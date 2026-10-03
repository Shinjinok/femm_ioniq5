import datetime
import os
import shutil
import time
from multiprocessing import Pool, cpu_count
import femm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 그래프용 모듈
import numpy as np
import pandas as pd
import pythoncom

# 한글 폰트 깨짐 방지 (Windows 환경 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

delete_temp_files = (
    True  # True로 설정하면 각 프로세스 종료 후 임시 파일 삭제
)


def worker_process(args):
  """개별 프로세스가 할당받은 전기각(Theta_e) 리스트와 지정된 여자상(excited_phase) 및 전류(current_val)로

  해석을 수행하고, 각 상의 평균 자계강도(H)를 추출합니다.
  """
  pythoncom.CoInitialize()  # Windows COM 초기화

  (
      worker_id,
      task_chunk,
      base_fem_path,
      magnet_material_name,
      rotor_group_no,
      phase_groups,  # A, B, C 상 권선들의 그룹 번호 딕셔너리 (예: {'A': [...], 'B': [...], 'C': [...]})
      pole_pairs,
      excited_phase,
      current_val,
  ) = args
  process_fem_path = f'model_worker_{worker_id}.fem'

  results = []

  try:
    for theta_e_rad in task_chunk:
      theta_e_deg = np.degrees(theta_e_rad)
      theta_m_deg = theta_e_deg / pole_pairs

      if os.path.exists(process_fem_path):
        os.remove(process_fem_path)
      shutil.copy(base_fem_path, process_fem_path)

      femm.openfemm(1)
      femm.opendocument(process_fem_path)

      # 1. 영구자석 물성치 변경 (순수 돌극성 추출: mu=1, Hc=0)
      try:
        femm.mi_modifymaterial(magnet_material_name, 1, 1.0)
        femm.mi_modifymaterial(magnet_material_name, 2, 1.0)
        femm.mi_modifymaterial(magnet_material_name, 3, 0.0)
      except Exception:
        pass

      # 2. 여자상(excited_phase)에만 전류 설정, 나머지는 0A
      currents = {'A': 0.0, 'B': 0.0, 'C': 0.0}
      currents[excited_phase] = current_val

      femm.mi_setcurrent('A', currents['A'])
      femm.mi_setcurrent('B', currents['B'])
      femm.mi_setcurrent('C', currents['C'])

      # 3. 회전자 기계각 회전 적용
      if rotor_group_no is not None and theta_m_deg != 0.0:
        femm.mi_seteditmode('group')
        femm.mi_clearselected()
        if isinstance(rotor_group_no, (list, tuple)):
          for g_no in rotor_group_no:
            femm.mi_selectgroup(g_no)
        else:
          femm.mi_selectgroup(rotor_group_no)
        femm.mi_moverotate(0.0, 0.0, theta_m_deg)

      # 4. 해석 실행 및 솔루션 로드
      femm.mi_analyze(1)
      femm.mi_loadsolution()

      # 5. a, b, c 상별 평균 자계강도(H) 또는 자속밀도 추출
      # FEMM에서 각 상 권선 영역의 평균 자계강도(H)를 구하기 위해 블록 적분 활용
      # (그룹 번호가 지정되어 있다고 가정)
      phase_H = {}
      for phase_name, g_list in phase_groups.items():
        femm.mo_clearblock()
        if isinstance(g_list, (list, tuple)):
          for g_no in g_list:
            femm.mo_groupselectblock(g_no)
        else:
          femm.mo_groupselectblock(g_list)

        # blockintegral 번호 안내:
        # 2: Volume (체적)
        # 8 또는 특정 적분 번호를 통해 H나 B 평균 계산 가능
        # 편의상 mo_blockintegral(1) [Stored Energy] 또는 mo_getblockinfo 활용 가능
        # 여기서는 예시로 블록 평균 자계강도(H) 혹은 대표적 블록 연산 결과 활용
        # * 주의: 모델의 권선 그룹 구조에 맞게 mo_blockintegral 인덱스 조정 필요
        try:
          # 예: 자계강도 H 관련 블록 적분 혹은 평균값 추출
          # mo_blockintegral(표준 번호) 활용 예시 (지정된 그룹의 평균 H값 산출)
          vol = femm.mo_blockintegral(2)  # 체적
          if vol > 0:
            # 예시로 에너지나 자속 관련 적분 후 체적으로 나누거나,
            # 특정 블록 정보 함수 활용 (FEMM API 버전에 따라 다름)
            # 안전하게 H 크기 평균을 원할 경우 mo_blockintegral 활용
            h_val = (
                femm.mo_blockintegral(10) / vol
                if hasattr(femm, 'mo_blockintegral')
                else 0.0
            )  # 모델 환경에 맞게 조정
          else:
            h_val = 0.0
        except:
          h_val = 0.0

        # 간단하게 특정 대표 지점의 H값을 원하실 경우 mo_getpointvals(x, y)를 사용할 수도 있습니다.
        phase_H[phase_name] = h_val

      results.append((
          theta_e_rad,
          theta_e_deg,
          theta_m_deg,
          excited_phase,
          current_val,
          phase_H.get('A', 0.0),
          phase_H.get('B', 0.0),
          phase_H.get('C', 0.0),
      ))
      femm.closefemm()

  finally:
    pythoncom.CoUninitialize()
    if delete_temp_files:
      for ext in ['.fem', '.ans', '.rec']:
        file_path = process_fem_path.replace('.fem', ext)
        if os.path.exists(file_path):
          try:
            os.remove(file_path)
          except:
            pass

  return results


def process_and_plot_3d_all_combinations(all_records):
  """A, B, C 상별 여자 조건에 따른 자계강도 조합을 피벗하고 3D 표면 그래프를 생성합니다."""
  df_all = pd.DataFrame(all_records)

  # 단위 변환 필요시 적용 (예: A/m 또는 kA/m)
  df_all['H_a_kAm'] = df_all['H_a'] * 1e-3
  df_all['H_b_kAm'] = df_all['H_b'] * 1e-3
  df_all['H_c_kAm'] = df_all['H_c'] * 1e-3

  combinations = [
      ('A', 'H_a_kAm', 'AA'),
      ('A', 'H_b_kAm', 'AB'),
      ('A', 'H_c_kAm', 'AC'),
      ('B', 'H_a_kAm', 'BA'),
      ('B', 'H_b_kAm', 'BB'),
      ('B', 'H_c_kAm', 'BC'),
      ('C', 'H_a_kAm', 'CA'),
      ('C', 'H_b_kAm', 'CB'),
      ('C', 'H_c_kAm', 'CC'),
  ]

  print(f'\n==================================================')
  print(f' [상호 자계강도 조합 피벗 저장 및 3D 그래프 생성 중]')
  print(f'==================================================')

  for excited_p, col_name, suffix in combinations:
    csv_filename = f'magnetic_field_H_{suffix}_pivot.csv'
    img_filename = f'magnetic_field_3d_{suffix}.png'
    title_text = f'Magnetic Field Intensity _{suffix} (Excited Phase {excited_p})'

    df_subset = df_all[df_all['Excited_Phase'] == excited_p]
    if df_subset.empty:
      continue

    df_pivot = df_subset.pivot(
        index='Theta_Elec_deg', columns='Current_A', values=col_name
    )
    df_pivot.to_csv(csv_filename, encoding='utf-8-sig')
    print(f' -> [{suffix}상 피벗 CSV 저장 완료] {csv_filename}')

    # 3D 표면 그래프 생성
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(projection='3d')

    X = df_pivot.columns.values  # 전류 [A]
    Y = df_pivot.index.values  # 전기각 [deg]
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z_grid = df_pivot.values  # 자계강도 [kA/m]

    surf = ax.plot_surface(
        X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9
    )
    fig.colorbar(
        surf, ax=ax, shrink=0.5, aspect=10, label='자계강도 [kA/m]'
    )

    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel(f'Phase {excited_p} 전류 [A]', fontsize=11, labelpad=10)
    ax.set_ylabel('전기각 [deg]', fontsize=11, labelpad=10)
    ax.set_zlabel(f'Magnetic Field _{suffix} [kA/m]', fontsize=11, labelpad=10)

    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.savefig(img_filename, dpi=300)
    plt.close()
    print(f' -> [{suffix}상 3D 그래프 저장 완료] {img_filename}')


def run_multi_current_sweep():
  base_fem_path = 'ioniq5-13.FEM'
  if not os.path.exists(base_fem_path):
    raise FileNotFoundError(f'기준 모델 파일을 찾을 수 없습니다: {base_fem_path}')

  magnet_material_name = 'Mag'
  rotor_group_no = [1, 20]
  
  # ★ 각 상 권선이 속한 FEMM 그룹 번호 설정 (사용하시는 모델의 그룹 번호에 맞게 수정 필요)
  phase_groups = {
      'A': [2],  # 예시 A상 그룹 번호
      'B': [3],  # 예시 B상 그룹 번호
      'C': [4],  # 예시 C상 그룹 번호
  }

  pole_number = 8
  pole_pairs = pole_number / 2

  current_list = [1, 25, 50, 100, 200, 400]
  theta_e_list = np.radians(np.arange(0, 361, 8))
  phases = ['A', 'B', 'C']

  all_records = []
  total_start_time = time.time()

  print(f'==================================================')
  print(f' [3상 여자 조건별 8극 모터 자계강도 스윕 해석]')
  print(f' 전류 범위: {current_list[0]}A ~ {current_list[-1]}A ')
  print(f'==================================================')

  for excited_phase in phases:
    print(
        f'\n================ [{excited_phase}상 여자 해석 시작]'
        ' ================='
    )
    for ia_val in current_list:
      if ia_val == 0:
        print(
            f'--- [{excited_phase}상 여자 | 전류 조건: 0A] 해석 생략 (데이터 0으로'
            ' 채우기) ---'
        )
        for theta_e_rad in theta_e_list:
          theta_e_deg = np.degrees(theta_e_rad)
          theta_m_deg = theta_e_deg / pole_pairs
          record = {
              'Excited_Phase': excited_phase,
              'Current_A': 0.0,
              'Theta_Elec_deg': theta_e_deg,
              'Theta_Mech_deg': theta_m_deg,
              'Theta_Elec_rad': theta_e_rad,
              'H_a': 0.0,
              'H_b': 0.0,
              'H_c': 0.0,
          }
          all_records.append(record)
        continue

      print(
          f'--- [{excited_phase}상 여자 | 전류 조건: {ia_val}A] 해석 진행 중 ---'
      )

      total_tasks = len(theta_e_list)
      num_processes = min(cpu_count(), total_tasks)
      task_chunks = np.array_split(theta_e_list, num_processes)

      worker_args = [
          (
              idx,
              list(chunk),
              base_fem_path,
              magnet_material_name,
              rotor_group_no,
              phase_groups,
              pole_pairs,
              excited_phase,
              ia_val,
          )
          for idx, chunk in enumerate(task_chunks)
          if len(chunk) > 0
      ]

      start_time_sec = time.time()
      with Pool(processes=len(worker_args)) as pool:
        chunk_results = pool.map(worker_process, worker_args)
      elapsed_sec = time.time() - start_time_sec
      print(
          f' -> [{excited_phase}상 / {ia_val}A] 완료 (소요 시간:'
          f' {int(elapsed_sec // 60)}분 {elapsed_sec % 60:.2f}초)'
      )

      for process_data in chunk_results:
        for (
            theta_e_rad,
            theta_e_deg,
            theta_m_deg,
            phase,
            current,
            ha,
            hb,
            hc,
        ) in process_data:
          record = {
              'Excited_Phase': phase,
              'Current_A': current,
              'Theta_Elec_deg': theta_e_deg,
              'Theta_Mech_deg': theta_m_deg,
              'Theta_Elec_rad': theta_e_rad,
              'H_a': ha,
              'H_b': hb,
              'H_c': hc,
          }
          all_records.append(record)

  process_and_plot_3d_all_combinations(all_records)

  total_elapsed = time.time() - total_start_time
  print(f'\n==================================================')
  print(
      ' 모든 상 여자 스윕 해석 총 소요 시간:'
      f' {int(total_elapsed // 60)}분 {total_elapsed % 60:.2f}초'
  )
  print(f'==================================================')


if __name__ == '__main__':
  import multiprocessing

  multiprocessing.freeze_support()
  run_multi_current_sweep()