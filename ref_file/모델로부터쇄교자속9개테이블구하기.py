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

  해석 수행 (0A가 아닌 경우에만 호출됨)
  """
  pythoncom.CoInitialize()  # Windows COM 초기화

  (
      worker_id,
      task_chunk,
      base_fem_path,
      magnet_material_name,
      rotor_group_no,
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

      # 5. a, b, c 상 쇄교 자속 추출 (턴 수 8 곱하기 반영)
      _, _, lambda_a = femm.mo_getcircuitproperties('A')
      _, _, lambda_b = femm.mo_getcircuitproperties('B')
      _, _, lambda_c = femm.mo_getcircuitproperties('C')

      results.append((
          theta_e_rad,
          theta_e_deg,
          theta_m_deg,
          excited_phase,
          current_val,
          lambda_a * 8,
          lambda_b * 8,
          lambda_c * 8,
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
  """A, B, C 상별 여자 조건에 따른 9가지 자속 쇄교 조합(_AA ~ _CC)을 피벗하고

  각각 3D 표면 그래프를 생성합니다.
  """
  df_all = pd.DataFrame(all_records)

  # mWb 단위 변환 컬럼 추가
  df_all['Lambda_a_mWb'] = df_all['Lambda_a'] * 1e3
  df_all['Lambda_b_mWb'] = df_all['Lambda_b'] * 1e3
  df_all['Lambda_c_mWb'] = df_all['Lambda_c'] * 1e3

  combinations = [
      ('A', 'Lambda_a_mWb', 'AA'),
      ('A', 'Lambda_b_mWb', 'AB'),
      ('A', 'Lambda_c_mWb', 'AC'),
      ('B', 'Lambda_a_mWb', 'BA'),
      ('B', 'Lambda_b_mWb', 'BB'),
      ('B', 'Lambda_c_mWb', 'BC'),
      ('C', 'Lambda_a_mWb', 'CA'),
      ('C', 'Lambda_b_mWb', 'CB'),
      ('C', 'Lambda_c_mWb', 'CC'),
  ]

  print(f'\n==================================================')
  print(f' [9가지 상호 인덕턴스/자속 쇄교 조합 피벗 저장 및 3D 그래프 생성 중]')
  print(f'==================================================')

  for excited_p, col_name, suffix in combinations:
    csv_filename = f'flux_linkage_{suffix}_pivot.csv'
    img_filename = f'flux_linkage_3d_{suffix}.png'
    title_text = f'Flux Linkage _{suffix} (Excited Phase {excited_p})'

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
    Z_grid = df_pivot.values  # 자속 쇄교수 [mWb]

    surf = ax.plot_surface(
        X_grid, Y_grid, Z_grid, cmap='viridis', edgecolor='none', alpha=0.9
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='자속 쇄교수 [mWb]')

    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel(f'Phase {excited_p} 전류 [A]', fontsize=11, labelpad=10)
    ax.set_ylabel('전기각 [deg]', fontsize=11, labelpad=10)
    ax.set_zlabel(f'Flux Linkage _{suffix} [mWb]', fontsize=11, labelpad=10)

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
  pole_number = 8
  pole_pairs = pole_number / 2

  current_list = [0, 25, 50, 100, 200, 400]
  theta_e_list = np.radians(np.arange(0, 361, 8))
  phases = ['A', 'B', 'C']

  all_records = []
  total_start_time = time.time()

  print(f'==================================================')
  print(f' [3상 여자 조건별 8극 모터 자속 쇄교수 스윕 해석]')
  print(f' 전류 범위: {current_list[0]}A ~ {current_list[-1]}A ')
  print(f'==================================================')

  for excited_phase in phases:
    print(
        f'\n================ [{excited_phase}상 여자 해석 시작]'
        ' ================='
    )
    for ia_val in current_list:
      # 전류가 0일 때는 FEMM 해석을 생략하고 0으로 데이터만 채움
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
              'Lambda_a': 0.0,
              'Lambda_b': 0.0,
              'Lambda_c': 0.0,
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
            la,
            lb,
            lc,
        ) in process_data:
          record = {
              'Excited_Phase': phase,
              'Current_A': current,
              'Theta_Elec_deg': theta_e_deg,
              'Theta_Mech_deg': theta_m_deg,
              'Theta_Elec_rad': theta_e_rad,
              'Lambda_a': la,
              'Lambda_b': lb,
              'Lambda_c': lc,
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