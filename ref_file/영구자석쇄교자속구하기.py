import datetime
import os
import shutil
import time
from multiprocessing import Pool, cpu_count
import femm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pythoncom

# 한글 폰트 깨짐 방지 (Windows 환경 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

delete_temp_files = (
    True  # True로 설정하면 각 프로세스 종료 후 임시 파일 삭제
)


def worker_process_pm(args):
  """전류 0A 조건에서 회전자 위치별 영구자석 쇄교자속 추출 워커 프로세스"""
  pythoncom.CoInitialize()  # Windows COM 초기화

  worker_id, task_chunk, base_fem_path, rotor_group_no, pole_pairs = args
  process_fem_path = f'model_worker_pm_{worker_id}.fem'

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

      # 1. 모든 상 전류를 0A로 설정 (영구자석 자속만 추출하기 위함)
      femm.mi_setcurrent('A', 0.0)
      femm.mi_setcurrent('B', 0.0)
      femm.mi_setcurrent('C', 0.0)

      # 2. 회전자 기계각 회전 적용
      if rotor_group_no is not None and theta_m_deg != 0.0:
        femm.mi_seteditmode('group')
        femm.mi_clearselected()
        if isinstance(rotor_group_no, (list, tuple)):
          for g_no in rotor_group_no:
            femm.mi_selectgroup(g_no)
        else:
          femm.mi_selectgroup(rotor_group_no)
        femm.mi_moverotate(0.0, 0.0, theta_m_deg)

      # 3. 해석 실행 및 솔루션 로드
      femm.mi_analyze(1)
      femm.mi_loadsolution()

      # 4. a, b, c 상 영구자석 쇄교 자속 추출 (턴 수 8 곱하기 반영)
      _, _, lambda_a = femm.mo_getcircuitproperties('A')
      _, _, lambda_b = femm.mo_getcircuitproperties('B')
      _, _, lambda_c = femm.mo_getcircuitproperties('C')

      results.append((
          theta_e_rad,
          theta_e_deg,
          theta_m_deg,
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


def run_pm_flux_sweep():
  base_fem_path = 'ioniq5-13.FEM'
  if not os.path.exists(base_fem_path):
    raise FileNotFoundError(f'기준 모델 파일을 찾을 수 없습니다: {base_fem_path}')

  rotor_group_no = [1, 20]
  pole_number = 8
  pole_pairs = pole_number / 2

  # 전기각 0도 ~ 360도, 2도 또는 4도 간격 설정 (원하시는 간격으로 조절 가능)
  theta_e_list = np.radians(np.arange(0, 361, 4))

  total_start_time = time.time()

  print(f'==================================================')
  print(f' [영구자석에 의한 상별 쇄교자속 스윕 해석 (전류 0A)]')
  print(f' 전기각 범위: 0° ~ 360° (간격: 4°)')
  print(f'==================================================')

  total_tasks = len(theta_e_list)
  num_processes = min(cpu_count(), total_tasks)
  task_chunks = np.array_split(theta_e_list, num_processes)

  worker_args = [
      (idx, list(chunk), base_fem_path, rotor_group_no, pole_pairs)
      for idx, chunk in enumerate(task_chunks)
      if len(chunk) > 0
  ]

  with Pool(processes=len(worker_args)) as pool:
    chunk_results = pool.map(worker_process_pm, worker_args)

  # 결과 데이터 수집
  records = []
  for process_data in chunk_results:
    for (
        theta_e_rad,
        theta_e_deg,
        theta_m_deg,
        la_pm,
        lb_pm,
        lc_pm,
    ) in process_data:
      records.append({
          'Theta_Elec_deg': theta_e_deg,
          'Theta_Mech_deg': theta_m_deg,
          'Theta_Elec_rad': theta_e_rad,
          'Lambda_a_pm': la_pm,
          'Lambda_b_pm': lb_pm,
          'Lambda_c_pm': lc_pm,
          # mWb 단위 컬럼 추가 (편의성)
          'Lambda_a_pm_mWb': la_pm * 1e3,
          'Lambda_b_pm_mWb': lb_pm * 1e3,
          'Lambda_c_pm_mWb': lc_pm * 1e3,
      })

  df_pm = pd.DataFrame(records)
  df_pm = df_pm.sort_values(by='Theta_Elec_deg').reset_index(drop=True)

  # CSV 저장
  csv_filename = 'permanent_magnet_flux_linkage_0A.csv'
  df_pm.to_csv(csv_filename, index=False, encoding='utf-8-sig')

  # 간단한 2D 파형 그래프 저장 (확인용)
  plt.figure(figsize=(10, 6))
  plt.plot(
      df_pm['Theta_Elec_deg'],
      df_pm['Lambda_a_pm_mWb'],
      label='A상 PM 자속 (Lambda_a_pm)',
      color='blue',
      lw=2,
  )
  plt.plot(
      df_pm['Theta_Elec_deg'],
      df_pm['Lambda_b_pm_mWb'],
      label='B상 PM 자속 (Lambda_b_pm)',
      color='green',
      lw=2,
  )
  plt.plot(
      df_pm['Theta_Elec_deg'],
      df_pm['Lambda_c_pm_mWb'],
      label='C상 PM 자속 (Lambda_c_pm)',
      color='orange',
      lw=2,
  )

  plt.title(
      '전류 0A 조건 - 회전자 위치(전기각)에 따른 영구자석 상 자속 쇄교수',
      fontsize=12,
      fontweight='bold',
  )
  plt.xlabel('전기각 [deg]', fontsize=10)
  plt.ylabel('자속 쇄교수 [mWb]', fontsize=10)
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend(loc='upper right', fontsize=10)
  plt.tight_layout()

  plot_filename = 'permanent_magnet_flux_linkage_0A.png'
  plt.savefig(plot_filename, dpi=300)
  plt.close()

  total_elapsed = time.time() - total_start_time
  print(f'\n==================================================')
  print(
      f' 해석 완료 총 소요 시간: {int(total_elapsed // 60)}분'
      f' {total_elapsed % 60:.2f}초'
  )
  print(f' 데이터 CSV 저장 완료: "{csv_filename}"')
  print(f' 파형 그래프 저장 완료: "{plot_filename}"')
  print(f'==================================================')

  return df_pm


if __name__ == '__main__':
  import multiprocessing

  multiprocessing.freeze_support()
  df_pm_result = run_pm_flux_sweep()