import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def load_flux_tables():
  """저장된 10개의 CSV 테이블을 불러와 보간기(Interpolator)를 구성합니다."""
  pm_df = pd.read_csv('permanent_magnet_flux_linkage_0A.csv')
  theta_vals = np.sort(pm_df['Theta_Elec_deg'].unique())

  lam_a_pm = pm_df.sort_values('Theta_Elec_deg')['Lambda_a_pm'].values
  lam_b_pm = pm_df.sort_values('Theta_Elec_deg')['Lambda_b_pm'].values
  lam_c_pm = pm_df.sort_values('Theta_Elec_deg')['Lambda_c_pm'].values

  pm_interp_a = RegularGridInterpolator(
      (theta_vals,), lam_a_pm, method='linear', bounds_error=False, fill_value=None
  )
  pm_interp_b = RegularGridInterpolator(
      (theta_vals,), lam_b_pm, method='linear', bounds_error=False, fill_value=None
  )
  pm_interp_c = RegularGridInterpolator(
      (theta_vals,), lam_c_pm, method='linear', bounds_error=False, fill_value=None
  )

  suffixes = ['AA', 'AB', 'AC', 'BA', 'BB', 'BC', 'CA', 'CB', 'CC']
  ind_interps = {}

  for sfx in suffixes:
    pivot_df = pd.read_csv(f'flux_linkage_{sfx}_pivot.csv')
    theta_deg = pivot_df['Theta_Elec_deg'].values
    currents = np.array(
        [float(c) for c in pivot_df.columns if c != 'Theta_Elec_deg']
    )
    matrix = pivot_df[pivot_df.columns[1:]].values * 1e-3  # mWb -> Wb 변환

    ind_interps[sfx] = RegularGridInterpolator(
        (theta_deg, currents),
        matrix,
        method='linear',
        bounds_error=False,
        fill_value=None,
    )

  return pm_interp_a, pm_interp_b, pm_interp_c, ind_interps


def calculate_flux_linkage(
    theta_elec_deg, ia, ib, ic, pm_a, pm_b, pm_c, ind_interps
):
  la = pm_a([theta_elec_deg])[0]
  lb = pm_b([theta_elec_deg])[0]
  lc = pm_c([theta_elec_deg])[0]

  la += (
      ind_interps['AA']((theta_elec_deg, abs(ia))) * np.sign(ia)
      if ia != 0
      else 0
  )
  la += (
      ind_interps['BA']((theta_elec_deg, abs(ib))) * np.sign(ib)
      if ib != 0
      else 0
  )
  la += (
      ind_interps['CA']((theta_elec_deg, abs(ic))) * np.sign(ic)
      if ic != 0
      else 0
  )

  lb += (
      ind_interps['AB']((theta_elec_deg, abs(ia))) * np.sign(ia)
      if ia != 0
      else 0
  )
  lb += (
      ind_interps['BB']((theta_elec_deg, abs(ib))) * np.sign(ib)
      if ib != 0
      else 0
  )
  lb += (
      ind_interps['CB']((theta_elec_deg, abs(ic))) * np.sign(ic)
      if ic != 0
      else 0
  )

  lc += (
      ind_interps['AC']((theta_elec_deg, abs(ia))) * np.sign(ia)
      if ia != 0
      else 0
  )
  lc += (
      ind_interps['BC']((theta_elec_deg, abs(ib))) * np.sign(ib)
      if ib != 0
      else 0
  )
  lc += (
      ind_interps['CC']((theta_elec_deg, abs(ic))) * np.sign(ic)
      if ic != 0
      else 0
  )

  return la, lb, lc


def generate_attached_format_matrices():
  print('==================================================')
  print(' [첨부 파일 형식에 맞춘 d/q 매트릭스 생성 및 저장 시작]')
  print('==================================================')

  pm_a, pm_b, pm_c, ind_interps = load_flux_tables()

  # 첨부 파일 형식: 행 = Beta(전류각 90~180도, 5도 간격), 열 = 전류 크기(0~340A, 34A 간격)
  idq_magnitudes = np.arange(0, 341, 34)  # [0.0, 34.0, ..., 340.0]
  gamma_angles_deg = np.arange(90, 181, 5)  # [90, 95, ..., 180]

  # 특정 전기각(예: 0도 또는 특정 대표 위치)에서의 매트릭스이거나, 전기각별 평균/최대값 등
  # 여기서는 첨부 파일 포맷과 정확히 일치하도록 0도 전기각 기준(또는 필요시 대표 위치)으로 매트릭스를 구성합니다.
  # 만약 특정 전기각 기준이 아니라면 원하는 theta_e를 선택할 수 있습니다.
  target_theta_e = 0.0

  d_matrix_data = []
  q_matrix_data = []

  for gamma_deg in gamma_angles_deg:
    gamma_rad = np.radians(gamma_deg)
    row_d = {'Beta': gamma_deg}
    row_q = {'Beta': gamma_deg}

    for Idq in idq_magnitudes:
      id_val = Idq * np.cos(gamma_rad)
      iq_val = Idq * np.sin(gamma_rad)

      theta_e_rad = np.radians(target_theta_e)
      ia = id_val * np.cos(theta_e_rad) - iq_val * np.sin(theta_e_rad)
      ib = id_val * np.cos(theta_e_rad - 2 * np.pi / 3) - iq_val * np.sin(
          theta_e_rad - 2 * np.pi / 3
      )
      ic = id_val * np.cos(theta_e_rad + 2 * np.pi / 3) - iq_val * np.sin(
          theta_e_rad + 2 * np.pi / 3
      )

      lam_a, lam_b, lam_c = calculate_flux_linkage(
          target_theta_e, ia, ib, ic, pm_a, pm_b, pm_c, ind_interps
      )

      # Park 변환 (mWb 단위로 환산)
      lam_d = (
          (2.0 / 3.0)
          * (
              lam_a * np.cos(theta_e_rad)
              + lam_b * np.cos(theta_e_rad - 2 * np.pi / 3)
              + lam_c * np.cos(theta_e_rad + 2 * np.pi / 3)
          )
          * 1e3
      )

      lam_q = (
          -(2.0 / 3.0)
          * (
              lam_a * np.sin(theta_e_rad)
              + lam_b * np.sin(theta_e_rad - 2 * np.pi / 3)
              + lam_c * np.sin(theta_e_rad + 2 * np.pi / 3)
          )
          * 1e3
      )

      # 열 이름을 실수형태(0.0, 34.0 등)로 매칭
      col_name = float(Idq)
      row_d[col_name] = round(lam_d, 2)
      row_q[col_name] = round(lam_q, 2)

    d_matrix_data.append(row_d)
    q_matrix_data.append(row_q)

  df_d_matrix = pd.DataFrame(d_matrix_data)
  df_q_matrix = pd.DataFrame(q_matrix_data)

  # 첨부 파일과 동일한 이름으로 저장
  d_output_filename = '모델로부터Lambda_d_Idq_matrix.csv'
  q_output_filename = '모델로부터Lambda_q_Idq_matrix.csv'

  df_d_matrix.to_csv(d_output_filename, index=False, encoding='utf-8-sig')
  df_q_matrix.to_csv(q_output_filename, index=False, encoding='utf-8-sig')

  print(f' -> d축 매트릭스 저장 완료: "{d_output_filename}"')
  print(f' -> q축 매트릭스 저장 완료: "{q_output_filename}"')


if __name__ == '__main__':
  generate_attached_format_matrices()