import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ==========================================
# 1. 전역 상수 및 파라미터 선언
# ==========================================
Rs = 1.2  # 고정자 저항 [ohm]
phi_m = 0.12  # 영구자석 쇄교자속 [Wb]
Rs_matrix = Rs * np.eye(3)
J = 0.0015  # 회전자 관성모멘트 [kg*m^2]
B = 0.0001  # 마찰계수 [N*m*s/rad]
T_load = 0.5  # 부하 토크 [N*m]
pole_pairs = 4  # 극쌍수 (필요시 설정)


def load_and_interpolate_inductance(
    csv_filename="ioniq5-13.FEM_inductance_table_1.csv",
):
  if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_filename}")

  df = pd.read_csv(csv_filename)
  print("=== 인덕턴스 테이블 데이터 로드 완료 ===")

  currents = df["Current_A"].values
  output_cols = [col for col in df.columns if col != "Current_A"]

  interpolators = {}
  for col in output_cols:
    interpolators[col] = interp1d(
        currents,
        df[col].values,
        kind="cubic",
        fill_value="extrapolate",
    )

  return interpolators, df
# 인덕턴스 테이블 로드 및 보간기 생성
interpolators, df_table = load_and_interpolate_inductance(
    "ioniq5-13.FEM_inductance_table_1.csv"
)


def get_inductance_matrix(i_abc, theta_e):
  """현재 상전류 크기 및 회전자 각도를 반영하여 안정적인 3x3 인덕턴스 행렬 구성"""
  i_norm = np.max(np.abs(i_abc))

  La = float(interpolators["Laa_Center_uH"](i_norm)) * 1e-6
  Lb = float(interpolators["Laa_Amplitude_uH"](i_norm)) * 1e-6

  La2 = float(interpolators["Lab_Center_uH"](i_norm)) * 1e-6
  Lb2 = float(interpolators["Lab_Amplitude_uH"](i_norm)) * 1e-6

  La3 = float(interpolators["Lac_Center_uH"](i_norm)) * 1e-6
  Lb3 = float(interpolators["Lac_Amplitude_uH"](i_norm)) * 1e-6

  l11 = La - Lb * np.cos(2 * theta_e)
  l12 = La2 - Lb2 * np.cos(2 * theta_e - 2 * np.pi / 3)
  l13 = La3 - Lb3 * np.cos(2 * theta_e + 2 * np.pi / 3)
  l21 = l12
  l22 = La - Lb * np.cos(2 * theta_e + 2 * np.pi / 3)
  l23 = La2 - Lb2 * np.cos(2 * theta_e)
  l31 = l13
  l32 = l23
  l33 = La - Lb * np.cos(2 * theta_e - 2 * np.pi / 3)

  Ls = np.array([[l11, l12, l13], [l21, l22, l23], [l31, l32, l33]])
  return Ls




# ==========================================
# 2. 파크/클라크 변환 함수 정의
# ==========================================
def abc_to_dq(i_abc, theta_e):
  ia, ib, ic = i_abc
  cos_th = np.cos(theta_e)
  sin_th = np.sin(theta_e)

  # Clark 변환 (abc -> alpha-beta)
  i_alpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
  i_beta = (2.0 / 3.0) * ((np.sqrt(3) / 2.0) * ib - (np.sqrt(3) / 2.0) * ic)

  # Park 변환 (alpha-beta -> dq)
  id_val = i_alpha * cos_th + i_beta * sin_th
  iq_val = -i_alpha * sin_th + i_beta * cos_th
  return np.array([id_val, iq_val])


def dq_to_abc(v_dq, theta_e):
  vd, vq = v_dq
  cos_th = np.cos(theta_e)
  sin_th = np.sin(theta_e)

  # 역 Park 변환 (dq -> alpha-beta)
  v_alpha = vd * cos_th - vq * sin_th
  v_beta = vd * sin_th + vq * cos_th

  # 역 Clark 변환 (alpha-beta -> abc)
  va = v_alpha
  vb = -0.5 * v_alpha + (np.sqrt(3) / 2.0) * v_beta
  vc = -0.5 * v_alpha - (np.sqrt(3) / 2.0) * v_beta
  return np.array([va, vb, vc])


def S_func(theta_e):
  return np.array(
      [
          -np.sin(theta_e),
          -np.sin(theta_e - 2 * np.pi / 3),
          -np.sin(theta_e + 2 * np.pi / 3),
      ]
  )

def motor_dynamics(i_abc, v_abc, omega_r, theta_r, dt):  

  Ls_matrix = get_inductance_matrix(i_abc, 0) #+ 1e-4*np.eye(3)  # 초기 인덕턴스 행렬
  inv_Ls_matrix = np.linalg.inv(Ls_matrix + np.eye(3) * 1e-6)
  theta_e = pole_pairs * theta_r
  S_theta_e = S_func(theta_e)
  
  term1 = np.dot(inv_Ls_matrix, np.dot(Rs_matrix, i_abc))
  term2 = np.dot(inv_Ls_matrix, v_abc)
  omega_e = pole_pairs * omega_r
  term3 = omega_e * phi_m *np.dot(inv_Ls_matrix, S_theta_e)
  di_dt = -term1 + term2 - term3

  # 전자기 토크 계산
  Tr = pole_pairs * phi_m * np.dot(np.transpose(S_theta_e), i_abc) 

  # 속도 및 각도 변화율 계산 (운동 방정식)
  domega_r_dt = (Tr - T_load - B * omega_r) / J
  
  
    # 오일러 적분 (수치 해석)
  i_abc += di_dt * dt
  omega_r += domega_r_dt * dt
  dtheta_r_dt = omega_r
  theta_r += dtheta_r_dt * dt
  theta_r = np.mod(theta_r, 2 * np.pi)  # 각도 범위 정규화

  return i_abc, omega_r, theta_r, Tr

