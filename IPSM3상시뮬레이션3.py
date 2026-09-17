import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_and_interpolate_inductance(
    csv_filename="ioniq5-13.FEM_all_currents_inductance_summary.csv",
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
    "ioniq5-13.FEM_all_currents_inductance_summary.csv"
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
# 1. 모터 및 시뮬레이션 파라미터 설정
# ==========================================
T_sim = 0.5  # 시뮬레이션 총 시간 [s]
dt = 1e-4  # 샘플링 시간 [s]
time = np.arange(0, T_sim, dt)
N = len(time)

Rs = 1.2*np.ones(3)  # 고정자 저항 [ohm]
Ld = 8.5e-3  # d축 인덕턴스 [H]
Lq = 8.5e-3  # q축 인덕턴스 [H]
phi_m = 0.12  # 영구자석 쇄교자속 [Wb]
pole_pairs = 4  # 극쌍수 (8극)
J = 0.0015  # 회전자 관성모멘트 [kg*m^2]
B = 0.0001  # 마찰계수 [N*m*s/rad]
T_load = 0.5  # 부하 토크 [N*m]

Ls_1 = get_inductance_matrix(200*np.ones(3), 0*np.pi/180)  # 초기
Ld = Ls_1[0, 0]

Ls_matrix = [[Ld+1e-4, -Ld/2, -Ld/2], [-Ld/2, Ld+1e-4, -Ld/2], [-Ld/2, -Ld/2, Ld+1e-4]]  # 초기


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


# ==========================================
# 3. 제어기(PI Controller) 변수 초기화
# ==========================================
# 속도 제어기 게인
Kp_spd, Ki_spd = 0.1, 1.0
integral_spd_err = 0.0

# 전류 제어기 게인 (d축, q축)
Kp_i, Ki_i = 2.0, 50.0
integral_id_err = 0.0
integral_iq_err = 0.0


# 속도 지령 함수 (0.1초일 때 50 rad/s로 스텩 입력)
def get_speed_ref(t):
  return 50.0 if t >= 0.1 else 0.0


# ==========================================
# 4. 시뮬레이션 상태 변수 초기화
# ==========================================
i_abc = np.zeros(3)
omega_r = 0.0
theta_r = 0.0

# 데이터 기록용 배열
i_history = np.zeros((3, N))
omega_history = np.zeros(N)
omega_ref_history = np.zeros(N)
Te_history = np.zeros(N)
#인덕턴스 행렬 설정
print("Ls_matrix:", Ls_matrix)
# ==========================================
# 5. 메인 제어 및 시뮬레이션 루프
# ==========================================
for k in range(N):
  t = time[k]
  theta_e = pole_pairs * theta_r  # 전기각 계산

  # --- [1] 속도 제어기 (Outer Loop) ---
  omega_ref = get_speed_ref(t)
  omega_ref_history[k] = omega_ref
  
  spd_err = omega_ref - omega_r
  integral_spd_err += spd_err * dt
  iq_ref = Kp_spd * spd_err + Ki_spd * integral_spd_err
  iq_ref = np.clip(iq_ref, -20, 20)  # 전류 지령 제한
  id_ref = 0.0  # SPMSM이므로 d축 전류는 0으로 제어
  
    # --- [2] 피드백 전류 측정 및 dq 변환 ---
  id_cur, iq_cur = abc_to_dq(i_abc, theta_e)
  
    # --- [3] 전류 제어기 (Inner Loop + 디커플링 제어) ---
  id_err = id_ref - id_cur
  integral_id_err += id_err * dt
  vd_ref = (
        Kp_i * id_err + Ki_i * integral_id_err - pole_pairs * omega_r * Lq * iq_cur
  )
  
  iq_err = iq_ref - iq_cur
  integral_iq_err += iq_err * dt
  vq_ref = (
        Kp_i * iq_err
        + Ki_i * integral_iq_err
        + pole_pairs * omega_r * (Ld * id_cur + phi_m)
  )

# --- [4] 전압 역변환 (dq -> abc) ---
  v_abc = dq_to_abc(np.array([vd_ref, vq_ref]), theta_e)

  # --- [5] 모터 모델 전개 (전기적/기계적 동역학) ---
  # di_abc/dt 계산 (간이 저항-인덕턴스 회로 방정식)
  di_dt = np.dot(np.linalg.inv(Ls_matrix), (v_abc - Rs * i_abc - pole_pairs * omega_r * phi_m * np.array([
      -np.sin(theta_e),
      -np.sin(theta_e - 2 * np.pi / 3),
      -np.sin(theta_e + 2 * np.pi / 3),
  ])))

  
  
  """ Ls_sub = get_inductance_matrix(i_abc, theta_e)
  inv_Ls = np.linalg.inv(Ls_sub + np.eye(3) * 1e-6)
  S_theta_e = S_func(theta_e)
  term1 = np.dot(inv_Ls, np.dot(Rs, i_abc))
  term2 = np.dot(inv_Ls, v_abc)
  term3 = 4 * omega_r * phi_m *np.dot(inv_Ls, S_theta_e)
  di_dt = -term1 + term2 - term3

  #Te = phi_m * np.dot(np.transpose(S_theta_e), i_abc) """
  # 전자기 토크 계산
  Te = 1.5 * pole_pairs * (phi_m * iq_cur + (Ld - Lq) * id_cur * iq_cur)
  Te_history[k] = Te
      # 속도 및 각도 변화율 계산 (운동 방정식)
  domega_dt = (Te - T_load - B * omega_r) / J
  dtheta_dt = omega_r
  
    # 오일러 적분 (수치 해석)
  i_abc += di_dt * dt
  omega_r += domega_dt * dt
  theta_r += dtheta_dt * dt
  theta_r = np.mod(theta_r, 2 * np.pi / pole_pairs)  # 각도 범위 정규화
  
    # 기록
  i_history[:, k] = i_abc
  omega_history[k] = omega_r

# ==========================================
# 6. 결과 시각화
# ==========================================
plt.figure(figsize=(10, 8))

# 속도 응답 그래프
plt.subplot(3, 1, 1)
plt.plot(time, omega_history, 'g-', label='실제 속도 ($\omega_r$)')
plt.plot(time, omega_ref_history, 'k--', label='지령 속도')
plt.ylabel('Speed [rad/s]')
plt.legend(loc='lower right')
plt.grid(True)
plt.title('PMSM 벡터 제어(DQ Control) 시뮬레이션')

# 상 전류 그래프
plt.subplot(3, 1, 2)
plt.plot(time, i_history[0], label='$i_a$', alpha=0.8)
plt.plot(time, i_history[1], label='$i_b$', alpha=0.8)
plt.plot(time, i_history[2], label='$i_c$', alpha=0.8)
plt.ylabel('Phase Current [A]')
plt.legend(loc='upper right')
plt.grid(True)

# 전자기 토크 그래프
plt.subplot(3, 1, 3)
plt.plot(time, Te_history, 'r-', label='전자기 토크 ($T_e$)')
plt.axhline(y=T_load, color='k', linestyle='--', label='부하 토크')
plt.xlabel('Time [s]')
plt.ylabel('Torque [N·m]')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()