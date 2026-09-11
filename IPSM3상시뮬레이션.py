import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 시뮬레이션 및 모터 파라미터 설정
# ==========================================
T_sim = 2.0  # 시뮬레이션 총 시간 [s]
dt = 1e-4  # 샘플링 시간 [s]
time = np.arange(0, T_sim, dt)
N = len(time)

Rs = 1.2  # 고정자 저항 [ohm]
Ll = 0.01  # 누설 인덕턴스 [H]
Lm = 0.005  # 자화 인덕턴스 [H]
Ls_val = Ll + Lm

# 3x3 고정자 인덕턴스 행렬 Ls 정의
Ls = np.array(
    [
        [Ls_val, -0.5 * Lm, -0.5 * Lm],
        [-0.5 * Lm, Ls_val, -0.5 * Lm],
        [-0.5 * Lm, -0.5 * Lm, Ls_val],
    ]
)
inv_Ls = np.linalg.inv(Ls)

phi_m = 0.12  # 영구자석 쇄교자속 [Wb]
J = 0.0015  # 회전자 관성모멘트 [kg*m^2]
B = 0.0001  # 마찰계수 [N*m*s/rad]
T_load = 0.5  # 부하 토크 [N*m]


# ==========================================
# 2. 파크/클라크 변환 함수 정의
# ==========================================
def abc_to_dq(i_abc, theta_r):
  ia, ib, ic = i_abc
  cos_th = np.cos(theta_r)
  sin_th = np.sin(theta_r)

  i_alpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
  i_beta = (2.0 / 3.0) * (
      (np.sqrt(3) / 2.0) * ib - (np.sqrt(3) / 2.0) * ic
  )

  id_val = i_alpha * cos_th + i_beta * sin_th
  iq_val = -i_alpha * sin_th + i_beta * cos_th
  return np.array([id_val, iq_val])


def dq_to_abc(v_dq, theta_r):
  vd, vq = v_dq
  cos_th = np.cos(theta_r)
  sin_th = np.sin(theta_r)

  v_alpha = vd * cos_th - vq * sin_th
  v_beta = vd * sin_th + vq * cos_th

  va = v_alpha
  vb = -0.5 * v_alpha + (np.sqrt(3) / 2.0) * v_beta
  vc = -0.5 * v_alpha - (np.sqrt(3) / 2.0) * v_beta
  return np.array([va, vb, vc])


def S_func(theta_r):
  return np.array(
      [
          -np.sin(theta_r),
          -np.sin(theta_r - 2 * np.pi / 3),
          -np.sin(theta_r + 2 * np.pi / 3),
      ]
  )


# ==========================================
# 3. 제어기 가인수(PI Gain) 설정
# ==========================================
Kp_spd, Ki_spd = 0.5, 10.0
integral_spd_err = 0.0

Kp_id, Ki_id = 15.0, 100.0
integral_id_err = 0.0

Kp_iq, Ki_iq = 15.0, 100.0
integral_iq_err = 0.0


def get_speed_ref(t):
  if t < 0.2:
    return 0.0
  else:
    return 50.0  # [rad/s]


# ==========================================
# 4. 상태 변수 초기화
# ==========================================
i = np.zeros(3)
omega_r = 0.0
theta_r = 0.0
theta_unwrapped = 0.0

i_history = np.zeros((3, N))
omega_history = np.zeros(N)
omega_ref_history = np.zeros(N)
Te_history = np.zeros(N)
theta_history = np.zeros(N)

# ==========================================
# 5. 수치 해석 및 제어 루프
# ==========================================
for k in range(N):
  t = time[k]

  # 1) 속도 제어기 (Outer Loop)
  omega_ref = get_speed_ref(t)
  omega_ref_history[k] = omega_ref

  spd_err = omega_ref - omega_r
  integral_spd_err += spd_err * dt
  iq_ref = Kp_spd * spd_err + Ki_spd * integral_spd_err
  id_ref = 0.0

  # 2) 피드백 전류 측정 및 dq 변환
  id_cur, iq_cur = abc_to_dq(i, theta_r)

  # 3) 전류 제어기 (Inner Loop + 디커플링 항)
  id_err = id_ref - id_cur
  integral_id_err += id_err * dt
  vd_ref = (
      Kp_id * id_err + Ki_id * integral_id_err - omega_r * Ls_val * iq_cur
  )

  iq_err = iq_ref - iq_cur
  integral_iq_err += iq_err * dt
  vq_ref = (
      Kp_iq * iq_err
      + Ki_iq * integral_iq_err
      + omega_r * (Ls_val * id_cur + phi_m)
  )

  # 4) 역변환을 통한 3상 지령 전압 생성
  V = dq_to_abc(np.array([vd_ref, vq_ref]), theta_r)

  # 5) 모터 모델 (전기적/기계적 동특성)
  S = S_func(theta_r)
  term1 = np.dot(inv_Ls, Rs * i)
  term2 = np.dot(inv_Ls, V)
  term3 = np.dot(inv_Ls, phi_m * omega_r * S)
  di_dt = -term1 + term2 - term3

  Te = phi_m * np.dot(S, i)
  T_fric = B * omega_r
  domega_dt = (Te - T_load - T_fric) / J
  dtheta_dt = omega_r

  # 오일러 적분
  i += di_dt * dt
  omega_r += domega_dt * dt
  theta_r += dtheta_dt * dt
  theta_unwrapped += dtheta_dt * dt

  theta_r = np.mod(theta_r + np.pi, 2 * np.pi) - np.pi

  # 결과 기록
  i_history[:, k] = i
  omega_history[k] = omega_r
  Te_history[k] = Te
  theta_history[k] = theta_unwrapped

# ==========================================
# 6. 결과 시각화
# ==========================================
plt.figure(figsize=(12, 10))

# 1. 속도 제어 결과
plt.subplot(4, 1, 1)
plt.plot(time, omega_history, "g-", label="Actual Speed (Omega_r)")
plt.plot(time, omega_ref_history, "k--", label="Reference Speed")
plt.ylabel("Speed [rad/s]")
plt.legend(loc="lower right")
plt.grid(True)
plt.title("PMSM Vector Control (dq Controller) Simulation")

# 2. 상 전류 결과
plt.subplot(4, 1, 2)
plt.plot(time, i_history[0], label="i_a", alpha=0.8)
plt.plot(time, i_history[1], label="i_b", alpha=0.8)
plt.plot(time, i_history[2], label="i_c", alpha=0.8)
plt.ylabel("Phase Current [A]")
plt.legend(loc="upper right")
plt.grid(True)

# 3. 전자기 토크 결과
plt.subplot(4, 1, 3)
plt.plot(time, Te_history, color="r", label="Te")
plt.axhline(y=T_load, color="k", linestyle="--", label="T_load")
plt.ylabel("Torque [N.m]")
plt.legend(loc="upper right")
plt.grid(True)

# 4. 회전자 각도 결과
plt.subplot(4, 1, 4)
plt.plot(
    time, theta_history, color="b", label="Rotor Angle (Unwrapped) [rad]"
)
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.show()