import Ioniq5_ev_model as ev_model
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 제어기 영역
# ==========================================
T_sim = 0.5
dt = 1e-4
time = np.arange(0, T_sim, dt)
N = len(time)
Ld = 8.5e-3  # d축 인덕턴스 [H]
Lq = 8.5e-3  # q축 인덕턴스 [H]

print(f'시뮬레이션 데이터 수 (N): {N}')
print(f'시간 배열 크기: {time.shape}')

# ==========================================
# 3. 제어기(PI Controller) 변수 초기화
# ==========================================
# 속도 제어기 게인
Kp_spd, Ki_spd = 0.2, 5.0
integral_spd_err = 0.0

# 전류 제어기 게인 (d축, q축)
Kp_i, Ki_i = 2.0, 50.0
integral_id_err = 0.0
integral_iq_err = 0.0


# 속도 지령 함수 (0.1초일 때 50 rad/s로 스텝 입력)
def get_speed_ref(t):
  return 6 if t >= 0.1 else 0.0


# ==========================================
# 4. 시뮬레이션 상태 변수 초기화
# ==========================================
i_abc = np.zeros(3)
omega_r = 0.0
theta_r = 0.0

# 데이터 기록용 배열 (v_history 추가)
i_history = np.zeros((3, N))
v_history = np.zeros((3, N))  # 3상 전압 기록용 배열
omega_r_history = np.zeros(N)
omega_r_ref_history = np.zeros(N)
theta_r_history = np.zeros(N)
Tr_history = np.zeros(N)

# ==========================================
# 5. 메인 제어 및 시뮬레이션 루프
# ==========================================
for k in range(N):
  t = time[k]
  theta_e = ev_model.pole_pairs * theta_r  # 전기각 계산
  omega_e = ev_model.pole_pairs * omega_r  # 전기각 속도 계산

  # --- [1] 속도 제어기 (Outer Loop) ---
  omega_r_ref = get_speed_ref(t)
  omega_r_ref_history[k] = omega_r_ref

  spd_err = omega_r_ref - omega_r
  integral_spd_err += spd_err * dt
  iq_ref = Kp_spd * spd_err + Ki_spd * integral_spd_err
  iq_ref = np.clip(iq_ref, -20, 20)  # 전류 지령 제한
  id_ref = 0  # SPMSM이므로 d축 전류는 0으로 제어

  # --- [2] 피드백 전류 측정 및 dq 변환 ---
  id_cur, iq_cur = ev_model.abc_to_dq(i_abc, theta_e)

  # --- [3] 전류 제어기 (Inner Loop + 디커플링 제어) ---
  id_err = id_ref - id_cur
  integral_id_err += id_err * dt
  vd_ref = (
      Kp_i * id_err + Ki_i * integral_id_err - omega_e * Lq * iq_cur
  )

  iq_err = iq_ref - iq_cur
  integral_iq_err += iq_err * dt
  vq_ref = (
      Kp_i * iq_err
      + Ki_i * integral_iq_err
      + omega_e * (Ld * id_cur + ev_model.phi_m)
  )

  # --- [4] 전압 역변환 (dq -> abc) ---
  v_abc = ev_model.dq_to_abc(np.array([vd_ref, vq_ref]), theta_e)
  v_history[:, k] = v_abc  # 3상 전압 기록

  # --- [5] 모터 동역학 계산 ---
  i_abc, omega_r, theta_r, Tr = ev_model.motor_dynamics(
      i_abc, v_abc, omega_r, theta_r, dt
  )

  # 기록
  i_history[:, k] = i_abc
  omega_r_history[k] = omega_r
  theta_r_history[k] = theta_r  # type: ignore
  Tr_history[k] = Tr

# ==========================================
# 6. 결과 시각화 (4개 서브플롯)
# ==========================================
plt.figure(figsize=(10, 10))

# 1. 속도 응답 그래프
plt.subplot(4, 1, 1)
plt.plot(time, omega_r_history, 'g-', label='실제 속도 ($\omega_r$)')
plt.plot(time, omega_r_ref_history, 'k--', label='지령 속도')
plt.ylabel('Speed [rad/s]')
plt.legend(loc='lower right')
plt.grid(True)
plt.title('PMSM 벡터 제어(DQ Control) 시뮬레이션')

# 2. 3상 전압 그래프 (추가됨)
plt.subplot(4, 1, 2)
plt.plot(time, v_history[0], label='$v_a$', alpha=0.8)
plt.plot(time, v_history[1], label='$v_b$', alpha=0.8)
plt.plot(time, v_history[2], label='$v_c$', alpha=0.8)
plt.ylabel('Phase Voltage [V]')
plt.legend(loc='upper right')
plt.grid(True)

# 3. 상 전류 그래프
plt.subplot(4, 1, 3)
plt.plot(time, i_history[0], label='$i_a$', alpha=0.8)
plt.plot(time, i_history[1], label='$i_b$', alpha=0.8)
plt.plot(time, i_history[2], label='$i_c$', alpha=0.8)
plt.ylabel('Phase Current [A]')
plt.legend(loc='upper right')
plt.grid(True)

# 4. 전자기 토크 그래프
plt.subplot(4, 1, 4)
plt.plot(time, Tr_history, 'r-', label='전자기 토크 ($T_r$)')
plt.axhline(y=ev_model.T_load, color='k', linestyle='--', label='부하 토크')
plt.xlabel('Time [s]')
plt.ylabel('Torque [N·m]')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()