import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 시뮬레이션 및 모터 파라미터 설정
# ==========================================
T_sim = 0.5   # 시뮬레이션 총 시간 [s]
dt = 1e-6     # 샘플링 시간 [s]
time = np.arange(0, T_sim, dt)
N = len(time)

Rs = 24.8*1e-3 * np.eye(3)  # 고정자 저항 [ohm]
phi_m = 0.0012         # 영구자석 쇄교자속 [Wb]
J = 0.03               # 회전자 관성모멘트 [kg*m^2]
B = 0.0               # 마찰계수 [N*m*s/rad]
T_load = 0.0          # 부하 토크 [N*m]

# 인가 전압 조건
f_voltage = 100
omega_e_ref = 2 * np.pi * f_voltage  # 전기각 주파수 [rad/s]
V_amplitude = 5.0                    # 전압 진폭 [V]

# 극쌍수 (p = 1)
pole_pairs = 1

# 고정자 인덕턴스 행렬 (Ls) 정의
L_self = 348*1e-6
L_liquid = 3.48*1e-6
L_mutual = -34.45*1e-6
Ls = np.array([
    [L_liquid + L_self, L_mutual, L_mutual],
    [L_mutual, L_liquid + L_self, L_mutual],
    [L_mutual, L_mutual, L_liquid + L_self]
])

inv_Ls = np.linalg.inv(Ls)
print("인버스 행렬 inv_Ls:\n", inv_Ls)
print("인버스 행렬 inv_Ls 고유치:", np.linalg.eigvals(inv_Ls))


def S_func(theta_r):
    th_e = pole_pairs * theta_r
    return np.array(
        [
            -np.sin(th_e),
            -np.sin(th_e - 2 * np.pi / 3),
            -np.sin(th_e + 2 * np.pi / 3),
        ]
    )


# ==========================================
# 2. 상태 변수 초기화
# ==========================================
i_abc = np.zeros(3)          # 상 전류 [ia, ib, ic]
omega_r = 0.0         # 회전자 기계각 속도 [rad/s]
theta_r = 0.0          # 회전자 기계각 [rad]
theta_unwrapped = 0.0

i_history = np.zeros((3, N))
omega_history = np.zeros(N)
Te_history = np.zeros(N)
v_history = np.zeros((3, N))
theta_m_history = np.zeros(N)
theta_e_wrapped_history = np.zeros(N)

# term1, term2, term3 기록용 배열 추가 (3상 전체 기록)
term1_history = np.zeros((3, N))
term2_history = np.zeros((3, N))
term3_history = np.zeros((3, N))

# ==========================================
# 3. 수치 해석 루프
# ==========================================
for k in range(N):
    t = time[k]

    # 1) 3상 평형 정현파 전압 생성
    theta_e_current = omega_e_ref * t
    va = V_amplitude * np.cos(theta_e_current)
    vb = V_amplitude * np.cos(theta_e_current - 2 * np.pi / 3)
    vc = V_amplitude * np.cos(theta_e_current + 2 * np.pi / 3)
    V = np.array([va, vb, vc])
    v_history[:, k] = V

    # 2) 모터 동특성 수치 적분 (dt 적용)
    omega_e_inst = pole_pairs * omega_r
    e_back = omega_e_inst * phi_m * S_func(theta_e_current)

    term0 = np.dot(-inv_Ls, Rs)
    term1 = np.dot(term0, i_abc)
    term2 = np.dot(inv_Ls, V)
    term3 = np.dot(-inv_Ls, e_back)
    di_dt = term1 + term2 + term3

    # 항 기록 (a상 기준 혹은 전체 상)
    term1_history[:, k] = term1
    term2_history[:, k] = term2
    term3_history[:, k] = term3

    Te = phi_m * np.dot(i_abc, S_func(theta_e_current))
    T_fric = B * omega_r
    domega_dt = (pole_pairs * Te - T_load - T_fric) / J
    dtheta_dt = omega_r

    i_abc += di_dt * dt
    omega_r += domega_dt * dt
    theta_r += dtheta_dt * dt
    theta_unwrapped += dtheta_dt * dt

    #theta_r = np.mod(theta_r + np.pi, 2 * np.pi) - np.pi

    # 결과 기록
    i_history[:, k] = i_abc
    omega_history[k] = omega_r
    Te_history[k] = Te
    theta_m_history[k] = theta_unwrapped
    
    th_e_val = pole_pairs * theta_unwrapped
    theta_e_wrapped_history[k] = np.mod(th_e_val + np.pi, 2 * np.pi) - np.pi

# ==========================================
# 4. 결과 시각화 (term1, term2, term3 플롯 추가)
# ==========================================
fig, axes = plt.subplots(8, 1, figsize=(11, 16), sharex=True)

# 1. 3상 입력 전압
axes[0].plot(time, v_history[0], label='v_a', color='r')
axes[0].plot(time, v_history[1], label='v_b', color='g')
axes[0].plot(time, v_history[2], label='v_c', color='b')
axes[0].set_ylabel('상전압 [V]')
axes[0].set_title('모터 구동 시뮬레이션 결과 (Term 성분 포함)', fontsize=12, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle='--', alpha=0.6)

# 2. 3상 전류 파형
axes[1].plot(time, i_history[0], label='i_a (Phase A)', color='r')
axes[1].plot(time, i_history[1], label='i_b (Phase B)', color='g')
axes[1].plot(time, i_history[2], label='i_c (Phase C)', color='b')
axes[1].set_ylabel('상전류 [A]')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.6)

# 3. 전자기 토크
axes[2].plot(time, Te_history, label='전자기 토크 (Te)', color='purple')
axes[2].axhline(y=T_load, color='k', linestyle='--', label='부하 토크 (T_load)')
axes[2].set_ylabel('토크 [N.m]')
#axes[2].set_ylim(bottom=min(0, np.min(Te_history) * 1.1))
axes[2].legend(loc='upper right')
axes[2].grid(True, linestyle='--', alpha=0.6)

# 4. 모터 속도 [deg/s]
axes[3].plot(time, omega_history * 2 * np.pi, label='회전자 속도', color='darkorange')
axes[3].set_ylabel('속도 [deg/s]')
#axes[3].set_ylim(bottom=0)
axes[3].legend(loc='upper right')
axes[3].grid(True, linestyle='--', alpha=0.6)

# 5. 회전자 기계각 [deg]
axes[4].plot(time, theta_m_history * 2 * np.pi, label='기계각 (Mechanical Angle)', color='blue', lw=1.5)
axes[4].set_ylabel('각도 [deg]')
axes[4].legend(loc='upper left')
axes[4].grid(True, linestyle='--', alpha=0.6)

# 6. 전압 방정식 항 비교 (Term1: 저항 강하 성분 di/dt 기여분)
axes[5].plot(time, term1_history[0], label='Term 1 (a상 - 저항 성분)', color='brown')
axes[5].set_ylabel('Term 1 [A/s]')
axes[5].legend(loc='upper right')
axes[5].grid(True, linestyle='--', alpha=0.6)

# 7. 전압 방정식 항 비교 (Term2: 인가 전압 성분, Term3: 역기전력 성분)
axes[6].plot(time, term2_history[0], label='Term 2 (a상 - 인가전압 성분)', color='crimson')

axes[6].set_xlabel('시간 [s]')
axes[6].set_ylabel('Term 2,3 [A/s]')
axes[6].legend(loc='upper right')
axes[6].grid(True, linestyle='--', alpha=0.6)

# 8. 전압 방정식 항 비교 (Term4: 자기력 성분)
axes[7].plot(time, term3_history[0], label='Term 3 (a상 - 역기전력 성분)', color='teal')
axes[7].set_xlabel('시간 [s]')
axes[7].set_ylabel('Term 4 [A/s]')
axes[7].legend(loc='upper right')
axes[7].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
output_img = 'simulation_with_terms.png'
plt.savefig(output_img, dpi=300)
print(f"[시뮬레이션 그래프 저장 완료] {output_img}")
plt.show()