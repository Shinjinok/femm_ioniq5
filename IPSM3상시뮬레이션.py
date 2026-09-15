import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_and_interpolate_inductance(
    csv_filename="ioniq5-13.FEM_inductance_table.csv",
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


# ==========================================
# 1. 시뮬레이션 및 모터 파라미터 설정
# ==========================================
T_sim = 1.0  # 시뮬레이션 총 시간 [s]
dt = 1e-4  # 샘플링 시간 [s]
time = np.arange(0, T_sim, dt)
N = len(time)

Rs = 1.2  # 고정자 저항 [ohm]
phi_m = 0.12  # 영구자석 쇄교자속 [Wb]
J = 0.0015  # 회전자 관성모멘트 [kg*m^2]
B = 0.0001  # 마찰계수 [N*m*s/rad]
T_load = 0.5  # 부하 토크 [N*m]

# 인덕턴스 테이블 로드 및 보간기 생성
interpolators, df_table = load_and_interpolate_inductance(
    "ioniq5-13.FEM_all_currents_inductance_summary.csv"
)


def get_inductance_matrix(i_abc, theta_r):
    """현재 상전류 크기 및 회전자 각도를 반영하여 안정적인 3x3 인덕턴스 행렬 구성"""
    i_norm = np.max(np.abs(i_abc))

    # 보간기로부터 Center 및 Amplitude 획득 (μH -> H 변환: * 1e-6)
    La = float(interpolators["Laa_Center_uH"](i_norm)) * 1e-6
    Lb = float(interpolators["Laa_Amplitude_uH"](i_norm)) * 1e-6

    La2 = float(interpolators["Lab_Center_uH"](i_norm)) * 1e-6
    Lb2 = float(interpolators["Lab_Amplitude_uH"](i_norm)) * 1e-6

    La3 = float(interpolators["Lac_Center_uH"](i_norm)) * 1e-6
    Lb3 = float(interpolators["Lac_Amplitude_uH"](i_norm)) * 1e-6

    th_e = 4 * theta_r  # 8극 모터 (극쌍수 = 4)

    # 공간 고조파 및 진폭 반영 (과도한 변동 방지를 위해 정규화된 형태 유지)
    l11 =  La - Lb * np.cos(2 * th_e)
    l12 =  La2 - Lb2 * np.cos(2 * th_e - 2 * np.pi / 3)
    l13 =  La3 - Lb3 * np.cos(2 * th_e + 2 * np.pi / 3)
    l22 =  La - Lb * np.cos(2 * th_e + 2 * np.pi / 3)
    l23 =  La2 - Lb2 * np.cos(2 * th_e)
    l33 =  La - Lb * np.cos(2 * th_e - 2 * np.pi / 3)

    # 대칭 3상 인덕턴스 행렬 구성
    Ls = np.array([[l11, l12, l13], 
                   [l12, l22, l23], 
                   [l13, l23, l33]])

    # 수치 발산 방지를 위한 행렬 양정정(Positive Definite) 보정
    # 최소 자기인덕턴스 보장 및 상호인덕턴스 크기 제한
    """ min_self = max(Laa_c * 0.5, 1e-5)
    for r in range(3):
        Ls[r, r] = max(Ls[r, r], min_self) """

    return Ls


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
# 3. 제어기 게인(PI Gain) 설정
# ==========================================
Kp_spd, Ki_spd = 0.5, 5.0
integral_spd_err = 0.0

Kp_id, Ki_id = 10.0, 50.0
integral_id_err = 0.0

Kp_iq, Ki_iq = 10.0, 50.0
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
# 5. 수치 해석 및 제어 루프 (안정화된 서브스테핑 적용)
# ==========================================
sub_steps = 5  # 수치 적분 안정성을 위한 서브스텝 분할
dt_sub = dt / sub_steps

for k in range(N):
    t = time[k]

    # 1) 속도 제어기 (Outer Loop)
    omega_ref = get_speed_ref(t)
    omega_ref_history[k] = omega_ref

    spd_err = omega_ref - omega_r
    integral_spd_err += spd_err * dt
    iq_ref = Kp_spd * spd_err + Ki_spd * integral_spd_err
    id_ref = 0.0

    # 안티윈드업 (적분기 포화 방지)
    integral_spd_err = np.clip(integral_spd_err, -200, 200)

    # 2) 피드백 전류 측정 및 dq 변환
    id_cur, iq_cur = abc_to_dq(i, theta_r)

    # 3) 현재 전류 상태에 맞는 등가 인덕턴스 계산 (디커플링 및 제어용)
    Ls_matrix = get_inductance_matrix(i, theta_r)
    Ls_equivalent = np.mean(np.diag(Ls_matrix))

    # 4) 전류 제어기 (Inner Loop + 디커플링 항)
    id_err = id_ref - id_cur
    integral_id_err += id_err * dt
    integral_id_err = np.clip(integral_id_err, -100, 100)
    vd_ref = (
        Kp_id * id_err
        + Ki_id * integral_id_err
        - omega_r * Ls_equivalent * iq_cur
    )

    iq_err = iq_ref - iq_cur
    integral_iq_err += iq_err * dt
    integral_iq_err = np.clip(integral_iq_err, -100, 100)
    vq_ref = (
        Kp_iq * iq_err
        + Ki_iq * integral_iq_err
        + omega_r * (Ls_equivalent * id_cur + phi_m)
    )

    # 5) 역변환을 통한 3상 지령 전압 생성
    V = dq_to_abc(np.array([vd_ref, vq_ref]), theta_r)

    # 6) 모터 모델 (수치 폭발 방지를 위한 서브스테핑 오일러 적분)
    for _ in range(sub_steps):
        Ls_sub = get_inductance_matrix(i, theta_r)
        # 역행렬 연산 시 조건수(Condition Number) 안정화 처리
        inv_Ls = np.linalg.inv(
            Ls_sub + np.eye(3) * 1e-7
        )  # 댐핑 팩터 추가로 발산 방지

        S = S_func(theta_r)
        term1 = np.dot(inv_Ls, Rs * i)
        term2 = np.dot(inv_Ls, V)
        term3 = np.dot(inv_Ls, phi_m * omega_r * S)
        di_dt = -term1 + term2 - term3

        Te = phi_m * np.dot(S, i)
        T_fric = B * omega_r
        domega_dt = (Te - T_load - T_fric) / J
        dtheta_dt = omega_r

        i += di_dt * dt_sub
        omega_r += domega_dt * dt_sub
        theta_r += dtheta_dt * dt_sub
        theta_unwrapped += dtheta_dt * dt_sub

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
plt.title(
    "PMSM Vector Control with Stabilized FEM Inductance Table Interpolation"
)

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