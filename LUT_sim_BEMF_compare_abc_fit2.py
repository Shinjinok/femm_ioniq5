import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# 한글 폰트 및 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 모터 기본 파라미터 설정
poles = 8
pole_pairs = poles / 2   # 극쌍수 (4)
Rs = 0.0618             # 고정자 권선 저항 [Ω]
J = 0.036


def dq_to_abc2(dq0, b):
    T = 2/3 * np.array([
        np.cos(b), np.cos(b - 2*np.pi/3), np.cos(b + 2*np.pi/3),
        -np.sin(b), -np.sin(b - 2*np.pi/3), -np.sin(b + 2*np.pi/3),
        1/2, 1/2, 1/2
    ]).reshape(3, 3)
    abc = np.linalg.inv(T) @ dq0
    return abc


def dq_to_abc(dq0, b):
    """dq0 -> abc 역변환 (b: 전기각 [rad])"""
    sr32 = np.sqrt(3) / 2
    clac2 = np.array([
        [1, -1/2, -1/2],
        [0, sr32, -sr32],
        [1, 1, 1]  # <- [1/2, 1/2, 1/2] 1/2에서 1로 변경해야 정확히 일치
    ])
    cb = np.cos(b)
    sb = np.sin(b)
    park_3x3 = np.array([
        [cb, sb, 0],
        [-sb, cb, 0],
        [0, 0, 1]
    ])
    abc = clac2.T @ park_3x3.T @ dq0
    return abc


def abc_to_dq(val_abc, b):
    """abc -> dq0 변환 (b: 전기각 [rad])"""
    sr32 = np.sqrt(3) / 2
    clac_3x3 = (2 / 3) * np.array([
        [1, -1/2, -1/2],
        [0, sr32, -sr32],
        [1/2, 1/2, 1/2]
    ])
    cb = np.cos(b)
    sb = np.sin(b)
    park_3x3 = np.array([
        [cb, sb, 0],
        [-sb, cb, 0],
        [0, 0, 1]
    ])
    dq0 = park_3x3 @ clac_3x3 @ val_abc
    return dq0


def abc_to_dq2(i_abc, b):
    T = 2/3 * np.array([
        np.cos(b), np.cos(b - 2*np.pi/3), np.cos(b + 2*np.pi/3),
        -np.sin(b), -np.sin(b - 2*np.pi/3), -np.sin(b + 2*np.pi/3),
        1/2, 1/2, 1/2
    ]).reshape(3, 3)
    dq0 = T @ i_abc
    return dq0


def load_lut_abc_interpolators():
    print("[정보] 3상(ABC) 쇄교자속 LUT CSV 파일 로딩 및 보간기 생성 중...")
    start_t = time.time()

    df_a = pd.read_csv('LUT_lamda_abcT_20261119_091223.csv')

    pivot_df_a = df_a.pivot_table(index=['Id_A', 'Iq_A'], columns='theta_e_deg', values='lamda_a_Wb')
    pivot_df_b = df_a.pivot_table(index=['Id_A', 'Iq_A'], columns='theta_e_deg', values='lamda_b_Wb')
    pivot_df_c = df_a.pivot_table(index=['Id_A', 'Iq_A'], columns='theta_e_deg', values='lamda_c_Wb')
    pivot_df_T = df_a.pivot_table(index=['Id_A', 'Iq_A'], columns='theta_e_deg', values='Torque')

    id_axis = pivot_df_a.index.get_level_values('Id_A').unique().values
    iq_axis = pivot_df_a.index.get_level_values('Iq_A').unique().values
    theta_axis = pivot_df_a.columns.values

    matrix_3d_a = pivot_df_a.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_b = pivot_df_b.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_c = pivot_df_c.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_T = pivot_df_T.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))

    interp_a = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_a, method="cubic", bounds_error=False, fill_value=None)
    interp_b = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_b, method="cubic", bounds_error=False, fill_value=None)
    interp_c = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_c, method="cubic", bounds_error=False, fill_value=None)
    interp_T = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_T, method="cubic", bounds_error=False, fill_value=None)

    print(f"[완료] ABC 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
    return interp_a, interp_b, interp_c, interp_T, id_axis, iq_axis, theta_axis


def eval_lamda_dq0(interp_a, interp_b, interp_c, id_val, iq_val, theta_elec_deg, theta_elec_rad):
    pt = np.array([[id_val, iq_val, theta_elec_deg]])
    lamda_abc = np.array([
        interp_a(pt)[0],
        interp_b(pt)[0],
        interp_c(pt)[0]
    ])
    lamda_dq0 = abc_to_dq(lamda_abc, theta_elec_rad)
    return lamda_abc, lamda_dq0


def compute_incremental_inductance_3x3(
    interp_a, interp_b, interp_c,
    id_val, iq_val, theta_elec_deg, theta_elec_rad,
    current_step=5.0, L00=0.0
):
    """
    3x3 증분 인덕턴스 행렬 L_inc 계산:
    [[L_dd, L_dq, L_d0],
     [L_qd, L_qq, L_q0],
     [L_0d, L_0q, L_00]]
    """
    _, lam_id_plus = eval_lamda_dq0(interp_a, interp_b, interp_c, id_val + current_step, iq_val, theta_elec_deg, theta_elec_rad)
    _, lam_id_minus = eval_lamda_dq0(interp_a, interp_b, interp_c, id_val - current_step, iq_val, theta_elec_deg, theta_elec_rad)
    dlam_did = (lam_id_plus - lam_id_minus) / (2.0 * current_step)  # [L_dd, L_qd, L_0d]

    _, lam_iq_plus = eval_lamda_dq0(interp_a, interp_b, interp_c, id_val, iq_val + current_step, theta_elec_deg, theta_elec_rad)
    _, lam_iq_minus = eval_lamda_dq0(interp_a, interp_b, interp_c, id_val, iq_val - current_step, theta_elec_deg, theta_elec_rad)
    dlam_diq = (lam_iq_plus - lam_iq_minus) / (2.0 * current_step)  # [L_dq, L_qq, L_0q]

    # LUT에 i0 축이 없으므로 d/di0 열은 [0.0, 0.0, L00]으로 구성
    dlam_di0 = np.array([0.0, 0.0, L00], dtype=float)

    L_inc_3x3 = np.column_stack([dlam_did, dlam_diq, dlam_di0])
    return L_inc_3x3


class SpeedControllerPI:
    """모터 속도 제어를 위한 PI 제어기"""
    def __init__(self, kp, ki, output_limits=(-340, 340)):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.min_out, self.max_out = output_limits

    def compute(self, target_speed, current_speed, dt):
        error = (target_speed - current_speed)
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.min_out / (self.ki + 1e-6), self.max_out / (self.ki + 1e-6))
        output = (self.kp * error) + (self.ki * self.integral)
        return np.clip(output, self.min_out, self.max_out)


def check_degree(degree):
    degree %= 360.0
    if degree < 0.0:
        degree += 360.0
    return np.radians(degree), degree


def run_bemf_simulation_at_141rpm():
    interp_a, interp_b, interp_c, interp_T, id_axis, iq_axis, theta_axis = load_lut_abc_interpolators()

    dt = 0.0001       # 제어 주기 [s]
    total_time = 0.2   # 총 시뮬레이션 시간 [s]
    total_steps = int(total_time / dt)

    current_radpsec = 0.0
    theta_mech_cum = 0.0  # 누적 기계각 [deg]

    time_history = []
    speed_history = []
    v_a_history = []
    v_a0_history = []
    v_d_history = []
    v_d1_history = []
    temp1_history, temp2_history = [], []
    idq0_history = []
    L_inc_history = []
    Torque_histroy = []
    Torque_histroy2 = []
    Torque_histroy3 = []

    id_val = 0.0
    iq_val = 0.0

    print("\n==================================================")
    print(" IPMSM 141 RPM BEMF(역기전력) 3상 전압 시뮬레이션 시작 (ABC LUT 연동)")
    print("==================================================")

    lamda_dq0_old = np.zeros(3)
    lamda_abc_old = np.zeros(3)

    xJ = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  0]
    ])
    load_torque = 0
    bemf = np.zeros(3)
    temp1 = np.zeros(3)
    temp2 = np.zeros(3)
    i_dq0 = np.zeros(3)
    v_dq0_cmd = np.zeros(3)  # 인가 전압 [Vd, Vq, V0]

    for step in range(total_steps):
        current_time = step * dt

        # 전기각 계산 (0 ~ 360도 정규화)
        theta_elec_rad, theta_elec_deg = check_degree(theta_mech_cum * pole_pairs)

        # 0 전류 쇄교자속
        _, lamda_dq0_0 = eval_lamda_dq0(interp_a, interp_b, interp_c, 0.0, 0.0, theta_elec_deg, theta_elec_rad)

        # 현재 쇄교자속 조회
        lamda_abc, lamda_dq0 = eval_lamda_dq0(interp_a, interp_b, interp_c, id_val, iq_val, theta_elec_deg, theta_elec_rad)

        # 3x3 증분 인덕턴스 행렬 L_inc 계산
        current_step = 1.0  # [A]
        L_inc_3x3 = compute_incremental_inductance_3x3(
            interp_a, interp_b, interp_c,
            id_val, iq_val, theta_elec_deg, theta_elec_rad,
            current_step=current_step,
            L00=0.0
        )
        L_dd, L_dq, L_d0 = L_inc_3x3[0, :]
        L_qd, L_qq, L_q0 = L_inc_3x3[1, :]
        L_0d, L_0q, L_00 = L_inc_3x3[2, :]

        # 각도 변위 미분 (gamma_dq0 = d(lamda_dq0)/d(theta_e))
        degree_step = 1
        dth1_rad, dth1_deg = check_degree(theta_elec_deg - degree_step)
        dth2_rad, dth2_deg = check_degree(theta_elec_deg + degree_step)

        _, lamda_dq0_th1 = eval_lamda_dq0(interp_a, interp_b, interp_c, id_val, iq_val, dth1_deg, dth1_rad)
        _, lamda_dq0_th2 = eval_lamda_dq0(interp_a, interp_b, interp_c, id_val, iq_val, dth2_deg, dth2_rad)

        gamma_dq0 = (lamda_dq0_th2 - lamda_dq0_th1) / (degree_step * np.pi / 90.0)

        We = current_radpsec * pole_pairs

        if step == 0:
            V_abc = np.zeros(3)
            bemf = np.zeros(3)
            temp1 = np.zeros(3)
            temp2 = np.zeros(3)
        else:
            V_abc = (lamda_abc - lamda_abc_old) / dt
            bemf = (lamda_dq0 - lamda_dq0_old) / dt + We * (xJ @ lamda_dq0)

            temp1 = np.array([-We * L_qq * iq_val, We * (L_dd * id_val + lamda_dq0_0[0]), 0.0])
            temp2 = np.cross(np.array([0.0, 0.0, We]), lamda_dq0)

            # v_dq0 = Rs * i_dq0 + L_inc_3x3 * (di_dq0/dt) + We * (gamma_dq0 + xJ * lamda_dq0)
            rhs_dq0 = v_dq0_cmd - Rs * i_dq0 - We * (gamma_dq0 + (xJ @ lamda_dq0))

            # LUT에 i0 축이 없으므로 dq 2x2 블록으로 did/dt, diq/dt 풀이 (i0는 0 유지)
            L_2x2 = L_inc_3x3[:2, :2]
            if abs(np.linalg.det(L_2x2)) > 1e-12:
                didq_dt = np.linalg.solve(L_2x2, rhs_dq0[:2])
            else:
                didq_dt = rhs_dq0[:2] / np.array([max(abs(L_dd), 1e-6), max(abs(L_qq), 1e-6)])

            i_dq0[0] += didq_dt[0] * dt
            i_dq0[1] += didq_dt[1] * dt
            i_dq0[2] = 0.0

        T_cog = 1.5 * pole_pairs * np.dot(gamma_dq0, i_dq0)
        id_val = i_dq0[0]
        iq_val = i_dq0[1]

        Te_z = 1.5 * pole_pairs * (lamda_dq0_0[0] * iq_val + (L_dd - L_qq) * id_val * iq_val) + T_cog
        Te_z2 = (1.5 * pole_pairs * np.cross(lamda_dq0, i_dq0))[2] + T_cog

        query_point_T = np.array([[id_val, iq_val, theta_elec_deg]])
        torque = interp_T(query_point_T)[0]

        Torque_histroy.append(Te_z)
        Torque_histroy2.append(Te_z2)
        Torque_histroy3.append(torque)

        # 모터 동역학 방정식 계산 및 회전 각도 산출
        accel_rad_s2 = (Te_z - load_torque) / J
        current_radpsec += accel_rad_s2 * dt
        current_speed_rpm = current_radpsec * 30.0 / np.pi

        lamda_abc_old = lamda_abc
        lamda_dq0_old = lamda_dq0

        V_abc_from_bemf = dq_to_abc(bemf, theta_elec_rad)
        V_dq0_from_abc = abc_to_dq(V_abc, theta_elec_rad)

        current_radpsec = 141/60*2*np.pi
        delta_mech_deg = (current_radpsec * dt) * (180.0 / np.pi)
        theta_mech_cum += delta_mech_deg

        time_history.append(current_time)
        speed_history.append(current_speed_rpm)
        v_a_history.append(V_abc)
        v_a0_history.append(V_abc_from_bemf)
        v_d_history.append(bemf)
        v_d1_history.append(V_dq0_from_abc)
        idq0_history.append(i_dq0.copy())
        L_inc_history.append(L_inc_3x3.copy())
        temp1_history.append(temp1)
        temp2_history.append(temp2)

    print("[완료] BEMF 시뮬레이션 연산 완료. 그래프 생성 중...")

    # --- 시각화 ---
    fig, axes = plt.subplots(2, 2, figsize=(16*2/3, 9*2/3), sharex=True)
    colors_rgb = ['red', 'green', 'blue']

    axes[0, 0].plot(time_history, Torque_histroy, label='$\\phi_f i_q + (L_{dd} - L_{qq}) i_d i_q$', color='blue', linewidth=1, alpha=0.5)
    axes[0, 0].plot(time_history, Torque_histroy2, label='$\\lambda \\times I$', color='green', linewidth=2, linestyle='--')
    axes[0, 0].plot(time_history, Torque_histroy3, label='$FEMM Torque$', color='red', linewidth=1)
    axes[0, 0].plot(time_history, speed_history, label='$Rpm$', color='purple', linewidth=1)
    axes[0, 0].set_ylabel('Torque')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].set_title('IPMSM Back-EMF (BEMF) 3-Phase Voltages at 141 RPM (ABC LUT)')

    labels = ["$V_a$", "$V_b$", "$V_c$"]
    for i in range(3):
        axes[1, 0].plot(time_history, np.array(v_a_history)[:, i], label=f'd$\\lambda$/dt ({labels[i]})', color=colors_rgb[i], linewidth=1.2, alpha=0.5)
        axes[1, 0].plot(time_history, np.array(v_a0_history)[:, i], label=f'd$\\lambda_x$/dt+$\\omega \\times \\lambda_x$ ({labels[i]})', color=colors_rgb[i], linewidth=1.2, linestyle='--')
    axes[1, 0].set_ylabel('BEMF Voltage [V]')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    labels = ["$V_d$", "$V_q$", "$V_0$"]
    for i in range(3):
        axes[0, 1].plot(time_history, np.array(v_d1_history)[:, i], label=f'V_dq0_from_abc({labels[i]})', color=colors_rgb[i], linewidth=1.2, alpha=0.5)
        axes[0, 1].plot(time_history, np.array(v_d_history)[:, i], label=f'$\\gamma+J\\lambda$ ({labels[i]})', color=colors_rgb[i], linewidth=1.2, linestyle='--')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('BEMF Voltage [V]')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    labels = ["$d$", "$q$", "$0$"]
    for i in range(3):
        axes[1, 1].plot(time_history, np.array(temp1_history)[:, i], label=f'temp1({labels[i]})', color=colors_rgb[i], linewidth=1.2, alpha=0.5)
        axes[1, 1].plot(time_history, np.array(temp2_history)[:, i], label=f'temp2({labels[i]})', color=colors_rgb[i], linewidth=1.2, linestyle='--')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('BEMF Voltage [V]')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('ipmsm_bemf_141rpm_abc.png', dpi=300)
    plt.show()
    print("[완료] BEMF 결과 그래프가 'ipmsm_bemf_141rpm_abc.png'로 저장되었습니다.")


if __name__ == '__main__':
    run_bemf_simulation_at_141rpm()