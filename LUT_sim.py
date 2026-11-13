import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# 한글 폰트 및 마이너스 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 모터 기본 파라미터 설정
poles = 8
pole_pairs = poles / 2  # 극쌍수 (4)
J = 0.036               # 회전자 관성 모멘트 [kg·m^2]
Rs = 0.0618             # 고정자 권선 저항 [Ω]

class SpeedControllerPI:
    """속도 제어를 위한 PI 제어기"""
    def __init__(self, kp, ki, output_limits=(-340, 340)):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.min_out, self.max_out = output_limits

    def compute(self, target_speed, current_speed, dt):
        error = target_speed - current_speed
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.min_out / (self.ki + 1e-6), self.max_out / (self.ki + 1e-6))
        output = (self.kp * error) + (self.ki * self.integral)
        return np.clip(output, self.min_out, self.max_out)

def dq_to_abc(dq0, b):
    """dq0 -> abc 역변환 (b: 전기각 [rad])"""
    sr32 = np.sqrt(3) / 2
    clac2 = np.array([
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
    abc = clac2.T @ park_3x3.T @ dq0
    return abc

def load_lut_interpolators():
    print("[정보] 쇄교자속 LUT CSV 파일 로딩 및 보간기 생성 중...")
    start_t = time.time()
    
    df_d = pd.read_csv('LUT_lamda_d.csv')
    df_q = pd.read_csv('LUT_lamda_q.csv')
    
    id_vals = np.sort(df_d['Id_A'].unique())
    iq_vals = np.sort(df_d['Iq_A'].unique())
    theta_vals = np.sort(df_d['theta_e_deg'].unique())
    
    n_id, n_iq, n_th = len(id_vals), len(iq_vals), len(theta_vals)
    
    grid_lamda_d = df_d['lamda_d_Wb'].values.reshape(n_th, n_iq, n_id).T
    grid_lamda_q = df_q['lamda_q_Wb'].values.reshape(n_th, n_iq, n_id).T
    
    interp_d = RegularGridInterpolator((id_vals, iq_vals, theta_vals), grid_lamda_d, bounds_error=False, fill_value=None)
    interp_q = RegularGridInterpolator((id_vals, iq_vals, theta_vals), grid_lamda_q, bounds_error=False, fill_value=None)
    
    print(f"[완료] LUT 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
    return interp_d, interp_q

def run_lut_ipmsm_simulation_with_voltage():
    interp_d, interp_q = load_lut_interpolators()
    
    dt = 0.0005             # 제어 주기 [s] (0.5ms)
    total_time = 0.05        # 총 시뮬레이션 시간 [s]
    total_steps = int(total_time / dt)
    
    target_speed_rpm = 1000.0  # 지령 속도 [RPM]
    load_torque = 50.0         # 부하 토크 [Nm]
    
    current_speed_rpm = 0.0
    current_radpsec = 0.0
    theta_mech_cum = 0.0       # 누적 기계각 [deg]
    
    speed_pi = SpeedControllerPI(kp=0.5, ki=10.0, output_limits=(-340, 340))
    
    # 히스토리 기록용 리스트
    time_history = []
    speed_history = []
    target_speed_history = []
    torque_history = []
    v_a_history, v_b_history, v_c_history = [], [], []
    lamda_d_history, lamda_q_history = [], []
    
    # 이전 자속 값 저장용 (미분 계산용)
    lamda_d_old = 0.0
    lamda_q_old = 0.0
    
    print(f"\n==================================================")
    print(f" IPMSM LUT 기반 FOC 시뮬레이션 (3상 전압 포함) 시작")
    print(f"==================================================")
    
    start_sim_time = time.time()
    
    for step in range(total_steps):
        current_time = step * dt
        
        # 전류 지령 생성
        id_ref = 0.0
        iq_ref = speed_pi.compute(target_speed_rpm, current_speed_rpm, dt)
        
        # 전기각 계산 (-180 ~ 180도 정규화)
        theta_elec_deg = (theta_mech_cum * pole_pairs) % 360.0
        if theta_elec_deg > 180.0:
            theta_elec_deg -= 360.0
        theta_elec_rad = np.radians(theta_elec_deg)
        omega_e = current_radpsec * pole_pairs
        
        # LUT 자속 조회
        query_point = np.array([[id_ref, iq_ref, theta_elec_deg]])
        lamda_d = interp_d(query_point)[0]
        lamda_q = interp_q(query_point)[0]
        
        # 자속 시간 미분 (첫 스텝 제외)
        if step == 0:
            d_lamda_d_dt = 0.0
            d_lamda_q_dt = 0.0
        else:
            d_lamda_d_dt = (lamda_d - lamda_d_old) / dt
            d_lamda_q_dt = (lamda_q - lamda_q_old) / dt
            
        lamda_d_old = lamda_d
        lamda_q_old = lamda_q
        
        # dq 전압 방정식 계산
        v_d = Rs * id_ref + d_lamda_d_dt - omega_e * lamda_q
        v_q = Rs * iq_ref + d_lamda_q_dt + omega_e * lamda_d
        
        # dq0 -> 3상 전압(Va, Vb, Vc) 변환
        v_abc = dq_to_abc(np.array([v_d, v_q, 0.0]), theta_elec_rad)
        
        # 전자기 토크 계산
        developed_torque = 1.5 * pole_pairs * (lamda_d * iq_ref - lamda_q * id_ref)
        
        # 기계 동역학 연동
        accel_rad_s2 = (developed_torque - load_torque) / J
        current_radpsec += accel_rad_s2 * dt
        current_speed_rpm = current_radpsec * 30 / np.pi
        
        delta_mech_deg = (current_radpsec * dt) * (180 / np.pi)
        theta_mech_cum += delta_mech_deg
        
        # 기록
        time_history.append(current_time)
        speed_history.append(current_speed_rpm)
        target_speed_history.append(target_speed_rpm)
        torque_history.append(developed_torque)
        v_a_history.append(v_abc[0])
        v_b_history.append(v_abc[1])
        v_c_history.append(v_abc[2])
        lamda_d_history.append(lamda_d)
        lamda_q_history.append(lamda_q)
        
        if step % 400 == 0:
            print(f"[Time {current_time:5.3f}s] 속도: {current_speed_rpm:7.2f} RPM | Va: {v_abc[0]:6.2f} V | 토크: {developed_torque:6.2f} Nm")

    print(f"\n[완료] 시뮬레이션 소요 시간: {time.time() - start_sim_time:.2f}초")

    # --- 시각화 ---
    print("[결과 그래프 생성 중...]")
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # 1. 속도
    axes[0].plot(time_history, speed_history, label='Actual Speed (RPM)', color='blue', linewidth=2)
    axes[0].plot(time_history, target_speed_history, label='Target Speed (RPM)', color='red', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel('Speed [RPM]')
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title('IPMSM LUT Simulation with 3-Phase Voltages')
    
    # 2. 토크
    axes[1].plot(time_history, torque_history, label='Developed Torque [Nm]', color='darkorange', linewidth=1.5)
    axes[1].axhline(y=load_torque, color='gray', linestyle=':', label='Load Torque')
    axes[1].set_ylabel('Torque [Nm]')
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. 3상 전압
    axes[2].plot(time_history, v_a_history, label='Phase A ($V_a$)', color='red', linewidth=1.2)
    axes[2].plot(time_history, v_b_history, label='Phase B ($V_b$)', color='green', linewidth=1.2)
    axes[2].plot(time_history, v_c_history, label='Phase C ($V_c$)', color='blue', linewidth=1.2)
    axes[2].set_ylabel('Phase Voltage [V]')
    axes[2].legend(loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # 4. 쇄교자속
    axes[3].plot(time_history, lamda_d_history, label='$\lambda_d$ [Wb]', color='purple', linewidth=1.5)
    axes[3].plot(time_history, lamda_q_history, label='$\lambda_q$ [Wb]', color='teal', linewidth=1.5)
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('Flux Linkage [Wb]')
    axes[3].legend(loc='best')
    axes[3].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('ipmsm_lut_simulation_with_voltage.png', dpi=300)
    plt.show()
    print("[완료] 'ipmsm_lut_simulation_with_voltage.png'로 저장되었습니다.")

if __name__ == '__main__':
    run_lut_ipmsm_simulation_with_voltage()