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
pole_pairs = poles / 2  # 극쌍수 (4)
Rs = 0.0618             # 고정자 권선 저항 [Ω]

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
    
    interp_d = RegularGridInterpolator((id_vals, iq_vals, theta_vals), grid_lamda_d,method='cubic', bounds_error=False, fill_value=None)
    interp_q = RegularGridInterpolator((id_vals, iq_vals, theta_vals), grid_lamda_q,method='cubic', bounds_error=False, fill_value=None)
    
    print(f"[완료] LUT 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
    return interp_d, interp_q

def run_bemf_simulation_at_141rpm():
    interp_d, interp_q = load_lut_interpolators()
    
    dt = 0.0001             # 제어 주기 [s] (정밀한 BEMF 파형을 위해 0.2ms 설정)
    total_time = 0.05        # 총 시뮬레이션 시간 [s] (141 RPM에서 약 2~3 전기각 주기 포함)
    total_steps = int(total_time / dt)
    
    target_speed_rpm = 141.0  # 목표 BEMF 측정 속도 [RPM]
    current_radpsec = target_speed_rpm * np.pi / 30.0
    omega_e = current_radpsec * pole_pairs
    
    theta_mech_cum = 0.0       # 누적 기계각 [deg]
    
    # 히스토리 기록용 리스트
    time_history = []
    speed_history = []
    v_a_history, v_b_history, v_c_history = [], [], []
    lamda_d_history, lamda_q_history = [], []
    
    id_val = 0.0  # 무부하 BEMF이므로 전류는 0
    iq_val = 0.0
    
    lamda_d_old = 0.0
    lamda_q_old = 0.0
    
    print(f"\n==================================================")
    print(f" IPMSM 141 RPM BEMF(역기전력) 3상 전압 시뮬레이션 시작")
    print(f"==================================================")
    
    for step in range(total_steps):
        current_time = step * dt
        
        # 전기각 계산 (-180 ~ 180도 정규화)
        theta_elec_deg = (theta_mech_cum * pole_pairs) % 360.0
        if theta_elec_deg > 180.0:
            theta_elec_deg -= 360.0
        theta_elec_rad = np.radians(theta_elec_deg)
        
        # LUT 자속 조회
        query_point = np.array([[id_val, iq_val, theta_elec_deg]])
        lamda_d = interp_d(query_point)[0]
        lamda_q = interp_q(query_point)[0]
        
        # 자속 시간 미분 (BEMF 계산 핵심: dλ/dt)
        if step == 0:
            d_lamda_d_dt = 0.0
            d_lamda_q_dt = 0.0
        else:
            d_lamda_d_dt = (lamda_d - lamda_d_old) / dt
            d_lamda_q_dt = (lamda_q - lamda_q_old) / dt
            
        lamda_d_old = lamda_d
        lamda_q_old = lamda_q
        
        # 역기전력(BEMF) dq 전압 방정식 계산 (전류=0 이므로 저항 낙차 무시 또는 포함)
        v_d = d_lamda_d_dt - omega_e * lamda_q
        v_q = d_lamda_q_dt + omega_e * lamda_d
        
        # dq0 -> 3상 BEMF 전압(Va, Vb, Vc) 변환
        v_abc = dq_to_abc(np.array([v_d, v_q, 0.0]), theta_elec_rad)
        
        # 기계각 업데이트 (141 RPM 유지)
        delta_mech_deg = (current_radpsec * dt) * (180 / np.pi)
        theta_mech_cum += delta_mech_deg
        
        # 기록
        time_history.append(current_time)
        speed_history.append(target_speed_rpm)
        v_a_history.append(v_abc[0])
        v_b_history.append(v_abc[1])
        v_c_history.append(v_abc[2])
        lamda_d_history.append(lamda_d)
        lamda_q_history.append(lamda_q)

    print("[완료] BEMF 시뮬레이션 연산 완료. 그래프 생성 중...")

    # --- 시각화 ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # 1. 속도 (141 RPM)
    axes[0].plot(time_history, speed_history, label='Speed (141 RPM)', color='blue', linewidth=2)
    axes[0].set_ylabel('Speed [RPM]')
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title('IPMSM Back-EMF (BEMF) 3-Phase Voltages at 141 RPM')
    
    # 2. 3상 BEMF 전압 파형
    axes[1].plot(time_history, v_a_history, label='Phase A BEMF ($V_a$)', color='red', linewidth=1.2)
    axes[1].plot(time_history, v_b_history, label='Phase B BEMF ($V_b$)', color='green', linewidth=1.2)
    axes[1].plot(time_history, v_c_history, label='Phase C BEMF ($V_c$)', color='blue', linewidth=1.2)
    axes[1].set_ylabel('BEMF Voltage [V]')
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. 쇄교자속
    axes[2].plot(time_history, lamda_d_history, label='$\lambda_d$ [Wb]', color='purple', linewidth=1.5)
    axes[2].plot(time_history, lamda_q_history, label='$\lambda_q$ [Wb]', color='teal', linewidth=1.5)
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('Flux Linkage [Wb]')
    axes[2].legend(loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('ipmsm_bemf_141rpm.png', dpi=300)
    plt.show()
    print("[완료] BEMF 결과 그래프가 'ipmsm_bemf_141rpm.png'로 저장되었습니다.")

if __name__ == '__main__':
    run_bemf_simulation_at_141rpm()