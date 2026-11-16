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

def dq_to_abc2(dq0, b):
    T = 2/3 * np.array([np.cos(b), np.cos(b - 2*np.pi/3), np.cos(b + 2*np.pi/3),
                        -np.sin(b), -np.sin(b - 2*np.pi/3), -np.sin(b + 2*np.pi/3),
                        1/2, 1/2, 1/2]).reshape(3, 3)
    abc = np.linalg.inv(T) @ dq0
    return abc

def load_lut_interpolators():
    print("[정보] 쇄교자속 LUT CSV 파일 로딩 및 보간기 생성 중...")
    start_t = time.time()
    
    df_d = pd.read_csv('LUT_lamda_d.csv')
    df_q = pd.read_csv('LUT_lamda_q.csv')
    df_0 = pd.read_csv('LUT_lamda_0.csv')
    
    pivot_df_d = df_d.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_d_Wb'
    )
    pivot_df_q = df_q.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_q_Wb'
    )
    pivot_df_0 = df_0.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_0_Wb'
    )
    
    # 피벗 테이블로부터 축(Axis) 값과 매트릭스 추출
    id_axis_d = pivot_df_d.index.get_level_values('Id_A').unique().values 
    iq_axis_d = pivot_df_d.index.get_level_values('Iq_A').unique().values  
    theta_axis_d = pivot_df_d.columns.values                               
    matrix_3d_d = pivot_df_d.values.reshape(len(id_axis_d), len(iq_axis_d), len(theta_axis_d))
    interp_d = RegularGridInterpolator((id_axis_d, iq_axis_d, theta_axis_d), matrix_3d_d, method="cubic")

    id_axis_q = pivot_df_q.index.get_level_values('Id_A').unique().values  
    iq_axis_q = pivot_df_q.index.get_level_values('Iq_A').unique().values  
    theta_axis_q = pivot_df_q.columns.values                                
    matrix_3d_q = pivot_df_q.values.reshape(len(id_axis_q), len(iq_axis_q), len(theta_axis_q))
    interp_q = RegularGridInterpolator((id_axis_q, iq_axis_q, theta_axis_q), matrix_3d_q, method="cubic")

    id_axis_0 = pivot_df_0.index.get_level_values('Id_A').unique().values  
    iq_axis_0 = pivot_df_0.index.get_level_values('Iq_A').unique().values  
    theta_axis_0 = pivot_df_0.columns.values                                
    matrix_3d_0 = pivot_df_0.values.reshape(len(id_axis_0), len(iq_axis_0), len(theta_axis_0))
    interp_0 = RegularGridInterpolator((id_axis_0, iq_axis_0, theta_axis_0), matrix_3d_0, method="cubic")
    
    print(f"[완료] LUT 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
    return interp_d, interp_q, interp_0  # 3개 모두 반환하도록 수정

def run_bemf_simulation_at_141rpm():
    interp_d, interp_q, interp_0 = load_lut_interpolators()  # 3개의 반환값 수신
    
    dt = 0.00001                # 제어 주기 [s]
    total_time = 1          # 총 시뮬레이션 시간 [s]
    total_steps = int(total_time / dt)
    
    target_speed_rpm = 141.0  # 목표 BEMF 측정 속도 [RPM]
    current_radpsec = target_speed_rpm * np.pi / 30.0
    omega_e = current_radpsec * pole_pairs
    
    theta_mech_cum = 0.0      # 누적 기계각 [deg]
    
    # 히스토리 기록용 리스트
    time_history = []
    speed_history = []
    v_a_history, v_b_history, v_c_history = [], [], []
    lamda_d_history, lamda_q_history = [], []
    
    id_val = 0.0  # 무부하 BEMF이므로 전류는 0
    iq_val = 0.0
    
    print(f"\n==================================================")
    print(f" IPMSM 141 RPM BEMF(역기전력) 3상 전압 시뮬레이션 시작")
    print(f"==================================================")
    
    theta_elec_rad_old = 0.0
    lamda_dq_old = np.zeros(3)
    
    # 3차원 공간(d, q, 0)에 맞게 회전 행렬 확장 (J_ 행렬 차원 오류 수정)
    J_ = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  0]
    ])
    
    for step in range(total_steps):
        current_time = step * dt
        
        # 전기각 계산 (0 ~ 360도 정규화)
        theta_elec_deg = (theta_mech_cum * pole_pairs) % 360.0
        if theta_elec_deg < 0.0:
            theta_elec_deg += 360.0
        theta_elec_rad = np.radians(theta_elec_deg)
        
        # LUT 자속 조회
        query_point = np.array([[id_val, iq_val, theta_elec_deg]])
        lamda_d = interp_d(query_point)[0]
        lamda_q = interp_q(query_point)[0]
        lamda_0 = interp_0(query_point)[0]
        lamda_dq = np.array([lamda_d, lamda_q, lamda_0])

        if step == 0:
            gamma_dq = np.zeros(3)
            Vdq = np.zeros(3)
        else:
            gamma_dq = (lamda_dq - lamda_dq_old) / (theta_elec_rad - theta_elec_rad_old)
            Vdq = omega_e * (gamma_dq + J_ @ lamda_dq)
            
        lamda_dq_old = lamda_dq
        theta_elec_rad_old = theta_elec_rad   
            
        # dq0 -> 3상 BEMF 전압(Va, Vb, Vc) 변환
        v_abc = dq_to_abc2(Vdq, theta_elec_rad)
        
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