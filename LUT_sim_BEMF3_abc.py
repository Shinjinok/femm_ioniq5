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

def load_lut_abc_interpolators():
    print("[정보] 3상(ABC) 쇄교자속 LUT CSV 파일 로딩 및 보간기 생성 중...")
    start_t = time.time()
    
    # 3상 원본 쇄교자속 CSV 파일 로드
    df_a = pd.read_csv('LUT_lamda_a.csv')
    df_b = pd.read_csv('LUT_lamda_b.csv')
    df_c = pd.read_csv('LUT_lamda_c.csv')
    
    pivot_df_a = df_a.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_a_Wb'
    )
    pivot_df_b = df_b.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_b_Wb'
    )
    pivot_df_c = df_c.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_c_Wb'
    )
    
    # 피벗 테이블로부터 축(Axis) 값과 매트릭스 추출
    id_axis_a = pivot_df_a.index.get_level_values('Id_A').unique().values 
    iq_axis_a = pivot_df_a.index.get_level_values('Iq_A').unique().values  
    theta_axis_a = pivot_df_a.columns.values                               
    matrix_3d_a = pivot_df_a.values.reshape(len(id_axis_a), len(iq_axis_a), len(theta_axis_a))
    interp_a = RegularGridInterpolator((id_axis_a, iq_axis_a, theta_axis_a), matrix_3d_a, method="cubic")

    id_axis_b = pivot_df_b.index.get_level_values('Id_A').unique().values  
    iq_axis_b = pivot_df_b.index.get_level_values('Iq_A').unique().values  
    theta_axis_b = pivot_df_b.columns.values                                
    matrix_3d_b = pivot_df_b.values.reshape(len(id_axis_b), len(iq_axis_b), len(theta_axis_b))
    interp_b = RegularGridInterpolator((id_axis_b, iq_axis_b, theta_axis_b), matrix_3d_b, method="cubic")

    id_axis_c = pivot_df_c.index.get_level_values('Id_A').unique().values  
    iq_axis_c = pivot_df_c.index.get_level_values('Iq_A').unique().values  
    theta_axis_c = pivot_df_c.columns.values                                
    matrix_3d_c = pivot_df_c.values.reshape(len(id_axis_c), len(iq_axis_c), len(theta_axis_c))
    interp_c = RegularGridInterpolator((id_axis_c, iq_axis_c, theta_axis_c), matrix_3d_c, method="cubic")
    
    print(f"[완료] ABC 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
    return interp_a, interp_b, interp_c

def run_bemf_simulation_at_141rpm():
    interp_a, interp_b, interp_c = load_lut_abc_interpolators()
    
    dt = 0.00001                # 제어 주기 [s]
    total_time = 0.1          # 총 시뮬레이션 시간 [s]
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
    print(f" IPMSM 141 RPM BEMF(역기전력) 3상 전압 시뮬레이션 시작 (ABC LUT 연동)")
    print(f"==================================================")
    
    theta_elec_rad_old = 0.0
    lamda_dq_old = np.zeros(3)
    

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
        
        # 1. ABC 쇄교자속 LUT 조회
        query_point = np.array([[id_val, iq_val, theta_elec_deg]])
        lamda_a = interp_a(query_point)[0]
        lamda_b = interp_b(query_point)[0]
        lamda_c = interp_c(query_point)[0]
        lamda_abc = np.array([lamda_a, lamda_b, lamda_c])
        
        # 2. ABC 자속을 Park 변환하여 d-q-0축 자속으로 변환
        lamda_dq0 = abc_to_dq(lamda_abc, theta_elec_rad)
        lamda_d = lamda_dq0[0]
        lamda_q = lamda_dq0[1]
        lamda_dq = lamda_dq0

        if step == 0:
            #gamma_dq = np.zeros(3)
            V_abc = np.zeros(3)
        else:
            #gamma_dq = (lamda_dq - lamda_dq_old) / (theta_elec_rad - theta_elec_rad_old)
            V_abc = (lamda_abc - lamda_old) / dt
            
        lamda_old = lamda_abc
        #theta_elec_rad_old = theta_elec_rad   
            
        # dq0 -> 3상 BEMF 전압(Va, Vb, Vc) 변환
        #v_abc = dq_to_abc2(Vdq, theta_elec_rad)
        
        # 기계각 업데이트 (141 RPM 유지)
        delta_mech_deg = (current_radpsec * dt) * (180 / np.pi)
        theta_mech_cum += delta_mech_deg
        
        # 기록
        time_history.append(current_time)
        speed_history.append(target_speed_rpm)
        v_a_history.append(V_abc[0])
        v_b_history.append(V_abc[1])
        v_c_history.append(V_abc[2])
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
    axes[0].set_title('IPMSM Back-EMF (BEMF) 3-Phase Voltages at 141 RPM (ABC LUT)')
    
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
    plt.savefig('ipmsm_bemf_141rpm_abc.png', dpi=300)
    plt.show()
    print("[완료] BEMF 결과 그래프가 'ipmsm_bemf_141rpm_abc.png'로 저장되었습니다.")

if __name__ == '__main__':
    run_bemf_simulation_at_141rpm()