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
    df_a = pd.read_csv('LUT_lamda_abc.csv')
   
    pivot_df_a = df_a.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_a_Wb'
    )
    pivot_df_b = df_a.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_b_Wb'
    )
    pivot_df_c = df_a.pivot_table(
        index=['Id_A', 'Iq_A'], 
        columns='theta_e_deg', 
        values='lamda_c_Wb'
    )
    pivot_df_T = df_a.pivot_table(
            index=['Id_A', 'Iq_A'], 
            columns='theta_e_deg', 
            values='Torque'
    )

    # 피벗 테이블로부터 축(Axis) 값과 매트릭스 추출
    id_axis = pivot_df_a.index.get_level_values('Id_A').unique().values 
    iq_axis = pivot_df_a.index.get_level_values('Iq_A').unique().values  
    theta_axis = pivot_df_a.columns.values                                 
    
    matrix_3d_a = pivot_df_a.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_b = pivot_df_b.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_c = pivot_df_c.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_T = pivot_df_T.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    # 범위 이탈 방지를 위한 bounds_error=False, fill_value=None 설정
    interp_a = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_a, method="cubic", bounds_error=False, fill_value=None)
    interp_b = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_b, method="cubic", bounds_error=False, fill_value=None)
    interp_c = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_c, method="cubic", bounds_error=False, fill_value=None)
    interp_T = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_T, method="cubic", bounds_error=False, fill_value=None)
    print(f"[완료] ABC 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
    return interp_a, interp_b, interp_c, interp_T, id_axis, iq_axis, theta_axis

class SpeedControllerPI:
    """모터 속도 제어를 위한 PI 제어기"""
    def __init__(self, kp, ki, output_limits=(-340, 340)):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.min_out, self.max_out = output_limits

    def compute(self, target_speed, current_speed, dt):
        error = (target_speed - current_speed) # rad/s
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
    interp_a, interp_b, interp_c, interp_T ,id_axis, iq_axis, theta_axis,= load_lut_abc_interpolators()
    
    dt = 0.0001                # 제어 주기 [s]
    total_time = 2          # 총 시뮬레이션 시간 [s]
    total_steps = int(total_time / dt)
    
    target_speed_rpm = 141.0  # 목표 BEMF 측정 속도 [RPM]
    current_radpsec = 0# target_speed_rpm * np.pi / 30.0
    omega_e = current_radpsec * pole_pairs
    
    theta_mech_cum = 0.0      # 누적 기계각 [deg]
    
    # 히스토리 기록용 리스트
    time_history = []
    speed_history = []
    v_a_history, v_b_history, v_c_history = [], [], []
    v_a0_history, v_b0_history, v_c0_history = [], [], []
    v_d_history, v_q_history, v_0_history = [], [],[]
    v_d1_history, v_q1_history, v_01_history = [], [],[]
    Torque_histroy = []
    Torque_histroy2 = []
    id_val = 0.0  # 무부하 BEMF이므로 전류는 0
    iq_val = 0.0
    
    print(f"\n==================================================")
    print(f" IPMSM 141 RPM BEMF(역기전력) 3상 전압 시뮬레이션 시작 (ABC LUT 연동)")
    print(f"==================================================")
    
    theta_elec_rad_old = 0.0
    lamda_dq0_old = np.zeros(3)
    lamda_abc_old = np.zeros(3)
    V_dq0 = np.zeros(3)
    Vd = 0
    Vq = 0

    J_ = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  0]
    ])
    load_torque = 10
    for step in range(total_steps):
        current_time = step * dt
        
        # 전기각 계산 (0 ~ 360도 정규화)
        theta_elec_rad, theta_elec_deg = check_degree(theta_mech_cum * pole_pairs)

        # 1. ABC 쇄교자속 LUT 조회
        query_point0 = np.array([[0, 0, theta_elec_deg]])
        lamda_abc_0 = np.array([interp_a(query_point0)[0], interp_b(query_point0)[0], interp_c(query_point0)[0]])
        lamda_dq0_0 = abc_to_dq(lamda_abc_0, theta_elec_rad)
        

        query_point = np.array([[id_val, iq_val, theta_elec_deg]])
        
        lamda_abc = np.array([interp_a(query_point)[0], interp_b(query_point)[0], interp_c(query_point)[0]])
        lamda_dq0 = abc_to_dq(lamda_abc, theta_elec_rad)

        query_point_id1 = np.array([[id_val+1, iq_val, theta_elec_deg]])
        query_point_iq1 = np.array([[id_val, iq_val+1, theta_elec_deg]])
        degree_step = 2

        dth1_rad, dth1_deg = check_degree(theta_elec_deg - degree_step)
        dth2_rad, dth2_deg = check_degree(theta_elec_deg + degree_step)

        query_point_th1 = np.array([[id_val, iq_val, dth1_deg]])
        query_point_th2 = np.array([[id_val, iq_val, dth2_deg]])
        
        lamda_abc_id1 = np.array([interp_a(query_point_id1)[0], interp_b(query_point_id1)[0], interp_c(query_point_id1)[0]])
        lamda_abc_iq1 = np.array([interp_a(query_point_iq1)[0], interp_b(query_point_iq1)[0], interp_c(query_point_iq1)[0]])
        
        lamda_dq0_id1 = abc_to_dq(lamda_abc_id1, theta_elec_rad)
        lamda_dq0_iq1 = abc_to_dq(lamda_abc_iq1, theta_elec_rad) 

        lamda_abc_th1 = np.array([interp_a(query_point_th1)[0], interp_b(query_point_th1)[0], interp_c(query_point_th1)[0]])
        lamda_abc_th2 = np.array([interp_a(query_point_th2)[0], interp_b(query_point_th2)[0], interp_c(query_point_th2)[0]])
        lamda_dq0_th1 = abc_to_dq(lamda_abc_th1, dth1_rad )
        lamda_dq0_th2 = abc_to_dq(lamda_abc_th2, dth2_rad ) 

        query_point_T = np.array([[id_val, iq_val, theta_elec_deg]])
                
        torque = interp_T(query_point_T)[0]
        

        # 인덕턴스 추정 안전장치
        Ld = lamda_dq0_id1[0] - lamda_dq0[0]
        Lq = lamda_dq0_iq1[1] - lamda_dq0[1]
        gamma_dq0 = (lamda_dq0_th2 - lamda_dq0_th1 ) / (2*degree_step*2*np.pi/360)
        We = current_radpsec*pole_pairs
        if step ==0 :
            V_abc = np.zeros(3)
            V_dq0 = np.zeros(3)
            V_wd =0
            V_wq=0
        else :

            V_abc = (lamda_abc - lamda_abc_old) / dt  #main reference
            bemf = (lamda_dq0  - lamda_dq0_old) /dt  +  We * J_ @ lamda_dq0  #V_abc와 일치
            did_dt = 1/Ld * (Vd - Rs*id_val + We * Lq * iq_val) 
            diq_dt = 1/Lq * (Vq - Rs*iq_val - We * Ld * id_val - We * lamda_dq0_0[0])
            
            #bemf = We * (gamma_dq0  +  J_ @ lamda_dq0) #BEMF
            #did_dt = 1/Ld * (Vd - Rs*id_val - bemf[0]) 
            #diq_dt = 1/Lq * (Vq - Rs*iq_val - bemf[1])
            
            id_val += did_dt * dt
            iq_val += diq_dt * dt
            
            V_wd = -We * Lq * iq_val
            V_wq =  We * (Ld * id_val + 0.14)
            V_dq0 =   bemf
            #print(f"step: {step} | Ld: {Ld:.5f} | Lq: {Lq:.5f} | id_val: {id_val:.3f} | iq_val: {iq_val:.3f} | bemf[0]: {bemf[0]:.3f} | bemf[1]: {bemf[1]:.3f}")

        Te_z = 1.5 * pole_pairs * (lamda_dq0_0[0] * iq_val + (Ld - Lq) * id_val * iq_val)
        Torque_histroy.append(Te_z)
        Te_cross = 1.5 * pole_pairs *np.linalg.cross(lamda_dq0 ,np.array([id_val,iq_val,0]))
        Torque_histroy2.append(torque)

        # 모터 동역학 방정식 계산 및 회전 각도 산출
        accel_rad_s2 = (Te_z - load_torque) / J
        current_radpsec += accel_rad_s2 * dt
        current_speed_rpm = current_radpsec * 30.0 / np.pi

        lamda_abc_old = lamda_abc
        lamda_dq0_old = lamda_dq0
        theta_elec_rad_old = theta_elec_rad

        V_abc_from_dq0 = dq_to_abc2(V_dq0, theta_elec_rad)
        V_dq0_from_abc = abc_to_dq(V_abc, theta_elec_rad)
        
        # 기계각 업데이트 (141 RPM 유지)
        delta_mech_deg = (current_radpsec * dt) * (180 / np.pi)
        theta_mech_cum += delta_mech_deg
        
        # 기록
        time_history.append(current_time)
        speed_history.append(-current_speed_rpm)
        v_a_history.append(V_abc[0])
        v_b_history.append(V_abc[1])
        v_c_history.append(V_abc[2])
        v_a0_history.append(V_abc_from_dq0[0])
        v_b0_history.append(V_abc_from_dq0[1])
        v_c0_history.append(V_abc_from_dq0[2])
        v_d_history.append(V_dq0[0])
        v_q_history.append(V_dq0[1])
        v_0_history.append(V_dq0[2])
        v_d1_history.append(V_dq0_from_abc[0])
        v_q1_history.append(V_dq0_from_abc[1])
        v_01_history.append(V_dq0_from_abc[2])
        
    print("[완료] BEMF 시뮬레이션 연산 완료. 그래프 생성 중...")

    # --- 시각화 ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # 1. 속도 (141 RPM)
    axes[0].plot(time_history, Torque_histroy, label='$\phi_f i_q + (L_d - L_q) i_d i_q$', color='blue', linewidth=2)
    axes[0].plot(time_history, Torque_histroy2, label='$\lambda x I_{dq0}$', color='green', linewidth=2)
    axes[0].plot(time_history, speed_history, label='RPM', color='red', linewidth=2)
    axes[0].set_ylabel('Torque')
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title('IPMSM Back-EMF (BEMF) 3-Phase Voltages at 141 RPM (ABC LUT)')
    
    # 2. 3상 BEMF 전압 파형
    axes[1].plot(time_history, v_a_history, label='d$\lambda$/dt ($V_a$)', color='red', linewidth=1.2, alpha=0.5)
    axes[1].plot(time_history, v_b_history, label='d$\lambda$/dt ($V_b$)', color='green', linewidth=1.2, alpha=0.5)
    axes[1].plot(time_history, v_c_history, label='d$\lambda$/dt ($V_c$)', color='blue', linewidth=1.2, alpha=0.5)
    axes[1].plot(time_history, v_a0_history, label='$\gamma+J\lambda$ ($V_a$)', color='red', linewidth=1.2,linestyle='--')
    axes[1].plot(time_history, v_b0_history, label='$\gamma+J\lambda$ ($V_b$)', color='green', linewidth=1.2,linestyle='--')
    axes[1].plot(time_history, v_c0_history, label='$\gamma+J\lambda$ ($V_c$)', color='blue', linewidth=1.2,linestyle='--')
    axes[1].set_ylabel('BEMF Voltage [V]')
    axes[1].legend(loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. 쇄교자속
    axes[2].plot(time_history, v_d1_history, label='V_dq0_from_abc', color='red', linewidth=1.2, alpha=0.5)
    axes[2].plot(time_history, v_q1_history, label='V_dq0_from_abc', color='green', linewidth=1.2, alpha=0.5)
    axes[2].plot(time_history, v_01_history, label='V_dq0_from_abc', color='blue', linewidth=1.2, alpha=0.5)
    axes[2].plot(time_history, v_d_history, label='$\gamma+J\lambda$ ($V_d$)', color='red', linewidth=1.2,linestyle='--')
    axes[2].plot(time_history, v_q_history, label='$\gamma+J\lambda$ ($V_q$)', color='green', linewidth=1.2,linestyle='--')
    axes[2].plot(time_history, v_0_history, label='$\gamma+J\lambda$ ($V_0$)', color='blue', linewidth=1.2,linestyle='--')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('BEMF Voltage [V]')
    axes[2].legend(loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('ipmsm_bemf_141rpm_abc.png', dpi=300)
    plt.show()
    print("[완료] BEMF 결과 그래프가 'ipmsm_bemf_141rpm_abc.png'로 저장되었습니다.")

if __name__ == '__main__':
    run_bemf_simulation_at_141rpm()