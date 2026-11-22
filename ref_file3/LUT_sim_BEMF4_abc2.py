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

load_torque = 0
J = 0.036      # 로터 관성 모멘트 [kg·m^2]
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
    df_a = pd.read_csv('LUT_lamda_abc_keep.csv')
   
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
    
    # 피벗 테이블로부터 축(Axis) 값과 매트릭스 추출
    id_axis = pivot_df_a.index.get_level_values('Id_A').unique().values 
    iq_axis = pivot_df_a.index.get_level_values('Iq_A').unique().values  
    theta_axis = pivot_df_a.columns.values                                 
    
    matrix_3d_a = pivot_df_a.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_b = pivot_df_b.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
    matrix_3d_c = pivot_df_c.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))

    # 범위 이탈 방지를 위한 bounds_error=False, fill_value=None 설정
    interp_a = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_a, method="cubic", bounds_error=False, fill_value=None)
    interp_b = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_b, method="cubic", bounds_error=False, fill_value=None)
    interp_c = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_c, method="cubic", bounds_error=False, fill_value=None)
    
    print(f"[완료] ABC 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
    return interp_a, interp_b, interp_c, id_axis, iq_axis, theta_axis

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

def run_bemf_simulation_at_141rpm():
    interp_a, interp_b, interp_c, id_axis, iq_axis, theta_axis = load_lut_abc_interpolators()
    
    dt = 0.0001            # 제어 주기 [s]
    total_time = 0.25            # 총 시뮬레이션 시간 [s]
    total_steps = int(total_time / dt)
    
    target_speed_rpm = 141.0    # 목표 BEMF 측정 속도 [RPM]
    current_speed_rpm = 0.0     # 초기 속도 [RPM]
    current_radpsec = target_speed_rpm * np.pi / 30.0
    omega_e = current_radpsec * pole_pairs
    
    theta_mech_cum = 0.0        # 누적 기계각 [deg]
    
    # 히스토리 기록용 리스트
    time_history = []
    speed_history = []
    v_a_history, v_b_history, v_c_history = [], [], []
    lamda_d_history, lamda_q_history = [], []
    
    print(f"\n==================================================")
    print(f" IPMSM 141 RPM BEMF(역기전력) 3상 전압 시뮬레이션 시작 (ABC LUT 연동)")
    print(f"==================================================")
    
    speed_pi = SpeedControllerPI(kp=0.1, ki=1, output_limits=(-200, 200))
    
    id = 0.0
    iq = 0.0
    Ld = 0.001
    Lq = 0.001
    Vd = 0.0
    Vq = 0.0
    cos_130 = np.cos(130*np.pi/180)
    sin_130 = np.sin(130*np.pi/180)
    for step in range(total_steps):
        current_time = step * dt
        
        # 전기각 계산 (0 ~ 360도 정규화)
        theta_elec_deg = (theta_mech_cum * pole_pairs) % 360.0
        if theta_elec_deg < 0.0:
            theta_elec_deg += 360.0
        theta_elec_rad = np.radians(theta_elec_deg)

        Vref = speed_pi.compute(target_speed_rpm, current_speed_rpm, dt)
        #Vq = sin_130 * Vref
        #Vd = cos_130 * Vref
        # 1. ABC 쇄교자속 LUT 조회 (실제 id, iq, theta_elec_deg 연동 및 클리핑 처리)
        query_point = np.array([[
            np.clip(id, id_axis[0], id_axis[-1]), 
            np.clip(iq, iq_axis[0], iq_axis[-1]), 
            np.clip(theta_elec_deg, theta_axis[0], theta_axis[-1])
        ]])
        query_point0 = np.array([[
            np.clip(0.0, id_axis[0], id_axis[-1]), 
            np.clip(0.0, iq_axis[0], iq_axis[-1]), 
            np.clip(theta_elec_deg, theta_axis[0], theta_axis[-1])
        ]])
        query_point_id1 = np.array([[
            np.clip(id+1, id_axis[0], id_axis[-1]), 
            np.clip(iq, iq_axis[0], iq_axis[-1]), 
            np.clip(theta_elec_deg, theta_axis[0], theta_axis[-1])
        ]])
        query_point_iq1 = np.array([[
            np.clip(id, id_axis[0], id_axis[-1]), 
            np.clip(iq+1, iq_axis[0], iq_axis[-1]), 
            np.clip(theta_elec_deg, theta_axis[0], theta_axis[-1])
        ]])
        lamda_abc = np.array([interp_a(query_point)[0], interp_b(query_point)[0], interp_c(query_point)[0]])
        lamda_abc0 = np.array([interp_a(query_point0)[0], interp_b(query_point0)[0], interp_c(query_point0)[0]])
        lamda_abc_id1 = np.array([interp_a(query_point_id1)[0], interp_b(query_point_id1)[0], interp_c(query_point_id1)[0]])
        lamda_abc_iq1 = np.array([interp_a(query_point_iq1)[0], interp_b(query_point_iq1)[0], interp_c(query_point_iq1)[0]])
        # 2. ABC 자속을 Park 변환하여 d-q-0축 자속으로 변환
        lamda_dq0 = abc_to_dq(lamda_abc, theta_elec_rad)
        lamda_dq0_0 = abc_to_dq(lamda_abc0, theta_elec_rad)
        lamda_dq0_id1 = abc_to_dq(lamda_abc_id1, theta_elec_rad)
        lamda_dq0_iq1 = abc_to_dq(lamda_abc_iq1, theta_elec_rad)
        
        # 인덕턴스 추정 안전장치
        Ld = lamda_dq0_id1[0] - lamda_dq0[0]
        Lq = lamda_dq0_iq1[1] - lamda_dq0[1]
        current_radpsec = target_speed_rpm / 60 * 2 * np.pi
        we = current_radpsec * pole_pairs
        
        did_dt = 1/Ld * (Vd - Rs*id + we * Lq * iq)
        diq_dt = 1/Lq * (Vq - Rs*iq - we * Ld * id - we * 0.14)

         
        id += did_dt * dt
        iq += diq_dt * dt
        Vd_ = -we * Lq * iq     
        Vq_ = we * Ld * id + we * 0.14
        i_abc = dq_to_abc2([Vd_, Vq_, 0], theta_elec_rad)
        
        # 전자기 토크 계산 (영구자석 토크 + 릴럭턴스 토크)
        lamda_f = lamda_dq0_0[0]
        Te_z = 1.5 * pole_pairs * (lamda_f * iq + (Ld - Lq) * id * iq)

        # 모터 동역학 방정식 계산 및 회전 각도 산출
        accel_rad_s2 = (Te_z - load_torque) / J
        current_radpsec += accel_rad_s2 * dt
        current_speed_rpm = current_radpsec * 30.0 / np.pi
        
        # 기계각 업데이트
        delta_mech_deg = (current_radpsec * dt) * (180 / np.pi)
        theta_mech_cum += delta_mech_deg
        theta_mech_cum %= 360

        print(f"step: {step} | Ld: {Ld:.5f} | Lq: {Lq:.5f} | id: {id:.3f} | iq: {iq:.3f} | Te: {Te_z:.2f} | lamda_dq0_0[0]: {lamda_dq0_0[0]:.4f} | lamda_dq0_0[1]: {lamda_dq0_0[1]:.4f}")
        
        # 기록
        time_history.append(current_time)
        speed_history.append(current_speed_rpm)
        v_a_history.append(i_abc[0])
        v_b_history.append(i_abc[1])
        v_c_history.append(i_abc[2])
        lamda_d_history.append(id)
        lamda_q_history.append(iq)

    print("[완료] BEMF 시뮬레이션 연산 완료. 그래프 생성 중...")

    # --- 시각화 ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    
    # 1. 속도
    axes[0].plot(time_history, speed_history, label='Speed (RPM)', color='blue', linewidth=2)
    axes[0].set_ylabel('Speed [RPM]')
    axes[0].legend(loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_title('IPMSM Back-EMF Simulation at 141 RPM (ABC LUT)')
    
    # 2. 3상 전류 파형
    axes[1].plot(time_history, v_a_history, label='Phase A ($i_a$)', color='red', linewidth=1.2)
    axes[1].plot(time_history, v_b_history, label='Phase B ($i_b$)', color='green', linewidth=1.2)
    axes[1].plot(time_history, v_c_history, label='Phase C ($i_c$)', color='blue', linewidth=1.2)
    axes[1].set_ylabel('Phase Current [A]')
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