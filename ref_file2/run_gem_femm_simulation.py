import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# 1. FEMM 포화 맵 보간(Interpolation) 클래스
class FemmSaturationMap:
    def __init__(self, torque_csv, lam_d_csv, lam_q_csv):
        df_t = pd.read_csv(torque_csv, header=None)
        df_ld = pd.read_csv(lam_d_csv, header=None)
        df_lq = pd.read_csv(lam_q_csv, header=None)
        
        self.mags = df_t.iloc[0, 1:].values.astype(float)
        self.betas = df_t.iloc[1:, 0].values.astype(float)
        
        torque_matrix = df_t.iloc[1:, 1:].values.astype(float) 
        lam_d_matrix = df_ld.iloc[1:, 1:].values.astype(float) * 1e-3
        lam_q_matrix = df_lq.iloc[1:, 1:].values.astype(float) * 1e-3
        
        grid_points = (self.betas, self.mags)
        self.interp_torque = RegularGridInterpolator(grid_points, torque_matrix, bounds_error=False, fill_value=None)
        self.interp_lam_d  = RegularGridInterpolator(grid_points, lam_d_matrix, bounds_error=False, fill_value=None)
        self.interp_lam_q  = RegularGridInterpolator(grid_points, lam_q_matrix, bounds_error=False, fill_value=None)

    def get_values(self, beta, mag):
        beta_clamped = np.clip(beta, self.betas[0], self.betas[-1])
        mag_clamped = np.clip(mag, self.mags[0], self.mags[-1])
        
        point = np.array([[beta_clamped, mag_clamped]])
        torque = self.interp_torque(point)[0]
        lam_d = self.interp_lam_d(point)[0]
        lam_q = self.interp_lam_q(point)[0]
        return lam_d, lam_q, torque

# 2. 포화 맵 기반 PMSM 시뮬레이터 클래스 (직접 수치 적분)
class SaturatedPMSMSimulator:
    def __init__(self, sat_map, p=8, r_s=0.06, dt=1e-4):
        self.sat_map = sat_map
        self.p = p
        self.r_s = r_s
        self.dt = dt
        
        # 상태 초기화: [i_d, i_q, omega_rotor, epsilon]
        self.state = np.array([0.0, 0.0, 100.0, 0.0]) # 회전속도 100 rad/s 가정

    def step(self, v_d, v_q):
        i_d, i_q, omega_rotor, epsilon = self.state
        
        mag = np.sqrt(i_d**2 + i_q**2)
        beta = np.degrees(np.arctan2(i_q, i_d))
        
        lam_d, lam_q, torque = self.sat_map.get_values(beta, mag)
        
        # 증분 인덕턴스 계산 (Finite Difference)
        delta = 0.5
        lam_d_plus, lam_q_plus, _ = self.sat_map.get_values(beta, mag + delta)
        lam_d_minus, lam_q_minus, _ = self.sat_map.get_values(beta, max(0.0, mag - delta))
        
        L_d_inc = max(1e-5, (lam_d_plus - lam_d_minus) / (2 * delta))
        L_q_inc = max(1e-5, (lam_q_plus - lam_q_minus) / (2 * delta))
        L_inc = np.array([[L_d_inc, 0.0], [0.0, L_q_inc]])
        
        # 전압 방정식에 따른 di/dt 계산
        res_v = np.array([
            v_d - (self.r_s * i_d - omega_rotor * lam_q),
            v_q - (self.r_s * i_q + omega_rotor * lam_d)
        ])
        
        di_dt = np.linalg.inv(L_inc) @ res_v
        
        # 오일러 적분으로 전류 업데이트
        i_d += di_dt[0] * self.dt
        i_q += di_dt[1] * self.dt
        
        # 전기각 업데이트 (omega_e = (p/2) * omega_r)
        omega_e = (self.p / 2) * omega_rotor
        epsilon += omega_e * self.dt
        epsilon = np.mod(epsilon, 2 * np.pi)
        
        self.state = np.array([i_d, i_q, omega_rotor, epsilon])
        return i_d, i_q, torque

# 3. 메인 실행부 (직접 Id, Iq 전류 PID 제어 + 포화 맵 시뮬레이션)
if __name__ == "__main__":
    sat_map = FemmSaturationMap(
        torque_csv='FEMM_Torque_matrix.csv',
        lam_d_csv='FEMM_Lambda_d_matrix.csv',
        lam_q_csv='FEMM_Lambda_q_matrix.csv'
    )

    p_poles = 8
    dt = 1e-4
    motor_sim = SaturatedPMSMSimulator(sat_map=sat_map, p=p_poles, r_s=0.06, dt=dt)

    # PID 제어기 이득 설정 (P, I, D 이득)
    kp_i = 0.01
    ki_i = 0
    kd_i = 0  # 미분(Derivative) 제어 이득 추가
    
    integral_id = 0.0
    integral_iq = 0.0
    e_id_prev = 0.0
    e_iq_prev = 0.0
    v_max = 400.0

    time_steps = []
    id_list = []
    iq_list = []
    id_ref_list = []
    iq_ref_list = []
    torque_list = []

    print("직접 Id, Iq 전류 PID 제어 시뮬레이션 실행 중 (총 1000 스텝)...")
    
    for step in range(1000):
        i_d, i_q, current_torque = motor_sim.state[0], motor_sim.state[1], 0.0
        
        # 1. 직접 전류 지령 스케줄 설정 (500스텝 전/후로 변경)
        id_ref = 0.0
        iq_ref = 10.0

            
        id_ref = np.clip(id_ref, -40.0, 40.0)
        iq_ref = np.clip(iq_ref, -40.0, 40.0)
        
        # 2. d축 전류 PID 제어기 (Anti-windup 적용)
        e_id = id_ref - i_d
        de_id = (e_id - e_id_prev) / dt  # 오차 미분값 계산
        v_d_unlimited = kp_i * e_id + ki_i * integral_id + kd_i * de_id
        v_d = np.clip(v_d_unlimited, -v_max, v_max)
        
        if np.isclose(v_d, v_d_unlimited):
            integral_id += e_id * dt
        e_id_prev = e_id  # 이전 오차 업데이트
        
        # 3. q축 전류 PID 제어기 (Anti-windup 적용)
        e_iq = iq_ref - i_q
        de_iq = (e_iq - e_iq_prev) / dt  # 오차 미분값 계산
        v_q_unlimited = kp_i * e_iq + ki_i * integral_iq + kd_i * de_iq
        v_q = np.clip(v_q_unlimited, -v_max, v_max)
        
        if np.isclose(v_q, v_q_unlimited):
            integral_iq += e_iq * dt
        e_iq_prev = e_iq  # 이전 오차 업데이트
        
        # 4. 모터 시뮬레이터 스텝 진행
        i_d_next, i_q_next, current_torque = motor_sim.step(v_d, v_q)
        
        time_steps.append(step)
        id_list.append(i_d_next)
        iq_list.append(i_q_next)
        id_ref_list.append(id_ref)
        iq_ref_list.append(iq_ref)
        torque_list.append(current_torque)

    print("시뮬레이션 완료!")

    # 5. 결과 시각화
    plt.figure(figsize=(10, 10))
    
    # d축 전류 비교
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, id_ref_list, label='d-axis Ref ($i_{d}^*$)', color='blue', linestyle='--', linewidth=1.5)
    plt.plot(time_steps, id_list, label='d-axis Actual ($i_d$)', color='navy', linewidth=1.5)
    plt.ylabel('Current (A)')
    plt.title('Direct $i_d, i_q$ Current PID Control with FEMM Saturation Map')
    plt.legend(loc='upper right')
    plt.grid(True)

    # q축 전류 비교
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, iq_ref_list, label='q-axis Ref ($i_{q}^*$)', color='orange', linestyle='--', linewidth=1.5)
    plt.plot(time_steps, iq_list, label='q-axis Actual ($i_q$)', color='darkorange', linewidth=1.5)
    plt.ylabel('Current (A)')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 발생하는 전자기 토크
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, torque_list, label='Actual Torque ($T_e$)', color='green', linewidth=1.5)
    plt.xlabel('Simulation Steps')
    plt.ylabel('Torque (Nm)')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('direct_current_pid_control_results.png', dpi=300)
    print("결과 그래프가 'direct_current_pid_control_results.png' 파일로 저장되었습니다.")
    plt.show()