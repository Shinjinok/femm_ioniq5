import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.electric_motors import PermanentMagnetSynchronousMotor
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

# 2. 포화 맵이 반영된 GEM 커스텀 PMSM 모터 클래스
class SaturatedFEMMPMSM(PermanentMagnetSynchronousMotor):
    def __init__(self, sat_map, p=8, r_s=0.06):
        super().__init__()
        self.sat_map = sat_map
        self.p = p
        self.r_s = r_s

    def electrical_ode(self, *args, **kwargs):
        if len(args) >= 4:
            state, prev_state, u_in, omega_rotor = args[0], args[1], args[2], args[3]
        else:
            state = kwargs.get('state', args[0] if len(args) > 0 else [0, 0])
            u_in = kwargs.get('u_in', args[2] if len(args) > 2 else [0, 0])
            omega_rotor = kwargs.get('omega_rotor', args[3] if len(args) > 3 else 0.0)

        i_d, i_q = float(state[0]), float(state[1])
        
        if np.isscalar(u_in):
            v_d, v_q = u_in, 0.0
        elif len(u_in) >= 2:
            v_d, v_q = u_in[0], u_in[1]
        else:
            v_d, v_q = u_in[0], 0.0
        
        mag = np.sqrt(i_d**2 + i_q**2)
        beta = np.degrees(np.arctan2(i_q, i_d))
        
        lam_d, lam_q, torque = self.sat_map.get_values(beta, mag)
        
        delta = 0.5
        lam_d_plus, lam_q_plus, _ = self.sat_map.get_values(beta, mag + delta)
        lam_d_minus, lam_q_minus, _ = self.sat_map.get_values(beta, max(0.0, mag - delta))
        
        L_d_inc = max(1e-5, (lam_d_plus - lam_d_minus) / (2 * delta))
        L_q_inc = max(1e-5, (lam_q_plus - lam_q_minus) / (2 * delta))
        L_inc = np.array([[L_d_inc, 0.0], [0.0, L_q_inc]])
        
        res_v = np.array([
            v_d - (self.r_s * i_d - omega_rotor * lam_q),
            v_q - (self.r_s * i_q + omega_rotor * lam_d)
        ])
        
        di_dt = np.linalg.inv(L_inc) @ res_v
        return di_dt

def extract_flat_elements(obs):
    flat_list = []
    if isinstance(obs, (list, tuple, np.ndarray)):
        for item in obs:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat_list.extend(np.atleast_1d(item).flatten())
            else:
                flat_list.append(item)
    else:
        flat_list.append(obs)
    return flat_list

# 3. 메인 실행부
if __name__ == "__main__":
    sat_map = FemmSaturationMap(
        torque_csv='FEMM_Torque_matrix.csv',
        lam_d_csv='FEMM_Lambda_d_matrix.csv',
        lam_q_csv='FEMM_Lambda_q_matrix.csv'
    )

    p_poles = 8
    motor = SaturatedFEMMPMSM(sat_map=sat_map, p=p_poles, r_s=0.06)

    # 전류 제어 환경 설정
    env = gem.make(
        'Cont-CC-PMSM-v0',
        motor=motor,
        visualization=None
    )

    obs, info = env.reset()

    # 제어기 파라미터 및 영구자석 자속 설정
    psi_m, _, _ = sat_map.get_values(0.0, 0.0)
    if psi_m <= 0:
        psi_m = 0.1

    kp_i = 5.0
    ki_i = 50.0
    integral_id = 0.0
    integral_iq = 0.0
    dt = 1e-4
    v_max = 400.0

    time_steps = []
    id_list = []
    iq_list = []
    torque_list = []
    torque_ref_list = []

    t = 0
    print("전류 제어 기반 토크 추종 시뮬레이션 실행 중 (총 1000 스텝)...")
    
    for step in range(1000):
        flat_obs = extract_flat_elements(obs)
        
        i_d = float(flat_obs[0]) if len(flat_obs) > 0 else 0.0
        i_q = float(flat_obs[1]) if len(flat_obs) > 1 else 0.0
        
        # 목표 토크 스케줄 (예시: 500스텝 전은 15 Nm, 이후는 -10 Nm)
        torque_ref = 15.0 if step < 500 else -10.0  
        
        # 1. 토크 지령 -> q축 전류 지령 변환 (Id = 0 제어)
        id_ref = 0.0
        iq_ref = torque_ref / (1.5 * (p_poles / 2) * psi_m + 1e-6)
        iq_ref = np.clip(iq_ref, -40.0, 40.0)
        
        # 2. 전류 PI 제어기 연산 (d축, q축 전압 계산)
        e_id = id_ref - i_d
        integral_id += e_id * dt
        v_d = kp_i * e_id + ki_i * integral_id
        
        e_iq = iq_ref - i_q
        integral_iq += e_iq * dt
        v_q = kp_i * e_iq + ki_i * integral_iq
        
        # 3. 역 파크 및 클라크 변환을 통해 d-q 전압을 3상(abc) 전압으로 변환
        # 물리 시스템 내부의 전기각(epsilon) 추출 (인덱스 2)
        epsilon = env.physical_system.state[2] if hasattr(env, 'physical_system') else 0.0
        
        v_alpha = v_d * np.cos(epsilon) - v_q * np.sin(epsilon)
        v_beta  = v_d * np.sin(epsilon) + v_q * np.cos(epsilon)
        
        v_a = v_alpha
        v_b = -0.5 * v_alpha + (np.sqrt(3) / 2.0) * v_beta
        v_c = -0.5 * v_alpha - (np.sqrt(3) / 2.0) * v_beta
        
        # 3상 전압 액션 배열 구성 및 정규화 (-1 ~ 1)
        action = np.array([v_a / v_max, v_b / v_max, v_c / v_max])
        action = np.clip(action, -1.0, 1.0)
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception:
            action = np.array([0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
        
        time_steps.append(t)
        id_list.append(i_d)
        iq_list.append(i_q)
        
        # 실제 포화 맵 기반 토크 계산
        mag = np.sqrt(i_d**2 + i_q**2)
        beta = np.degrees(np.arctan2(i_q, i_d))
        _, _, current_torque = sat_map.get_values(beta, mag)
        
        torque_list.append(current_torque)
        torque_ref_list.append(torque_ref)
        
        t += 1
        if terminated or truncated:
            print(f"에피소드가 {t}스텝 만에 조기 종료되었습니다.")
            break

    env.close()
    print(f"시뮬레이션 완료 (총 {len(time_steps)} 스텝 수집)")

    # 4. 결과 시각화
    plt.figure(figsize=(10, 9))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, id_list, label='d-axis current ($i_d$)', color='blue', linewidth=1.5)
    plt.plot(time_steps, iq_list, label='q-axis current ($i_q$)', color='orange', linewidth=1.5)
    plt.ylabel('Current (A)')
    plt.title('Torque Tracking via Current Control (CC-PMSM) with FEMM Map')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_steps, torque_list, label='Actual Torque ($T_e$)', color='green', linewidth=1.5)
    plt.ylabel('Torque (Nm)')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_steps, torque_ref_list, label='Target Torque Reference ($T_{ref}$)', color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Simulation Steps')
    plt.ylabel('Ref Torque (Nm)')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('torque_tracking_cc_results.png', dpi=300)
    print("결과 그래프가 'torque_tracking_cc_results.png' 파일로 저장되었습니다.")
    plt.show()