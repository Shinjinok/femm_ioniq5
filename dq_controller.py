import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from ioq5_model import IPMSMSimulator

# --- 전류 PI 제어기 클래스 ---
class CurrentPIController:
    def __init__(self, kp, ki, output_limit):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.output_limit = output_limit
        
    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.output_limit, self.output_limit)
        output = self.kp * error + self.ki * self.integral
        return output
    
    def reset(self):
        self.integral = 0.0

# --- 속도 PI 제어기 클래스 ---
class SpeedPIController:
    def __init__(self, kp, ki, output_limit):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.output_limit = output_limit
        
    def compute(self, target_rpm, current_rpm, dt):
        error = target_rpm - current_rpm
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.output_limit, self.output_limit)
        output = self.kp * error + self.ki * self.integral
        return np.clip(output, -self.output_limit, self.output_limit)
    
    def reset(self):
        self.integral = 0.0

# --- 스피드 제어 시뮬레이션 구동 루틴 ---
if __name__ == "__main__":
    sim = IPMSMSimulator('FEMM_Lambda_d_matrix.csv', 'FEMM_Lambda_q_matrix.csv', pole_pairs=4, rs=0.06)
    
    dt = 0.0001  # 100 us 샘플링 타임
    total_time = 0.5  # 총 0.5초 시뮬레이션
    time_steps = np.arange(0, total_time, dt)
    
    # 제어기 게인 설정
    pi_d = CurrentPIController(kp=10.0, ki=200.0, output_limit=50.0)
    pi_q = CurrentPIController(kp=10.0, ki=200.0, output_limit=50.0)
    pi_speed = SpeedPIController(kp=0.3, ki=1.0, output_limit=50.0)
    
    # 모터 초기 상태 변수
    id_curr = 0.0
    iq_curr = 0.0
    wm_curr = 0.0
    theta_m_curr = 0.0
    
    # 기록용 리스트
    history_time = []
    history_target_rpm = []
    history_output_rpm = []
    history_torque = []
    history_id = []
    history_iq = []
    history_load_torque = []
    load_torque = 0.0  # 외부 부하 토크 (Nm)
    print("속도 제어 시뮬레이션을 실행 중입니다...")
    
    for t in time_steps:
        # 1. 목표 스피드 지령 설정: 0.05초 이후부터 3,000 RPM 지령
        target_rpm = 10000 if t >= 0.01 else 0.0
        
        # 2. 현재 스피드 [RPM] 계산
        current_rpm = wm_curr * (60.0 / (2.0 * np.pi))
        
        # 3. 속도 PI 제어기를 통한 q축 전류 지령치(iq_ref) 생성
        iq_ref = pi_speed.compute(target_rpm, current_rpm, dt)
        id_ref = pi_speed.compute(target_rpm, current_rpm, dt)  # Id=0 제어 적용
        
        # 4. 전기각 및 속도(rad/s) 계산
        theta_e = sim.p * theta_m_curr
        omega_e = sim.p * wm_curr
        
        # 5. 자속 쇄교수 조회 (디커플링용)
        current_mag = np.sqrt(id_curr**2 + iq_curr**2)
        beta_deg = 0.0 if current_mag < 1e-6 else np.degrees(np.arctan2(iq_curr, id_curr))
        ld, lq = sim.get_flux(current_mag, beta_deg)
        
        # 6. 전류 PI 제어기를 통한 전압 지령치 계산
        v_d_pi = pi_d.compute(id_ref, id_curr, dt)
        v_q_pi = pi_q.compute(iq_ref, iq_curr, dt)
        
        # 디커플링 보상항 적용
        vd_star = v_d_pi# - omega_e * lq
        vq_star = v_q_pi# + omega_e * ld
        
        # 7. 역 Park 변환을 통해 3상 전압 생성
        va, vb, vc = sim.dq_to_abc(np.array([vd_star, vq_star]), theta_e)
        
        # 8. 모터 시뮬레이터 1스텝 실행[cite: 1]
        ia, ib, ic, Te, wm_curr, theta_m_curr, id_curr, iq_curr, vd, vq, beta_deg = sim.simulate_step(
            va, vb, vc, id_curr, iq_curr, wm_curr, theta_m_curr, dt, load_torque)
        
        load_torque = 10 if t >= 0.25 else 0.0
        # 9. 데이터 기록
        history_time.append(t * 1000)  # [ms]
        history_target_rpm.append(target_rpm)
        history_output_rpm.append(current_rpm)
        history_torque.append(Te)
        history_id.append(id_curr)
        history_iq.append(iq_curr)
        history_load_torque.append(load_torque)
    # --- 결과 시각화 (3개 서브플롯) ---
    plt.figure(figsize=(10, 9))
    
    # 1. 목표 스피드 vs 출력 스피드 그래프
    plt.subplot(3, 1, 1)
    plt.plot(history_time, history_target_rpm, 'r--', label='Target RPM', lw=1.5)
    plt.plot(history_time, history_output_rpm, 'g-', label='Output RPM', lw=1.5)
    plt.title('Speed Control Response (Target vs Output)')
    plt.ylabel('Speed [RPM]')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 전자기 토크 그래프
    plt.subplot(3, 1, 2)
    plt.plot(history_time, history_torque, 'b-', label='Torque (Te)', lw=1.5)
    plt.plot(history_time, history_load_torque, 'r-', label='Load Torque', lw=1.5)
    plt.title('Electromagnetic Torque')
    plt.ylabel('Torque [Nm]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. d-q축 전류 (id, iq) 그래프
    plt.subplot(3, 1, 3)
    plt.plot(history_time, history_id, 'm-', label='id current', lw=1.5)
    plt.plot(history_time, history_iq, 'c-', label='iq current', lw=1.5)
    plt.title('d-q Axis Currents (id, iq)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Current [A]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    input("Press Enter to exit...")