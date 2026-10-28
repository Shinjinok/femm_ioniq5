import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from ref_file2.ioq5_model import IPMSMSimulator


# --- 강제 속도 인가 및 BEMF 측정 구동 예제 ---
if __name__ == "__main__":
    sim = IPMSMSimulator('FEMM_Lambda_d_matrix.csv', 'FEMM_Lambda_q_matrix.csv', 'FEMM_Lambda_0_matrix.csv', pole_pairs=4, rs=0.0618)
    
    dt = 0.0001  # 100 us 샘플링 타임
    total_time = 0.1  # 0.5초 시뮬레이션
    time_steps = np.arange(0, total_time, dt)
    
    # 강제로 인가할 목표 속도 설정 [RPM] (예: 3,000 RPM)
    target_rpm = 141.03
    wm_forced = target_rpm * (2.0 * np.pi / 60.0)  # rad/s로 변환
    
    id_curr, iq_curr = 0.0, 0.0
    theta_m_curr = 0.0
    
    # 기록용 리스트
    history_time = []
    history_ea = []
    history_eb = []
    history_ec = []
    history_ed = []
    history_eq = []
    history_e0 = []
    history_beta = []

    #a, b = sim.get_flux(0.0, 0.0)
    
    print(f"강제 속도 {target_rpm} RPM 인가에 따른 역기전력(BEMF)을 계산 중입니다...")
    Vabc =  np.zeros(3)
    Iabc = np.zeros(3)
    for t in time_steps:
        # 1. 속도 제어 없이 강제 회전각(theta_m) 및 전기각(theta_e) 적분 업데이트
        theta_m_curr += wm_forced * dt
        theta_m_curr = np.mod(theta_m_curr, 2.0 * np.pi)
        theta_e = sim.p * theta_m_curr
        omega_e = sim.p * wm_forced
        
        # 2. 무부하 상태(전압 인가 없음: Vabc=0) 가정 하에 시뮬레이터 스텝 실행
        # (순수 회전에 의한 역기전력 성분 추출)
        Iabc, Te, wm_curr, theta_m_curr, Vdq0, Idq0 = sim.simulate_step(
            Vabc, Iabc, wm_forced, theta_m_curr, dt, load_torque=0.0)
        # 데이터 기록
        history_time.append(t * 1000)
        history_ea.append(Iabc[0]*sim.Rs)  # ia * Rs = ea (역기전력)
        history_eb.append(Iabc[1]*sim.Rs)
        history_ec.append(Iabc[2]*sim.Rs)
        history_ed.append(Idq0[0])
        history_eq.append(Idq0[1])
        history_e0.append(Idq0[2])
        current_mag = np.linalg.norm(Idq0)
        
        beta_deg = np.degrees(np.arctan2(Idq0[1], Idq0[0]))
        history_beta.append(beta_deg)  # rad -> deg 변환
    # --- 결과 시각화 (3상 및 dq축 역기전력) ---
    plt.figure(figsize=(10, 8))
    
    # 1. 3상 역기전력 그래프
    plt.subplot(3, 1, 1)
    plt.plot(history_time, history_ea, 'r-', label='ea', alpha=0.9)
    plt.plot(history_time, history_eb, 'g-', label='eb', alpha=0.9)
    plt.plot(history_time, history_ec, 'b-', label='ec', alpha=0.9)
    plt.title(f'Three-Phase Back-EMF at {target_rpm} RPM (ea, eb, ec) :Max peak :{max(history_ea):.2f} V')
    plt.ylabel('Back-EMF [V]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. dq축 역기전력 그래프
    plt.subplot(3, 1, 2)
    plt.plot(history_time, history_beta, 'm-', label='beta (deg)', lw=1.5)
    #plt.plot(history_time, history_eq, 'c-', label='eq (omega_e * ld)', lw=1.5)
    plt.title('d-q Axis Back-EMF Components (beta )')
    plt.xlabel('Time [ms]')
    plt.ylabel('deg')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. dq축 전류 그래프
    plt.subplot(3, 1, 3)
    plt.plot(history_time, history_ed, 'm-', label='id', lw=1.5)
    plt.plot(history_time, history_eq, 'c-', label='iq', lw=1.5)
    plt.plot(history_time, history_e0, 'y-', label='i0', lw=1.5)
    plt.title('d-q Axis Currents (id, iq, i0)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Current [A]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    input("Press Enter to exit...")