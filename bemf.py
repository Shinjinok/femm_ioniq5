import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from ioq5_model import IPMSMSimulator


# --- 강제 속도 인가 및 BEMF 측정 구동 예제 ---
if __name__ == "__main__":
    sim = IPMSMSimulator('FEMM_Lambda_d_matrix.csv', 'FEMM_Lambda_q_matrix.csv', pole_pairs=4, rs=0.0618)
    
    dt = 0.0001  # 100 us 샘플링 타임
    total_time = 0.3  # 0.3초 시뮬레이션
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
    history_theta_m = []

    #a, b = sim.get_flux(0.0, 0.0)
    
    print(f"강제 속도 {target_rpm} RPM 인가에 따른 역기전력(BEMF)을 계산 중입니다...")
    va, vb, vc = 0.0, 0.0, 0.0
    for t in time_steps:
        # 1. 속도 제어 없이 강제 회전각(theta_m) 및 전기각(theta_e) 적분 업데이트
        theta_m_curr += wm_forced * dt
        theta_m_curr = np.mod(theta_m_curr, 2.0 * np.pi)
        theta_e = sim.p * theta_m_curr
        omega_e = sim.p * wm_forced
        
        # 2. 무부하 상태(전압 인가 없음: va=0, vb=0, vc=0) 가정 하에 시뮬레이터 스텝 실행
        # (순수 회전에 의한 역기전력 성분 추출)
        ia, ib, ic, Te, wm_curr, theta_m_curr, id_curr, iq_curr, vd, vq, beta_deg = sim.simulate_step(
            va, vb, vc, id_curr, iq_curr, wm_forced, theta_m_curr, dt, load_torque=0.0)
        # 데이터 기록
        history_time.append(t * 1000)
        history_ea.append(ia*sim.Rs)  # ia * Rs = ea (역기전력)
        history_eb.append(ib*sim.Rs)
        history_ec.append(ic*sim.Rs)
        history_ed.append(id_curr)
        history_eq.append(iq_curr)
        history_theta_m.append(theta_m_curr*(180.0/np.pi))  # rad -> deg 변환
    # --- 결과 시각화 (3상 및 dq축 역기전력) ---
    plt.figure(figsize=(10, 8))
    
    # 1. 3상 역기전력 그래프
    plt.subplot(3, 1, 1)
    plt.plot(history_time, history_ea, 'r-', label='ea', alpha=0.9)
    plt.plot(history_time, history_eb, 'g-', label='eb', alpha=0.9)
    plt.plot(history_time, history_ec, 'b-', label='ec', alpha=0.9)
    plt.title(f'Three-Phase Back-EMF at {target_rpm} RPM (ea, eb, ec)')
    plt.ylabel('Back-EMF [V]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. dq축 역기전력 그래프
    plt.subplot(3, 1, 2)
    plt.plot(history_time, history_theta_m, 'm-', label='theta_m(deg)', lw=1.5)
    #plt.plot(history_time, history_eq, 'c-', label='eq (omega_e * ld)', lw=1.5)
    plt.title('d-q Axis Back-EMF Components (theta_m )')
    plt.xlabel('Time [ms]')
    plt.ylabel('deg')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. dq축 전류 그래프
    plt.subplot(3, 1, 3)
    plt.plot(history_time, history_ed, 'm-', label='ed', lw=1.5)
    plt.plot(history_time, history_eq, 'c-', label='eq', lw=1.5)
    plt.title('d-q Axis Currents (ed, eq)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Current [A]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    input("Press Enter to exit...")