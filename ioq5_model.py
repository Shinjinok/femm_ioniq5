import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class IPMSMSimulator:
    def __init__(self, csv_d_path, csv_q_path, csv_0_path, pole_pairs=4, rs=0.05, J=0.01, B=0.00):
        """
        IPMSM 시뮬레이터 초기화
        - csv_d_path: d축 자속쇄교수 CSV 파일 경로
        - csv_q_path: q축 자속쇄교수 CSV 파일 경로
        - csv_0_path: 0축 자속쇄교수 CSV 파일 경로
        - pole_pairs: 극쌍수 (Default: 4)
        - rs: 고정자 권선 저항 [ohm]
        - J: 회전자 관성모멘트 [kg·m^2]
        - B: 마찰 계수 [Nm/(rad/s)]
        """
        self.p = pole_pairs
        self.Rs = rs
        self.J = J
        self.B = B
        
        # 1. CSV 데이터 로드
        df_d = pd.read_csv(csv_d_path)
        df_q = pd.read_csv(csv_q_path)
        df_0 = pd.read_csv(csv_0_path)

        # Beta (전류각, 도) 및 Current (전류 크기, A) 추출
        self.betas = df_d['Beta'].values
        self.currents = df_d.columns[1:].astype(float).values
        
        # 자속쇄교수 행렬 추출 (단위: mWb -> Wb 변환을 위해 1e-3 곱셈 적용)
        lambda_d_mat = df_d.iloc[:, 1:].values * 1e-3
        lambda_q_mat = df_q.iloc[:, 1:].values * 1e-3
        lambda_0_mat = df_0.iloc[:, 1:].values * 1e-3
        # 2. 2차원 보간기(Interpolator) 생성 (RegularGridInterpolator 사용)
        self.interp_d = RegularGridInterpolator((self.betas, self.currents), lambda_d_mat, method='linear', bounds_error=False, fill_value=None)
        self.interp_q = RegularGridInterpolator((self.betas, self.currents), lambda_q_mat, method='linear', bounds_error=False, fill_value=None)
        self.interp_0 = RegularGridInterpolator((self.betas, self.currents), lambda_0_mat, method='linear', bounds_error=False, fill_value=None)  

    def get_flux(self, I, beta_deg):
        """전류 크기(I)와 전류각(beta)에 따른 d, q축 자속쇄교수 반환"""
        I_clamped = np.clip(I, self.currents[0], self.currents[-1])
        beta_clamped = np.clip(beta_deg, self.betas[0], self.betas[-1])
        
        pts = np.array([[beta_clamped, I_clamped]])
        ld = self.interp_d(pts)[0]
        lq = self.interp_q(pts)[0]
        l0 = self.interp_0(pts)[0]

        return ld, lq, l0

    def calculate_torque(self, ld, lq, id_, iq_):
        """전자기 토크 계산 [Nm]"""
        torque = 1.5 * self.p * (ld * iq_ - lq * id_)
        return torque

    def abc_to_dq(self, i_abc, b):
        clac = 2/3 * np.array([[1, -1/2, -1/2], 
                               [0, np.sqrt(3)/2, -np.sqrt(3)/2]],
                               [1/2, 1/2, 1/2]] )
        
        park = np.array([[np.cos(b), np.sin(b)], 
                         [-np.sin(b), np.cos(b)]])
        dq = park @ clac @ i_abc
        return dq

    def dq_to_abc(self, v_dq, b):
        clac = np.array([[1, -1/2, -1/2], 
                         [0, np.sqrt(3)/2, -np.sqrt(3)/2]],
                         [1/2, 1/2, 1/2]])
        iclac = np.transpose(clac)
        ipark = np.array([[np.cos(b), -np.sin(b)], 
                          [np.sin(b), np.cos(b)]])
        v = iclac @ ipark @ v_dq
        return v

    def simulate_step(self, va, vb, vc, id_prev, iq_prev, wm_prev, theta_m_prev, dt, load_torque=0.0):
        """단일 샘플링 타임(dt) 동안의 모터 상태 업데이트 함수 (증분 인덕턴스 적용)"""
        theta_e = self.p * theta_m_prev
         
        vd, vq, v0 = self.abc_to_dq(np.array([va, vb, vc]), theta_e)
        
        # 1. 현재 전류 크기 및 전류각 계산
        current_mag = np.sqrt(id_prev**2 + iq_prev**2)
        if current_mag < 1e-6:
            beta_deg = 0.0
        else:
            beta_deg = np.degrees(np.arctan2(iq_prev, id_prev))
           
        ld, lq, l0 = self.get_flux(current_mag, beta_deg)
        Te = self.calculate_torque(ld, lq, id_prev, iq_prev)
        
        omega_e = self.p * wm_prev
        
        # --- 2. 증분 인덕턴스 (Incremental Inductance) 수치 미분 계산 ---
        delta_i = 1e-3  # 미소 전류 변동분 [A]
        
        def _get_flux_from_dq(id_val, iq_val):
            mag = np.sqrt(id_val**2 + iq_val**2)
            if mag < 1e-6:
                b = 0.0
            else:
                b = np.degrees(np.arctan2(iq_val, id_val))
            return self.get_flux(mag, b)

        ld_p, _ = _get_flux_from_dq(id_prev + delta_i, iq_prev)
        ld_m, _ = _get_flux_from_dq(id_prev - delta_i, iq_prev)
        L_d_approx = max((ld_p - ld_m) / (2.0 * delta_i), 0.001)
        
        _, lq_p = _get_flux_from_dq(id_prev, iq_prev + delta_i)
        _, lq_m = _get_flux_from_dq(id_prev, iq_prev - delta_i)
        L_q_approx = max((lq_p - lq_m) / (2.0 * delta_i), 0.001)
        # -----------------------------------------------------------
        
        did_dt = (vd - self.Rs * id_prev + omega_e * lq) / L_d_approx
        diq_dt = (vq - self.Rs * iq_prev - omega_e * ld) / L_q_approx

       
        id_next = id_prev + did_dt * dt
        iq_next = iq_prev + diq_dt * dt
        
        dwm_dt = (Te - load_torque - self.B * wm_prev) / self.J
        wm_next = wm_prev + dwm_dt * dt
        
        theta_m_next = theta_m_prev + wm_prev * dt
        theta_m_next = np.mod(theta_m_next, 2.0 * np.pi)
        
        ia, ib, ic = self.dq_to_abc(np.array([id_next, iq_next]), theta_e)
        
        current_mag_next = np.sqrt(id_next**2 + iq_next**2)
        if current_mag_next < 1e-6:
            beta_deg_next = 0.0
        else:
            beta_deg_next = np.degrees(np.arctan2(iq_next, id_next))

        return ia, ib, ic, Te, wm_next, theta_m_next, id_next, iq_next, vd, vq, beta_deg_next


# --- 시뮬레이터 구동 및 테스트 예제 ---
if __name__ == "__main__":
    sim = IPMSMSimulator('FEMM_Lambda_d_matrix.csv', 'FEMM_Lambda_q_matrix.csv', 'FEMM_Lambda_0_matrix.csv', pole_pairs=4, rs=0.05)
    
    dt = 0.001 # 100 us
    total_time = 10.0 # 1 s
    time_steps = np.arange(0, total_time, dt)
    
    id_curr = 0.0
    iq_curr = 0.0
    wm_curr = 0.0
    theta_m_curr = 0.0
    
    # 기록용 리스트
    history_ia = []
    history_vd = []
    history_vq = []
    history_torque = []
    history_wm = []
    history_beta = [] 
    history_theta_m = [] # 기계적 위치 기록용 리스트 추가
    
    f_sys = 1
    V_peak = 1
    
    for t in time_steps:
        theta_rr = 2.0 * np.pi * f_sys * t
        va = V_peak * np.sin(theta_rr)
        vb = V_peak * np.sin(theta_rr - 2.0 * np.pi / 3.0)
        vc = V_peak * np.sin(theta_rr + 2.0 * np.pi / 3.0)
        
        ia, ib, ic, Te, wm_curr, theta_m_curr, id_curr, iq_curr, vd, vq, beta_deg = sim.simulate_step(
            va, vb, vc, id_curr, iq_curr, wm_curr, theta_m_curr, dt, 0.0
        )
        
        history_ia.append(ia)
        history_vd.append(vd)
        history_vq.append(vq)
        history_torque.append(Te)
        history_wm.append(wm_curr * (60.0 / (2.0 * np.pi)))
        history_beta.append(beta_deg)
        history_theta_m.append(np.degrees(theta_m_curr)) # 도(deg) 단위로 변환 저장

    # 결과 플롯 확인 (6개 서브플롯 구성)
    plt.figure(figsize=(8, 14))
    final_t = time_steps[-1] * 1000
    
    # 1. Phase A Current
    plt.subplot(6, 1, 1)
    plt.plot(time_steps * 1000, history_ia, 'r-')
    plt.title(f'Phase A Current (ia) | Final: {history_ia[-1]:.4f} A')
    plt.ylabel('Current [A]')
    plt.grid(True)
    plt.plot(final_t, history_ia[-1], 'ro')
    plt.text(final_t, history_ia[-1], f'  {history_ia[-1]:.4f} A', color='r', verticalalignment='center')
    
    # 2. d-q Axis Voltages
    plt.subplot(6, 1, 2)
    plt.plot(time_steps * 1000, history_vd, 'm-', label='vd')
    plt.plot(time_steps * 1000, history_vq, 'c-', label='vq')
    plt.title(f'd-q Axis Voltages | Final vd: {history_vd[-1]:.4f} V, vq: {history_vq[-1]:.4f} V')
    plt.ylabel('Voltage [V]')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.plot(final_t, history_vd[-1], 'mo')
    plt.text(final_t, history_vd[-1], f'  {history_vd[-1]:.4f}', color='m', verticalalignment='center')
    plt.plot(final_t, history_vq[-1], 'co')
    plt.text(final_t, history_vq[-1], f'  {history_vq[-1]:.4f}', color='c', verticalalignment='center')
    
    # 3. Electromagnetic Torque
    plt.subplot(6, 1, 3)
    plt.plot(time_steps * 1000, history_torque, 'b-')
    plt.title(f'Electromagnetic Torque (Te) | Final: {history_torque[-1]:.4f} Nm')
    plt.ylabel('Torque [Nm]')
    plt.grid(True)
    plt.plot(final_t, history_torque[-1], 'bo')
    plt.text(final_t, history_torque[-1], f'  {history_torque[-1]:.4f} Nm', color='b', verticalalignment='center')
    
    # 4. Mechanical Speed
    plt.subplot(6, 1, 4)
    plt.plot(time_steps * 1000, history_wm, 'g-')
    plt.title(f'Mechanical Speed | Final: {history_wm[-1]:.2f} RPM')
    plt.ylabel('Speed [RPM]')
    plt.grid(True)
    plt.plot(final_t, history_wm[-1], 'go')
    plt.text(final_t, history_wm[-1], f'  {history_wm[-1]:.2f} RPM', color='g', verticalalignment='center')
    
    # 5. Current Angle (Beta)
    plt.subplot(6, 1, 5)
    plt.plot(time_steps * 1000, history_beta, 'k-')
    plt.title(f'Current Angle (Beta) | Final: {history_beta[-1]:.2f} deg')
    plt.ylabel('Angle [deg]')
    plt.grid(True)
    plt.plot(final_t, history_beta[-1], 'ko')
    plt.text(final_t, history_beta[-1], f'  {history_beta[-1]:.2f} deg', color='k', verticalalignment='center')

    # 6. Mechanical Rotor Angle (Theta_m)
    plt.subplot(6, 1, 6)
    plt.plot(time_steps * 1000, history_theta_m, color='darkorange')
    plt.title(f'Rotor Angle (Theta_m) | Final: {history_theta_m[-1]:.2f} deg')
    plt.xlabel('Time [ms]')
    plt.ylabel('Angle [deg]')
    plt.grid(True)
    plt.plot(final_t, history_theta_m[-1], 'o', color='darkorange')
    plt.text(final_t, history_theta_m[-1], f'  {history_theta_m[-1]:.2f} deg', color='darkorange', verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

    input("Press Enter to exit...")