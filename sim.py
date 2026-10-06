import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class IPMSMSimulator:
    def __init__(self, csv_d_path, csv_q_path, pole_pairs=4, rs=0.05, J=0.001, B=0.0):
        """
        IPMSM 시뮬레이터 초기화
        - csv_d_path: d축 자속쇄교수 CSV 파일 경로
        - csv_q_path: q축 자속쇄교수 CSV 파일 경로
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
        
        # Beta (전류각, 도) 및 Current (전류 크기, A) 추출
        self.betas = df_d['Beta'].values
        self.currents = df_d.columns[1:].astype(float).values
        
        # 자속쇄교수 행렬 추출 (단위: mWb -> Wb 변환을 위해 1e-3 곱셈 적용)
        lambda_d_mat = df_d.iloc[:, 1:].values * 1e-3
        lambda_q_mat = df_q.iloc[:, 1:].values * 1e-3
        
        # 2. 2차원 보간기(Interpolator) 생성 (RegularGridInterpolator 사용)
        self.interp_d = RegularGridInterpolator((self.betas, self.currents), lambda_d_mat, method='linear', bounds_error=False, fill_value=None)
        self.interp_q = RegularGridInterpolator((self.betas, self.currents), lambda_q_mat, method='linear', bounds_error=False, fill_value=None)
        
    def get_flux(self, I, beta_deg):
        """전류 크기(I)와 전류각(beta)에 따른 d, q축 자속쇄교수 반환"""
        I_clamped = np.clip(I, self.currents[0], self.currents[-1])
        beta_clamped = np.clip(beta_deg, self.betas[0], self.betas[-1])
        
        pts = np.array([[beta_clamped, I_clamped]])
        ld = self.interp_d(pts)[0]
        lq = self.interp_q(pts)[0]
        return ld, lq

    def calculate_torque(self, ld, lq, id_, iq_):
        """전자기 토크 계산 [Nm]"""
        torque = 1.5 * self.p * (ld * iq_ - lq * id_)
        return torque

    def abc_to_dq(self,i_abc, b):
    
        #dq = np.array([0, 0])
        clac=2/3*np.array([[1, -1/2, -1/2], 
                            [0,np.sqrt(3)/2, -np.sqrt(3)/2]])
        park=np.array([[np.cos(b), np.sin(b)], 
                        [-np.sin(b), np.cos(b)]])
        dq= park@clac@i_abc
        return dq


    def dq_to_abc(self,v_dq, b):
        clac=np.array([[1, -1/2, -1/2], 
                        [0,np.sqrt(3)/2, -np.sqrt(3)/2]])
        iclac=np.transpose(clac)
        ipark=np.array([[np.cos(b), -np.sin(b)], 
                        [np.sin(b), np.cos(b)]])
        v=iclac@ipark@v_dq
        return v

    def simulate_step(self, va, vb, vc, id_prev, iq_prev, wm_prev, theta_m_prev, dt, load_torque=0.0):
        """
        단일 샘플링 타임(dt) 동안의 모터 상태 업데이트 함수
        """
        # 1. 전기각 계산 (theta_e = p * theta_m)
        theta_e = self.p * theta_m_prev
        v=np.array([va, vb, vc])
        # 2. 3상 전압을 dq 전압으로 변환
        vd, vq = self.abc_to_dq(v, theta_e)
        
        # 3. 이전 전류로부터 전류 크기(I)와 전류각(beta) 계산
        current_mag = np.sqrt(id_prev**2 + iq_prev**2)
        if current_mag < 1e-6:
            beta_deg = 0.0
        else:
            beta_deg = np.degrees(np.arctan2(iq_prev, -id_prev))
            
        # 4. 자속 쇄교수 및 토크 계산
        ld, lq = self.get_flux(current_mag, beta_deg+90)
        Te = self.calculate_torque(ld, lq, id_prev, iq_prev)
        
        # 5. 전기각 속도
        omega_e = self.p * wm_prev
        
        # 6. 전압 방정식을 통한 전류 미분값 계산
        L_d_approx = max(ld / max(current_mag, 1.0), 0.001)
        L_q_approx = max(lq / max(current_mag, 1.0), 0.001)
        
        did_dt = (vd - self.Rs * id_prev + omega_e * lq) / L_d_approx
        diq_dt = (vq - self.Rs * iq_prev - omega_e * ld) / L_q_approx
        
        id_next = id_prev + did_dt * dt
        iq_next = iq_prev + diq_dt * dt
        
        # 7. 기계적 운동 방정식
        dwm_dt = (Te - load_torque - self.B * wm_prev) / self.J
        wm_next = wm_prev + dwm_dt * dt
        
        theta_m_next = theta_m_prev + wm_prev * dt
        theta_m_next = np.mod(theta_m_next, 2.0 * np.pi)
        
        # 8. 역 Park 변환을 통해 3상 전류 계산
        ia, ib, ic = self.dq_to_abc(np.array([id_next, iq_next]), theta_e)
        
        
        return ia, ib, ic, Te, wm_next, theta_m_next, id_next, iq_next, vd, vq


# --- 시뮬레이터 구동 및 테스트 예제 ---
if __name__ == "__main__":
    sim = IPMSMSimulator('FEMM_Lambda_d_matrix.csv', 'FEMM_Lambda_q_matrix.csv', pole_pairs=4, rs=0.05)
    
    dt = 0.001 # 100 us
    total_time = 1 # 50 ms
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
    history_beta = [] # 전류각 기록용 리스트 추가
    
    f_sys = 60.0
    V_peak = 100.0
    
    for t in time_steps:
        
        omega_v = 2.0 * np.pi * f_sys * t
        va = V_peak * np.sin(omega_v)
        vb = V_peak * np.sin(omega_v + 2.0 * np.pi / 3.0)
        vc = V_peak * np.sin(omega_v - 2.0 * np.pi / 3.0)
        
        ia, ib, ic, Te, wm_curr, theta_m_curr, id_curr, iq_curr, vd, vq = sim.simulate_step(
            va, vb, vc, id_curr, iq_curr, wm_curr, theta_m_curr, dt, 0.0
        )
        
        # 현재 스텝의 전류각(Beta, 도) 계산
        current_mag = np.sqrt(id_curr**2 + iq_curr**2)
        if current_mag < 1e-6:
            beta_deg = 0.0
        else:
            beta_deg = np.degrees(np.arctan2(iq_curr, -id_curr))
        
        history_ia.append(ia)
        history_vd.append(vd)
        history_vq.append(vq)
        history_torque.append(Te)
        history_wm.append(wm_curr * (60.0 / (2.0 * np.pi)))
        history_beta.append(beta_deg)

    # 결과 플롯 확인 (5개 서브플롯 구성)
    plt.figure(figsize=(12, 12))
    
    plt.subplot(5, 1, 1)
    plt.plot(time_steps * 1000, history_ia, 'r-')
    plt.title('Phase A Current (ia)')
    plt.ylabel('Current [A]')
    plt.grid(True)
    
    plt.subplot(5, 1, 2)
    plt.plot(time_steps * 1000, history_vd, 'm-', label='vd')
    plt.plot(time_steps * 1000, history_vq, 'c-', label='vq')
    plt.title('d-q Axis Voltages (vd, vq)')
    plt.ylabel('Voltage [V]')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.subplot(5, 1, 3)
    plt.plot(time_steps * 1000, history_torque, 'b-')
    plt.title('Electromagnetic Torque (Te)')
    plt.ylabel('Torque [Nm]')
    plt.grid(True)
    
    plt.subplot(5, 1, 4)
    plt.plot(time_steps * 1000, history_wm, 'g-')
    plt.title('Mechanical Speed')
    plt.ylabel('Speed [RPM]')
    plt.grid(True)
    
    plt.subplot(5, 1, 5)
    plt.plot(time_steps * 1000, history_beta, 'k-')
    plt.title('Current Angle (Beta)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Angle [deg]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    input("Press Enter to exit...")