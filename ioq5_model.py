import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class IPMSMSimulator:
    def __init__(self, csv_d_path, csv_q_path, csv_0_path, pole_pairs=4, rs=0.06, J=0.01, B=0.00):
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

    def get_flux(self, Idq0):
        """전류 크기(I)와 전류각(beta)에 따른 d, q축 자속쇄교수 반환"""
        current_mag = np.linalg.norm(Idq0)
        if current_mag < 1e-6:
            beta_deg = 0.0
        else:
            beta_deg = np.degrees(np.arctan2(Idq0[1], Idq0[0]))
            
        I_clamped = np.clip(current_mag, self.currents[0], self.currents[-1])
        beta_clamped = np.clip(beta_deg, self.betas[0], self.betas[-1])
        
        pts = np.array([[beta_clamped, I_clamped]])
        ld = self.interp_d(pts)[0]
        lq = self.interp_q(pts)[0]
        l0 = self.interp_0(pts)[0]

        return np.array([ld, lq, l0])

    def calculate_torque(self, lamda, Idq0):
        """전자기 토크 계산 [Nm]"""
        #torque = 1.5 * self.p * (ld * iq_ - lq * id_)
        torque = 1.5 * self.p * np.cross(lamda, Idq0)

        return torque

    def abc_to_dq(self,i_abc, b):
        # 전체를 감싸는 바깥쪽 대괄호([ ])가 반드시 있어야 합니다.
        clac_3x3 = 2/3 * np.array([
            [1, -1/2, -1/2], 
            [0, np.sqrt(3)/2, -np.sqrt(3)/2],
            [1/2, 1/2, 1/2]
        ])
        
        park_3x3 = np.array([
            [np.cos(b), np.sin(b), 0], 
            [-np.sin(b), np.cos(b), 0],
            [0, 0, 1]
        ])
        
        dq0 = park_3x3 @ clac_3x3 @ i_abc
        return dq0

    def dq_to_abc(self, dq0, b):
        clac2 = np.array([[1, -1/2, -1/2], 
                        [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                        [1/2, 1/2, 1/2]])
        park_3x3 = np.array([
                    [np.cos(b), np.sin(b), 0], 
                    [-np.sin(b), np.cos(b), 0],
                    [0, 0, 1]
                ])
        ipark = np.linalg.inv(park_3x3)
        abc = clac2.T @ park_3x3.T @ dq0
        return abc

    
    def compute_incremental_inductance(self, Idq0, delta=1e-6):
        """
        dq0 좌표계에서 철심 포화와 상호 간섭을 고려한 3x3 증분 인덕턴스 행렬(L_inc) 계산
        - flux_map_func: 전류 [id, iq, i0]를 입력받아 자속 [lam_d, lam_q, lam_0]를 반환하는 함수
        """
        
        
        L_inc = np.zeros((3, 3))
               
        # 수치미분(Finite Difference)을 통한 Jacobian 행렬 구성
        for j in range(3):
            I_perturbed = Idq0.copy()
            I_perturbed[j] += delta  # 미소 전류 변화
                    
            # d(lambda)_i / d(i)_j 계산
            L_inc[:, j] = (self.get_flux(I_perturbed) - self.get_flux(Idq0)) / delta

        epsilon = 1e-10
        L_inc_reg = L_inc + epsilon * np.eye(L_inc.shape[0])    
        return L_inc_reg
    
    def compute_incremental_inductance2(self, Idq0, delta=1e-3):
            """
            dq0 좌표계에서 철심 포화와 상호 간섭을 고려한 3x3 증분 인덕턴스 행렬(L_inc) 계산
            - flux_map_func: 전류 [id, iq, i0]를 입력받아 자속 [lam_d, lam_q, lam_0]를 반환하는 함수
            """
                
            
            ld_p, lq_p ,l0_p= self.get_flux([Idq0[0] + delta, Idq0[1], Idq0[2]])
            ld_m, lq_m ,l0_m= self.get_flux([Idq0[0] - delta, Idq0[1], Idq0[2]])
            L_d1 = max((ld_p - ld_m) / (2.0 * delta), 0.001)
            L_d2 = max((lq_p - lq_m) / (2.0 * delta), 0.001)
            L_d3 = max((l0_p - l0_m) / (2.0 * delta), 0.001)
            
            ld_p, lq_p ,l0_p= self.get_flux([Idq0[0], Idq0[1] + delta, Idq0[2]])
            ld_m, lq_m ,l0_m= self.get_flux([Idq0[0], Idq0[1] - delta, Idq0[2]])
            L_q1 = max((ld_p - ld_m) / (2.0 * delta), 0.001)
            L_q2 = max((lq_p - lq_m) / (2.0 * delta), 0.001)
            L_q3 = max((l0_p - l0_m) / (2.0 * delta), 0.001)

            ld_p, lq_p ,l0_p = self.get_flux([Idq0[0], Idq0[1], Idq0[2]+delta])
            ld_m, lq_m ,l0_m = self.get_flux([Idq0[0], Idq0[1], Idq0[2]-delta])
            L_01 = max((ld_p - ld_m) / (2.0 * delta), 0.001)
            L_02 = max((lq_p - lq_m) / (2.0 * delta), 0.001)
            L_03 = max((l0_p - l0_m) / (2.0 * delta), 0.001)
            L_inc = np.array([[L_d1,L_d2, L_d3],
                            [L_q1, L_q2, L_q3],
                            [L_01, L_02, L_03]])    
            return L_inc
    
    def simulate_step(self, Vabc, Iabc, wm, theta_m, dt, load_torque=0.0):
        
        theta_e = np.mod(self.p * theta_m, 2.0 * np.pi) 
        Vdq0 = self.abc_to_dq(Vabc, theta_e)
        Idq0 = self.abc_to_dq(Iabc, theta_e)

      
        
        
        omega_e = self.p * wm
        
        # 2. 증분 인덕턴스 행렬 계산
        L_inc = self.compute_incremental_inductance(Idq0)
        
        # 3. 속도 기전력 항 수정 (전류 대신 자속 이용 예시)
        # 회전 좌표계 속도 기전력 전압 벡터: [-omega_e * lamda[1], omega_e * lamda[0], 0]
        omega = np.array([0, 0, omega_e])
        lamda = self.get_flux(Idq0)
        speed_emf = np.cross(omega, lamda)
        # 전류 변화율 계산 (V = Rs*i + d_lam/dt + speed_emf 형태 반영)
        try:
            # 역행렬 계산 시도
            L_inv = np.linalg.inv(L_inc)
            dIdq0_dt = L_inv @ (Vdq0 - self.Rs * Idq0 - speed_emf)

        except np.linalg.LinAlgError:
            print(
                "[Warning] L_inc is singular! Switching to Pseudoinverse (유사역행렬)."
            )
            # 특이 행렬일 때 무한대 폭발을 막기 위해 pinv(유사역행렬) 사용
            dIdq0_dt = np.linalg.pinv(L_inc) @ (Vdq0 - self.Rs * Idq0 - speed_emf)
        
        Idq0 = Idq0 + dIdq0_dt * dt

        lamda = self.get_flux(Idq0)
        Te_vector = 1.5 * self.p * np.cross(lamda, Idq0)
        Te_z = Te_vector[2]  # 스칼라 토크 값
        dwm_dt = (Te_z - load_torque - self.B * wm) / self.J
        
        wm = wm + dwm_dt * dt
        
        theta_m = theta_m + wm * dt
        theta_m = np.mod(theta_m, 2.0 * np.pi)
        theta_e = np.mod(self.p * theta_m, 2.0 * np.pi)
        Iabc = self.dq_to_abc(Idq0, theta_e)
        
        return Iabc, Te_z, wm, theta_m, Vdq0, Idq0, speed_emf


# --- 강제 속도 인가 및 BEMF 측정 구동 예제 ---
if __name__ == "__main__":
    sim = IPMSMSimulator('Rotate_FEMM_Lambda_d_matrix.csv', 'Rotate_FEMM_Lambda_q_matrix.csv', 'Rotate_FEMM_Lambda_0_matrix.csv', pole_pairs=4, rs=0.0618)
    
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
    history_Te = []

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
        Iabc, Te, wm_curr, theta_m_curr, Vdq0, Idq0, speed_emf = sim.simulate_step(
            Vabc, Iabc, wm_forced, theta_m_curr, dt, load_torque=0.0)
        # 데이터 기록
        history_time.append(t * 1000)
        history_ea.append(Iabc[0]*sim.Rs)  # ia * Rs = ea (역기전력)
        history_eb.append(Iabc[1]*sim.Rs)
        history_ec.append(Iabc[2]*sim.Rs)
        history_ed.append(speed_emf[0])
        history_eq.append(speed_emf[1])
        history_e0.append(speed_emf[2])
        current_mag = np.linalg.norm(Idq0)
        
        beta_deg = np.degrees(np.arctan2(Idq0[1], Idq0[0]))
        history_Te.append(Te)  # rad -> deg 변환
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
    plt.plot(history_time, history_Te, 'm-', label='Te', lw=1.5)
    #plt.plot(history_time, history_eq, 'c-', label='eq (omega_e * ld)', lw=1.5)
    plt.title('d-q Axis Back-EMF Components (Torque and EMF)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Torque [Nm]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. dq축 전류 그래프
    plt.subplot(3, 1, 3)
    plt.plot(history_time, history_ed, 'm-', label='speed_emf[0]', lw=1.5)
    plt.plot(history_time, history_eq, 'c-', label='speed_emf[1]', lw=1.5)
    plt.plot(history_time, history_e0, 'y-', label='speed_emf[2]', lw=1.5)
    plt.title('speed_emf Components (omega_e * lambda)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Back-EMF [V]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    input("Press Enter to exit...")