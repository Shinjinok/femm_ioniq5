import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class IPMSMSimulator:
    def __init__(self, csv_d_path, csv_q_path, pole_pairs=4, rs=0.05, J=0.01, B=0.0001):        """
        IPMSM 시뮬레이터 초기화
        - csv_d_path: d축 자속쇄교수 CSV 파일 경로
        - csv_q_path: q축 자속쇄교수 CSV 파일 경로
        - pole_pairs: 극쌍수 (Default: 4)
        - rs: 고정자 권선 저항 [ohm]
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
        """
        전류 크기(I)와 전류각(beta)에 따른 d, q축 자속쇄교수 반환
        """
        I_clamped = np.clip(I, self.currents[0], self.currents[-1])
        beta_clamped = np.clip(beta_deg, self.betas[0], self.betas[-1])
        
        pts = np.array([[beta_clamped, I_clamped]])
        ld = self.interp_d(pts)[0]
        lq = self.interp_q(pts)[0]
        return ld, lq

    def calculate_torque(self, ld, lq, id_, iq_):
        """
        전자기 토크 계산 [Nm]
        Te = 1.5 * p * (lambda_d * iq - lambda_q * id)
        """
        torque = 1.5 * self.p * (ld * iq_ - lq * id_)
        return torque
    
    def abc_to_dq0(self, V, theta_e):
        """3상 전압(abc)을 회전 자속 좌표계(dq) 전압으로 변환 (Park 변환)"""
        # 파크 변환 행렬 적용 (Clarke -> Park)
        cost = np.cos(theta_e)
        sint = np.sin(theta_e)
        
        # 클라크 변환 (1부 분모 2/3)
        v_alpha = 2.0 / 3.0 * (V[0] - 0.5 * V[1] - 0.5 * V[2])
        v_beta = 2.0 / 3.0 * (np.sqrt(3.0) / 2.0 * V[1] - np.sqrt(3.0) / 2.0 * V[2])
        
        # 파크 변환
        vd = v_alpha * cost + v_beta * sint
        vq = -v_alpha * sint + v_beta * cost
        return vd, vq

    def dq0_to_abc(self, ia, ib, ic, theta_e):
        """(역변환용 참고용 메서드) 필요시 사용"""
        pass

    def simulate_step(self, V, id_prev, iq_prev, wm_prev, theta_m_prev, dt, load_torque=0.0):
        """
        단일 샘플링 타임(dt) 동안의 모터 상태 업데이트 함수
        - 입력: 3상 전압 (va, vb, vc), 이전 d-q 전류, 이전 기계각속도, 이전 기계각도, dt, 부하 토크
        - 출력: 3상 전류 (ia, ib, ic), 전자기 토크 (Te), 기계 각속도 (wm), 기계 각도 (theta_m), 업데이트된 d-q 전류
        """
        # 1. 전기각 계산 (theta_e = p * theta_m)
        theta_e = self.p * theta_m_prev
        
        # 2. 3상 전압을 dq 전압으로 변환
        vd, vq = self.abc_to_dq0(V, theta_e)
        
        # 3. 이전 전류로부터 전류 크기(I)와 전류각(beta) 계산
        current_mag = np.sqrt(id_prev**2 + iq_prev**2)
        if current_mag < 1e-6:
            beta_deg = 0.0
        else:
            # iq = I*sin(beta), -id = I*cos(beta) 관계 이용
            beta_deg = np.degrees(np.arctan2(iq_prev, -id_prev))
            
        # 4. 자속 쇄교수 및 토크 계산
        ld, lq = self.get_flux(current_mag, beta_deg)
        Te = self.calculate_torque(ld, lq, id_prev, iq_prev)
        
        # 5. 전기각 속도 (omega_e = p * omega_m)
        omega_e = self.p * wm_prev
        
        # 6. 전압 방정식을 통한 전류 미분값 계산 (di_d/dt, di_q/dt)
        # v_d = Rs*id + L_d*(did/dt) - omega_e * lambda_q  =>  did/dt = (vd - Rs*id + omega_e*lq) / L_d (간이 인덕턴스 근사 대신 자속 미분 반영)
        # 수치 안정성을 위해 동적 미분 방정식 반영 (여기서는 평균 등가 인덕턴스 혹은 직접 자속 변화율 기반 미분 적용)
        # 범용적인 수치 해석을 위해 역기전력 항을 분리하여 전류 변화율 산출
        L_d_approx = max(ld / max(current_mag, 1.0), 0.001) # 대략적인 미분 인덕턴스 근사
        L_q_approx = max(lq / max(current_mag, 1.0), 0.001)
        
        did_dt = (vd - self.Rs * id_prev + omega_e * lq) / L_d_approx
        diq_dt = (vq - self.Rs * iq_prev - omega_e * ld) / L_q_approx
        
        # 오일러 적분으로 다음 스텝 전류 계산
        id_next = id_prev + did_dt * dt
        iq_next = iq_prev + diq_dt * dt
        
        # 7. 기계적 운동 방정식 (Motion Equation): J * d(wm)/dt = Te - T_load - B * wm
        dwm_dt = (Te - load_torque - self.B * wm_prev) / self.J
        wm_next = wm_prev + dwm_dt * dt
        
        # 기계 각도 업데이트 (theta_m = theta_m + wm * dt)
        theta_m_next = theta_m_prev + wm_prev * dt
        theta_m_next = np.mod(theta_m_next, 2.0 * np.pi) # 0 ~ 2*pi 주기화
        
        # 8. 역 Park 변환을 통해 3상 전류(ia, ib, ic) 계산
        # i_alpha = id*cos - iq*sin
        # i_beta  = id*sin + iq*cos
        cost = np.cos(theta_e)
        sint = np.sin(theta_e)
        i_alpha = id_next * cost - iq_next * sint
        i_beta  = id_next * sint + iq_next * cost
        
        ia = i_alpha
        ib = -0.5 * i_alpha + (np.sqrt(3.0) / 2.0) * i_beta
        ic = -0.5 * i_alpha - (np.sqrt(3.0) / 2.0) * i_beta
        
        return ia, ib, ic, Te, wm_next, theta_m_next, id_next, iq_next    
    
    def find_mtpa(self, target_current):
        """
        주어진 전류 크기에서 최대 토크를 내는 최적의 전류각(Beta) 찾기 (MTPA 탐색)
        """
        best_beta = 0.0
        max_torque = -float('inf')
        
        # 전류각 0도 ~ 180도 탐색
        test_betas = np.linspace(0, 180, 361)
        for beta in test_betas:
            id_ = -target_current * np.cos(np.radians(beta))
            iq_ = target_current * np.sin(np.radians(beta))
            ld_t, lq_t = self.get_flux(target_current, beta)
            T = self.calculate_torque(ld_t, lq_t, id_, iq_)
            if T > max_torque:
                max_torque = T
                best_beta = beta
                
        id_opt = -target_current * np.cos(np.radians(best_beta))
        iq_opt = target_current * np.sin(np.radians(best_beta))
        return best_beta, id_opt, iq_opt, max_torque

# --- 시뮬레이터 실행 및 시각화 예제 ---
if __name__ == "__main__":
    # 시뮬레이터 객체 생성 (파일 경로 지정)
    sim = IPMSMSimulator('FEMM_Lambda_d_matrix.csv', 'FEMM_Lambda_q_matrix.csv', pole_pairs=4, rs=0.05)
    
    # 1. 특정 전류점에서 자속 및 토크 계산 테스트
    I_test = 204 # [A]
    beta_test = 135.0 # [deg]
    id_t = I_test * np.cos(np.radians(beta_test))
    iq_t = I_test * np.sin(np.radians(beta_test))
    ld_t, lq_t = sim.get_flux(I_test, beta_test)
    torque_t = sim.calculate_torque(ld_t, lq_t, id_t, iq_t)
    
    print(f"=== 단일 운전점 테스트 (I={I_test}A, Beta={beta_test}deg) ===")
    print(f"id = {id_t:.2f} A, iq = {iq_t:.2f} A")
    print(f"lambda_d = {ld_t*1000:.2f} mWb, lambda_q = {lq_t*1000:.2f} mWb")
    print(f"전자기 토크 = {torque_t:.2f} Nm\n")
    
    # 2. 전류 크기별 MTPA(최대 토크 제어) 궤적 계산 (340A 제한까지)
    max_limit_current = min(340.0, max(sim.currents))
    current_range = np.linspace(10, max_limit_current, 30)
    mtpa_betas = []
    mtpa_torques = []
    mtpa_ids = []
    mtpa_iqs = []
    
    for I in current_range:
        beta_opt, id_opt, iq_opt, T_opt = sim.find_mtpa(I)
        mtpa_betas.append(beta_opt)
        mtpa_torques.append(T_opt)
        mtpa_ids.append(id_opt)
        mtpa_iqs.append(iq_opt)
        
    # 3. d-q 평면 격자 데이터 생성 (최대 340A 기준)
    grid_max = max(340.0, max(sim.currents))
    id_vals = np.linspace(-grid_max, 0, 100)
    iq_vals = np.linspace(0, grid_max, 100)
    ID, IQ = np.meshgrid(id_vals, iq_vals)
    
    Torque_grid = np.zeros_like(ID)
    Voltage_grid = np.zeros_like(ID)
    
    # 전압 제한 타원 계산을 위한 운전 속도 설정 (예: 2000 RPM)
    N_rpm = 2000.0 
    omega_e = 2.0 * np.pi * N_rpm * (sim.p / 60.0) # 전기각 속도 [rad/s]
    V_max = 220.0 # 최대 허용 전압 [V] (예시 값, 모터 스펙에 맞게 조절 가능)

    for i in range(ID.shape[0]):
        for j in range(ID.shape[1]):
            curr = np.sqrt(ID[i, j]**2 + IQ[i, j]**2)
            if curr <= sim.currents[-1]:
                beta = np.degrees(np.arctan2(IQ[i, j], -ID[i, j]))
                ld, lq = sim.get_flux(curr, beta)
                Torque_grid[i, j] = sim.calculate_torque(ld, lq, ID[i, j], IQ[i, j])
                
                # 고정자 전압 크기 계산 (V_s = sqrt(V_d^2 + V_q^2))
                # V_d = Rs * id - omega_e * lq
                # V_q = Rs * iq + omega_e * ld
                vd = sim.Rs * ID[i, j] - omega_e * lq
                vq = sim.Rs * IQ[i, j] + omega_e * ld
                Voltage_grid[i, j] = np.sqrt(vd**2 + vq**2)
            else:
                Torque_grid[i, j] = np.nan
                Voltage_grid[i, j] = np.nan

    # 4. 결과 그래프 출력 
    # [그림 1] MTPA 특성 곡선 (전류각 및 토크)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(current_range, mtpa_betas, 'r-o', linewidth=2)
    plt.title('MTPA Optimal Current Angle')
    plt.xlabel('Current Magnitude [A]')
    plt.ylabel('Optimal Beta [deg]')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(current_range, mtpa_torques, 'b-o', linewidth=2)
    plt.title('MTPA Maximum Torque')
    plt.xlabel('Current Magnitude [A]')
    plt.ylabel('Torque [Nm]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # [그림 2] d-q 평면: 등토크 곡선 + 340A 전류 제한원 + 전압 제한 타원 + MTPA 궤적
    plt.figure(figsize=(8, 8))
    
    # 등토크 곡선
    cs = plt.contour(ID, IQ, Torque_grid, levels=15, cmap='viridis')
    plt.clabel(cs, inline=True, fontsize=9, fmt='%.1f Nm')
    
    # 전압 제한 타원 (V_max 경계선 표시)
    v_contour = plt.contour(ID, IQ, Voltage_grid, levels=[V_max], colors='magenta', linewidths=2.0, linestyles='dashdot')
    plt.clabel(v_contour, inline=True, fontsize=9, fmt=f'Voltage Limit ({V_max}V)')

    # MTPA 궤적 표시
    plt.plot(mtpa_ids, mtpa_iqs, 'r-o', linewidth=2.5, label='MTPA Trajectory')
    
    # 전류 제한원 (340A까지 85A 간격으로 표시)
    circle_currents = [85, 170, 255, 340]
    theta = np.linspace(0, np.pi/2, 100)
    for I_circle in circle_currents:
        if I_circle <= sim.currents[-1]:
            c_id = -I_circle * np.cos(theta)
            c_iq = I_circle * np.sin(theta)
            plt.plot(c_id, c_iq, 'k--', alpha=0.5)
            plt.text(c_id[50], c_iq[50] + 5, f'{int(I_circle)}A', fontsize=9, color='k', weight='bold')

    plt.title(f'd-q Plane: Torque, Current Circles (340A), Voltage Ellipse @ {int(N_rpm)}RPM')
    plt.xlabel('d-axis Current id [A]')
    plt.ylabel('q-axis Current iq [A]')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.axhline(0, color='gray', linewidth=0.8)
    plt.axvline(0, color='gray', linewidth=0.8)
    #plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()

    # 시뮬레이션 설정 (예: 0.1초 동안 100us 간격으로 시뮬레이션 수행)
    dt = 0.0001 # 100 us
    total_time = 0.05 # 50 ms
    time_steps = np.arange(0, total_time, dt)
    
    # 초기 상태 변수
    id_curr = 0.0
    iq_curr = 0.0
    wm_curr = 0.0 # 기계 각속도 [rad/s]
    theta_m_curr = 0.0 # 기계 각도 [rad]
    
    # 기록용 리스트
    history_ia = []
    history_torque = []
    history_wm = []
    history_theta = []
    
    # 인가할 3상 전압 예시 (예: 60Hz 정현파 전압 입력)
    f_sys = 60.0
    V_peak = 100.0
    
    for t in time_steps:
        omega_v = 2.0 * np.pi * f_sys * t
        va = V_peak * np.sin(omega_v)
        vb = V_peak * np.sin(omega_v - 2.0 * np.pi / 3.0)
        vc = V_peak * np.sin(omega_v + 2.0 * np.pi / 3.0)
        
        # 함수 호출
        ia, ib, ic, Te, wm_curr, theta_m_curr, id_curr, iq_curr = sim.simulate_step(
            va, vb, vc, id_curr, iq_curr, wm_curr, theta_m_curr, dt, load_torque=5.0
        )
        
        history_ia.append(ia)
        history_torque.append(Te)
        history_wm.append(wm_curr * (60.0 / (2.0 * np.pi))) # rad/s -> RPM 변환
        history_theta.append(theta_m_curr)

    # 결과 플롯 확인
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_steps * 1000, history_ia, 'r-')
    plt.title('Phase A Current')
    plt.ylabel('Current [A]')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_steps * 1000, history_torque, 'b-')
    plt.title('Electromagnetic Torque')
    plt.ylabel('Torque [Nm]')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_steps * 1000, history_wm, 'g-')
    plt.title('Mechanical Speed')
    plt.xlabel('Time [ms]')
    plt.ylabel('Speed [RPM]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()