import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

class IPMSMSimulator:
    def __init__(self, csv_d_path, csv_q_path, csv_0_path, pole_pairs=4, rs=0.06, J=0.01, B=0.00):
        """
        IPMSM 시뮬레이터 초기화 (전류, 전류각, 쇄교자속의 3차원 보간 적용)
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
        self.L_epsilon = 1e-3

        # 1. CSV 데이터 로드
        df_d = pd.read_csv(csv_d_path)
        df_q = pd.read_csv(csv_q_path)
        df_0 = pd.read_csv(csv_0_path)

        # 3차원 축 구성: 전류(Current), 전류각(Beta), 쇄교자속(Flux_linkage)
        # 각 CSV 파일에 정의된 전류 크기, 전류각, 그리고 입력 쇄교자속 범위를 격자축으로 활용합니다.
        self.currents = np.sort(df_d['Current'].unique()) if 'Current' in df_d.columns else np.sort(df_d.columns[1:].astype(float).values)
        self.betas = np.sort(df_d['Beta'].unique())

        # 3번째 차원인 쇄교자속(Flux Linkage) 축 정의 (데이터 내 범위 또는 고유값 기준 추출)
        if 'Flux' in df_d.columns:
            self.fluxes = np.sort(df_d['Flux'].unique())
        else:
            # 기본 쇄교자속 대표 성분 격자 생성 (예시: 0Wb에서 0.5Wb 범위를 격자화)
            self.fluxes = np.linspace(0.0, 0.5, 10)

        num_current = len(self.currents)
        num_beta = len(self.betas)
        num_flux = len(self.fluxes)

        # 3D 자속 맵 데이터 배열 구축 [Current, Beta, Flux]
        # 실제 3차원 곡면 스윕 데이터 형태에 맞게 정렬하여 할당합니다.
        if 'Current' in df_d.columns and 'Beta' in df_d.columns and 'Flux' in df_d.columns:
            df_d_sorted = df_d.sort_values(by=['Current', 'Beta', 'Flux'])
            df_q_sorted = df_q.sort_values(by=['Current', 'Beta', 'Flux'])
            df_0_sorted = df_0.sort_values(by=['Current', 'Beta', 'Flux'])

            lambda_d_mat = df_d_sorted['Flux_d'].values.reshape(num_current, num_beta, num_flux) * 1e-3
            lambda_q_mat = df_q_sorted['Flux_q'].values.reshape(num_current, num_beta, num_flux) * 1e-3
            lambda_0_mat = df_0_sorted['Flux_0'].values.reshape(num_current, num_beta, num_flux) * 1e-3
        else:
            # 기존 2D 파일 호환 데코레이터 구조 생성
            lambda_d_2d = df_d.iloc[:, 1:].values * 1e-3
            # 전치 연산을 통해 [Current, Beta] 형상 확보 후 3차원으로 확장
            if lambda_d_2d.shape == (num_beta, num_current):
                lambda_d_2d = lambda_d_2d.T
                lambda_q_2d = df_q.iloc[:, 1:].values.T * 1e-3
                lambda_0_2d = df_0.iloc[:, 1:].values.T * 1e-3
            else:
                lambda_q_2d = df_q.iloc[:, 1:].values * 1e-3
                lambda_0_2d = df_0.iloc[:, 1:].values * 1e-3

            lambda_d_mat = np.repeat(lambda_d_2d[:, :, np.newaxis], num_flux, axis=2)
            lambda_q_mat = np.repeat(lambda_q_2d[:, :, np.newaxis], num_flux, axis=2)
            lambda_0_mat = np.repeat(lambda_0_2d[:, :, np.newaxis], num_flux, axis=2)

        # 3차원 RegularGridInterpolator 생성 [Current, Beta, Flux]
        self.interp_d = RegularGridInterpolator((self.currents, self.betas, self.fluxes), lambda_d_mat, method='linear', bounds_error=False, fill_value=None)
        self.interp_q = RegularGridInterpolator((self.currents, self.betas, self.fluxes), lambda_q_mat, method='linear', bounds_error=False, fill_value=None)
        self.interp_0 = RegularGridInterpolator((self.currents, self.betas, self.fluxes), lambda_0_mat, method='linear', bounds_error=False, fill_value=None)

    def get_flux(self, Idq0):
        """전류 크기(I), 전류각(beta), 0축 전류(i0)에 따른 d, q, 0축 자속쇄교수 반환 (3D 보간)"""
        current_mag = np.sqrt(Idq0[0]**2 + Idq0[1]**2)
        if current_mag < 1e-6:
            beta_deg = 0.0
        else:
            beta_deg = np.degrees(np.arctan2(Idq0[1], Idq0[0]))

        # 세 번째 입력값으로 활용할 대표 자속 크기값 산출 (전체 자속의 노름 활용)
        flux_est = np.sqrt(Idq0[0]**2 + Idq0[1]**2 + Idq0[2]**2) * 1e-2 # 크기 가중 계수 반영

        # 입력값 격자 범위 내 클램핑
        I_clamped = np.clip(current_mag, self.currents[0], self.currents[-1])
        beta_clamped = np.clip(beta_deg, self.betas[0], self.betas[-1])
        flux_clamped = np.clip(flux_est, self.fluxes[0], self.fluxes[-1])

        # 3차원 보간 좌표 생성 [Current, Beta, Flux]
        pts = np.array([[I_clamped, beta_clamped, flux_clamped]])
        ld = self.interp_d(pts)[0]
        lq = self.interp_q(pts)[0]
        l0 = self.interp_0(pts)[0]

        return np.array([ld, lq, l0])

    def calculate_torque(self, lamda, Idq0):
        """전자기 토크 계산 [Nm]"""
        torque = 1.5 * self.p * np.cross(lamda, Idq0)
        return torque

    def abc_to_dq(self,i_abc, b):
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
        abc = clac2.T @ park_3x3.T @ dq0
        return abc

    def compute_incremental_inductance2(self, Idq0, delta=1e-3):
        """
        dq0 좌표계에서 철심 포화와 상호 간섭(Cross-saturation)을 고려한 3x3 증분 인덕턴스 행렬(L_inc) 계산
        """
        L_inc = np.zeros((3, 3))

        for j in range(3):
            I_perturbed_p = Idq0.copy()
            I_perturbed_m = Idq0.copy()
            I_perturbed_p[j] += delta
            I_perturbed_m[j] -= delta

            # d(lambda_i) / d(i_j) 수치미분
            L_inc[:, j] = (self.get_flux(I_perturbed_p) - self.get_flux(I_perturbed_m)) / (2.0 * delta)

        L_inc_reg = L_inc + self.L_epsilon * np.eye(3)
        return L_inc_reg

    def compute_incremental_inductance(self, Idq0, delta=1e-3):
        """
        dq0 좌표계에서 철심 포화와 상호 간섭을 고려한 3x3 증분 인덕턴스 행렬(L_inc) 계산
        """
        # d축 변위
        ld_p, lq_p, l0_p = self.get_flux([Idq0[0] + delta, Idq0[1], Idq0[2]])
        ld_m, lq_m, l0_m = self.get_flux([Idq0[0] - delta, Idq0[1], Idq0[2]])
        L_dd = (ld_p - ld_m) / (2.0 * delta)
        L_qd = (lq_p - lq_m) / (2.0 * delta)
        L_0d = (l0_p - l0_m) / (2.0 * delta)

        # q축 변위
        ld_p, lq_p, l0_p = self.get_flux([Idq0[0], Idq0[1] + delta, Idq0[2]])
        ld_m, lq_m, l0_m = self.get_flux([Idq0[0], Idq0[1] - delta, Idq0[2]])
        L_dq = (ld_p - ld_m) / (2.0 * delta)
        L_qq = (lq_p - lq_m) / (2.0 * delta)
        L_0q = (l0_p - l0_m) / (2.0 * delta)

        # 0축 변위
        ld_p, lq_p, l0_p = self.get_flux([Idq0[0], Idq0[1], Idq0[2] + delta])
        ld_m, lq_m, l0_m = self.get_flux([Idq0[0], Idq0[1], Idq0[2] - delta])
        L_d0 = (ld_p - ld_m) / (2.0 * delta)
        L_q0 = (lq_p - lq_m) / (2.0 * delta)
        L_00 = (l0_p - l0_m) / (2.0 * delta)

        L_inc = np.array([[L_dd, L_dq, L_d0],
                          [L_qd, L_qq, L_q0],
                          [L_0d, L_0q, L_00]]) + np.eye(3,3)*self.L_epsilon
        return L_inc

    def simulate_step(self, Vabc, Iabc, wm, theta_m, dt, load_torque=0.0):
        theta_e = np.mod(self.p * theta_m, 2.0 * np.pi)
        Vdq0 = self.abc_to_dq(Vabc, theta_e)
        Idq0 = self.abc_to_dq(Iabc, theta_e)

        omega_e = self.p * wm
        L_inc = self.compute_incremental_inductance2(Idq0)

        omega = np.array([0, 0, omega_e])
        lamda = self.get_flux(Idq0)
        speed_emf = np.cross(omega, lamda)

        try:
            L_inv = np.linalg.inv(L_inc)
            dIdq0_dt = L_inv @ (Vdq0 - self.Rs * Idq0 - speed_emf)
        except np.linalg.LinAlgError:
            dIdq0_dt = np.linalg.pinv(L_inc) @ (Vdq0 - self.Rs * Idq0 - speed_emf)

        Idq0 = Idq0 + dIdq0_dt * dt
        lamda = self.get_flux(Idq0)
        Te_vector = 1.5 * self.p * np.cross(lamda, Idq0)
        Te_z = Te_vector[2]
        dwm_dt = (Te_z - load_torque - self.B * wm) / self.J

        wm = wm + dwm_dt * dt
        theta_m = theta_m + wm * dt
        theta_m = np.mod(theta_m, 2.0 * np.pi)
        theta_e = np.mod(self.p * theta_m, 2.0 * np.pi)
        Iabc = self.dq_to_abc(Idq0, theta_e)
        Vdq0 = self.abc_to_dq(Vabc, theta_e)
        return Iabc, Te_z, wm, theta_m, Vdq0, Idq0, speed_emf


# --- 강제 속도 인가 및 BEMF 측정 구동 예제 ---
if __name__ == "__main__":
    #csv_d_path = 'FEMM_Lambda_d_matrix.csv'
    #csv_q_path = 'FEMM_Lambda_q_matrix.csv'
    #csv_0_path = 'FEMM_Lambda_0_matrix.csv'
    csv_d_path = 'Rotate_FEMM_Lambda_d_matrix.csv'
    csv_q_path = 'Rotate_FEMM_Lambda_q_matrix.csv'
    csv_0_path = 'Rotate_FEMM_Lambda_0_matrix.csv'

    sim = IPMSMSimulator(csv_d_path, csv_q_path, csv_0_path, pole_pairs=4, rs=0.0618)


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
    plt.subplot(2, 1, 1)
    plt.plot(history_time, history_ea, 'r-', label='ea', alpha=0.9)
    plt.plot(history_time, history_eb, 'g-', label='eb', alpha=0.9)
    plt.plot(history_time, history_ec, 'b-', label='ec', alpha=0.9)
    plt.title(f'Three-Phase Back-EMF at {target_rpm} RPM (ea, eb, ec) :Max peak :{max(history_ea):.2f} V')
    plt.ylabel('Back-EMF [V]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    """ # 2. dq축 역기전력 그래프
    plt.subplot(3, 1, 2)
    plt.plot(history_time, history_Te, 'm-', label='Te', lw=1.5)
    #plt.plot(history_time, history_eq, 'c-', label='eq (omega_e * ld)', lw=1.5)
    plt.title('d-q Axis Back-EMF Components (Torque and EMF)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Torque [Nm]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7) """

    # 3. dq축 전류 그래프
    plt.subplot(2, 1, 2)
    plt.plot(history_time, history_ed, 'm-', label='emf_d', lw=1.5)
    plt.plot(history_time, history_eq, 'c-', label='emf_q', lw=1.5)
    plt.plot(history_time, history_e0, 'y-', label='emf_0', lw=1.5)
    plt.title('speed_emf Components (omega_e * lambda)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Back-EMF [V]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 격자 데이터 생성 (전류 크기와 전류각 그리드 추출)
current_grid, beta_grid = np.meshgrid(sim.currents, sim.betas)

# 대표 자속(Flux)은 중간값으로 고정하여 3차원 곡면 데이터 계산
fixed_flux = sim.fluxes[len(sim.fluxes) // 2]

lambda_d_surf = np.zeros_like(current_grid)
lambda_q_surf = np.zeros_like(current_grid)
lambda_0_surf = np.zeros_like(current_grid)

for i in range(current_grid.shape[0]):
    for j in range(current_grid.shape[1]):
        # Current와 Beta 성분 기반으로 역산한 Idq0 가상 전동기 벡터 입력
        I_val = current_grid[i, j]
        beta_rad = np.radians(beta_grid[i, j])

        # 임의의 dq0 가상 전류점 매핑
        id_temp = I_val * np.cos(beta_rad)
        iq_temp = I_val * np.sin(beta_rad)
        i0_temp = 0.0

        # 보간기를 통한 자속 검출
        flux_vector = sim.get_flux([id_temp, iq_temp, i0_temp])
        lambda_d_surf[i, j] = flux_vector[0]
        lambda_q_surf[i, j] = flux_vector[1]
        lambda_0_surf[i, j] = flux_vector[2]

# 3차원 그래프 시각화
fig = plt.figure(figsize=(20, 6))

# 1. d축 자속쇄교수 곡면
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(current_grid, beta_grid, lambda_d_surf, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.set_title('d-axis Flux Linkage ($\\lambda_d$) [Wb]', fontsize=12)
ax1.set_xlabel('Current [A]')
ax1.set_ylabel('Beta [deg]')
ax1.set_zlabel('Flux [Wb]')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

# 2. q축 자속쇄교수 곡면
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(current_grid, beta_grid, lambda_q_surf, cmap='plasma', edgecolor='none', alpha=0.9)
ax2.set_title('q-axis Flux Linkage ($\\lambda_q$) [Wb]', fontsize=12)
ax2.set_xlabel('Current [A]')
ax2.set_ylabel('Beta [deg]')
ax2.set_zlabel('Flux [Wb]')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

# 3. 0축 자속쇄교수 곡면
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(current_grid, beta_grid, lambda_0_surf, cmap='inferno', edgecolor='none', alpha=0.9)
ax3.set_title('0-axis Flux Linkage ($\\lambda_0$) [Wb]', fontsize=12)
ax3.set_xlabel('Current [A]')
ax3.set_ylabel('Beta [deg]')
ax3.set_zlabel('Flux [Wb]')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()