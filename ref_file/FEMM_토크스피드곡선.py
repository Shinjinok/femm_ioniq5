import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

def generate_speed_torque_curve():
    # 1. CSV 맵 데이터 로드
    try:
        df_torque = pd.read_csv("FEMM_Torque_matrix.csv", index_col=0)
        df_lambda_d = pd.read_csv("FEMM_Lambda_d_matrix.csv", index_col=0)
        df_lambda_q = pd.read_csv("FEMM_Lambda_q_matrix.csv", index_col=0)
    except FileNotFoundError as e:
        print(f"필요한 CSV 파일을 찾을 수 없습니다: {e}")
        return

    betas = df_torque.index.values.astype(float)
    idqs = df_torque.columns.values.astype(float)
    
    # 2. 3차 보간기(Cubic Interpolator) 생성
    interp_torque = RegularGridInterpolator((betas, idqs), df_torque.values, method='cubic', bounds_error=False, fill_value=None)
    interp_lambda_d = RegularGridInterpolator((betas, idqs), df_lambda_d.values, method='cubic', bounds_error=False, fill_value=None)
    interp_lambda_q = RegularGridInterpolator((betas, idqs), df_lambda_q.values, method='cubic', bounds_error=False, fill_value=None)

    # 3. 모터 및 인버터 구동 제원 설정 (700V DC Link 기준)
    P_poles = 8
    p_pair = P_poles / 2
    I_max = 340.0
    V_max = 700.0 / np.sqrt(3)  # SVPWM 기준 상전압 피크 (약 404.15V)
    flux_divisor = 1000.0

    # 4. 속도(RPM) 그리드 설정 (100 RPM 단위)
    rpm_list = np.arange(500, 15001, 100)
    max_torques = []
    powers = []

    beta_min, beta_max = betas[0], betas[-1]
    idq_min, idq_max = idqs[0], idqs[-1]

    print("연속 최적화 기반 스피드-토크 곡선 연산 중 (잠시만 기다려주세요)...")
    
    for rpm in rpm_list:
        omega_m = rpm * (2 * np.pi / 60)
        omega_e = omega_m * p_pair
        
        # 목적 함수: 토크 최대화
        def objective(x):
            beta, idq = x
            t = interp_torque([[beta, idq]])[0]
            return -t

        # 제약 조건 1: 전압 한계 만족
        def constraint_voltage(x):
            beta, idq = x
            ld = interp_lambda_d([[beta, idq]])[0] / flux_divisor
            lq = interp_lambda_q([[beta, idq]])[0] / flux_divisor
            v_mag = omega_e * np.sqrt(ld**2 + lq**2)
            return V_max - v_mag

        # 제약 조건 2: 전류 한계 만족
        def constraint_current(x):
            beta, idq = x
            return I_max - idq

        constraints = [
            {'type': 'ineq', 'fun': constraint_voltage},
            {'type': 'ineq', 'fun': constraint_current}
        ]
        bounds = [(beta_min, beta_max), (idq_min, idq_max)]

        # 다중 시작점 최적화 수행
        best_t = 0.0
        initial_guesses = [
            [135.0, I_max],
            [135.0, 100.0],
            [160.0, I_max],
            [175.0, 50.0]
        ]
        
        for guess in initial_guesses:
            res = minimize(objective, guess, method='SLSQP', bounds=bounds, constraints=constraints)
            if res.success:
                current_t = -res.fun
                if current_t > best_t:
                    best_t = current_t

        max_torques.append(best_t)
        powers.append(best_t * omega_m / 1000.0)

    # 5. 결과 시각화
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # 토크 곡선 (좌측 Y축)
    ax1.set_xlabel('Speed (RPM)', fontsize=12)
    ax1.set_ylabel('Torque (Nm)', color='tab:blue', fontsize=12)
    ax1.plot(rpm_list, max_torques, color='tab:blue', linewidth=2.5, label='Torque')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # X축 / Y축 그리드 설정
    ax1.xaxis.set_major_locator(MultipleLocator(500))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
    plt.xticks(rotation=45)

    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))

    # 출력 곡선 (우측 Y축)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (kW)', color='tab:red', fontsize=12)
    ax2.plot(rpm_list, powers, color='tab:red', linestyle='--', linewidth=2.5, label='Power')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('EV Motor Speed-Torque & Power Curve (Continuous Optimization)', fontsize=14)
    
    ax1.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()

    # --- [자기 자신의 스크립트 이름을 파일명으로 활용하여 플롯 저장] ---
    try:
        script_name = os.path.basename(__file__)
        base_name, _ = os.path.splitext(script_name)
    except NameError:
        # 대화형 환경(Jupyter 등)에서 실행되어 __file__이 없는 경우 예외 처리
        base_name = "script_plot"

    output_filename = f"{base_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"성공적으로 플롯이 저장되었습니다: '{output_filename}'")

    plt.show()

if __name__ == '__main__':
    generate_speed_torque_curve()