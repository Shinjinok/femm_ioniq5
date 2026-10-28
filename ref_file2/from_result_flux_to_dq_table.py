import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    input_file = 'results_flux.csv'
    
    # 1. results_flux.csv 파일 읽기
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return

    # 전기각 및 각 상의 쇄교 자속 데이터 추출
    theta_e_deg = df['Electrical_Angle'].values
    theta_e_rad = np.radians(theta_e_deg)
    
    fa = df['Phase_A'].values
    fb = df['Phase_B'].values
    fc = df['Phase_C'].values

    # 2. 파크 변환 (Park Transformation - Amplitude-invariant) 수행
    cos_th = np.cos(theta_e_rad)
    sin_th = np.sin(theta_e_rad)
    cos_th_120 = np.cos(theta_e_rad - 2 * np.pi / 3)
    sin_th_120 = np.sin(theta_e_rad - 2 * np.pi / 3)
    cos_th_240 = np.cos(theta_e_rad + 2 * np.pi / 3)
    sin_th_240 = np.sin(theta_e_rad + 2 * np.pi / 3)

    lambda_d = (2.0 / 3.0) * (fa * cos_th + fb * cos_th_120 + fc * cos_th_240)
    lambda_q = -(2.0 / 3.0) * (fa * sin_th + fb * sin_th_120 + fc * sin_th_240)
    lambda_0 = (1.0 / 3.0) * (fa + fb + fc)

    # 데이터프레임에 dq0 성분 컬럼 추가
    df['Flux_d'] = lambda_d*1000
    df['Flux_q'] = lambda_q*1000
    df['Flux_0'] = lambda_0*1000

    # 3. 변환 결과 CSV 파일로 저장
    output_csv = 'results_flux_dq0.csv'
    df.to_csv(output_csv, index=False)
    print(f"[저장 완료] {output_csv}")

    # 4. Matplotlib 시각화 및 이미지 저장
    plt.figure(figsize=(10, 6))
    plt.plot(df['Electrical_Angle'], df['Flux_d'], label='Flux d (lambda_d)', linewidth=2)
    plt.plot(df['Electrical_Angle'], df['Flux_q'], label='Flux q (lambda_q)', linewidth=2)
    plt.plot(df['Electrical_Angle'], df['Flux_0'], label='Flux 0 (lambda_0)', linestyle='--', linewidth=1.5)
    
    plt.title('dq0 Flux Linkage Components', fontsize=12)
    plt.xlabel('Electrical Angle [deg]', fontsize=10)
    plt.ylabel('Flux Linkage [Wb-turns]', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()

    output_plot = 'flux_dq0_plot.png'
    plt.savefig(output_plot, dpi=300)
    print(f"[플롯 저장 완료] {output_plot}")
    plt.show()

if __name__ == '__main__':
    main()