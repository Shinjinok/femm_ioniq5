import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows 환경 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_actual_vs_fitted_mWb(csv_filename='flux_linkage_data_34A.csv'):
    # 1. CSV 파일 읽기
    try:
        df = pd.read_csv(csv_filename)
        print(f"'{csv_filename}' 파일을 성공적으로 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: '{csv_filename}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    theta_deg = df['Theta_Elec_deg'].values
    theta_rad = df['Theta_Elec_rad'].values
    
    # 데이터를 처음부터 mWb 단위로 변환 (Wb -> mWb)
    lam_a = df['Lambda_a'].values * 1e3
    lam_b = df['Lambda_b'].values * 1e3
    lam_c = df['Lambda_c'].values * 1e3

    # 2. 푸리에 급수 모델 피팅 함수 (mWb 기준, 돌극성 모터 특성에 맞춘 2고조파 중심 피팅)
    def fit_fourier_mWb(theta, y, harmonics=[2, 4]):
        # DC 오프셋 (중심값) [mWb]
        a0 = np.mean(y)
        fit_y = np.full_like(y, a0)
        
        components = [{'term': 'DC', 'value': a0}]
        
        for k in harmonics:
            cos_k = np.cos(k * theta)
            sin_k = np.sin(k * theta)
            
            ck = 2 * np.mean(y * cos_k)
            sk = 2 * np.mean(y * sin_k)
            
            amp = np.sqrt(ck**2 + sk**2) # [mWb]
            phase = np.arctan2(-sk, ck)
            
            fit_y += amp * np.cos(k * theta - phase)
            components.append({'harmonic': k, 'amp': amp, 'phase': phase})
            
        return fit_y, components

    # 각 상별 수식 피팅 수행
    fit_a, comp_a = fit_fourier_mWb(theta_rad, lam_a, harmonics=[2, 4])
    fit_b, comp_b = fit_fourier_mWb(theta_rad, lam_b, harmonics=[2, 4])
    fit_c, comp_c = fit_fourier_mWb(theta_rad, lam_c, harmonics=[2, 4])

    print("\n================ 상별 유도 수식 요약 (단위: mWb) ================")
    print(f"[A상 수식]")
    print(f"  Lambda_a(theta) = {comp_a[0]['value']:.3f} + {comp_a[1]['amp']:.3f}*cos(2*theta - {np.degrees(comp_a[1]['phase']):.2f}°) [mWb]")
    
    print(f"\n[B상 수식]")
    print(f"  Lambda_b(theta) = {comp_b[0]['value']:.3f} + {comp_b[1]['amp']:.3f}*cos(2*theta - {np.degrees(comp_b[1]['phase']):.2f}°) [mWb]")
    
    print(f"\n[C상 수식]")
    print(f"  Lambda_c(theta) = {comp_c[0]['value']:.3f} + {comp_c[1]['amp']:.3f}*cos(2*theta - {np.degrees(comp_c[1]['phase']):.2f}°) [mWb]")
    print("==================================================================")

    # 3. 실제 데이터와 유도된 수식 곡선 비교 플롯 (mWb 단위)
    plt.figure(figsize=(12, 7))

    # 실제 데이터 (실선/투명도 적용)
    plt.plot(theta_deg, lam_a, label='A상 실제 데이터', color='blue', lw=2, alpha=0.6)
    #plt.plot(theta_deg, lam_b, label='B상 실제 데이터', color='green', lw=2, alpha=0.6)
    plt.plot(theta_deg, lam_c, label='C상 실제 데이터', color='orange', lw=2, alpha=0.6)

    # 유도된 수식 모델 곡선 (점선)
    plt.plot(theta_deg, fit_a, label='A상 유도 수식 (Fit)', color='darkblue', lw=1.5, linestyle='--')
    #plt.plot(theta_deg, fit_b, label='B상 유도 수식 (Fit)', color='darkgreen', lw=1.5, linestyle='--')
    plt.plot(theta_deg, fit_c, label='C상 유도 수식 (Fit)', color='darkorange', lw=1.5, linestyle='--')

    plt.title("3상 자속 쇄교수 실제 데이터 vs 유도 수식 모델 비교 (mWb)", fontsize=13, fontweight='bold')
    plt.xlabel("전기각 (Theta_Elec) [deg]", fontsize=11)
    plt.ylabel("자속 쇄교수 [mWb]", fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()

    output_image_path = "flux_linkage_mWb_actual_vs_fitted.png"
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"\n[비교 플롯 저장 완료] 그래프가 '{output_image_path}' 파일로 저장되었습니다.")

if __name__ == '__main__':
    plot_actual_vs_fitted_mWb()