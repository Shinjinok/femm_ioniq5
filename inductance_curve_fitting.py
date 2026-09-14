import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def fit_all_outputs_poly6(csv_filename="ioniq5-13.FEM_all_currents_inductance_summary.csv"):
    # 1. 데이터 로드
    df = pd.read_csv(csv_filename)
    currents = df['Current_A'].values
    
    output_cols = [
        'Laa_Center_uH', 'Laa_Amplitude_uH', 
        'Lab_Center_uH', 'Lab_Amplitude_uH', 
        'Lac_Center_uH', 'Lac_Amplitude_uH'
    ]

    # 6차 다항식 피팅 함수 정의 (y = ax^6 + bx^5 + cx^4 + dx^3 + ex^2 + fx + g)
    def poly6(x, a, b, c, d, e, f, g):
        return a * x**6 + b * x**5 + c * x**4 + d * x**3 + e * x**2 + f * x + g

    # 3x2 서브플롯 생성
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes = axes.flatten()

    x_fit = np.linspace(currents.min(), currents.max(), 300)

    for i, col in enumerate(output_cols):
        y_data = df[col].values
        
        # 6차 곡선 피팅 수행
        popt, _ = curve_fit(poly6, currents, y_data)
        a, b, c, d, e, f, g = popt
        y_fit = poly6(x_fit, *popt)
        
        print(f"[{col} 6차 다항식 피팅 방정식]")
        print(f"y = {a:.6e}*x^6 + {b:.6e}*x^5 + {c:.6e}*x^4 + {d:.6e}*x^3 + {e:.6e}*x^2 + {f:.6e}*x + {g:.6e}\n")

        # 그래프 시각화
        axes[i].scatter(currents, y_data, color='blue', label='FEM 데이터', zorder=5)
        axes[i].plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2, label='6차 다항식 피팅')
        
        axes[i].set_title(f'입력 전류 vs {col} (6차 피팅)', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('입력 전류 [A]', fontsize=10)
        axes[i].set_ylabel('인덕턴스 [μH]', fontsize=10)
        axes[i].grid(True, which='both', linestyle='--', alpha=0.6)
        axes[i].legend(fontsize=9)

    plt.tight_layout()
    output_image = 'all_outputs_curve_fit_poly6.png'
    plt.savefig(output_image, dpi=300)
    print(f"[6차 피팅 그래프 저장 완료] {output_image}")
    plt.show()

if __name__ == '__main__':
    fit_all_outputs_poly6()