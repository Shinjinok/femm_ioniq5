import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_inductance_summary(csv_filename="ioniq5-13.FEM_all_currents_inductance_summary.csv"):
    # CSV 파일 읽기
    df = pd.read_csv(csv_filename)
    
    currents = df['Current_A']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. 상별 중심값(Center) 그래프
    ax1.plot(currents, df['Laa_Center_uH'], marker='o', label='Laa Center', color='blue')
    ax1.plot(currents, df['Lab_Center_uH'], marker='s', label='Lab Center', color='green')
    ax1.plot(currents, df['Lac_Center_uH'], marker='^', label='Lac Center', color='orange')
    ax1.set_title('전류에 따른 상별 인덕턴스 중심값(Center) 변화', fontsize=12, fontweight='bold')
    ax1.set_ylabel('중심 인덕턴스 [μH]', fontsize=10)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # 2. 상별 진폭(Amplitude) 그래프
    ax2.plot(currents, df['Laa_Amplitude_uH'], marker='o', label='Laa Amplitude', color='blue')
    ax2.plot(currents, df['Lab_Amplitude_uH'], marker='s', label='Lab Amplitude', color='green')
    ax2.plot(currents, df['Lac_Amplitude_uH'], marker='^', label='Lac Amplitude', color='orange')
    ax2.set_title('전류에 따른 상별 인덕턴스 진폭(Amplitude) 변화', fontsize=12, fontweight='bold')
    ax2.set_xlabel('전류 [A]', fontsize=10)
    ax2.set_ylabel('인덕턴스 진폭 [μH]', fontsize=10)
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    output_img = 'inductance_summary_plot.png'
    plt.savefig(output_img, dpi=300)
    print(f"[요약 플롯 저장 완료] {output_img}")
    plt.show()

if __name__ == '__main__':
    plot_inductance_summary()