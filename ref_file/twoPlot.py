import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 파일 경로 설정
file_100A = "a_phase_100A_8pole_inductance.csv"
file_600A = "a_phase_600A_8pole_inductance.csv"

# 파일 존재 여부 확인
for f in [file_100A, file_600A]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"필수 CSV 파일을 찾을 수 없습니다: {f}")

# 2. 데이터 불러오기
df_100 = pd.read_csv(file_100A)
df_600 = pd.read_csv(file_600A)

# X축 열 이름 확인 (Theta_Elec_deg 또는 Theta_deg)
x_col = 'Theta_Elec_deg' if 'Theta_Elec_deg' in df_100.columns else 'Theta_deg'

# 인덕턴스 열 이름 확인 (Laa, Lab, Lac)
col_aa = 'Laa' if 'Laa' in df_100.columns else 'La'
col_ab = 'Lab' if 'Lab' in df_100.columns else 'Lb'
col_ac = 'Lac' if 'Lac' in df_100.columns else 'Lc'

# 3. 플롯 생성 (2단 서브플롯: 위쪽은 자기 인덕턴스 Laa 비교, 아래쪽은 상호 인덕턴스 Lab, Lac 비교)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# --- [상단] 자기 인덕턴스 (Laa) 비교 ---
ax1.plot(df_100[x_col], df_100[col_aa] * 1e3, marker='o', label='Self Inductance ($L_{aa}$) @ 100A', linewidth=2, color='#1f77b4')
ax1.plot(df_600[x_col], df_600[col_aa] * 1e3, marker='s', linestyle='--', label='Self Inductance ($L_{aa}$) @ 600A', linewidth=2, color='#ff7f0e')

ax1.set_title('Self Inductance ($L_{aa}$) Comparison: 100A vs 600A', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylabel('Inductance [mH]', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10, loc='best')

# --- [하단] 상호 인덕턴스 (Lab, Lac) 비교 ---
ax2.plot(df_100[x_col], df_100[col_ab] * 1e3, marker='o', label='Mutual Inductance ($L_{ab}$) @ 100A', linewidth=2, color='#2ca02c')
ax2.plot(df_600[x_col], df_600[col_ab] * 1e3, marker='s', linestyle='--', label='Mutual Inductance ($L_{ab}$) @ 600A', linewidth=2, color='#d62728')

ax2.plot(df_100[x_col], df_100[col_ac] * 1e3, marker='^', label='Mutual Inductance ($L_{ac}$) @ 100A', linewidth=2, color='#9467bd')
ax2.plot(df_600[x_col], df_600[col_ac] * 1e3, marker='v', linestyle='--', label='Mutual Inductance ($L_{ac}$) @ 600A', linewidth=2, color='#8c564b')

ax2.set_title('Mutual Inductances ($L_{ab}, L_{ac}$) Comparison: 100A vs 600A', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Electrical Angle [deg]', fontsize=11)
ax2.set_ylabel('Inductance [mH]', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=10, loc='best')

# 4. 레이아웃 정리 및 고해상도 이미지 저장
plt.tight_layout()
output_image = "inductance_comparison_100A_600A.png"
plt.savefig(output_image, dpi=300)
print(f"'{output_image}' 비교 플롯 이미지 저장 완료!")
plt.show()