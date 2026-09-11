import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 파일 불러오기 (최신 8극 결과 파일 지정)
csv_filename = "a_phase_600A_8pole_inductance.csv"  # 혹은 기존 파일명
df = pd.read_csv(csv_filename)

# 2. X축으로 사용할 각도 열 이름 자동 감지 (Theta_Elec_deg 또는 Theta_deg)
x_col = 'Theta_Elec_deg' if 'Theta_Elec_deg' in df.columns else ('Theta_deg' if 'Theta_deg' in df.columns else df.columns[0])
print(f"플롯에 사용되는 X축 열: {x_col}")

# 3. 인덕턴스 열 이름 호환성 처리 (Laa 또는 La 등)
col_a = 'Laa' if 'Laa' in df.columns else ('La' if 'La' in df.columns else None)
col_b = 'Lab' if 'Lab' in df.columns else ('Lb' if 'Lb' in df.columns else None)
col_c = 'Lac' if 'Lac' in df.columns else ('Lc' if 'Lc' in df.columns else None)

# 4. 2단 서브플롯 생성
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# --- [상단] Phase Flux Linkages 플롯 ---
if 'Lambda_a' in df.columns:
    ax1.plot(df[x_col], df['Lambda_a'] * 1000, marker='o', label='Flux Linkage $\\lambda_a$', linewidth=2, color='#1f77b4')
    ax1.plot(df[x_col], df['Lambda_b'] * 1000, marker='s', label='Flux Linkage $\\lambda_b$', linewidth=2, color='#ff7f0e')
    ax1.plot(df[x_col], df['Lambda_c'] * 1000, marker='^', label='Flux Linkage $\\lambda_c$', linewidth=2, color='#2ca02c')
    ax1.set_title('Phase Flux Linkages under A-Phase Excitation vs. Electrical Angle', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Flux Linkage [mWb-turns]', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10, loc='best')

# --- [하단] Inductances 플롯 ---
if col_a and col_b and col_c:
    ax2.plot(df[x_col], df[col_a] * 1e3, marker='o', label=f'Inductance ({col_a})', linewidth=2, color='#1f77b4')
    ax2.plot(df[x_col], df[col_b] * 1e3, marker='s', label=f'Inductance ({col_b})', linewidth=2, color='#ff7f0e')
    ax2.plot(df[x_col], df[col_c] * 1e3, marker='^', label=f'Inductance ({col_c})', linewidth=2, color='#2ca02c')

ax2.set_title('Phase Inductances vs. Electrical Angle (Pure Saliency, 8-Pole)', fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel('Electrical Angle [deg]', fontsize=11)
ax2.set_ylabel('Inductance [mH]', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=10, loc='best')

# 5. 레이아웃 정리 및 이미지 저장
plt.tight_layout()
output_plot_filename = "a_phase_600A_8pole_inductance.png"
plt.savefig(output_plot_filename, dpi=300)
print(f"'{output_plot_filename}' 플롯 이미지 저장 완료!")
plt.show()