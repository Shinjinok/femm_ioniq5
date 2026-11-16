import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. lamda_q 데이터 불러오기 및 텐서 변환
df_q = pd.read_csv('LUT_lamda_q.csv')
ids = np.sort(df_q['Id_A'].unique())
iqs = np.sort(df_q['Iq_A'].unique())
thetas = np.sort(df_q['theta_e_deg'].unique())

matrix_q = df_q.pivot_table(
    index=['Id_A', 'Iq_A'], columns='theta_e_deg', values='lamda_q_Wb'
)
print("\n=== 2. 피벗 테이블 변환 결과 ===")
print(matrix_q)
lamda_q_3d = matrix_q.reshape(len(ids), len(iqs), len(thetas))

print(lamda_q_3d.shape)
# 2. 3차원 그리드 좌표 생성 (X: Id, Y: Iq, Z: Theta)
ID, IQ, THETA = np.meshgrid(ids, iqs, thetas, indexing='ij')

# 3. 시각화 가독성을 위해 theta 방향으로 샘플링 (예: 5간격)
step = 5
Id_flat = ID[:, :, ::step].flatten()
Iq_flat = IQ[:, :, ::step].flatten()
Theta_flat = THETA[:, :, ::step].flatten()
Lamda_q_flat = lamda_q_3d[:, :, ::step].flatten()

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(projection='3d')

# 4. 3차원 산점도 생성 (Color는 lamda_q_Wb 값 기준)
scatter = ax.scatter(
    Id_flat,
    Iq_flat,
    Theta_flat,
    c=Lamda_q_flat,
    cmap='jet',
    s=12,
    alpha=0.6,
    edgecolors='none',
)

# 5. 축 및 컬러바 설정
ax.set_title(
    '3D Scatter Plot for Lamda_q (X: Id, Y: Iq, Z: Theta)', fontsize=13, pad=15
)
ax.set_xlabel('Id Current [A]', labelpad=10)
ax.set_ylabel('Iq Current [A]', labelpad=10)
ax.set_zlabel('Electrical Angle (theta) [deg]', labelpad=10)

# 컬러바 추가
cbar = fig.colorbar(
    scatter, ax=ax, shrink=0.6, aspect=12, pad=0.1
)
cbar.set_label('Lamda_q [Wb]', fontsize=12)

plt.show()