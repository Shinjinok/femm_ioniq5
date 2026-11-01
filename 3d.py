import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D


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

# 1. 데이터 로드 및 격자 설정
csv_d_path = 'Rotate_FEMM_Lambda_d_matrix3.csv'
df_d = pd.read_csv(csv_d_path, index_col=0)

betas = df_d.index.values.astype(float)
idqs = df_d.columns.values.astype(float)
Z = df_d.values

# 2. 지정된 지점 (-51도, 251A) 보간 계산
# RegularGridInterpolator는 (betas, idqs) 순서로 생성
interp_d = RegularGridInterpolator((betas, idqs), Z, method='cubic', bounds_error=False, fill_value=None)

query_beta = -0
query_current = 340 
# 인터폴레이터 입력 형식: [[Beta, Current]]
query_z = interp_d(np.array([[query_beta, query_current]]))[0]

print(f"보간된 쇄교자속 값 (Beta={query_beta} deg, Current={query_current} A) => lambda_d: {query_z:.4f} Wb")

# 3. 3차원 그래프 및 타겟 포인트 시각화
IDQ, BETA = np.meshgrid(idqs, betas)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 표면도 그리기
surf = ax.plot_surface(IDQ, BETA, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# 타겟 포인트를 빨간색 큰 점으로 표시

ax.scatter(query_current, query_beta, query_z, color='red', s=150, edgecolor='black', depthshade=False, label=f'Target Point (-51°, 251A): {query_z:.2f} Wb')

ax.set_title('3D Flux Linkage Matrix ($\lambda_d$) with Target Point', fontsize=14, pad=20)
ax.set_xlabel('Current / $I_{dq}$ [A]', fontsize=11, labelpad=10)
ax.set_ylabel('Beta [deg]', fontsize=11, labelpad=10)
ax.set_zlabel('Flux Linkage [Wb]', fontsize=11, labelpad=10)

# 범례 및 컬러바 추가
ax.legend(loc='upper left', fontsize=11)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

plt.tight_layout()
plt.show()