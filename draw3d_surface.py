import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 한글 폰트 설정 및 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# d, q, 0, Torque 파일과 그래프 제목, 컬러맵 정보를 담은 딕셔너리
target_plots = {
    'Lambda_d': ('Rotate_FEMM_Lambda_d_matrix3.csv', 'd-axis Flux Linkage (Lambda_d)', 'viridis'),
    'Lambda_q': ('Rotate_FEMM_Lambda_q_matrix3.csv', 'q-axis Flux Linkage (Lambda_q)', 'plasma'),
    'Lambda_0': ('Rotate_FEMM_Lambda_0_matrix3.csv', 'Zero-sequence Flux (Lambda_0)', 'cividis'),
    'Torque': ('Rotate_FEMM_Torque_matrix2.csv', 'Torque', 'inferno')
}

# 각 파일별로 반복문을 돌며 3차원 그래프 생성
for key, (file_name, title_desc, cmap_name) in target_plots.items():
    # 1. CSV 파일 불러오기
    df = pd.read_csv(file_name, index_col=0)

    # 인덱스와 컬럼 이름을 숫자형(float)으로 변환
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    # 2. 3차원 좌표계용 X, Y 그리드 생성
    X, Y = np.meshgrid(df.columns, df.index)
    Z = df.values

    # 3. 3차원 서피스 그래프 플롯팅
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 서피스 그래프 생성
    surf = ax.plot_surface(X, Y, Z, cmap=cmap_name, edgecolor='none', alpha=0.9)

    # 4. 축 라벨 및 제목 설정
    ax.set_title(f'3D Surface Plot: {title_desc}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('I_dq (Current [A])', fontsize=11, labelpad=10)
    ax.set_ylabel('Beta (Angle [°])', fontsize=11, labelpad=10)
    ax.set_zlabel('Value', fontsize=11, labelpad=10)

    # 5. 컬러바 추가
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, label='Magnitude')

    # 보기 좋은 각도로 시점(뷰) 조절
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    
    # 화면에 띄우거나 파일로 저장하려면 아래 주석 해제
    # plt.savefig(f'{key}_3d_plot.png', dpi=300)
    
    plt.show()