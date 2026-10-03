import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_inductance_maps():
    # 1. 저장된 CSV 파일 불러오기 (인덱스와 컬럼을 원래 숫자로 로드)
    try:
        df_Ld = pd.read_csv("Ld_map.csv", index_col=0)
        df_Lq = pd.read_csv("Lq_map.csv", index_col=0)
    except FileNotFoundError:
        print("오류: Ld_map.csv 또는 Lq_map.csv 파일을 찾을 수 없습니다. 시뮬레이션을 먼저 실행해주세요.")
        return

    # 데이터프레임의 인덱스와 컬럼을 숫자형 배열로 변환
    id_list = df_Ld.index.astype(float).values
    iq_list = df_Ld.columns.astype(float).values
    
    # Meshgrid 생성 (X: iq, Y: id, Z: Inductance)
    IQ, ID = np.meshgrid(iq_list, id_list)
    Ld_values = df_Ld.values
    Lq_values = df_Lq.values

    # 2. 3차원 플롯 그리기
    fig = plt.figure(figsize=(14, 6))

    # --- (1) Ld 3D Surface Plot ---
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(IQ, ID, Ld_values, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title("d-axis Inductance ($L_d$) Map")
    ax1.set_xlabel("I_q (A)")
    ax1.set_ylabel("I_d (A)")
    ax1.set_zlabel("L_d (H)")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # --- (2) Lq 3D Surface Plot ---
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(IQ, ID, Lq_values, cmap='plasma', edgecolor='none', alpha=0.9)
    ax2.set_title("q-axis Inductance ($L_q$) Map")
    ax2.set_xlabel("I_q (A)")
    ax2.set_ylabel("I_d (A)")
    ax2.set_zlabel("L_q (H)")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_3d_inductance_maps()