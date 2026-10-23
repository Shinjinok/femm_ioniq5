import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 한글 폰트 설정 (시스템 환경에 맞게 수정 가능)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
# d축 자속
pe_d = pd.read_csv('폴페어일렉트로닉스d축자속.csv').set_index('Beta')
femm_d = pd.read_csv('femm_sim_lamda_d.csv').set_index('Beta')

# q축 자속 (불필요한 열이 있을 경우 제거)
pe_q = pd.read_csv('폴페어일렉트로닉스q축자속.csv')

pe_q = pe_q.set_index('Beta')
femm_q = pd.read_csv('femm_sim_lamda_q.csv').set_index('Beta')

# 토크 (Torque)
pe_t = pd.read_csv('폴페어일렉트로닉스Torque.csv').set_index('Beta')
femm_t = pd.read_csv('femm_sim_lamda_Torque.csv').set_index('Beta')


# 2. 공통 열(Column) 기준 데이터 정렬 및 추출 함수
fn_align = lambda pe_df, femm_df: (
    pe_df[
        [c for c in pe_df.columns if c in femm_df.columns]
    ].astype(float),
    femm_df[
        [c for c in femm_df.columns if c in pe_df.columns]
    ].astype(float),
)

pe_d_al, femm_d_al = fn_align(pe_d, femm_d)
pe_q_al, femm_q_al = fn_align(pe_q, femm_q)
pe_t_al, femm_t_al = fn_align(pe_t, femm_t)

# 3. 오차(Absolute Error) 계산 및 평균 오차 도출
err_d = np.abs(pe_d_al - femm_d_al)
mean_err_d = err_d.values.mean()

err_q = np.abs(pe_q_al - femm_q_al)
mean_err_q = err_q.values.mean()

err_t = np.abs(pe_t_al - femm_t_al)
mean_err_t = err_t.values.mean()

# 4. 히트맵 시각화 (1행 3열 구성: d축, q축, 토크)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# (1) d축 자속 오차 히트맵
sns.heatmap(
    err_d,
    ax=axes[0],
    cmap='coolwarm',
    annot=False,
    cbar_kws={'label': 'Absolute Error'},
)
axes[0].set_title(
    f'd-Axis Flux Error\n(Mean Error: {mean_err_d:.4f} mWb)',
    fontsize=13,
    fontweight='bold',
)
axes[0].set_xlabel('Current / Angle Index', fontsize=11)
axes[0].set_ylabel('Beta', fontsize=11)

# (2) q축 자속 오차 히트맵
sns.heatmap(
    err_q,
    ax=axes[1],
    cmap='coolwarm',
    annot=False,
    cbar_kws={'label': 'Absolute Error'},
)
axes[1].set_title(
    f'q-Axis Flux Error\n(Mean Error: {mean_err_q:.4f} mWb)',
    fontsize=13,
    fontweight='bold',
)
axes[1].set_xlabel('Current / Angle Index', fontsize=11)
axes[1].set_ylabel('Beta', fontsize=11)

# (3) 토크(Torque) 오차 히트맵
sns.heatmap(
    err_t,
    ax=axes[2],
    cmap='coolwarm',
    annot=False,
    cbar_kws={'label': 'Absolute Error'},
)
axes[2].set_title(
    f'Torque Error\n(Mean Error: {mean_err_t:.4f} Nm)',
    fontsize=13,
    fontweight='bold',
)
axes[2].set_xlabel('Current / Angle Index', fontsize=11)
axes[2].set_ylabel('Beta', fontsize=11)

plt.tight_layout()
plt.show()