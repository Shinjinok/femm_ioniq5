import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 자속 테이블 CSV 파일 읽기
df_d = pd.read_csv('FEMM_Lambda_d_matrix.csv')
df_q = pd.read_csv('FEMM_Lambda_q_matrix.csv')

# 2. 모터 제원 설정
P = 4  # 극쌍수 (Pole Pairs)

# 3. 데이터 추출 (Beta: 행, Is: 열)
beta_deg = df_d['Beta'].values
beta_rad = np.radians(beta_deg)
Is_cols = df_d.columns[1:].astype(float).values

# 4. 쇄교자속 단위 변환 (mWb -> Wb)
# 자속 데이터가 mWb 단위인 경우 Wb로 변환하기 위해 1000으로 나눔
Lambda_d = df_d.iloc[:, 1:].values / 1000.0
Lambda_q = df_q.iloc[:, 1:].values / 1000.0

# 5. 전류 좌표 변환 (id, iq 계산)
# id = Is * cos(beta), iq = Is * sin(beta)
Id_mat = np.outer(np.cos(beta_rad), Is_cols)
Iq_mat = np.outer(np.sin(beta_rad), Is_cols)

# 6. 전자기 토크 계산 (Te = 1.5 * P * (lambda_d * iq - lambda_q * id))
Torque_mat = 1.5 * P * (Lambda_d * Iq_mat - Lambda_q * Id_mat)

# 7. 토크 테이블 DataFrame 생성 및 CSV 저장
df_torque = pd.DataFrame(Torque_mat, index=beta_deg, columns=df_d.columns[1:])
df_torque.index.name = 'Beta'
df_torque.to_csv('Torque_matrix.csv')
print("Torque_matrix.csv 파일 저장이 완료되었습니다.")

# 8. 토크 특성 시각화 (그래프 저장)
plt.figure(figsize=(10, 6))
for col in df_torque.columns:
    plt.plot(df_torque.index, df_torque[col], label=f'Is = {col} A')

plt.title('IPMSM Torque vs. Current Angle (Beta) for various Current Magnitudes')
plt.xlabel('Current Angle Beta [deg]')
plt.ylabel('Electromagnetic Torque [N·m]')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('torque_table_plot.png', dpi=300)
plt.show()