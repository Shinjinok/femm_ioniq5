import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import scipy.io as sio  # MATLAB .mat 파일 저장을 위한 라이브러리
# 1. 파일 데이터 로드 (텍스트 데이터를 복사해 파일로 저장했다고 가정)
# 여기서는 파일 1을 d축 자속, 파일 2를 q축 자속 데이터로 가정합니다.
df_flux1 = pd.read_csv('FEMM_Lambda_d_matrix.csv', index_col=0) # 행 인덱스를 Beta로 지정
df_flux2 = pd.read_csv('FEMM_Lambda_q_matrix.csv', index_col=0)

# 전류 크기(Columns)와 Beta각(Index) 추출
current_amplitudes = np.array(df_flux1.columns.astype(float))
beta_angles = np.array(df_flux1.index.astype(float))

# 2. 불규칙한 (id, iq) 데이터 포인트 수집
points_id = []
points_iq = []
values_lambda_d = []
values_lambda_q = []

for beta in beta_angles:
    for im in current_amplitudes:
        # 일반적인 PMSM의 dq 매핑 (제어 정의에 따라 sin, cos 위치나 부호가 달라질 수 있습니다)
        # 예시: Beta=0일 때 q축 정렬, 전류가 음수 d축으로 투입되는 일반 공식 기준
        beta_rad = np.radians(beta)
        id_val = im * np.cos(beta_rad)
        iq_val = im * np.sin(beta_rad)
        
        points_id.append(id_val)
        points_iq.append(iq_val)
        values_lambda_d.append(df_flux1.loc[beta, str(im)])
        values_lambda_q.append(df_flux2.loc[beta, str(im)])

# 3. 새로운 직교 격자 (Regular Grid) 정의 (Simulink/파이썬 모델용 입력 Grid)
# 변환된 id, iq의 최솟값과 최댓값을 기준으로 균일한 격자를 만듭니다.
id_grid = np.linspace(min(points_id), max(points_id), 50)
iq_grid = np.linspace(min(points_iq), max(points_iq), 50)
grid_ID, grid_IQ = np.meshgrid(id_grid, iq_grid)

# 4. griddata를 이용해 정형 격자 위로 자속 데이터 보간 (Linear 또는 Cubic)
lut_lambda_d = griddata((points_id, points_iq), values_lambda_d, (grid_ID, grid_IQ), method='linear')
lut_lambda_q = griddata((points_id, points_iq), values_lambda_q, (grid_ID, grid_IQ), method='linear')

# 결과 확인 (lut_lambda_d와 lut_lambda_q는 이제 50x50 사이즈의 완성된 2D LUT Matrix입니다)
print("d축 자속 LUT 생성 완료. 크기:", lut_lambda_d.shape)
print("q축 자속 LUT 생성 완료. 크기:", lut_lambda_q.shape)
print("d/q축 자속 직교 격자 LUT 변환 완료.")

# ==========================================
# 4. 자속 룩업테이블(LUT) 데이터 저장 코드 추가
# ==========================================


# --- [방법 B] 범용적으로 확인 가능한 2D CSV 파일로 저장 ---
# 보기 좋게 가로축을 id_grid, 세로축을 iq_grid로 배치하여 저장합니다.
df_lut_d = pd.DataFrame(lut_lambda_d, index=iq_grid, columns=id_grid)
df_lut_q = pd.DataFrame(lut_lambda_q, index=iq_grid, columns=id_grid)

df_lut_d.to_csv('lut_lambda_d_output.csv')
df_lut_q.to_csv('lut_lambda_q_output.csv')

# 격자 축(Breakpoints) 정보를 따로 텍스트 파일로 보관
np.savetxt('id_grid_axis.csv', id_grid, delimiter=',')
np.savetxt('iq_grid_axis.csv', iq_grid, delimiter=',')
print("-> [성공] CSV 파일 저장 완료: lut_lambda_d_output.csv, lut_lambda_q_output.csv")

print("-> [성공] 파이썬 이진 파일(.npy) 저장 완료.")
