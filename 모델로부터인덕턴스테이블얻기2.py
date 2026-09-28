import numpy as np
import pandas as pd
import FEMM_Ioniq5_ev_model as femm_model

def generate_idq_flux_linkage_table():
    # 1. 조건 설정
    # 베타 각도: 90도 ~ 180도 (5도 간격) -> 세로축 (행)
    beta_list = np.arange(90, 181, 5)
    # 전류 크기(Idq): 0A ~ 340A (34A 간격) -> 가로축 (열)
    idq_list = np.arange(0, 341, 34)
    
    theta_e = 0.0  # 전기각 고정
    data_rows = []

    print("인덕턴스 행렬 변환(클라크·파크) 및 쇄교자속 매핑 연산 중...")

    for beta in beta_list:
        beta_rad = np.radians(beta)
        
        # 2. Park 변환 행렬 (T) 및 역변환 행렬 (T_inv) 정의
        # 진폭 불변(Amplitude-invariant) 기준 Park 변환 행렬
        cos_th = np.cos(beta_rad)
        sin_th = np.sin(beta_rad)
        cos_th_120 = np.cos(beta_rad - 2.0 * np.pi / 3.0)
        sin_th_120 = np.sin(beta_rad - 2.0 * np.pi / 3.0)
        cos_th_p120 = np.cos(beta_rad + 2.0 * np.pi / 3.0)
        sin_th_p120 = np.sin(beta_rad + 2.0 * np.pi / 3.0)

        # T 행렬 (3x3 또는 2x3 형태 중 3상 -> 2상 변환용 클라크·파크 통합 행렬)
        # 여기서는 3상 쇄교자속/인덕턴스를 dq로 변환하기 위해 표준 Park 변환 행렬 구성
        T = (2.0 / 3.0) * np.array([
            [cos_th, cos_th_120, cos_th_p120],
            [-sin_th, -sin_th_120, -sin_th_p120],
            [0.5, 0.5, 0.5] # 영상분(0축) 성분을 포함한 3x3 확장 행렬
        ])
        
        # 역변환 행렬 (T_inv)
        T_inv = np.array([
            [cos_th, -sin_th, 1.0],
            [cos_th_120, -sin_th_120, 1.0],
            [cos_th_p120, -sin_th_p120, 1.0]
        ])

        for idq in idq_list:
            # 3) FEMM 모델로부터 3x3 인덕턴스 행렬 획득
            Ls = femm_model.get_inductance_matrix(idq, theta_e)
            
            # 4) [핵심] 인덕턴스 행렬에 클라크·파크 변환 양쪽에서 곱하기 (닮음 변환: L_dq = T * L_s * T_inv)
            L_dq_matrix = np.dot(np.dot(T, Ls), T_inv)
            
            # d축, q축 인덕턴스 추출 (행렬의 대각 성분)
            L_d = L_dq_matrix[0, 0]
            L_q = L_dq_matrix[1, 1]
            
            # 5) 전류와 영구자석 자속을 이용한 d-q축 쇄교자속 계산
            # (가정: id = idq, iq = 0 인 조건 또는 설정된 전류 분배 방식에 따름)
            # 기존 코드의 전류 및 자속 벡터 연산 방식을 반영
            i_d = idq*np.cos(beta_rad)  # d축 전류
            i_q = idq*np.sin(beta_rad)  # q축 전류

            # 쇄교자속 계산: lambda_dq = L_dq * i_dq + lambda_f_dq
            lambda_d = L_d * i_d + femm_model.phi_m  # d축은 영구자석 자속(phi_m) 포함
            lambda_q = L_q * i_q                     # q축 전류가 0인 경우

            data_rows.append({
                'Beta': beta,
                'Idq': float(idq),
                'L_d_mH': round(L_d * 1000, 4),                     # d축 인덕턴스 (mH)
                'L_q_mH': round(L_q * 1000, 4),                     # q축 인덕턴스 (mH)
                'Lambda_d_Wb': round(lambda_d * 1000, 2),           # d축 쇄교자속 (mWb)
                'Lambda_q_Wb': round(lambda_q * 1000, 2),           # q축 쇄교자속 (mWb)
            })

    df_results = pd.DataFrame(data_rows)

    # 6. 피벗 테이블 변환 (가로축: Idq 전류 크기, 세로축: Beta 각도)
    df_lambdad_pivot = df_results.pivot(index='Beta', columns='Idq', values='Lambda_d_Wb')
    df_lambdaq_pivot = df_results.pivot(index='Beta', columns='Idq', values='Lambda_q_Wb')

    # 7. CSV 파일로 저장
    df_lambdad_pivot.to_csv("모델로부터Lambda_d_Idq_matrix.csv", encoding="utf-8-sig")
    df_lambdaq_pivot.to_csv("모델로부터Lambda_q_Idq_matrix.csv", encoding="utf-8-sig")

    print("=== 인덕턴스 행렬 변환 기반 쇄교자속 맵 테이블 생성 및 CSV 저장 완료 ===")
    return df_lambdad_pivot, df_lambdaq_pivot

if __name__ == '__main__':
    generate_idq_flux_linkage_table()