import numpy as np
import pandas as pd
from FEMM_Ioniq5_ev_model import interpolators, get_inductance_matrix

def generate_idq_inductance_table():
    # 1. 조건 설정
    # 베타 각도: 90도 ~ 180도 (5도 간격) -> 세로축 (행)
    beta_list = np.arange(90, 181, 5)
    # 전류 크기(Idq): 0A ~ 340A (34A 간격) -> 가로축 (열)
    idq_list = np.arange(0, 341, 34)
    
    theta_r = 0.0  # 전기각 고정
    data_rows = []

    print("전류 크기(Idq) 및 베타 각도에 따른 인덕턴스 매핑 연산 중...")

    for beta in beta_list:
        beta_rad = np.radians(beta)
        for idq in idq_list:
            # Idq 전류 크기와 베타 각도로부터 d-q 축 전류(id, iq) 역산
            id_val = idq * np.cos(beta_rad)
            iq_val = idq * np.sin(beta_rad)

            # 1) Park/Clark 변환 역과정으로 3상 전류(i_abc) 계산
            ia = id_val * np.cos(theta_r) - iq_val * np.sin(theta_r)
            ib = id_val * np.cos(theta_r - 2.0*np.pi/3.0) - iq_val * np.sin(theta_r - 2.0*np.pi/3.0)
            ic = -ia - ib
            i_abc = np.array([ia, ib, ic])

            # 2) 모델의 인덕턴스 행렬 함수 호출 (3x3 Ls matrix)
            Ls = get_inductance_matrix(i_abc, theta_r)

            # 3) d-q 축 및 3상 주요 인덕턴스 성분 추출
            l_d_val = Ls[0, 0] * 1000.0  # 단위 mH 변환 예시
            l_q_val = Ls[1, 1] * 1000.0

            data_rows.append({
                'Beta': beta,
                'Idq': float(idq),
                'L_d_mH': round(l_d_val, 4),
                'L_q_mH': round(l_q_val, 4),
                'L_11_uH': round(Ls[0, 0] * 1e6, 2)  # 3상 a상 자기인덕턴스 (uH)
            })

    df_results = pd.DataFrame(data_rows)

    # 4. 피벗 테이블 변환 (가로축: Idq 전류 크기, 세로축: Beta 각도)
    df_ld_pivot = df_results.pivot(index='Beta', columns='Idq', values='L_d_mH')
    df_lq_pivot = df_results.pivot(index='Beta', columns='Idq', values='L_q_mH')
    df_l11_pivot = df_results.pivot(index='Beta', columns='Idq', values='L_11_uH')

    # 5. CSV 파일로 저장
    df_ld_pivot.to_csv("Inductance_Ld_Idq_matrix.csv", encoding="utf-8-sig")
    df_lq_pivot.to_csv("Inductance_Lq_Idq_matrix.csv", encoding="utf-8-sig")
    df_l11_pivot.to_csv("Inductance_Phase_L11_Idq_matrix.csv", encoding="utf-8-sig")

    print("=== Idq 및 베타 각도 기준 인덕턴스 맵 테이블 생성 및 CSV 저장 완료 ===")
    print("저장된 파일: Inductance_Ld_Idq_matrix.csv, Inductance_Lq_Idq_matrix.csv, Inductance_Phase_L11_Idq_matrix.csv")

    return df_ld_pivot, df_lq_pivot

if __name__ == '__main__':
    generate_idq_inductance_table()