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

    print("전류 크기(Idq) 및 베타 각도에 따른 쇄교자속(Flux Linkage) 매핑 연산 중...")

    for beta in beta_list:
        beta_rad = np.radians(beta)
        for idq in idq_list:

            Ls = femm_model.get_inductance_matrix(idq, theta_e)
            i_abc = femm_model.dq_to_abc(np.array([idq, 0]), beta_rad)  # Idq -> abc 변환
            cos_f =np.array([np.cos(theta_e), np.cos(theta_e - 2 * np.pi / 3), np.cos(beta_rad + 2 * np.pi / 3)])
            lambda_abc = np.dot(Ls, i_abc) + femm_model.phi_m * cos_f
            # 쇄교자속 계산
            print(f"lamda_abc: {lambda_abc}")
            #lambda_d, lambda_q = femm_model.abc_to_dq(lambda_abc, theta_e)
            # 4) [핵심] 3상 쇄교자속(lambda_abc)을 d-q축 쇄교자속으로 변환 (Park 변환)
            cos_th = np.cos(beta_rad)
            sin_th = np.sin(beta_rad)
            cos_th_120 = np.cos(beta_rad - 2.0 * np.pi / 3.0)
            sin_th_120 = np.sin(beta_rad - 2.0 * np.pi / 3.0)
            cos_th_p120 = np.cos(beta_rad + 2.0 * np.pi / 3.0)
            sin_th_p120 = np.sin(beta_rad + 2.0 * np.pi / 3.0)

            # Park 변환 공식 적용
            lambda_d = (2.0 / 3.0) * (
                lambda_abc[0] * cos_th + 
                lambda_abc[1] * cos_th_120 + 
                lambda_abc[2] * cos_th_p120
            )

            lambda_q = -(2.0 / 3.0) * (
                lambda_abc[0] * sin_th + 
                lambda_abc[1] * sin_th_120 + 
                lambda_abc[2] * sin_th_p120
            )
            data_rows.append({
                'Beta': beta,
                'Idq': float(idq),
                'Lambda_d_Wb': round(lambda_d*1000, 2),          # d축 쇄교자속 (Wb)
                'Lambda_q_Wb': round(lambda_q*1000, 2),         # q축 쇄교자속 (Wb)
            })

    df_results = pd.DataFrame(data_rows)

    # 4. 피벗 테이블 변환 (가로축: Idq 전류 크기, 세로축: Beta 각도)
    df_lambdad_pivot = df_results.pivot(index='Beta', columns='Idq', values='Lambda_d_Wb')
    df_lambdaq_pivot = df_results.pivot(index='Beta', columns='Idq', values='Lambda_q_Wb')


    # 5. CSV 파일로 저장
    df_lambdad_pivot.to_csv("모델로부터Lambda_d_Idq_matrix.csv", encoding="utf-8-sig")
    df_lambdaq_pivot.to_csv("모델로부터Lambda_q_Idq_matrix.csv", encoding="utf-8-sig")


    print("=== Idq 및 베타 각도 기준 쇄교자속 맵 테이블 생성 및 CSV 저장 완료 ===")
    print("저장된 파일: 모델로부터Lambda_d_Idq_matrix.csv, 모델로부터Lambda_q_Idq_matrix.csv")

    return df_lambdad_pivot, df_lambdaq_pivot

if __name__ == '__main__':
    generate_idq_flux_linkage_table()