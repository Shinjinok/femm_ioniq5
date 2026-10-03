import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def load_and_interpolate_inductance(csv_filename="ioniq5-13.FEM_inductance_table.csv", query_current=75.0):
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_filename}")

    # 1. CSV 파일 로드 및 테이블 생성
    df = pd.read_csv(csv_filename)
    print("=== 인덕턴스 테이블 데이터 로드 완료 ===")
    print(df.head(), "\n")

    currents = df['Current_A'].values
    output_cols = [col for col in df.columns if col != 'Current_A']

    # 2. 상별/특성별 보간 함수(Interpolator) 생성
    # 'cubic'(3차 스플라인) 또는 'linear'(선택 가능) 보간법 적용
    interpolators = {}
    for col in output_cols:
        interpolators[col] = interp1d(
            currents, 
            df[col].values, 
            kind='cubic',             # 3차 스플라인 보간 (부드러운 곡선 연결)
            fill_value="extrapolate"  # 범위를 벗어날 경우 외삽 허용
        )

    # 3. 임의의 입력 전류에 대한 아웃풋 보간 값 계산 테스트
    print(f"--- 입력 전류 [{query_current} A]에 대한 보간 결과 ---")
    interpolated_results = {}
    for col in output_cols:
        val = float(interpolators[col](query_current))
        interpolated_results[col] = val
        print(f"  {col}: {val:.4f} μH")

    return interpolators, df

if __name__ == '__main__':
    # 원하는 입력 전류값(예: 75A)을 넣어서 테스트 가능
    interpolators, df = load_and_interpolate_inductance(query_current=75.0)
    