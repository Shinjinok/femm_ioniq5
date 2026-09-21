import os
import shutil
import time
import datetime
import numpy as np
import pandas as pd
import femm
from multiprocessing import Pool, cpu_count

def worker_process(args):
    """
    개별 프로세스가 할당받은 (beta_val, id_val, iq_val) 조합 리스트를 순회하며 FEMM 해석을 수행하는 함수
    """
    worker_id, task_chunk, base_fem_path = args
    
    # 프로세스 충돌 방지를 위해 작업용 독립 .fem 파일 복사 생성
    process_fem_path = f"model_worker_{worker_id}.fem"
    shutil.copy(base_fem_path, process_fem_path)
    
    # 각 프로세스별 독립된 FEMM 인스턴스 열기 (1: 백그라운드 실행 또는 가시화 설정)
    femm.openfemm(1)
    femm.opendocument(process_fem_path)
    
    results = []
    theta_r = 0.0  # 전기각 고정
    
    try:
        for beta_val, id_val, iq_val in task_chunk:
            # 1. Park/Clark 변환 역과정 (3상 전류 계산)
            ia = id_val * np.cos(theta_r) - iq_val * np.sin(theta_r)
            ib = id_val * np.cos(theta_r - 2.0*np.pi/3.0) - iq_val * np.sin(theta_r - 2.0*np.pi/3.0)
            ic = -ia - ib
            
            # 2. 회로 전류 설정
            femm.mi_setcurrent('A', ia)
            femm.mi_setcurrent('B', ib)
            femm.mi_setcurrent('C', ic)
            
            # 3. 해석 실행 및 솔루션 로드
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            
            # 4. 쇄교 자속 추출
            _, _, lambda_a = femm.mo_getcircuitproperties('A')
            _, _, lambda_b = femm.mo_getcircuitproperties('B')
            _, _, lambda_c = femm.mo_getcircuitproperties('C')
            
            # 5. d-q 축 자속 변환
            lambda_d = 2.0 / 3.0 * (lambda_a * np.cos(theta_r) + 
                                    lambda_b * np.cos(theta_r - 2.0*np.pi/3.0) + 
                                    lambda_c * np.cos(theta_r + 2.0*np.pi/3.0))
                                    
            lambda_q = -2.0 / 3.0 * (lambda_a * np.sin(theta_r) + 
                                     lambda_b * np.sin(theta_r - 2.0*np.pi/3.0) + 
                                     lambda_c * np.sin(theta_r + 2.0*np.pi/3.0))
            
            results.append((beta_val, id_val, iq_val, lambda_d, lambda_q))
            print(f"[Worker {worker_id}] Beta: {beta_val}°, Id: {id_val:.2f}, Iq: {iq_val:.2f} 완료")
                
    finally:
        # 작업 종료 후 FEMM 닫기 및 임시 파일 정리
        femm.closefemm()
        if os.path.exists(process_fem_path):
            os.remove(process_fem_path)
            ans_path = process_fem_path.replace('.fem', '.ans')
            if os.path.exists(ans_path):
                os.remove(ans_path)
                
    return results

def calculate_dq_inductance_map_parallel():
    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    # 전류 크기 0 ~ 340A (34A 간격), 위상각 90 ~ 180도 (5도 간격)
    idq_list = np.arange(0, 341, 34)
    ibeta_list = np.arange(90, 181, 5)
    
    # Meshgrid를 통한 d, q 전류 격자 생성
    IDQ, IBETA = np.meshgrid(idq_list, ibeta_list)
    id_grid = IDQ * np.cos(np.radians(IBETA))
    iq_grid = IDQ * np.sin(np.radians(IBETA))
    
    # 1. 모든 (beta, id, iq) 조합 리스트 생성 (추적을 위해 beta값도 함께 묶음)
    all_tasks = []
    for beta_val, id_row, iq_row in zip(ibeta_list, id_grid, iq_grid):
        for id_val, iq_val in zip(id_row, iq_row):
            all_tasks.append((beta_val, id_val, iq_val))
            
    total_tasks = len(all_tasks)
    
    # 2. 시스템 최대 가용 스레드 수 확인
    num_processes = cpu_count()
    num_processes = min(num_processes, total_tasks)
    
    print(f"==================================================")
    print(f" 총 연산 조합 수 : {total_tasks}개")
    print(f" 활용 스레드 수  : {num_processes}개")
    print(f"==================================================")
    
    # 3. 전체 조합(Task)을 코어 수에 맞게 균등 분할(Chunking)
    task_chunks = np.array_split(all_tasks, num_processes)
    
    worker_args = []
    for idx, chunk in enumerate(task_chunks):
        if len(chunk) > 0:
            worker_args.append((idx, list(chunk), base_fem_path))
            
    # --- [시간 측정 시작] ---
    start_time_sec = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" 시뮬레이션 시작 시간: {start_time_str}")
    print(f"--------------------------------------------------")
    
    # 4. 멀티프로세싱 풀 실행
    with Pool(processes=len(worker_args)) as pool:
        chunk_results = pool.map(worker_process, worker_args)
        
    # --- [시간 측정 종료] ---
    end_time_sec = time.time()
    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_sec = end_time_sec - start_time_sec
    
    elapsed_min = elapsed_sec // 60
    remaining_sec = elapsed_sec % 60
    
    print(f"--------------------------------------------------")
    print(f" 시뮬레이션 종료 시간: {end_time_str}")
    print(f" 총 소요 시간      : {int(elapsed_min)}분 {remaining_sec:.2f}초 (총 {elapsed_sec:.2f}초)")
    print(f"==================================================")
    
    # 5. 결과를 평탄화하여 데이터프레임으로 변환 (8000 곱하기 및 소수점 2째 자리 반올림 적용)
    flat_results = []
    for process_data in chunk_results:
        flat_results.extend(process_data)
        
    data_rows = []
    for beta_val, id_val, iq_val, lambda_d, lambda_q in flat_results:
        # 전류 크기(idq) 역산 혹은 원본 인덱싱 매칭을 위해 계산 (또는 IDQ에서 직접 찾기)
        idq_val = round(np.sqrt(id_val**2 + iq_val**2), 2)
        
        scaled_lambda_d = round(lambda_d * 8000.0, 2)
        scaled_lambda_q = round(lambda_q * 8000.0, 2)
        
        data_rows.append({
            'Beta': beta_val,
            'I_dq': idq_val,
            'Lambda_d': scaled_lambda_d,
            'Lambda_q': scaled_lambda_q
        })
        
    df_results = pd.DataFrame(data_rows)
    
    # 6. 행: Beta(위상각), 열: I_dq(전류크기) 형태로 피벗 테이블 변환
    df_lambda_d_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_d')
    df_lambda_q_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_q')
    
    # 7. 각각 별도의 CSV 파일로 저장
    file_d = "Lambda_d_matrix.csv"
    file_q = "Lambda_q_matrix.csv"
    
    df_lambda_d_pivot.to_csv(file_d, encoding="utf-8-sig")
    df_lambda_q_pivot.to_csv(file_q, encoding="utf-8-sig")
    
    print(f" '{file_d}' 및 '{file_q}' 저장 완료! (행: 위상각°, 열: 전류크기A)")

    return df_lambda_d_pivot, df_lambda_q_pivot

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    df_d, df_q = calculate_dq_inductance_map_parallel()