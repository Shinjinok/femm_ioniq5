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
    개별 프로세스가 할당받은 (id, iq) 조합 리스트를 순회하며 FEMM 해석을 수행하는 함수
    """
    worker_id, task_chunk, base_fem_path = args
    
    # 프로세스 충돌 방지를 위해 작업용 독립 .fem 파일 복사 생성
    process_fem_path = f"model_worker_{worker_id}.fem"
    shutil.copy(base_fem_path, process_fem_path)
    
    # 각 프로세스별 독립된 FEMM 인스턴스 열기 (0: GUI 숨김 백그라운드 실행)
    femm.openfemm(1)
    femm.opendocument(process_fem_path)
    
    results = []
    theta_r = 0.0  # 전기각 고정
    
    try:
        for id_val, iq_val in task_chunk:
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
            
            results.append((id_val, iq_val, lambda_d, lambda_q))
            print(f"[Worker {worker_id}] Id: {id_val}, Iq: {iq_val} 완료")
                
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
    base_fem_path = "ioniq5-6.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    id_list = np.linspace(-100, 0, 6)
    iq_list = np.linspace(0, 180, 10)
    
    # 1. 모든 (id, iq) 조합 생성 (총 66개)
    all_tasks = [(id_val, iq_val) for id_val in id_list for iq_val in iq_list]
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
    
    # 5. 결과를 전체 맵 그리드로 재구성
    Ld_map = np.zeros((len(id_list), len(iq_list)))
    Lq_map = np.zeros((len(id_list), len(iq_list)))
    
    id_to_idx = {val: i for i, val in enumerate(id_list)}
    iq_to_idx = {val: i for i, val in enumerate(iq_list)}
    
    for process_data in chunk_results:
        for id_val, iq_val, l_d, l_q in process_data:
            i = id_to_idx[id_val]
            j = iq_to_idx[iq_val]
            Ld_map[i, j] = l_d 
            Lq_map[i, j] = l_q

    # 6. Pandas를 이용해 CSV 파일로 저장
    # 행(Index): id_list, 열(Columns): iq_list
    df_Ld = pd.DataFrame(Ld_map, index=id_list, columns=iq_list)
    df_Lq = pd.DataFrame(Lq_map, index=id_list, columns=iq_list)
    
    df_Ld.to_csv("Ld_map.csv", encoding="utf-8-sig")
    df_Lq.to_csv("Lq_map.csv", encoding="utf-8-sig")
    print(" Ld_map.csv 및 Lq_map.csv 저장 완료!")

    return id_list, iq_list, Ld_map, Lq_map

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    id_list, iq_list, Ld_map, Lq_map = calculate_dq_inductance_map_parallel()