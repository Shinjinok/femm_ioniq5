import os
import shutil
import time
import datetime
import numpy as np
import pandas as pd
import femm
import pythoncom  # Windows COM 초기화용 모듈
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# 한글 폰트 깨짐 방지 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def abc_to_dq(i_abc, b):
    clac = 2/3 * np.array([[1, -1/2, -1/2], 
                           [0, np.sqrt(3)/2, -np.sqrt(3)/2]])
    park = np.array([[np.cos(b), np.sin(b)], 
                     [-np.sin(b), np.cos(b)]])
    dq = park @ clac @ i_abc
    return dq


def dq_to_abc(v_dq, b):
    clac = np.array([[1, -1/2, -1/2], 
                     [0, np.sqrt(3)/2, -np.sqrt(3)/2]])
    iclac = np.transpose(clac)
    ipark = np.array([[np.cos(b), -np.sin(b)], 
                     [np.sin(b), np.cos(b)]])
    v = iclac @ ipark @ v_dq
    return v


def worker_sweep_process(args):
    """
    개별 프로세스가 할당받은 (I_mag, beta_deg) 조합 리스트를 순회하며 
    FEMM 해석 및 3상 자속 추출을 수행하는 멀티프로세스 워커 함수
    """
    worker_id, task_chunk, base_fem_path = args
    
    # Windows COM 라이브러리 초기화
    pythoncom.CoInitialize()
    
    process_fem_path = f"model_sweep_worker_{worker_id}.fem"
    shutil.copy(base_fem_path, process_fem_path)
    
    femm.openfemm(1)
    femm.opendocument(process_fem_path)
    
    results = []
    
    try:
        for I_mag, beta_deg in task_chunk:
            beta_rad = np.radians(beta_deg)
            id_val = I_mag * np.cos(beta_rad)
            iq_val = I_mag * np.sin(beta_rad)
            
            ia, ib, ic = dq_to_abc(np.array([id_val, iq_val]), 0.0)
            
            femm.mi_setcurrent('A', ia)
            femm.mi_setcurrent('B', ib)
            femm.mi_setcurrent('C', ic)
            
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            
            _, _, lambda_a = femm.mo_getcircuitproperties('A')
            _, _, lambda_b = femm.mo_getcircuitproperties('B')
            _, _, lambda_c = femm.mo_getcircuitproperties('C')
            
            # 8배수 스케일링 적용
            results.append((I_mag, beta_deg, lambda_a * 8000.0, lambda_b * 8000.0, lambda_c * 8000.0))
            print(f"[Worker {worker_id}] 전류: {I_mag}A, Beta: {beta_deg}° 완료")
                
    finally:
        try:
            femm.closefemm()
        except:
            pass
        pythoncom.CoUninitialize()
        
        if os.path.exists(process_fem_path):
            os.remove(process_fem_path)
            ans_path = process_fem_path.replace('.fem', '.ans')
            if os.path.exists(ans_path):
                os.remove(ans_path)
                
    return results


def run_flux_sweep_0_to_360_parallel(base_fem_path):
    """
    베타를 0도~360도(5도 간격) 회전시키며 각 전류 크기별 3상 쇄교자속을 
    멀티프로세싱으로 빠르게 구하고 그래프로 시각화
    """
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    target_currents = [1, 10, 50]
    sweep_angles = np.arange(0, 361, 5) # 0 ~ 360도 (5도 간격)
    
    # 1. 모든 (전류크기, 위상각) 조합 태스크 생성
    all_tasks = []
    for I_mag in target_currents:
        for beta_deg in sweep_angles:
            all_tasks.append((I_mag, beta_deg))
            
    total_tasks = len(all_tasks)
    num_processes = min(cpu_count(), total_tasks)
    
    print(f"==================================================")
    print(f" 0~360도 스윕 총 연산 조합 수 : {total_tasks}개")
    print(f" 활용 멀티프로세스 수       : {num_processes}개")
    print(f"==================================================")
    
    # 2. 태스크를 프로세스 수에 맞게 균등 분할
    task_chunks = np.array_split(all_tasks, num_processes)
    worker_args = [(idx, list(chunk), base_fem_path) for idx, chunk in enumerate(task_chunks) if len(chunk) > 0]
    
    # --- [시간 측정 시작] ---
    start_time_sec = time.time()
    
    # 3. 멀티프로세싱 풀 실행
    with Pool(processes=len(worker_args)) as pool:
        chunk_results = pool.map(worker_sweep_process, worker_args)
        
    # --- [시간 측정 종료] ---
    end_time_sec = time.time()
    print(f"--------------------------------------------------")
    print(f" 0~360도 스윕 총 소요 시간: {end_time_sec - start_time_sec:.2f}초")
    print(f"==================================================")
    
    # 4. 결과 통합 및 정렬
    flat_results = []
    for process_data in chunk_results:
        flat_results.extend(process_data)
        
    temp_data = {I: [] for I in target_currents}
    for I_mag, beta_deg, la, lb, lc in flat_results:
        temp_data[I_mag].append((beta_deg, la, lb, lc))
        
    results_by_current = {I: {'angles': [], 'lambda_a': [], 'lambda_b': [], 'lambda_c': []} for I in target_currents}
    for I_mag in target_currents:
        temp_data[I_mag].sort(key=lambda x: x[0])  # 각도순 정렬
        for beta_deg, la, lb, lc in temp_data[I_mag]:
            results_by_current[I_mag]['angles'].append(beta_deg)
            results_by_current[I_mag]['lambda_a'].append(la)
            results_by_current[I_mag]['lambda_b'].append(lb)
            results_by_current[I_mag]['lambda_c'].append(lc)

    # 5. 그래프 시각화
    plt.figure(figsize=(15, 5 * len(target_currents)))
    
    for idx, I_mag in enumerate(target_currents):
        plt.subplot(len(target_currents), 1, idx + 1)
        angles = results_by_current[I_mag]['angles']
        
        plt.plot(angles, results_by_current[I_mag]['lambda_a'], label='Phase A Flux ($\lambda_a$)', color='red', linewidth=1.5)
        plt.plot(angles, results_by_current[I_mag]['lambda_b'], label='Phase B Flux ($\lambda_b$)', color='green', linewidth=1.5)
        plt.plot(angles, results_by_current[I_mag]['lambda_c'], label='Phase C Flux ($\lambda_c$)', color='blue', linewidth=1.5)
        
        plt.title(f'3상 쇄교 자속 파형 (전류 크기: {I_mag}A)', fontsize=13)
        plt.xlabel('전기각 (Beta / Degrees)', fontsize=11)
        plt.ylabel('쇄교 자속 (Wb-turns)', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
        plt.xticks(np.arange(0, 361, 30))
        plt.legend(loc='upper right')
        
    plt.tight_layout()
    plot_filename = "FEMM_3Phase_Flux_Waveforms_0_360.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\n[+] 3상 쇄교 자속 그래프 저장 완료: {plot_filename}")
    plt.show()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 0 ~ 360도 멀티프로세싱 자속 파형 스윕 실행
    run_flux_sweep_0_to_360_parallel("ioniq5-13.FEM")