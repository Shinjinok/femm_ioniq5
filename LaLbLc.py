import os
import shutil
import time
import datetime
import numpy as np
import pandas as pd
import pythoncom
import femm
from multiprocessing import Pool, cpu_count

def worker_process(args):
    """
    개별 프로세스가 할당받은 전기각(Theta_e) 리스트를 순회하며 
    8극 모터의 전기각/기계각 관계에 따른 로터 회전 및 A상 100A 고정 인가 해석 수행
    """
    pythoncom.CoInitialize()  # Windows COM 초기화
    
    worker_id, task_chunk, base_fem_path, magnet_material_name, rotor_group_no, pole_pairs = args
    
    # 프로세스 충돌 방지를 위해 작업용 독립 .fem 파일 복사 생성
    process_fem_path = f"model_worker_{worker_id}.fem"
    
    results = []
    
    try:
        for theta_e_rad in task_chunk:
            theta_e_deg = np.degrees(theta_e_rad)
            
            # [8극 모터 반영] 전기각을 극쌍수(pole_pairs = 4)로 나누어 실제 기계각(Mechanical Angle) 계산
            theta_m_deg = theta_e_deg / pole_pairs
            
            # 매 스텝마다 신규 파일 복사 및 FEMM 인스턴스 재시작으로 초기 상태 보장
            if os.path.exists(process_fem_path):
                os.remove(process_fem_path)
            shutil.copy(base_fem_path, process_fem_path)
            
            femm.openfemm(1)
            femm.opendocument(process_fem_path)
            
            # 1. 영구자석 물성치 변경 (순수 돌극성 추출: mu=1, Hc=0)
            try:
                femm.mi_modifymaterial(magnet_material_name, 1, 1.0)  # mu_x = 1.0
                femm.mi_modifymaterial(magnet_material_name, 2, 1.0)  # mu_y = 1.0
                femm.mi_modifymaterial(magnet_material_name, 3, 0.0)  # Coercivity = 0
            except Exception as e:
                pass

            # 2. A상 고정 전류 설정 (600A 일정), B상/C상은 0A
            ia_const = 600.0
            femm.mi_setcurrent('A', ia_const)
            femm.mi_setcurrent('B', 0.0)
            femm.mi_setcurrent('C', 0.0)

            # 3. 회전자 기계각 회전 적용
            if rotor_group_no is not None and theta_m_deg != 0.0:
                femm.mi_seteditmode("group")
                femm.mi_clearselected()
                femm.mi_selectgroup(rotor_group_no)
                femm.mi_moverotate(0.0, 0.0, theta_m_deg)
            
            # 4. 해석 실행 및 솔루션 로드
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            
            # 5. a, b, c 상 쇄교 자속 추출
            _, _, lambda_a = femm.mo_getcircuitproperties('A')
            _, _, lambda_b = femm.mo_getcircuitproperties('B')
            _, _, lambda_c = femm.mo_getcircuitproperties('C')
            
            results.append((theta_e_rad, theta_e_deg, theta_m_deg, ia_const, lambda_a, lambda_b, lambda_c))
            
            # FEMM 인스턴스 닫기
            femm.closefemm()
            
    finally:
        pythoncom.CoUninitialize()  # COM 해제
        
        # 임시 파일 및 부산물 정리
        for ext in ['.fem', '.ans', '.rec']:
            file_path = process_fem_path.replace('.fem', ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
                
    return results

def calculate_rotor_sweep_inductance_parallel():
    base_fem_path = "ioniq5-6.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    # 모델 설정 값
    magnet_material_name = "NdFeB 40 MGOe" 
    rotor_group_no = 1     # FEMM 모델 내 회전자 영역의 그룹 번호 (모델에 맞게 수정 필요)
    pole_number = 8        # 폴수 (8극)
    pole_pairs = pole_number / 2  # 극쌍수 (4)

    # 전기각 기준 0도 ~ 180도, 10도 간격 리스트 생성
    theta_e_list = np.radians(np.arange(0, 181, 10))  # 0°, 10°, 20°, ..., 180°
    
    total_tasks = len(theta_e_list)
    num_processes = min(cpu_count(), total_tasks)
    
    print(f"==================================================")
    print(f" [8극 모터 로터 회전 및 A상 600A 여자 해석]")
    print(f" 폴수: {pole_number}극 (극쌍수 p = {int(pole_pairs)})")
    print(f" 총 연산 전기각 수 : {total_tasks}개 (0° ~ 180°, 10° 간격)")
    print(f" 활용 멀티스레드 수 : {num_processes}개")
    print(f"==================================================")
    
    # 작업 분할 (Chunking)
    task_chunks = np.array_split(theta_e_list, num_processes)
    
    worker_args = []
    for idx, chunk in enumerate(task_chunks):
        if len(chunk) > 0:
            worker_args.append((idx, list(chunk), base_fem_path, magnet_material_name, rotor_group_no, pole_pairs))
            
    # --- [시간 측정 시작] ---
    start_time_sec = time.time()
    print(f" 시뮬레이션 시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"--------------------------------------------------")
    
    # 멀티프로세싱 풀 실행
    with Pool(processes=len(worker_args)) as pool:
        chunk_results = pool.map(worker_process, worker_args)
        
    # --- [시간 측정 종료] ---
    elapsed_sec = time.time() - start_time_sec
    print(f"--------------------------------------------------")
    print(f" 시뮬레이션 종료 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 총 소요 시간      : {int(elapsed_sec // 60)}분 {elapsed_sec % 60:.2f}초")
    print(f"==================================================")
    
    # 결과를 데이터프레임으로 재구성 및 인덕턴스 계산
    data_records = []
    for process_data in chunk_results:
        for theta_e_rad, theta_e_deg, theta_m_deg, ia, la, lb, lc in process_data:
            Laa = la / ia if abs(ia) > 1e-5 else 0.0
            Lab = lb / ia if abs(ia) > 1e-5 else 0.0
            Lac = lc / ia if abs(ia) > 1e-5 else 0.0
            
            data_records.append({
                'Theta_Elec_deg': theta_e_deg,
                'Theta_Mech_deg': theta_m_deg,
                'Theta_Elec_rad': theta_e_rad,
                'Ia': ia,
                'Lambda_a': la, 'Lambda_b': lb, 'Lambda_c': lc,
                'Laa': Laa, 'Lab': Lab, 'Lac': Lac
            })

    # 정렬 및 CSV 저장
    df_results = pd.DataFrame(data_records)
    df_results = df_results.sort_values(by='Theta_Elec_deg').reset_index(drop=True)
    
    output_filename = "a_phase_600A_8pole_inductance.csv"
    df_results.to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f" '{output_filename}' 저장 완료!")

    return df_results

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    df_results = calculate_rotor_sweep_inductance_parallel()