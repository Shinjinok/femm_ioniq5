import os
import shutil
import time
import datetime
import glob
import subprocess
import numpy as np
import pandas as pd
from femm import FemmSession
from multiprocessing import Pool, cpu_count
import multiprocessing
import psutil

def kill_lingering_femm():
    """기존에 열려 있거나 좀비로 남은 FEMM 프로세스들을 강제 종료"""
    for proc_name in ["femm.exe", "wfemm.exe", "femm_gui.exe"]:
        try:
            subprocess.run(["taskkill", "/f", "/im", proc_name], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print(" 백그라운드 잔여 FEMM 프로세스 정리 완료.")

def abc_to_dq(i_abc, b):
    clac_3x3 = 2/3 * np.array([
        [1, -1/2, -1/2], 
        [0, np.sqrt(3)/2, -np.sqrt(3)/2],
        [1/2, 1/2, 1/2]
    ])
    
    park_3x3 = np.array([
        [np.cos(b), np.sin(b), 0], 
        [-np.sin(b), np.cos(b), 0],
        [0, 0, 1]
    ])
    
    dq0 = park_3x3 @ clac_3x3 @ i_abc
    return dq0

def check_and_release_files(file_list):
    for file_path in file_list:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'a'):
                    pass
            except IOError:
                raise PermissionError(
                    f"\n[오류] 대상 파일 '{file_path}'이(가) 다른 프로그램(엑셀 등)에서 열려 있습니다. "
                    f"해당 파일을 닫은 후 다시 실행해 주세요."
                )

# --- 멀티프로세싱 전역 변수 (워커별 세션 유지) ---
_session = None
_base_fem_path = None

def init_worker(base_fem_path, base_port=48010):
    """
    워커 프로세스 생성 시 최초 1회만 실행:
    각 프로세스마다 고유 포트를 할당하여 FemmSession을 한 번만 생성하고 유지합니다.
    """
    global _session, _base_fem_path
    _base_fem_path = base_fem_path
    
    try:
        worker_idx = multiprocessing.current_process()._identity[0] - 1
    except (AttributeError, IndexError):
        worker_idx = 0
        
    port = base_port + worker_idx
    
    try:
        # 각 워커 프로세스당 단 하나의 세션만 생성 (헤드리스 모드)
        _session = FemmSession(port=port, headless=True)
    except Exception as e:
        print(f"[워커 초기화 오류] Port {port}: {e}")
        raise e

def worker_task(task_args):
    """
    이미 생성되어 있는 전역 _session 객체를 재사용하여 연산을 수행합니다.
    """
    global _session, _base_fem_path
    idq_val, beta_deg = task_args
    
    pid = os.getpid()
    worker_fem_path = f"model_worker_theta_{beta_deg}_idq_{idq_val}_p{pid}.fem"
    shutil.copy(_base_fem_path, worker_fem_path)
    
    try:
        _session.opendocument(worker_fem_path)
        
        # 1. 로터 회전 처리 (그룹 1, 20)
        if abs(beta_deg) > 1e-5:
            _session.mi_clearselected()
            _session.mi_selectgroup(1)
            _session.mi_selectgroup(20)
            _session.mi_moverotate(0, 0, beta_deg/4)

        # 2. 삼상 전류 고정 설정
        _session.mi_setcurrent('A', idq_val * 1.0)
        _session.mi_setcurrent('B', idq_val * (-0.5))
        _session.mi_setcurrent('C', idq_val * (-0.5))
        
        # 3. 해석 실행 및 솔루션 로드
        _session.mi_analyze(1)
        _session.mi_loadsolution()
        
        # 4. 쇄교 자속 추출
        _, _, lambda_a = _session.mo_getcircuitproperties('A')
        _, _, lambda_b = _session.mo_getcircuitproperties('B')
        _, _, lambda_c = _session.mo_getcircuitproperties('C')
        
        # 5. d-q 축 자속 변환
        lambda_d, lambda_q, lambda_0 = abc_to_dq(np.array([lambda_a, lambda_b, lambda_c]), np.radians(beta_deg))
        
        # 6. 토크 추출
        _session.mo_clearblock()
        _session.mo_groupselectblock(1)
        torque_val = _session.mo_blockintegral(22)
        
        print(f"[Theta: {beta_deg}°, I_dq: {idq_val}A] 완료 (Torque: {torque_val:.4f})")
        return (beta_deg, idq_val, lambda_d, lambda_q, lambda_0, torque_val)
        
    except Exception as e:
        print(f"[오류 발생] (Theta: {beta_deg}°, I_dq: {idq_val}A): {e}")
        raise e
    finally:
        # 연산에 사용된 임시 파일(.fem, .ans 등) 정리
        for ext in ['.fem', '.ans', '.anh', '.log']:
            target_file = worker_fem_path.replace('.fem', ext)
            if os.path.exists(target_file):
                try:
                    os.remove(target_file)
                except:
                    pass

def calculate_dq_inductance_map_parallel():
    kill_lingering_femm()

    base_fem_path = "ioniq5-13.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    idq_list = np.arange(0, 341, 34)
    theta_list = np.arange(-180, 181, 5)

    all_tasks = []
    for beta_deg in theta_list:
        for idq_val in idq_list:
            all_tasks.append((idq_val, beta_deg))
            
    total_tasks = len(all_tasks)
    num_processes = min(64, cpu_count())
    
    print(f"==================================================")
    print(f" 총 연산 조합 수 : {total_tasks}개")
    print(f" 병렬 프로세스 수 : {num_processes}개 (FemmSession 세션 재사용 구조)")
    print(f"==================================================")
    
    start_time_sec = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" 시뮬레이션 시작 시간: {start_time_str}")
    print(f"--------------------------------------------------")
    
    flat_results = []
    # Pool 생성 시 initializer로 워커별 세션 초기화 실행
    with Pool(processes=num_processes, initializer=init_worker, initargs=(base_fem_path, 48010)) as pool:
        try:
            for result in pool.imap_unordered(worker_task, all_tasks):
                flat_results.append(result)
        finally:
            pool.terminate()
            pool.join()
            
    end_time_sec = time.time()
    elapsed_sec = end_time_sec - start_time_sec
    print(f"--------------------------------------------------")
    print(f" 시뮬레이션 총 소요 시간: {int(elapsed_sec // 60)}분 {elapsed_sec % 60:.2f}초")
    print(f"==================================================")
    
    data_rows = []
    for beta_deg, idq_val, lambda_d, lambda_q, lambda_0, torque_val in flat_results:
        data_rows.append({
            'Beta': beta_deg,
            'I_dq': idq_val,
            'Lambda_d': (lambda_d * 8000.0, 2),
            'Lambda_q': (lambda_q * 8000.0, 2),
            'Lambda_0': (lambda_0 * 8000.0, 2),
            'Torque': (torque_val * 8.0, 2)
        })
        
    df_results = pd.DataFrame(data_rows)
    
    df_lambda_d_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_d')
    df_lambda_q_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_q')
    df_lambda_0_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_0')
    df_torque_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Torque')
    
    check_and_release_files(target_files)
    
    df_lambda_d_pivot.to_csv(file_d, encoding="utf-8-sig")
    df_lambda_q_pivot.to_csv(file_q, encoding="utf-8-sig")
    df_lambda_0_pivot.to_csv(file_0, encoding="utf-8-sig")
    df_torque_pivot.to_csv(file_t, encoding="utf-8-sig")
    
    print(f" '{file_d}', '{file_q}', '{file_0}', '{file_t}' 저장 완료!")
    kill_lingering_femm()

file_d = "Rotate_FEMM_Lambda_d_matrix.csv"
file_q = "Rotate_FEMM_Lambda_q_matrix.csv"
file_0 = "Rotate_FEMM_Lambda_0_matrix.csv"
file_t = "Rotate_FEMM_Torque_matrix.csv"
target_files = [file_d, file_q, file_0, file_t]

if __name__ == '__main__':
    check_and_release_files(target_files)
    kill_lingering_femm()
    
    multiprocessing.freeze_support()
    calculate_dq_inductance_map_parallel()