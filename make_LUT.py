import os
import shutil
import time
import subprocess
import csv
import sys
import socket
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from femm import FemmSession

poles = 8                # 극수
pole_pairs = poles / 2   # 극쌍수 (4)
CIRCUIT_NAMES = ["A", "B", "C"]
HEADLESS_MODE = True     # True: GUI 없이 FEMM 고속 실행

# 워커 프로세스 전역 변수
_worker_session = None
_worker_fem_path = None

def kill_lingering_femm():
    """기존에 열려 있거나 좀비로 남은 FEMM 프로세스들을 강제 종료"""
    for proc_name in ["femm.exe", "wfemm.exe", "femm_gui.exe"]:
        try:
            subprocess.run(
                ["taskkill", "/f", "/im", proc_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
    time.sleep(1.0)

def find_free_port():
    """운영체제(OS)로부터 현재 사용 가능한 빈 TCP 포트를 동적으로 할당받음"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

def dq_to_abc(dq0, b):
    sr32 = np.sqrt(3) / 2
    clac2 = np.array([
        [1, -1/2, -1/2],
        [0, sr32, -sr32],
        [1/2, 1/2, 1/2]
    ])
    cb = np.cos(b)
    sb = np.sin(b)
    park_3x3 = np.array([
        [cb, sb, 0],
        [-sb, cb, 0],
        [0, 0, 1]
    ])
    abc = clac2.T @ park_3x3.T @ dq0
    return abc

def abc_to_dq(val_abc, b):
    sr32 = np.sqrt(3) / 2
    clac_3x3 = (2 / 3) * np.array([
        [1, -1/2, -1/2],
        [0, sr32, -sr32],
        [1/2, 1/2, 1/2]
    ])
    cb = np.cos(b)
    sb = np.sin(b)
    park_3x3 = np.array([
        [cb, sb, 0],
        [-sb, cb, 0],
        [0, 0, 1]
    ])
    dq0 = park_3x3 @ clac_3x3 @ val_abc
    return dq0

def run_femm_flux_only(session, Iabc, theta_mech_deg):
    session.mi_setcurrent('A', float(Iabc[0]))
    session.mi_setcurrent('B', float(Iabc[1]))
    session.mi_setcurrent('C', float(Iabc[2]))
    
    session.mi_modifyboundprop('slidingAirGap', 10, float(theta_mech_deg))
    session.mi_analyze(1)
    session.mi_loadsolution()
    
    lamda = []
    for c_name in CIRCUIT_NAMES:
        prop = session.mo_getcircuitproperties(c_name)
        lamda.append(prop[2] * pole_pairs * 2)
        
    return np.array(lamda)

def init_worker(base_fem_path):
    """각 워커 프로세스 시작 시 OS가 보장하는 빈 포트를 동적으로 할당받아 세션 초기화"""
    global _worker_session, _worker_fem_path
    
    name = mp.current_process().name
    try:
        worker_id = int(name.split('-')[-1]) - 1
    except:
        worker_id = abs(hash(name)) % 1000
        
    # OS가 충돌 없는 빈 포트를 직접 찾아주므로 포트 겹침 현상 원천 차단
    port = find_free_port()
    _worker_fem_path = f"model_working_worker_{worker_id}_{port}.fem"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if os.path.exists(_worker_fem_path):
                os.remove(_worker_fem_path)
            shutil.copy(base_fem_path, _worker_fem_path)
            
            _worker_session = FemmSession(port=port, headless=HEADLESS_MODE)
            _worker_session.opendocument(_worker_fem_path)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Worker {worker_id} (포트 {port}) 초기화 실패: {e}")
            time.sleep(1.0)

def worker_chunk_task(chunk_data):
    """할당된 세션을 재사용하여 청크 연산 수행"""
    global _worker_session
    results_d = []
    results_q = []
    
    try:
        for id_val, iq_val, theta_e_deg in chunk_data:
            theta_e_rad = np.radians(theta_e_deg)
            theta_mech_deg = theta_e_deg / pole_pairs
            
            Iabc = dq_to_abc(np.array([id_val, iq_val, 0.0]), theta_e_rad)
            lamda_abc = run_femm_flux_only(_worker_session, Iabc, theta_mech_deg)
            
            lamda_dq0 = abc_to_dq(lamda_abc, theta_e_rad)
            lamda_d = lamda_dq0[0]
            lamda_q = lamda_dq0[1]
            
            results_d.append((id_val, iq_val, theta_e_deg, lamda_d))
            results_q.append((id_val, iq_val, theta_e_deg, lamda_q))
                
    except Exception as e:
        return None, None, f"연산 중 오류 발생: {e}"
        
    return results_d, results_q, None

def generate_flux_lut_csv_multiprocessing():
    kill_lingering_femm()

    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    id_grid = np.arange(-100, 100, 25)         # 69개
    iq_grid = np.arange(-100, 100, 25)         # 69개
    theta_e_grid = np.arange(-180, 180, 1)    # 37개

    all_cases = []
    for theta_e_deg in theta_e_grid:
        for id_val in id_grid:
            for iq_val in iq_grid:
                all_cases.append((id_val, iq_val, theta_e_deg))

    total_cases = len(all_cases)  # 총 176,157개
    
    chunk_size = 50
    chunks = []
    for i in range(0, total_cases, chunk_size):
        chunks.append(all_cases[i:i + chunk_size])

    # 사용 가능한 모든 CPU 코어를 최대로 활용
    max_cores = mp.cpu_count()
    
    sys.stdout.write("==================================================\n")
    sys.stdout.write(" FEMM 3D 쇄교자속 LUT 생성 (최대 코어 + 동적 포트 할당)\n")
    sys.stdout.write(f" - 최대 활용 코어 수: {max_cores} 개 (100% 풀가동)\n")
    sys.stdout.write(f" - 총 해석 케이스 수: {total_cases:,} 회 (총 {len(chunks):,}개 묶음)\n")
    sys.stdout.write("==================================================\n")
    sys.stdout.flush()

    file_ld = "LUT_lamda_d.csv"
    file_lq = "LUT_lamda_q.csv"

    start_time = time.time()
    completed_cases = 0

    with open(file_ld, mode="w", newline="", encoding="utf-8") as f_d, \
         open(file_lq, mode="w", newline="", encoding="utf-8") as f_q:
        
        writer_d = csv.writer(f_d)
        writer_q = csv.writer(f_q)

        writer_d.writerow(["Id_A", "Iq_A", "theta_e_deg", "lamda_d_Wb"])
        writer_q.writerow(["Id_A", "Iq_A", "theta_e_deg", "lamda_q_Wb"])

        with Pool(processes=max_cores, initializer=init_worker, initargs=(base_fem_path,)) as pool:
            for res_d, res_q, err in pool.imap_unordered(worker_chunk_task, chunks):
                if err is not None:
                    sys.stdout.write(f"[오류 발생]: {err}\n")
                    sys.stdout.flush()
                    continue

                for row in res_d:
                    writer_d.writerow([row[0], row[1], row[2], f"{row[3]:.8f}"])
                for row in res_q:
                    writer_q.writerow([row[0], row[1], row[2], f"{row[3]:.8f}"])
                
                f_d.flush()
                f_q.flush()
                
                completed_cases += len(res_d)
                elapsed = time.time() - start_time
                progress = (completed_cases / total_cases) * 100
                
                if completed_cases > 0:
                    avg_time_per_case = elapsed / completed_cases
                    remaining_sec = avg_time_per_case * (total_cases - completed_cases)
                else:
                    remaining_sec = 0
                
                rem_min = int(remaining_sec // 60)
                rem_sec = int(remaining_sec % 60)
                
                msg = (f"[진행률 {progress:6.2f}%] 해석 케이스: {completed_cases:,} / {total_cases:,} 회 | "
                       f"경과 시간: {elapsed/60:.1f}분 | "
                       f"종료 예상(남은 시간): 약 {rem_min}분 {rem_sec}초\n")
                
                sys.stdout.write(msg)
                sys.stdout.flush()

    total_elapsed = time.time() - start_time
    t_hour = int(total_elapsed // 3600)
    t_min = int((total_elapsed % 3600) // 60)
    t_sec = int(total_elapsed % 60)

    sys.stdout.write("\n==================================================\n")
    sys.stdout.write(" [완료] 멀티프로세싱 3D 쇄교자속 LUT 추출 성공!\n")
    sys.stdout.write(f" - 총 소요 시간: {t_hour}시간 {t_min}분 {t_sec}초 ({total_elapsed:.1f}초)\n")
    sys.stdout.write(f" - λd 파일: {os.path.abspath(file_ld)}\n")
    sys.stdout.write(f" - λq 파일: {os.path.abspath(file_lq)}\n")
    sys.stdout.write("==================================================\n")
    sys.stdout.flush()
    
    kill_lingering_femm()

if __name__ == '__main__':
    generate_flux_lut_csv_multiprocessing()