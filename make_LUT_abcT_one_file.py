import os
import shutil
import time
import subprocess
import csv
import sys
import socket
import glob
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from femm import FemmSession
from tqdm import tqdm  # 실시간 프로그래스 바 출력을 위한 라이브러리
from datetime import datetime

poles = 8                  # 극수
pole_pairs = poles // 2    # 극쌍수 (4)
CIRCUIT_NAMES = ["A", "B", "C"]
HEADLESS_MODE = True       # True: GUI 없이 FEMM 고속 실행

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

def cleanup_working_files():
    """작업 중 생성된 model_working_worker_* 파일들을 일괄 삭제"""
    for temp_file in glob.glob("model_working_worker_*"):
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

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

    # 1/8 대칭 모델 스케일링 적용 (8배)
    session.mo_groupselectblock(1)
    raw_torque = session.mo_blockintegral(22)
    torque = raw_torque * 8.0
        
    return np.array(lamda), torque

def init_worker(base_fem_path):
    """각 워커 프로세스 시작 시 OS가 보장하는 빈 포트를 동적으로 할당받아 세션 초기화"""
    global _worker_session, _worker_fem_path
    
    name = mp.current_process().name
    try:
        worker_id = int(name.split('-')[-1]) - 1
    except:
        worker_id = abs(hash(name)) % 1000
        
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

def worker_single_task(case_data):
    """개별 케이스 연산 수행 (청크 단위를 없애고 1개 단위로 유연하게 분배)"""
    global _worker_session
    id_val, iq_val, theta_e_deg = case_data
    
    try:
        theta_e_rad = np.radians(theta_e_deg)
        theta_mech_deg = theta_e_deg / pole_pairs
        
        # d-q 전류를 abc상 전류로 변환하여 FEMM 입력
        Iabc = dq_to_abc(np.array([id_val, iq_val, 0.0]), theta_e_rad)
        
        # FEMM에서 계산된 3상 원본 쇄교자속 획득
        lamda_abc, Torque = run_femm_flux_only(_worker_session, Iabc, theta_mech_deg)
        
        return (id_val, iq_val, theta_e_deg, lamda_abc[0], lamda_abc[1], lamda_abc[2],Torque), None
    except Exception as e:
        return None, f"연산 중 오류 발생 (Id:{id_val}, Iq:{iq_val}, Theta:{theta_e_deg}): {e}"

def generate_flux_lut_csv_multiprocessing():
    kill_lingering_femm()
    cleanup_working_files()

    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    #id_grid = np.linspace(-350, 350, 70)        
    #iq_grid = np.linspace(-350, 350, 70)   
    id_grid = np.linspace(-350, 350, 51)        
    iq_grid = np.linspace(-350, 350, 51)     
    theta_e_grid = np.linspace(0, 360, 181)     

    all_cases = []
    for theta_e_deg in theta_e_grid:
        for id_val in id_grid:
            for iq_val in iq_grid:
                all_cases.append((id_val, iq_val, theta_e_deg))

    total_cases = len(all_cases)  
    max_cores = mp.cpu_count()
    
    sys.stdout.write("==================================================\n")
    sys.stdout.write(" FEMM 3D 3상(ABC) 쇄교자속 LUT 생성 (tqdm 실시간 진행률 적용)\n")
    sys.stdout.write(f" - 최대 활용 코어 수: {max_cores} 개\n")
    sys.stdout.write(f" - 총 해석 케이스 수: {total_cases:,} 회\n")
    sys.stdout.write("==================================================\n")
    sys.stdout.flush()

    # 현재 날짜 및 시간 가져오기
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 파일명 생성
    file = f"LUT_lamda_abcT_{now}.csv"
    start_time = time.time()

    try:
        with open(file, mode="w", newline="", encoding="utf-8") as f_a:
            writer_a = csv.writer(f_a)
            writer_a.writerow(["Id_A", "Iq_A", "theta_e_deg", "lamda_a_Wb", "lamda_b_Wb", "lamda_c_Wb","Torque"])
            
            with Pool(processes=max_cores, initializer=init_worker, initargs=(base_fem_path,)) as pool:
                # chunksize를 적절히 조절하여 IPC 오버헤드를 줄이면서도 실시간 반영 (예: 5 또는 10)
                # tqdm을 사용하여 터미널에 아름다운 프로그래스 바와 남은 시간(ETA)을 출력합니다.
                with tqdm(total=total_cases, desc="LUT 생성 중", unit="case") as pbar:
                    for res, err in pool.imap_unordered(worker_single_task, all_cases, chunksize=10):
                        if err is not None:
                            tqdm.write(f"[오류 발생]: {err}")
                            pbar.update(1)
                            continue

                        if res is not None:
                            writer_a.writerow([res[0], res[1], res[2], f"{res[3]:.8f}", f"{res[4]:.8f}", f"{res[5]:.8f}", f"{res[6]:.8f}"])
                            f_a.flush()
                        
                        pbar.update(1)
    finally:
        kill_lingering_femm()
        cleanup_working_files()

    total_elapsed = time.time() - start_time
    t_hour = int(total_elapsed // 3600)
    t_min = int((total_elapsed % 3600) // 60)
    t_sec = int(total_elapsed % 60)

    sys.stdout.write("\n==================================================\n")
    sys.stdout.write(" [완료] 멀티프로세싱 3상(ABC) 쇄교자속 LUT 추출 성공!\n")
    sys.stdout.write(f" - 총 소요 시간: {t_hour}시간 {t_min}분 {t_sec}초 ({total_elapsed:.1f}초)\n")
    sys.stdout.write(f" - 파일 경로: {os.path.abspath(file)}\n")
    sys.stdout.write("==================================================\n")
    sys.stdout.flush()

if __name__ == '__main__':
    generate_flux_lut_csv_multiprocessing()