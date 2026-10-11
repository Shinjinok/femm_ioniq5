from asyncio import subprocess
import os
import shutil
import time
import datetime
import atexit
import numpy as np
import pandas as pd
import femm
from multiprocessing import Pool, cpu_count


def cleanup_all_temp_files():
    """모든 연산 및 파일 저장이 끝난 후 생성된 임시 파일(.fem, .ans 등) 일괄 삭제"""
    try:
        femm.closefemm()
    except:
        pass
        
    for ext in ['*.fem', '*.ans']:
        for file_path in glob.glob(f"model_worker_theta_*{ext[1:]}"):
            try:
                os.remove(file_path)
            except:
                pass
    print(" 모든 임시 파일(.fem, .ans)이 안전하게 정리되었습니다.")


def check_and_release_files(file_list):
    """저장할 파일들이 열려있는지 확인하고 사용 중이면 예외 발생"""
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
# --- 백그라운드 잔여 FEMM 프로세스 강제 정리 ---
def kill_lingering_femm():
    """기존에 열려 있거나 좀비로 남은 FEMM 프로세스들을 강제 종료"""
    for proc_name in ["femm.exe", "wfemm.exe"]:
        try:
            subprocess.run(["taskkill", "/f", "/im", proc_name], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    print(" 백그라운드 잔여 FEMM 프로세스 정리 완료.")
    
def abc_to_dq(i_abc, b):
    # 전체를 감싸는 바깥쪽 대괄호([ ])가 반드시 있어야 합니다.
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


def dq_to_abc(dq0, b):
    clac = np.array([[1, -1/2, -1/2], 
                        [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                        [1/2, 1/2, 1/2]])
    iclac = np.transpose(clac)
    park_3x3 = np.array([
                [np.cos(b), np.sin(b), 0], 
                [-np.sin(b), np.cos(b), 0],
                [0, 0, 1]
            ])
    ipark = np.transpose(park_3x3)
    abc = iclac @ ipark @ dq0
    return abc


# --- 멀티프로세싱 워커 관련 전역 변수 및 초기화/정리 함수 ---
_worker_fem_path = None

def cleanup_worker():
    """프로세스 종료 시 FEMM 인스턴스 닫기 및 임시 파일 정리"""
    try:
        femm.closefemm()
    except:
        pass
    global _worker_fem_path
    if _worker_fem_path and os.path.exists(_worker_fem_path):
        try:
            os.remove(_worker_fem_path)
        except:
            pass
        ans_path = _worker_fem_path.replace('.fem', '.ans')
        if os.path.exists(ans_path):
            try:
                os.remove(ans_path)
            except:
                pass

def init_worker(base_fem_path):
    """프로세스 생성 시 최초 1회 실행: 고유 fem 복사 및 FEMM 세션 오픈"""
    global _worker_fem_path
    pid = os.getpid()
    _worker_fem_path = f"model_worker_{pid}.fem"
    shutil.copy(base_fem_path, _worker_fem_path)
    
    femm.openfemm(1)
    femm.opendocument(_worker_fem_path)
    atexit.register(cleanup_worker)


def worker_task(task_args):
    """개별 (beta_val, id_val, iq_val) 조합을 받아 FEMM 해석 및 결과 추출 수행"""
    beta_val, id_val, iq_val = task_args
    theta_r = 0.0  # 전기각 고정
    
    try:
        # 1. Park/Clark 변환 역과정 (3상 전류 계산)
        ia, ib, ic = dq_to_abc(np.array([id_val, iq_val, 0.0]), theta_r)
        
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
        lambda_d, lambda_q, lambda_0 = abc_to_dq(np.array([lambda_a, lambda_b, lambda_c]), theta_r)
        
        # 6. 토크 추출
        femm.mo_clearblock()
        femm.mo_groupselectblock(1)
        torque_val = femm.mo_blockintegral(22)  # 22: Torque via Weighted Stress Tensor
        
        print(f"[PID {os.getpid()}] Beta: {beta_val}°, Id: {id_val:.2f}, Iq: {iq_val:.2f} 완료 (Torque: {torque_val:.4f})")
        return (beta_val, id_val, iq_val, lambda_d, lambda_q, lambda_0, torque_val)
        
    except Exception as e:
        print(f"[PID {os.getpid()}] 오류 발생 (Beta: {beta_val}°, Id: {id_val:.2f}, Iq: {iq_val:.2f}): {e}")
        raise e


def calculate_dq_inductance_map_parallel():
    base_fem_path = "ioniq5-13.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    # 전류 크기 0 ~ 340A (34A 간격), 위상각 -180 ~ 180도 (5도 간격)
    idq_list = np.arange(0, 341, 34)
    ibeta_list = np.arange(-180, 181, 5)
    
    # Meshgrid를 통한 d, q 전류 격자 생성
    IDQ, IBETA = np.meshgrid(idq_list, ibeta_list)
    id_grid = IDQ * np.cos(np.radians(IBETA))
    iq_grid = IDQ * np.sin(np.radians(IBETA))
    
    # 1. 모든 개별 (beta, id, iq) 조합 리스트 생성
    all_tasks = []
    for beta_val, id_row, iq_row in zip(ibeta_list, id_grid, iq_grid):
        for id_val, iq_val in zip(id_row, iq_row):
            all_tasks.append((beta_val, id_val, iq_val))
            
    total_tasks = len(all_tasks)
    
    # 2. 고정 프로세스 수 설정 (최대 60개, 단 전체 태스크 수가 60개보다 적으면 태스크 수만큼만 생성)
    num_processes = min(60, total_tasks)
    
    print(f"==================================================")
    print(f" 총 연산 조합 수 : {total_tasks}개")
    print(f" 고정 프로세스 수 : {num_processes}개 (동적 작업 분배 방식)")
    print(f"==================================================")
    
    # --- [시간 측정 시작] ---
    start_time_sec = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f" 시뮬레이션 시작 시간: {start_time_str}")
    print(f"--------------------------------------------------")
    
    # 3. 60개의 프로세스 풀 생성 및 imap_unordered를 통한 동적 작업 처리
    flat_results = []
    with Pool(processes=num_processes, initializer=init_worker, initargs=(base_fem_path,)) as pool:
        for result in pool.imap_unordered(worker_task, all_tasks):
            flat_results.append(result)
            
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
    
    # 4. 결과를 데이터프레임으로 변환 (8배수 및 스케일링 적용)
    data_rows = []
    for beta_val, id_val, iq_val, lambda_d, lambda_q, lambda_0, torque_val in flat_results:
        idq_val = round(np.sqrt(id_val**2 + iq_val**2), 2)
        
        scaled_lambda_d = round(lambda_d * 8000.0, 2)
        scaled_lambda_q = round(lambda_q * 8000.0, 2)
        scaled_lambda_0 = round(lambda_0 * 8000.0, 2)
        scaled_torque = round(torque_val * 8.0, 2)
        
        data_rows.append({
            'Beta': beta_val,
            'I_dq': idq_val,
            'Lambda_d': scaled_lambda_d,
            'Lambda_q': scaled_lambda_q,
            'Lambda_0': scaled_lambda_0,
            'Torque': scaled_torque
        })
        
    df_results = pd.DataFrame(data_rows)
    
    # 5. 행: Beta(위상각), 열: I_dq(전류크기) 형태로 피벗 테이블 변환
    df_lambda_d_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_d')
    df_lambda_q_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_q')
    df_lambda_0_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_0')   
    df_torque_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Torque')
    
    # 6. 각각 별도의 CSV 파일로 저장
    file_d = "FEMM_Lambda_d_matrix.csv"
    file_q = "FEMM_Lambda_q_matrix.csv"
    file_0 = "FEMM_Lambda_0_matrix.csv"
    file_t = "FEMM_Torque_matrix.cdv" if False else "FEMM_Torque_matrix.csv"
    
    df_lambda_d_pivot.to_csv(file_d, encoding="utf-8-sig")
    df_lambda_q_pivot.to_csv(file_q, encoding="utf-8-sig")
    df_lambda_0_pivot.to_csv(file_0, encoding="utf-8-sig")
    df_torque_pivot.to_csv(file_t, encoding="utf-8-sig")
    
    print(f" '{file_d}', '{file_q}', '{file_t}' 저장 완료! (행: 위상각°, 열: 전류크기A)")

    return df_lambda_d_pivot, df_lambda_q_pivot, df_torque_pivot


if __name__ == '__main__':
    kill_lingering_femm()  # 기존에 열려 있거나 좀비로 남은 FEMM 프로세스들을 강제 종료

    import multiprocessing
    multiprocessing.freeze_support()
    
    df_d, df_q, df_t = calculate_dq_inductance_map_parallel()
    cleanup_worker()  # 모든 연산 및 파일 저장이 끝난 후 생성된 임시 파일(.fem, .ans 등) 일괄 삭제