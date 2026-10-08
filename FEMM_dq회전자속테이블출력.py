import os
import shutil
import time
import datetime
import glob
import subprocess
import numpy as np
import pandas as pd
import femm
from multiprocessing import Pool, cpu_count

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


# --- 멀티프로세싱 워커 관련 전역 변수 및 초기화 함수 ---
_base_fem_path = None

def init_worker(base_fem_path):
    """프로세스 생성 시 최초 1회 실행: 기준 경로 저장 및 FEMM 세션 오픈"""
    global _base_fem_path
    _base_fem_path = base_fem_path
    femm.openfemm(1)


def worker_task(task_args):
    """개별 (idq_val, beta_deg) 조합별로 각도/전류 이름의 .fem 복사, 회전, 전류 설정, FEMM 해석 수행"""
    global _base_fem_path
    idq_val, beta_deg = task_args
    
    # PID 대신 각도와 전류를 포함한 고유 파일명 생성
    worker_fem_path = f"model_worker_theta_{beta_deg}_idq_{idq_val}.fem"
    shutil.copy(_base_fem_path, worker_fem_path)
    
    try:
        femm.opendocument(worker_fem_path)
        
        # 1. 로터 회전 처리 (그룹 1, 20)
        if abs(beta_deg) > 1e-5:
            femm.mi_clearselected()
            femm.mi_selectgroup(1)   # 로터 그룹 1 선택
            femm.mi_selectgroup(20)  # 로터 그룹 20 선택
            femm.mi_moverotate(0, 0, beta_deg/4)

        # 2. 삼상 전류 고정 설정: idq * [1, -1/2, -1/2]
        femm.mi_setcurrent('A', idq_val * 1.0)
        femm.mi_setcurrent('B', idq_val * (-0.5))
        femm.mi_setcurrent('C', idq_val * (-0.5))
        
        # 3. 해석 실행 및 솔루션 로드
        femm.mi_analyze(1)
        femm.mi_loadsolution()
        
        # 4. 쇄교 자속 추출
        _, _, lambda_a = femm.mo_getcircuitproperties('A')
        _, _, lambda_b = femm.mo_getcircuitproperties('B')
        _, _, lambda_c = femm.mo_getcircuitproperties('C')
        
        # 5. d-q 축 자속 변환
        lambda_d, lambda_q, lambda_0 = abc_to_dq(np.array([lambda_a, lambda_b, lambda_c]), np.radians(beta_deg))
        
        # 6. 토크 추출 (그룹 1, 20)
        femm.mo_clearblock()
        femm.mo_groupselectblock(1)
        torque_val = femm.mo_blockintegral(22)  # 22: Torque via Weighted Stress Tensor
        
        print(f"[Theta: {beta_deg}°, I_dq: {idq_val}A] 완료 (Torque: {torque_val:.4f})")
        return (beta_deg, idq_val, lambda_d, lambda_q, lambda_0, torque_val)
        
    except Exception as e:
        print(f"[오류 발생] (Theta: {beta_deg}°, I_dq: {idq_val}A): {e}")
        raise e


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


def calculate_dq_inductance_map_parallel():
    # 0. 실행 전 기존에 떠있던 잔여 FEMM 프로세스 일괄 청소
    kill_lingering_femm()

    base_fem_path = "ioniq5-13.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    # 전류 크기 및 회전 각도 범위 설정
    idq_list = np.arange(0, 341, 34)
    theta_list = np.arange(-180, 181, 5)

    all_tasks = []
    for beta_deg in theta_list:
        for idq_val in idq_list:
            all_tasks.append((idq_val, beta_deg))
            
    total_tasks = len(all_tasks)
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
    
    flat_results = []
    try:
        with Pool(processes=num_processes, initializer=init_worker, initargs=(base_fem_path,)) as pool:
            for result in pool.imap_unordered(worker_task, all_tasks):
                flat_results.append(result)
    finally:
        pass
            
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
    
    # 4. 결과를 데이터프레임으로 변환 (스케일링 적용)
    data_rows = []
    for beta_deg, idq_val, lambda_d, lambda_q, lambda_0, torque_val in flat_results:
        scaled_lambda_d = round(lambda_d * 8000.0, 2)
        scaled_lambda_q = round(lambda_q * 8000.0, 2)
        scaled_lambda_0 = round(lambda_0 * 8000.0, 2)
        scaled_torque = round(torque_val * 8.0, 2)
        
        data_rows.append({
            'Beta': beta_deg,
            'I_dq': idq_val,
            'Lambda_d': scaled_lambda_d,
            'Lambda_q': scaled_lambda_q,
            'Lambda_0': scaled_lambda_0,
            'Torque': scaled_torque
        })
        
    df_results = pd.DataFrame(data_rows)
    
    # 5. 행: Beta(전류각), 열: I_dq(전류크기) 형태로 피벗 테이블 변환
    df_lambda_d_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_d')
    df_lambda_q_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_q')
    df_lambda_0_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Lambda_0')
    df_torque_pivot = df_results.pivot(index='Beta', columns='I_dq', values='Torque')
    
    # 파일이 열려있는지 먼저 체크
    check_and_release_files(target_files)
    
    # 7. 각각 별도의 CSV 파일로 저장
    df_lambda_d_pivot.to_csv(file_d, encoding="utf-8-sig")
    df_lambda_q_pivot.to_csv(file_q, encoding="utf-8-sig")
    df_lambda_0_pivot.to_csv(file_0, encoding="utf-8-sig")
    df_torque_pivot.to_csv(file_t, encoding="utf-8-sig")
    
    print(f" '{file_d}', '{file_q}', '{file_0}', '{file_t}' 저장 완료! (행: 회전각도°, 열: 전류크기A)")

    # 8. 파일 저장까지 모두 완료된 후 임시 파일 일괄 삭제
    cleanup_all_temp_files()
    kill_lingering_femm()

    return df_lambda_d_pivot, df_lambda_q_pivot, df_lambda_0_pivot, df_torque_pivot

file_d = "FEMM_Lambda_d_matrix.csv"
file_q = "FEMM_Lambda_q_matrix.csv"
file_0 = "FEMM_Lambda_0_matrix.csv"
file_t = "FEMM_Torque_matrix.csv"

target_files = [file_d, file_q, file_0, file_t]

if __name__ == '__main__':
    # 파일이 열려있는지 먼저 체크
    check_and_release_files(target_files)
    cleanup_all_temp_files()
    kill_lingering_femm()
    import multiprocessing
    multiprocessing.freeze_support()
    
    df_d, df_q, df_0, df_t = calculate_dq_inductance_map_parallel()