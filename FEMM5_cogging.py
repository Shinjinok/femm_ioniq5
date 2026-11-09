import os
import shutil
import time
import csv
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from femm import FemmSession

# 한글 폰트 설정 및 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def worker_cogging_torque(args):
    """멀티프로세싱 워커 함수: 포트 충돌 및 연결 에러 발생 시 재시도(Retry) 로직 포함"""
    index, angle_mech, base_fem_path, base_port, worker_id = args
    working_fem_path = f"model_cogging_{index}_{os.getpid()}.fem"
    shutil.copy(base_fem_path, working_fem_path)
    
    port = base_port + worker_id
    session = None
    torque = None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 동시 접속으로 인한 포트 충돌을 막기 위해 워커별·시도별 시차 부여
            time.sleep(0.15 * worker_id + 0.2 * attempt)
            
            session = FemmSession(port=port, headless=True)
            session.opendocument(working_fem_path)
            
            # 코깅 토크 측정 시 상전류는 모두 0으로 설정
            session.mi_setcurrent('A', 0.0)
            session.mi_setcurrent('B', 0.0)
            session.mi_setcurrent('C', 0.0)
            
            # 슬라이딩 에어갭 바운더리를 통한 회전자 회전 (기계각 기준)
            session.mi_modifyboundprop('slidingAirGap', 10, angle_mech)
            
            session.mi_analyze(1)
            session.mi_loadsolution()
            
            # 1/8 대칭 모델 스케일링 적용 (8배)
            session.mo_groupselectblock(1)
            raw_torque = session.mo_blockintegral(22)
            torque = raw_torque * 8.0
            
            session.close()
            break  # 성공 시 재시도 루프 탈출
            
        except Exception as e:
            if session:
                try:
                    session.close()
                except:
                    pass
            if attempt == max_retries - 1:
                print(f"[오류] 기계각 {angle_mech:.1f}° 연산 최종 실패 (포트 {port}): {e}")
                torque = 0.0
            else:
                # 포트가 해제되기를 잠시 대기 후 재시도
                time.sleep(0.5)
        finally:
            if os.path.exists(working_fem_path):
                try:
                    os.remove(working_fem_path)
                except:
                    pass
                
    return angle_mech, torque

def calculate_cogging_torque_parallel():
    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")
    
    # 소수점 오차 방지를 위해 round 적용 (0도 ~ 90도, 0.5도 간격)
    angles = [round(x, 1) for x in np.arange(0.0, 90.1, 0.1)]
    base_port = 48100
    
    # 안정적인 병렬 처리를 위해 워커 수를 6~8개로 유지 (포트 충돌 최소화)
    max_workers = 60
    
    print(f"==================================================")
    print(f" IPMSM 코깅 토크 멀티프로세싱 해석 시작 (재시도 로직 적용)")
    print(f" 각도 범위: 0° ~ 90° (간격: 0.5°, 총 {len(angles)}개 포인트)")
    print(f" 동시 프로세스 수 (Workers/Ports): {max_workers}")
    print(f"==================================================")
    
    tasks = [(i, ang, base_fem_path, base_port, i % max_workers) for i, ang in enumerate(angles)]
    
    start_time = time.time()
    results = []
    
    # 병렬 프로세싱 실행
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_cogging_torque, task): task[1] for task in tasks}
        for future in as_completed(futures):
            ang, tq = future.result()
            results.append((ang, tq))
            print(f"[완료] 기계각: {ang:5.1f}° | 코깅 토크: {tq:8.4f} Nm")
            
    elapsed = time.time() - start_time
    print(f"\n해석 완료! 총 소요 시간: {elapsed:.2f}초")
    
    # 각도 기준 정렬
    results.sort(key=lambda x: x[0])
    sorted_angles = [r[0] for r in results]
    sorted_torques = [r[1] for r in results]
    
    # 1. CSV 결과 파일 저장
    csv_filename = "cogging_torque_result.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Mechanical_Angle_deg", "Cogging_Torque_Nm"])
        for ang, tq in results:
            writer.writerow([ang, tq])
    print(f"결과 CSV 저장 완료: {csv_filename}")
    
    # 2. 그래프 시각화 및 이미지 저장
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_angles, sorted_torques, color='darkorange', linewidth=2, marker='o', markersize=3)
    plt.title("IPMSM Cogging Torque Profile (0° ~ 90°)", fontsize=14)
    plt.xlabel("Mechanical Angle [deg]", fontsize=12)
    plt.ylabel("Cogging Torque [Nm]", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_filename = "cogging_torque_curve.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"그래프 이미지 저장 완료: {plot_filename}")
    
    plt.show()

if __name__ == '__main__':
    calculate_cogging_torque_parallel()