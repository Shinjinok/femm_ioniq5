import os
import shutil
import time
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from ref_file3.femm import FemmSession

# 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def worker_cogging_torque(args):
    """멀티프로세싱 워커 함수: 포트 충돌 및 연결 에러 발생 시 재시도(Retry) 로직 포함"""
    index, angle_mech, base_fem_path, base_port, worker_id = args
    working_fem_path = f"model_cogging_{index}_{os.getpid()}.fem"
    shutil.copy(base_fem_path, working_fem_path)

    port = base_port + worker_id
    session = None
    torque = 0.0
    torque2 = 0.0

    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.05 * (worker_id % 12) + 0.2 * attempt)

            session = FemmSession(port=port, headless=True)
            session.opendocument(working_fem_path)

            # 코깅 토크 측정 시 상전류는 0으로 설정
            session.mi_setcurrent('A', 0.0)
            session.mi_setcurrent('B', 0.0)
            session.mi_setcurrent('C', 0.0)

            # 슬라이딩 에어갭 경계를 통한 회전자 회전 (기계각 기준)
            session.mi_modifyboundprop('slidingAirGap', 10, angle_mech)

            session.mi_analyze(1)
            session.mi_loadsolution()

            # 1) 슬라이딩 에어갭 적분
            torque2 = float(session.mo_gapintegral("slidingAirGap", 0))

            # 2) 블록 선택 기반 가중 응력 텐서 토크 (1/8 대칭 모델 -> 8배 스케일링)
            session.mo_clearblock()
            session.mo_groupselectblock(1)
            raw_torque = float(session.mo_blockintegral(22))
            session.mo_clearblock()
            torque = raw_torque * 8.0

            session.close()
            session = None
            break

        except Exception as e:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
                session = None

            if attempt == max_retries - 1:
                print(f"[오류] 기계각 {angle_mech:.5f}° 연산 최종 실패 (포트 {port}): {e}")
                torque = 0.0
                torque2 = 0.0
            else:
                time.sleep(0.5)

        finally:
            if os.path.exists(working_fem_path):
                try:
                    os.remove(working_fem_path)
                except Exception:
                    pass

    return angle_mech, torque, torque2


def calculate_cogging_torque_parallel():
    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    step_deg = 0.01
    angles = np.arange(0.0, 5.0 + step_deg * 0.5, step_deg)
    base_port = 48100
    max_workers = 16  # PC 코어/메모리 및 FEMM 세션 안정성에 맞춰 조정 가능

    print("=" * 52)
    print(" IPMSM 코깅 토크 멀티프로세싱 해석 시작")
    print(f" 각도 범위: 0.0° ~ 90.0° (간격: {step_deg}°, 총 {len(angles)}개 포인트)")
    print(f" 동시 프로세스 수 (Workers/Ports): {max_workers}")
    print("=" * 52)

    tasks = [
        (i, ang, base_fem_path, base_port, i % max_workers)
        for i, ang in enumerate(angles)
    ]

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker_cogging_torque, task): task[1] for task in tasks}
        for future in as_completed(futures):
            ang, tq, tq2 = future.result()
            results.append((ang, tq, tq2))
            print(f"[완료] 기계각: {ang:5.5f}° | Stress*8: {tq:9.5f} Nm | Gap: {tq2:9.5f} Nm")

    elapsed = time.time() - start_time
    print(f"\n해석 완료! 총 소요 시간: {elapsed:.2f}초")

    results.sort(key=lambda x: x[0])
    sorted_angles = [r[0] for r in results]
    sorted_torques = [r[1] for r in results]
    sorted_torques2 = [r[2] for r in results]

    csv_filename = "cogging_torque_result.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["Mechanical_Angle_deg", "Cogging_Torque_Stress8_Nm", "Cogging_Torque_Gap_Nm"])
        for ang, tq, tq2 in results:
            writer.writerow([ang, tq, tq2])
    print(f"결과 CSV 저장 완료: {csv_filename}")

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_angles, sorted_torques, label='Stress Torque (x8)', color='red', linewidth=2, marker='o', markersize=2)
    plt.plot(sorted_angles, sorted_torques2, label='Sliding Band Torque', color='green', linewidth=2, marker='s', markersize=2)
    plt.title("IPMSM Cogging Torque Profile (0° ~ 90°)", fontsize=14)
    plt.xlabel("Mechanical Angle [deg]", fontsize=12)
    plt.ylabel("Cogging Torque [Nm]", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = "cogging_torque_curve.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"그래프 이미지 저장 완료: {plot_filename}")

    plt.show()


if __name__ == '__main__':
    calculate_cogging_torque_parallel()