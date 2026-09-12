from concurrent.futures import ProcessPoolExecutor
import math
import os
import shutil
import sys
import time  # 소요 시간 측정을 위한 모듈 추가
from tkinter import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 필수 모듈 확인
try:
    import femm
except ImportError:
    print("오류: 'pyfemm' 모듈이 설치되지 않았습니다.")
    print("설치 명령: pip install pyfemm")
    sys.exit(1)

I_max = 228
image_save = False  # 이미지 저장 여부 (True: 저장, False: 저장 안함)
temp_file_delete = True  # 임시 파일 삭제 여부 (True: 삭제, False: 삭제 안함)
base_file_path = "ioniq5-13.FEM"  # 원본 FEM 파일 경로
femm_path = r"C:\femm42\bin"  # 사용자 지정 FEMM 경로


def run_simulation_only(args):
    """1단계: 각 프로세스에서 독립적으로 FEMM을 열어 해석(.ans 생성까지만) 수행"""
    angle_deg, base_file_path, femm_path, worker_id = args

    temp_file = f"temp_model_worker_{worker_id}_{angle_deg}deg.fem"

    try:
        shutil.copy(base_file_path, temp_file)
    except Exception as e:
        print(f"[{angle_deg}도] 파일 복사 실패: {e}")
        return angle_deg, False, temp_file

    femm_opened = False
    try:
        # 독립된 백그라운드 FEMM 인스턴스 실행
        femm.openfemm(1)
        femm_opened = True

        femm.opendocument(temp_file)

        # 1. 3상 전류 계산
        theta_rad = math.radians(angle_deg)
        i_a = I_max * math.sin(theta_rad)
        i_b = I_max * math.sin(theta_rad - math.radians(120))
        i_c = I_max * math.sin(theta_rad + math.radians(120))

        # 회로 속성 수정
        circuit_names = ["A", "B", "C"]
        currents = [i_a, i_b, i_c]

        for c_name, curr in zip(circuit_names, currents):
            try:
                femm.mi_modifycircprop(c_name, 1, curr)
            except Exception:
                pass

        # 2. 메시 생성 및 분석 실행 (해석 파일 .ans 생성)
        femm.mi_analyze(1)
        print(f"[해석 완료] 전기적 회전각 {angle_deg:2d}도")

        return angle_deg, True, temp_file

    except Exception as e:
        print(f"[오류] {angle_deg}도 해석 중 예외 발생: {e}")
        return angle_deg, False, temp_file

    finally:
        if femm_opened:
            try:
                femm.closefemm()
            except Exception:
                pass


def calculate_torque_and_save_plots(
    angle_deg, temp_file, femm_path, output_dir="femm_plots"
):
    """2단계: 결과(.ans)를 열어 토크 계산 및 화면 그림(이미지)으로 저장"""
    ans_file = temp_file.replace(".fem", ".ans")
    torque = 0.0
    femm_opened = False

    # 저장할 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        if not os.path.exists(ans_file):
            print(f"[{angle_deg}도] 결과 파일({ans_file})이 존재하지 않습니다.")
            return 0.0

        femm.openfemm()
        femm_opened = True

        # femm.opendocument(temp_file)
        femm.opendocument(ans_file)
        # femm.mi_loadsolution()

        # 1. 그룹 선택 및 토크 계산 (Weighted Stress Tensor: 22)
        femm.mo_clearblock()
        # for g_id in [1, 4]:  # 회전자 그룹 ID
        try:
            femm.mo_groupselectblock(2)  # 고정자 ID
        except Exception:
            pass

        torque = femm.mo_blockintegral(22) * 8
        print(
            f"[토크 계산완료] 전기적 회전각 {angle_deg:2d}도 | 토크: {torque:.4f} Nm"
        )

        # 2. 결과 화면 이미지로 저장하기
        if image_save:
            femm.mo_clearblock()
            femm.main_resize(1200, 1000)
            femm.mo_zoomnatural()
            femm.mo_showdensityplot(1, 0, 4, 1e-4, "bmag")
            # femm.mo_reload()
            img_path = os.path.join(
                output_dir, f"torque_result_{angle_deg}deg.png"
            )
            femm.mo_savebitmap(img_path)
            print(f"[이미지 저장완료] {img_path}")

    except Exception as e:
        print(f"[{angle_deg}도] 토크 계산 또는 이미지 저장 중 오류 발생: {e}")
        torque = 0.0

    if image_save:
        output_filename = img_path
        try:
            img = Image.open(output_filename)
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("malgun.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            text_position = (20, 20)
            text_color = (0, 0, 0)
            draw.text(text_position, temp_file, fill=text_color, font=font)
            img.save(output_filename)
            print(f"[이미지 및 텍스트 합성 완료] {output_filename}")

        except Exception as e:
            print(
                f"[이미지 저장 완료, 텍스트 합성 실패]: {output_filename} (오류: {e})"
            )

    if temp_file_delete:
        if femm_opened:
            try:
                femm.closefemm()
            except Exception:
                pass

        for f_path in [temp_file, ans_file]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception:
                    pass

    return torque


def main():
    # 전체 작업 시작 시간 기록
    start_time = time.time()

    if not os.path.exists(base_file_path):
        print(f"오류: {base_file_path} 파일을 찾을 수 없습니다.")
        return

    # 0부터 360도까지 10도 간격 설정
    angles = list(range(0, 360, 10))

    # 3상 전류 미리 계산 (그래프 표현용)
    ia_list, ib_list, ic_list = [], [], []
    for ang in angles:
        rad = math.radians(ang)
        ia_list.append(I_max * math.sin(rad))
        ib_list.append(I_max * math.sin(rad - math.radians(120)))
        ic_list.append(I_max * math.sin(rad + math.radians(120)))

    print("=== 1단계: 병렬 FEMM 해석(Simulation) 시작 ===")
    task_args = [
        (angle, base_file_path, femm_path, i) for i, angle in enumerate(angles)
    ]

    simulation_results = []
    max_workers = len(angles)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            simulation_results = list(
                executor.map(run_simulation_only, task_args)
            )
    except Exception as e:
        print(f"오류: 병렬 해석 중 문제가 발생했습니다: {e}")
        return

    print("\n=== 2단계: 후처리 토크 계산 및 이미지 저장 시작 ===")
    results = []
    for angle_deg, success, temp_file in simulation_results:
        if success:
            torque = calculate_torque_and_save_plots(
                angle_deg, temp_file, femm_path, output_dir="femm_result_images"
            )
            results.append((angle_deg, torque))
        else:
            results.append((angle_deg, 0.0))

    if not results:
        print("오류: 시뮬레이션 결과가 없습니다.")
        return

    results.sort(key=lambda x: x[0])
    sorted_angles = [r[0] for r in results]
    torques = [r[1] for r in results]

    # 결과 요약 출력
    print("\n--- 모든 시뮬레이션 및 토크 계산 완료 ---")
    for ang, tq, ia, ib, ic in zip(
        sorted_angles, torques, ia_list, ib_list, ic_list
    ):
        print(
            f"각도: {ang:3d}도 | 토크: {tq:7.4f} Nm || Ia: {ia:6.2f}A, Ib: {ib:6.2f}A, Ic: {ic:6.2f}A"
        )

    # Matplotlib 서브플롯 시각화
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(
            sorted_angles,
            ia_list,
            marker="o",
            linestyle="-",
            color="r",
            label="Phase A",
        )
        ax1.plot(
            sorted_angles,
            ib_list,
            marker="s",
            linestyle="-",
            color="g",
            label="Phase B",
        )
        ax1.plot(
            sorted_angles,
            ic_list,
            marker="^",
            linestyle="-",
            color="b",
            label="Phase C",
        )
        ax1.set_title(f"3-Phase Input Currents (Max {I_max}A)", fontsize=13)
        ax1.set_ylabel("Current [A]", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        ax2.plot(
            sorted_angles,
            torques,
            marker="o",
            linestyle="-",
            color="darkorange",
            linewidth=2,
            label="Torque (Group 1 & 4)",
        )
        ax2.set_title(
            "Electrical Angle (°) vs Electromagnetic Torque", fontsize=13
        )
        ax2.set_xlabel("Electrical Angle [deg]", fontsize=11)
        ax2.set_ylabel("Torque [Nm]", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="upper right")

        ax2.set_xlim(min(sorted_angles), max(sorted_angles))
        plt.tight_layout()

        plt.savefig("torque_current_summary.png", dpi=300)
        print("\n[종합 그래프 저장완료] torque_current_summary.png")

        # 전체 작업 종료 시간 기록 및 총 소요 시간 계산
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(
            f"\n[총 소요 시간]: {minutes}분 {seconds:.2f}초 (총 {elapsed_time:.2f}초)"
        )

        plt.show()
    except Exception as e:
        print(f"경고: 그래프 생성 중 오류 성공/실패 여부 확인 필요: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n예기치 않은 오류가 발생했습니다: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)