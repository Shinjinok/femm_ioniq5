from concurrent.futures import ProcessPoolExecutor
import math
import os
import shutil
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 필수 모듈 확인
try:
    import femm
except ImportError:
    print("오류: 'pyfemm' 모듈이 설치되지 않았습니다.")
    print("설치 명령: pip install pyfemm")
    sys.exit(1)

# 회전자 그룹 번호 (사용자 FEM 모델의 회전자 그룹 ID로 설정)
ROTOR_GROUP_ID = 1


def run_simulation_only(args):
    """1단계: 각 프로세스에서 회전자를 해당 각도만큼 회전시킨 후 FEMM 해석 수행"""
    angle_deg, base_file_path, worker_id = args

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

        # 1. 코깅 토크 해석을 위한 전류 0A (무부하) 설정
        circuit_names = ["A", "B", "C"]
        for c_name in circuit_names:
            try:
                femm.mi_modifycircprop(c_name, 1, 0.0)
            except Exception:
                pass

        # 2. 회전자 그룹(ROTOR_GROUP_ID) 선택 및 해당 각도만큼 회전
        if angle_deg != 0:
            femm.mi_clearselected()
            femm.mi_selectgroup(ROTOR_GROUP_ID)
            # mi_moverotate(x_center, y_center, angle_deg)
            femm.mi_moverotate(0, 0, angle_deg)

        # 3. 변경사항 저장 및 해석 실행 (.ans 생성)
        femm.mi_saveas(temp_file)
        femm.mi_analyze(1)
        print(f"[해석 완료] 기계적 회전각 {angle_deg:.2f}도")

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
    angle_deg, temp_file, output_dir="femm_plots"
):
    """2단계: 결과(.ans)를 열어 코깅 토크 계산 및 결과 이미지 저장"""
    ans_file = temp_file.replace(".fem", ".ans")
    torque = 0.0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        if not os.path.exists(ans_file):
            print(
                f"[{angle_deg}도] 결과 파일({ans_file})이 존재하지 않습니다."
            )
            return 0.0

        femm.openfemm(1)

        femm.opendocument(ans_file)

        # 1. 회전자 그룹 선택 및 맥스웰 응력 텐서(22) 기반 토크 계산
        femm.mo_clearblock()
        try:
            femm.mo_groupselectblock(1)
            femm.mo_groupselectblock(4)
        except Exception:
            pass

        torque = femm.mo_blockintegral(22)
        print(
            f"[토크 계산완료] 기계적 회전각 {angle_deg:.2f}도 | 코깅 토크: {torque:.4f} Nm"
        )

        # 2. 결과 자속밀도 화면 저장
        femm.main_resize(1200, 1000)
        femm.mo_zoomnatural()
        femm.mo_showdensityplot(1, 0, 4, 1e-4, "bmag")

        img_path = os.path.join(
            output_dir, f"cogging_result_{angle_deg:.1f}deg.png"
        )
        femm.mo_savebitmap(img_path)

        femm.closefemm()

    except Exception as e:
        print(f"[{angle_deg}도] 토크 계산 또는 이미지 저장 중 오류: {e}")
        torque = 0.0
        return torque

    # 3. 이미지 워터마크/텍스트 표기
    try:
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("malgun.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        text_position = (20, 20)
        text_color = (0, 0, 0)
        text_str = f"Angle: {angle_deg} deg | Cogging Torque: {torque:.4f} Nm"

        draw.text(text_position, text_str, fill=text_color, font=font)
        img.save(img_path)

    except Exception as e:
        print(f"[이미지 텍스트 합성 실패]: {img_path} (오류: {e})")

    # 정리: 생성된 임시 .fem 및 .ans 삭제 (필요시 주석 해제)
    """
    for f_path in [temp_file, ans_file]:
        if os.path.exists(f_path):
            try:
                os.remove(f_path)
            except Exception:
                pass
    """

    return torque


def main():
    base_file_path = "ioniq5-6.FEM"  # 원본 FEM 파일 경로

    if not os.path.exists(base_file_path):
        print(f"오류: {base_file_path} 파일을 찾을 수 없습니다.")
        return

    # -------------------------------------------------------------
    # 코깅 토크 해석 범위 설정 (기계각 기준)
    # 정밀한 파형 관찰을 위해 0.5도 또는 1도 간격 추천
    # (예: 한 슬롯 피치 구간인 0~15도 또는 0~360도 전체)
    # -------------------------------------------------------------
    angles = [round(a, 2) for a in list(np.arange(0, 15.5, 0.5))]

    print("=== 1단계: 병렬 FEMM 코깅 토크 해석 시작 ===")
    task_args = [
        (angle, base_file_path, i) for i, angle in enumerate(angles)
    ]

    # CPU 코어 수 이하로 worker 제한 (시스템 부하 방지)
    max_workers = min(os.cpu_count() or 4, len(angles))

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            simulation_results = list(
                executor.map(run_simulation_only, task_args)
            )
    except Exception as e:
        print(f"오류: 병렬 해석 중 문제가 발생했습니다: {e}")
        return

    print("\n=== 2단계: 후처리 코깅 토크 계산 및 시각화 ===")
    results = []
    for angle_deg, success, temp_file in simulation_results:
        if success:
            torque = calculate_torque_and_save_plots(
                angle_deg, temp_file, output_dir="cogging_result_images"
            )
            results.append((angle_deg, torque))
        else:
            results.append((angle_deg, 0.0))

    results.sort(key=lambda x: x[0])
    sorted_angles = [r[0] for r in results]
    torques = [r[1] for r in results]

    # 결과 요약 출력
    print("\n--- 코깅 토크 해석 결과 요약 ---")
    for ang, tq in zip(sorted_angles, torques):
        print(f"회전각: {ang:5.2f}도 | 코깅 토크: {tq:8.5f} Nm")

    # Matplotlib 코깅 토크 플롯 생성
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(
            sorted_angles,
            torques,
            marker="o",
            linestyle="-",
            color="crimson",
            linewidth=1.8,
            markersize=4,
            label="Cogging Torque",
        )

        # Peak-to-Peak 값 계산 및 표시
        p2p_torque = max(torques) - min(torques)
        plt.title(
            f"Cogging Torque Profile (Peak-to-Peak: {p2p_torque:.4f} Nm)",
            fontsize=13,
        )
        plt.xlabel("Mechanical Angle [deg]", fontsize=11)
        plt.ylabel("Torque [Nm]", fontsize=11)
        plt.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper right")
        plt.xlim(min(sorted_angles), max(sorted_angles))

        plt.tight_layout()

        # 그래프 파일 저장
        plot_save_path = "cogging_torque_plot.png"
        plt.savefig(plot_save_path, dpi=300)
        print(f"\n[플롯 저장 완료] {plot_save_path}")

        plt.show()

    except Exception as e:
        print(f"경고: 그래프 생성 중 오류 발생: {e}")


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