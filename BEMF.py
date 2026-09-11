from concurrent.futures import ProcessPoolExecutor
import math
import os
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np

try:
    import femm
except ImportError:
    print("pyfemm이 설치되어 있지 않습니다. (pip install pyfemm)")
    sys.exit(1)

# ==========================================
# 설정 변수
# ==========================================
BASE_FILE = "ioniq5-8.FEM"  # 원본 FEM 파일
ROTOR_GROUP_ID = 1  # 회전자 Group 번호
SPEED_RPM = 141.03  # 회전 속도 (RPM)
CIRCUIT_NAMES = ["A", "B", "C"]  # FEMM 내 상 회로 이름

# 기계적 회전각 범위 및 간격 (도 단위)
# 백기전력 계산 시 미분 정확도를 위해 0.5도~1도 정도의 조촘한 간격 권장
DEG_STEP = 1
ANGLES = [round(a, 2) for a in np.arange(0, 31, DEG_STEP)]


def run_simulation(args):
    """각 각도별 회전자 회전 후 .ans 생성"""
    angle_deg, base_file, worker_id = args
    temp_file = f"temp_emf_worker_{worker_id}_{angle_deg:.2f}deg.fem"

    try:
        shutil.copy(base_file, temp_file)
        femm.openfemm(1)
        femm.opendocument(temp_file)

        # 1. 무부하 상태 설정 (전류 0A)
        for c_name in CIRCUIT_NAMES:
            try:
                femm.mi_modifycircprop(c_name, 1, 0.0)
            except Exception:
                pass

        # 2. 회전자 이동
        if angle_deg != 0:
            femm.mi_clearselected()
            femm.mi_selectgroup(ROTOR_GROUP_ID)
            femm.mi_moverotate(0, 0, angle_deg)  # (0,0) 중심 회전

        femm.mi_saveas(temp_file)
        femm.mi_analyze(1)

        femm.closefemm()
        return angle_deg, True, temp_file
    except Exception as e:
        print(f"[{angle_deg}도] 해석 중 오류: {e}")
        return angle_deg, False, temp_file


def extract_flux_linkage(temp_file):
    """결과 파일(.ans)에서 각 상의 Flux Linkage[Wb-turns] 추출"""
    ans_file = temp_file.replace(".fem", ".ans")
    flux_dict = {c: 0.0 for c in CIRCUIT_NAMES}

    if not os.path.exists(ans_file):
        return flux_dict

    try:
        femm.openfemm(1)
        femm.opendocument(ans_file)

        for c_name in CIRCUIT_NAMES:
            # mo_getcircuitproperties(circuit_name) -> (current, volts, flux_linkage)
            prop = femm.mo_getcircuitproperties(c_name)
            flux_dict[c_name] = prop[2]  # Index 2 : Flux Linkage

        femm.closefemm()
    except Exception as e:
        print(f"Flux linkage 추출 실패: {e}")

    # 임시 파일 정리
    for f in [temp_file, ans_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    return flux_dict


def main():
    if not os.path.exists(BASE_FILE):
        print(f"파일을 찾을 수 없습니다: {BASE_FILE}")
        return

    print("=== 1단계: 병렬 FEMM 쇄교 자속(Flux Linkage) 해석 ===")
    task_args = [(ang, BASE_FILE, i) for i, ang in enumerate(ANGLES)]
    max_workers = min(os.cpu_count() or 4, len(ANGLES))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sim_results = list(executor.map(run_simulation, task_args))

    print("\n=== 2단계: 쇄교 자속(Flux Linkage) 데이터 수집 ===")
    flux_data = {c: [] for c in CIRCUIT_NAMES}

    # 각도순 정렬
    sim_results.sort(key=lambda x: x[0])

    for angle_deg, success, temp_file in sim_results:
        if success:
            f_dict = extract_flux_linkage(temp_file)
            for c in CIRCUIT_NAMES:
                flux_data[c].append(f_dict[c])
        else:
            for c in CIRCUIT_NAMES:
                flux_data[c].append(0.0)

    # ==========================================
    # 3단계: Back EMF 계산 (-dFlux/dt)
    # ==========================================
    omega_m = SPEED_RPM * (2 * math.pi / 60.0)  # rad/s
    omega_e = 4 * omega_m  # rad/s
    angles_rad = np.radians(ANGLES)  # deg -> rad
    angles_rad_e = angles_rad * 4  # deg -> rad

    emf_data = {}
    for c in CIRCUIT_NAMES:
        flux_array = np.array(flux_data[c])
        # 수치 미분: d(Flux)/d(theta) [Wb/rad]
        dflux_dtheta = np.gradient(flux_array, angles_rad_e)

        # e = - omega_m * (dFlux / dtheta)
        emf = -omega_e * dflux_dtheta
        emf_data[c] = emf

    # ==========================================
    # 4단계: Matplotlib 시각화
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Subplot 1: 쇄교 자속 (Flux Linkage)
    for c in CIRCUIT_NAMES:
        ax1.plot(ANGLES, flux_data[c], label=f"Phase {c}")
    ax1.set_title("Stator Flux Linkage (No-Load)", fontsize=12)
    ax1.set_ylabel("Flux Linkage [Wb-turns]", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    # Subplot 2: 역기전력 (Back EMF)
    for c in CIRCUIT_NAMES:
        ax2.plot(ANGLES, emf_data[c], label=f"Phase {c}")
    ax2.set_title(f"Back EMF @ {SPEED_RPM:.0f} RPM", fontsize=12)
    ax2.set_xlabel("Mechanical Angle [deg]", fontsize=10)
    ax2.set_ylabel("Back EMF Voltage [V]", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("back_emf_plot.png", dpi=300)
    print("\n[플롯 저장 완료] back_emf_plot.png")
    plt.show()


if __name__ == "__main__":
    main()