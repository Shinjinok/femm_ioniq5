from concurrent.futures import ProcessPoolExecutor, as_completed
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
BASE_FILE = "ioniq5-13.FEM"  # 원본 FEM 파일
ROTOR_GROUP_IDS = [1, 20]   # 회전자 Group 번호
SPEED_RPM = 141.03  # 회전 속도 (RPM)
CIRCUIT_NAMES = ["A", "B", "C"]  # FEMM 내 상 회로 이름
POLE_PAIRS = 4  # 극쌍수 (Ioniq 5 등 주파수 계산용)

# 전기각 0도 ~ 360도를 만족하기 위한 기계각 설정 (0 ~ 360 / POLE_PAIRS)
MAX_MECH_ANGLE = 180.0 / POLE_PAIRS
DEG_STEP = 5/4  # 미분 및 FFT 정확도를 위한 0.5도 간격 권장
ANGLES = [round(a, 2) for a in np.arange(-MAX_MECH_ANGLE, MAX_MECH_ANGLE + 1e-5, DEG_STEP)]
temp_file_delete = True  # 시뮬레이션 후 임시 파일 삭제 여부 (True: 삭제, False: 유지)

def run_simulation(args):
    """각 각도별 회전자 회전 후 .ans 생성 (예외 안전 처리)"""
    angle_deg, base_file, worker_id = args
    temp_file = f"temp_emf_worker_{worker_id}_{angle_deg:.2f}deg.fem"
    ans_file = temp_file.replace(".fem", ".ans")

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

            for g_id in ROTOR_GROUP_IDS:
                femm.mi_selectgroup(g_id)
            femm.mi_moverotate(0, 0, angle_deg)  # (0,0) 중심 회전

        femm.mi_saveas(temp_file)
        femm.mi_analyze(1)

        femm.closefemm()
        return angle_deg, True, temp_file
    except Exception as e:
        print(f"⚠️ [{angle_deg}도] 해석 중 예외 발생: {e}")
        for f in [temp_file, ans_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
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
            prop = femm.mo_getcircuitproperties(c_name)
            flux_dict[c_name] = prop[2]  # Index 2 : Flux Linkage

        femm.closefemm()
    except Exception as e:
        print(f"⚠️ Flux linkage 추출 실패: {e}")
    
    if temp_file_delete:
        for f in [temp_file, ans_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    return flux_dict


def main():
    if not os.path.exists(BASE_FILE):
        print(f"❌ 파일을 찾을 수 없습니다: {BASE_FILE}")
        return

    print("=== 1단계: 병렬 FEMM 쇄교 자속(Flux Linkage) 해석 시작 ===")
    task_args = [(ang, BASE_FILE, i % 60) for i, ang in enumerate(ANGLES)]

    # Windows ProcessPoolExecutor 제한(max_workers <= 61)을 준수하기 위해 최대 60으로 제한
    cpu_count = os.cpu_count() or 4
    max_workers = min(60, cpu_count)
    print(
        f"ℹ️ 시스템 코어 수: {cpu_count}개 | 적용된 최대 병렬 워커 수(max_workers): {max_workers}개"
    )

    sim_results = []

    # 단일 ProcessPoolExecutor를 생성하여 모든 작업을 한 번에 등록
    # 작업이 끝나는 대로 비동기로 다음 작업이 자동으로 할당됩니다.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_angle = {
            executor.submit(run_simulation, arg): arg[0] for arg in task_args
        }

        for future in as_completed(future_to_angle):
            ang = future_to_angle[future]
            try:
                result = future.result()
                sim_results.append(result)
                print(f"✔ [완료] 각도 {ang}도 해석 성공 여부: {result[1]}")
            except Exception as exc:
                print(f"❌ [오류] 각도 {ang}도 프로세스 실행 중 예외: {exc}")
                sim_results.append((ang, False, ""))

    print("\n=== 2단계: 쇄교 자속(Flux Linkage) 데이터 수집 및 저장 ===")
    flux_data = {c: [] for c in CIRCUIT_NAMES}
    sim_results.sort(key=lambda x: x[0])  # 각도순 정렬

    for angle_deg, success, temp_file in sim_results:
        if success and temp_file:
            f_dict = extract_flux_linkage(temp_file)
            for c in CIRCUIT_NAMES:
                flux_data[c].append(f_dict[c])
        else:
            print(f"⚠️ 각도 {angle_deg}도 데이터 누락으로 기본값(0.0) 처리됩니다.")
            for c in CIRCUIT_NAMES:
                flux_data[c].append(0.0)

    # 전기각 계산 (Deg)
    angles_mech = np.array(ANGLES)
    angles_elec = angles_mech * POLE_PAIRS

    # 쇄교 자속 데이터 저장 (회전 턴수 보정 필요시 계수 조정, 예: *8)
    flux_arrays = {}
    flux_csv_data = [angles_mech, angles_elec]
    for c in CIRCUIT_NAMES:
        arr = np.array(flux_data[c]) * 8
        flux_arrays[c] = arr
        flux_csv_data.append(arr)

    np.savetxt(
        "results_flux.csv",
        np.column_stack(flux_csv_data),
        delimiter=",",
        header="Mechanical_Angle,Electrical_Angle,Phase_A,Phase_B,Phase_C",
        comments="",
    )
    print("[저장 완료] results_flux.csv")

    # ==========================================
    # 3단계: Back EMF 계산 (-dFlux/dt) 및 저장
    # ==========================================
    omega_m = SPEED_RPM * (2 * math.pi / 60.0)  # 기계 각속도 [rad/s]
    omega_e = omega_m * POLE_PAIRS  # 전기 각속도 [rad/s]
    angles_rad_e = np.radians(angles_elec)

    emf_data = {}
    emf_csv_data = [angles_mech, angles_elec]
    for c in CIRCUIT_NAMES:
        dflux_dtheta_e = np.gradient(flux_arrays[c], angles_rad_e)
        emf = -omega_e * dflux_dtheta_e
        emf_data[c] = emf
        emf_csv_data.append(emf)

    np.savetxt(
        "results_emf.csv",
        np.column_stack(emf_csv_data),
        delimiter=",",
        header="Mechanical_Angle,Electrical_Angle,Phase_A,Phase_B,Phase_C",
        comments="",
    )
    print("[저장 완료] results_emf.csv")

    # ==========================================
    # 4단계: FFT (고조파 분석) 수행 및 저장
    # ==========================================
    N = len(angles_mech)
    fft_freqs = np.fft.rfftfreq(N, d=np.radians(DEG_STEP * POLE_PAIRS))
    harmonic_orders = np.arange(len(fft_freqs))

    fft_data = {}
    fft_csv_data = [harmonic_orders]
    for c in CIRCUIT_NAMES:
        fft_val = np.fft.rfft(emf_data[c])
        magnitude = np.abs(fft_val) / N * 2
        if len(magnitude) > 0:
            magnitude[0] = magnitude[0] / 2  # DC 성분 보정
        fft_data[c] = magnitude
        fft_csv_data.append(magnitude)

    np.savetxt(
        "results_fft.csv",
        np.column_stack(fft_csv_data),
        delimiter=",",
        header="Harmonic_Order,Phase_A,Phase_B,Phase_C",
        comments="",
    )
    print("[저장 완료] results_fft.csv")

    # ==========================================
    # 5단계: Matplotlib 시각화
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    # 1. 쇄교 자속 플롯
    for c in CIRCUIT_NAMES:
        axes[0].plot(angles_elec, flux_arrays[c], label=f"Phase {c}")
    axes[0].set_title("Stator Flux Linkage (No-Load)", fontsize=11)
    axes[0].set_ylabel("Flux Linkage [Wb-turns]", fontsize=10)
    axes[0].set_xlim(0, 360)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # 2. 역기전력 플롯
    for c in CIRCUIT_NAMES:
        axes[1].plot(angles_elec, emf_data[c], label=f"Phase {c}")
    axes[1].set_title(f"Back EMF @ {SPEED_RPM:.2f} RPM", fontsize=11)
    axes[1].set_ylabel("Back EMF Voltage [V]", fontsize=10)
    axes[1].set_xlabel("Electrical Angle [deg]", fontsize=10)
    axes[1].set_xlim(0, 360)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    # 3. FFT 고조파 스펙트럼 플롯
    max_harmonic = min(31, len(harmonic_orders) - 1)
    for c in CIRCUIT_NAMES:
        axes[2].bar(
            harmonic_orders[: max_harmonic + 1]
            + (0.2 if c == "B" else (0.4 if c == "C" else 0.0)),
            fft_data[c][: max_harmonic + 1],
            width=0.2,
            label=f"Phase {c}",
        )
    axes[2].set_title("Back EMF Harmonic Spectrum (FFT)", fontsize=11)
    axes[2].set_ylabel("Peak Amplitude [V]", fontsize=10)
    axes[2].set_xlabel("Harmonic Order", fontsize=10)
    axes[2].set_xlim(0, max_harmonic + 1)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plot_filename = "back_emf_fft_analysis.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\n[플롯 저장 완료] {plot_filename}")
    plt.show()


if __name__ == "__main__":
    main()