import os
import shutil
import time
import datetime
import numpy as np
import pandas as pd
import pythoncom
import femm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# 한글 폰트 깨짐 방지 (Windows 환경 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

delete_temp_files = True  # True로 설정하면 각 프로세스 종료 후 임시 파일 삭제

def worker_process(args):
    """
    개별 프로세스가 할당받은 전기각(Theta_e) 리스트와 지정된 전류(ia_val)로 
    8극 모터의 회전자 회전 및 A, B, C 상별 순차 인가 해석 수행 (총 3회 해석)
    """
    pythoncom.CoInitialize()  # Windows COM 초기화
    
    worker_id, task_chunk, base_fem_path, magnet_material_name, rotor_group_no, pole_pairs, ia_val = args
    process_fem_path = f"model_worker_{worker_id}.fem"
    
    results = []
    
    try:
        for theta_e_rad in task_chunk:
            theta_e_deg = np.degrees(theta_e_rad)
            theta_m_deg = theta_e_deg / pole_pairs
            
            if os.path.exists(process_fem_path):
                os.remove(process_fem_path)
            shutil.copy(base_fem_path, process_fem_path)
            
            femm.openfemm(1)
            femm.opendocument(process_fem_path)
            
            # 1. 영구자석 물성치 변경 (순수 돌극성 추출: mu=1, Hc=0)
            try:
                femm.mi_modifymaterial(magnet_material_name, 1, 1.0)
                femm.mi_modifymaterial(magnet_material_name, 2, 1.0)
                femm.mi_modifymaterial(magnet_material_name, 3, 0.0)
            except Exception:
                pass

            # 회전자 기계각 회전 적용 (동일 각도에서 3번의 이상 전류 인가를 위해 미리 회전)
            if rotor_group_no is not None and theta_m_deg != 0.0:
                femm.mi_seteditmode("group")
                femm.mi_clearselected()
                femm.mi_selectgroup(rotor_group_no)
                femm.mi_moverotate(0.0, 0.0, theta_m_deg)

            # --- [Case 1] A상 여자 (A상에만 전류 인가, B=0, C=0) ---
            femm.mi_setcurrent('A', ia_val)
            femm.mi_setcurrent('B', 0.0)
            femm.mi_setcurrent('C', 0.0)
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            _, _, la_A = femm.mo_getcircuitproperties('A')
            _, _, lb_A = femm.mo_getcircuitproperties('B')
            _, _, lc_A = femm.mo_getcircuitproperties('C')

            # --- [Case 2] B상 여자 (B상에만 전류 인가, A=0, C=0) ---
            femm.mi_setcurrent('A', 0.0)
            femm.mi_setcurrent('B', ia_val)
            femm.mi_setcurrent('C', 0.0)
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            _, _, la_B = femm.mo_getcircuitproperties('A')
            _, _, lb_B = femm.mo_getcircuitproperties('B')
            _, _, lc_B = femm.mo_getcircuitproperties('C')

            # --- [Case 3] C상 여자 (C상에만 전류 인가, A=0, B=0) ---
            femm.mi_setcurrent('A', 0.0)
            femm.mi_setcurrent('B', 0.0)
            femm.mi_setcurrent('C', ia_val)
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            _, _, la_C = femm.mo_getcircuitproperties('A')
            _, _, lb_C = femm.mo_getcircuitproperties('B')
            _, _, lc_C = femm.mo_getcircuitproperties('C')

            # 결과 저장 (권선 턴수 8 곱하기 반영)
            results.append((
                theta_e_rad, theta_e_deg, theta_m_deg, ia_val,
                la_A * 8, lb_A * 8, lc_A * 8,
                la_B * 8, lb_B * 8, lc_B * 8,
                la_C * 8, lb_C * 8, lc_C * 8
            ))
            
            femm.closefemm()
            
    finally:
        pythoncom.CoUninitialize()
        if delete_temp_files:
            for ext in ['.fem', '.ans', '.rec']:
                file_path = process_fem_path.replace('.fem', ext)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
                
    return results

def plot_inductance_results(df_results, ia_val, output_image_path="inductance_plot.png"):
    """특정 전류 조건의 9개 인덕턴스 파형 결과 플롯 및 그래프 저장"""
    theta_deg = df_results['Theta_Elec_deg']
    
    # 9개 인덕턴스 μH 단위 변환
    laa = df_results['Laa'] * 1e6
    lab = df_results['Lab'] * 1e6
    lac = df_results['Lac'] * 1e6
    lba = df_results['Lba'] * 1e6
    lbb = df_results['Lbb'] * 1e6
    lbc = df_results['Lbc'] * 1e6
    lca = df_results['Lca'] * 1e6
    lcb = df_results['Lcb'] * 1e6
    lcc = df_results['Lcc'] * 1e6

    plt.figure(figsize=(12, 8))

    # A상 루프
    plt.plot(theta_deg, laa, label='Laa', color='blue', lw=2)
    plt.plot(theta_deg, lab, label='Lab', color='dodgerblue', lw=1.5, linestyle='--')
    plt.plot(theta_deg, lac, label='Lac', color='deepskyblue', lw=1.5, linestyle=':')

    # B상 루프
    plt.plot(theta_deg, lba, label='Lba', color='green', lw=1.5, linestyle='--')
    plt.plot(theta_deg, lbb, label='Lbb', color='forestgreen', lw=2)
    plt.plot(theta_deg, lbc, label='Lbc', color='limegreen', lw=1.5, linestyle=':')

    # C상 루프
    plt.plot(theta_deg, lca, label='Lca', color='orange', lw=1.5, linestyle='--')
    plt.plot(theta_deg, lcb, label='Lcb', color='darkorange', lw=1.5, linestyle=':')
    plt.plot(theta_deg, lcc, label='Lcc', color='red', lw=2)

    plt.title(f"전류 {ia_val}A 조건 - 3상 9개 인덕턴스 프로파일", fontsize=13, fontweight='bold')
    plt.xlabel("전기각 [deg]", fontsize=11)
    plt.ylabel("인덕턴스 [μH]", fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=9, ncol=3)
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"[전류 {ia_val}A 9개 인덕턴스 파형 플롯 저장 완료] {output_image_path}")

def run_multi_current_sweep():
    base_fem_path = "ioniq5-13.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    magnet_material_name = "Mag" 
    rotor_group_no = 1 
    pole_number = 8 
    pole_pairs = pole_number / 2 

    current_list = [1, 5, 10, 20, 30] + list(np.arange(50, 401, 50))
    theta_e_list = np.radians(np.arange(0, 360, 6))  # 0° ~ 360°, 6° 간격
    
    summary_records = []
    total_start_time = time.time()

    print(f"==================================================")
    print(f" [다중 전류 조건별 8극 모터 3상 인덕턴스(9컴포넌트) 스윕]")
    print(f" 전류 범위: {current_list[0]}A ~ {current_list[-1]}A ")
    print(f" 총 전류 조건 수: {len(current_list)}개")
    print(f"==================================================")

    for ia_val in current_list:
        print(f"\n--- [현재 전류 조건: {ia_val}A] 해석 진행 중 ---")
        
        total_tasks = len(theta_e_list)
        num_processes = min(cpu_count(), total_tasks)
        task_chunks = np.array_split(theta_e_list, num_processes)
        
        worker_args = [(idx, list(chunk), base_fem_path, magnet_material_name, rotor_group_no, pole_pairs, ia_val) 
                       for idx, chunk in enumerate(task_chunks) if len(chunk) > 0]
        
        start_time_sec = time.time()
        with Pool(processes=len(worker_args)) as pool:
            chunk_results = pool.map(worker_process, worker_args)
        elapsed_sec = time.time() - start_time_sec
        print(f" -> {ia_val}A 해석 완료 (소요 시간: {int(elapsed_sec // 60)}분 {elapsed_sec % 60:.2f}초)")
        
        # 결과 데이터 변환 (9개 인덕턴스 계산)
        current_records = []
        for process_data in chunk_results:
            for item in process_data:
                theta_e_rad, theta_e_deg, theta_m_deg, ia, \
                la_A, lb_A, lc_A, \
                la_B, lb_B, lc_B, \
                la_C, lb_C, lc_C = item
                
                inv_ia = 1.0 / ia if abs(ia) > 1e-5 else 0.0

                # A상 루프 인덕턴스 (A상, B상, C상 여자 시 A상 쇄교자속)
                Laa = la_A * inv_ia
                Lab = la_B * inv_ia
                Lac = la_C * inv_ia
                
                # B상 루프 인덕턴스 (A상, B상, C상 여자 시 B상 쇄교자속)
                Lba = lb_A * inv_ia
                Lbb = lb_B * inv_ia
                Lbc = lb_C * inv_ia
                
                # C상 루프 인덕턴스 (A상, B상, C상 여자 시 C상 쇄교자속)
                Lca = lc_A * inv_ia
                Lcb = lc_B * inv_ia
                Lcc = lc_C * inv_ia
                
                current_records.append({
                    'Current_A': ia,
                    'Theta_Elec_deg': theta_e_deg,
                    'Theta_Mech_deg': theta_m_deg,
                    'Theta_Elec_rad': theta_e_rad,
                    'Laa': Laa, 'Lab': Lab, 'Lac': Lac,
                    'Lba': Lba, 'Lbb': Lbb, 'Lbc': Lbc,
                    'Lca': Lca, 'Lcb': Lcb, 'Lcc': Lcc
                })

        df_current = pd.DataFrame(current_records)
        df_current = df_current.sort_values(by='Theta_Elec_deg').reset_index(drop=True)

        # 1. 각 전류별 전체 인덕턴스 파형 플롯 저장
        plot_inductance_results(df_current, ia_val=ia_val, output_image_path=f"inductance_plot_{int(ia_val)}A.png")

        # 2. 각 성분별 요약 (중심값 및 진폭) 계산 후 추가
        summary_dict = {'Current_A': ia_val}
        for name in ['Laa', 'Lab', 'Lac', 'Lba', 'Lbb', 'Lbc', 'Lca', 'Lcb', 'Lcc']:
            vals = df_current[name] * 1e6
            summary_dict[f'{name}_Center_uH'] = np.mean(vals)
            summary_dict[f'{name}_Amplitude_uH'] = (np.max(vals) - np.min(vals)) / 2.0
            
        summary_records.append(summary_dict)

    # 모든 전류 계산 완료 후 요약 데이터 CSV 저장
    df_summary = pd.DataFrame(summary_records)
    summary_csv_filename = f"{base_fem_path}_inductance_matrix_summary.csv"
    df_summary.to_csv(summary_csv_filename, index=False, encoding="utf-8-sig")

    total_elapsed = time.time() - total_start_time
    print(f"\n==================================================")
    print(f" 모든 전류 조건 스윕 해석 총 소요 시간: {int(total_elapsed // 60)}분 {total_elapsed % 60:.2f}초")
    print(f" 3상 인덕턴스 매트릭스 요약 CSV 파일 저장 완료: '{summary_csv_filename}'")
    print(f"==================================================")

    return df_summary

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    df_summary = run_multi_current_sweep()