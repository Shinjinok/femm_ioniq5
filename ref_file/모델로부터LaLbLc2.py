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
    8극 모터의 로터 회전 및 고정 인가 해석 수행
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

            # 2. A상 고정 전류 설정, B상/C상은 0A
            femm.mi_setcurrent('A', ia_val)
            femm.mi_setcurrent('B', 0.0)
            femm.mi_setcurrent('C', 0.0)

            # 3. 회전자 기계각 회전 적용
            if rotor_group_no is not None and theta_m_deg != 0.0:
                femm.mi_seteditmode("group")
                femm.mi_clearselected()
                if isinstance(rotor_group_no, (list, tuple)):
                    for g_no in rotor_group_no:
                        femm.mi_selectgroup(g_no)
                else:
                    femm.mi_selectgroup(rotor_group_no)
                femm.mi_moverotate(0.0, 0.0, theta_m_deg)
            
            # 4. 해석 실행 및 솔루션 로드
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            
            # [추가] 각 해석(.ans 결과) 화면을 이미지(비트맵)로 저장
            # 뷰 옵션 설정 (예: 1번 플롯/밀도 플롯 표시 등, 필요시 커스텀 가능)
            try:
                ans_image_dir = f"ans_images_{int(ia_val)}A"
                os.makedirs(ans_image_dir, exist_ok=True)
                ans_image_path = os.path.abspath(os.path.join(ans_image_dir, f"flux_dist_theta_{int(theta_e_deg)}deg.png"))
                
                # 자속 밀도 맵 및 메쉬 등 표시 설정 후 비트맵 저장
                femm.mo_showdensityplot(1, 0, 2.0, 0.0, "b") # 자속밀도 컬러맵 표시
                femm.main_resize(1000, 1000)
                femm.mo_zoom(-100, -100, 100, 100)
                femm.mo_savebitmap(ans_image_path)
            except Exception as e:
                print(f"ANS 이미지 저장 중 오류 발생 ({theta_e_deg}deg): {e}")

            # 5. a, b, c 상 쇄교 자속 추출
            _, _, lambda_a = femm.mo_getcircuitproperties('A')
            _, _, lambda_b = femm.mo_getcircuitproperties('B')
            _, _, lambda_c = femm.mo_getcircuitproperties('C')

            results.append((theta_e_rad, theta_e_deg, theta_m_deg, ia_val, lambda_a * 8, lambda_b * 8, lambda_c * 8))
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

def plot_flux_linkage_results(df_results, ia_val, output_image_path="flux_linkage_plot.png"):
    """특정 전류 조건의 자속 쇄교수 파형 결과 플롯 및 그래프 저장"""
    theta_deg = df_results['Theta_Elec_deg']
    lam_a = df_results['Lambda_a'] * 1e3  # mWb 단위 변환
    lam_b = df_results['Lambda_b'] * 1e3
    lam_c = df_results['Lambda_c'] * 1e3

    metrics = {}
    for name, val in [('Lambda_a', lam_a), ('Lambda_b', lam_b), ('Lambda_c', lam_c)]:
        center = np.mean(val)
        amplitude = (np.max(val) - np.min(val)) / 2.0
        metrics[name] = {'center': center, 'amp': amplitude}

    plt.figure(figsize=(11, 7))

    plt.plot(theta_deg, lam_a, label='A상 자속 쇄교수 (Lambda_a)', color='blue', lw=2)
    plt.plot(theta_deg, lam_b, label='B상 자속 쇄교수 (Lambda_b)', color='green', lw=2)
    plt.plot(theta_deg, lam_c, label='C상 자속 쇄교수 (Lambda_c)', color='orange', lw=2)

    colors = {'Lambda_a': 'blue', 'Lambda_b': 'green', 'Lambda_c': 'orange'}
    annot_configs = {
        'Lambda_a': {'x': 45,  'y_offset': 0},
        'Lambda_b': {'x': 90,  'y_offset': 0},
        'Lambda_c': {'x': 135, 'y_offset': 0}
    }

    for name, data in metrics.items():
        c = data['center']
        a = data['amp']
        plt.axhline(c, color=colors[name], linestyle='--', alpha=0.6, lw=1)
        
        cfg = annot_configs[name]
        display_name = 'A상' if 'a' in name else ('B상' if 'b' in name else 'C상')
        annot_text = f"[{display_name}]\n중심: {c:.2f} mWb\n진폭: {a:.2f} mWb"
        
        plt.annotate(annot_text, 
                     xy=(cfg['x'], c), 
                     xytext=(cfg['x'], c + cfg['y_offset']),
                     arrowprops=dict(arrowstyle="->", color=colors[name], lw=1),
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=colors[name], alpha=0.9))

    plt.title(f"전류 {ia_val}A 조건 - 전기각에 따른 상 자속 쇄교수 프로파일", fontsize=13, fontweight='bold')
    plt.xlabel("전기각 [deg]", fontsize=11)
    plt.ylabel("자속 쇄교수 [mWb]", fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"[전류 {ia_val}A 자속 파형 플롯 저장 완료] {output_image_path}")

def run_multi_current_sweep():
    base_fem_path = "ioniq5-13.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    magnet_material_name = "Mag" 
    rotor_group_no = [1, 20]
    pole_number = 8 
    pole_pairs = pole_number / 2 

    current_list = list(np.arange(0, 341, 34))  # 테스트용 전류 설정 (필요시 확장)
    theta_e_list = np.radians(np.arange(0, 361, 4))  # 0° ~ 360°, 16° 간격
    
    summary_records = []
    total_start_time = time.time()

    print(f"==================================================")
    print(f" [다중 전류 조건별 8극 모터 자속 쇄교수 스윕 해석]")
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
        
        current_records = []
        for process_data in chunk_results:
            for theta_e_rad, theta_e_deg, theta_m_deg, ia, la, lb, lc in process_data:
                current_records.append({
                    'Current_A': ia,
                    'Theta_Elec_deg': theta_e_deg,
                    'Theta_Mech_deg': theta_m_deg,
                    'Theta_Elec_rad': theta_e_rad,
                    'Lambda_a': la, 
                    'Lambda_b': lb, 
                    'Lambda_c': lc
                })

        df_current = pd.DataFrame(current_records)
        df_current = df_current.sort_values(by='Theta_Elec_deg').reset_index(drop=True)

        # 1. 각 전류별 자속 파형 그래프 저장
        plot_flux_linkage_results(df_current, ia_val=ia_val, output_image_path=f"flux_linkage_plot_{int(ia_val)}A.png")

        # 2. 각 상별 자속 중심값과 진폭 계산 후 요약 리스트에 추가
        lam_a_vals = df_current['Lambda_a'] * 1e3
        lam_b_vals = df_current['Lambda_b'] * 1e3
        lam_c_vals = df_current['Lambda_c'] * 1e3

        summary_records.append({
            'Current_A': ia_val,
            'Lambda_a_Center_mWb': np.mean(lam_a_vals),
            'Lambda_a_Amplitude_mWb': (np.max(lam_a_vals) - np.min(lam_a_vals)) / 2.0,
            'Lambda_b_Center_mWb': np.mean(lam_b_vals),
            'Lambda_b_Amplitude_mWb': (np.max(lam_b_vals) - np.min(lam_b_vals)) / 2.0,
            'Lambda_c_Center_mWb': np.mean(lam_c_vals),
            'Lambda_c_Amplitude_mWb': (np.max(lam_c_vals) - np.min(lam_c_vals)) / 2.0,
        })

    df_summary = pd.DataFrame(summary_records)
    summary_csv_filename = f"{base_fem_path}_flux_linkage_table.csv"
    df_summary.to_csv(summary_csv_filename, index=False, encoding="utf-8-sig")

    total_elapsed = time.time() - total_start_time
    print(f"\n==================================================")
    print(f" 모든 전류 조건 스윕 해석 총 소요 시간: {int(total_elapsed // 60)}분 {total_elapsed % 60:.2f}초")
    print(f" 각 전류별 자속 중심값/진폭 요약 CSV 파일 저장 완료: '{summary_csv_filename}'")
    print(f"==================================================")

    return df_summary

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    df_summary = run_multi_current_sweep()