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
                femm.mi_selectgroup(rotor_group_no)
                femm.mi_moverotate(0.0, 0.0, theta_m_deg)
            
            # 4. 해석 실행 및 솔루션 로드
            femm.mi_analyze(1)
            femm.mi_loadsolution()
            
            # 5. a, b, c 상 쇄교 자속 추출
            _, _, lambda_a = femm.mo_getcircuitproperties('A')
            _, _, lambda_b = femm.mo_getcircuitproperties('B')
            _, _, lambda_c = femm.mo_getcircuitproperties('C')
            
            results.append((theta_e_rad, theta_e_deg, theta_m_deg, ia_val, lambda_a, lambda_b, lambda_c))
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
    """특정 전류 조건의 인덕턴스 파형 결과 플롯 및 그래프 저장"""
    theta_deg = df_results['Theta_Elec_deg']
    laa = df_results['Laa'] * 1e6  # μH 단위 변환
    lab = df_results['Lab'] * 1e6
    lac = df_results['Lac'] * 1e6

    metrics = {}
    for name, val in [('Laa', laa), ('Lab', lab), ('Lac', lac)]:
        center = np.mean(val)
        amplitude = (np.max(val) - np.min(val)) / 2.0
        metrics[name] = {'center': center, 'amp': amplitude}

    plt.figure(figsize=(11, 7))

    plt.plot(theta_deg, laa, label='Laa (자기인덕턴스)', color='blue', lw=2)
    plt.plot(theta_deg, lab, label='Lab (상호인덕턴스 A-B)', color='green', lw=2)
    plt.plot(theta_deg, lac, label='Lac (상호인덕턴스 A-C)', color='orange', lw=2)

    colors = {'Laa': 'blue', 'Lab': 'green', 'Lac': 'orange'}
    annot_configs = {
        'Laa': {'x': 45,  'y_offset': 25},
        'Lab': {'x': 90,  'y_offset': -35},
        'Lac': {'x': 135, 'y_offset': 25}
    }

    for name, data in metrics.items():
        c = data['center']
        a = data['amp']
        plt.axhline(c, color=colors[name], linestyle='--', alpha=0.6, lw=1)
        
        cfg = annot_configs[name]
        annot_text = f"[{name}]\n중심: {c:.2f} μH\n진폭: {a:.2f} μH"
        
        plt.annotate(annot_text, 
                     xy=(cfg['x'], c), 
                     xytext=(cfg['x'], c + cfg['y_offset']),
                     arrowprops=dict(arrowstyle="->", color=colors[name], lw=1),
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=colors[name], alpha=0.9))

    plt.title(f"전류 {ia_val}A 조건 - 전기각에 따른 상 인덕턴스 프로파일", fontsize=13, fontweight='bold')
    plt.xlabel("전기각 [deg]", fontsize=11)
    plt.ylabel("인덕턴스 [μH]", fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"[전류 {ia_val}A 파형 플롯 저장 완료] {output_image_path}")

def run_multi_current_sweep():
    base_fem_path = "ioniq5-13.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    magnet_material_name = "Mag" 
    rotor_group_no = 1 
    pole_number = 8 
    pole_pairs = pole_number / 2 

    # 10A ~ 350A, 50A 간격 설정
    current_list = np.arange(0, 200, 50)
    theta_e_list = np.radians(np.arange(0, 360, 6))  # 0° ~ 360°, 6° 간격
    
    summary_records = []
    total_start_time = time.time()

    print(f"==================================================")
    print(f" [다중 전류 조건별 8극 모터 인덕턴스 스윕 해석]")
    print(f" 전류 범위: {current_list[0]}A ~ {current_list[-1]}A (간격: 50A)")
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
        
        # 결과 데이터 변환
        current_records = []
        for process_data in chunk_results:
            for theta_e_rad, theta_e_deg, theta_m_deg, ia, la, lb, lc in process_data:
                Laa = la / ia if abs(ia) > 1e-5 else 0.0
                Lab = lb / ia if abs(ia) > 1e-5 else 0.0
                Lac = lc / ia if abs(ia) > 1e-5 else 0.0
                
                current_records.append({
                    'Current_A': ia,
                    'Theta_Elec_deg': theta_e_deg,
                    'Theta_Mech_deg': theta_m_deg,
                    'Theta_Elec_rad': theta_e_rad,
                    'Lambda_a': la, 'Lambda_b': lb, 'Lambda_c': lc,
                    'Laa': Laa, 'Lab': Lab, 'Lac': Lac
                })

        df_current = pd.DataFrame(current_records)
        df_current = df_current.sort_values(by='Theta_Elec_deg').reset_index(drop=True)

        # 1. 각 전류별 계산 완료 즉시 파형 그래프 저장
        plot_inductance_results(df_current, ia_val=ia_val, output_image_path=f"inductance_plot_{int(ia_val)}A.png")

        # 2. 각 상별 중심값(Center)과 진폭(Amplitude) 계산 후 요약 리스트에 추가
        laa_vals = df_current['Laa'] * 1e6
        lab_vals = df_current['Lab'] * 1e6
        lac_vals = df_current['Lac'] * 1e6

        summary_records.append({
            'Current_A': ia_val,
            'Laa_Center_uH': np.mean(laa_vals),
            'Laa_Amplitude_uH': (np.max(laa_vals) - np.min(laa_vals)) / 2.0,
            'Lab_Center_uH': np.mean(lab_vals),
            'Lab_Amplitude_uH': (np.max(lab_vals) - np.min(lab_vals)) / 2.0,
            'Lac_Center_uH': np.mean(lac_vals),
            'Lac_Amplitude_uH': (np.max(lac_vals) - np.min(lac_vals)) / 2.0,
        })

    # 모든 전류 계산이 끝난 후 요약 데이터 CSV 저장
    df_summary = pd.DataFrame(summary_records)
    summary_csv_filename = f"{base_fem_path}_inductance_table.csv"
    df_summary.to_csv(summary_csv_filename, index=False, encoding="utf-8-sig")

    total_elapsed = time.time() - total_start_time
    print(f"\n==================================================")
    print(f" 모든 전류 조건 스윕 해석 총 소요 시간: {int(total_elapsed // 60)}분 {total_elapsed % 60:.2f}초")
    print(f" 각 전류별 중심값/진폭 요약 CSV 파일 저장 완료: '{summary_csv_filename}'")
    print(f"==================================================")

    return df_summary

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    df_summary = run_multi_current_sweep()