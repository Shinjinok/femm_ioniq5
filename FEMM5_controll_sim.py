import os
import shutil
import time
import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt
from femm import FemmSession

# 한글 폰트 설정 및 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

poles = 8   # 극수
pole_pairs = poles / 2  # 극쌍수 (4)

def kill_lingering_femm():
    """기존에 열려 있거나 좀비로 남은 FEMM 프로세스들을 강제 종료"""
    for proc_name in ["femm.exe", "wfemm.exe", "femm_gui.exe"]:
        try:
            subprocess.run(["taskkill", "/f", "/im", proc_name], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(0.5)

def dq_to_abc(dq0, b):
    clac2 = np.array([[1, -1/2, -1/2],
                    [0, np.sqrt(3)/2, -np.sqrt(3)/2],
                    [1/2, 1/2, 1/2]])
    park_3x3 = np.array([
                [np.cos(b), np.sin(b), 0],
                [-np.sin(b), np.cos(b), 0],
                [0, 0, 1]
            ])
    abc = clac2.T @ park_3x3.T @ dq0
    return abc

class SpeedControllerPI:
    """모터 속도 제어를 위한 PI 제어기"""
    def __init__(self, kp, ki, output_limits=(-340, 340)):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.min_out, self.max_out = output_limits

    def compute(self, target_speed, current_speed, dt):
        error = (target_speed - current_speed) # rad/s
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.min_out / (self.ki + 1e-6), self.max_out / (self.ki + 1e-6))
        
        output = (self.kp * error) + (self.ki * self.integral)
        return np.clip(output, self.min_out, self.max_out)

def run_realtime_femm_foc_simulation():
    kill_lingering_femm()

    base_fem_path = "ioniq5-13.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    print(f"==================================================")
    print(f" FEMM 단일 세션 유지 및 증분 회전(Delta Degree) FOC 제어 시작")
    print(f"==================================================")
    
    target_speed_rpm = 36   # 목표 속도 [RPM]
    target_radpsec = target_speed_rpm /60 * 2* np.pi

    current_speed_rpm = 0.0    # 초기 속도 [RPM]
    current_radpsec = current_speed_rpm /60 * 2 * np.pi
    
    theta_mech_cum = 0.0       # 누적 기계각 [°] (계속 증가)
    delta_deg = 0.0            # 이번 스텝에서 회전할 증분 각도 [°]
    
    speed_pi = SpeedControllerPI(kp=2, ki=5.0, output_limits=(-340, 340))
    
    dt = 0.001    # 제어 주기 (1ms)
    J = 0.01      # 로터 관성 모멘트 [kg·m^2]
    load_torque = 0.0  # 부하 토크 [Nm]
    
    total_steps = 36  # 실시간 연동 시뮬레이션 스텝 수
    
    # 데이터 기록용 리스트
    time_history = []
    speed_history = []
    target_speed_history = []
    iq_ref_history = []
    torque_history = []
    i_a_history = []
    i_b_history = []
    i_c_history = []
    
    working_fem_path = "model_working.fem"
    shutil.copy(base_fem_path, working_fem_path)
    
    session = FemmSession(port=48010, headless=False)
    
    try:
        session.opendocument(working_fem_path)
        
        for step in range(total_steps):
            loop_start_time = time.time()
            current_time = step * dt
            
            # 1. 속도 PI 제어기 계산 (q축 기준 전류 iq_ref 산출)
            iq_ref = speed_pi.compute(target_radpsec, current_radpsec, dt)
            id_ref = 0.0
            
            # 2. 전기각 계산 및 역 Park-Clarke 변환으로 삼상 전류 계산
            # 전기각(Electrical Angle) = 기계각 * 극쌍수(pole_pairs)
            theta_elec_deg = (theta_mech_cum * pole_pairs) % 360.0
            theta_rad = np.radians(theta_elec_deg) % np.pi
            i_a, i_b, i_c = dq_to_abc([id_ref, iq_ref, 0], theta_rad)
            
            # 3. 삼상 전류 고정 설정
            session.mi_setcurrent('A', float(i_a))
            session.mi_setcurrent('B', float(i_b))
            session.mi_setcurrent('C', float(i_c))
            
            # 4. 증분 회전 적용 (반드시 누적 각도가 아닌 증분 각도인 delta_deg 전달)
            
            if abs(delta_deg) > 1e-5:
                session.mi_clearselected()
                session.mi_selectgroup(1)
                session.mi_selectgroup(20)
                session.mi_moverotate(0, 0, delta_deg)

            # 5. FEMM 해석 실행 및 솔루션 로드
            session.mi_analyze(1)
            session.mi_loadsolution()
            
            # 6. 토크 추출 (그룹 1 블록 적분)
            session.mo_clearblock()
            session.mo_groupselectblock(1)
            raw_torque = session.mo_blockintegral(22)
            developed_torque = raw_torque * 8.0  # 스케일링 적용
            
            # 7. 모터 동역학 방정식 계산 및 다음 스텝 회전 각도(delta_deg) 산출
            accel_rad_s2 = (developed_torque - load_torque) / J
            current_radpsec = accel_rad_s2 * dt
            
            # [핵심 수정] RPM을 각도(Degree)로 변환: RPM * (360 / 60) * dt = RPM * dt * 6.0
            current_rad = current_radpsec * dt
            delta_deg = current_rad / (2*np.pi) *360
            theta_mech_cum += delta_deg
            
            # 데이터 기록
            time_history.append(current_time)
            speed_history.append(current_speed_rpm)
            target_speed_history.append(target_speed_rpm)
            iq_ref_history.append(iq_ref)
            i_a_history.append(i_a)
            i_b_history.append(i_b)
            i_c_history.append(i_c)
            torque_history.append(developed_torque)
            
            print(f"[Step {step+1:02d}] 속도: {current_speed_rpm:7.2f} RPM | iq_ref: {iq_ref:6.2f} A | 전류(Va): {i_a:6.2f} A | 토크: {developed_torque:6.2f} Nm")
                 
    except Exception as e:
        print(f"[실시간 제어 루프 오류 발생]: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        try:
            session.close()
        except:
            pass
            
        for ext in ['.fem', '.ans', '.anh', '.log']:
            t_file = working_fem_path.replace('.fem', ext)
            if os.path.exists(t_file):
                try:
                    os.remove(t_file)
                except:
                    pass
        kill_lingering_femm()
        print("실시간 제어 시뮬레이션 종료 및 임시 파일 정리 완료.")

    # --- CSV 파일로 결과 저장 ---
    csv_filename = "foc_simulation_results.csv"
    print(f"\n[CSV 파일 저장 중: {csv_filename}]")
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time [s]', 'Actual Speed [RPM]', 'Target Speed [RPM]', 
                'iq_ref [A]', 'Phase A Current [A]', 'Phase B Current [A]', 
                'Phase C Current [A]', 'Torque [Nm]'
            ])
            for t, sp, tsp, iq, va, vb, vc, tq in zip(
                time_history, speed_history, target_speed_history, 
                iq_ref_history, i_a_history, i_b_history, i_c_history, torque_history
            ):
                writer.writerow([t, sp, tsp, iq, va, vb, vc, tq])
        print(f"CSV 저장 완료: {os.path.abspath(csv_filename)}")
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")

    # --- 시뮬레이션 결과 그래프 표시 ---
    print("\n[그래프 생성 중...]")
    plt.figure(figsize=(10, 10))
    
    # 1. 속도 그래프
    plt.subplot(4, 1, 1)
    plt.plot(time_history, speed_history, label='Actual Speed (RPM)', color='blue', linewidth=2)
    plt.plot(time_history, target_speed_history, label='Target Speed (RPM)', color='red', linestyle='--', linewidth=1.5)
    plt.ylabel('Speed [RPM]')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('FOC Real-time Simulation Results (Incremental Rotation)')
    
    # 2. 제어 입력 전류 그래프
    plt.subplot(4, 1, 2)
    plt.plot(time_history, iq_ref_history, label='q-axis Current Ref (iq_ref)', color='green', linewidth=2)
    plt.ylabel('Input Current [A]')
    plt.legend(loc='best')
    plt.grid(True)
    
    # 3. 3상 전류 그래프 (사인파 형태 확인)
    plt.subplot(4, 1, 3)
    plt.plot(time_history, i_a_history, label='Phase A Current ($I_a$)', color='red', linewidth=1.5)
    plt.plot(time_history, i_b_history, label='Phase B Current ($I_b$)', color='green', linewidth=1.5)
    plt.plot(time_history, i_c_history, label='Phase C Current ($I_c$)', color='blue', linewidth=1.5)
    plt.ylabel('Current [A]')
    plt.legend(loc='best')
    plt.grid(True)
    
    # 4. 축력(전자기 토크) 그래프
    plt.subplot(4, 1, 4)
    plt.plot(time_history, torque_history, label='Developed Torque (Shaft)', color='purple', linewidth=2)
    plt.axhline(y=load_torque, color='orange', linestyle='--', label=f'Load Torque ({load_torque} Nm)')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    kill_lingering_femm()
    run_realtime_femm_foc_simulation()