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



CIRCUIT_NAMES = ["A","B","C"]
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

def abc_to_dq(i_abc, b):
    # 전체를 감싸는 바깥쪽 대괄호([ ])가 반드시 있어야 합니다.
    clac_3x3 = 2/3 * np.array([
        [1, -1/2, -1/2], 
        [0, np.sqrt(3)/2, -np.sqrt(3)/2],
        [1/2, 1/2, 1/2]
    ])
    
    park_3x3 = np.array([
        [np.cos(b), np.sin(b), 0], 
        [-np.sin(b), np.cos(b), 0],
        [0, 0, 1]
    ])
    
    dq0 = park_3x3 @ clac_3x3 @ i_abc
    return dq0


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
    # ⏱️ 총 소요 시간 측정 시작
    start_time_global = time.time()
    
    kill_lingering_femm()

    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    print(f"==================================================")
    print(f" FEMM 단일 세션 유지 및 증분 회전(Delta Degree) FOC 제어 시작")
    print(f"==================================================")
    
    
    
    
    
    dt = 0.001    # 제어 주기 (1ms)
    J = 0.036      # 로터 관성 모멘트 [kg·m^2]
    load_torque = 0.0  # 부하 토크 [Nm]
    current_radpsec = 0.0
    total_steps = 100  # 실시간 연동 시뮬레이션 스텝 수
    
    # 데이터 기록용 리스트
    time_history = []
    speed_history = []
    target_speed_history = []
    iq_ref_history = []
    torque_history = []
    i_a_history = []
    i_b_history = []
    i_c_history = []
    v_a_history = []
    v_b_history = []
    v_c_history = []
    theta_m_histroy = []
    inductance = []

    
    working_fem_path = "model_working.fem"
    shutil.copy(base_fem_path, working_fem_path)
    
    session = FemmSession(port=48010, headless=False)

    try:
        
        poles = 8    # 극수
        pole_pairs = poles / 2  # 극쌍수 (4)
        
        Ld = 1
        Lq = 1
        
        Rd = 0.0064
        Rq = 0.0064
        pf = 0.273
        id=0.0
        iq=0.0
        theta_e_rad= 0.0
        theta_m_deg = 0.0
        theta_m_rad = 0.0
        omega_e_rad = 0.0
        omega_m_rad = 0.0
       
        #speed_pi = SpeedControllerPI(kp=0.5, ki=1, output_limits=(-340, 340))
        session.opendocument(working_fem_path)
        #session.mi_modifymaterial("Mag", 3, 0)
        load_torque = -10
        Vq = 0
        Vd = 0
        for step in range(total_steps):
            current_time = step * dt
        
            did_dt = 1/Ld * (Vd - Rd * id + omega_e_rad * Lq * iq)
            diq_dt = 1/Lq * (Vq - Rq * iq - omega_e_rad * (Ld * id + pf) )
                                    
            id = id + did_dt * dt
            iq = iq + diq_dt * dt

            out_d = Rd * id
            out_q = Rq * iq    

            i_a, i_b, i_c = dq_to_abc([id, iq, 0], theta_e_rad)
            v_a, v_b, v_c = dq_to_abc([out_d, out_q, 0], theta_e_rad)

            # 3. 삼상 전류 고정 설정
            session.mi_setcurrent('A', float(i_a))
            session.mi_setcurrent('B', float(i_b))
            session.mi_setcurrent('C', float(i_c))
            
            session.mi_modifyboundprop('slidingAirGap',10,theta_m_deg)
            # 5. FEMM 해석 실행 및 솔루션 로드
            session.mi_analyze(1)
            session.mi_loadsolution()
            
            # 6. 토크 추출 (그룹 1 블록 적분)
            session.mo_groupselectblock(1)
            raw_torque = session.mo_blockintegral(22)
            developed_torque = raw_torque * 8.0  # 스케일링 적용
            
            # 7. 모터 동역학 방정식 계산 및 다음 스텝 회전 각도(delta_deg) 산출
            accel_m_rad_s2 = (developed_torque - load_torque) / J       # 각가속도 
            omega_m_rad += accel_m_rad_s2 * dt                          # 각속도 (라디안/초)
            omega_e_rad = omega_m_rad * pole_pairs
            theta_m_rad += omega_m_rad * dt                              # 각도 (라디안)
            theta_e_rad = theta_m_rad * pole_pairs
            theta_m_deg = np.degrees(theta_m_rad)

            
            lamda = []
            current = []
            for c_name in CIRCUIT_NAMES:
                prop = session.mo_getcircuitproperties(c_name)
                lamda.append(prop[2])
                current.append(prop[0])

            lamdad, lamdaq,_ = abc_to_dq(lamda,theta_e_rad)
            currentd, currentq,_ = abc_to_dq(current,theta_e_rad)

            if currentd * currentq != 0 :
                Ld = lamdad / currentd
                Lq = lamdaq / currentq
            print(f"Ld:{Ld},Lq:{Lq}")



            # 데이터 기록
            time_history.append(current_time)
            speed_history.append(np.degrees(omega_m_rad))
            #target_speed_history.append(target_speed_rpm)
            iq_ref_history.append(iq)
            i_a_history.append(v_a)
            i_b_history.append(v_b)
            i_c_history.append(v_c)
            #v_a_history.apeend(v_ref)
            
            torque_history.append(developed_torque)  # 🛠️ 토크 데이터 올바르게 저장
            target_speed_history.append(theta_m_deg)
            theta_m_histroy.append(theta_m_deg)
            
            # ⚡ 출력 속도 최적화: 매 스텝 대신 10스텝마다 한 번씩 콘솔 출력
            #if (step + 1) % 10 == 0 or step == 0:
            print(f"[Step {step+1:04d}] 속도: {np.degrees(omega_m_rad):7.2f} RPM | iq: {iq:6.2f} A | id: {id:6.2f} A |theta(deg): {theta_m_deg:3.2f} deg | 토크: {developed_torque:6.2f} Nm")
                
    except Exception as e:
        print(f"[실시간 제어 루프 오류 발생]: {e}")
        import traceback
        traceback.print_exc()
        
    """ finally:
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
        print("실시간 제어 시뮬레이션 종료 및 임시 파일 정리 완료.") """

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

    # ⏱️ 총 소요 시간 계산 및 출력
    elapsed_time = time.time() - start_time_global
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"\n==================================================")
    if minutes > 0:
        print(f" ⏱️ 총 소요 시간: {minutes}분 {seconds:.2f}초 ({elapsed_time:.2f}초)")
    else:
        print(f" ⏱️ 총 소요 시간: {seconds:.2f}초")
    print(f"==================================================")

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
    
    # 4. 전자기 토크 그래프
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