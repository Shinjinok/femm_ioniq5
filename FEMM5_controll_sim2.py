import os
import shutil
import time
import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt
from femm import FemmSession
import ctypes
from ctypes import wintypes

# 한글 폰트 설정 및 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

poles = 8    # 극수
pole_pairs = poles / 2  # 극쌍수 (4)
CIRCUIT_NAMES = ["A","B","C"]
# 2. Windows API 함수 로드
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
HEADLESS_MODE = True  # True: GUI 없이 FEMM 실행, False: GUI 모드로 FEMM 실행

SW_RESTORE = 9
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

def get_process_name_from_hwnd(hwnd):
    """윈도우 핸들(HWND)로부터 실행 중인 실제 .exe 파일 경로를 조회"""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == 0:
        return ""
    
    # PID를 통해 프로세스 열기
    h_process = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h_process:
        return ""
    
    exe_buf = ctypes.create_unicode_buffer(1024)
    size = wintypes.DWORD(len(exe_buf))
    success = kernel32.QueryFullProcessImageNameW(h_process, 0, exe_buf, ctypes.byref(size))
    kernel32.CloseHandle(h_process)
    
    return exe_buf.value.lower() if success else ""

def set_femm_window_size(width=400, height=300, x=100, y=100):
    found_hwnds = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_windows_proc(hwnd, lparam):
        # 화면에 보이는 최상위 창인지 확인
        if user32.IsWindowVisible(hwnd) and user32.GetParent(hwnd) == 0:
            exe_path = get_process_name_from_hwnd(hwnd)
            # 타이틀 대신 프로세스 파일명이 femm.exe / wfemm.exe 인지 직접 확인
            if exe_path.endswith(("femm_gui.exe", "wfemm.exe")):
                found_hwnds.append(hwnd)
        return True

    user32.EnumWindows(enum_windows_proc, 0)

    if not found_hwnds:
        print("[경고] FEMM 프로세스(femm.exe/wfemm.exe)의 창을 찾지 못했습니다.")
        return False

    for hwnd in found_hwnds:
        # 최대화 상태 해제 후 지정 좌표/크기로 강제 설정
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.05)
        user32.SetWindowPos(
            hwnd, 
            0, 
            int(x), int(y), int(width), int(height), 
            SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW
        )
        print(f"[성공] FEMM 프로세스 창 크기 조절 완료: {width}x{height} (위치: x={x}, y={y})")

    return True

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
    sr32= np.sqrt(3)/2
    clac2 = np.array([[1, -1/2, -1/2],
                    [0, sr32, -sr32],
                    [1/2, 1/2, 1/2]])
    cb = np.cos(b)
    sb = np.sin(b)   
    park_3x3 = np.array([
                [cb, sb, 0],
                [-sb, cb, 0],
                [0, 0, 1]
            ])
    abc = clac2.T @ park_3x3.T @ dq0
    return abc
def dq_to_abc2(dq0, b):

    T = 2/3 * np.array([np.cos(b), np.cos(b - 2*np.pi/3), np.cos(b + 2*np.pi/3),
                            -np.sin(b), -np.sin(b - 2*np.pi/3), -np.sin(b + 2*np.pi/3),
                            1/2, 1/2, 1/2]).reshape(3, 3)
    abc = np.linalg.inv(T) @ dq0
    return abc
def abc_to_dq(i_abc, b):
    # 전체를 감싸는 바깥쪽 대괄호([ ])가 반드시 있어야 합니다.
    sr32= np.sqrt(3)/2
    clac_3x3 = 2/3 * np.array([
        [1, -1/2, -1/2], 
        [0, sr32, -sr32],
        [1/2, 1/2, 1/2]
    ])
    cb = np.cos(b)
    sb = np.sin(b)
    park_3x3 = np.array([
        [cb, sb, 0], 
        [-sb, cb, 0],
        [0, 0, 1]
    ])

    dq0 = park_3x3 @ clac_3x3 @ i_abc
    return dq0
def abc_to_dq2(i_abc, b):
    # 전체를 감싸는 바깥쪽 대괄호([ ])가 반드시 있어야 합니다.
    T = 2/3 * np.array([np.cos(b), np.cos(b - 2*np.pi/3), np.cos(b + 2*np.pi/3),
                        -np.sin(b), -np.sin(b - 2*np.pi/3), -np.sin(b + 2*np.pi/3),
                        1/2, 1/2, 1/2]).reshape(3, 3)
    dq0 = T @ i_abc

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

def run_femm(session, Iabc, theta_mech_cum):
    # 1. 삼상 전류 입력
    session.mi_setcurrent('A', float(Iabc[0]))
    session.mi_setcurrent('B', float(Iabc[1]))
    session.mi_setcurrent('C', float(Iabc[2]))
    
    # 2. 증분 회전 적용
    session.mi_modifyboundprop('slidingAirGap',10,theta_mech_cum)
    
    # 3. FEMM 해석 실행 및 솔루션 로드
    session.mi_analyze(1)
    session.mi_loadsolution()
    
    # 4. 토크 추출 (그룹 1 블록 적분)
    session.mo_groupselectblock(1)
    raw_torque = session.mo_blockintegral(22)
    developed_torque = raw_torque * 8.0  # 스케일링 적용
    #session.mo_clearblock()
    session.mo_showvectorplot(1, 100)

    current = []
    lamda = []
    for c_name in CIRCUIT_NAMES:
        prop = session.mo_getcircuitproperties(c_name)
        current.append(prop[0])  # Index 0 : Current
        #voltage_drop = prop[1]  # Index 1 : Voltage Drop
        lamda.append(prop[2] * 8)  # Index 2 : Flux Linkage
        #Ls.append(prop[2] / prop[0])  # Inductance 계산 (H) = Flux Linkage / Current
    
    return np.array(lamda), np.array(current), developed_torque
     

def run_realtime_femm_foc_simulation():
    # ⏱️ 총 소요 시간 측정 시작
    start_time_global = time.time()
    
    kill_lingering_femm()

    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    print(f"==================================================")
    print(f" FEMM IPMSM 1/8 슬라이딩 바운더리 모델 FOC 제어 시작")
    print(f"==================================================")
    
    
    # 데이터 기록용 리스트
    time_history = []
    speed_history = []
    target_speed_history = []
    vq_ref_history = []
    torque_history = []
    i_a_history = []
    i_b_history = []
    i_c_history = []
    v_a_history = []
    v_b_history = []
    v_c_history = []
    theta_m_histroy = []
    
    
    
    

    inductance_history = {c: 0.0 for c in CIRCUIT_NAMES}
    working_fem_path = "model_working.fem"
    shutil.copy(base_fem_path, working_fem_path)
    
    session = FemmSession(port=48010, headless=HEADLESS_MODE)
    theta_elec_deg = 0
    try:
        dt = 0.001    # 제어 주기 (1ms)
        
        total_steps = 500  # 실시간 연동 시뮬레이션 스텝 수

        session.opendocument(working_fem_path)
        #time.sleep(0.5)  # FEMM GUI가 완전히 로드될 시간을 확보
        # 400x300 크기로 조정
        #set_femm_window_size(width=480, height=480, x=0, y=0)
        cos_130 = np.cos(130*np.pi/180)
        sin_130 = np.sin(130*np.pi/180)
        
        
        J = 0.036      # 로터 관성 모멘트 [kg·m^2]
        Rs = 0.0618  # 선간저항
        fm = 0.016497753*8 #np.array([0.016497753, -0.0086603404, -0.0086541372])  # 예: 각 상의 전기자 반작용 계수 (V·s/A) 가정

        current_speed_rpm = 0.0    # 초기 속도 [RPM]
        current_radpsec = 0.0    
        theta_mech_cum = 0.0       # 누적 기계각 [°] (계속 증가)
        delta_deg = 0.0            # 이번 스텝에서 회전할 증분 각도 [°]
        load_torque = 300.0  # 부하 토크 [Nm]
        target_speed_rpm = 100   # 목표 속도 [RPM]
        lamda_old = np.zeros(3)  # 각 상의 자속 링크 [Wb]
        Vabc = np.zeros(3)  # 각 상의 전압 [V]
        Iabc = np.zeros(3)  # 다음 스텝에서의 전류 [A]
        dlamda_dt = np.zeros(3)  # 자속 링크의 시간 변화율 [Wb/s]
        speed_pi = SpeedControllerPI(kp=0.5, ki=10, output_limits=(-340, 340))
        lamda_old, _, _ = run_femm(session, Iabc, theta_mech_cum)

        for step in range(total_steps):
            current_time = step * dt
            
            # 1. 속도 PI 제어기 계산 (q축 기준 전류 iq_ref 산출)
            i_ref = speed_pi.compute(target_speed_rpm, current_speed_rpm, dt)
            iq_ref = sin_130 * i_ref
            id_ref = cos_130 * i_ref
            
            # 2. 전기각 계산 
            theta_elec_deg = (theta_mech_cum * pole_pairs) 
            theta_elec_rad = np.radians(theta_elec_deg)

            #제어입력
            Iabc = dq_to_abc2(np.array([id_ref, iq_ref, 0]), theta_elec_rad)
            
            lamda, _, developed_torque = run_femm(session, Iabc, theta_mech_cum)
            dlamda_dt = (lamda - lamda_old) / dt
            Vabc = Rs*Iabc + dlamda_dt
            lamda_old = lamda


            # 7. 모터 동역학 방정식 계산 및 다음 스텝 회전 각도(delta_deg) 산출
            accel_rad_s2 = (developed_torque - load_torque) / J
            current_radpsec += accel_rad_s2 * dt

            #current_radpsec = 141 * 2 * np.pi / 60  # 목표 속도 [rad/s]로 강제 설정

            current_rad = current_radpsec * dt
            current_speed_rpm = current_radpsec * 30 /np.pi 
            delta_deg = current_rad * 180 / np.pi
            theta_mech_cum += delta_deg


            # 데이터 기록
            v_a_history.append(Vabc[0])
            v_b_history.append(Vabc[1])
            v_c_history.append(Vabc[2])
            
            time_history.append(current_time)
            speed_history.append(current_speed_rpm)
            target_speed_history.append(target_speed_rpm)

            i_a_history.append(Iabc[0])
            i_b_history.append(Iabc[1])
            i_c_history.append(Iabc[2])
            
            torque_history.append(developed_torque)  # 🛠️ 토크 데이터 올바르게 저장
            theta_m_histroy.append(theta_mech_cum)
            
            print(f"[Step {step+1:04d}] 속도: {current_speed_rpm:7.2f} RPM | v_a: {Vabc[0]:6.2f} V | delta_deg: {delta_deg:3.2f} deg | 토크: {developed_torque:6.2f} Nm")
                
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
                'vq_ref [V]', 'Phase A Current [A]', 'Phase B Current [A]', 
                'Phase C Current [A]', 'Torque [Nm]'
            ])
            for t, sp, tsp, vq, va, vb, vc, tq in zip(
                time_history, speed_history, target_speed_history, 
                vq_ref_history, i_a_history, i_b_history, i_c_history, torque_history
            ):
                writer.writerow([t, sp, tsp, vq, va, vb, vc, tq])
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
    plt.plot(time_history, i_a_history, label='Phase A Current (I_a)', color='red', linewidth=2)
    plt.plot(time_history, i_b_history, label='Phase B Current (I_b)', color='green', linewidth=2)
    plt.plot(time_history, i_c_history, label='Phase C Current (I_c)', color='blue', linewidth=2)
    plt.ylabel('Current [A]')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Input Current (Iabc) vs Time')
    
    # 3. 3상 전류 그래프 (사인파 형태 확인)
    plt.subplot(4, 1, 3)
    
    plt.plot(time_history, v_a_history, label='Phase A Voltage ($V_a$)', color='red', linewidth=1.5)
    plt.plot(time_history, v_b_history, label='Phase B Voltage ($V_b$)', color='green', linewidth=1.5)
    plt.plot(time_history, v_c_history, label='Phase C Voltage ($V_c$)', color='blue', linewidth=1.5)
    plt.ylabel('Voltage [V]')
    plt.legend(loc='best')
    plt.grid(True)
    
    # 4. 전자기 토크 그래프
    plt.subplot(4, 1, 4)
    plt.plot(time_history, torque_history, label='Developed Torque (Shaft)', color='red', linewidth=2)
    #plt.plot(time_history, torque_history1, label='Calculated Torque1 (Shaft)', color='green', linewidth=2)
    #plt.plot(time_history, torque_history2, label='Calculated Torque2 (Shaft)', color='blue', linewidth=2)
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