import os
import shutil
import time
import subprocess
import csv
from matplotlib.pylab import beta
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
        if user32.IsWindowVisible(hwnd) and user32.GetParent(hwnd) == 0:
            exe_path = get_process_name_from_hwnd(hwnd)
            if exe_path.endswith(("femm_gui.exe", "wfemm.exe")):
                found_hwnds.append(hwnd)
        return True

    user32.EnumWindows(enum_windows_proc, 0)

    if not found_hwnds:
        print("[경고] FEMM 프로세스 창을 찾지 못했습니다.")
        return False

    for hwnd in found_hwnds:
        user32.ShowWindow(hwnd, SW_RESTORE)
        time.sleep(0.05)
        user32.SetWindowPos(
            hwnd, 0, int(x), int(y), int(width), int(height), 
            SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW
        )
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
    sr32 = np.sqrt(3)/2
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
    sr32 = np.sqrt(3)/2
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

class SpeedControllerPI:
    """모터 속도 제어를 위한 PI 제어기"""
    def __init__(self, kp, ki, output_limits=(-340, 340)):
        self.kp = kp
        self.ki = ki
        self.integral = 0.0
        self.min_out, self.max_out = output_limits

    def compute(self, target_speed, current_speed, dt):
        error = (target_speed - current_speed)
        self.integral += error * dt
        self.integral = np.clip(self.integral, self.min_out / (self.ki + 1e-6), self.max_out / (self.ki + 1e-6))
        
        output = (self.kp * error) + (self.ki * self.integral)
        return np.clip(output, self.min_out, self.max_out)

def run_femm(session, Iabc, theta_mech_cum):
    session.mi_setcurrent('A', float(Iabc[0]))
    session.mi_setcurrent('B', float(Iabc[1]))
    session.mi_setcurrent('C', float(Iabc[2]))
    
    session.mi_modifyboundprop('slidingAirGap', 10, theta_mech_cum)
    
    session.mi_analyze(1)
    session.mi_loadsolution()
    
    session.mo_groupselectblock(1)
    raw_torque = session.mo_blockintegral(22)
    developed_torque = raw_torque * 8.0  # 스케일링 적용
    session.mo_showvectorplot(1, 100)

    current = []
    lamda = []
    for c_name in CIRCUIT_NAMES:
        prop = session.mo_getcircuitproperties(c_name)
        current.append(prop[0])  # Index 0 : Current
        lamda.append(prop[2]  * pole_pairs * 2)
    
    return np.array(lamda), np.array(current), developed_torque

def run_realtime_femm_foc_simulation():
    start_time_global = time.time()
    kill_lingering_femm()

    base_fem_path = "ioniq5-14.FEM"
    if not os.path.exists(base_fem_path):
        raise FileNotFoundError(f"기준 모델 파일을 찾을 수 없습니다: {base_fem_path}")

    print(f"==================================================")
    print(f" FEMM IPMSM 1/8 슬라이딩 바운더리 모델 FOC 제어 시작")
    print(f"==================================================")
    
    time_history = []
    speed_history = []
    target_speed_history = []
    vq_ref_history = []
    torque_history = []
    torque_history2 = []
    torque_history3 = []
    i_a_history = []
    i_b_history = []
    i_c_history = []
    v_a_history = []
    v_b_history = []
    v_c_history = []
    
    working_fem_path = "model_working.fem"
    shutil.copy(base_fem_path, working_fem_path)
    
    session = FemmSession(port=48010, headless=HEADLESS_MODE)
    try:
        dt = 0.001
        total_steps = 100  # 빠른 시뮬레이션을 위해 조절 가능

        session.opendocument(working_fem_path)
        cos_130 = np.cos(130*np.pi/180)
        sin_130 = np.sin(130*np.pi/180)
        
        J = 0.036
        Rs = 0.0618
        fm = 0.016497753*8
        current_speed_rpm = 0.0
        current_radpsec = 0.0
        theta_mech_cum = 0.0
        delta_deg = 0.0
        load_torque = 100
        target_speed_rpm = 200
        
        lamda_old = np.zeros(3)
        Vabc = np.zeros(3)
        Iabc = np.zeros(3)
        speed_pi = SpeedControllerPI(kp=0.5, ki=10, output_limits=(-340, 340))
        lamda_old, _, _ = run_femm(session, Iabc, theta_mech_cum)
        id = 0.0
        iq = 0.0
        id_old = 0.0
        iq_old = 0.0
        
        for step in range(total_steps):
            current_time = step * dt
            
            i_ref = speed_pi.compute(target_speed_rpm, current_speed_rpm, dt)
            iq_ref = sin_130 * i_ref
            id_ref = cos_130 * i_ref


            
            theta_elec_deg = (theta_mech_cum * pole_pairs) 
            theta_elec_rad = np.radians(theta_elec_deg)
            we = current_radpsec * pole_pairs

            we = target_speed_rpm * 2 * np.pi / 60 * pole_pairs

            Iabc = dq_to_abc2(np.array([id_ref, iq_ref, 0]), theta_elec_rad)
            Iabc = np.zeros(3)  # 테스트용으로 전류를 0으로 설정 (실제 시뮬레이션에서는 제어기 출력에 따라 Iabc를 계산해야 함)
            
            lamda, current, developed_torque = run_femm(session, Iabc, theta_mech_cum)

            

            dlamda_dt = (lamda - lamda_old) / dt
            Vabc = Rs * Iabc + dlamda_dt
            lamda_old = lamda.copy()

            lamda_d, lamda_q, lamda_0 = abc_to_dq(lamda, theta_elec_rad)

            id, iq, i0 = abc_to_dq(current, theta_elec_rad)
            Ld = (lamda_d - fm) / id if id != 0 else 1
            Lq = lamda_q / iq if iq != 0 else 1

            """ vd_ref = 2 * Rs * id_ref
            vq_ref = 2 * Rs * iq_ref

            id_dt = (vd_ref - Rs * id + we * lamda_q ) / Ld
            iq_dt = (vq_ref - Rs * iq - we * lamda_d) / Lq

            id += id_dt * dt
            iq += iq_dt * dt """
            vd_ref =  Rs * id + Ld * (id - id_old) / dt - we * Lq * iq
            vq_ref =  Rs * iq + Lq * (iq - iq_old) / dt + we * Ld * id + we * fm

            id_old = id
            iq_old = iq

            Vabc2 = dq_to_abc2(np.array([vd_ref, vq_ref, 0]), theta_elec_rad)

           
            Te_vector = 1.5 * pole_pairs * np.cross([lamda_d, lamda_q, 0], [id, iq, 0])  # 전자기 토크 계산 (벡터 연산)

            s= np.array([-np.sin(-theta_elec_rad), -np.sin(theta_elec_rad - 2/3*np.pi), -np.sin(theta_elec_rad + 2/3*np.pi)])

            s2t = np.sin(2*theta_elec_rad)
            s2t_m23p = np.sin(2*theta_elec_rad - 2/3*np.pi)
            s2t_p23p = np.sin(2*theta_elec_rad + 2/3*np.pi)
            s2t_2pi = np.sin(2*theta_elec_rad + 2*np.pi)
            Lb = 0.0034
            dLsdt = 2* Lb * np.array([[s2t, s2t_m23p, s2t_p23p],
                                    [s2t_m23p, s2t_p23p, s2t_2pi],
                                    [s2t_p23p, s2t_2pi, s2t_m23p]])
           
            Te_vector2 = 0.5 * current @ dLsdt @ current.T #+ fm * s @ current.T  # 전자기 토크 계산 (벡터 연산)
            
            accel_rad_s2 = (developed_torque - load_torque) / J
            current_radpsec += accel_rad_s2 * dt
            current_rad = current_radpsec * dt

            current_radpsec = target_speed_rpm * 2 * np.pi / 60
            current_speed_rpm = current_radpsec * 30 / np.pi 
            delta_deg = current_rad * 180 / np.pi

            delta_deg = target_speed_rpm * 6 * dt

            theta_mech_cum += delta_deg

            v_a_history.append(Vabc[0])
            v_b_history.append(Vabc[1])
            v_c_history.append(Vabc[2])
            
            time_history.append(current_time)
            speed_history.append(current_speed_rpm)
            target_speed_history.append(target_speed_rpm)

            i_a_history.append(Vabc2[0])
            i_b_history.append(Vabc2[1])
            i_c_history.append(Vabc2[2])
            
            torque_history.append(developed_torque)
            torque_history2.append(Te_vector[2])  # q축 전자기 토크 기록
            torque_history3.append(Te_vector2)  # d축 전자기 토크 기록
            print(f"[Step {step+1:04d}] 속도: {current_speed_rpm:7.2f} RPM | v_a: {Vabc[0]:6.2f} V | 토크: {developed_torque:6.2f} Nm")
                
    except Exception as e:
        print(f"[실시간 제어 루프 오류 발생]: {e}")
        import traceback
        traceback.print_exc()

    

    # --- 시뮬레이션 결과 그래프 표시 (속도-토크 곡선 포함) ---
    print("\n[그래프 생성 중...]")
    fig = plt.figure(figsize=(12, 12))
    
    # 1. 속도 그래프
    plt.subplot(5, 1, 1)
    plt.plot(time_history, speed_history, label='Actual Speed (RPM)', color='blue', linewidth=2)
    plt.plot(time_history, target_speed_history, label='Target Speed (RPM)', color='red', linestyle='--', linewidth=1.5)
    plt.ylabel('Speed [RPM]')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('FOC Real-time Simulation & Speed-Torque Curve')
    
    # 2. 제어 입력 전류 그래프
    plt.subplot(5, 1, 2)
    plt.plot(time_history, v_a_history, label='Phase A Voltage ($V_a$)', color='red', linewidth=1.5)
    plt.plot(time_history, i_a_history, label='Phase A2 Voltage ($I_a$)', color='green', linewidth=1.5)
    
    plt.ylabel('Voltage [V]')
    plt.legend(loc='best')
    plt.grid(True)
    
    # 3. 3상 전압 그래프
    plt.subplot(5, 1, 3)
    plt.plot(time_history, v_b_history, label='Phase B Voltage ($V_a$)', color='red', linewidth=1.5)
    plt.plot(data:=time_history, i_b_history, label='Phase B2 Voltage ($V_b$)', color='green', linewidth=1.5)
    
    plt.ylabel('Voltage [V]')
    plt.legend(loc='best')
    plt.grid(True)
    
    # 4. 전자기 토크(시간축) 그래프
    plt.subplot(5, 1, 4)
    plt.plot(time_history, v_c_history, label='Phase C Voltage ($V_a$)', color='red', linewidth=1.5)
    plt.plot(data:=time_history, i_c_history, label='Phase C2 Voltage ($V_c$)', color='green', linewidth=1.5)
    
    plt.ylabel('Voltage [V]')
    plt.legend(loc='best')
    plt.grid(True)
    

    # 5. [추가] 스피드-토크 곡선 (Speed-Torque Curve) 그래프
    plt.subplot(5, 1, 5)
    plt.plot(time_history, torque_history,  color='red', linewidth=2, label='WST Torque [Nm]')
    plt.plot(time_history, torque_history2, color='blue', linewidth=2, label='dq cross Torque [Nm]')
    plt.plot(time_history, torque_history3,  color='green', linewidth=2, label='energy differential Torque [Nm]')
    plt.ylabel('Torque [Nm]')
    plt.legend(loc='best')
    plt.grid(True)

    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    kill_lingering_femm()
    run_realtime_femm_foc_simulation()