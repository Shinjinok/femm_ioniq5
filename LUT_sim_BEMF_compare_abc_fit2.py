import os
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from iq5model import iq5model 

# 한글 폰트 및 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


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


def run_bemf_simulation_at_141rpm():
    
    iq5 = iq5model(lut="LUT_lamda_abcT_20261119_091223.csv")
    dt = 0.0001
    
    total_time = 2
    target_speed_rpm = 141 # 목표 BEMF 측정 속도 [RPM]

    #total_time = 1/(target_speed_rpm*4/60)    # 총 시뮬레이션 시간 [s]
    #total_steps = 1000
    #dt = total_time/total_steps
    #print(total_time, dt)
    total_steps = int(total_time / dt)
    
    
    # 히스토리 기록용 리스트
    time_history = []
    speed_history = []
    v_a_history = []
    v_a0_history = []
    v_d_history = []
    v_d1_history = []
    temp1_history, temp2_history =[],[]
    idq0_history = []
    Torque_histroy = []
    Torque_histroy2 = []
    Torque_histroy3 = []
    
    
    print(f"\n==================================================")
    print(f" IPMSM 141 RPM BEMF(역기전력) 3상 전압 시뮬레이션 시작 (ABC LUT 연동)")
    print(f"==================================================")

    xJ = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  0]
    ])
    load_torque =0
    bemf = np.zeros(3)
    temp1 = np.zeros(3)
    temp2 = np.zeros(3)
    i_dq0 = np.zeros(3)
   
    V_abc = np.zeros(3)
    V_dq0 = np.zeros(3)
    V_dq1 = np.zeros(3)
    dlamda_dq0_dt2 = np.zeros(3)
    dlamda_dq0_dt = np.zeros(3)
    lamda_abc = np.zeros(3)
    lamda_dq0_old = np.zeros(3)
    lamda_abc_old = np.zeros(3)
    V_dq0 = np.zeros(3)
    current_radpsec = 0
    theta_mech_cum = 0.0 
    did_dt=np.zeros(3)  
    i_dq0_old = np.zeros(3)
    We_cross_lamda_dq0 = np.zeros(3)
    current_speed_rpm = 0.0
    speed = SpeedControllerPI(0.01,0.1)

    for step in range(total_steps):
        current_time = step * dt

        
        # 전기각 계산 (0 ~ 360도 정규화)
        theta_elec_rad, theta_elec_deg = iq5.check_degree(theta_mech_cum * iq5.pole_pairs)
        lamda_abc, lamda_dq0_0, lamda_dq0, Ldq, gamma_dq0 = iq5.get_lambda(i_dq0[0], i_dq0[1], theta_elec_deg)
       
        We = current_radpsec * iq5.pole_pairs
    
        if step > 0 :
                    
            V_abc = (lamda_abc - lamda_abc_old) / dt 

            dlamda_dq0_dt =  (lamda_dq0  - lamda_dq0_old) /dt
            We_cross_lamda_dq0 =  np.linalg.cross(np.array([0,0,We]) , lamda_dq0) #=We * xJ @ lamda_dq0 동일 값
            V_dq0 = dlamda_dq0_dt + We_cross_lamda_dq0
 
            Ls_di_dq0_dt = Ldq @ (i_dq0 - i_dq0_old)/dt
            We_gamma_dq0 = We * gamma_dq0
            dlamda_dq0_dt2 = Ls_di_dq0_dt + We_gamma_dq0
            V_dq1 = Ls_di_dq0_dt + We_gamma_dq0  +  We_cross_lamda_dq0 
            
            v_ref = speed.compute(target_speed_rpm, current_speed_rpm, dt)
            vq_ref = np.sin(np.radians(130)) * v_ref
            vd_ref = np.cos(np.radians(130)) * v_ref
            V_dq0[0] = vd_ref
            V_dq0[1] = vq_ref
            V_dq0[2] = 0
           
            did_dt[0] = 1/max(Ldq[0,0],1e-5) * (V_dq0[0] - iq5.Rs*i_dq0[0] + We * Ldq[1,1] * i_dq0[1]) 
            did_dt[1] = 1/max(Ldq[1,1],1e-5) * (V_dq0[1] - iq5.Rs*i_dq0[1] - We * (Ldq[0,0] * i_dq0[0] + lamda_dq0_0[0]))

            #did_dt = (-Rs*i_dq0 -  We * (  xJ @ lamda_dq0) )/ np.array([Ldq[0,0], Ldq[1,1], 1]) 
            did_dt[2] = 0
            #did_dt = np.linalg.inv(Ldq) @ (-Rs*i_dq0 -  We * (gamma_dq0 +  xJ @ lamda_dq0) )
            i_dq0 += did_dt * dt

        
        T_cog = 1.5 * iq5.pole_pairs * gamma_dq0 @ i_dq0
  
        #Te_z = 1.5 * pole_pairs * (lamda_dq0_0[0] * i_dq0[1] + (Ld - Lq) * i_dq0[0] * i_dq0[1]) + T_cog

        Te_z = 1.5 * iq5.pole_pairs * (lamda_dq0_0[0] * i_dq0[1] + (Ldq[0,0] - Ldq[1,1]) * i_dq0[0] * i_dq0[1]) + T_cog
        
        Te_z2 = (1.5 * iq5.pole_pairs *np.linalg.cross(lamda_dq0 ,i_dq0))[2] + T_cog

        query_point_T = np.array([[i_dq0[0], i_dq0[1], theta_elec_deg]])
        torque = iq5.interp_T(query_point_T)[0]

        Torque_histroy.append(Te_z)
        Torque_histroy2.append(Te_z2)
        Torque_histroy3.append(torque)

        # 모터 동역학 방정식 계산 및 회전 각도 산출
        accel_rad_s2 = (torque - load_torque) / iq5.Inertia
        current_radpsec += accel_rad_s2 * dt
        current_speed_rpm = current_radpsec * 30.0 / np.pi

        lamda_abc_old = lamda_abc
        lamda_dq0_old = lamda_dq0
        theta_elec_deg_old = theta_elec_deg
        i_dq0_old = i_dq0.copy()

     
        
        # 기계각 업데이트 (141 RPM 유지)
        #current_radpsec = target_speed_rpm/60*2*np.pi
        delta_mech_deg = (current_radpsec * dt) * (180 / np.pi)
        theta_mech_cum += delta_mech_deg
        
        # 기록
        time_history.append(current_time)
        speed_history.append(current_speed_rpm)
        v_a_history.append(V_abc)
        v_a0_history.append(iq5.dq_to_abc(V_dq1, theta_elec_rad))
        
        v_d_history.append(iq5.abc_to_dq(V_abc,theta_elec_rad))
        v_d1_history.append(We_cross_lamda_dq0)
        
        idq0_history.append(i_dq0)
        temp1_history.append(dlamda_dq0_dt)
        temp2_history.append(dlamda_dq0_dt2)
    print("[완료] BEMF 시뮬레이션 연산 완료. 그래프 생성 중...")

    # --- 시각화 ---
    fig, axes = plt.subplots(2, 2, figsize=(16*2/3, 9*2/3), sharex=True)
    colors_rgb=['red','green','blue']
    
    # 1. 속도 (141 RPM)
    #axes[0,0].plot(time_history, Torque_histroy, label=rf'$\phi_f i_q + (L_d - L_q) i_d i_q$', color='blue', linewidth=1, alpha=0.5)
    #axes[0,0].plot(time_history, Torque_histroy2, label=rf'$\lambda_{{dq}} x I_{{dq}}$', color='green', linewidth=2,linestyle='--')
    #axes[0,0].plot(time_history, Torque_histroy3, label='$FEMM Torque$', color='red', linewidth=1)
    #axes[0,0].plot(time_history, speed_history, label='RPM', color='red', linewidth=2)
    #axes[0,0].set_ylabel('Torque')
    #axes[0,0].legend(loc='best')
    #axes[0,0].grid(True, linestyle='--', alpha=0.6)
    #axes[0,0].set_title(rf'IPMSM Back-EMF (BEMF) 3-Phase Voltages at {-current_speed_rpm:.1f} RPM (ABC LUT)')
    # 1. 토크(좌축) 및 속도(우축, RPM)
    ax = axes[0, 0]
    ax_rpm = ax.twinx()

    l1, = ax.plot(time_history, Torque_histroy, label=rf'$\phi_f i_q + (L_d - L_q) i_d i_q$', color='blue', linewidth=1, alpha=0.5)
    l2, = ax.plot(time_history, Torque_histroy2, label=rf'$\lambda_{{dq}} x I_{{dq}}$', color='green', linewidth=2, linestyle='--')
    l3, = ax.plot(time_history, Torque_histroy3, label='$FEMM Torque$', color='red', linewidth=1)
    l4, = ax_rpm.plot(time_history, speed_history, label='RPM', color='purple', linewidth=2)

    ax.set_ylabel('Torque [Nm]')
    ax_rpm.set_ylabel('Speed [RPM]')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(rf'IPMSM Back-EMF (BEMF) 3-Phase Voltages at {-current_speed_rpm:.1f} RPM (ABC LUT)')

    # 좌축·우축 범례 통합
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='best')
    
    # 2. 3상 BEMF 전압 파형
    
    labels = ["$V_a$", "$V_b$", "$V_c$"]
    for  i in range(3):
        axes[1,0].plot(time_history, np.array(v_a_history)[:,i], label=rf'd$\lambda_{{abc}}$/dt ({labels[i]})', color=colors_rgb[i], linewidth=1.2, alpha=0.5)
        axes[1,0].plot(time_history, np.array(v_a0_history)[:,i], label=rf'$L_sI_{{dq}}+\omega\gamma_{{dq}}+\omega x \lambda_{{dq}}$ ({labels[i]})', color=colors_rgb[i], linewidth=1.2,linestyle='--')
        #axes[1,0].plot(time_history, np.array(v_a0_history)[:,i], label=f'd$\lambda_{{dq}}/dt+\omega x \lambda_{{dq}}$ ({labels[i]})', color=colors_rgb[i], linewidth=1.2,linestyle='--')
    axes[1,0].set_ylabel('BEMF Voltage [V]')
    axes[1,0].legend(loc='best')
    axes[1,0].grid(True, linestyle='--', alpha=0.6)
    
    # 3. 쇄교자속
    labels = ["$V_d$", "$V_q$", "$V_0$"]
    for  i in range(3):
        axes[0,1].plot(time_history, np.array(v_d_history)[:,i], label=rf'd$\lambda_{{abc}}$/dt({labels[i]})', color=colors_rgb[i], linewidth=1.2, alpha=0.5)
        axes[0,1].plot(time_history, np.array(v_d1_history)[:,i], label=f'd$\omega x \lambda_{{dq}}$ ({labels[i]})', color=colors_rgb[i], linewidth=1.2,linestyle='--')
        #axes[0,1].plot(time_history, np.array(v_d1_history)[:,i], label=rf'$L_sI_{{dq}}+\omega\gamma_{{dq}}+\omega x \lambda_{{dq}}$ ({labels[i]})', color=colors_rgb[i], linewidth=1.2,linestyle='--')
        
    axes[0,1].set_xlabel('Time [s]')
    axes[0,1].set_ylabel('BEMF Voltage [V]')
    axes[0,1].legend(loc='best')
    axes[0,1].grid(True, linestyle='--', alpha=0.6)
    # 4. 전류
    labels = ["$d$", "$q$", "$0$"]
    for  i in range(3):
        #axes[1,1].plot(time_history, np.array(temp1_history)[:,i], label=f'temp1({labels[i]})', color=colors_rgb[i], linewidth=1.2, alpha=0.5)
        #axes[1,1].plot(time_history, np.array(temp2_history)[:,i], label=f'temp2({labels[i]})', color=colors_rgb[i], linewidth=1.2,linestyle='--')
        axes[1,1].plot(time_history, np.array(idq0_history)[:,i], label=rf'i_{{dq0}}({labels[i]})', color=colors_rgb[i], linewidth=1.2, alpha=0.5)
    axes[1,1].set_xlabel('Time [s]')
    #axes[1,1].set_ylabel('Voltage [V]')
    axes[1,1].set_ylabel('Amphere [A]')
    axes[1,1].legend(loc='best')
    axes[1,1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('ipmsm_bemf_141rpm_abc.png', dpi=300)
    plt.show()
    print("[완료] BEMF 결과 그래프가 'ipmsm_bemf_141rpm_abc.png'로 저장되었습니다.")

if __name__ == '__main__':
    run_bemf_simulation_at_141rpm()