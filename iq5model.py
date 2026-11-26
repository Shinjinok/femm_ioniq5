import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# 모터 기본 파라미터 설정
#poles = 8
#pole_pairs = poles / 2   # 극쌍수 (4)
#Rs = 0.0618             # 고정자 권선 저항 [Ω]
#J = 0.036

class iq5model:
    def __init__(self, pole=8, Rs=0.0618*2, Inertia = 0.036, lut="" ):
        self.pole = pole
        self.pole_pairs = pole / 2.0
        self.Rs = Rs
        self.Inertia = Inertia
        self.interp_a = None
        self.interp_b = None
        self.interp_c = None
        self.interp_T = None
        if lut:
            try:
                self.interp_a, self.interp_b, self.interp_c, self.interp_T = (
                    self.load_lut_abc_interpolators(lut)
                )
            except FileNotFoundError:
                print(f"[경고] LUT 파일을 찾을 수 없습니다: {lut}")
        else:
            print("[정보] LUT 파일 경로가 지정되지 않았습니다.")

    def dq_to_abc(self, dq0, b):
        """dq0 -> abc 역변환 (b: 전기각 [rad])"""
        sr32 = np.sqrt(3) / 2
        clac2 = np.array([
            [1, -1/2, -1/2],
            [0, sr32, -sr32],
            [1, 1, 1] #<- [1/2, 1/2, 1/2] 1/2에서 1로 변경해야 정확히 일치
        ])
        cb = np.cos(b)
        sb = np.sin(b)
        park_3x3 = np.array([
            [cb, sb, 0],
            [-sb, cb, 0],
            [0, 0, 1]
        ])
        abc = clac2.T @ park_3x3.T @ dq0
        return abc
    def abc_to_dq(self,val_abc, b):
        """abc -> dq0 변환 (b: 전기각 [rad])"""
        sr32 = np.sqrt(3) / 2
        clac_3x3 = (2 / 3) * np.array([
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
        dq0 = park_3x3 @ clac_3x3 @ val_abc
        return dq0
    def check_degree(self, degree):
        degree %= 360.0
        if degree < 0.0:
            degree += 360.0
        return np.radians(degree), degree
    def load_lut_abc_interpolators(self, lut):
        print("[정보] 3상(ABC) 쇄교자속 LUT CSV 파일 로딩 및 보간기 생성 중...")
        start_t = time.time()
        df_a = pd.read_csv(lut)
    
        pivot_df_a = df_a.pivot_table(
            index=['Id_A', 'Iq_A'], 
            columns='theta_e_deg', 
            values='lamda_a_Wb'
        )
        pivot_df_b = df_a.pivot_table(
            index=['Id_A', 'Iq_A'], 
            columns='theta_e_deg', 
            values='lamda_b_Wb'
        )
        pivot_df_c = df_a.pivot_table(
            index=['Id_A', 'Iq_A'], 
            columns='theta_e_deg', 
            values='lamda_c_Wb'
        )
        pivot_df_T = df_a.pivot_table(
                index=['Id_A', 'Iq_A'], 
                columns='theta_e_deg', 
                values='Torque'
        )

        # 피벗 테이블로부터 축(Axis) 값과 매트릭스 추출
        id_axis = pivot_df_a.index.get_level_values('Id_A').unique().values 
        iq_axis = pivot_df_a.index.get_level_values('Iq_A').unique().values  
        theta_axis = pivot_df_a.columns.values                                 
        
        matrix_3d_a = pivot_df_a.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
        matrix_3d_b = pivot_df_b.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
        matrix_3d_c = pivot_df_c.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
        matrix_3d_T = pivot_df_T.values.reshape(len(id_axis), len(iq_axis), len(theta_axis))
        # 범위 이탈 방지를 위한 bounds_error=False, fill_value=None 설정
        interp_a = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_a, method="cubic", bounds_error=False, fill_value=None)
        interp_b = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_b, method="cubic", bounds_error=False, fill_value=None)
        interp_c = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_c, method="cubic", bounds_error=False, fill_value=None)
        interp_T = RegularGridInterpolator((id_axis, iq_axis, theta_axis), matrix_3d_T, method="cubic", bounds_error=False, fill_value=None)
        print(f"[완료] ABC 보간기 생성 완료 (소요 시간: {time.time() - start_t:.2f}초)")
        return interp_a, interp_b, interp_c, interp_T
    
    def get_lambda(self,id_val,iq_val,theta_elec_deg):
        theta_elec_rad, theta_elec_deg = self.check_degree(theta_elec_deg)
        query_point0 = np.array([[0, 0, theta_elec_deg]])
        lamda_abc_0 = np.array([self.interp_a(query_point0)[0], self.interp_b(query_point0)[0], self.interp_c(query_point0)[0]])
        lamda_dq0_0 = self.abc_to_dq(lamda_abc_0, theta_elec_rad)
        
        #현재 쇄교자속 조회
        query_point = np.array([[id_val, iq_val, theta_elec_deg]])
        lamda_abc = np.array([self.interp_a(query_point)[0], self.interp_b(query_point)[0], self.interp_c(query_point)[0]])
        lamda_dq0 = self.abc_to_dq(lamda_abc, theta_elec_rad)
    
        #전류 스텝
        current_step = 1 #A
        #Ld 계산
        query_point_id1 = np.array([[id_val - current_step, iq_val, theta_elec_deg]])
        query_point_id2 = np.array([[id_val + current_step, iq_val, theta_elec_deg]])
        lamda_abc_id1 = np.array([self.interp_a(query_point_id1)[0], self.interp_b(query_point_id1)[0], self.interp_c(query_point_id1)[0]])
        lamda_abc_id2 = np.array([self.interp_a(query_point_id2)[0], self.interp_b(query_point_id2)[0], self.interp_c(query_point_id2)[0]])
        lamda_dq0_id1 = self.abc_to_dq(lamda_abc_id1, theta_elec_rad)-lamda_dq0_0
        lamda_dq0_id2 = self.abc_to_dq(lamda_abc_id2, theta_elec_rad)-lamda_dq0_0
    
        
        Ld = (lamda_dq0_id2 - lamda_dq0_id1)/(2*current_step)
        
    
        #Lq 계산
        query_point_iq1 = np.array([[id_val, iq_val - current_step, theta_elec_deg]])
        query_point_iq2 = np.array([[id_val, iq_val + current_step, theta_elec_deg]])
        lamda_abc_iq1 = np.array([self.interp_a(query_point_iq1)[0], self.interp_b(query_point_iq1)[0], self.interp_c(query_point_iq1)[0]])
        lamda_abc_iq2 = np.array([self.interp_a(query_point_iq2)[0], self.interp_b(query_point_iq2)[0], self.interp_c(query_point_iq2)[0]])
        lamda_dq0_iq1 = self.abc_to_dq(lamda_abc_iq1, theta_elec_rad)
        lamda_dq0_iq2 = self.abc_to_dq(lamda_abc_iq2, theta_elec_rad)
    
        
        Lq = (lamda_dq0_iq2 - lamda_dq0_iq1)/(2*current_step)
        
        Ldq = np.array([[Ld[0],Lq[0],0],[Ld[1],Lq[1],0],[Ld[2],Lq[2],0]])
        #각도 변위
        degree_step = 1.0
        #감마 계산
        dth1_rad, dth1_deg = self.check_degree(theta_elec_deg - degree_step)
        dth2_rad, dth2_deg = self.check_degree(theta_elec_deg + degree_step)
        
        query_point_th1 = np.array([[id_val, iq_val, dth1_deg]])
        query_point_th2 = np.array([[id_val, iq_val, dth2_deg]])
        lamda_abc_th1 = np.array([self.interp_a(query_point_th1)[0], self.interp_b(query_point_th1)[0], self.interp_c(query_point_th1)[0]])
        lamda_abc_th2 = np.array([self.interp_a(query_point_th2)[0], self.interp_b(query_point_th2)[0], self.interp_c(query_point_th2)[0]])
        lamda_dq0_th1 = self.abc_to_dq(lamda_abc_th1, dth1_rad )
        lamda_dq0_th2 = self.abc_to_dq(lamda_abc_th2, dth2_rad ) 
        
        gamma_dq0 = (lamda_dq0_th2 - lamda_dq0_th1 ) / (2 * np.radians(degree_step))
    
        return lamda_abc, lamda_dq0_0, lamda_dq0, Ldq, gamma_dq0






