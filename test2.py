import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


class ThreePhaseSineTorqueSim:
    def __init__(self, lut_path, poles=8, Rs=0.0618, inertia=0.036):
        self.poles = poles
        self.pole_pairs = poles / 2.0
        self.Rs = Rs
        self.inertia = inertia
        self.Jx = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 0.0],
        ], dtype=float)

        self.interp_a, self.interp_b, self.interp_c, self.interp_T = (
            self._load_interpolators(lut_path)
        )

    @staticmethod
    def abc_to_dq(val_abc, theta_rad):
        """abc -> dq0 변환 (theta_rad: 전기각 [rad])"""
        sr32 = np.sqrt(3.0) / 2.0
        clarke = (2.0 / 3.0) * np.array([
            [1.0, -0.5, -0.5],
            [0.0, sr32, -sr32],
            [0.5,  0.5,  0.5],
        ], dtype=float)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        park = np.array([
            [ c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        return park @ (clarke @ np.asarray(val_abc, dtype=float))

    @staticmethod
    def norm_deg(deg):
        d = float(deg) % 360.0
        return 0.0 if d == 0.0 else d

    @staticmethod
    def _pad_periodic_theta(theta_axis, mat_3d):
        if len(theta_axis) > 1 and not np.isclose(theta_axis[-1], 360.0):
            theta_pad = np.append(theta_axis, 360.0)
            mat_pad = np.concatenate([mat_3d, mat_3d[:, :, :1]], axis=2)
            return theta_pad, mat_pad
        return theta_axis, mat_3d

    def _load_interpolators(self, lut_path):
        t0 = time.time()
        df = pd.read_csv(lut_path)

        def make_interp(col):
            piv = df.pivot_table(
                index=["Id_A", "Iq_A"],
                columns="theta_e_deg",
                values=col,
                aggfunc="mean",
            )
            id_ax = np.sort(piv.index.get_level_values("Id_A").unique().to_numpy(dtype=float))
            iq_ax = np.sort(piv.index.get_level_values("Iq_A").unique().to_numpy(dtype=float))
            th_ax = np.sort(piv.columns.to_numpy(dtype=float))

            full_idx = pd.MultiIndex.from_product([id_ax, iq_ax], names=["Id_A", "Iq_A"])
            piv = piv.reindex(index=full_idx, columns=th_ax)

            mat_3d = piv.to_numpy(dtype=float).reshape(len(id_ax), len(iq_ax), len(th_ax))
            th_per, mat_per = self._pad_periodic_theta(th_ax, mat_3d)
            return RegularGridInterpolator(
                (id_ax, iq_ax, th_per),
                mat_per,
                method="cubic",
                bounds_error=False,
                fill_value=None,
            )

        interp_a = make_interp("lamda_a_Wb")
        interp_b = make_interp("lamda_b_Wb")
        interp_c = make_interp("lamda_c_Wb")
        interp_T = make_interp("Torque")
        print(f"[정보] LUT 보간기 로드 완료 ({time.time() - t0:.2f}s)")
        return interp_a, interp_b, interp_c, interp_T

    def eval_lambda_dq0(self, id_val, iq_val, theta_deg, theta_rad):
        d_norm = self.norm_deg(theta_deg)
        pt = np.array([[id_val, iq_val, d_norm]], dtype=float)
        lam_abc = np.array([
            self.interp_a(pt)[0],
            self.interp_b(pt)[0],
            self.interp_c(pt)[0],
        ], dtype=float)
        return self.abc_to_dq(lam_abc, theta_rad)

    def get_flux_and_derivatives(self, id_val, iq_val, theta_deg, di=0.5, dtheta_deg=1.0):
        d_norm = self.norm_deg(theta_deg)
        th_rad = np.radians(d_norm)

        lam_dq0 = self.eval_lambda_dq0(id_val, iq_val, d_norm, th_rad)

        lam_id_p = self.eval_lambda_dq0(id_val + di, iq_val, d_norm, th_rad)
        lam_id_m = self.eval_lambda_dq0(id_val - di, iq_val, d_norm, th_rad)
        dlam_did = (lam_id_p - lam_id_m) / (2.0 * di)

        lam_iq_p = self.eval_lambda_dq0(id_val, iq_val + di, d_norm, th_rad)
        lam_iq_m = self.eval_lambda_dq0(id_val, iq_val - di, d_norm, th_rad)
        dlam_diq = (lam_iq_p - lam_iq_m) / (2.0 * di)

        dlam_di0 = np.array([0.0, 0.0, 1e-4], dtype=float)
        L_inc = np.column_stack([dlam_did, dlam_diq, dlam_di0])

        th_p_deg = d_norm + dtheta_deg
        th_m_deg = d_norm - dtheta_deg
        lam_th_p = self.eval_lambda_dq0(id_val, iq_val, th_p_deg, np.radians(th_p_deg))
        lam_th_m = self.eval_lambda_dq0(id_val, iq_val, th_m_deg, np.radians(th_m_deg))
        dlam_dtheta = (lam_th_p - lam_th_m) / (np.radians(th_p_deg) - np.radians(th_m_deg))

        pt = np.array([[id_val, iq_val, d_norm]], dtype=float)
        T_lut = float(self.interp_T(pt)[0])

        return lam_dq0, L_inc, dlam_dtheta, T_lut

    def compute_torque_dq0(self, lam_dq0, i_dq0, dlam_dtheta):
        cross_z = lam_dq0[0] * i_dq0[1] - lam_dq0[1] * i_dq0[0]
        T_em = 1.5 * self.pole_pairs * cross_z
        T_cog = 1.5 * self.pole_pairs * float(np.dot(dlam_dtheta, i_dq0))
        return T_em + T_cog

    @staticmethod
    def make_v_abc_sine(v_amp, elec_freq_hz, phase_deg=0.0):
        """3상 평형 정현파 전압 [va, vb, vc] 생성 함수"""
        we = 2.0 * np.pi * elec_freq_hz
        phi = np.radians(phase_deg)

        def v_abc_fn(t):
            ang = we * t + phi
            return np.array([
                v_amp * np.sin(ang),
                v_amp * np.sin(ang - 2.0 * np.pi / 3.0),
                v_amp * np.sin(ang + 2.0 * np.pi / 3.0),
            ], dtype=float)

        return v_abc_fn

    def simulate_141rpm(
        self,
        v_amp=10.0,
        phase_deg=0.0,
        total_time=0.4,
        dt=1e-4,
        fixed_speed=True,
        load_torque_fn=lambda t, wm: 0.0,
    ):
        """
        141 RPM 조건에서 3상 정현파 V_abc를 인가하고 토크(Te_dq0, Te_lut)를 출력합니다.
        - fixed_speed=True  : 141 RPM 일정 회전 상태에서 인가 전압에 따른 전류 및 토크 응답 확인
        - fixed_speed=False : 초기 141 RPM에서 출발해 전자기 토크와 부하 토크 차이에 따라 속도 변동
        """
        target_rpm = 141.0
        wm_141 = target_rpm * 2.0 * np.pi / 60.0        # 기계각속도 [rad/s]
        elec_freq_hz = (target_rpm * self.pole_pairs) / 60.0  # 전기 주파수 [Hz] (141*4/60 = 9.4 Hz)
        v_abc_fn = self.make_v_abc_sine(v_amp=v_amp, elec_freq_hz=elec_freq_hz, phase_deg=phase_deg)

        n_steps = int(total_time / dt)
        i_dq0 = np.zeros(3, dtype=float)
        wm = wm_141
        theta_mech_rad = 0.0

        hist = {
            "t": [],
            "va": [], "vb": [], "vc": [],
            "vd": [], "vq": [],
            "id": [], "iq": [],
            "rpm": [],
            "Te_dq0": [],
            "Te_lut": [],
        }

        for k in range(n_steps):
            t = k * dt
            theta_elec_rad = theta_mech_rad * self.pole_pairs
            theta_elec_deg = np.degrees(theta_elec_rad)
            we = wm * self.pole_pairs

            # 1) 3상 정현파 입력 전압 -> dq0 변환
            v_abc = v_abc_fn(t)
            v_dq0 = self.abc_to_dq(v_abc, theta_elec_rad)

            # 2) LUT 기반 자속, 증분 인덕턴스 행렬, 각도 미분, LUT 토크 조회
            lam_dq0, L_inc, dlam_dtheta, T_lut = self.get_flux_and_derivatives(
                i_dq0[0], i_dq0[1], theta_elec_deg
            )

            # 3) 이미지 전압 방정식 기반 di_dq0/dt 계산[cite: 2]
            rhs = (
                v_dq0
                - self.Rs * i_dq0
                - we * dlam_dtheta
                - we * (self.Jx @ lam_dq0)
            )
            di_dq0_dt = np.linalg.solve(L_inc + 1e-9 * np.eye(3), rhs)

            # 4) 출력 토크 계산 [Nm]
            Te_dq0 = self.compute_torque_dq0(lam_dq0, i_dq0, dlam_dtheta)

            # 5) 상태 적분
            i_dq0 = i_dq0 + di_dq0_dt * dt

            if not fixed_speed:
                T_load = float(load_torque_fn(t, wm))
                wm = wm + ((Te_dq0 - T_load) / self.inertia) * dt

            theta_mech_rad = theta_mech_rad + wm * dt

            # 6) 결과 기록
            hist["t"].append(t)
            hist["va"].append(v_abc[0]); hist["vb"].append(v_abc[1]); hist["vc"].append(v_abc[2])
            hist["vd"].append(v_dq0[0]); hist["vq"].append(v_dq0[1])
            hist["id"].append(i_dq0[0]); hist["iq"].append(i_dq0[1])
            hist["rpm"].append(wm * 30.0 / np.pi)
            hist["Te_dq0"].append(Te_dq0)
            hist["Te_lut"].append(T_lut)

        return {k: np.array(v) for k, v in hist.items()}


if __name__ == "__main__":
    sim = ThreePhaseSineTorqueSim(
        lut_path="LUT_lamda_abcT_20261119_091223.csv",
        poles=8,
        Rs=0.0618,
        inertia=0.036,
    )

    res = sim.simulate_141rpm(
        v_amp=1.0,       # 3상 정현파 상전압 진폭 [V] (필요에 따라 조정)
        phase_deg=90.0,   # 동기 기준 위상각 [deg]
        total_time=0.4,
        dt=1e-4,
        fixed_speed=True, # 141 RPM 고정 속도에서 출력 토크 확인 (자유 가속 시 False)
    )

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(res["t"], res["va"], label="Va [V]")
    axes[0].plot(res["t"], res["vb"], label="Vb [V]")
    axes[0].plot(res["t"], res["vc"], label="Vc [V]")
    axes[0].set_ylabel("V_abc [V]")
    axes[0].set_title("3-Phase Sine Voltage Input (141 RPM, 9.4 Hz) -> Torque Output")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(res["t"], res["vd"], label="Vd [V]")
    axes[1].plot(res["t"], res["vq"], label="Vq [V]")
    axes[1].set_ylabel("V_dq [V]")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].plot(res["t"], res["id"], label="Id [A]")
    axes[2].plot(res["t"], res["iq"], label="Iq [A]")
    axes[2].set_ylabel("Current [A]")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    axes[3].plot(res["t"], res["Te_dq0"], label="Output Torque (dq0) [Nm]")
    axes[3].plot(res["t"], res["Te_lut"], "--", label="Output Torque (LUT) [Nm]")
    axes[3].set_ylabel("Torque [Nm]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("sine_141rpm_torque.png", dpi=300)
    plt.show()