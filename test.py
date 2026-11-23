import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


class SVPWMGenerator:
    """DC 링크 전압 V_dc 기준 3상 SVPWM 전압 생성기"""

    def __init__(self, v_dc=700.0, pwm_freq_hz=10000.0, mod_index=0.8):
        self.v_dc = float(v_dc)
        self.pwm_freq_hz = float(pwm_freq_hz)
        self.mod_index = float(np.clip(mod_index, 0.0, 1.0))
        self.v_phase_max_linear = self.v_dc / np.sqrt(3.0)

    @staticmethod
    def _triangle_carrier(t, pwm_freq_hz):
        """[-1, +1] 대칭 삼각파 캐리어 생성"""
        period = 1.0 / pwm_freq_hz
        phase = (t % period) / period
        if phase < 0.5:
            return -1.0 + 4.0 * phase
        return 3.0 - 4.0 * phase

    def compute_svpwm_abc(self, t, v_amp, elec_freq_hz, phase_deg=0.0, mode="carrier"):
        """
        시각 t에서 SVPWM 3상 출력 전압 [va, vb, vc] [V] 반환
        - mode="carrier" : 삼각파 캐리어 비교 기반 실제 스위칭 3상 전압
        - mode="average" : 스위칭 주기 평균 SVPWM 상전압(3차 고조파/영상분 포함)
        """
        v_ref_amp = float(np.clip(v_amp, 0.0, self.v_phase_max_linear))
        we = 2.0 * np.pi * float(elec_freq_hz)
        phi = np.radians(phase_deg)
        ang = we * t + phi

        va_ref = v_ref_amp * np.sin(ang)
        vb_ref = v_ref_amp * np.sin(ang - 2.0 * np.pi / 3.0)
        vc_ref = v_ref_amp * np.sin(ang + 2.0 * np.pi / 3.0)

        v_max = max(va_ref, vb_ref, vc_ref)
        v_min = min(va_ref, vb_ref, vc_ref)
        v_offset = -0.5 * (v_max + v_min)

        va_sv = va_ref + v_offset
        vb_sv = vb_ref + v_offset
        vc_sv = vc_ref + v_offset

        if mode == "average":
            return np.array([va_sv, vb_sv, vc_sv], dtype=float)

        half_vdc = 0.5 * self.v_dc
        da = np.clip(0.5 + va_sv / self.v_dc, 0.0, 1.0)
        db = np.clip(0.5 + vb_sv / self.v_dc, 0.0, 1.0)
        dc = np.clip(0.5 + vc_sv / self.v_dc, 0.0, 1.0)

        carrier = self._triangle_carrier(t, self.pwm_freq_hz)
        sa = 1.0 if (2.0 * da - 1.0) >= carrier else -1.0
        sb = 1.0 if (2.0 * db - 1.0) >= carrier else -1.0
        sc = 1.0 if (2.0 * dc - 1.0) >= carrier else -1.0

        v_an = sa * half_vdc
        v_bn = sb * half_vdc
        v_cn = sc * half_vdc

        v_sn = (v_an + v_bn + v_cn) / 3.0
        return np.array([v_an - v_sn, v_bn - v_sn, v_cn - v_sn], dtype=float)


class SVPMTorqueSim:
    def __init__(self, lut_path, poles=8, Rs=0.0618, inertia=0.036, interp_method="cubic"):
        self.poles = poles
        self.pole_pairs = poles / 2.0
        self.Rs = Rs
        self.inertia = inertia
        self.interp_method = interp_method
        self.Jx = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 0.0],
        ], dtype=float)

        (
            self.interp_a,
            self.interp_b,
            self.interp_c,
            self.interp_T,
            self.id_bounds,
            self.iq_bounds,
        ) = self._load_interpolators(lut_path)

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

        id_all = df["Id_A"].to_numpy(dtype=float)
        iq_all = df["Iq_A"].to_numpy(dtype=float)
        id_bounds = (float(np.min(id_all)), float(np.max(id_all)))
        iq_bounds = (float(np.min(iq_all)), float(np.max(iq_all)))

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
                method=self.interp_method,
                bounds_error=False,
                fill_value=None,
            )

        interp_a = make_interp("lamda_a_Wb")
        interp_b = make_interp("lamda_b_Wb")
        interp_c = make_interp("lamda_c_Wb")
        interp_T = make_interp("Torque")
        print(
            f"[정보] LUT 보간기 로드 완료 ({time.time() - t0:.2f}s) | "
            f"Id: [{id_bounds[0]}, {id_bounds[1]}] A, Iq: [{iq_bounds[0]}, {iq_bounds[1]}] A"
        )
        return interp_a, interp_b, interp_c, interp_T, id_bounds, iq_bounds

    def _clip_currents_to_lut(self, id_val, iq_val, margin=1e-6):
        id_c = float(np.clip(id_val, self.id_bounds[0] + margin, self.id_bounds[1] - margin))
        iq_c = float(np.clip(iq_val, self.iq_bounds[0] + margin, self.iq_bounds[1] - margin))
        return id_c, iq_c

    def eval_lambda_dq0(self, id_val, iq_val, theta_deg, theta_rad):
        d_norm = self.norm_deg(theta_deg)
        id_c, iq_c = self._clip_currents_to_lut(id_val, iq_val)
        pt = np.array([[id_c, iq_c, d_norm]], dtype=float)
        lam_abc = np.array([
            self.interp_a(pt)[0],
            self.interp_b(pt)[0],
            self.interp_c(pt)[0],
        ], dtype=float)
        return self.abc_to_dq(lam_abc, theta_rad)

    def get_flux_and_derivatives(self, id_val, iq_val, theta_deg, di=0.5, dtheta_deg=1.0):
        d_norm = self.norm_deg(theta_deg)
        th_rad = np.radians(d_norm)
        id_c, iq_c = self._clip_currents_to_lut(id_val, iq_val)

        lam_dq0 = self.eval_lambda_dq0(id_c, iq_c, d_norm, th_rad)

        lam_id_p = self.eval_lambda_dq0(id_c + di, iq_c, d_norm, th_rad)
        lam_id_m = self.eval_lambda_dq0(id_c - di, iq_c, d_norm, th_rad)
        dlam_did = (lam_id_p - lam_id_m) / (2.0 * di)

        lam_iq_p = self.eval_lambda_dq0(id_c, iq_c + di, d_norm, th_rad)
        lam_iq_m = self.eval_lambda_dq0(id_c, iq_c - di, d_norm, th_rad)
        dlam_diq = (lam_iq_p - lam_iq_m) / (2.0 * di)

        dlam_di0 = np.array([0.0, 0.0, 1e-4], dtype=float)
        L_inc = np.column_stack([dlam_did, dlam_diq, dlam_di0])

        th_p_deg = d_norm + dtheta_deg
        th_m_deg = d_norm - dtheta_deg
        lam_th_p = self.eval_lambda_dq0(id_c, iq_c, th_p_deg, np.radians(th_p_deg))
        lam_th_m = self.eval_lambda_dq0(id_c, iq_c, th_m_deg, np.radians(th_m_deg))
        dlam_dtheta = (lam_th_p - lam_th_m) / (np.radians(th_p_deg) - np.radians(th_m_deg))

        pt = np.array([[id_c, iq_c, d_norm]], dtype=float)
        T_lut = float(self.interp_T(pt)[0])

        return lam_dq0, L_inc, dlam_dtheta, T_lut

    def compute_torque_dq0(self, lam_dq0, i_dq0, dlam_dtheta):
        cross_z = float(lam_dq0[0]) * float(i_dq0[1]) - float(lam_dq0[1]) * float(i_dq0[0])
        T_em = 1.5 * self.pole_pairs * cross_z
        T_cog = 1.5 * self.pole_pairs * float(np.dot(dlam_dtheta, i_dq0))
        return T_em + T_cog

    def simulate_svpwm(
        self,
        target_rpm=141.0,
        v_dc=700.0,
        mod_index=0.2,
        phase_deg=130.0,
        pwm_freq_hz=10000.0,
        svpwm_mode="average",
        total_time=0.01,
        dt=1e-5,
        max_di_dt=1e7,
        fixed_speed=True,
    ):
        svpwm = SVPWMGenerator(v_dc=v_dc, pwm_freq_hz=pwm_freq_hz, mod_index=mod_index)
        v_amp = mod_index * svpwm.v_phase_max_linear

        wm = float(target_rpm) * 2.0 * np.pi / 60.0
        elec_freq_hz = (float(target_rpm) * self.pole_pairs) / 60.0
        theta_mech_rad = 0.0
        i_dq0 = np.zeros(3, dtype=float)

        n_steps = max(1, int(round(total_time / dt)))
        hist = {
            "t": [],
            "va": [], "vb": [], "vc": [],
            "vd": [], "vq": [],
            "id": [], "iq": [],
            "Te_dq0": [],
            "Te_lut": [],
        }

        for k in range(n_steps):
            t = k * dt
            theta_elec_rad = theta_mech_rad * self.pole_pairs
            theta_elec_deg = np.degrees(theta_elec_rad)
            we = wm * self.pole_pairs

            v_abc = svpwm.compute_svpwm_abc(
                t=t,
                v_amp=v_amp,
                elec_freq_hz=elec_freq_hz,
                phase_deg=phase_deg,
                mode=svpwm_mode,
            )
            v_dq0 = self.abc_to_dq(v_abc, theta_elec_rad)

            lam_dq0, L_inc, dlam_dtheta, T_lut = self.get_flux_and_derivatives(
                i_dq0[0], i_dq0[1], theta_elec_deg
            )

            rhs = (
                v_dq0
                - self.Rs * i_dq0
                - we * dlam_dtheta
                - we * (self.Jx @ lam_dq0)
            )
            try:
                di_dq0_dt = np.linalg.solve(L_inc + 1e-7 * np.eye(3), rhs)
            except np.linalg.LinAlgError:
                di_dq0_dt, _, _, _ = np.linalg.lstsq(
                    L_inc + 1e-5 * np.eye(3), rhs, rcond=None
                )

            di_dq0_dt = np.clip(di_dq0_dt, -max_di_dt, max_di_dt)
            Te_dq0 = self.compute_torque_dq0(lam_dq0, i_dq0, dlam_dtheta)

            i_dq0 = i_dq0 + di_dq0_dt * dt
            if not np.all(np.isfinite(i_dq0)):
                break

            if not fixed_speed:
                wm = wm + (Te_dq0 / self.inertia) * dt

            theta_mech_rad = theta_mech_rad + wm * dt

            hist["t"].append(t)
            hist["va"].append(v_abc[0]); hist["vb"].append(v_abc[1]); hist["vc"].append(v_abc[2])
            hist["vd"].append(v_dq0[0]); hist["vq"].append(v_dq0[1])
            hist["id"].append(i_dq0[0]); hist["iq"].append(i_dq0[1])
            hist["Te_dq0"].append(Te_dq0)
            hist["Te_lut"].append(T_lut)

        return {k: np.array(v) for k, v in hist.items()}, v_amp


if __name__ == "__main__":
    sim = SVPMTorqueSim(
        lut_path="LUT_lamda_abcT_20261119_091223.csv",
        poles=8,
        Rs=0.0618,
        inertia=0.036,
        interp_method="cubic",
    )

    res, applied_v_amp = sim.simulate_svpwm(
        target_rpm=141.0,
        v_dc=700.0,
        mod_index=0.05,
        phase_deg=90.0,
        pwm_freq_hz=5000.0,
        svpwm_mode="carrier",
        total_time=0.2,
        dt=1e-5,
        fixed_speed=True,
    )
    print(f"[정보] V_dc=700V 기준 SVPWM 적용 상전압 진폭: {applied_v_amp:.2f} V")

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(res["t"], res["va"], label="Va (SVPWM) [V]", linewidth=0.9)
    axes[0].plot(res["t"], res["vb"], label="Vb (SVPWM) [V]", linewidth=0.9)
    axes[0].plot(res["t"], res["vc"], label="Vc (SVPWM) [V]", linewidth=0.9)

    axes[0].set_ylabel("V [V]")
    axes[0].set_title(f"SVPWM Input (V_dc=700V, V_amp={applied_v_amp:.1f}V) -> Torque Output")
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
    plt.savefig("svpwm_700v_torque.png", dpi=300)
    plt.show()