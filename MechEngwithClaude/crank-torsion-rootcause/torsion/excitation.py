"""Single-cylinder gas + inertia torque and its order spectrum.

Cycle angle theta_c in [0, 720) deg, 0 = firing TDC of that cylinder:
    0-180 expansion, 180-360 exhaust, 360-540 intake, 540-720 compression.
Order k (k = 0.5, 1, 1.5 ...) complex amplitude A_k such that
    T(theta) = T_mean + sum_k Re( A_k exp(j k theta) )
A cylinder firing at crank angle theta_fire therefore contributes A_k exp(-j k theta_fire).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Engine, PressureModel

N_CYCLE = 1440                                    # 0.5 deg resolution over 720 deg
THETA_DEG = np.arange(N_CYCLE) * 720.0 / N_CYCLE
THETA = np.deg2rad(THETA_DEG)


# ----------------------------------------------------------------------------- kinematics
def piston_kinematics(phi, r, l):
    """Distance crank centre -> piston pin along the cylinder axis and its phi-derivatives."""
    s, c = np.sin(phi), np.cos(phi)
    S = np.sqrt(l**2 - (r * s) ** 2)
    g = s * c
    x = r * c + S
    x1 = -r * s - r**2 * g / S
    x2 = -r * c - r**2 * (np.cos(2 * phi) / S + r**2 * g**2 / S**3)
    return x, x1, x2


def geometry(eng: Engine):
    r, l = eng.r_mm * 1e-3, eng.l_mm * 1e-3
    A = np.pi * (eng.bore_mm * 1e-3) ** 2 / 4
    return r, l, A


# ----------------------------------------------------------------------------- pressure
class PressureSource:
    """p(theta_c) in bar for a given rpm, either from the Wiebe model or from a CSV."""

    def __init__(self, pm: PressureModel, eng: Engine):
        self.pm, self.eng = pm, eng
        r, l, A = geometry(eng)
        Vd = A * 2 * r
        self.Vd = Vd
        Vc = Vd / (pm.compression_ratio - 1)
        x, x1, _ = piston_kinematics(THETA, r, l)
        self.V = Vc + A * (l + r - x)                  # cylinder volume over the cycle grid
        self.dV = -A * x1                              # dV/dtheta
        if pm.csv_path:
            self._load_csv(pm.csv_path)
        else:
            self._precompute_closed_cycle(Vc, A, r, l)

    # --- Wiebe closed cycle: p = p_ivc * pa + Q * pb (linear ODE -> superposition)
    def _precompute_closed_cycle(self, Vc, A, r, l):
        pm = self.pm
        g = pm.gamma
        tc = np.arange(-180.0, pm.evo_deg + 1e-9, 0.5)          # IVC (BDC) ... EVO
        V = lambda t: Vc + A * (l + r - piston_kinematics(np.deg2rad(t), r, l)[0])
        dV = lambda t: -A * piston_kinematics(np.deg2rad(t), r, l)[1] * np.pi / 180  # per deg

        def dxb(t):                                              # Wiebe burn rate per deg
            u = np.clip((t - pm.soc_deg) / pm.duration_deg, 0, 1)
            m, a = pm.wiebe_m, pm.wiebe_a
            return np.where((t > pm.soc_deg) & (t < pm.soc_deg + pm.duration_deg),
                            a * (m + 1) / pm.duration_deg * u**m * np.exp(-a * u ** (m + 1)), 0.0)

        def rhs(t, p, q):                                        # first law, single zone
            return (g - 1) / V(t) * q * dxb(t) - g * p / V(t) * dV(t)

        def integrate(p0, q):
            p = np.empty_like(tc)
            p[0] = p0
            h = tc[1] - tc[0]
            for i in range(len(tc) - 1):
                t, y = tc[i], p[i]
                k1 = rhs(t, y, q)
                k2 = rhs(t + h / 2, y + h / 2 * k1, q)
                k3 = rhs(t + h / 2, y + h / 2 * k2, q)
                k4 = rhs(t + h, y + h * k3, q)
                p[i + 1] = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return p

        self._tc = tc
        self._pa = integrate(1.0, 0.0)          # Pa per Pa of p_ivc
        self._pb = integrate(0.0, 1.0)          # Pa per J of heat release

    def _cycle_from_closed(self, Q):
        pm = self.pm
        p_int, p_exh = pm.p_intake_bar * 1e5, pm.p_exhaust_bar * 1e5
        pc = p_int * self._pa + Q * self._pb                     # theta_c = -180 ... EVO
        tc_wrapped = np.mod(self._tc, 720.0)
        p = np.empty(N_CYCLE)
        # gas exchange
        p[(THETA_DEG >= 180) & (THETA_DEG < 360)] = p_exh
        p[(THETA_DEG >= 360) & (THETA_DEG < 540)] = p_int
        # closed part
        idx = np.rint(tc_wrapped / (720.0 / N_CYCLE)).astype(int) % N_CYCLE
        p[idx] = pc
        # blow-down EVO -> 180
        m = (THETA_DEG > pm.evo_deg) & (THETA_DEG < 180)
        s = (THETA_DEG[m] - pm.evo_deg) / (180 - pm.evo_deg)
        p[m] = pc[-1] + (p_exh - pc[-1]) * (1 - np.cos(np.pi * s)) / 2
        return p

    def imep(self, p_pa):
        return np.sum(p_pa * self.dV) * (THETA[1] - THETA[0]) / self.Vd

    def p_bar(self, rpm: float) -> np.ndarray:
        if self.pm.csv_path:
            return self._csv_p(rpm)
        target = np.interp(rpm, *zip(*self.pm.imep_curve)) * 1e5
        i0 = self.imep(self._cycle_from_closed(0.0))
        i1 = self.imep(self._cycle_from_closed(1.0)) - i0
        return self._cycle_from_closed((target - i0) / i1) / 1e5

    # --- CSV: columns theta_deg, p_bar_<rpm> ...  (theta 0 = firing TDC, any 720 deg window)
    def _load_csv(self, path):
        import csv
        with open(path) as f:
            rows = list(csv.reader(f))
        head, data = rows[0], np.array(rows[1:], float)
        th = np.mod(data[:, 0], 720.0)
        order = np.argsort(th)
        self._csv_rpm = np.array([float(h.split("_")[-1]) for h in head[1:]])
        self._csv_p_grid = np.array([np.interp(THETA_DEG, th[order], data[order, j + 1], period=720.0)
                                     for j in range(len(self._csv_rpm))])

    def _csv_p(self, rpm):
        rr = self._csv_rpm
        if rpm <= rr[0]:
            return self._csv_p_grid[0]
        if rpm >= rr[-1]:
            return self._csv_p_grid[-1]
        j = np.searchsorted(rr, rpm)
        w = (rpm - rr[j - 1]) / (rr[j] - rr[j - 1])
        return (1 - w) * self._csv_p_grid[j - 1] + w * self._csv_p_grid[j]


# ----------------------------------------------------------------------------- torques
def cylinder_torques(eng: Engine, p_bar: np.ndarray, rpm: float):
    """Gas and oscillating-mass torque on the crank of ONE cylinder over its 720 deg cycle [Nm]."""
    r, l, A = geometry(eng)
    _, x1, x2 = piston_kinematics(THETA, r, l)
    w = rpm * np.pi / 30
    F_gas = -(p_bar - eng.p_crankcase_bar) * 1e5 * A          # force on piston along +x (to head)
    F_in = -eng.m_osc_g * 1e-3 * w**2 * x2                     # d'Alembert force of the piston mass
    return F_gas * x1, F_in * x1                               # T = F_x dx/dtheta


def order_spectrum(T: np.ndarray, orders: np.ndarray):
    c = np.fft.rfft(T) / len(T)
    idx = np.rint(orders * 2).astype(int)
    return c[0].real, 2 * c[idx]


@dataclass
class Excitation:
    rpm: np.ndarray
    orders: np.ndarray
    Cg: np.ndarray            # [n_rpm, n_order] complex, single cylinder gas
    Ci: np.ndarray            # [n_rpm, n_order] complex, single cylinder inertia
    Tmean: np.ndarray         # [n_rpm] mean torque per cylinder
    pmax_bar: np.ndarray
    imep_bar: np.ndarray


def build_excitation(eng: Engine, pm: PressureModel, rpm: np.ndarray, orders: np.ndarray) -> Excitation:
    src = PressureSource(pm, eng)
    n, m = len(rpm), len(orders)
    Cg = np.zeros((n, m), complex)
    Ci = np.zeros((n, m), complex)
    Tm, pmax, imep = np.zeros(n), np.zeros(n), np.zeros(n)
    for a, n_ in enumerate(rpm):
        p = src.p_bar(n_)
        Tg, Ti = cylinder_torques(eng, p, n_)
        mg, Cg[a] = order_spectrum(Tg, orders)
        mi, Ci[a] = order_spectrum(Ti, orders)
        Tm[a], pmax[a], imep[a] = mg + mi, p.max(), src.imep(p * 1e5) / 1e5
    return Excitation(rpm, orders, Cg, Ci, Tm, pmax, imep)
