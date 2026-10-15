"""rpm sweep: node excitation -> chain response -> section stress -> fatigue damage."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import fatigue
from .chain import Model, build_model, harmonic_response, modal_response, static_spring_torques
from .config import Study
from .excitation import THETA, Excitation, build_excitation

# filter(order) -> (gas_scale, inertia_scale); None = untouched
HarmonicFilter = Callable[[float], tuple[float, float]]


@dataclass
class Result:
    name: str
    study: Study
    model: Model
    rpm: np.ndarray
    orders: np.ndarray
    F: np.ndarray                 # [rpm, order, node] complex node torques (Nm)
    tau_k: np.ndarray             # [rpm, order] complex section shear stress (MPa)
    tau_k_mode: np.ndarray        # [rpm, order, mode] modal contributions (MPa)
    tau_mean: np.ndarray          # [rpm]
    tau_t: np.ndarray             # [rpm, theta] reconstructed stress over 720 deg
    tau_amp: np.ndarray           # [rpm] (max-min)/2
    tau_amp_mode: np.ndarray      # [rpm, mode]
    D_cycle: np.ndarray           # [rpm] damage per 720 deg cycle
    tau_aeq_max: np.ndarray       # [rpm] largest mean-corrected cycle amplitude
    G: np.ndarray                 # [mode, order] modal excitation (geometry) factor
    damage_rate_h: float = 0.0
    life_h: float = np.inf
    bin_damage: np.ndarray = field(default_factory=lambda: np.zeros(0))


def section_stress_factor(st: Study) -> float:
    """MPa per Nm of section torque, incl. Kt."""
    D, d = st.section.d_outer_mm * 1e-3, st.section.d_inner_mm * 1e-3
    return st.section.kt * 16 * D / (np.pi * (D**4 - d**4)) / 1e6


def node_harmonics(st: Study, m: Model, exc: Excitation, filt: Optional[HarmonicFilter]):
    n_r, n_o, n_n = len(exc.rpm), len(exc.orders), len(m.names)
    Cg, Ci = exc.Cg.copy(), exc.Ci.copy()
    if filt:
        for b, k in enumerate(exc.orders):
            sg, si = filt(k)
            Cg[:, b] *= sg
            Ci[:, b] *= si
    C = Cg + Ci
    F = np.zeros((n_r, n_o, n_n), complex)
    Fmean = np.zeros((n_r, n_n))
    for c in st.engine.cylinders:
        j = m.idx(c.node)
        ph = np.exp(-1j * exc.orders * np.deg2rad(c.fire_deg))
        F[:, :, j] += C * ph[None, :]
        Fmean[:, j] += exc.Tmean
    return F, Fmean


def modal_excitation_factor(st: Study, m: Model, orders: np.ndarray) -> np.ndarray:
    """G[r, k] = sum_cyl phi_r(node_cyl) exp(-j k theta_fire,cyl)   (phase star x mode shape)."""
    G = np.zeros((m.Phi.shape[1], len(orders)), complex)
    for c in st.engine.cylinders:
        G += np.outer(m.Phi[m.idx(c.node)], np.exp(-1j * orders * np.deg2rad(c.fire_deg)))
    return G


def run(st: Study, exc: Excitation, name: str = "baseline",
        filt: Optional[HarmonicFilter] = None, model: Optional[Model] = None) -> Result:
    m = model or build_model(st.chain, st.engine)
    F, Fmean = node_harmonics(st, m, exc, filt)
    s = m.spring_index(*st.section.spring_ab)
    i, j, k_s = m.springs[s]
    fac = section_stress_factor(st)
    n_r, n_o = F.shape[:2]
    n_m = m.Phi.shape[1]
    tau_k = np.zeros((n_r, n_o), complex)
    tau_km = np.zeros((n_r, n_o, n_m), complex)
    tau_mean = np.zeros(n_r)
    for a, rpm in enumerate(exc.rpm):
        w = rpm * np.pi / 30
        for b, k in enumerate(exc.orders):
            X = modal_response(m, F[a, b], k * w)            # [node, mode], exact for modal damping
            tau_km[a, b] = fac * k_s * (X[i] - X[j])
        tau_mean[a] = fac * static_spring_torques(m, Fmean[a])[s]
    tau_k = tau_km.sum(axis=2)
    E = np.exp(1j * np.outer(exc.orders, THETA))
    tau_t = tau_mean[:, None] + np.real(tau_k @ E)
    tau_amp = (tau_t.max(1) - tau_t.min(1)) / 2
    tau_amp_mode = np.zeros((n_r, n_m))
    for r in range(n_m):
        tr = np.real(tau_km[:, :, r] @ E)
        tau_amp_mode[:, r] = (tr.max(1) - tr.min(1)) / 2
    D = np.zeros(n_r)
    aeq = np.zeros(n_r)
    for a in range(n_r):
        D[a], aeq[a] = fatigue.damage_per_period(tau_t[a], st.material)
    res = Result(name, st, m, exc.rpm, exc.orders, F, tau_k, tau_km, tau_mean, tau_t, tau_amp,
                 tau_amp_mode, D, aeq, modal_excitation_factor(st, m, exc.orders))
    _duty_damage(res)
    return res


def _duty_damage(res: Result):
    du = res.study.duty
    bins = np.asarray(du.rpm_bins, float)
    frac = np.asarray(du.time_frac, float)
    frac = frac / frac.sum()
    half = np.diff(bins).mean() / 2 if len(bins) > 1 else res.study.rpm_step
    per_h = np.zeros(len(bins))
    for b, (c, f) in enumerate(zip(bins, frac)):
        sel = (res.rpm >= c - half) & (res.rpm < c + half)
        if not sel.any():
            sel = np.abs(res.rpm - c) == np.abs(res.rpm - c).min()
        cycles_h = res.rpm[sel] / 2 * 60                    # 720 deg cycles per hour
        per_h[b] = f * np.mean(res.D_cycle[sel] * cycles_h)
    res.bin_damage = per_h
    res.damage_rate_h = float(per_h.sum())
    res.life_h = 1.0 / res.damage_rate_h if res.damage_rate_h > 0 else np.inf


# ----------------------------------------------------------------------------- study helpers
def sweep_grid(st: Study):
    rpm = np.arange(st.rpm_min, st.rpm_max + 1e-9, st.rpm_step)
    orders = np.arange(1, int(round(2 * st.orders_max)) + 1) / 2
    return rpm, orders


def excitation_for(st: Study) -> Excitation:
    rpm, orders = sweep_grid(st)
    return build_excitation(st.engine, st.pressure, rpm, orders)


def only(order: float, gas=True, inertia=True) -> HarmonicFilter:
    """Keep everything, but scale the given order's gas/inertia part to 0 when flag is False."""
    return lambda k: ((1.0 if gas else 0.0), (1.0 if inertia else 0.0)) if np.isclose(k, order) else (1.0, 1.0)


def single_order(order: float) -> HarmonicFilter:
    return lambda k: (1.0, 1.0) if np.isclose(k, order) else (0.0, 0.0)


def crossings(res: Result, rpm_range=None):
    """Order/mode resonance crossings inside the sweep (or rpm_range) with peak stress near them."""
    lo, hi = rpm_range or (res.rpm[0], res.rpm[-1])
    out = []
    for r, f in enumerate(res.model.f_hz):
        if f < 1e-3:
            continue
        for b, k in enumerate(res.orders):
            n = 60 * f / k
            if lo <= n <= hi:
                a = int(np.argmin(np.abs(res.rpm - n)))
                win = slice(max(a - 3, 0), a + 4)
                out.append(dict(mode=r, f_hz=f, order=k, rpm=n,
                                tau_order=float(np.abs(res.tau_k[win, b]).max()),
                                G=float(np.abs(res.G[r, b]))))
    return sorted(out, key=lambda d: -d["tau_order"])
