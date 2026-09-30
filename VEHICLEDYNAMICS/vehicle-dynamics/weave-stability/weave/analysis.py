"""
weave.analysis
==============
Eigenvalue analysis on top of :mod:`weave.multibody`: mode identification by
modal participation, speed sweeps, critical speeds and model-variant helpers
(gyro switches, engine relocation).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

from .multibody import LinearBike, linearize
from .params import BikeModel, Body, Powertrain

MODES = ("capsize", "weave", "wobble")


@dataclass
class Mode:
    name: str
    eigenvalue: complex
    growth: float            # real part [1/s]; >0 unstable
    frequency: float         # Hz (nan for real modes)
    damping: float           # damping ratio (nan for real modes)

    @property
    def stable(self) -> bool:
        return self.growth < 0


def _participation(A):
    ev, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    P = np.abs(V * W.T)
    return ev, P / P.sum(axis=0)


def identify_modes(A: np.ndarray, names: List[str]) -> Dict[str, Optional[Mode]]:
    """
    Classify eigenvalues of the reduced state matrix by modal participation
    (scale-free, from right/left eigenvectors):

    capsize : real mode with the largest roll participation
    wobble  : oscillatory mode with the largest steer participation (>25 %)
    weave   : remaining oscillatory mode with the smallest tyre-lag participation
              (roll/yaw/lateral-dominated).  Where the weave merges with real
              tyre/lateral roots (~19 m/s here) it is followed by whichever
              oscillatory pair then carries the least tyre-lag participation.
    Returns None for a mode that does not exist (e.g. weave that has become
    two real roots at very low speed).
    """
    ev, P = _participation(A)
    idx = {n: i for i, n in enumerate(names)}
    roll = P[idx["phi"]] + P[idx["phidot"]]
    steer = P[idx["delta"]] + P[idx["deltadot"]]
    tyre = sum(P[idx[n]] for n in names if n.startswith("F_")) if any(
        n.startswith("F_") for n in names) else np.zeros(len(ev))

    def mk(name, k):
        e = ev[k]
        wn = abs(e)
        osc = abs(e.imag) > 1e-9 * max(1.0, wn)
        return Mode(name, e, float(e.real),
                    float(abs(e.imag) / (2 * np.pi)) if osc else np.nan,
                    float(-e.real / wn) if osc else np.nan)

    out: Dict[str, Optional[Mode]] = {m: None for m in MODES}
    real = [k for k in range(len(ev)) if abs(ev[k].imag) <= 1e-9 * max(1.0, abs(ev[k]))]
    osc = [k for k in range(len(ev)) if ev[k].imag > 1e-9 * max(1.0, abs(ev[k]))]
    if real:
        out["capsize"] = mk("capsize", max(real, key=lambda k: roll[k]))
    cand = list(osc)
    if cand:
        kw = max(cand, key=lambda k: steer[k])
        if steer[kw] > 0.25:
            out["wobble"] = mk("wobble", kw)
            cand.remove(kw)
    if cand:
        # weave family: smallest tyre-lag participation (the tyre-relaxation /
        # lateral-slip family carries the F states, the weave does not)
        kv = min(cand, key=lambda k: tyre[k] - 1e-3 * (roll[k] + P[idx["v"], k] + P[idx["r"], k]))
        out["weave"] = mk("weave", kv)
    return out


def _mk_mode(name, e):
    wn = abs(e)
    osc = abs(e.imag) > 1e-9 * max(1.0, wn)
    return Mode(name, e, float(e.real),
                float(abs(e.imag) / (2 * np.pi)) if osc else np.nan,
                float(-e.real / wn) if osc else np.nan)


def modes_at(lb: LinearBike, u: float) -> Dict[str, Optional[Mode]]:
    """Capsize / weave / wobble at one speed (see :func:`identify_modes`)."""
    A, names, _ = lb.state_space(u)
    return identify_modes(A, names)


# ---------------------------------------------------------------------------
def speed_sweep(lb: LinearBike, speeds: Iterable[float]) -> Dict[str, np.ndarray]:
    """Modes over a speed range.  Keys: u, and for each mode m in
    {capsize, weave, wobble}: m_re, m_freq, m_zeta (nan where not defined);
    plus 'eigs' (all eigenvalues, one row per speed, sorted by real part)."""
    speeds = np.asarray(list(speeds), float)
    out = {"u": speeds}
    for m in MODES:
        for s in ("re", "freq", "zeta"):
            out[f"{m}_{s}"] = np.full(speeds.size, np.nan)
    eigs = []
    for i, u in enumerate(speeds):
        A, names, _ = lb.state_space(u)
        modes_at_cached = identify_modes(A, names)
        for m in MODES:
            md = modes_at_cached[m]
            if md is not None:
                out[f"{m}_re"][i] = md.growth
                out[f"{m}_freq"][i] = md.frequency
                out[f"{m}_zeta"][i] = md.damping
        e = np.linalg.eigvals(A)
        eigs.append(e[np.argsort(-e.real)])
    out["eigs"] = np.array(eigs)
    return out


def critical_speeds(lb: LinearBike, mode: str, u_lo=5.0, u_hi=90.0, n=120):
    """Speeds at which the growth rate of ``mode`` changes sign
    (list of (speed, 'S->U' | 'U->S')), linearly interpolated on a tracked sweep."""
    sw = speed_sweep(lb, np.linspace(u_lo, u_hi, n))
    us, g = sw["u"], sw[f"{mode}_re"]
    res = []
    for i in range(n - 1):
        a, b = g[i], g[i + 1]
        if np.isnan(a) or np.isnan(b) or a * b > 0:
            continue
        res.append((float(us[i] + (0 - a) * (us[i + 1] - us[i]) / (b - a)),
                    "S->U" if b > a else "U->S"))
    return res


# ---------------------------------------------------------------------------
# model variants
# ---------------------------------------------------------------------------
def without_gyro(model: BikeModel, wheels: bool = True, powertrain: bool = True,
                 which_wheels: Tuple[str, ...] = ("front", "rear")) -> BikeModel:
    """Copy of the model with selected rotating inertias' spin switched off."""
    bodies = []
    for b in model.bodies:
        if wheels and b.spin_axis is not None:
            is_front = b.parent == "front"
            if (is_front and "front" in which_wheels) or (not is_front and "rear" in which_wheels):
                b = replace(b, spin_per_speed=0.0)
        bodies.append(b)
    return replace(model, bodies=bodies,
                   powertrain=None if powertrain else model.powertrain)


def shift_body(model: BikeModel, name: str, dx: float = 0.0, dz: float = 0.0,
               keep_com: bool = True) -> BikeModel:
    """Move body ``name`` by (dx, dz).  With ``keep_com`` the COM of the body
    called 'rear frame+rider' is moved the opposite way so the total COM is
    unchanged - this isolates the change of the inertia distribution."""
    bodies = []
    m_move = None
    for b in model.bodies:
        if b.name == name:
            m_move = b.mass
            b = replace(b, pos=(b.pos[0] + dx, b.pos[1], b.pos[2] + dz))
        bodies.append(b)
    if m_move is None:
        raise KeyError(name)
    if keep_com:
        out = []
        for b in bodies:
            if b.name == "rear frame+rider":
                f = m_move / b.mass
                b = replace(b, pos=(b.pos[0] - f * dx, b.pos[1], b.pos[2] - f * dz))
            out.append(b)
        bodies = out
    return replace(model, bodies=bodies)


def scale_rotor(model: BikeModel, factor: float, names: Optional[List[str]] = None) -> BikeModel:
    """Scale the axial inertia of powertrain rotors (all, or the named ones)."""
    pt = model.powertrain
    if pt is None:
        return model
    rotors = [replace(r, I=r.I * factor) if (names is None or r.name in names) else r
              for r in pt.rotors]
    return replace(model, powertrain=replace(pt, rotors=rotors))


def gyro_ratio(model: BikeModel, u: float = 1.0) -> Dict[str, float]:
    """Angular momenta [N m s] at speed u: front wheel, rear wheel, powertrain
    (net, signed) and the ratio powertrain / (front+rear wheels)."""
    g = model.geom
    Hf = Hr = 0.0
    for b in model.bodies:
        if b.spin_axis is not None:
            H = b.inertia[1, 1] * b.spin_per_speed * u
            if b.parent == "front":
                Hf += H
            else:
                Hr += H
    Hp = model.powertrain.angular_momentum(u, g.r_rear) if model.powertrain else 0.0
    return {"front_wheel": Hf, "rear_wheel": Hr, "powertrain": Hp,
            "ratio": Hp / (Hf + Hr) if (Hf + Hr) else np.nan}
