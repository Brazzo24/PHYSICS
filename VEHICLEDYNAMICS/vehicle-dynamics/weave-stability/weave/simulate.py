"""
weave.simulate
==============
Time-domain simulation of the linear model (exact zero-order-hold
discretisation via a matrix exponential - no SciPy needed).
"""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np

from .multibody import LinearBike


def expm(M: np.ndarray, terms: int = 24) -> np.ndarray:
    """Matrix exponential by scaling and squaring (Taylor series)."""
    nrm = np.linalg.norm(M, 1)
    s = max(0, int(np.ceil(np.log2(nrm))) + 2) if nrm > 0.5 else 0
    X = M / (2.0 ** s)
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, terms):
        term = term @ X / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def rider_roll_pd(lb: LinearBike, u: float, kp: float = 400.0, kd: float = 10.0) -> np.ndarray:
    """State-feedback row K (T = -K x) of a minimal 'rider': handlebar torque
    opposing lean angle and lean rate, T = -kp*phi - kd*phidot [N m/rad, N m s/rad].
    The defaults stabilise the (hands-off unstable) capsize mode at 20-60 m/s so
    that time simulations show the weave/wobble dynamics rather than a fall."""
    A, names, _ = lb.state_space(u)
    K = np.zeros((1, A.shape[0]))
    K[0, names.index("phi")] = kp
    K[0, names.index("phidot")] = kd
    return K


def closed_loop_eigs(lb: LinearBike, u: float, K: np.ndarray) -> np.ndarray:
    A, _, _ = lb.state_space(u)
    B = lb.steer_torque_input(u)
    return np.linalg.eigvals(A - B @ K)


def simulate(lb: LinearBike, u: float, t_end: float = 5.0, dt: float = 1e-3,
             torque: Optional[Callable[[float], float]] = None,
             x0: Optional[np.ndarray] = None,
             K: Optional[np.ndarray] = None) -> dict:
    """
    Simulate at constant speed u with an optional external handlebar torque
    T(t) and an optional rider feedback  T_rider = -K x  (see rider_roll_pd);
    without K the response is hands-off (and capsize grows, as on a real bike).

    States: v [m/s], r [rad/s], phi, delta [rad], phidot, deltadot, tyre-lag
    forces; plus integrated yaw angle psi and world lateral position Y
    (Ydot = v + u psi).  Returns a dict of arrays keyed by name (angles in rad).
    """
    A, names, _ = lb.state_space(u)
    B = lb.steer_torque_input(u)
    n = A.shape[0]
    # augment with psi and Y :  psidot = r,  Ydot = v + u psi
    Aa = np.zeros((n + 2, n + 2))
    Aa[:n, :n] = A
    Aa[n, 1] = 1.0
    Aa[n + 1, 0] = 1.0
    Aa[n + 1, n] = u
    Ba = np.vstack([B, [[0.0], [0.0]]])
    # ZOH discretisation with the augmented [[A, B],[0, 0]] exponential
    Mx = np.zeros((n + 3, n + 3))
    Mx[:n + 2, :n + 2] = Aa
    Mx[:n + 2, n + 2:] = Ba
    E = expm(Mx * dt)
    Ad, Bd = E[:n + 2, :n + 2], E[:n + 2, n + 2:]
    if K is not None:
        Kf = np.zeros((1, n + 2))
        Kf[0, :n] = K
        Ad = Ad - Bd @ Kf
    steps = int(round(t_end / dt))
    x = np.zeros(n + 2)
    if x0 is not None:
        x[:len(x0)] = x0
    X = np.zeros((steps + 1, n + 2))
    T = np.zeros(steps + 1)
    X[0] = x
    for k in range(steps):
        tk = k * dt
        Tk = torque(tk) if torque else 0.0
        T[k] = Tk
        x = Ad @ x + Bd[:, 0] * Tk
        X[k + 1] = x
    out = {"t": np.arange(steps + 1) * dt, "torque": T}
    for i, nm in enumerate(names + ["psi", "Y"]):
        out[nm] = X[:, i]
    return out
