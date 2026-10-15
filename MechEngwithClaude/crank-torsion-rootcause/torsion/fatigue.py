"""Rainflow counting, mean-stress correction and Miner damage for periodic stress signals."""
from __future__ import annotations

import numpy as np

from .config import Material


def _reversals(x):
    nz = np.r_[True, np.diff(x) != 0]           # drop flat steps
    x = x[nz]
    d = np.diff(x)
    if len(d) < 2:
        return x
    turn = np.r_[True, d[1:] * d[:-1] < 0, True]
    return x[turn]


def rainflow_periodic(x: np.ndarray):
    """Closed cycles (amplitude, mean) of ONE period of a periodic signal (ASTM E1049, 3-point)."""
    k = int(np.argmax(x))
    x = np.r_[x[k:], x[:k], x[k]]                # start and end at the absolute maximum
    r = _reversals(x)
    stack, amps, means = [], [], []
    for v in r:
        stack.append(v)
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X < Y:
                break
            amps.append(Y / 2)
            means.append((stack[-2] + stack[-3]) / 2)
            del stack[-3:-1]
    return np.array(amps), np.array(means)


def equivalent_amplitude(tau_a, tau_m, mat: Material):
    """FKM-like linear mean-stress correction (tension-side mean only increases damage)."""
    return tau_a + mat.mean_sens_M * np.maximum(tau_m, 0.0)


def cycles_to_failure(tau_a_eq, mat: Material):
    ta = np.maximum(np.asarray(tau_a_eq, float), 1e-9)
    k = mat.k_slope
    N = mat.N_D * (mat.tau_w_MPa / ta) ** k
    if mat.haibach:
        N_low = mat.N_D * (mat.tau_w_MPa / ta) ** (2 * k - 1)
        N = np.where(ta < mat.tau_w_MPa, N_low, N)
    else:
        N = np.where(ta < mat.tau_w_MPa, np.inf, N)
    return N


def damage_per_period(tau: np.ndarray, mat: Material):
    a, m = rainflow_periodic(tau)
    if len(a) == 0:
        return 0.0, 0.0
    aeq = equivalent_amplitude(a, m, mat)
    return float(np.sum(1.0 / cycles_to_failure(aeq, mat))), float(aeq.max())
