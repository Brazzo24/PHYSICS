"""Lumped torsional model: modes, FE tuning, steady-state harmonic response."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .config import Chain, Engine


@dataclass
class Model:
    names: list[str]
    M: np.ndarray
    K: np.ndarray
    springs: list[tuple[int, int, float]]
    omega: np.ndarray          # rad/s, ascending (mode 0 = rigid body if free)
    Phi: np.ndarray            # mass-normalised mode shapes, columns
    zeta: np.ndarray
    C: np.ndarray
    output: int
    stiffness_scale: float = 1.0

    @property
    def f_hz(self):
        return self.omega / (2 * np.pi)

    def idx(self, name):
        return self.names.index(name)

    def spring_index(self, a, b):
        ia, ib = self.idx(a), self.idx(b)
        for s, (i, j, _) in enumerate(self.springs):
            if {i, j} == {ia, ib}:
                return s
        raise KeyError(f"no spring {a}-{b}")


def _assemble(ch: Chain, eng: Engine, scale: float) -> Model:
    names = list(ch.nodes)
    J = np.array([ch.nodes[n] for n in names], float)
    if ch.add_recip_inertia:
        r = eng.r_mm * 1e-3
        for c in eng.cylinders:
            J[names.index(c.node)] += eng.m_osc_g * 1e-3 * r**2 / 2
    n = len(names)
    K = np.zeros((n, n))
    springs = []
    for s in ch.springs:
        i, j = names.index(s.a), names.index(s.b)
        k = s.k_Nm_rad * (scale if s.tunable else 1.0)
        K[[i, j], [i, j]] += k
        K[i, j] -= k
        K[j, i] -= k
        springs.append((i, j, k))
    M = np.diag(J)
    Mh = np.diag(1 / np.sqrt(J))
    w2, V = np.linalg.eigh(Mh @ K @ Mh)
    w2 = np.clip(w2, 0, None)
    omega = np.sqrt(w2)
    Phi = Mh @ V
    zeta = np.where(omega > 1e-6 * omega.max(), ch.zeta, 0.0)
    C = M @ Phi @ np.diag(2 * zeta * omega) @ Phi.T @ M
    return Model(names, M, K, springs, omega, Phi, zeta, C, names.index(ch.output_node), scale)


def build_model(ch: Chain, eng: Engine) -> Model:
    """Assemble and (optionally) scale the tunable springs to hit the FE frequency."""
    scale = 1.0
    m = _assemble(ch, eng, scale)
    if ch.tune_to_fe_hz:
        for _ in range(50):
            f = m.f_hz[ch.tune_mode]
            if abs(f / ch.tune_to_fe_hz - 1) < 1e-8:
                break
            scale *= (ch.tune_to_fe_hz / f) ** 2
            m = _assemble(ch, eng, scale)
    return m


def scaled_frequency(ch: Chain, factor: float) -> Chain:
    """Copy of the chain whose tuned target frequency is multiplied by ``factor``."""
    ch = copy.deepcopy(ch)
    if ch.tune_to_fe_hz:
        ch.tune_to_fe_hz *= factor
    else:
        for s in ch.springs:
            s.k_Nm_rad *= factor**2
    return ch


def harmonic_response(m: Model, F: np.ndarray, Omega: float) -> np.ndarray:
    """Complex angle amplitudes for complex node torques F at circular frequency Omega."""
    Z = m.K - Omega**2 * m.M + 1j * Omega * m.C
    return np.linalg.solve(Z, F)


def modal_response(m: Model, F: np.ndarray, Omega: float) -> np.ndarray:
    """Per-mode contribution to the angles, shape [n_nodes, n_modes] (sum = harmonic_response)."""
    q = (m.Phi.T @ F) / (m.omega**2 - Omega**2 + 2j * m.zeta * m.omega * Omega)
    return m.Phi * q[None, :]


def static_spring_torques(m: Model, Fmean: np.ndarray) -> np.ndarray:
    """Mean spring torques with the total mean torque reacted at the output node."""
    keep = [i for i in range(len(m.names)) if i != m.output]
    x = np.zeros(len(m.names))
    x[keep] = np.linalg.solve(m.K[np.ix_(keep, keep)], Fmean[keep])
    return np.array([k * (x[i] - x[j]) for i, j, k in m.springs])
