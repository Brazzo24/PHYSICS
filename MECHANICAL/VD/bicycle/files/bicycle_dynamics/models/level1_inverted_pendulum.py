"""
models/level1_inverted_pendulum.py
===================================
Level-1 bicycle model: linearised inverted-pendulum roll dynamics
with steer angle δ as the direct control input.

    J φ'' − mgh φ = (DV/b) δ' + (mV²h/b) δ          (Eq. 1)

States  : x = [φ, φ']
Input   : δ  (steer angle, can be a time-series from a controller)
Output  : x(t)

Key results from the paper reproduced here:
  - Open-loop poles  : ±sqrt(mgh/J)           (Eq. 2)
  - Zero             : −mVh/D ≈ −V/a          (Eq. 3)
  - Transfer function: Gφδ(s)                 (Eq. 4)
  - Stabilising P-control with δ = −k2·φ      (Eq. 5, 6)

Reference
---------
Åström, Klein, Lennartsson (2005), Eqs. (1)–(6).
"""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from utils.params import BicycleParams, BIKE_WITH_RIDER


# ─────────────────────────────────────────────────────────────────────────────
# Transfer-function poles & zero  (analytical, Eqs. 2–4)
# ─────────────────────────────────────────────────────────────────────────────

def open_loop_poles(p: BicycleParams) -> Tuple[float, float]:
    """
    Open-loop pendulum poles (Eq. 2):
        p₁,₂ = ±sqrt(mgh / J)
    """
    val = np.sqrt(p.m * p.g * p.h / p.J)
    return (-val, val)   # (stable, unstable)


def transfer_function_zero(V: float, p: BicycleParams) -> float:
    """
    RHP zero of Gφδ(s) (Eq. 3):
        z = −mVh / D  ≈  −V/a
    Note: negative ⟹ zero in left half-plane for positive V
          (minimum-phase in the steer-angle → roll transfer).
    """
    return -p.m * V * p.h / p.D


def gain_V(V: float, p: BicycleParams) -> float:
    """DC-like gain prefactor V·D / (b·J) from Eq. (4)."""
    return V * p.D / (p.b * p.J)


# ─────────────────────────────────────────────────────────────────────────────
# State-space  (continuous time)
# ─────────────────────────────────────────────────────────────────────────────

def state_matrices(V: float, p: BicycleParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (A, B) for   ẋ = A x + B u,  x = [φ, φ'], u = δ

    From Eq. (1), rewritten as:
        φ'' = (mgh/J) φ + (DV/bJ) δ' + (mV²h/bJ) δ

    We treat δ as a differentiable signal and introduce an augmented
    state  [φ, φ', δ]  to absorb the δ' term cleanly, giving a 3rd
    order system:

        d/dt [φ, φ', δ] = A3 [φ, φ', δ] + B3 u

    where u = δ_cmd (desired steer angle, assumed piecewise smooth).

    For stability analysis we often just need the 2×2 system assuming
    δ = const (quasi-static), returned when augment=False.
    """
    mgh_J   = p.m * p.g * p.h / p.J
    DV_bJ   = p.D * V / (p.b * p.J)
    mV2h_bJ = p.m * V**2 * p.h / (p.b * p.J)

    # ── Augmented 3-state form: x = [φ, φ', δ] ──────────────────────────────
    A = np.array([
        [0.0,      1.0,        0.0     ],
        [mgh_J,    0.0,        mV2h_bJ ],
        [0.0,      0.0,        0.0     ],   # δ updated by input
    ])
    B = np.array([
        [0.0],
        [DV_bJ],
        [1.0],
    ])
    return A, B


def closed_loop_matrices_P(V: float, k2: float,
                            p: BicycleParams) -> np.ndarray:
    """
    Closed-loop A matrix under proportional steer-angle feedback
        δ = −k2 · φ                          (Eq. 5)
    Returns the 2×2 A_cl for states [φ, φ'].

    Stability condition (Eq. 6):  k2 > b·g / V²
    """
    mgh_J   = p.m * p.g * p.h / p.J
    DV_bJ   = p.D * V / (p.b * p.J)
    mV2h_bJ = p.m * V**2 * p.h / (p.b * p.J)

    # Substitute δ = −k2 φ  →  δ' = −k2 φ'
    a21 = mgh_J - mV2h_bJ * k2
    a22 = -DV_bJ * k2

    A_cl = np.array([
        [0.0,  1.0],
        [a21,  a22],
    ])
    return A_cl


def minimum_gain_for_stability(V: float, p: BicycleParams) -> float:
    """
    Minimum proportional gain k2 needed to stabilise the P-controlled
    bicycle (from Eq. 6):
        k2_min = b·g / V²
    """
    return p.b * p.g / V**2


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    t:   np.ndarray   # time  [s]
    phi: np.ndarray   # roll angle  [rad]
    dphi: np.ndarray  # roll rate   [rad/s]
    delta: np.ndarray # steer angle [rad]


def simulate(
    V: float,
    t_span: Tuple[float, float],
    x0: Tuple[float, float, float],          # (φ₀, φ̇₀, δ₀)
    controller: Optional[Callable[[float, np.ndarray], float]] = None,
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 2000,
) -> SimResult:
    """
    Simulate the Level-1 bicycle model (Eq. 1).

    Parameters
    ----------
    V          : forward speed [m/s], assumed constant
    t_span     : (t_start, t_end)
    x0         : initial state (φ, φ̇, δ)
    controller : callable(t, x) → δ_cmd  (if None, δ is held constant at x0[2])
    p          : BicycleParams
    n_eval     : number of output time points

    Returns
    -------
    SimResult
    """
    A, B = state_matrices(V, p)

    def rhs(t, x):
        delta_cmd = controller(t, x) if controller is not None else x[2]
        return (A @ x + B.flatten() * delta_cmd).tolist()

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(rhs, t_span, list(x0), t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-10)

    return SimResult(
        t=sol.t,
        phi=sol.y[0],
        dphi=sol.y[1],
        delta=sol.y[2],
    )


def simulate_P_control(
    V: float,
    k2: float,
    t_span: Tuple[float, float],
    phi0: float,
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 2000,
) -> SimResult:
    """Simulate with δ = −k2·φ  (Eq. 5)."""
    def ctrl(t, x):
        phi, dphi, _ = x
        return -k2 * phi
    return simulate(V, t_span, (phi0, 0.0, 0.0), controller=ctrl, p=p,
                    n_eval=n_eval)
