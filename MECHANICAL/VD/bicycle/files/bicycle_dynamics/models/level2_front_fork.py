"""
models/level2_front_fork.py
============================
Level-2 bicycle model: Level-1 roll dynamics coupled with the
*static* torque balance of the front fork, giving a self-stabilising
bicycle that accepts handlebar torque T as input.

System
------
    Frame (Eq. 14):
        J φ'' − mgh φ = (DV sinλ / b) δ' + (m(V²h − acg) sinλ / b) δ

    Front fork static balance (Eq. 11):
        δ = k₁(V) T − k₂(V) φ

    with (Eqs. 12, 13):
        k₁(V) = b² / ((V² sinλ − bg cosλ) mac sinλ)
        k₂(V) = bg / (V² sinλ − bg cosλ)

Combined into one second-order ODE in φ with input T  (Eq. 23, simplified):
    J φ'' + [DVk₂(V)/b] φ' + [mV²hk₂(V)/b − mgh] φ
        = (DVk₁(V)/b) T' + (mV²k₁(V)/b) T

Self-alignment / critical velocities (Eqs. 10, 16):
    V_sa = sqrt(bg cotλ)    (front fork self-aligns)
    V_c  = sqrt(bg cotλ)    (== V_sa for the static model)

Stability conditions (Eqs. 16, 17):
    V > V_c   AND   bh > ac tanλ

Reference
---------
Åström, Klein, Lennartsson (2005), Eqs. (9)–(17), (23).
"""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from utils.params import BicycleParams, BIKE_WITH_RIDER


# ─────────────────────────────────────────────────────────────────────────────
# Front-fork gain functions  (Eqs. 12, 13)
# ─────────────────────────────────────────────────────────────────────────────

def _denom(V: float, p: BicycleParams) -> float:
    """V² sinλ − bg cosλ  — denominator in k₁, k₂."""
    return V**2 * np.sin(p.lam) - p.b * p.g * np.cos(p.lam)


def k1(V: float, p: BicycleParams) -> float:
    """
    k₁(V) = b² / ((V² sinλ − bg cosλ) · m·a·c·sinλ)   (Eq. 12)
    Gain from handlebar torque T to steer angle δ.
    Undefined (singular) at V = V_sa.
    """
    return p.b**2 / (_denom(V, p) * p.m * p.a * p.c * np.sin(p.lam))


def k2(V: float, p: BicycleParams) -> float:
    """
    k₂(V) = bg / (V² sinλ − bg cosλ)                   (Eq. 13)
    Negative feedback gain from tilt φ to steer angle δ.
    Positive and stabilising when V > V_sa.
    """
    return p.b * p.g / _denom(V, p)


def self_alignment_velocity(p: BicycleParams) -> float:
    """V_sa = sqrt(bg cotλ)  (Eq. 10)."""
    return np.sqrt(p.b * p.g / np.tan(p.lam))


def critical_velocity(p: BicycleParams) -> float:
    """
    V_c = sqrt(bg cotλ)  (Eq. 16).
    Equals V_sa in the static front-fork model.
    Second condition (Eq. 17): bh > ac tanλ must also hold.
    """
    return self_alignment_velocity(p)


def stability_condition_geometry(p: BicycleParams) -> bool:
    """Check geometric condition (Eq. 17): bh > ac tanλ."""
    return p.b * p.h > p.a * p.c * np.tan(p.lam)


# ─────────────────────────────────────────────────────────────────────────────
# State-space: combined frame + static front-fork (Eq. 23, simplified)
# ─────────────────────────────────────────────────────────────────────────────

def state_matrices(V: float,
                   p: BicycleParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (A, B_T, B_Td) for the *combined* second-order ODE:

        J φ'' + c1 φ' + c0 φ = b1 T' + b0 T

    Augmented state  x = [φ, φ', T]  so that T' = u (torque rate as input),
    yielding:
        ẋ = A x + B u

    Returns
    -------
    A   : 3×3
    B   : 3×1  (coefficient of scalar input u = dT/dt)
    C   : 1×3  (output = φ, i.e. C = [1, 0, 0])
    """
    _K1 = k1(V, p)
    _K2 = k2(V, p)

    # Coefficients from the combined ODE (Eq. 23, T-input, no rider lean)
    c1 =  p.D * V * _K2 / p.b                              # damping
    c0 =  p.m * V**2 * p.h * _K2 / p.b - p.m * p.g * p.h  # stiffness
    b1 =  p.D * V * _K1 / p.b                              # T' coefficient
    b0 =  p.m * V**2 * _K1 / p.b                           # T coefficient

    # Augmented: x = [φ, φ', T],  u = dT/dt
    A = np.array([
        [0.0,      1.0,          0.0   ],
        [-c0/p.J, -c1/p.J,       b0/p.J],
        [0.0,      0.0,          0.0   ],
    ])
    B = np.array([[0.0], [b1/p.J], [1.0]])
    C = np.array([[1.0, 0.0, 0.0]])
    return A, B, C


# ─────────────────────────────────────────────────────────────────────────────
# Steer angle from tilt  (closed-loop view, Eq. 11)
# ─────────────────────────────────────────────────────────────────────────────

def steer_from_tilt(phi: np.ndarray, T: np.ndarray,
                    V: float, p: BicycleParams) -> np.ndarray:
    """Recover steer angle δ = k₁(V)·T − k₂(V)·φ  (Eq. 11)."""
    return k1(V, p) * T - k2(V, p) * phi


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    t:     np.ndarray   # time [s]
    phi:   np.ndarray   # roll angle [rad]
    dphi:  np.ndarray   # roll rate  [rad/s]
    T:     np.ndarray   # handlebar torque [N·m]
    delta: np.ndarray   # steer angle [rad]  (derived via Eq. 11)


def simulate(
    V: float,
    t_span: Tuple[float, float],
    x0: Tuple[float, float, float],              # (φ₀, φ̇₀, T₀)
    torque_cmd: Callable[[float, np.ndarray], float],
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 2000,
) -> SimResult:
    """
    Simulate the Level-2 bicycle (frame + static front-fork).

    Parameters
    ----------
    V          : forward speed [m/s]
    t_span     : (t0, tf)
    x0         : initial state (φ, φ̇, T)
    torque_cmd : callable(t, x) → dT/dt  (rate of change of applied torque)
    p          : BicycleParams
    n_eval     : number of output points
    """
    A, B, _ = state_matrices(V, p)

    def rhs(t, x):
        u = torque_cmd(t, x)
        return (A @ x + B.flatten() * u).tolist()

    t_eval = np.linspace(*t_span, n_eval)
    sol = solve_ivp(rhs, t_span, list(x0), t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-11)

    T_arr = sol.y[2]
    phi   = sol.y[0]
    delta = steer_from_tilt(phi, T_arr, V, p)

    return SimResult(t=sol.t, phi=phi, dphi=sol.y[1], T=T_arr, delta=delta)


def simulate_free(
    V: float,
    t_span: Tuple[float, float],
    phi0: float,
    dphi0: float = 0.0,
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 3000,
) -> SimResult:
    """
    Simulate with zero applied torque (T = 0 throughout).
    Tests self-stabilisation for V > V_c.
    """
    return simulate(V, t_span, (phi0, dphi0, 0.0),
                    torque_cmd=lambda t, x: 0.0,
                    p=p, n_eval=n_eval)


def simulate_constant_torque(
    V: float,
    T_val: float,
    t_span: Tuple[float, float],
    phi0: float = 0.0,
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 2000,
) -> SimResult:
    """
    Apply a constant torque step T_val at t=0 (dT/dt = 0 after t=0).
    Reproduces the inverse-response behaviour in Fig. 5 of the paper.
    """
    def ctrl(t, x):
        # T is already in state; keep it clamped to T_val via a proportional
        # corrector:  dT/dt = large_gain * (T_val − T_current)
        T_current = x[2]
        return 500.0 * (T_val - T_current)   # effectively instantaneous step

    return simulate(V, t_span, (phi0, 0.0, 0.0),
                    torque_cmd=ctrl, p=p, n_eval=n_eval)
