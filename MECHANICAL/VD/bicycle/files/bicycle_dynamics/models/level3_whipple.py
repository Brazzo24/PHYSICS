"""
models/level3_whipple.py
=========================
Level-3 bicycle model: the full linearised fourth-order Whipple model.

    M q'' + C(V) q' + (K₀ + K₂ V²) q = f               (Eq. 24)

where  q = [φ, δ]ᵀ  (roll + steer angles),  f = [0, T]ᵀ.

The matrices M, C₁, K₀, K₂ are 2×2 and depend on the bicycle geometry
and mass distribution.  C(V) = V · C₁.

Numerical values from Eq. (25) / Table 1 are stored in utils.params.

This module provides:
  - `eigenvalues_at_speed(V)`          — 4 eigenvalues of the system
  - `stability_range(V_vec)`           — sweep to find self-stable window
  - `state_matrices(V)`                — (A, B) for ẋ = Ax + Bu  (u = T)
  - `simulate(...)`:                   — time-domain integration
  - `simulate_torque_step(...)`:       — reproduce Fig. 5 (inverse response)

Reference
---------
Åström, Klein, Lennartsson (2005), Eqs. (24)–(25); Table 1.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

from utils.params import BicycleParams, BIKE_WITH_RIDER


# ─────────────────────────────────────────────────────────────────────────────
# System matrices
# ─────────────────────────────────────────────────────────────────────────────

def K_total(V: float, p: BicycleParams) -> np.ndarray:
    """K₀ + K₂ V²"""
    return p.K0_w + p.K2_w * V**2


def C_total(V: float, p: BicycleParams) -> np.ndarray:
    """C(V) = V · C₁"""
    return V * p.C1_w


def state_matrices(V: float,
                   p: BicycleParams = BIKE_WITH_RIDER
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the 2nd-order matrix ODE (Eq. 24) to first-order form.

        ẋ = A x + B u

    State:   x = [φ, δ, φ', δ']ᵀ     (4×1)
    Input:   u = T  (scalar handlebar torque)

    Returns
    -------
    A : 4×4 state matrix
    B : 4×1 input vector
    """
    Minv = np.linalg.inv(p.M_w)
    C    = C_total(V, p)
    K    = K_total(V, p)

    # Partitioned form:  ẋ = [q', -M⁻¹(C q' + K q) + M⁻¹ f]
    A = np.block([
        [np.zeros((2, 2)),      np.eye(2)           ],
        [-Minv @ K,            -Minv @ C            ],
    ])

    # Input: f = [0, T]ᵀ  →  contribution to q'' is  M⁻¹ f
    f_col = np.array([[0.0], [1.0]])
    B = np.vstack([np.zeros((2, 1)), Minv @ f_col])

    return A, B


# ─────────────────────────────────────────────────────────────────────────────
# Eigenvalue analysis
# ─────────────────────────────────────────────────────────────────────────────

def eigenvalues_at_speed(V: float,
                          p: BicycleParams = BIKE_WITH_RIDER) -> np.ndarray:
    """Return the 4 eigenvalues of A(V) as a complex array."""
    A, _ = state_matrices(V, p)
    return eigvals(A)


def is_stable_at_speed(V: float, p: BicycleParams = BIKE_WITH_RIDER) -> bool:
    """True if ALL eigenvalues have negative real part."""
    return bool(np.all(eigenvalues_at_speed(V, p).real < 0))


def stability_range(V_vec: np.ndarray,
                    p: BicycleParams = BIKE_WITH_RIDER
                    ) -> Tuple[Optional[float], Optional[float]]:
    """
    Sweep over velocities and return the (V_low, V_high) window in
    which the bicycle is self-stable (all poles have Re < 0).

    Returns (None, None) if no stable window is found.
    """
    stable_mask = np.array([is_stable_at_speed(v, p) for v in V_vec])
    indices = np.where(stable_mask)[0]
    if len(indices) == 0:
        return None, None
    return float(V_vec[indices[0]]), float(V_vec[indices[-1]])


def eigenvalue_sweep(V_vec: np.ndarray,
                     p: BicycleParams = BIKE_WITH_RIDER) -> np.ndarray:
    """
    Returns shape (len(V_vec), 4) complex array of eigenvalues,
    sorted by imaginary part for consistent colouring.
    """
    eigs = np.array([eigenvalues_at_speed(v, p) for v in V_vec])
    # sort by imaginary part at each speed for smooth curves
    idx = np.argsort(eigs.imag, axis=1)
    return eigs[np.arange(len(V_vec))[:, None], idx]


# ─────────────────────────────────────────────────────────────────────────────
# Steady turning: path geometry
# ─────────────────────────────────────────────────────────────────────────────

def yaw_rate_from_steer(delta: float, V: float,
                         p: BicycleParams = BIKE_WITH_RIDER) -> float:
    """
    Kinematic yaw rate ψ' = V δ_eff / b  ≈  V δ / b
    (small-angle approximation, straight-line linearisation).
    """
    return V * delta / p.b


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain simulation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimResult:
    t:    np.ndarray    # time [s]
    phi:  np.ndarray    # roll angle  φ  [rad]
    delta: np.ndarray   # steer angle δ  [rad]
    dphi:  np.ndarray   # roll rate   φ' [rad/s]
    ddelta: np.ndarray  # steer rate  δ' [rad/s]
    # Path deviation (small-angle, Eq. after 20):
    eta:  np.ndarray    # lateral path deviation [m]


def _integrate(V, t_span, x0, torque_fn, p, n_eval):
    A, B = state_matrices(V, p)

    def rhs(t, x):
        u = torque_fn(t, x)
        return A @ x + B.flatten() * u

    t_eval = np.linspace(*t_span, n_eval)
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval,
                    method="RK45", rtol=1e-9, atol=1e-11)
    return sol


def simulate(
    V: float,
    t_span: Tuple[float, float],
    x0: Tuple[float, float, float, float],  # (φ, δ, φ', δ')
    torque_fn: Callable[[float, np.ndarray], float],
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 3000,
) -> SimResult:
    """
    General simulator for the Whipple 4th-order model.

    Parameters
    ----------
    V         : forward speed [m/s] (constant)
    t_span    : (t0, tf)
    x0        : initial state [φ₀, δ₀, φ̇₀, δ̇₀]
    torque_fn : callable(t, x) → T  [N·m]
    p         : BicycleParams
    n_eval    : output resolution
    """
    sol = _integrate(V, t_span, list(x0), torque_fn, p, n_eval)

    phi    = sol.y[0]
    delta  = sol.y[1]
    dphi   = sol.y[2]
    ddelta = sol.y[3]

    # Integrate path deviation η: dη/dt = V ψ,  dψ/dt = V δ / b  (linear)
    psi = np.zeros(len(sol.t))
    eta = np.zeros(len(sol.t))
    for i in range(1, len(sol.t)):
        dt = sol.t[i] - sol.t[i-1]
        psi[i] = psi[i-1] + (V / p.b) * delta[i-1] * dt
        eta[i] = eta[i-1] + V * psi[i-1] * dt

    return SimResult(t=sol.t, phi=phi, delta=delta,
                     dphi=dphi, ddelta=ddelta, eta=eta)


def simulate_free(
    V: float,
    phi0: float = 0.05,
    dphi0: float = 0.0,
    t_span: Tuple[float, float] = (0.0, 10.0),
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 3000,
) -> SimResult:
    """Riderless / free simulation (T = 0). Tests self-stabilisation."""
    return simulate(V, t_span, (phi0, 0.0, dphi0, 0.0),
                    torque_fn=lambda t, x: 0.0, p=p, n_eval=n_eval)


def simulate_torque_step(
    V: float,
    T_step: float = 1.0,
    t_span: Tuple[float, float] = (0.0, 3.0),
    p: BicycleParams = BIKE_WITH_RIDER,
    n_eval: int = 3000,
) -> SimResult:
    """
    Apply a constant torque step T_step at t = 0.
    Reproduces the inverse-response of Fig. 5 of the paper.
    """
    return simulate(V, t_span, (0.0, 0.0, 0.0, 0.0),
                    torque_fn=lambda t, x: T_step, p=p, n_eval=n_eval)
