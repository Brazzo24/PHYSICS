"""
utils/params.py
===============
Bicycle and rider parameters from Table 1 of:

    Åström, Klein, Lennartsson (2005) — "Bicycle Dynamics and Control"
    IEEE Control Systems Magazine, 25(4), 26-47.

All SI units (kg, m, s, rad).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BicycleParams:
    """
    Geometric and inertial parameters for a bicycle (+optional rider).

    Notation follows the paper directly. Parameters marked with a comment
    reference the equation or table where they appear.
    """
    # ── Geometry ────────────────────────────────────────────────────────────
    b: float = 1.00        # Wheelbase [m]
    c: float = 0.08        # Trail [m]  (Fig. 1)
    lam: float = np.radians(70.0)  # Head angle λ [rad] (70° in Table 1)
    Rrw: float = 0.35      # Rear wheel radius [m]
    Rfw: float = 0.35      # Front wheel radius [m]

    # ── Combined frame + rider mass properties ───────────────────────────────
    # (rear frame with rider rigidly attached, Table 1)
    m: float = 87.0 + 2.0 + 1.5 + 1.5   # Total mass [kg]  (≈92 kg)
    h: float = 1.028       # Height of centre of mass [m]
    a: float = 0.492       # Longitudinal CoM distance from rear contact [m]
    J: float = 3.28        # Roll moment of inertia [kg·m²]  (Jxx rear frame)
    D: float = 0.603       # Inertia product |Jxz| [kg·m²]   (D = −Jxz)

    # ── Front-fork assembly ─────────────────────────────────────────────────
    mf: float = 2.0        # Front fork mass [kg]
    hf: float = 0.676      # CoM height of front fork [m]
    Jf: float = 0.08       # Roll inertia front fork [kg·m²]
    Jfw_yy: float = 0.14   # Spin inertia of front wheel [kg·m²]

    # ── Rear wheel ───────────────────────────────────────────────────────────
    Jrw_yy: float = 0.14   # Spin inertia of rear wheel [kg·m²]

    # ── Gravity ──────────────────────────────────────────────────────────────
    g: float = 9.81        # [m/s²]

    # ── Whipple 4th-order matrices (Eq. 24, numerically given in Eq. 25) ────
    # These are the bicycle-with-rider values from the paper.
    # Stored as 2×2 numpy arrays.
    M_w: np.ndarray = field(default_factory=lambda: np.array([
        [ 96.8,  -3.57],
        [ -3.57,  0.258]
    ]))
    C1_w: np.ndarray = field(default_factory=lambda: np.array([
        [  0.0,   -50.8],
        [  0.436,   2.20]
    ]))
    K0_w: np.ndarray = field(default_factory=lambda: np.array([
        [-901.0,  35.17],
        [  35.17, -12.03]
    ]))
    K2_w: np.ndarray = field(default_factory=lambda: np.array([
        [  0.0,  -87.06],
        [  0.0,    3.50]
    ]))

    # ── Optional rider lean sub-model (Eq. 22) ───────────────────────────────
    Jr: float = 1.0        # Rider upper-body roll inertia [kg·m²]  (approx)
    mr: float = 60.0       # Rider upper-body mass [kg]             (approx)
    hr: float = 0.6        # Rider upper-body CoM height [m]        (approx)


@dataclass
class BicycleParamsNoRider(BicycleParams):
    """
    Parameters for the bicycle WITHOUT a rider (values in parentheses
    in Table 1 of the paper).
    """
    m: float = 2.0 + 1.5 + 1.5   # frame + 2 wheels only  ≈ 5 kg
    h: float = 0.579
    a: float = 0.439
    J: float = 0.476
    D: float = 0.274

    M_w: np.ndarray = field(default_factory=lambda: np.array([
        [ 3.57,  -0.472],
        [-0.472,  0.152]
    ]))
    C1_w: np.ndarray = field(default_factory=lambda: np.array([
        [ 0.0,    -5.84],
        [ 0.436,   0.666]
    ]))
    K0_w: np.ndarray = field(default_factory=lambda: np.array([
        [-91.72,  7.51],
        [  7.51, -2.57]
    ]))
    K2_w: np.ndarray = field(default_factory=lambda: np.array([
        [ 0.0,   -9.54],
        [ 0.0,    0.848]
    ]))


# Convenience singletons
BIKE_WITH_RIDER    = BicycleParams()
BIKE_WITHOUT_RIDER = BicycleParamsNoRider()
