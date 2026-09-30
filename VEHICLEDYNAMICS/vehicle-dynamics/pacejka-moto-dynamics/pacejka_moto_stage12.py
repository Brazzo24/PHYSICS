"""
pacejka_moto_stage12.py
=======================
Pacejka "Tyre and Vehicle Dynamics" – Chapter 11, Motorcycle Dynamics
Stage 1 : Geometry & Kinematics   (§11.2.1 – §11.2.2)
Stage 2 : Tyre Forces              (Magic Formula Fy, Mx, Mz)

State vector (steady-state subset used here):
    phi   – mainframe roll angle          [rad]  (positive = lean right)
    delta – steer angle                   [rad]  (positive = steer right)
    u     – forward speed                 [m/s]
    v     – lateral speed                 [m/s]
    r     – yaw rate                      [rad/s]
    omega_prime – front-frame twist angle [rad]  (set 0 for rigid frame)

All angles in radians unless stated otherwise.
Sign convention follows Pacejka (2006) Ch.11.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# 1.  VEHICLE PARAMETERS
# ---------------------------------------------------------------------------

@dataclass
class MotorcycleParams:
    """
    Geometric and mass parameters for the four-body motorcycle model.
    Defaults are representative of a mid-size sports motorcycle.
    """
    # ---- masses [kg] -------------------------------------------------------
    mm: float = 180.0   # mainframe + rear wheel + lower rider
    mf: float =  10.0   # front upper frame
    ms: float =  15.0   # front subframe + front wheel
    mr: float =  70.0   # rider upper torso

    # ---- CoG heights above ground [m] (upright) ----------------------------
    hm: float = 0.58    # mainframe
    hf: float = 0.80    # front upper frame
    hs: float = 0.55    # front subframe
    hr: float = 0.90    # rider upper torso

    # ---- longitudinal CoG positions from rear axle [m] --------------------
    # (ac = distance from A to front axle, bc = distance from A to rear axle)
    ac: float = 0.75    # reference point A to front axle
    bc: float = 0.65    # reference point A to rear axle

    # ---- front-frame CoG offsets from A in x-direction [m] ----------------
    ef: float = 0.05    # front upper frame
    es: float = 0.03    # front subframe

    # ---- steering geometry -------------------------------------------------
    lam: float = np.radians(27.0)   # rake (steering head) angle  [rad]
    tc:  float = 0.10               # caster length  [m]

    # ---- tyre crown radii [m] ----------------------------------------------
    rc1: float = 0.06   # front tyre crown radius
    rc2: float = 0.08   # rear  tyre crown radius

    # ---- nominal tyre radii [m] --------------------------------------------
    r1: float = 0.305   # front wheel radius
    r2: float = 0.305   # rear  wheel radius

    # ---- Pacejka Magic Formula coefficients (same for both tyres here) -----
    # Lateral force  Fy = D*sin(C*atan(B*alpha - E*(B*alpha - atan(B*alpha))))
    # + camber thrust  Fy_gamma
    pCy1: float =  1.30   # shape factor C
    pDy1: float =  1.00   # peak factor scale
    pDy2: float = -0.08
    pEy1: float = -1.60   # curvature factor E
    pKy1: float = 21.0    # cornering stiffness factor
    pKy2: float =  2.0
    pKy3: float =  0.0
    # Camber thrust stiffness (linear approximation)
    pCy_gamma: float = 1.05  # camber force C
    pDy_gamma: float = 0.30  # camber force D (peak fraction)
    # Self-aligning moment  Mz = -t * Fy  (simplified; t = pneumatic trail)
    t0: float = 0.045   # nominal pneumatic trail  [m]
    # Overturning couple  Mx = -rc * Fz * sin(gamma)  (simplified)
    # (rc used per wheel from rc1/rc2 above)

    def __post_init__(self):
        self._recompute()

    def _recompute(self):
        """Derived geometry (Eq. 11.2)."""
        self.m   = self.mm + self.mf + self.ms + self.mr
        self.mmr = self.mm + self.mr
        self.l   = self.ac + self.bc

        b_num = (self.mmr * self.bc
                 + self.mf * (self.af_upright() + self.bc)
                 + self.ms * (self.as_upright() + self.bc))
        self.b = b_num / self.m
        self.a = self.l - self.b

        self.h = (self.hm*self.mm + self.hf*self.mf
                  + self.hs*self.ms + self.hr*self.mr) / self.m

        # weighted average crown radius
        self.rc = (self.b / self.l) * self.rc1 + (self.a / self.l) * self.rc2

    def af_upright(self) -> float:
        """Eq. 11.1 – af at phi=0."""
        return self.ac - (self.hf * np.sin(self.lam) - (self.ef + self.tc)) / np.cos(self.lam)

    def as_upright(self) -> float:
        """Eq. 11.1 – as at phi=0."""
        return self.ac - (self.hs * np.sin(self.lam) - (self.es + self.tc)) / np.cos(self.lam)

    def h_rolled(self, phi: float) -> float:
        """Effective CoG height at roll angle phi (Eq. 11.5)."""
        return self.h + self.rc * (1.0 - np.cos(phi)) / np.cos(phi)

    def tc_rolled(self, phi: float) -> float:
        """Effective caster length at roll angle phi (Eq. 11.6)."""
        return (self.tc
                + self.rc1 * np.sin(self.lam) * (1.0 - np.cos(phi)) / np.cos(phi))


# ---------------------------------------------------------------------------
# 2.  STAGE 1 – KINEMATICS: slip angles and camber angles
# ---------------------------------------------------------------------------

def front_wheel_spin_axis(phi: float, delta: float, lam: float,
                           omega_prime: float = 0.0) -> np.ndarray:
    """
    Unit vector s along the front wheel spin axis in the moving horizontal
    frame, via successive rotation matrices (Eq. 11.7 / 11.8).

    Rotations applied in order:
        s_y (initial) → R_x(phi) → R_z(psi=0) → R_z_lam(lam) →
        R_z_delta(delta) → R_x(omega_prime)

    Returns 3-element ndarray [sx, sy, sz].
    """
    # Start: unit vector along y-axis in wheel frame
    s = np.array([0.0, 1.0, 0.0])

    # 1. Twist angle (front-frame torsion, usually small / zero)
    ct, st = np.cos(omega_prime), np.sin(omega_prime)
    R_twist = np.array([[1,  0,   0 ],
                        [0,  ct, -st],
                        [0,  st,  ct]])
    s = R_twist @ s

    # 2. Steer angle about z_lambda axis (rotated by rake angle lambda)
    cd, sd = np.cos(delta), np.sin(delta)
    R_steer = np.array([[ cd, -sd, 0],
                        [ sd,  cd, 0],
                        [  0,   0, 1]])
    s = R_steer @ s

    # 3. Rake (inclination of steering axis in mainframe centre plane)
    cl, sl = np.cos(lam), np.sin(lam)
    R_rake = np.array([[ cl,  0, sl],
                       [  0,  1,  0],
                       [-sl,  0, cl]])
    s = R_rake @ s

    # 4. Roll of mainframe
    cp, sp = np.cos(phi), np.sin(phi)
    R_roll = np.array([[1,  0,   0 ],
                       [0,  cp, -sp],
                       [0,  sp,  cp]])
    s = R_roll @ s

    return s


def steer_and_camber_angles(phi: float, delta: float, lam: float,
                             omega_prime: float = 0.0,
                             linearise: bool = False
                             ) -> Tuple[float, float]:
    """
    Front wheel ground-steer angle eta and camber angle gamma1.

    Non-linear: Eq. 11.10 / 11.11
    Linear    : Eq. 11.12 / 11.13
    """
    if linearise:
        # Eq. 11.12 / 11.13
        eta    = delta * np.cos(lam) - phi * np.sin(lam)
        gamma1 = phi + delta * np.sin(lam) + omega_prime * np.cos(lam)
    else:
        s = front_wheel_spin_axis(phi, delta, lam, omega_prime)
        sx, sy, sz = s
        eta    = np.arctan2(-sx, sy)          # Eq. 11.10
        gamma1 = np.arcsin(sz)                # Eq. 11.11 (= arcsin(sz))
    return eta, gamma1


def slip_angles_steady(phi: float, delta: float, u: float,
                        v: float, r: float, p: MotorcycleParams,
                        omega_prime: float = 0.0
                        ) -> Tuple[float, float]:
    """
    Steady-state front (alpha1) and rear (alpha2) slip angles.
    Eq. 11.15 (non-linear steer angle) or 11.16 (dynamic, used here for
    the steady-state case by setting d/dt terms to zero).
    """
    eta, _ = steer_and_camber_angles(phi, delta, p.lam, omega_prime,
                                      linearise=False)
    alpha1 = eta - (1.0 / u) * (v - p.ac * r)
    alpha2 =      -(1.0 / u) * (v + p.bc * r)
    return alpha1, alpha2


def camber_angles_both(phi: float, delta: float, p: MotorcycleParams,
                        omega_prime: float = 0.0
                        ) -> Tuple[float, float]:
    """
    Front camber gamma1 (Eq. 11.11) and rear camber gamma2 = phi (Eq. 11.14).
    """
    _, gamma1 = steer_and_camber_angles(phi, delta, p.lam, omega_prime)
    gamma2    = phi   # Eq. 11.14
    return gamma1, gamma2


# ---------------------------------------------------------------------------
# 3.  VERTICAL WHEEL LOADS  (quasi-static, no acceleration here)
# ---------------------------------------------------------------------------

def normal_loads(p: MotorcycleParams,
                 ax: float = 0.0,
                 Fd: float = 0.0,
                 hd: float = 0.6) -> Tuple[float, float]:
    """
    Static + longitudinal load transfer (Eq. 11.21–11.24).
    ax  : longitudinal acceleration [m/s²]
    Fd  : aero drag force [N]
    hd  : drag application height [m]
    Returns (Fz1, Fz2) – front and rear vertical loads [N].
    """
    g = 9.81
    Fz1o = (p.b / p.l) * p.m * g
    Fz2o = (p.a / p.l) * p.m * g
    delta_Fz = (1.0 / p.l) * (hd * Fd + p.h * p.m * ax)
    Fz1 = Fz1o - delta_Fz
    Fz2 = Fz2o + delta_Fz
    return Fz1, Fz2


# ---------------------------------------------------------------------------
# 4.  STAGE 2 – MAGIC FORMULA TYRE MODEL
# ---------------------------------------------------------------------------

def _Dy(pDy1: float, pDy2: float, Fz: float, Fz0: float) -> float:
    """Peak lateral force factor D."""
    dfz = (Fz - Fz0) / Fz0
    return Fz * (pDy1 + pDy2 * dfz)


def _Ky(pKy1: float, pKy2: float, Fz: float, Fz0: float) -> float:
    """Cornering stiffness BCD."""
    return pKy1 * Fz0 * np.sin(2.0 * np.arctan(Fz / (pKy2 * Fz0)))


def magic_formula_Fy(alpha: float, gamma: float, Fz: float,
                      p: MotorcycleParams,
                      Fz0: float = 1500.0) -> float:
    """
    Lateral force via Pacejka Magic Formula including camber thrust.

    Fy = D * sin(C * atan(B*alpha_eq - E*(B*alpha_eq - atan(B*alpha_eq))))
         + Fy_gamma

    where Fy_gamma is the camber component (linearised).
    """
    C   = p.pCy1
    D   = _Dy(p.pDy1, p.pDy2, Fz, Fz0)
    BCD = _Ky(p.pKy1, p.pKy2, Fz, Fz0)
    B   = BCD / (C * D + 1e-9)
    E   = p.pEy1

    x = B * alpha
    Fy_slip = D * np.sin(C * np.arctan(x - E * (x - np.arctan(x))))

    # Camber thrust (simplified linear model)
    Fy_gamma = p.pDy_gamma * Fz * np.sin(p.pCy_gamma * np.arctan(gamma))

    return Fy_slip + Fy_gamma


def magic_formula_Mz(alpha: float, Fy: float,
                      p: MotorcycleParams,
                      t: float = None) -> float:
    """
    Self-aligning torque Mz ≈ -t * Fy  (simplified).
    t defaults to p.t0 (nominal pneumatic trail).
    """
    if t is None:
        t = p.t0
    return -t * Fy


def magic_formula_Mx(gamma: float, Fz: float,
                      rc: float) -> float:
    """
    Overturning couple Mx due to tyre crown radius and camber.
    Mx ≈ -rc * Fz * sin(gamma)  (Pacejka eq. approx.)
    """
    return -rc * Fz * np.sin(gamma)


# ---------------------------------------------------------------------------
# 5.  HIGH-LEVEL: FULL TYRE FORCE PACKAGE FOR ONE WHEEL
# ---------------------------------------------------------------------------

@dataclass
class TyreForces:
    Fy:  float   # lateral force       [N]
    Mz:  float   # self-aligning torque [N·m]
    Mx:  float   # overturning couple   [N·m]
    Fz:  float   # normal load          [N]
    alpha: float # slip angle           [rad]
    gamma: float # camber angle         [rad]


def compute_tyre_forces(alpha: float, gamma: float, Fz: float,
                         rc: float, p: MotorcycleParams,
                         Fz0: float = 1500.0) -> TyreForces:
    """Compute Fy, Mz, Mx for one wheel."""
    Fy = magic_formula_Fy(alpha, gamma, Fz, p, Fz0)
    Mz = magic_formula_Mz(alpha, Fy, p)
    Mx = magic_formula_Mx(gamma, Fz, rc)
    return TyreForces(Fy=Fy, Mz=Mz, Mx=Mx, Fz=Fz,
                      alpha=alpha, gamma=gamma)


# ---------------------------------------------------------------------------
# 6.  STEADY-STATE SOLVER  – given (phi, u, R) find delta and tyre forces
# ---------------------------------------------------------------------------

def steady_state_cornering(phi: float, u: float, R: float,
                            p: MotorcycleParams,
                            omega_prime: float = 0.0
                            ) -> dict:
    """
    For a given roll angle phi, speed u and radius R:
      - compute r = u/R  (yaw rate)
      - assume v ≈ 0  (small sideslip of vehicle reference point)
      - find delta from the kinematic steer angle (small-steer linearisation)
      - compute all tyre angles and forces

    Returns a dict with all quantities of interest.
    """
    g   = 9.81
    r   = u / R                         # yaw rate [rad/s]
    ay  = u * r                         # lateral acceleration [m/s²]
    v   = 0.0                           # lateral velocity at A (small)

    # Kinematic delta from lean equilibrium: tan(phi) ≈ ay/g
    # Steer angle from linearised ground-steer: eta ≈ delta*cos(lam) - phi*sin(lam)
    # and  eta ≈ l/R  (Ackermann)  →  solve for delta
    eta_kin  = p.l / R                  # kinematic ground steer angle
    delta    = (eta_kin + phi * np.sin(p.lam)) / np.cos(p.lam)

    # Camber & slip angles
    gamma1, gamma2 = camber_angles_both(phi, delta, p, omega_prime)
    alpha1, alpha2 = slip_angles_steady(phi, delta, u, v, r, p, omega_prime)

    # Normal loads (no braking/driving here)
    Fz1, Fz2 = normal_loads(p)

    # Tyre forces
    tf1 = compute_tyre_forces(alpha1, gamma1, Fz1, p.rc1, p)
    tf2 = compute_tyre_forces(alpha2, gamma2, Fz2, p.rc2, p)

    return dict(
        phi=phi, delta=delta, u=u, R=R, r=r, ay=ay,
        alpha1=alpha1, alpha2=alpha2,
        gamma1=gamma1, gamma2=gamma2,
        Fz1=Fz1, Fz2=Fz2,
        front=tf1, rear=tf2,
    )


# ---------------------------------------------------------------------------
# 7.  QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = MotorcycleParams()
    print("=== Motorcycle Parameters ===")
    print(f"  Total mass m  = {p.m:.1f} kg")
    print(f"  Wheelbase  l  = {p.l:.3f} m")
    print(f"  CoG height h  = {p.h:.3f} m")
    print(f"  CoG front  a  = {p.a:.3f} m  (from front axle)")
    print(f"  CoG rear   b  = {p.b:.3f} m  (from rear axle)")
    print(f"  Avg crown  rc = {p.rc:.4f} m")

    print("\n=== Geometry at phi=35°, delta=2° ===")
    phi   = np.radians(35.0)
    delta = np.radians(2.0)
    eta, g1 = steer_and_camber_angles(phi, delta, p.lam, linearise=False)
    print(f"  Ground steer angle eta = {np.degrees(eta):.3f}°")
    print(f"  Front camber  gamma1   = {np.degrees(g1):.3f}°")
    print(f"  Rear  camber  gamma2   = {np.degrees(phi):.3f}°")
    print(f"  CoG height (rolled)    = {p.h_rolled(phi):.4f} m")

    print("\n=== Steady-state cornering: u=30 m/s, R=100 m ===")
    phi_ss = np.arctan(30.0**2 / (100.0 * 9.81))   # equilibrium lean
    res    = steady_state_cornering(phi_ss, 30.0, 100.0, p)
    print(f"  phi   = {np.degrees(res['phi']):.2f}°")
    print(f"  delta = {np.degrees(res['delta']):.3f}°")
    print(f"  ay    = {res['ay']:.2f} m/s²  ({res['ay']/9.81:.2f} g)")
    print(f"  alpha1 (front slip) = {np.degrees(res['alpha1']):.3f}°")
    print(f"  alpha2 (rear  slip) = {np.degrees(res['alpha2']):.3f}°")
    print(f"  Fy1   = {res['front'].Fy:.1f} N")
    print(f"  Fy2   = {res['rear'].Fy:.1f} N")
    print(f"  Mz1   = {res['front'].Mz:.2f} N·m")
    print(f"  Mx1   = {res['front'].Mx:.2f} N·m")
