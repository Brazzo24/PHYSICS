"""
pacejka_moto_stage3.py
======================
Pacejka "Tyre and Vehicle Dynamics" – Chapter 11, Motorcycle Dynamics
Stage 3 : Linearised Equations of Motion + Eigenvalue / Stability Analysis

Builds directly on pacejka_moto_stage12 (imported as sibling module).

State vector  x = [phi, p, delta, delta_dot, v, r]
  phi       – mainframe roll angle            [rad]
  p         – roll rate  d(phi)/dt            [rad/s]
  delta     – steer angle                     [rad]
  delta_dot – steer rate                      [rad/s]
  v         – lateral velocity at ref. pt A   [m/s]
  r         – yaw rate                        [rad/s]

The linearised model follows the Newton-Euler treatment in Pacejka Ch.11
(section 11.3), reduced to the rigid-rider, rigid-frame case.
Eigenvalues reproduce the three classical modes:
  Capsize  - real eigenvalue, slow (un)stable
  Weave    - complex pair, ~2-4 Hz, stable above weave speed
  Wobble   - complex pair, ~8-12 Hz, speed-dependent damping
"""

import numpy as np
from dataclasses import dataclass
from typing import List, NamedTuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pacejka_moto_stage12 import (
    MotorcycleParams, _Ky, normal_loads
)


# ---------------------------------------------------------------------------
# 1.  LINEARISED TYRE STIFFNESSES
# ---------------------------------------------------------------------------

def cornering_stiffness(Fz: float, p: MotorcycleParams,
                         Fz0: float = 1500.0) -> float:
    """Cy = dFy/dalpha at alpha=0 (= BCD from Magic Formula)."""
    return _Ky(p.pKy1, p.pKy2, Fz, Fz0)


def camber_stiffness(Fz: float, p: MotorcycleParams) -> float:
    """Cgamma = dFy/dgamma at gamma=0."""
    return p.pDy_gamma * Fz * p.pCy_gamma


# ---------------------------------------------------------------------------
# 2.  INERTIA PARAMETERS
# ---------------------------------------------------------------------------

@dataclass
class InertiaParams:
    mm0:      float   # mainframe + rider mass          [kg]
    hm0:      float   # combined CoG height             [m]
    Imx0:     float   # roll moment of inertia          [kg m2]
    Imxz0:    float   # roll-yaw product of inertia     [kg m2]
    mf_tot:   float   # front assembly mass (mf+ms)     [kg]
    If_steer: float   # steering-axis moment of inertia [kg m2]
    Iwy1:     float   # front wheel spin inertia        [kg m2]
    Iwy2:     float   # rear  wheel spin inertia        [kg m2]


def build_inertia(p: MotorcycleParams,
                  Irx:      float = 3.0,
                  Irxz:     float = 0.1,
                  Imx:      float = 28.0,
                  Imxz:     float = -2.5,
                  If_steer: float = 0.25,
                  Iwy1:     float = 0.6,
                  Iwy2:     float = 0.9) -> InertiaParams:
    """
    Assemble InertiaParams. Defaults representative of mid-size sports bike.
    """
    mm0  = p.mm + p.mr
    hm0  = (p.hm * p.mm + p.hr * p.mr) / mm0
    Imx0 = (Imx + p.hm**2 * p.mm + Irx + p.hr**2 * p.mr - hm0**2 * mm0)
    Imxz0 = Imxz + Irxz
    return InertiaParams(
        mm0=mm0, hm0=hm0, Imx0=Imx0, Imxz0=Imxz0,
        mf_tot=p.mf + p.ms, If_steer=If_steer,
        Iwy1=Iwy1, Iwy2=Iwy2
    )


# ---------------------------------------------------------------------------
# 3.  A-MATRIX ASSEMBLY
# ---------------------------------------------------------------------------

def build_A_matrix(u: float, p: MotorcycleParams,
                   ip: InertiaParams, Fz0: float = 1500.0,
                   cs: float = 15.0) -> np.ndarray:
    """
    cs : steering damper coefficient [N*m*s/rad] — adds -cs/Ids to A[3,3].
         Set cs=0 to study the undamped system.
    """
    """
    6x6 state matrix for x = [phi, p, delta, delta_dot, v, r].

    Four dynamics + two kinematics:
      [0]  phi_dot   = p
      [1]  p_dot     = roll equation
      [2]  delta_dot = delta_dot
      [3]  ddot_dot  = steer equation
      [4]  v_dot     = lateral Newton
      [5]  r_dot     = yaw equation
    """
    g   = 9.81
    lam = p.lam
    tc  = p.tc
    l   = p.l
    h   = p.h
    cl  = np.cos(lam)
    sl  = np.sin(lam)

    # Tyre loads and stiffnesses at operating point
    Fz1, Fz2 = normal_loads(p)
    Cy1 = cornering_stiffness(Fz1, p, Fz0)
    Cy2 = cornering_stiffness(Fz2, p, Fz0)
    Cg1 = camber_stiffness(Fz1, p)
    Cg2 = camber_stiffness(Fz2, p)
    t1  = p.t0
    t2  = p.t0 * 0.85

    # -------------------------------------------------------------------
    # Linearised tyre force partials  (Eq. 11.12-11.16 at phi=delta=0)
    #
    # Front slip:   alpha1 = delta*cl - phi*sl - (v - ac*r)/u
    # Front camber: gamma1 = phi + delta*sl
    # Fy1 = Cy1*alpha1 + Cg1*gamma1
    #
    # Rear slip:    alpha2 = -(v + bc*r)/u
    # Rear camber:  gamma2 = phi
    # Fy2 = Cy2*alpha2 + Cg2*phi
    # -------------------------------------------------------------------
    dFy1_phi   = -Cy1*sl + Cg1
    dFy1_delta =  Cy1*cl + Cg1*sl
    dFy1_v     = -Cy1 / u
    dFy1_r     =  Cy1 * p.ac / u

    dFy2_phi   =  Cg2
    dFy2_delta =  0.0
    dFy2_v     = -Cy2 / u
    dFy2_r     = -Cy2 * p.bc / u

    dFy_phi   = dFy1_phi   + dFy2_phi
    dFy_delta = dFy1_delta + dFy2_delta
    dFy_v     = dFy1_v     + dFy2_v
    dFy_r     = dFy1_r     + dFy2_r

    # Aligning torques  Mz = -t*Fy
    dMz1_phi   = -t1 * dFy1_phi;   dMz1_delta = -t1 * dFy1_delta
    dMz1_v     = -t1 * dFy1_v;     dMz1_r     = -t1 * dFy1_r
    dMz2_phi   = -t2 * dFy2_phi;   dMz2_delta = -t2 * dFy2_delta
    dMz2_v     = -t2 * dFy2_v;     dMz2_r     = -t2 * dFy2_r

    # Tyre overturning couples  Mx = -rc*Fz*gamma  (linearised)
    dMx_phi = -(p.rc1*Fz1 + p.rc2*Fz2)

    # Gyroscopic coefficients
    gyro1 = ip.Iwy1 * u / p.r1
    gyro2 = ip.Iwy2 * u / p.r2

    # Effective inertias
    m   = p.m
    Ixx = ip.Imx0
    Ixz = ip.Imxz0
    Izz = m * p.a * p.b           # yaw inertia approx (parallel-axis point masses)
    Ids = ip.If_steer

    # Steer-axis castoring gravity restoring torque (front assembly only)
    e_front = (p.ef + p.es) / 2.0
    steer_grav = ip.mf_tot * g * e_front * sl   # [N m / rad]

    # -------------------------------------------------------------------
    A = np.zeros((6, 6))

    # Row 0: phi_dot = p  (kinematics)
    A[0, 1] = 1.0

    # Row 2: delta_dot = delta_dot  (kinematics)
    A[2, 3] = 1.0

    # Row 1: roll  (Ixx * p_dot = ...)
    # Driving terms: gravity overturning (+), tyre forces (+), Mx (+), gyro (+)
    A[1, 0] = (m*g*h + h*dFy_phi   + dMx_phi) / Ixx
    A[1, 2] = (        h*dFy_delta           ) / Ixx
    A[1, 4] = (        h*dFy_v               ) / Ixx
    A[1, 5] = (-m*u*h + h*dFy_r + gyro1*cl + gyro2) / Ixx

    # Row 4: lateral Newton  (m * v_dot = Fy - m*u*r)
    A[4, 0] = dFy_phi   / m
    A[4, 2] = dFy_delta / m
    A[4, 4] = dFy_v     / m
    A[4, 5] = (dFy_r - m*u) / m

    # Row 5: yaw  (Izz * r_dot = ...)
    A[5, 0] = (p.ac*dFy1_phi  - p.bc*dFy2_phi  + dMz1_phi  + dMz2_phi ) / Izz
    A[5, 1] = -(gyro1 + gyro2) / Izz
    A[5, 2] = (p.ac*dFy1_delta                  + dMz1_delta           ) / Izz
    A[5, 4] = (p.ac*dFy1_v    - p.bc*dFy2_v    + dMz1_v    + dMz2_v   ) / Izz
    A[5, 5] = (p.ac*dFy1_r    - p.bc*dFy2_r    + dMz1_r    + dMz2_r   ) / Izz

    # Row 3: steer  (Ids * delta_ddot = ...)
    # Restoring: aligning torque + caster force (projected along steer axis)
    # Excitation: front-assembly castoring gravity + gyroscopic coupling
    caster = t1*cl + tc   # effective moment arm along steer axis
    A[3, 0] = ( steer_grav - caster * dFy1_phi  ) / Ids
    A[3, 1] = -(gyro1 * cl)                        / Ids
    A[3, 2] = (             -caster * dFy1_delta) / Ids
    A[3, 4] = (             -caster * dFy1_v    ) / Ids
    A[3, 5] = (             -caster * dFy1_r    ) / Ids

    # Steering damper (standard on real motorcycles)
    A[3, 3] -= cs / Ids

    return A


# ---------------------------------------------------------------------------
# 4.  EIGENVALUE ANALYSIS AND MODE CLASSIFICATION
# ---------------------------------------------------------------------------

class Mode(NamedTuple):
    name:       str
    eigenvalue: complex
    frequency:  float    # Hz — NaN for real modes
    damping:    float    # damping ratio — NaN for real modes
    growth:     float    # real part lambda [1/s];  >0 = unstable
    stable:     bool


def classify_modes(eigenvalues: np.ndarray) -> List[Mode]:
    """Classify eigenvalues as Capsize / Weave / Wobble / Real."""
    thresh = 0.05
    real_eigs = [e for e in eigenvalues
                 if abs(e.imag) < thresh * (abs(e.real) + 1e-6)]
    osc_eigs  = sorted([e for e in eigenvalues if e.imag > thresh],
                        key=lambda e: abs(e.imag))

    modes = []
    for e in real_eigs:
        modes.append(Mode('Real', e, float('nan'), float('nan'),
                          e.real, e.real < 0))
    for e in osc_eigs:
        wn   = abs(e)
        freq = wn / (2*np.pi)
        zeta = -e.real / wn if wn > 0 else 0.0
        modes.append(Mode('Osc', e, freq, zeta, e.real, e.real < 0))

    osc   = sorted([m for m in modes if not np.isnan(m.frequency)],
                    key=lambda m: m.frequency)
    reals = sorted([m for m in modes if     np.isnan(m.frequency)],
                    key=lambda m: abs(m.growth))

    named = []
    for m, name in zip(osc,   ['Weave', 'Wobble'][:len(osc)]):
        named.append(m._replace(name=name))
    for m, name in zip(reals, (['Capsize'] + ['Real']*10)[:len(reals)]):
        named.append(m._replace(name=name))
    return named


def stability_analysis(u: float, p: MotorcycleParams,
                        ip: InertiaParams, cs: float = 15.0) -> List[Mode]:
    """Full pipeline: A -> eigenvalues -> classified modes."""
    return classify_modes(np.linalg.eigvals(build_A_matrix(u, p, ip, cs=cs)))


# ---------------------------------------------------------------------------
# 5.  SPEED SWEEP
# ---------------------------------------------------------------------------

def speed_sweep(u_range: np.ndarray, p: MotorcycleParams,
                ip: InertiaParams, cs: float = 15.0) -> dict:
    """Eigenvalue sweep; returns dict of float arrays."""
    keys = ['u','capsize_re','weave_re','weave_freq','weave_damp',
            'wobble_re','wobble_freq','wobble_damp']
    out  = {k: [] for k in keys}

    for u in u_range:
        modes = stability_analysis(u, p, ip, cs=cs)
        cap = next((m for m in modes if m.name == 'Capsize'), None)
        wea = next((m for m in modes if m.name == 'Weave'),   None)
        wob = next((m for m in modes if m.name == 'Wobble'),  None)
        out['u'].append(u)
        out['capsize_re'].append(cap.growth     if cap else float('nan'))
        out['weave_re'].append(  wea.growth     if wea else float('nan'))
        out['weave_freq'].append(wea.frequency  if wea else float('nan'))
        out['weave_damp'].append(wea.damping    if wea else float('nan'))
        out['wobble_re'].append( wob.growth     if wob else float('nan'))
        out['wobble_freq'].append(wob.frequency if wob else float('nan'))
        out['wobble_damp'].append(wob.damping   if wob else float('nan'))

    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 6.  SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p  = MotorcycleParams()
    ip = build_inertia(p)

    print("=== Inertia ===")
    print(f"  Ixx={ip.Imx0:.2f} kg*m2   Izz_approx={p.m*p.a*p.b:.2f} kg*m2")
    print(f"  gyro1(20m/s)={(ip.Iwy1*20/p.r1):.2f}   gyro2={(ip.Iwy2*20/p.r2):.2f}")

    print("\n=== Modes at selected speeds ===")
    for u in [5.0, 10.0, 20.0, 30.0, 50.0]:
        modes = stability_analysis(u, p, ip)
        cap = next((m for m in modes if m.name=='Capsize'), None)
        wea = next((m for m in modes if m.name=='Weave'),   None)
        wob = next((m for m in modes if m.name=='Wobble'),  None)
        cap_s = f"{cap.growth:+.3f}" if cap else "  n/a "
        wea_s = (f"{wea.growth:+.3f} {wea.frequency:.1f}Hz z={wea.damping:+.3f}"
                 if wea else "  n/a")
        wob_s = (f"{wob.growth:+.3f} {wob.frequency:.1f}Hz z={wob.damping:+.3f}"
                 if wob else "  n/a")
        print(f"  u={u:4.0f} m/s  cap={cap_s}  weave={wea_s}  wobble={wob_s}")

    print("\n=== Stability transitions ===")
    u_arr = np.linspace(1, 70, 600)
    sw    = speed_sweep(u_arr, p, ip)
    for key, name in [('capsize_re','Capsize'),('weave_re','Weave'),('wobble_re','Wobble')]:
        arr = sw[key]
        mask = ~np.isnan(arr)
        if mask.sum() < 2: continue
        sc = np.where(np.diff(np.sign(arr[mask])))[0]
        uc_arr = u_arr[mask]
        ac_arr = arr[mask]
        for idx in sc:
            uc = np.interp(0, [ac_arr[idx], ac_arr[idx+1]],
                              [uc_arr[idx], uc_arr[idx+1]])
            d  = "S->U" if ac_arr[idx] < ac_arr[idx+1] else "U->S"
            print(f"  {name:8s}: {d} at {uc:.1f} m/s ({uc*3.6:.0f} km/h)")
