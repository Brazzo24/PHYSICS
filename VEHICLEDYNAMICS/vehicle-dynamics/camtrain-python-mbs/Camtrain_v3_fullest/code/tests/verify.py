"""Verification of the pycamtrain skeleton against closed-form solutions.

Each case has an analytic answer. Nothing about the camtrain physics is
trusted until these pass, following Rule 5 of Section 16.3: refactor in
validated increments.

Run:  python3 tests/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pycamtrain import (CompliantSpurGearMesh, Housing, MBSystem, PendulumBody,
                        Shaft, TorqueSource, TorsionalSpringDamper, simulate)

RESULTS = []


def check(name, computed, expected, tol_rel, unit=""):
    """Relative-error check against a closed-form value."""
    err = abs(computed - expected) / max(abs(expected), 1e-30)
    ok = err <= tol_rel
    RESULTS.append((name, computed, expected, err, tol_rel, ok, unit))
    return ok


def check_small(name, value, threshold, unit=""):
    """Absolute check that a quantity which should vanish actually does."""
    ok = abs(value) <= threshold
    RESULTS.append((name, value, threshold, abs(value) / threshold, 1.0, ok, unit))
    return ok


def zero_crossing_frequency(t, x):
    """Oscillation frequency from linearly interpolated upward zero crossings.

    Far more accurate than an FFT bin for a lightly damped single mode, and it
    does not need a long record to resolve the frequency.
    """
    x = np.asarray(x, float) - np.mean(x)
    up = np.where((x[:-1] < 0.0) & (x[1:] >= 0.0))[0]
    if len(up) < 2:
        raise RuntimeError("fewer than two zero crossings -- signal not oscillatory")
    frac = -x[up] / (x[up + 1] - x[up])
    tc = t[up] + frac * (t[up + 1] - t[up])
    return (len(tc) - 1) / (tc[-1] - tc[0])


# ---------------------------------------------------------------------------
# Case 1 -- two inertias coupled by a torsional spring (free-free)
#          analytic:  f_n = 1/(2*pi) * sqrt(c*(J1+J2)/(J1*J2))
# ---------------------------------------------------------------------------
def case_two_inertia():
    J1, J2, c = 2.0e-3, 5.0e-3, 4.0e3

    sys_ = MBSystem("two_inertia")
    h = Housing(sys_)
    s1 = Shaft(sys_, "shaft1", J=J1, housing=h, phi0=0.01)
    s2 = Shaft(sys_, "shaft2", J=J2, housing=h, phi0=0.0)
    TorsionalSpringDamper(sys_, "coupling", s1.flange, s2.flange, c=c, d=0.0)

    sys_.assemble()
    res = simulate(sys_, t_end=0.5, n_points=20001, method="DOP853",
                   rtol=1e-10, atol=1e-12)

    f_num = zero_crossing_frequency(res.t, res["shaft1.phi"])
    f_ana = np.sqrt(c * (J1 + J2) / (J1 * J2)) / (2 * np.pi)
    check("two-inertia eigenfrequency", f_num, f_ana, 1e-5, "Hz")

    # angular momentum must be conserved exactly (free-free, no damping)
    L = J1 * res["shaft1.w"] + J2 * res["shaft2.w"]
    check_small("angular momentum drift", float(np.max(np.abs(L - L[0]))),
                1e-14, "N m s")

    # energy conservation
    E = (0.5 * J1 * res["shaft1.w"] ** 2 + 0.5 * J2 * res["shaft2.w"] ** 2
         + 0.5 * c * (res["shaft1.phi"] - res["shaft2.phi"]) ** 2)
    check_small("energy drift (relative)",
                float(np.max(np.abs(E - E[0])) / E[0]), 1e-7, "-")
    return res


# ---------------------------------------------------------------------------
# Case 2 -- compliant spur gear pair driven by a constant torque
#          analytic (stiff mesh):  alpha_a = T / (J_a + J_b*i^2),  i = r_a/r_b
#          steady ratio: w_b/w_a = -i
# ---------------------------------------------------------------------------
def case_gear_pair():
    Ja, Jb = 1.0e-3, 3.0e-3
    ra, rb = 0.024, 0.040          # crank -> step stage, z = 24/40
    i = ra / rb                    # 0.6, matching Section 5.2
    c_mesh, d_mesh = 1.0e8, 2.0e2  # stiff mesh so the rigid limit applies
    T_drive = 5.0

    sys_ = MBSystem("gear_pair")
    h = Housing(sys_)
    a = Shaft(sys_, "gearA", J=Ja, housing=h)
    b = Shaft(sys_, "gearB", J=Jb, housing=h)
    CompliantSpurGearMesh(sys_, "mesh", a.flange, b.flange,
                          r_a=ra, r_b=rb, c=c_mesh, d=d_mesh)
    TorqueSource(sys_, "drive", a.flange, func=lambda t: T_drive)

    sys_.assemble()
    res = simulate(sys_, t_end=0.02, n_points=4001, method="Radau",
                   rtol=1e-10, atol=1e-12)

    # fit the acceleration over the second half, after the mesh transient
    m = res.t > 0.01
    alpha_num = np.polyfit(res.t[m], res["gearA.w"][m], 1)[0]
    alpha_ana = T_drive / (Ja + Jb * i ** 2)
    check("gear-pair rigid-limit acceleration", alpha_num, alpha_ana, 1e-3, "rad/s^2")

    ratio_num = float(np.mean(res["gearB.w"][m] / res["gearA.w"][m]))
    check("gear-pair signed speed ratio", ratio_num, -i, 1e-4, "-")

    # Mean compatibility residual: mesh ringing averages out, a genuine ratio
    # error would not. Section 5.2 / Appendix A.
    resid = float(np.mean(res["mesh.residual"][m]) / np.max(np.abs(res["gearA.w"][m])))
    check_small("mesh compatibility residual (mean, normalised)", resid, 1e-4, "-")

    # Transmitted mesh force against the rigid-limit value F = J_b*alpha_b/r_b
    F_num = float(np.mean(res["mesh.force"][m]))
    # Sign: a positive drive torque on A opens the mesh (delta > 0), so the
    # mesh force is positive and both reaction torques are negative.
    F_ana = Jb * (i * alpha_ana) / rb
    check("mesh force (rigid limit)", F_num, F_ana, 5e-3, "N")

    # Static mesh deflection must match F/c
    delta_num = float(np.mean(res["mesh.deflection"][m]))
    check("mesh deflection (F/c)", delta_num, F_ana / c_mesh, 5e-3, "m")
    return res


# ---------------------------------------------------------------------------
# Case 3 -- spatial pendulum, small-angle period
#          analytic: T = 2*pi*sqrt((J_com + m*l^2)/(m*g*l))
# ---------------------------------------------------------------------------
def case_pendulum():
    m, l, J_com, g = 1.7, 0.23, 4.0e-3, 9.81

    sys_ = MBSystem("pendulum")
    Housing(sys_)
    PendulumBody(sys_, "pend", mass=m, l=l, J_com=J_com, g=g, phi0=0.01)

    sys_.assemble()
    res = simulate(sys_, t_end=4.0, n_points=40001, method="DOP853",
                   rtol=1e-11, atol=1e-13)

    f_num = zero_crossing_frequency(res.t, res["pend.phi"])
    f_ana = np.sqrt(m * g * l / (J_com + m * l ** 2)) / (2 * np.pi)
    check("pendulum small-angle frequency", f_num, f_ana, 1e-4, "Hz")
    return res


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("pycamtrain skeleton verification")
    print("=" * 78)

    runs = {"two_inertia": case_two_inertia(),
            "gear_pair": case_gear_pair(),
            "pendulum": case_pendulum()}

    print(f"\n{'check':<42}{'computed':>13}{'expected':>13}{'rel err':>10}  ")
    print("-" * 78)
    for name, comp, exp, err, tol, ok, unit in RESULTS:
        flag = "PASS" if ok else "FAIL"
        print(f"{name:<42}{comp:>13.6g}{exp:>13.6g}{err:>10.2e}  {flag}")

    print("\n" + "=" * 78)
    print("run metadata (wall time)")
    print("=" * 78)
    for k, r in runs.items():
        print(f"\n[{k}]")
        print(r.meta)

    n_fail = sum(1 for r in RESULTS if not r[5])
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
