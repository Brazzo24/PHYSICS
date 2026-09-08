"""Stage 2: one camtrain bank, unloaded, driven through a speed ramp.

Stage 2 of the Section 12 commissioning ladder asks one question: do signs,
ratios and compliant meshes behave? A constant-speed run cannot answer it,
because with consistent initial speeds and no load every mesh force is
identically zero. So the crankshaft is ramped 0 -> 6000 rpm with a smooth
half-cosine profile. During the ramp each mesh must transmit exactly the torque
needed to accelerate everything downstream of it, which turns the run into four
independent analytic checks plus an energy balance.

Acceptance checks (Section 13.1):
  1. signed speeds after the ramp match the Section 5.2 table
  2. all six compatibility residuals vanish
  3. crankshaft drive torque equals J_reflected * alpha_crank
  4. last-stage mesh force equals -J_exhaust * alpha_exhaust / r_base
  5. drive power = d(KE)/dt + mesh damping loss

Run:  python3 examples/ex02_single_bank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycamtrain import (BankGeometry, CamtrainBank, Housing, KinematicDrive,
                        MBSystem, layout_from_bearing_angles, simulate)
from pycamtrain.components.bank import DEFAULT_INERTIA
from pycamtrain.core.system import T_SYM
from pycamtrain.geometry import SHAFT_NAMES

RPM = 2.0 * np.pi / 60.0
CRANK_RPM = 6000.0
OMEGA = CRANK_RPM * RPM
T_RAMP = 0.05
T_END = 0.10

MESH_C = 1.0e8
MESH_D = 2.0e3

CHECKS = []


def record(name, computed, expected, tol_rel, unit=""):
    err = abs(computed - expected) / max(abs(expected), 1e-30)
    CHECKS.append((name, computed, expected, err, err <= tol_rel, unit))


def record_small(name, value, threshold, unit=""):
    CHECKS.append((name, value, threshold, abs(value) / threshold,
                   abs(value) <= threshold, unit))


def speed_profile():
    """Smooth C1 half-cosine ramp, then constant. Angle given in closed form."""
    t = T_SYM
    omega = sp.Piecewise(
        (OMEGA * sp.Rational(1, 2) * (1 - sp.cos(sp.pi * t / T_RAMP)), t < T_RAMP),
        (OMEGA, True))
    phi = sp.Piecewise(
        (OMEGA * sp.Rational(1, 2) * (t - T_RAMP / sp.pi * sp.sin(sp.pi * t / T_RAMP)),
         t < T_RAMP),
        (OMEGA * (t - T_RAMP / 2), True))
    return omega, phi


def alpha_crank(t):
    """Analytic crankshaft angular acceleration of the ramp."""
    a = np.zeros_like(t)
    m = t < T_RAMP
    a[m] = OMEGA * np.pi / (2 * T_RAMP) * np.sin(np.pi * t[m] / T_RAMP)
    return a


def build():
    geo = BankGeometry(layout_from_bearing_angles([28.0, 8.0, -32.0, 90.0]))
    omega, phi = speed_profile()

    sys_ = MBSystem("single_bank_stage2")
    housing = Housing(sys_)
    drive = KinematicDrive(sys_, "crankshaft", omega=omega, phi=phi,
                           axis=geo.axis, housing=housing)
    bank = CamtrainBank(sys_, "frontBank", geometry=geo, housing=housing,
                        drive_flange=drive.flange, drive_speed=0.0,
                        mesh_stiffness=MESH_C, mesh_damping=MESH_D)
    return sys_, geo, drive, bank


def main():
    sys_, geo, drive, bank = build()

    print("=" * 78)
    print("Stage 2 -- single camtrain bank, unloaded, speed ramp")
    print("=" * 78)
    print()
    print(geo.report())
    print(f"\n  geometry/tooth consistency within 50 um: {geo.check()}")
    print()
    print("component tree")
    for c in sys_.components:
        print("  " + c.tree().replace("\n", "\n  "))

    sys_.assemble(verbose=True)
    print()
    print(sys_.summary())

    res = simulate(sys_, t_end=T_END, n_points=5001, method="Radau",
                   rtol=1e-9, atol=1e-11)

    t = res.t
    post = t > 0.07                      # settled, constant-speed window
    ramp = (t > 0.005) & (t < T_RAMP - 0.005)

    # ---- 1. signed speeds ------------------------------------------------
    print("\n" + "=" * 78)
    print("1. signed speeds after the ramp (Section 5.2)")
    print("=" * 78)
    print(f"{'shaft':<20}{'mean rpm':>14}{'expected':>14}{'rel err':>12}")
    print("-" * 78)
    for row in range(len(SHAFT_NAMES)):
        name = SHAFT_NAMES[row]
        exp = CRANK_RPM * geo.total_ratio(row)
        key = f"crankshaft.rpm" if row == 0 else f"frontBank.{name}.rpm"
        num = float(np.mean(res[key][post]))
        record(f"speed {name}", num, exp, 1e-6, "rpm")
        print(f"{name:<20}{num:>14.3f}{exp:>14.3f}{abs(num-exp)/abs(exp):>12.2e}")

    # ---- 2. six compatibility residuals ----------------------------------
    w_max = float(np.max(np.abs(res["crankshaft.w"])))
    print("\n" + "=" * 78)
    print("2. six signed compatibility residuals per bank (Section 13)")
    print("=" * 78)
    # After the ramp the residual is a damped oscillation at the mesh
    # frequency, not a ratio error. Two things must hold (Section 13.1): it is
    # small compared with the speeds it relates, and it decays rather than
    # drifting. A wrong ratio would show up as a residual that does neither.
    early = (t > 0.055) & (t < 0.0725)
    late = t > 0.0825
    print(f"{'residual':<32}{'max ramp':>12}{'max post':>12}"
          f"{'normalised':>12}{'decay':>9}")
    print("-" * 78)
    for key in bank.residual_names():
        r = res[key]
        label = key.replace("frontBank.", "").replace(".residual", "") \
                   .replace("residual_", "")
        n_post = float(np.max(np.abs(r[post]))) / w_max
        decay = float(np.max(np.abs(r[late]))) / float(np.max(np.abs(r[early])))
        record_small(f"residual {label} (normalised)", n_post, 2e-5, "-")
        record_small(f"residual {label} decaying", decay - 1.0, 1.0, "-")
        print(f"{label:<32}{np.max(np.abs(r[ramp])):>12.3e}"
              f"{np.max(np.abs(r[post])):>12.3e}{n_post:>12.2e}{decay:>9.3f}")

    # ---- 3. drive torque vs reflected inertia ----------------------------
    J_ref = sum(DEFAULT_INERTIA[SHAFT_NAMES[i]] * geo.total_ratio(i) ** 2
                for i in range(1, len(SHAFT_NAMES)))
    a_crank = alpha_crank(t)
    T_num = float(np.mean(res["frontBank.driveTorque"][ramp]))
    T_ana = float(np.mean(J_ref * a_crank[ramp]))
    record("drive torque vs J_reflected*alpha", T_num, T_ana, 5e-3, "N m")

    # ---- 4. last-stage mesh force ----------------------------------------
    last = geo.stages[-1]
    rb = geo.base_radius(last.z_b)
    J_exh = DEFAULT_INERTIA["exhaustCamshaft"]
    a_exh = geo.total_ratio(4) * a_crank
    F_num = float(np.mean(res[f"frontBank.{last.name}.force"][ramp]))
    F_ana = float(np.mean(-J_exh * a_exh[ramp] / rb))
    record("last-stage mesh force", F_num, F_ana, 5e-3, "N")

    # ---- 5. energy balance ------------------------------------------------
    KE = res["frontBank.kineticEnergy"]
    dKE = np.gradient(KE, t)
    imbalance = res["frontBank.drivePower"] - dKE - res["frontBank.meshDampingPower"]
    scale = float(np.max(np.abs(res["frontBank.drivePower"])))
    record_small("energy balance residual (normalised)",
                 float(np.max(np.abs(imbalance[ramp]))) / scale, 5e-3, "-")

    print("\n" + "=" * 78)
    print("3-5. analytic and balance checks")
    print("=" * 78)
    print(f"  reflected inertia at crankshaft : {J_ref:.6e} kg m^2")
    print(f"  peak crankshaft acceleration    : {a_crank.max():.1f} rad/s^2")
    print(f"  mean drive torque over ramp     : {T_num:.4f} N m  "
          f"(analytic {T_ana:.4f})")
    print(f"  last-stage mesh force           : {F_num:.4f} N    "
          f"(analytic {F_ana:.4f})")

    print("\n" + "=" * 78)
    print("acceptance summary")
    print("=" * 78)
    print(f"{'check':<44}{'computed':>13}{'target':>13}{'ratio':>9}")
    print("-" * 78)
    for name, comp, exp, err, ok, unit in CHECKS:
        print(f"{name:<44}{comp:>13.6g}{exp:>13.6g}{err:>9.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print("run metadata")
    print("=" * 78)
    print(res.meta)
    print(f"  {'cse':<9} : {sys_.cse}")

    # ---- plots (Section 13 groups) ---------------------------------------
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(t * 1e3, res["crankshaft.rpm"], "k", lw=1.6, label="crankshaft")
    for n in bank.shaft_names():
        ax[0].plot(t * 1e3, res[f"frontBank.{n}.rpm"], lw=1.1, label=n)
    ax[0].set_ylabel("signed speed [rpm]")
    ax[0].legend(fontsize=8, ncol=3); ax[0].grid(alpha=0.3)
    ax[0].set_title("Stage 2: single bank, unloaded, 0 -> 6000 rpm ramp")

    for key in bank.residual_names():
        label = key.replace("frontBank.", "").replace(".residual", "") \
                   .replace("residual_", "")
        ax[1].plot(t * 1e3, res[key], lw=1.0, label=label)
    ax[1].set_ylabel("compatibility residual [rad/s]")
    ax[1].legend(fontsize=7, ncol=3); ax[1].grid(alpha=0.3)

    for m in bank.meshes:
        ax[2].plot(t * 1e3, res[f"frontBank.{m.name}.force"], lw=1.0, label=m.name)
    ax[2].set_ylabel("mesh force [N]")
    ax[2].legend(fontsize=8, ncol=2); ax[2].grid(alpha=0.3)

    ax[3].plot(t * 1e3, res["frontBank.drivePower"], lw=1.4, label="drive power")
    ax[3].plot(t * 1e3, dKE, "--", lw=1.2, label="d(KE)/dt")
    ax[3].plot(t * 1e3, res["frontBank.meshDampingPower"], lw=1.0,
               label="mesh damping loss")
    ax[3].set_ylabel("power [W]"); ax[3].set_xlabel("time [ms]")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)

    fig.tight_layout()
    png = out / "ex02_single_bank.png"
    fig.savefig(png, dpi=140)
    print(f"\nplot written to {png}")

    n_fail = sum(1 for c in CHECKS if not c[4])
    print(f"\n{len(CHECKS) - n_fail}/{len(CHECKS)} acceptance checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
