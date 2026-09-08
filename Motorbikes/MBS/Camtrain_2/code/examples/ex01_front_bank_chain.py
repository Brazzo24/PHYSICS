"""Front-bank geartrain: the Section 5.2 ratio table, unloaded.

This is the first example that exercises the real camtrain topology instead of
a two-body toy: five shafts, four compliant meshes, consistent initial speeds.

It reproduces two acceptance checks from the model package guide:

  * Section 5.2  -- signed speeds at 6000 rpm crankshaft:
                    +6000, -3600, +1764.71, -3000, +3000 rpm
  * Section 11.1 -- consistent initial speeds derived from the crankshaft
                    reference and the gear ratios avoid mesh shock at t = 0

The crankshaft is represented here by a large flywheel with no external
torque, so the chain simply coasts. Prescribed-speed excitation, the PI
controller and the measured cam loads come in the next stage.

Run:  python3 examples/ex01_front_bank_chain.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycamtrain import (CompliantSpurGearMesh, Housing, MBSystem, Shaft,
                        TorsionalSpringDamper, simulate)

RPM = 2.0 * np.pi / 60.0
MODULE = 1.5e-3          # gear module [m]

# tooth numbers, driving -> driven, from Section 5.2
STAGES = [("crank_to_step",    24, 40),
          ("step_to_idler",    25, 51),
          ("idler_to_intake",  51, 30),
          ("intake_to_exhaust", 30, 30)]

SHAFTS = ["crankshaft", "stepShaft", "idlerCamShaft", "intakeCamshaft",
          "exhaustCamshaft"]

# polar inertias [kg m^2] -- placeholder magnitudes, replace with CAD values
INERTIA = {"crankshaft": 2.5e-2, "stepShaft": 3.0e-4, "idlerCamShaft": 4.5e-4,
           "intakeCamshaft": 6.0e-4, "exhaustCamshaft": 6.0e-4}

MESH_STIFFNESS = 1.0e8   # N/m along the line of action
MESH_DAMPING = 2.0e3     # N s/m
CRANK_RPM = 6000.0


def signed_speeds(crank_rpm: float):
    """Signed shaft speeds from the ratio chain. Every external mesh reverses."""
    speeds = [crank_rpm]
    for _, z_a, z_b in STAGES:
        speeds.append(-speeds[-1] * z_a / z_b)
    return speeds


def build():
    expected_rpm = signed_speeds(CRANK_RPM)

    sys_ = MBSystem("front_bank_chain")
    housing = Housing(sys_)

    shafts = {}
    for name, rpm0 in zip(SHAFTS, expected_rpm):
        shafts[name] = Shaft(sys_, name, J=INERTIA[name], housing=housing,
                             w0=rpm0 * RPM)

    meshes = []
    for (mesh_name, z_a, z_b), a, b in zip(STAGES, SHAFTS[:-1], SHAFTS[1:]):
        meshes.append(CompliantSpurGearMesh(
            sys_, mesh_name, shafts[a].flange, shafts[b].flange,
            r_a=MODULE * z_a / 2.0, r_b=MODULE * z_b / 2.0,
            c=MESH_STIFFNESS, d=MESH_DAMPING))

    return sys_, shafts, meshes, expected_rpm


def main():
    sys_, shafts, meshes, expected_rpm = build()
    sys_.assemble(verbose=True)
    print()
    print(sys_.summary())

    res = simulate(sys_, t_end=0.02, n_points=4001, method="Radau",
                   rtol=1e-9, atol=1e-11)

    print("\n" + "=" * 74)
    print("signed speeds (Section 5.2)")
    print("=" * 74)
    print(f"{'shaft':<20}{'mean rpm':>14}{'expected':>14}{'rel err':>12}")
    print("-" * 74)
    ok = True
    for name, exp in zip(SHAFTS, expected_rpm):
        num = float(np.mean(res[f"{name}.rpm"]))
        err = abs(num - exp) / abs(exp)
        ok &= err < 1e-4
        print(f"{name:<20}{num:>14.2f}{exp:>14.2f}{err:>12.2e}")

    print("\n" + "=" * 74)
    print("mesh state (Section 11.1: no shock from consistent initial speeds)")
    print("=" * 74)
    print(f"{'mesh':<20}{'max |force| N':>16}{'max |defl| um':>16}"
          f"{'mean residual':>16}")
    print("-" * 74)
    for m in meshes:
        F = np.abs(res[f"{m.name}.force"]).max()
        d = np.abs(res[f"{m.name}.deflection"]).max() * 1e6
        r = float(np.mean(res[f"{m.name}.residual"]))
        print(f"{m.name:<20}{F:>16.2f}{d:>16.3f}{r:>16.2e}")

    print("\n" + "=" * 74)
    print("run metadata")
    print("=" * 74)
    print(res.meta)

    # ---- plots ------------------------------------------------------
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name in SHAFTS:
        ax[0].plot(res.t * 1e3, res[f"{name}.rpm"], label=name, lw=1.2)
    ax[0].set_ylabel("signed speed [rpm]")
    ax[0].legend(fontsize=8, ncol=3)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Front-bank geartrain, unloaded, 6000 rpm crankshaft")

    for m in meshes:
        ax[1].plot(res.t * 1e3, res[f"{m.name}.force"], label=m.name, lw=1.0)
    ax[1].set_ylabel("mesh force [N]")
    ax[1].set_xlabel("time [ms]")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    png = out / "ex01_front_bank_chain.png"
    fig.savefig(png, dpi=140)
    print(f"\nplot written to {png}")

    print("\n" + ("ACCEPTANCE: signed speeds match the ratio table"
                  if ok else "ACCEPTANCE FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
