"""
Example 2 — braking behaviour.

Sweeps longitudinal deceleration and plots:
  * wheel loads
  * fork compression
  * anti-dive-moderated pitch
"""

import matplotlib.pyplot as plt
import numpy as np

from motodyn.presets import generic_sportbike
from motodyn.state import OperatingPoint
from motodyn.models.chassis_rigid_body import chassis_equilibrium
from motodyn.models.load_transfer import stoppie_threshold_deceleration


def main():
    bike = generic_sportbike()
    stoppie = stoppie_threshold_deceleration(bike)
    print(f"Stoppie threshold deceleration: {stoppie:.2f} m/s² ({stoppie / 9.81:.2f} g)")

    decels = np.linspace(0.0, 0.95 * stoppie, 30)       # stop just short of wheel lift
    Nf, Nr, fork, pitch = [], [], [], []

    for d in decels:
        op = OperatingPoint(speed=25.0, a_x=-d)         # braking
        eq = chassis_equilibrium(bike, op)
        Nf.append(eq.N_front)
        Nr.append(eq.N_rear)
        fork.append(eq.front_compression * 1000)        # mm
        pitch.append(np.degrees(eq.pitch_angle))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    decels_g = decels / 9.81

    axes[0].plot(decels_g, Nf, label="Front", linewidth=2)
    axes[0].plot(decels_g, Nr, label="Rear", linewidth=2)
    axes[0].set_xlabel("Deceleration [g]"); axes[0].set_ylabel("Wheel load [N]")
    axes[0].set_title("Wheel loads"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(decels_g, fork, linewidth=2, color="C2")
    axes[1].set_xlabel("Deceleration [g]"); axes[1].set_ylabel("Fork compression [mm]")
    axes[1].set_title("Fork dive"); axes[1].grid(alpha=0.3)

    axes[2].plot(decels_g, pitch, linewidth=2, color="C3")
    axes[2].set_xlabel("Deceleration [g]"); axes[2].set_ylabel("Pitch angle [deg]")
    axes[2].set_title("Chassis pitch (− = nose down)"); axes[2].grid(alpha=0.3)

    fig.suptitle(f"{bike.name}: braking study  (anti-dive baseline)")
    fig.tight_layout()
    fig.savefig("braking_study.png", dpi=120)
    print("Saved braking_study.png")


if __name__ == "__main__":
    main()
