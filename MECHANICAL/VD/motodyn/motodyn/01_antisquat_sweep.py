"""
Example 1 — Anti-squat parameter study.

Sweeps the swingarm pivot height and measures how anti-squat changes.
This is the prototypical "parameter study" use case.
"""

import matplotlib.pyplot as plt
import numpy as np

from motodyn.presets import generic_sportbike
from motodyn.models.suspension_kinematics import anti_squat_ratio
from motodyn.analyses.parameter_sweep import sweep


def main():
    bike = generic_sportbike()

    # Sweep swingarm pivot height across a realistic tuning range (~4 cm)
    baseline_pivot = bike.suspension.swingarm_pivot
    heights = np.linspace(0.30, 0.34, 25)
    values = [(baseline_pivot[0], z) for z in heights]

    results = sweep(
        bike,
        param_path="suspension.swingarm_pivot",
        values=values,
        metric=anti_squat_ratio,
    )

    z_vals = [v[1] for v, _ in results]
    as_pct = [m * 100 for _, m in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z_vals, as_pct, "o-", linewidth=2, markersize=5)
    ax.axhline(100, color="grey", linestyle="--", alpha=0.6, label="100 % (neutral)")
    ax.set_xlabel("Swingarm pivot height above ground [m]")
    ax.set_ylabel("Anti-squat [%]")
    ax.set_title(f"{bike.name}: anti-squat vs. swingarm pivot height")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("antisquat_sweep.png", dpi=120)
    print(f"Baseline anti-squat: {anti_squat_ratio(bike) * 100:.1f} %")
    print(f"Range over sweep: {min(as_pct):.1f} % → {max(as_pct):.1f} %")
    print("Saved antisquat_sweep.png")


if __name__ == "__main__":
    main()
