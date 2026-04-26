"""
Example 3 — 2D grid sweep.

Anti-squat as a function of swingarm pivot height AND countershaft
sprocket height. Produces a heatmap.
"""

import matplotlib.pyplot as plt
import numpy as np

from motodyn.presets import generic_sportbike
from motodyn.models.suspension_kinematics import anti_squat_ratio


def main():
    bike = generic_sportbike()

    pivot_z = np.linspace(0.24, 0.36, 15)
    sprkt_z = np.linspace(0.22, 0.34, 15)
    pivot_x = bike.suspension.swingarm_pivot[0]
    sprkt_x = bike.suspension.sprocket_center[0]

    AS = np.zeros((len(sprkt_z), len(pivot_z)))
    for i, sz in enumerate(sprkt_z):
        for j, pz in enumerate(pivot_z):
            modified = bike.with_changes(
                **{
                    "suspension.swingarm_pivot": (pivot_x, pz),
                    "suspension.sprocket_center": (sprkt_x, sz),
                }
            )
            AS[i, j] = anti_squat_ratio(modified) * 100

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(pivot_z, sprkt_z, AS, shading="auto", cmap="RdYlGn")
    cs = ax.contour(pivot_z, sprkt_z, AS, levels=[50, 75, 100, 125, 150],
                    colors="black", linewidths=0.8)
    ax.clabel(cs, fmt="%d %%")
    ax.set_xlabel("Swingarm pivot height [m]")
    ax.set_ylabel("Countershaft sprocket height [m]")
    ax.set_title("Anti-squat [%]")
    fig.colorbar(im, ax=ax, label="Anti-squat [%]")
    fig.tight_layout()
    fig.savefig("antisquat_grid.png", dpi=120)
    print("Saved antisquat_grid.png")


if __name__ == "__main__":
    main()
