"""Shared plot style and a side-view sketch of the model (cf. Fig. 4.2)."""
from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# categorical colours in fixed order (colour-blind safe), ink and grid
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.prop_cycle": mpl.cycler(color=C),
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110,
    })


def side_view(kin, k, ax=None, labels=True):
    """Draw the bike in the plane of the frame (d1, d2) from a config dict."""
    ax = ax or plt.subplots(figsize=(9, 4.2))[1]
    p = kin.p
    O = np.array([k["q"][6], 0.0, k["q"][7]])
    RD = k["R_D"]
    P2 = lambda n: (RD.T @ (k["r_" + n] - O))[:2]
    th = np.linspace(0, 2 * np.pi, 200)
    for axle, R in (("D", p.Rr), ("G", p.Rf)):
        c = P2(axle)
        ax.plot(c[0] + R * np.cos(th), c[1] + R * np.sin(th), color=MUTED, lw=1.2)
    seg = lambda a, b, **kw: ax.plot(*np.column_stack([P2(a), P2(b)]), **kw)
    seg("D", "P", color=C[0], lw=3)                      # swing-arm
    seg("P", "H", color=C[1], lw=3)                      # main frame
    seg("H", "Q", color=C[2], lw=3)                      # steering assembly
    seg("Q", "G", color=C[3], lw=3)                      # fork lower
    for n in "ABCDFG":
        ax.plot(*P2(n), "o", color=INK, ms=5)
    for n in "PHQR":
        ax.plot(*P2(n), "s", color=MUTED, ms=4)
    if labels:
        for n in "ABCDFGPHQR":
            ax.annotate(n, P2(n), xytext=(5, 5), textcoords="offset points")
        for n in "ST":
            ax.annotate(n, P2(n), xytext=(4, -13), textcoords="offset points")
    for n in "ST":
        ax.plot(*P2(n), "^", color=INK, ms=6)
    ax.axhline(0, color=INK, lw=1)
    ax.set_aspect("equal")
    ax.set_xlabel("$\\hat d_1$ [m]"); ax.set_ylabel("$\\hat d_2$ [m]")
    ax.grid(False)
    return ax
