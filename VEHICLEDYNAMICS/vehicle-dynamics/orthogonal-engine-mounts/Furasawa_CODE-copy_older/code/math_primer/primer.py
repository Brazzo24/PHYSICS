"""Shared plotting style and small drawing helpers for the math primer notebooks.

Nothing here is physics -- it is only so the seven notebooks look like one
document and so the toy spring-mass pictures do not have to be re-drawn by hand
in every notebook.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# One categorical palette, used in fixed order everywhere in the primer.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.labelsize": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "font.size": 9,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "axes.spines.top": False, "axes.spines.right": False,
    })
    np.set_printoptions(precision=4, suppress=True, linewidth=110)


def tidy(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


# ---------------------------------------------------------------------------
#  Toy drawings
# ---------------------------------------------------------------------------

def spring(ax, x0, x1, y=0.0, coils=7, amp=0.09, **kw):
    """Draw a zig-zag spring between x0 and x1 at height y."""
    kw.setdefault("color", INK2)
    kw.setdefault("lw", 1.3)
    n = coils * 2
    lead = 0.16 * (x1 - x0)
    xs = [x0, x0 + lead]
    ys = [y, y]
    body = np.linspace(x0 + lead, x1 - lead, n + 1)
    for i, xv in enumerate(body):
        xs.append(xv)
        ys.append(y + (amp if i % 2 else -amp) if 0 < i < n else y)
    xs += [x1 - lead, x1]
    ys += [y, y]
    ax.plot(xs, ys, **kw)


def block(ax, xc, y=0.0, w=0.34, h=0.34, color=SERIES[0], label=None):
    """Draw a mass block centred at xc."""
    ax.add_patch(plt.Rectangle((xc - w / 2, y - h / 2), w, h,
                               facecolor=color, edgecolor="white", lw=1.5,
                               zorder=3))
    if label:
        ax.text(xc, y, label, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=4)


def wall(ax, x, y=0.0, h=0.7, side=1):
    """Draw a hatched ground/wall at x."""
    ax.plot([x, x], [y - h / 2, y + h / 2], color=INK, lw=2, zorder=2)
    for yy in np.linspace(y - h / 2, y + h / 2, 8):
        ax.plot([x, x - 0.09 * side], [yy, yy - 0.07], color=INK2, lw=1)


def chain_2dof(ax, u=(0.0, 0.0), scale=1.0, labels=("m1", "m2"),
               colors=(SERIES[0], SERIES[1]), y=0.0):
    """Draw the standard 2-mass chain: wall - k1 - m1 - k2 - m2.

    ``u`` are the two displacements; they are drawn scaled by ``scale``.
    """
    x1, x2 = 1.0 + scale * u[0], 2.2 + scale * u[1]
    wall(ax, 0.0, y=y)
    spring(ax, 0.0, x1 - 0.17, y=y)
    spring(ax, x1 + 0.17, x2 - 0.17, y=y)
    block(ax, x1, y=y, color=colors[0], label=labels[0])
    block(ax, x2, y=y, color=colors[1], label=labels[1])
    ax.set_xlim(-0.35, 3.1)
    ax.set_ylim(y - 0.55, y + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")


def strobe(ax, mode, n=7, amp=0.35, labels=("m1", "m2"), title=""):
    """Draw one mode shape as a strobe: several phases of the motion stacked."""
    phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for i, ph in enumerate(phases):
        alpha = 0.18 + 0.82 * (i == 0)
        u = np.real(mode) * np.cos(ph) * amp
        y = 0.0
        x1, x2 = 1.0 + u[0], 2.2 + u[1]
        ax.plot([x1], [y], "o", ms=11, color=SERIES[0], alpha=alpha, zorder=3)
        ax.plot([x2], [y], "o", ms=11, color=SERIES[1], alpha=alpha, zorder=3)
    wall(ax, 0.0)
    ax.set_xlim(-0.35, 3.1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", pad=6)
