"""
utils/plotting.py
=================
Shared matplotlib helpers for the bicycle dynamics project.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Output directory ─────────────────────────────────────────────────────────
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(FIGURE_DIR, name if name.endswith(".png") else name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved: {path}")
    return path


def style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def make_fig(nrows=1, ncols=1, title="", **kwargs):
    style()
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig, axes


def plot_poles_on_complex_plane(ax, poles, label="", color="C0", marker="o", size=60):
    """Scatter poles on the complex plane."""
    poles = np.atleast_1d(poles)
    stable = poles[poles.real <= 0]
    unstable = poles[poles.real > 0]
    if len(stable):
        ax.scatter(stable.real, stable.imag, c=color, marker=marker, s=size,
                   zorder=5, label=label + " (stable)" if label else "stable")
    if len(unstable):
        ax.scatter(unstable.real, unstable.imag, c="red", marker=marker, s=size,
                   zorder=5, label=label + " (unstable)" if label else "unstable")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Real part  [rad/s]")
    ax.set_ylabel("Imag part  [rad/s]")


def annotate_velocity_range(ax, v_stable, v_unstable, ypos=0.95):
    """Add a horizontal bar showing the self-stable speed range."""
    ax.axvspan(v_stable, v_unstable, alpha=0.12, color="green",
               label=f"Self-stable: {v_stable:.2f}–{v_unstable:.2f} m/s")
