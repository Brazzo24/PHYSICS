"""
weave.plots
===========
Matplotlib figures used by the examples.  Styling follows a small fixed
categorical palette (assigned in a fixed order, never cycled), thin lines,
recessive grid, a legend on every multi-series chart and a secondary encoding
(dash pattern) so identity never relies on colour alone.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]          # blue, orange, aqua, magenta
DASHES = ["-", (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 1.5, 1.5, 1.5))]
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#2a78d6", "#dcdad2", "#eb6834"])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 10, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 110,
    })


def _plot(ax, x, y, i, label):
    ax.plot(x, y, color=SERIES[i % 4], ls=DASHES[i % 4], label=label)


def modes_vs_speed(sweeps: dict, path: str, title: str = "", umin=None):
    """sweeps: label -> speed_sweep() result.  Five panels: capsize growth,
    weave / wobble damping ratio and frequency."""
    style()
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.6))
    panels = [("capsize_re", "Capsize growth rate [1/s]", axs[0, 0]),
              ("weave_zeta", "Weave damping ratio [-]", axs[0, 1]),
              ("wobble_zeta", "Wobble damping ratio [-]", axs[0, 2]),
              ("weave_freq", "Weave frequency [Hz]", axs[1, 0]),
              ("wobble_freq", "Wobble frequency [Hz]", axs[1, 1])]
    for key, ttl, ax in panels:
        for i, (lab, sw) in enumerate(sweeps.items()):
            _plot(ax, sw["u"] * 3.6, sw[key], i, lab)
        ax.set_title(ttl)
        ax.set_xlabel("speed [km/h]")
        if key == "capsize_re":
            ax.axhline(0, color=INK2, lw=0.8)
    axs[1, 2].axis("off")
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[1, 2].legend(h, l, loc="center left", fontsize=10)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def delta_vs_speed(base: dict, variants: dict, path: str, title: str = ""):
    """Change relative to ``base`` (percent of the baseline value) for the
    weave / wobble damping ratio, weave frequency and capsize growth rate."""
    style()
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    keys = [("capsize_re", "Capsize growth rate"), ("weave_zeta", "Weave damping ratio"),
            ("weave_freq", "Weave frequency"), ("wobble_zeta", "Wobble damping ratio")]
    for (key, ttl), ax in zip(keys, axs):
        for i, (lab, sw) in enumerate(variants.items()):
            d = 100.0 * (sw[key] - base[key]) / np.abs(base[key])
            _plot(ax, sw["u"] * 3.6, d, i, lab)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(ttl + "  [Δ %]")
        ax.set_xlabel("speed [km/h]")
    axs[0].legend(fontsize=8.5, loc="best")
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def sensitivity_maps(maps: dict, speeds, scales, path: str, title="", unit="Δ weave damping ratio [%]"):
    """maps: label -> 2-D array (len(scales) x len(speeds)) of percent change."""
    style()
    n = len(maps)
    vmax = max(np.nanmax(np.abs(m)) for m in maps.values())
    fig, axs = plt.subplots(1, n, figsize=(5.6 * n + 1.2, 4.4), squeeze=False)
    for ax, (lab, M) in zip(axs[0], maps.items()):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                       extent=[speeds[0] * 3.6, speeds[-1] * 3.6, scales[0], scales[-1]])
        ax.set_title(lab)
        ax.set_xlabel("speed [km/h]")
        ax.grid(False)
    axs[0, 0].set_ylabel("rotating-inertia scale factor [-]")
    cb = fig.colorbar(im, ax=axs[0].tolist(), shrink=0.9, pad=0.02)
    cb.set_label(unit)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def gear_bars(labels, values, ratios, path, ylabel, title=""):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=SERIES[0], width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("gear (crank / wheel speed ratio below)")
    for x, v, r in zip(xs, values, ratios):
        ax.text(x, v, f"{r:.1f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5, color=INK2)
    ax.grid(axis="x", visible=False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def time_response(runs: dict, path: str, title=""):
    """runs: label -> simulate() result.  Roll, steer, yaw-rate panels."""
    style()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for i, (lab, r) in enumerate(runs.items()):
        _plot(axs[0], r["t"], np.degrees(r["phi"]), i, lab)
        _plot(axs[1], r["t"], np.degrees(r["delta"]), i, lab)
        _plot(axs[2], r["t"], np.degrees(r["r"]), i, lab)
    for ax, t in zip(axs, ("Roll angle φ [deg]", "Steer angle δ [deg]", "Yaw rate r [deg/s]")):
        ax.set_title(t)
        ax.set_xlabel("time [s]")
        ax.axhline(0, color=INK2, lw=0.8)
    axs[0].legend(fontsize=8.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pole_map(sweeps: dict, path: str, title=""):
    """Weave and wobble pole trajectories in the complex plane."""
    style()
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for i, (lab, sw) in enumerate(sweeps.items()):
        for key, mk in (("weave", "o"), ("wobble", "s")):
            re, f = sw[f"{key}_re"], sw[f"{key}_freq"] * 2 * np.pi
            ax.plot(re, f, color=SERIES[i % 4], ls=DASHES[i % 4], marker=mk, ms=3.5,
                    label=f"{lab} – {key}" if key == "weave" else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("real part [1/s]")
    ax.set_ylabel("imaginary part [rad/s]")
    ax.set_title(title or "Weave (●) and wobble (■) poles, 20–250 km/h")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
