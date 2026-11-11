"""Shared matplotlib style for the notebook series (import once per notebook)."""
import matplotlib as mpl
from cycler import cycler

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"


def use():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 120, "figure.facecolor": "white",
        "axes.prop_cycle": cycler(color=COLORS),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.8, "lines.markersize": 4, "legend.frameon": False,
        "legend.fontsize": 9, "font.size": 10, "figure.constrained_layout.use": True,
    })
    return COLORS


def mult_axis(ax, axis="x", ticks=(0.2, 0.3, 0.5, 0.7, 1, 1.4, 2, 3, 5)):
    """Log axis with plain multiplier tick labels (0.5, 1, 2 ...) inside the current limits."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    ax_ = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    t = [v for v in ticks if lo * 0.999 <= v <= hi * 1.001]
    if len(t) > 5:
        t = [v for v in t if v in (0.2, 0.5, 1, 2, 5)]
    ax_.set_major_locator(FixedLocator(t))
    ax_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_.set_minor_locator(NullLocator())
    ax_.set_minor_formatter(NullFormatter())
