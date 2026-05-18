"""
analysis/stability.py
======================
Stability analysis for all three model levels.

Produces
--------
  - Fig. A1 : Level-1 open-loop poles & P-control stabilisation boundary
  - Fig. A2 : Level-2 front-fork gains k₁(V), k₂(V); critical velocity
  - Fig. A3 : Level-3 root locus (pole trajectories vs V)  — paper Fig. 8
  - Fig. A4 : Level-3 real parts of poles vs V             — paper Fig. 9
              (rear-steered bicycle shown with negative V)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import level1_inverted_pendulum as L1
from models import level2_front_fork         as L2
from models import level3_whipple            as L3
from utils.params import BIKE_WITH_RIDER, BIKE_WITHOUT_RIDER
from utils.plotting import make_fig, save, annotate_velocity_range


# ─────────────────────────────────────────────────────────────────────────────
# Fig. A1  —  Level-1: poles & min gain for P-control
# ─────────────────────────────────────────────────────────────────────────────

def plot_level1_stability(p=BIKE_WITH_RIDER, V_range=(1.0, 15.0)):
    fig, axes = make_fig(1, 2, figsize=(11, 4),
                         title="Level-1 Inverted Pendulum — Stability")
    V_vec = np.linspace(*V_range, 300)

    # Left: open-loop poles (constant; independent of V)
    p_stable, p_unstable = L1.open_loop_poles(p)
    ax = axes[0]
    ax.axhline(p_unstable, color="red",  lw=2, label=f"Unstable pole  p = +{p_unstable:.2f}")
    ax.axhline(p_stable,   color="blue", lw=2, label=f"Stable pole    p = {p_stable:.2f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Forward speed V [m/s]")
    ax.set_ylabel("Pole  [rad/s]")
    ax.set_title("Open-loop poles (Eq. 2)")
    ax.legend(fontsize=9)

    # Right: minimum k₂ for stability vs V
    ax = axes[1]
    k2_min = [L1.minimum_gain_for_stability(v, p) for v in V_vec]
    ax.plot(V_vec, k2_min, "C1", lw=2)
    ax.fill_between(V_vec, k2_min, alpha=0.15, color="C1", label="k₂ > b·g/V² required")
    ax.set_xlabel("Forward speed V [m/s]")
    ax.set_ylabel("Minimum gain k₂")
    ax.set_title("Min. proportional gain for stability (Eq. 6)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, "A1_level1_stability")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig. A2  —  Level-2: front-fork gains & critical velocity
# ─────────────────────────────────────────────────────────────────────────────

def plot_level2_front_fork_gains(p=BIKE_WITH_RIDER):
    V_sa = L2.self_alignment_velocity(p)
    V_vec_lo = np.linspace(0.1, V_sa * 0.97, 200)
    V_vec_hi = np.linspace(V_sa * 1.03, 20.0, 400)
    V_vec = np.concatenate([V_vec_lo, V_vec_hi])

    fig, axes = make_fig(1, 2, figsize=(11, 4),
                         title="Level-2 Front-Fork Gains (Eqs. 12–13)")

    ax = axes[0]
    ax.plot(V_vec_lo, [L2.k2(v, p) for v in V_vec_lo], "C3", lw=1.8)
    ax.plot(V_vec_hi, [L2.k2(v, p) for v in V_vec_hi], "C0", lw=2,
            label="k₂(V)  [stabilising for V > V_sa]")
    ax.axvline(V_sa, ls="--", color="k", label=f"V_sa = {V_sa:.2f} m/s")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 20); ax.set_ylim(-5, 10)
    ax.set_xlabel("V [m/s]"); ax.set_ylabel("k₂  [1/m]")
    ax.set_title("Tilt→steer gain k₂(V)  (Eq. 13)")
    ax.legend(fontsize=9)

    ax = axes[1]
    k1_hi = [L2.k1(v, p) for v in V_vec_hi]
    k1_lo = [L2.k1(v, p) for v in V_vec_lo]
    ax.plot(V_vec_lo, k1_lo, "C3", lw=1.8)
    ax.plot(V_vec_hi, k1_hi, "C0", lw=2, label="k₁(V)  [torque→steer gain]")
    ax.axvline(V_sa, ls="--", color="k", label=f"V_sa = {V_sa:.2f} m/s")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0, 20); ax.set_ylim(-0.05, 0.15)
    ax.set_xlabel("V [m/s]"); ax.set_ylabel("k₁  [m/N]")
    ax.set_title("Torque→steer gain k₁(V)  (Eq. 12)")
    ax.legend(fontsize=9)

    geom_ok = L2.stability_condition_geometry(p)
    fig.text(0.5, 0.01,
             f"Geometric condition bh > ac tanλ: {'✓ satisfied' if geom_ok else '✗ NOT satisfied'}",
             ha="center", fontsize=10,
             color="green" if geom_ok else "red")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "A2_level2_front_fork_gains")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig. A3  —  Level-3: root locus (paper Fig. 8)
# ─────────────────────────────────────────────────────────────────────────────

def plot_level3_root_locus(p=BIKE_WITH_RIDER, V_max=15.0):
    V_vec = np.linspace(0.01, V_max, 800)
    eigs  = L3.eigenvalue_sweep(V_vec, p)   # shape (N, 4)

    fig, ax = make_fig(1, 1, figsize=(8, 6),
                       title="Level-3 Root Locus vs Velocity (paper Fig. 8)")

    colors = ["C0", "C1", "C2", "C3"]
    labels = ["Weave / pendulum 1", "Weave / pendulum 2",
              "Capsize / fork 1",   "Capsize / fork 2"]

    for i in range(4):
        re = eigs[:, i].real
        im = eigs[:, i].imag
        # colour by stability
        col_arr = np.where(re > 0, "red", colors[i])
        # scatter with speed-based alpha
        sc = ax.scatter(re, im, c=V_vec, cmap="viridis",
                        s=6, zorder=4, label=labels[i])

    plt.colorbar(sc, ax=ax, label="Speed V [m/s]", fraction=0.04)
    ax.axvline(0, color="k", lw=1.2, ls="--")
    ax.axhline(0, color="k", lw=0.6, ls="--")

    # Mark zero-velocity poles
    eigs0 = L3.eigenvalues_at_speed(0.01, p)
    ax.scatter(eigs0.real, eigs0.imag, c="red", s=80, marker="o",
               zorder=6, label="Poles at V≈0")

    ax.set_xlabel("Real part  [rad/s]")
    ax.set_ylabel("Imaginary part  [rad/s]")
    ax.set_xlim(-16, 16)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "A3_level3_root_locus")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Fig. A4  —  Level-3: real parts of poles vs velocity (paper Fig. 9)
# ─────────────────────────────────────────────────────────────────────────────

def plot_level3_real_parts(p=BIKE_WITH_RIDER, V_max=15.0):
    """
    Mirrors paper Fig. 9.  Negative V values represent rear-wheel steering
    (sign reversal of V in the model, as described in the paper).
    """
    V_fwd  = np.linspace(0.05, V_max, 500)
    V_rear = np.linspace(0.05, V_max, 500)   # plotted at negative axis

    eigs_fwd  = L3.eigenvalue_sweep(V_fwd,  p)
    eigs_rear = L3.eigenvalue_sweep(V_rear, p)   # rear = −V in original model

    fig, ax = make_fig(1, 1, figsize=(9, 5),
                       title="Level-3 Real Parts of Poles vs Velocity (paper Fig. 9)")

    n_modes = 4
    cmap = cm.get_cmap("tab10")

    for i in range(n_modes):
        re_f = eigs_fwd[:, i].real
        re_r = eigs_rear[:, i].real    # rear-steered (plotted at −V)

        # Forward bicycle
        stable_mask = re_f <= 0
        if np.any(stable_mask):
            ax.plot(V_fwd[stable_mask], re_f[stable_mask],
                    color=cmap(i), lw=1.8)
        if np.any(~stable_mask):
            ax.plot(V_fwd[~stable_mask], re_f[~stable_mask],
                    color="red", lw=1.8)

        # Rear-steered bicycle (plotted at −V)
        re_r_mod = eigs_rear[:, i].real
        stable_r = re_r_mod <= 0
        if np.any(stable_r):
            ax.plot(-V_rear[stable_r], re_r_mod[stable_r],
                    color=cmap(i), lw=1.8, ls="--")
        if np.any(~stable_r):
            ax.plot(-V_rear[~stable_r], re_r_mod[~stable_r],
                    color="red", lw=1.8, ls="--")

    # Shade stable window
    V_lo, V_hi = L3.stability_range(V_fwd, p)
    if V_lo is not None:
        annotate_velocity_range(ax, V_lo, V_hi)
        ax.axvline(V_lo, color="green", ls=":", lw=1.2)
        ax.axvline(V_hi, color="green", ls=":", lw=1.2)
        ax.text(0.5*(V_lo+V_hi), ax.get_ylim()[0]*0.9,
                f"self-stable\n{V_lo:.1f}–{V_hi:.1f} m/s",
                ha="center", fontsize=8, color="green")

    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Velocity V [m/s]  (negative = rear-steered)")
    ax.set_ylabel("Re(pole)  [rad/s]")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "A4_level3_real_parts")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    print("[stability] Level-1 stability analysis...")
    plot_level1_stability()

    print("[stability] Level-2 front-fork gain curves...")
    plot_level2_front_fork_gains()

    print("[stability] Level-3 root locus...")
    plot_level3_root_locus()

    print("[stability] Level-3 real parts vs velocity...")
    plot_level3_real_parts()


if __name__ == "__main__":
    run_all()
    plt.show()
