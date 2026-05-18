"""
analysis/self_stabilization.py
================================
Analysis of the self-stabilisation mechanism across all model levels.

Scenarios
---------
  C1 : Critical velocity from Level-2 vs geometry (c, λ sweep)
  C2 : Level-3 stable speed window: with vs without rider
  C3 : Effect of front-wheel inertia (gyroscopic effect) — paper Fig. 13
  C4 : Comparison of critical velocities across all three model levels
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import level1_inverted_pendulum as L1
from models import level2_front_fork         as L2
from models import level3_whipple            as L3
from utils.params import (BicycleParams, BIKE_WITH_RIDER,
                           BIKE_WITHOUT_RIDER)
from utils.plotting import make_fig, save, annotate_velocity_range


# ─────────────────────────────────────────────────────────────────────────────
# C1  Level-2 critical velocity vs trail c and head angle λ
# ─────────────────────────────────────────────────────────────────────────────

def plot_C1_Vc_geometry():
    print("  C1: V_c vs geometry (Level-2)")
    fig, axes = make_fig(1, 2, figsize=(11, 4),
                         title="C1 — Critical Velocity vs Geometry (Level-2, Eq. 16)")

    p_base = BicycleParams()

    # Vary trail c
    c_vals = np.linspace(0.01, 0.15, 200)
    Vc_c = []
    for c in c_vals:
        p = BicycleParams(c=c)
        Vc_c.append(L2.critical_velocity(p))
    axes[0].plot(c_vals * 100, Vc_c, "C0", lw=2)
    axes[0].axvline(p_base.c * 100, ls="--", color="k",
                    label=f"Default c = {p_base.c*100:.0f} cm")
    axes[0].set_xlabel("Trail c [cm]")
    axes[0].set_ylabel("Critical velocity V_c [m/s]")
    axes[0].set_title("V_c vs trail c  (Eq. 10)")
    axes[0].legend(fontsize=9)

    # Vary head angle λ
    lam_vals = np.linspace(np.radians(60), np.radians(85), 200)
    Vc_lam = []
    for lam in lam_vals:
        p = BicycleParams(lam=lam)
        Vc_lam.append(L2.critical_velocity(p))
    axes[1].plot(np.degrees(lam_vals), Vc_lam, "C1", lw=2)
    axes[1].axvline(np.degrees(p_base.lam), ls="--", color="k",
                    label=f"Default λ = {np.degrees(p_base.lam):.0f}°")
    axes[1].set_xlabel("Head angle λ [°]")
    axes[1].set_ylabel("Critical velocity V_c [m/s]")
    axes[1].set_title("V_c vs head angle λ  (Eq. 16)")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    save(fig, "C1_critical_velocity_geometry")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# C2  Level-3: stable window with vs without rider
# ─────────────────────────────────────────────────────────────────────────────

def plot_C2_stable_window_rider():
    print("  C2: Stable window with/without rider (Level-3)")
    V_vec = np.linspace(0.1, 15.0, 600)

    configs = {
        "With rider":    BIKE_WITH_RIDER,
        "Without rider": BIKE_WITHOUT_RIDER,
    }

    fig, axes = make_fig(1, 2, figsize=(12, 5),
                         title="C2 — Stable Speed Window: With vs Without Rider (Level-3)")

    for ax, (label, params) in zip(axes, configs.items()):
        eigs = L3.eigenvalue_sweep(V_vec, params)
        V_lo, V_hi = L3.stability_range(V_vec, params)

        for i in range(4):
            re = eigs[:, i].real
            stable = re <= 0
            color = "C" + str(i)
            if np.any(stable):
                ax.plot(V_vec[stable], re[stable], color=color, lw=1.8)
            if np.any(~stable):
                ax.plot(V_vec[~stable], re[~stable], color="red", lw=1.8)

        if V_lo is not None:
            annotate_velocity_range(ax, V_lo, V_hi)
            ax.text((V_lo+V_hi)/2, -1.0,
                    f"{V_lo:.1f}–{V_hi:.1f} m/s",
                    ha="center", fontsize=9, color="green")

        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Re(pole) [rad/s]")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "C2_stable_window_rider_comparison")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# C3  Gyroscopic effect: vary front wheel moment of inertia (paper Fig. 13)
# ─────────────────────────────────────────────────────────────────────────────

def plot_C3_gyroscopic_effect():
    """
    Reproduces the spirit of paper Fig. 13.
    We vary the front-wheel spin inertia Jfw_yy (entering C₁ via gyroscopic
    coupling) and observe how the stable speed window shifts.

    Note: the full Whipple model's C₁ matrix in Eq. (25) already encodes the
    gyroscopic contribution from Jfw_yy.  Here we scale the off-diagonal
    gyroscopic terms in C₁ proportionally to Jfw_yy / Jfw_yy_nominal.
    """
    print("  C3: Gyroscopic effect (vary front wheel inertia)")
    import copy, dataclasses

    V_vec = np.linspace(0.1, 15.0, 400)
    Jfw_nom = BIKE_WITH_RIDER.Jfw_yy       # nominal = 0.14 kg·m²
    Jfw_vals = np.logspace(-3, np.log10(0.184), 30)

    Vc_stable   = []
    Vc_capsize  = []

    for Jfw in Jfw_vals:
        # Scale the gyroscopic column of C₁ (upper-right element encodes
        # gyroscopic coupling of front wheel onto roll equation)
        scale = Jfw / Jfw_nom
        p_mod = copy.deepcopy(BIKE_WITH_RIDER)
        C1_scaled = BIKE_WITH_RIDER.C1_w.copy()
        # Column 1 of C₁ (coupling δ̇ → φ̈) is gyroscopic in origin
        C1_scaled[:, 1] *= scale
        p_mod.C1_w = C1_scaled

        # Also scale off-diagonal K terms slightly
        V_lo, V_hi = L3.stability_range(V_vec, p_mod)
        Vc_stable.append(V_lo if V_lo else np.nan)
        Vc_capsize.append(V_hi if V_hi else np.nan)

    fig, ax = make_fig(1, 1, figsize=(7, 5),
                       title="C3 — Critical Velocities vs Front Wheel Inertia (cf. Fig. 13)")

    ax.loglog(Jfw_vals, Vc_stable,  "C0", lw=2, label="V_stable (weave mode)")
    ax.loglog(Jfw_vals, Vc_capsize, "C1", lw=2, ls="--", label="V_capsize (capsize mode)")
    ax.axvline(Jfw_nom, ls=":", color="k", label=f"Nominal J_fw = {Jfw_nom} kg·m²")
    ax.set_xlabel("Front wheel spin inertia J_fw [kg·m²]")
    ax.set_ylabel("Velocity [m/s]")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "C3_gyroscopic_effect")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# C4  Summary: critical velocities across model levels
# ─────────────────────────────────────────────────────────────────────────────

def print_C4_summary():
    print("\n── C4: Critical Velocity Summary ──────────────────────────────")
    p = BIKE_WITH_RIDER

    # Level-1 (unstable, no self-stabilisation)
    pole_stable, pole_unstable = L1.open_loop_poles(p)
    print(f"  Level-1 open-loop poles: {pole_stable:.3f}, +{pole_unstable:.3f} rad/s")
    print(f"  Level-1 has NO self-stabilisation (no front fork)")

    # Level-2
    Vsa = L2.self_alignment_velocity(p)
    Vc2 = L2.critical_velocity(p)
    geom = L2.stability_condition_geometry(p)
    print(f"\n  Level-2  V_sa = V_c = {Vc2:.3f} m/s  (static front-fork model, Eq. 16)")
    print(f"           Geometric condition (Eq. 17): {'satisfied ✓' if geom else 'NOT satisfied ✗'}")

    # Level-3
    V_vec = np.linspace(0.1, 20.0, 1000)
    V_lo, V_hi = L3.stability_range(V_vec, p)
    print(f"\n  Level-3  V_low = {V_lo:.3f} m/s,  V_high = {V_hi:.3f} m/s")
    print(f"           Self-stable window: {V_hi-V_lo:.3f} m/s wide")
    print(f"           (Paper reports: 5.96–10.36 m/s for bicycle with rider)")
    print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    plot_C1_Vc_geometry()
    plot_C2_stable_window_rider()
    plot_C3_gyroscopic_effect()
    print_C4_summary()


if __name__ == "__main__":
    run_all()
    plt.show()
