"""
analysis/step_response.py
==========================
Time-domain step-response scenarios for all three model levels.

Scenarios
---------
  B1 : Level-1 — open-loop (unstable) roll response to initial lean
  B2 : Level-1 — P-controlled bicycle at various gains and speeds
  B3 : Level-2 — inverse response to handlebar torque step (paper Fig. 5)
  B4 : Level-3 — inverse response (path deviation η) with Whipple model
  B5 : Level-3 — free (T=0) roll recovery above & below critical speed
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import level1_inverted_pendulum as L1
from models import level2_front_fork         as L2
from models import level3_whipple            as L3
from utils.params import BIKE_WITH_RIDER
from utils.plotting import make_fig, save


p = BIKE_WITH_RIDER


# ─────────────────────────────────────────────────────────────────────────────
# B1  Level-1: open-loop instability
# ─────────────────────────────────────────────────────────────────────────────

def plot_B1_openloop(V=5.0):
    print(f"  B1: Level-1 open-loop at V={V} m/s")
    res = L1.simulate(V, (0, 2.0), (0.05, 0.0, 0.0),
                      controller=None, p=p)   # δ held at 0
    fig, axes = make_fig(1, 2, figsize=(10, 4),
                         title="B1 — Level-1 Open-Loop Roll (unstable)")
    axes[0].plot(res.t, np.degrees(res.phi), "C0")
    axes[0].set_xlabel("Time [s]"); axes[0].set_ylabel("Roll angle φ [°]")
    axes[0].set_title(f"V = {V} m/s  (open loop, δ = 0)")

    axes[1].plot(res.t, np.degrees(res.dphi), "C1")
    axes[1].set_xlabel("Time [s]"); axes[1].set_ylabel("Roll rate φ̇ [°/s]")
    axes[1].set_title("Roll rate")

    fig.tight_layout()
    save(fig, "B1_level1_openloop")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# B2  Level-1: P-control at different gains
# ─────────────────────────────────────────────────────────────────────────────

def plot_B2_pcontrol(V=6.0):
    print(f"  B2: Level-1 P-control at V={V} m/s")
    k2_min = L1.minimum_gain_for_stability(V, p)
    gains  = [0.5 * k2_min, k2_min, 2 * k2_min, 5 * k2_min]
    labels = [f"k₂={k:.2f} ({'stable' if k>k2_min else 'unstable'})"
              for k in gains]

    fig, ax = make_fig(1, 1, figsize=(8, 5),
                       title=f"B2 — Level-1 P-Control  (V = {V} m/s, k₂_min = {k2_min:.3f})")
    for k2_val, lbl in zip(gains, labels):
        res = L1.simulate_P_control(V, k2_val, (0, 4.0), phi0=0.1, p=p)
        ax.plot(res.t, np.degrees(res.phi), label=lbl, lw=1.8)

    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Roll angle φ [°]")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "B2_level1_pcontrol")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# B3  Level-2: inverse response — reproduction of paper Fig. 5
# ─────────────────────────────────────────────────────────────────────────────

def plot_B3_inverse_response_L2(V=5.0, T_step=1.0):
    print(f"  B3: Level-2 torque step (inverse response) at V={V} m/s")
    res = L2.simulate_constant_torque(V, T_step, (0, 3.0), phi0=0.0, p=p)

    # Compute approximate path deviation by double-integrating steer
    # (same linearisation as paper: dη/dt = V ψ, dψ/dt = V δ / b)
    psi = np.zeros_like(res.t)
    eta = np.zeros_like(res.t)
    for i in range(1, len(res.t)):
        dt = res.t[i] - res.t[i-1]
        psi[i] = psi[i-1] + (V / p.b) * res.delta[i-1] * dt
        eta[i] = eta[i-1] + V * psi[i-1] * dt

    fig, axes = make_fig(3, 1, figsize=(7, 8), sharex=True,
                         title=f"B3 — Level-2 Torque Step (paper Fig. 5)  V={V} m/s, T={T_step} N·m")

    axes[0].plot(res.t, eta, "C0", lw=2)
    axes[0].axhline(0, color="k", lw=0.6, ls="--")
    axes[0].set_ylabel("Path deviation η [m]")
    axes[0].set_title("(a) Path deviation — note inverse (non-minimum phase) response")

    axes[1].plot(res.t, np.degrees(res.phi), "C1", lw=2)
    axes[1].axhline(0, color="k", lw=0.6, ls="--")
    axes[1].set_ylabel("Roll angle φ [°]")
    axes[1].set_title("(b) Roll angle")

    axes[2].plot(res.t, np.degrees(res.delta), "C2", lw=2)
    axes[2].axhline(0, color="k", lw=0.6, ls="--")
    axes[2].set_ylabel("Steer angle δ [°]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("(c) Steer angle")

    fig.tight_layout()
    save(fig, "B3_level2_inverse_response")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# B4  Level-3: inverse response with full Whipple model
# ─────────────────────────────────────────────────────────────────────────────

def plot_B4_inverse_response_L3(V=5.0, T_step=1.0):
    print(f"  B4: Level-3 torque step at V={V} m/s")
    res = L3.simulate_torque_step(V, T_step=T_step, t_span=(0, 3.0), p=p)

    fig, axes = make_fig(3, 1, figsize=(7, 8), sharex=True,
                         title=f"B4 — Level-3 (Whipple) Torque Step  V={V} m/s")

    axes[0].plot(res.t, res.eta, "C0", lw=2)
    axes[0].axhline(0, color="k", lw=0.6, ls="--")
    axes[0].set_ylabel("Path deviation η [m]")
    axes[0].set_title("(a) Path deviation")

    axes[1].plot(res.t, np.degrees(res.phi), "C1", lw=2)
    axes[1].axhline(0, color="k", lw=0.6, ls="--")
    axes[1].set_ylabel("Roll angle φ [°]")
    axes[1].set_title("(b) Roll angle")

    axes[2].plot(res.t, np.degrees(res.delta), "C2", lw=2)
    axes[2].axhline(0, color="k", lw=0.6, ls="--")
    axes[2].set_ylabel("Steer angle δ [°]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("(c) Steer angle")

    fig.tight_layout()
    save(fig, "B4_level3_inverse_response")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# B5  Level-3: free roll above & below critical velocity
# ─────────────────────────────────────────────────────────────────────────────

def plot_B5_free_roll():
    print("  B5: Level-3 free roll above/below critical speed")
    from models.level3_whipple import stability_range
    V_vec = np.linspace(0.5, 15.0, 500)
    V_lo, V_hi = stability_range(V_vec, p)

    speeds = {
        f"V={V_lo*0.6:.1f} m/s (sub-critical)": V_lo * 0.6,
        f"V={V_lo:.2f} m/s (≈ critical)":       V_lo,
        f"V={(V_lo+V_hi)/2:.1f} m/s (self-stable)": (V_lo + V_hi) / 2,
        f"V={V_hi*1.1:.1f} m/s (above stable)": V_hi * 1.1,
    }

    fig, axes = make_fig(len(speeds), 1, figsize=(9, 10), sharex=False,
                         title="B5 — Level-3 Free Roll: Above & Below Critical Speed")

    for ax, (label, V) in zip(axes, speeds.items()):
        res = L3.simulate_free(V, phi0=0.05, t_span=(0, 8.0), p=p)
        ax.plot(res.t, np.degrees(res.phi), "C0", lw=1.8, label="φ (roll)")
        ax.plot(res.t, np.degrees(res.delta), "C1", lw=1.4,
                ls="--", label="δ (steer)")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_ylabel("Angle [°]")
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    save(fig, "B5_level3_free_roll")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    plot_B1_openloop()
    plot_B2_pcontrol()
    plot_B3_inverse_response_L2()
    plot_B4_inverse_response_L3()
    plot_B5_free_roll()


if __name__ == "__main__":
    run_all()
    plt.show()
