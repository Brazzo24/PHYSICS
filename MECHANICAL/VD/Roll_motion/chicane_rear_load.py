"""
Rear Load Loss Analysis — Corner Exit with Rapid Roll Recovery
==============================================================
Focuses on the apex-to-exit phase:
  - Demonstrates WHY rapid phi_dot unloads the rear via Mozzi axis tilt
  - Simulates four improvement strategies side by side
  - Shows traction margin (how close the rear is to spinning)

Builds on chicane_mozzi.py — same model, extended scenario.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field

# ══════════════════════════════════════════════════════════════════
# 1.  PARAMETERS  (dataclass so we can vary them per scenario)
# ══════════════════════════════════════════════════════════════════
@dataclass
class BikeParams:
    m         : float = 220.0   # total mass [kg]
    h_com     : float = 0.58    # CoM height [m]
    wheelbase : float = 1.40    # [m]
    I_roll    : float = 30.0    # roll moment of inertia [kg·m²]
    g         : float = 9.81    # [m/s²]
    v_fwd     : float = 14.0    # forward speed [m/s]  (constant here)

    # Pacejka
    B : float = 10.0
    C : float = 1.9
    D : float = 1.0    # peak friction coefficient
    E : float = -1.0

    # PD lean controller gains
    Kp : float = 120.0
    Kd : float = 25.0

    # Anti-squat factor: shifts rear static load baseline
    # 0.0 = neutral geometry, 0.15 = 15% extra rear static preload
    anti_squat : float = 0.0

    # Traction control: clamp throttle when rear load < this fraction of static
    tc_threshold : float = 0.0   # 0 = off

    label : str = "Baseline"
    color : str = "royalblue"


# ══════════════════════════════════════════════════════════════════
# 2.  CORE PHYSICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def mozzi_axis(omega, v_P, r_P=np.zeros(3)):
    omega = np.asarray(omega, float)
    v_P   = np.asarray(v_P,   float)
    r_P   = np.asarray(r_P,   float)
    w2    = float(omega @ omega)
    if w2 < 1e-10:
        return None, None, None
    h      = float(omega @ v_P) / w2
    r_axis = r_P + np.cross(omega, v_P) / w2
    return h, r_axis, h * omega


def bike_mozzi_pitch(phi, phi_dot, psi_dot, p: BikeParams):
    """Return Mozzi screw pitch and axis point for current bike state."""
    omega = np.array([phi_dot * np.cos(phi),
                      phi_dot * np.sin(phi),
                      psi_dot])
    r_com = np.array([0.0,
                      -p.h_com * np.sin(phi),
                       p.h_com * np.cos(phi)])
    v_com = np.array([p.v_fwd,
                      0.0,
                      phi_dot * p.h_com * np.cos(phi)])
    h_sc, r_ax, _ = mozzi_axis(omega, v_com, r_com)
    return h_sc, r_ax


def vertical_loads(phi, phi_dot, psi_dot, p: BikeParams):
    """
    Vertical tyre loads with Mozzi-derived load transfer + anti-squat offset.
    """
    F_static = p.m * p.g / 2.0

    # Anti-squat geometry: shifts baseline rear load up
    anti_squat_N = p.anti_squat * F_static
    F_static_f   = F_static - anti_squat_N
    F_static_r   = F_static + anti_squat_N

    h_sc, r_ax = bike_mozzi_pitch(phi, phi_dot, psi_dot, p)

    if r_ax is not None:
        axis_pitch = np.arctan2(r_ax[2], r_ax[0] + 1e-9)
        a_roll     = phi_dot**2 * p.h_com * np.cos(phi)
        # Coupling coefficient κ: fraction of inertial moment transferred
        # to longitudinal axle load — empirically ~0.25 for typical geometry
        kappa      = 0.25
        delta_F    = kappa * p.m * a_roll * np.tan(axis_pitch)
        delta_F    = np.clip(delta_F, -F_static * 0.55, F_static * 0.55)
    else:
        delta_F = 0.0

    F_front = max(F_static_f + delta_F, 10.0)
    F_rear  = max(F_static_r - delta_F, 10.0)
    return F_front, F_rear


def pacejka_Fy(alpha, F_z, p: BikeParams):
    Fz0  = p.m * p.g / 2.0
    mu   = p.D * (F_z / Fz0)
    phi_ = (1 - p.E) * alpha + (p.E / p.B) * np.arctan(p.B * alpha)
    return mu * F_z * np.sin(p.C * np.arctan(p.B * phi_))


def pacejka_Fx_max(F_z, p: BikeParams):
    """
    Peak longitudinal force capacity of rear tyre (simplified: same Pacejka
    shape, peak at zero slip angle — full traction budget).
    μ_x ≈ D for bias-ply; slightly less for combined slip.
    """
    Fz0 = p.m * p.g / 2.0
    return p.D * F_z   # peak Fx available at this F_z


# ══════════════════════════════════════════════════════════════════
# 3.  CORNER-EXIT SCENARIO
#     Starts at apex lean (phi_max), picks bike up while accelerating.
#     phi_ref goes from apex lean → 0 with a configurable pick-up rate.
# ══════════════════════════════════════════════════════════════════
def phi_ref_exit(t, phi_apex_deg, pickup_rate_deg_s):
    """
    Desired lean: from apex angle, reduce at pickup_rate deg/s toward 0.
    """
    phi_apex = np.radians(phi_apex_deg)
    ramp     = np.radians(pickup_rate_deg_s) * t
    return max(phi_apex - ramp, 0.0)


def make_ode(p: BikeParams, phi_apex_deg=32.0, pickup_rate=40.0,
             throttle_Fx=800.0):
    """
    Returns an ODE rhs for corner-exit scenario.
    throttle_Fx : demanded rear longitudinal force [N]
    """
    def rhs(t, y):
        phi, phi_dot = y[0], y[1]

        phi_ref    = phi_ref_exit(t, phi_apex_deg, pickup_rate)
        phi_err    = phi_ref - phi
        tau_roll   = p.Kp * phi_err + p.Kd * (0.0 - phi_dot)

        psi_dot    = p.v_fwd * np.tan(phi) / p.wheelbase
        F_front, F_rear = vertical_loads(phi, phi_dot, psi_dot, p)

        # Traction control: reduce throttle demand if rear is unloaded
        Fx_available = pacejka_Fx_max(F_rear, p)
        if p.tc_threshold > 0:
            load_ratio = F_rear / (p.m * p.g / 2.0)
            if load_ratio < p.tc_threshold:
                tc_factor  = load_ratio / p.tc_threshold
                Fx_demand  = throttle_Fx * tc_factor
            else:
                Fx_demand  = throttle_Fx
        else:
            Fx_demand = throttle_Fx

        # Traction margin: positive = grip available, negative = spinning
        traction_margin = Fx_available - Fx_demand

        alpha      = np.arctan2(psi_dot * p.wheelbase, p.v_fwd)
        Fy_f       = pacejka_Fy( alpha, F_front, p)
        Fy_r       = pacejka_Fy(-alpha, F_rear,  p)

        lat_moment  = (Fy_f + Fy_r) * p.h_com * np.cos(phi)
        grav_moment =  p.m * p.g * p.h_com * np.sin(phi)
        phi_ddot    = (tau_roll + grav_moment - lat_moment) / p.I_roll

        return [phi_dot, phi_ddot]

    return rhs


def run_scenario(p: BikeParams, phi_apex_deg=32.0,
                 pickup_rate=40.0, throttle_Fx=800.0, n_pts=600):
    """Integrate and return full post-processed result dict."""
    rhs    = make_ode(p, phi_apex_deg, pickup_rate, throttle_Fx)
    t_span = (0.0, 2.5)
    t_eval = np.linspace(*t_span, n_pts)
    y0     = [np.radians(phi_apex_deg), 0.0]

    sol    = solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                       method='RK45', rtol=1e-7, atol=1e-9)

    t      = sol.t
    phi    = sol.y[0]
    phid   = sol.y[1]

    # Post-process
    pitches   = np.zeros(len(t))
    F_fronts  = np.zeros(len(t))
    F_rears   = np.zeros(len(t))
    Fx_avail  = np.zeros(len(t))
    Fx_demand = np.zeros(len(t))
    trac_marg = np.zeros(len(t))
    phi_refs  = np.array([phi_ref_exit(ti, phi_apex_deg, pickup_rate)
                          for ti in t])

    for i in range(len(t)):
        psid         = p.v_fwd * np.tan(phi[i]) / p.wheelbase
        h_sc, _      = bike_mozzi_pitch(phi[i], phid[i], psid, p)
        pitches[i]   = h_sc if h_sc is not None else 0.0

        Ff, Fr       = vertical_loads(phi[i], phid[i], psid, p)
        F_fronts[i]  = Ff
        F_rears[i]   = Fr
        Fx_avail[i]  = pacejka_Fx_max(Fr, p)

        load_ratio   = Fr / (p.m * p.g / 2.0)
        if p.tc_threshold > 0 and load_ratio < p.tc_threshold:
            Fx_demand[i] = throttle_Fx * (load_ratio / p.tc_threshold)
        else:
            Fx_demand[i] = throttle_Fx

        trac_marg[i] = Fx_avail[i] - Fx_demand[i]

    return dict(t=t, phi=phi, phid=phid, phi_ref=phi_refs,
                pitch=pitches, F_front=F_fronts, F_rear=F_rears,
                Fx_avail=Fx_avail, Fx_demand=Fx_demand,
                trac_margin=trac_marg, label=p.label, color=p.color)


# ══════════════════════════════════════════════════════════════════
# 4.  DEFINE SCENARIOS
# ══════════════════════════════════════════════════════════════════
THROTTLE  = 600.0   # N longitudinal demand (aggressive corner exit)
APEX_LEAN = 32.0    # deg

scenarios = [
    # Baseline: aggressive pick-up rate
    (BikeParams(label="Baseline",            color="royalblue"),   40.0),

    # Fix 1: Slower pick-up rate (rider technique / smoother steering)
    (BikeParams(label="Slower pick-up",      color="seagreen"),    18.0),

    # Fix 2: Lower CoM (e.g. lower suspension, crouching)
    (BikeParams(h_com=0.48, label="Lower CoM (-10 cm)", color="darkorange"), 40.0),

    # Fix 3: Traction control (cuts throttle below 80% rear load)
    (BikeParams(tc_threshold=0.80,
                label="Traction Control",    color="crimson"),     40.0),

    # Fix 4: Anti-squat geometry (+15% rear static load baseline)
    (BikeParams(anti_squat=0.15,
                label="Anti-squat +15%",     color="purple"),      40.0),
]

results = [run_scenario(p, APEX_LEAN, rate, THROTTLE)
           for p, rate in scenarios]


# ══════════════════════════════════════════════════════════════════
# 5.  FIGURE 1 — Main analysis: Mozzi pitch → rear load → traction
# ══════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
fig1.suptitle(
    "Corner Exit: Rear Load Loss via Mozzi Axis — Baseline Anatomy",
    fontsize=13, fontweight='bold', y=0.99)

base = results[0]
t    = base['t']
F_st = BikeParams().m * BikeParams().g / 2.0

# — Panel 1: Lean angle & roll rate —
ax1a = axes[0]
ax1b = ax1a.twinx()
ax1a.plot(t, np.degrees(base['phi']),  'royalblue', lw=2,   label='φ actual [°]')
ax1a.plot(t, np.degrees(base['phi_ref']), 'k--',   lw=1.5, label='φ ref [°]')
ax1b.plot(t, np.degrees(base['phid']), 'cornflowerblue',
          lw=1.5, ls=':', label='φ̇ [°/s]')
ax1a.set_ylabel("Lean angle [°]", color='royalblue')
ax1b.set_ylabel("Roll rate φ̇ [°/s]", color='cornflowerblue')
ax1a.set_title("① Roll Angle (picking up from apex)")
lines1 = ax1a.get_lines() + ax1b.get_lines()
ax1a.legend(lines1, [l.get_label() for l in lines1], fontsize=8, loc='upper right')
ax1a.grid(True, alpha=0.3)
ax1a.axhline(0, color='k', lw=0.5)

# — Panel 2: Mozzi pitch —
ax2 = axes[1]
ax2.plot(t, base['pitch'], 'crimson', lw=2)
ax2.fill_between(t, base['pitch'], 0,
                 where=base['pitch'] > 0, alpha=0.18, color='crimson',
                 label='h>0 → front loaded')
ax2.fill_between(t, base['pitch'], 0,
                 where=base['pitch'] < 0, alpha=0.18, color='navy',
                 label='h<0 → rear loaded')
ax2.axhline(0, color='k', lw=0.8, ls='--')
ax2.set_ylabel("Mozzi pitch h [m/rad]")
ax2.set_title("② Mozzi Screw Pitch  (encodes inertial load transfer direction)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# — Panel 3: Vertical loads —
ax3 = axes[2]
ax3.plot(t, base['F_rear'],  'tomato', lw=2.5, label='F_z rear')
ax3.plot(t, base['F_front'], 'steelblue', lw=2.5, label='F_z front')
ax3.axhline(F_st, color='k', lw=0.9, ls='--', label=f'Static ({F_st:.0f} N)')
ax3.fill_between(t, base['F_rear'], F_st,
                 where=base['F_rear'] < F_st,
                 alpha=0.20, color='tomato', label='Rear load deficit')
ax3.set_ylabel("Vertical Force [N]")
ax3.set_title("③ Vertical Tyre Loads  (rear deficit = spin risk)")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# — Panel 4: Traction margin —
ax4 = axes[3]
ax4.plot(t, base['Fx_avail'],  'seagreen', lw=2,   label='Fx available (grip)')
ax4.plot(t, base['Fx_demand'], 'k',        lw=2,   label='Fx demand (throttle)', ls='--')
ax4.fill_between(t, base['Fx_avail'], base['Fx_demand'],
                 where=base['trac_margin'] < 0,
                 alpha=0.30, color='red', label='⚠ SPINNING (demand > grip)')
ax4.fill_between(t, base['Fx_avail'], base['Fx_demand'],
                 where=base['trac_margin'] >= 0,
                 alpha=0.12, color='seagreen', label='Traction margin OK')
ax4.set_ylabel("Longitudinal Force [N]")
ax4.set_xlabel("Time after apex [s]")
ax4.set_title("④ Traction Budget  (Pacejka Fx_max vs. throttle demand)")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("/mnt/user-data/outputs/rear_load_anatomy.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved.")


# ══════════════════════════════════════════════════════════════════
# 6.  FIGURE 2 — Four strategies compared
# ══════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig2.suptitle(
    "Four Improvement Strategies — Rear Load & Traction Margin Comparison",
    fontsize=13, fontweight='bold', y=0.99)

for r in results:
    t = r['t']
    axes2[0].plot(t, r['F_rear'],      color=r['color'], lw=2, label=r['label'])
    axes2[1].plot(t, r['pitch'],       color=r['color'], lw=2, label=r['label'])
    axes2[2].plot(t, r['trac_margin'], color=r['color'], lw=2, label=r['label'])

axes2[0].axhline(F_st, color='k', lw=0.9, ls='--', label='Static')
axes2[0].set_ylabel("F_z rear [N]")
axes2[0].set_title("Rear Vertical Load")
axes2[0].legend(fontsize=8); axes2[0].grid(True, alpha=0.3)

axes2[1].axhline(0, color='k', lw=0.9, ls='--')
axes2[1].set_ylabel("Mozzi pitch h [m/rad]")
axes2[1].set_title("Mozzi Screw Pitch  (smaller |h| = less load transfer)")
axes2[1].legend(fontsize=8); axes2[1].grid(True, alpha=0.3)

axes2[2].axhline(0, color='k', lw=1.2, ls='-')
axes2[2].fill_between(results[0]['t'],
                       np.zeros(len(results[0]['t'])),
                       results[0]['trac_margin'],
                       where=results[0]['trac_margin'] < 0,
                       alpha=0.08, color='red')
axes2[2].set_ylabel("Traction margin [N]\n(negative = spinning)")
axes2[2].set_xlabel("Time after apex [s]")
axes2[2].set_title("Traction Margin  (Fx_available − Fx_demand)")
axes2[2].legend(fontsize=8); axes2[2].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("/mnt/user-data/outputs/rear_load_strategies.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved.")


# ══════════════════════════════════════════════════════════════════
# 7.  FIGURE 3 — Phase portrait: Mozzi pitch vs rear load deficit
#     This is the key diagnostic plot: shows each strategy's
#     "operating trajectory" in the (h, ΔF_rear) space.
# ══════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(9, 7))
ax.set_title("Phase Portrait: Mozzi Pitch vs. Rear Load Deficit\n"
             "(top-right = most dangerous — high h, high deficit)",
             fontsize=11)

for r in results:
    delta_F = r['F_rear'] - F_st   # negative = deficit
    ax.plot(r['pitch'], delta_F,
            color=r['color'], lw=2, label=r['label'], alpha=0.85)
    # Mark start (apex) and end
    ax.scatter(r['pitch'][0],  delta_F[0],
               marker='o', s=90, color=r['color'], zorder=5)
    ax.scatter(r['pitch'][-1], delta_F[-1],
               marker='s', s=70, color=r['color'], zorder=5)

# Danger zone
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.fill_between([0, ax.get_xlim()[1] if ax.get_xlim()[1] > 0.5 else 2.0],
                [-F_st*0.4, -F_st*0.4], [0, 0],
                alpha=0.08, color='red')
ax.text(0.05, -F_st*0.18, "⚠  Spin risk zone\n   (h>0, rear unloaded)",
        color='red', fontsize=9, alpha=0.8)

ax.set_xlabel("Mozzi screw pitch h [m/rad]")
ax.set_ylabel("Rear load deviation ΔF_z [N]  (negative = deficit)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate direction of travel
ax.annotate("● apex", xy=(results[0]['pitch'][0],
            results[0]['F_rear'][0] - F_st),
            fontsize=8, color='royalblue',
            xytext=(0.3, -20), arrowprops=dict(arrowstyle='->', color='grey'))

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/rear_load_phase_portrait.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved.")


# ══════════════════════════════════════════════════════════════════
# 8.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print(f"{'Strategy':<25} {'Min F_rear [N]':>14} {'Max |h|':>10} "
      f"{'Spin time [s]':>14} {'Max deficit [N]':>16}")
print("═"*70)

for r in results:
    min_Fr   = np.min(r['F_rear'])
    max_h    = np.max(np.abs(r['pitch']))
    spin_dur = np.sum(r['trac_margin'] < 0) * (r['t'][-1]/len(r['t']))
    max_def  = F_st - min_Fr
    print(f"{r['label']:<25} {min_Fr:>14.1f} {max_h:>10.4f} "
          f"{spin_dur:>14.3f} {max_def:>16.1f}")

print("═"*70)
print("\n✓ All outputs written.")
