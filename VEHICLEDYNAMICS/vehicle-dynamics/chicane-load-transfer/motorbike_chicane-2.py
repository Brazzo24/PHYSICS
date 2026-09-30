"""
Motorbike chicane — roll dynamics + tyre forces
================================================

Physics model
─────────────
State: [φ, φ̇], integrated with scipy RK45.

Roll equation (closed-loop 2nd-order lean tracking):
  φ̈ = ω_n²·(φ_eq − φ) − 2·ζ·ω_n·φ̇
  φ_eq = arctan(v²·|κ|/g)   ← always positive; sign handled separately

Normal load distribution — three contributions:

  (a) Static split
        Fzf_stat = m·g·Lr/L,   Fzr_stat = m·g·Lf/L

  (b) Roll-axis centripetal load transfer  ← KEY FIX vs. previous version
      On a single-track vehicle the bike always leans INTO the corner,
      so the centripetal force is always directed toward the CoM from
      the contact patch — regardless of left or right turn direction.
      The magnitude of the longitudinal (pitch-axis) load transfer is
      therefore proportional to |a_y|, NOT signed a_y:

        ΔFz_lat = m·v²·|κ|·(h − h_ra) / L
        → Fzf decreases,  Fzr increases  (same for left AND right corner)

  (c) Roll-axis inertial couple during φ̈ transients
      The angular acceleration Ix·φ̈ has a component along the roll axis:
        M_ra = Ix·φ̈·sin(ε_ra)
      This IS signed — it reverses during the chicane direction change
      (φ̈ changes sign as the bike crosses the upright position).
      Effect: front loads briefly during one half of the transition,
      rear loads during the other half.

Tyre lateral forces — Pacejka Magic Formula (simplified):
  Fy = Fz · D · sin(C · arctan(B·α − E·(B·α − arctan(B·α))))
  Slip angles: α_f = Lf·κ,  α_r = −Lr·κ  (signed — same as before)

Roll-axis geometry:
  Roll axis from front contact (h_rc_f ≈ 0.03 m, telescopic fork)
  to rear contact (h_rc_r ≈ 0.17 m, swing-arm).
  Inclination: ε_ra = arctan((h_rc_f − h_rc_r)/L) ≈ −5.5°
  Height at CoM: h_ra = h_rc_r + (h_rc_f − h_rc_r)·(Lr/L)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────
p = dict(
    m       = 210.0,   # total mass  [kg]
    Ix      = 28.0,    # roll moment of inertia  [kg·m²]
    h       = 0.62,    # CoM height  [m]
    L       = 1.45,    # wheelbase  [m]
    Lf      = 0.72,    # CoM–front distance  [m]  → Lr = 0.73 m
    v       = 18.0,    # speed  [m/s]  (≈ 65 km/h)
    g       = 9.81,

    # Roll response (sport bike: ω_n ≈ 10 rad/s, ζ ≈ 0.7)
    omega_n = 10.0,
    zeta    =  0.7,

    # Roll-axis contact heights
    h_rc_f  = 0.03,    # front  [m]
    h_rc_r  = 0.17,    # rear   [m]

    # Pacejka tyre parameters
    B = 10.0,  C = 1.9,  D = 1.05,  E = 0.97,

    R_turn  = 50.0,    # corner radius  [m]
)

SIM_T = 7.0


# ─────────────────────────────────────────────────────────────────
# CURVATURE PROFILE  κ(t)  [1/m]
# ─────────────────────────────────────────────────────────────────
def sigmoid(x, k=18.0):
    return 1.0 / (1.0 + np.exp(-k * x))

def curvature_profile(t, R):
    """Signed curvature: +1/R = left, -1/R = right."""
    inv = 1.0 / R
    return (
          inv * sigmoid(t - 0.7)
        - inv * sigmoid(t - 2.7)
        - inv * sigmoid(t - 2.7)
        + inv * sigmoid(t - 5.0)
    )


# ─────────────────────────────────────────────────────────────────
# PACEJKA MAGIC FORMULA
# ─────────────────────────────────────────────────────────────────
def magic_formula(alpha, Fz, B, C, D, E):
    Fz = max(float(Fz), 0.0)
    return Fz * D * np.sin(C * np.arctan(
        B*alpha - E*(B*alpha - np.arctan(B*alpha))
    ))


# ─────────────────────────────────────────────────────────────────
# ODE  — state = [φ, φ̇]
# ─────────────────────────────────────────────────────────────────
def equations(t, state, p):
    phi, phi_dot = state
    v  = p['v'];  g  = p['g']
    wn = p['omega_n'];  z = p['zeta']

    kappa  = curvature_profile(t, p['R_turn'])
    # Equilibrium lean depends only on |curvature| (left/right symmetric)
    phi_eq = np.sign(kappa) * np.arctan(v**2 * abs(kappa) / g)

    phi_ddot = wn**2 * (phi_eq - phi) - 2*z*wn * phi_dot
    return [phi_dot, phi_ddot]


# ─────────────────────────────────────────────────────────────────
# INTEGRATE
# ─────────────────────────────────────────────────────────────────
t_eval = np.linspace(0, SIM_T, 1400)
sol = solve_ivp(
    lambda t, y: equations(t, y, p),
    (0.0, SIM_T), [0.0, 0.0],
    t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10
)
t       = sol.t
phi     = sol.y[0]
phi_dot = sol.y[1]


# ─────────────────────────────────────────────────────────────────
# POST-PROCESS
# ─────────────────────────────────────────────────────────────────
m   = p['m'];  g  = p['g'];  h  = p['h']
L   = p['L'];  Lf = p['Lf']; Lr = L - Lf
Ix  = p['Ix']; v  = p['v']
h_rc_f = p['h_rc_f']; h_rc_r = p['h_rc_r']
B, C, D, E = p['B'], p['C'], p['D'], p['E']

h_ra       = h_rc_r + (h_rc_f - h_rc_r) * (Lr / L)
epsilon_ra = np.arctan2(h_rc_f - h_rc_r, L)

kappa      = np.array([curvature_profile(ti, p['R_turn']) for ti in t])
phi_eq_arr = np.sign(kappa) * np.arctan(v**2 * np.abs(kappa) / g)
phi_ddot   = np.gradient(phi_dot, t)

# (b) Roll-axis centripetal load transfer — use |κ|, symmetric left/right
#     The bike always leans into the corner; relative to the lean axis,
#     left and right are identical. The pitch moment arm is (h - h_ra).
a_y_mag     = v**2 * np.abs(kappa)             # |lateral acceleration|  [m/s²]
M_pitch_lat = m * a_y_mag * (h - h_ra)         # always positive → always rear-biased

# (c) Roll-axis inertial couple — this IS signed (reverses during chicane)
M_rollaxis  = Ix * phi_ddot * np.sin(epsilon_ra)

# Normal loads
delta_Fz = (M_pitch_lat + M_rollaxis) / L
Fzf = np.clip(m*g*Lr/L - delta_Fz, 0, None)
Fzr = np.clip(m*g*Lf/L + delta_Fz, 0, None)

# Lateral tyre forces — slip angles remain signed (centripetal direction)
alpha_f =  Lf * kappa
alpha_r = -Lr * kappa
Fyf = np.array([magic_formula(a, fz, B, C, D, E) for a, fz in zip(alpha_f, Fzf)])
Fyr = np.array([magic_formula(a, fz, B, C, D, E) for a, fz in zip(alpha_r, Fzr)])
util_f = np.abs(Fyf) / np.maximum(D * Fzf, 1.0)
util_r = np.abs(Fyr) / np.maximum(D * Fzr, 1.0)

# Bird's-eye trajectory
heading = np.zeros(len(t))
bx, by  = np.zeros(len(t)), np.zeros(len(t))
for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    heading[i] = heading[i-1] - kappa[i-1]*v*dt
    bx[i] = bx[i-1] + v*dt*np.cos(heading[i-1])
    by[i] = by[i-1] + v*dt*np.sin(heading[i-1])

lean_deg = np.degrees(phi)
Fzf_s, Fzr_s = m*g*Lr/L, m*g*Lf/L


# ─────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#f8f7f4',
    'axes.facecolor':   '#f8f7f4',
    'axes.grid':        True,
    'grid.color':       '#e0ddd6',
    'grid.linewidth':   0.5,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'font.family':      'sans-serif',
    'font.size':        10,
})

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    f'Motorbike chicane  |  v = {v} m/s ({v*3.6:.0f} km/h)  |  '
    f'R = {p["R_turn"]} m  |  m = {m} kg  |  '
    f'ω_n = {p["omega_n"]} rad/s  ζ = {p["zeta"]}',
    fontsize=12, fontweight='500', y=0.99
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.42)
ax_track = fig.add_subplot(gs[0:2, 0])
ax_lean  = fig.add_subplot(gs[0, 1])
ax_rate  = fig.add_subplot(gs[0, 2])
ax_fzf   = fig.add_subplot(gs[1, 1])
ax_fzr   = fig.add_subplot(gs[1, 2])
ax_fyf   = fig.add_subplot(gs[2, 0])
ax_fyr   = fig.add_subplot(gs[2, 1])
ax_util  = fig.add_subplot(gs[2, 2])

# Track
norm_c = plt.Normalize(-35, 35)
cmap_c = plt.cm.RdYlBu_r
for i in range(len(t)-1):
    ax_track.plot(bx[i:i+2], by[i:i+2],
                  color=cmap_c(norm_c(lean_deg[i])), lw=2.2)
ax_track.plot(bx[0],  by[0],  'o', color='#3266ad', ms=7, zorder=5, label='Start')
ax_track.plot(bx[-1], by[-1], 's', color='#C04828', ms=7, zorder=5, label='End')
ax_track.set_aspect('equal')
ax_track.set_title("Bird's-eye track  (colour = lean)")
ax_track.set_xlabel('x [m]'); ax_track.set_ylabel('y [m]')
ax_track.legend(fontsize=9, frameon=False)
sm = plt.cm.ScalarMappable(cmap=cmap_c, norm=norm_c); sm.set_array([])
fig.colorbar(sm, ax=ax_track, fraction=0.04, pad=0.04).set_label('Lean [°]', fontsize=9)

# Roll angle
ax_lean.plot(t, lean_deg,               color='#3266ad', lw=1.8, label='φ actual')
ax_lean.plot(t, np.degrees(phi_eq_arr), color='#3266ad', lw=1.0,
             ls='--', alpha=0.5,         label='φ_eq ideal')
ax_lean.axhline(0, color='#aaa', lw=0.6, ls=':')
ax_lean.fill_between(t, lean_deg, alpha=0.10, color='#3266ad')
ax_lean.set_title('Roll angle φ'); ax_lean.set_xlabel('Time [s]')
ax_lean.set_ylabel('φ [°]'); ax_lean.legend(fontsize=9, frameon=False)

# Roll rate
ax_rate.plot(t, np.degrees(phi_dot), color='#C04828', lw=1.8)
ax_rate.axhline(0, color='#aaa', lw=0.6, ls=':')
ax_rate.fill_between(t, np.degrees(phi_dot), alpha=0.10, color='#C04828')
ax_rate.set_title('Roll rate φ̇'); ax_rate.set_xlabel('Time [s]')
ax_rate.set_ylabel('φ̇ [°/s]')
idx_peak = np.argmax(np.abs(phi_dot))
ax_rate.annotate(f'  peak {np.degrees(phi_dot[idx_peak]):.1f}°/s',
                 xy=(t[idx_peak], np.degrees(phi_dot[idx_peak])),
                 fontsize=8, color='#C04828', va='center')

# Normal loads
for ax, Fz, col, Fzs, ttl in [
    (ax_fzf, Fzf, '#1D9E75', Fzf_s, 'Front normal load Fzf'),
    (ax_fzr, Fzr, '#534AB7', Fzr_s, 'Rear normal load Fzr'),
]:
    ax.plot(t, Fz, color=col, lw=1.8)
    ax.axhline(Fzs, color='#aaa', lw=0.8, ls='--', label=f'Static {Fzs:.0f} N')
    ax.fill_between(t, Fz, Fzs, where=Fz <  Fzs, color='#E24B4A', alpha=0.18, label='unloaded')
    ax.fill_between(t, Fz, Fzs, where=Fz >= Fzs, color=col,       alpha=0.18, label='loaded')
    ax.set_title(ttl); ax.set_xlabel('Time [s]')
    ax.set_ylabel('Fz [N]'); ax.legend(fontsize=8, frameon=False)

# Lateral forces
for ax, Fy, col, ttl in [
    (ax_fyf, Fyf, '#1D9E75', 'Front lateral force Fyf'),
    (ax_fyr, Fyr, '#534AB7', 'Rear lateral force Fyr'),
]:
    ax.plot(t, Fy, color=col, lw=1.8)
    ax.axhline(0, color='#aaa', lw=0.6, ls=':')
    ax.fill_between(t, Fy, alpha=0.12, color=col)
    ax.set_title(ttl); ax.set_xlabel('Time [s]'); ax.set_ylabel('Fy [N]')

# Utilisation
ax_util.plot(t, util_f*100, color='#1D9E75', lw=1.8, label='Front')
ax_util.plot(t, util_r*100, color='#534AB7', lw=1.8, ls='--', label='Rear')
ax_util.axhline(100, color='#E24B4A', lw=0.9, ls=':', label='Grip limit')
ax_util.set_title('Tyre utilisation |Fy|/(μ·Fz)'); ax_util.set_xlabel('Time [s]')
ax_util.set_ylabel('Utilisation [%]'); ax_util.set_ylim(0, 115)
ax_util.legend(fontsize=9, frameon=False)

# Phase shading
phases = [
    (0.0, 0.7, '#d3d1c7', 'Straight'),
    (0.7, 2.7, '#b5d4f4', 'Left\ncorner'),
    (2.7, 4.0, '#fac775', 'Chicane\ntransition'),
    (4.0, 5.0, '#f5c4b3', 'Right\ncorner'),
    (5.0, 7.0, '#d3d1c7', 'Exit'),
]
ts_axes = [ax_lean, ax_rate, ax_fzf, ax_fzr, ax_fyf, ax_fyr, ax_util]
for t0, t1, col, label in phases:
    for ax in ts_axes:
        ax.axvspan(t0, t1, color=col, alpha=0.22, lw=0)
    yhi = ax_lean.get_ylim()[1]
    ax_lean.text((t0+t1)/2, yhi*0.95, label,
                 ha='center', va='top', fontsize=8, color='#444',
                 multialignment='center')

plt.savefig('/mnt/user-data/outputs/chicane_simulation.png',
            dpi=150, bbox_inches='tight')
print("Saved → /mnt/user-data/outputs/chicane_simulation.png")

print(f"\n── Peak values ──────────────────────────────────────────")
print(f"  Max lean angle    : {np.max(np.abs(lean_deg)):.1f}°"
      f"  (equilibrium: {np.max(np.abs(np.degrees(phi_eq_arr))):.1f}°)")
print(f"  Max roll rate     : {np.max(np.abs(np.degrees(phi_dot))):.1f} °/s"
      f"  at t = {t[np.argmax(np.abs(phi_dot))]:.2f} s")
print(f"  Front Fz range    : {np.min(Fzf):.0f} – {np.max(Fzf):.0f} N"
      f"  (static = {Fzf_s:.0f} N)")
print(f"  Rear  Fz range    : {np.min(Fzr):.0f} – {np.max(Fzr):.0f} N"
      f"  (static = {Fzr_s:.0f} N)")
print(f"  Max |Fyf|         : {np.max(np.abs(Fyf)):.0f} N")
print(f"  Max |Fyr|         : {np.max(np.abs(Fyr)):.0f} N")
print(f"  Peak front util   : {np.max(util_f)*100:.1f} %")
print(f"  Peak rear  util   : {np.max(util_r)*100:.1f} %")
print(f"  Roll-axis angle ε : {np.degrees(epsilon_ra):.2f}°")
M_ra_max = np.max(np.abs(Ix * phi_ddot * np.sin(epsilon_ra)))
print(f"  Max roll-axis M   : {M_ra_max:.1f} N·m  → ΔFz = {M_ra_max/L:.0f} N")

# Verify symmetry: steady-state Fzf in left vs right corner
mask_left  = (t > 1.5) & (t < 2.5)
mask_right = (t > 4.2) & (t < 4.9)
print(f"\n── Left/right symmetry check ────────────────────────────")
print(f"  Fzf in left corner  : {np.mean(Fzf[mask_left]):.0f} N")
print(f"  Fzf in right corner : {np.mean(Fzf[mask_right]):.0f} N  (should match ↑)")
print(f"  Fzr in left corner  : {np.mean(Fzr[mask_left]):.0f} N")
print(f"  Fzr in right corner : {np.mean(Fzr[mask_right]):.0f} N  (should match ↑)")

plt.show()
