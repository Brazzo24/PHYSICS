"""
Mozzi-Axis Rear Tire Load Change Analysis
==========================================
During corner exit (rapid roll recovery + acceleration), the rear tire can lose
200–300 N of vertical load briefly — enough to break traction and kill drive.

This script analyses which parameters drive dF_z_rear and how sensitive it is,
using two complementary views:
  1. Line plots  – sweep phi_dot, compare one secondary parameter (v or h)
  2. Heatmaps    – full 2-D grid: phi_dot × v  and  phi_dot × h

Physics recap
-------------
  dF_z_roll   = -m * h * phi_dot²          (centrifugal load shift from CoM height)
  lambda_rear = phi_dot² / (phi_dot² + (theta_dot * v / L)²)
                                             (Mozzi-axis fraction going to rear)
  dF_z_rear   = lambda_rear * dF_z_roll     (rear share of the roll-rate load loss)
  dF_z_front  = (1 - lambda_rear) * dF_z_roll
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
#  Physics functions
# ─────────────────────────────────────────────

def calc_dF_z_roll(m, h, phi_dot):
    """Total vertical load change due to roll rate [N].  Always ≤ 0."""
    return -m * h * phi_dot**2

def calc_lambda_rear(phi_dot, theta_dot, v, L):
    """
    Fraction of dF_z_roll acting on the rear tire.
    Derived from the Mozzi / instantaneous screw axis geometry.
    phi_dot  – roll rate  [rad/s]
    theta_dot – yaw rate  [rad/s]   (constant default: 0.05 rad/s)
    v         – forward speed [m/s]
    L         – wheelbase [m]
    """
    num   = phi_dot**2
    denom = num + (theta_dot * v / L)**2
    # Guard against phi_dot = 0 → lambda = 0 (no roll, no load shift)
    return np.where(denom > 0, num / denom, 0.0)

def calc_dF_z_axle(lambda_rear, dF_z_roll):
    """Returns (dF_z_rear, dF_z_front) tuple."""
    dF_z_rear  = lambda_rear       * dF_z_roll
    dF_z_front = (1 - lambda_rear) * dF_z_roll
    return dF_z_rear, dF_z_front

# ─────────────────────────────────────────────
#  Default / nominal parameters
# ─────────────────────────────────────────────

DEFAULTS = dict(
    m         = 229,    # total mass bike + rider [kg]
    h         = 0.658,  # CoM height [m]
    L         = 1.4,    # wheelbase [m]
    theta_dot = 0.05,   # yaw rate [rad/s]  (typical MotoGP corner exit)
    v         = 27.78,  # forward speed [m/s]  (~100 km/h)
)

# Danger zone reference (from chicane simulation)
LOAD_LOSS_TARGET = -250  # N  — the 200–300 N rear load loss we want to explain

# ─────────────────────────────────────────────
#  Sweep ranges
# ─────────────────────────────────────────────

phi_dot_vals = np.linspace(0, 2.0, 200)   # roll rate  [rad/s]  — 2 rad/s is aggressive
v_vals       = np.linspace(0, 40,  100)   # speed      [m/s]
h_vals       = np.linspace(0.3, 1.0, 80)  # CoM height [m]

v_lines      = [0, 5, 10, 15, 20, 30, 40]   # discrete v for line plots
h_lines      = [0.35, 0.50, 0.658, 0.80, 0.95]  # discrete h for line plots


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 – Line plots: sweep phi_dot, colour = velocity or CoM height
# ─────────────────────────────────────────────────────────────────────────────

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9))
fig1.suptitle(
    "Mozzi-Axis Rear Load Loss — Line Sweeps\n"
    f"Nominal: m={DEFAULTS['m']} kg, h={DEFAULTS['h']} m, "
    f"L={DEFAULTS['L']} m, θ̇={DEFAULTS['theta_dot']} rad/s, "
    f"v={DEFAULTS['v']:.1f} m/s",
    fontsize=12
)
fig1.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.5, w_pad=3)

cmap_v = plt.cm.plasma
cmap_h = plt.cm.viridis


def _add_danger_line(ax):
    ax.axhline(LOAD_LOSS_TARGET, color='red', lw=1.2, ls='--', alpha=0.7,
               label=f'Target: {LOAD_LOSS_TARGET} N')


# ── Row 0: vary velocity ──────────────────────────────────────────────────────

ax = axes1[0, 0]
ax.set_title("dF_z_roll  (independent of v)")
for v in v_lines:
    fz = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], phi_dot_vals)
    ax.plot(phi_dot_vals, fz, color='steelblue', lw=1.5)   # same curve — show once visibly
    break  # only one curve; loop kept for structural symmetry
ax.plot(phi_dot_vals, fz, color='steelblue', lw=2, label=f"h={DEFAULTS['h']} m")
_add_danger_line(ax)
ax.set_xlabel("φ̇  [rad/s]")
ax.set_ylabel("dF_z_roll  [N]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

ax = axes1[0, 1]
ax.set_title("λ_rear  vs  φ̇   (param: speed v)")
norm_v = mcolors.Normalize(vmin=min(v_lines), vmax=max(v_lines))
for v in v_lines:
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], v, DEFAULTS['L'])
    ax.plot(phi_dot_vals, lam, color=cmap_v(norm_v(v)), lw=1.8, label=f"v={v} m/s")
ax.set_xlabel("φ̇  [rad/s]")
ax.set_ylabel("λ_rear  [–]")
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

ax = axes1[0, 2]
ax.set_title("dF_z_rear  vs  φ̇   (param: speed v)")
for v in v_lines:
    fz   = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], phi_dot_vals)
    lam  = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], v, DEFAULTS['L'])
    fzr, _ = calc_dF_z_axle(lam, fz)
    ax.plot(phi_dot_vals, fzr, color=cmap_v(norm_v(v)), lw=1.8, label=f"v={v} m/s")
_add_danger_line(ax)
ax.set_xlabel("φ̇  [rad/s]")
ax.set_ylabel("dF_z_rear  [N]")
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

# ── Row 1: vary CoM height ────────────────────────────────────────────────────

ax = axes1[1, 0]
ax.set_title("dF_z_roll  vs  φ̇   (param: CoM height h)")
norm_h = mcolors.Normalize(vmin=min(h_lines), vmax=max(h_lines))
for h in h_lines:
    fz = calc_dF_z_roll(DEFAULTS['m'], h, phi_dot_vals)
    ax.plot(phi_dot_vals, fz, color=cmap_h(norm_h(h)), lw=1.8, label=f"h={h} m")
_add_danger_line(ax)
ax.set_xlabel("φ̇  [rad/s]")
ax.set_ylabel("dF_z_roll  [N]")
ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

ax = axes1[1, 1]
ax.set_title("λ_rear  vs  φ̇   (param: CoM height h)")
for h in h_lines:
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], DEFAULTS['v'], DEFAULTS['L'])
    ax.plot(phi_dot_vals, lam, color=cmap_h(norm_h(h)), lw=1.8, label=f"h={h} m")
ax.set_xlabel("φ̇  [rad/s]")
ax.set_ylabel("λ_rear  [–]")
ax.set_title("λ_rear  (h has no effect — geometry only)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

ax = axes1[1, 2]
ax.set_title("dF_z_rear  vs  φ̇   (param: CoM height h)")
for h in h_lines:
    fz   = calc_dF_z_roll(DEFAULTS['m'], h, phi_dot_vals)
    lam  = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], DEFAULTS['v'], DEFAULTS['L'])
    fzr, _ = calc_dF_z_axle(lam, fz)
    ax.plot(phi_dot_vals, fzr, color=cmap_h(norm_h(h)), lw=1.8, label=f"h={h} m")
_add_danger_line(ax)
ax.set_xlabel("φ̇  [rad/s]")
ax.set_ylabel("dF_z_rear  [N]")
ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

# plt.savefig("/mnt/user-data/outputs/fig1_line_sweeps.png", dpi=150, bbox_inches='tight')
# print("Saved fig1_line_sweeps.png")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 – Heatmaps: dF_z_rear on 2-D grids
#  Grid A: phi_dot  ×  v          (h = nominal)
#  Grid B: phi_dot  ×  h          (v = nominal)
#  Grid C: phi_dot  ×  theta_dot  (v, h = nominal)  ← new: yaw-rate sensitivity
# ─────────────────────────────────────────────────────────────────────────────

theta_dot_vals = np.linspace(0.01, 0.3, 80)   # yaw rate [rad/s]

PHI, V   = np.meshgrid(phi_dot_vals, v_vals,        indexing='ij')
PHI2, H  = np.meshgrid(phi_dot_vals, h_vals,        indexing='ij')
PHI3, TD = np.meshgrid(phi_dot_vals, theta_dot_vals, indexing='ij')

# Grid A
FZ_A   = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], PHI)
LAM_A  = calc_lambda_rear(PHI, DEFAULTS['theta_dot'], V, DEFAULTS['L'])
FZR_A, _ = calc_dF_z_axle(LAM_A, FZ_A)

# Grid B
FZ_B   = calc_dF_z_roll(DEFAULTS['m'], H, PHI2)
LAM_B  = calc_lambda_rear(PHI2, DEFAULTS['theta_dot'], DEFAULTS['v'], DEFAULTS['L'])
FZR_B, _ = calc_dF_z_axle(LAM_B, FZ_B)

# Grid C
FZ_C   = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], PHI3)
LAM_C  = calc_lambda_rear(PHI3, TD, DEFAULTS['v'], DEFAULTS['L'])
FZR_C, _ = calc_dF_z_axle(LAM_C, FZ_C)

# Shared colour scale so all three heatmaps are comparable
vmin = min(FZR_A.min(), FZR_B.min(), FZR_C.min())
vmax = 0   # load change is always ≤ 0, so anchor upper bound at 0

fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
fig2.suptitle(
    "Mozzi-Axis Rear Load Loss — 2-D Sensitivity Heatmaps\n"
    f"Red dashed contour = {LOAD_LOSS_TARGET} N  (corner-exit danger zone)",
    fontsize=12
)
fig2.tight_layout(rect=[0, 0, 1, 0.90], w_pad=4)

cmap_heat = 'RdYlGn'   # green = small loss, red = big loss (reversed below)

def _heatmap(ax, X, Y, Z, xlabel, ylabel, title):
    im = ax.pcolormesh(X, Y, Z, cmap=cmap_heat + '_r',
                       vmin=vmin, vmax=vmax, shading='auto')
    cs = ax.contour(X, Y, Z, levels=[LOAD_LOSS_TARGET],
                    colors='red', linewidths=1.5, linestyles='--')
    ax.clabel(cs, fmt=f'{LOAD_LOSS_TARGET} N', fontsize=8, colors='red')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im

im = _heatmap(axes2[0], PHI,  V,  FZR_A,
              "φ̇  [rad/s]", "v  [m/s]",
              f"dF_z_rear   (h={DEFAULTS['h']} m, θ̇={DEFAULTS['theta_dot']} rad/s)")

_heatmap(axes2[1], PHI2, H,  FZR_B,
         "φ̇  [rad/s]", "h  [m]",
         f"dF_z_rear   (v={DEFAULTS['v']:.1f} m/s, θ̇={DEFAULTS['theta_dot']} rad/s)")

_heatmap(axes2[2], PHI3, TD, FZR_C,
         "φ̇  [rad/s]", "θ̇  [rad/s]",
         f"dF_z_rear   (v={DEFAULTS['v']:.1f} m/s, h={DEFAULTS['h']} m)")

# Shared colourbar
cbar = fig2.colorbar(im, ax=axes2.tolist(), fraction=0.015, pad=0.02)
cbar.set_label("dF_z_rear  [N]")

# plt.savefig("/mnt/user-data/outputs/fig2_heatmaps.png", dpi=150, bbox_inches='tight')
# print("Saved fig2_heatmaps.png")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 – "Corner exit scenario" — what combination gets us to -250 N?
#  Show dF_z_rear vs phi_dot for a realistic range of (v, h) combinations,
#  with the target band shaded.
# ─────────────────────────────────────────────────────────────────────────────

corner_exit_cases = [
    dict(v=10, h=0.55, label="slow corner, low h"),
    dict(v=15, h=0.60, label="slow corner, nominal h"),
    dict(v=20, h=0.658, label="medium corner, nominal h"),
    dict(v=27, h=0.658, label="fast corner, nominal h  ★"),
    dict(v=27, h=0.80,  label="fast corner, high h"),
    dict(v=35, h=0.658, label="very fast corner, nominal h"),
]

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.axhspan(LOAD_LOSS_TARGET - 50, LOAD_LOSS_TARGET + 50,
            color='red', alpha=0.12, label='±50 N danger band')
ax3.axhline(LOAD_LOSS_TARGET, color='red', lw=1.5, ls='--', label=f'{LOAD_LOSS_TARGET} N target')
ax3.axhline(0, color='k', lw=0.8, ls=':')

colors = plt.cm.tab10(np.linspace(0, 0.9, len(corner_exit_cases)))
for case, color in zip(corner_exit_cases, colors):
    fz  = calc_dF_z_roll(DEFAULTS['m'], case['h'], phi_dot_vals)
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], case['v'], DEFAULTS['L'])
    fzr, _ = calc_dF_z_axle(lam, fz)
    ax3.plot(phi_dot_vals, fzr, color=color, lw=2, label=case['label'])

ax3.set_xlabel("φ̇  (roll rate)  [rad/s]", fontsize=11)
ax3.set_ylabel("dF_z_rear  [N]", fontsize=11)
ax3.set_title(
    "Corner Exit: Rear Tire Load Loss vs Roll Rate\n"
    f"m={DEFAULTS['m']} kg, L={DEFAULTS['L']} m, θ̇={DEFAULTS['theta_dot']} rad/s",
    fontsize=12
)
ax3.legend(fontsize=8, loc='lower left')
ax3.grid(True, alpha=0.35)
ax3.set_ylim(bottom=-700)

plt.tight_layout()
# plt.savefig("/mnt/user-data/outputs/fig3_corner_exit_scenario.png", dpi=150, bbox_inches='tight')
# print("Saved fig3_corner_exit_scenario.png")

# ─────────────────────────────────────────────────────────────────────────────
#  Console summary: at what phi_dot does each case cross the -250 N threshold?
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Corner-exit threshold crossings (dF_z_rear ≤ −250 N) ──")
print(f"{'Case':<40} {'φ̇_threshold [rad/s]':>22}")
print("─" * 64)
for case in corner_exit_cases:
    fz  = calc_dF_z_roll(DEFAULTS['m'], case['h'], phi_dot_vals)
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], case['v'], DEFAULTS['L'])
    fzr, _ = calc_dF_z_axle(lam, fz)
    crossings = phi_dot_vals[fzr <= LOAD_LOSS_TARGET]
    if len(crossings):
        print(f"{case['label']:<40} {crossings[0]:>22.3f}")
    else:
        print(f"{case['label']:<40} {'never crossed':>22}")

plt.show()