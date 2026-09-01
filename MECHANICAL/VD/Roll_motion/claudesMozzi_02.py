"""
Mozzi-Axis + Longitudinal Acceleration: Rear Tire Load Change Analysis
=======================================================================
During corner exit (rapid roll recovery + acceleration), two effects act
simultaneously on the rear tire vertical load:

  1. Roll-rate load shift  (Mozzi-axis, centrifugal)   — always NEGATIVE for rear
  2. Acceleration load transfer (pitch quasi-static)   — POSITIVE for rear under drive,
                                                          NEGATIVE under braking

Their sum determines whether traction is lost.

Physics
-------
  dF_z_roll      = -m * h * phi_dot^2
  lambda_rear    = phi_dot^2 / (phi_dot^2 + (theta_dot * v / L)^2)
  dF_z_rear_roll = lambda_rear * dF_z_roll

  dF_z_accel     = +m * a_x * h / L      (+ = rear gains load under throttle)
                   sign: a_x > 0 = acceleration, a_x < 0 = braking

  dF_z_rear_TOTAL = dF_z_rear_roll + dF_z_accel

Note: under hard acceleration dF_z_accel is positive, partially compensating the
roll-rate loss. The dangerous window is when phi_dot is large AND a_x is still
small — a brief overlap at corner exit before throttle builds.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
#  Physics functions
# ─────────────────────────────────────────────

def calc_dF_z_roll(m, h, phi_dot):
    """Total centrifugal vertical load change due to roll rate [N]. Always <= 0."""
    return -m * h * phi_dot**2

def calc_lambda_rear(phi_dot, theta_dot, v, L):
    """Mozzi-axis fraction of dF_z_roll acting on the rear tire."""
    num   = phi_dot**2
    denom = num + (theta_dot * v / L)**2
    return np.where(denom > 0, num / denom, 0.0)

def calc_dF_z_axle(lambda_rear, dF_z_roll):
    """Split dF_z_roll into rear and front shares."""
    return lambda_rear * dF_z_roll, (1 - lambda_rear) * dF_z_roll

def calc_dF_z_accel(m, h, a_x, L):
    """
    Quasi-static longitudinal load transfer [N].
    a_x > 0: acceleration -> rear gains load (+)
    a_x < 0: braking      -> rear loses load (-)
    """
    return m * a_x * h / L

# ─────────────────────────────────────────────
#  Nominal parameters
# ─────────────────────────────────────────────

DEFAULTS = dict(
    m         = 229,    # kg
    h         = 0.658,  # CoM height [m]
    L         = 1.4,    # wheelbase [m]
    theta_dot = 0.05,   # yaw rate [rad/s]
    v         = 27.78,  # forward speed [m/s]  (~100 km/h)
)

LOAD_LOSS_TARGET = -250   # N — rear load loss that triggers grip loss

# ─────────────────────────────────────────────
#  Sweep ranges
# ─────────────────────────────────────────────

phi_dot_vals   = np.linspace(0, 2.0,  200)
v_vals         = np.linspace(0, 40,   100)
h_vals         = np.linspace(0.3, 1.0, 80)
a_x_vals       = np.linspace(-1, 10,  100)   # m/s^2
theta_dot_vals = np.linspace(0.01, 0.3, 80)

v_lines   = [0, 5, 10, 15, 20, 30, 40]
h_lines   = [0.35, 0.50, 0.658, 0.80, 0.95]
a_x_lines = [-1, 0, 1, 2, 4, 6, 8]   # m/s^2 — throttle ramp at corner exit

corner_exit_cases = [
    dict(v=10,  h=0.55,  label="slow corner, low h"),
    dict(v=15,  h=0.60,  label="slow corner, nominal h"),
    dict(v=20,  h=0.658, label="medium corner, nominal h"),
    dict(v=27,  h=0.658, label="fast corner, nominal h  *"),
    dict(v=27,  h=0.80,  label="fast corner, high h"),
    dict(v=35,  h=0.658, label="very fast corner, nominal h"),
]


def _add_danger_line(ax):
    ax.axhline(LOAD_LOSS_TARGET, color='red', lw=1.2, ls='--', alpha=0.7,
               label=f'Target: {LOAD_LOSS_TARGET} N')


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1 – Line sweeps (roll-rate term only)
# ─────────────────────────────────────────────────────────────────────────────

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9))
fig1.suptitle(
    "Mozzi-Axis Rear Load Loss — Line Sweeps  (roll-rate term only)\n"
    f"Nominal: m={DEFAULTS['m']} kg, h={DEFAULTS['h']} m, "
    f"L={DEFAULTS['L']} m, theta_dot={DEFAULTS['theta_dot']} rad/s, "
    f"v={DEFAULTS['v']:.1f} m/s",
    fontsize=12
)
fig1.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.5, w_pad=3)

cmap_v = plt.cm.plasma
cmap_h = plt.cm.viridis
norm_v = mcolors.Normalize(vmin=min(v_lines), vmax=max(v_lines))
norm_h = mcolors.Normalize(vmin=min(h_lines), vmax=max(h_lines))

# Row 0: vary v
ax = axes1[0, 0]
ax.set_title("dF_z_roll  (independent of v)")
fz = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], phi_dot_vals)
ax.plot(phi_dot_vals, fz, color='steelblue', lw=2, label=f"h={DEFAULTS['h']} m")
_add_danger_line(ax)
ax.set_xlabel("phi_dot  [rad/s]"); ax.set_ylabel("dF_z_roll  [N]")
ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

ax = axes1[0, 1]
ax.set_title("lambda_rear  vs  phi_dot   (param: speed v)")
for v in v_lines:
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], v, DEFAULTS['L'])
    ax.plot(phi_dot_vals, lam, color=cmap_v(norm_v(v)), lw=1.8, label=f"v={v} m/s")
ax.set_xlabel("phi_dot  [rad/s]"); ax.set_ylabel("lambda_rear  [-]")
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

ax = axes1[0, 2]
ax.set_title("dF_z_rear_roll  vs  phi_dot   (param: speed v)")
for v in v_lines:
    fz  = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], phi_dot_vals)
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], v, DEFAULTS['L'])
    fzr, _ = calc_dF_z_axle(lam, fz)
    ax.plot(phi_dot_vals, fzr, color=cmap_v(norm_v(v)), lw=1.8, label=f"v={v} m/s")
_add_danger_line(ax)
ax.set_xlabel("phi_dot  [rad/s]"); ax.set_ylabel("dF_z_rear_roll  [N]")
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.4)

# Row 1: vary h
ax = axes1[1, 0]
ax.set_title("dF_z_roll  vs  phi_dot   (param: CoM height h)")
for h in h_lines:
    fz = calc_dF_z_roll(DEFAULTS['m'], h, phi_dot_vals)
    ax.plot(phi_dot_vals, fz, color=cmap_h(norm_h(h)), lw=1.8, label=f"h={h} m")
_add_danger_line(ax)
ax.set_xlabel("phi_dot  [rad/s]"); ax.set_ylabel("dF_z_roll  [N]")
ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

ax = axes1[1, 1]
ax.set_title("lambda_rear  (h has no effect — geometry only)")
for h in h_lines:
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], DEFAULTS['v'], DEFAULTS['L'])
    ax.plot(phi_dot_vals, lam, color=cmap_h(norm_h(h)), lw=1.8, label=f"h={h} m")
ax.set_xlabel("phi_dot  [rad/s]"); ax.set_ylabel("lambda_rear  [-]")
ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

ax = axes1[1, 2]
ax.set_title("dF_z_rear_roll  vs  phi_dot   (param: CoM height h)")
for h in h_lines:
    fz  = calc_dF_z_roll(DEFAULTS['m'], h, phi_dot_vals)
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], DEFAULTS['v'], DEFAULTS['L'])
    fzr, _ = calc_dF_z_axle(lam, fz)
    ax.plot(phi_dot_vals, fzr, color=cmap_h(norm_h(h)), lw=1.8, label=f"h={h} m")
_add_danger_line(ax)
ax.set_xlabel("phi_dot  [rad/s]"); ax.set_ylabel("dF_z_rear_roll  [N]")
ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

#plt.savefig("/mnt/user-data/outputs/fig1_line_sweeps.png", dpi=150, bbox_inches='tight')
#print("Saved fig1_line_sweeps.png")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2 – Heatmaps: roll-rate term only
# ─────────────────────────────────────────────────────────────────────────────

PHI,  V_  = np.meshgrid(phi_dot_vals, v_vals,         indexing='ij')
PHI2, H_  = np.meshgrid(phi_dot_vals, h_vals,         indexing='ij')
PHI3, TD_ = np.meshgrid(phi_dot_vals, theta_dot_vals, indexing='ij')

FZR_A, _ = calc_dF_z_axle(calc_lambda_rear(PHI,  DEFAULTS['theta_dot'], V_,             DEFAULTS['L']), calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], PHI ))
FZR_B, _ = calc_dF_z_axle(calc_lambda_rear(PHI2, DEFAULTS['theta_dot'], DEFAULTS['v'],  DEFAULTS['L']), calc_dF_z_roll(DEFAULTS['m'], H_,             PHI2))
FZR_C, _ = calc_dF_z_axle(calc_lambda_rear(PHI3, TD_,                   DEFAULTS['v'],  DEFAULTS['L']), calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], PHI3))

vmin_roll = min(FZR_A.min(), FZR_B.min(), FZR_C.min())

def _heatmap(ax, X, Y, Z, xlabel, ylabel, title, vmin, vmax=0):
    im = ax.pcolormesh(X, Y, Z, cmap='RdYlGn_r', vmin=vmin, vmax=vmax, shading='auto')
    try:
        cs = ax.contour(X, Y, Z, levels=[LOAD_LOSS_TARGET],
                        colors='red', linewidths=1.5, linestyles='--')
        ax.clabel(cs, fmt=f'{LOAD_LOSS_TARGET} N', fontsize=8, colors='red')
    except Exception:
        pass
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    return im

fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5))
fig2.suptitle(
    "Mozzi-Axis Rear Load Loss — 2-D Sensitivity Heatmaps  (roll-rate term only)\n"
    f"Red dashed contour = {LOAD_LOSS_TARGET} N",
    fontsize=12
)
fig2.tight_layout(rect=[0, 0, 1, 0.90], w_pad=4)

im2 = _heatmap(axes2[0], PHI,  V_,  FZR_A, "phi_dot  [rad/s]", "v  [m/s]",
               f"dF_z_rear_roll  (h={DEFAULTS['h']} m)", vmin_roll)
_heatmap(axes2[1], PHI2, H_,  FZR_B, "phi_dot  [rad/s]", "h  [m]",
         f"dF_z_rear_roll  (v={DEFAULTS['v']:.1f} m/s)", vmin_roll)
_heatmap(axes2[2], PHI3, TD_, FZR_C, "phi_dot  [rad/s]", "theta_dot  [rad/s]",
         f"dF_z_rear_roll  (v={DEFAULTS['v']:.1f} m/s, h={DEFAULTS['h']} m)", vmin_roll)

fig2.colorbar(im2, ax=axes2.tolist(), fraction=0.015, pad=0.02).set_label("dF_z_rear_roll  [N]")
#plt.savefig("/mnt/user-data/outputs/fig2_heatmaps.png", dpi=150, bbox_inches='tight')
#print("Saved fig2_heatmaps.png")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3 – Corner exit scenario (roll-rate term only)
# ─────────────────────────────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.axhspan(LOAD_LOSS_TARGET - 50, LOAD_LOSS_TARGET + 50, color='red', alpha=0.12, label='+-50 N danger band')
ax3.axhline(LOAD_LOSS_TARGET, color='red', lw=1.5, ls='--', label=f'{LOAD_LOSS_TARGET} N target')
ax3.axhline(0, color='k', lw=0.8, ls=':')

colors3 = plt.cm.tab10(np.linspace(0, 0.9, len(corner_exit_cases)))
for case, color in zip(corner_exit_cases, colors3):
    fz  = calc_dF_z_roll(DEFAULTS['m'], case['h'], phi_dot_vals)
    lam = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], case['v'], DEFAULTS['L'])
    fzr, _ = calc_dF_z_axle(lam, fz)
    ax3.plot(phi_dot_vals, fzr, color=color, lw=2, label=case['label'])

ax3.set_xlabel("phi_dot  (roll rate)  [rad/s]", fontsize=11)
ax3.set_ylabel("dF_z_rear_roll  [N]", fontsize=11)
ax3.set_title("Corner Exit: Rear Load Loss vs Roll Rate  (roll-rate term only)\n"
              f"m={DEFAULTS['m']} kg, L={DEFAULTS['L']} m, theta_dot={DEFAULTS['theta_dot']} rad/s", fontsize=12)
ax3.legend(fontsize=8, loc='lower left')
ax3.grid(True, alpha=0.35)
ax3.set_ylim(bottom=-700)
plt.tight_layout()
#plt.savefig("/mnt/user-data/outputs/fig3_corner_exit_scenario.png", dpi=150, bbox_inches='tight')
#print("Saved fig3_corner_exit_scenario.png")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4 – NEW: phi_dot x a_x heatmap (combined effect)
#
#  The "corner exit operating map":
#    x-axis: roll rate phi_dot  (how fast you pick up the bike)
#    y-axis: longitudinal acceleration a_x  (how hard on the throttle)
#    colour: total rear load change = roll-rate loss + accel gain
#
#  Red −250 N contour = grip-loss boundary.
#  Anything to the right of / below it = rear loses traction.
# ─────────────────────────────────────────────────────────────────────────────

PHI4, AX_ = np.meshgrid(phi_dot_vals, a_x_vals, indexing='ij')

FZ_roll4     = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], PHI4)
LAM4         = calc_lambda_rear(PHI4, DEFAULTS['theta_dot'], DEFAULTS['v'], DEFAULTS['L'])
FZR_roll4, _ = calc_dF_z_axle(LAM4, FZ_roll4)
FZ_accel4    = calc_dF_z_accel(DEFAULTS['m'], DEFAULTS['h'], AX_, DEFAULTS['L'])
FZR_total4   = FZR_roll4 + FZ_accel4

vmin4 = -600
vmax4 = +400

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.suptitle(
    "Combined Rear Load Change: Roll Rate x Longitudinal Acceleration\n"
    f"m={DEFAULTS['m']} kg,  h={DEFAULTS['h']} m,  L={DEFAULTS['L']} m,  "
    f"v={DEFAULTS['v']:.1f} m/s,  theta_dot={DEFAULTS['theta_dot']} rad/s",
    fontsize=12
)
fig4.tight_layout(rect=[0, 0, 1, 0.90], w_pad=4)

def _heatmap_sym(ax, X, Y, Z, xlabel, ylabel, title, vmin, vmax,
                 danger_level=LOAD_LOSS_TARGET, zero_contour=False):
    im = ax.pcolormesh(X, Y, Z, cmap='RdYlGn', vmin=vmin, vmax=vmax, shading='auto')
    try:
        cs = ax.contour(X, Y, Z, levels=[danger_level],
                        colors='red', linewidths=2.0, linestyles='--')
        ax.clabel(cs, fmt=f'{danger_level} N', fontsize=9, colors='red')
    except Exception:
        pass
    if zero_contour:
        try:
            cs0 = ax.contour(X, Y, Z, levels=[0],
                             colors='black', linewidths=1.0, linestyles=':')
            ax.clabel(cs0, fmt='0 N', fontsize=8, colors='black')
        except Exception:
            pass
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    return im

im4 = _heatmap_sym(axes4[0], PHI4, AX_, FZR_total4,
                   "phi_dot  [rad/s]", "a_x  [m/s^2]",
                   "dF_z_rear_TOTAL  =  roll-rate  +  accel",
                   vmin4, vmax4, zero_contour=True)

_heatmap_sym(axes4[1], PHI4, AX_, FZR_roll4,
             "phi_dot  [rad/s]", "a_x  [m/s^2]",
             "dF_z_rear_roll  only  (Mozzi term)",
             vmin4, vmax4)

_heatmap_sym(axes4[2], PHI4, AX_, FZ_accel4,
             "phi_dot  [rad/s]", "a_x  [m/s^2]",
             "dF_z_accel only  (longitudinal transfer)",
             vmin4, vmax4, danger_level=99999)  # no danger contour on this panel

# Mark reference acceleration lines on panel C
for a_ref in [2, 5, 8]:
    axes4[2].axhline(a_ref, color='steelblue', lw=0.8, ls=':', alpha=0.6)
    axes4[2].text(1.92, a_ref + 0.15, f"{a_ref} m/s^2", fontsize=7,
                  color='steelblue', ha='right')

fig4.colorbar(im4, ax=axes4.tolist(), fraction=0.015, pad=0.02).set_label("dF_z_rear  [N]")
#plt.savefig("/mnt/user-data/outputs/fig4_combined_heatmap.png", dpi=150, bbox_inches='tight')
#print("Saved fig4_combined_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5 – NEW: Line plot of TOTAL dF_z_rear vs phi_dot, coloured by a_x
#
#  "Money plot": shows how much throttle compensates a given roll rate,
#  and at what phi_dot you cross the danger line for each throttle level.
# ─────────────────────────────────────────────────────────────────────────────

fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle(
    "Corner Exit Operating Map: Total Rear Load Change vs Roll Rate\n"
    f"m={DEFAULTS['m']} kg,  L={DEFAULTS['L']} m,  theta_dot={DEFAULTS['theta_dot']} rad/s",
    fontsize=12
)
fig5.tight_layout(rect=[0, 0, 1, 0.92], w_pad=4)

norm_a = mcolors.Normalize(vmin=min(a_x_lines), vmax=max(a_x_lines))
cmap_a = plt.cm.RdYlGn

def _plot_combined_lines(ax, h_val, v_val, title):
    ax.axhspan(LOAD_LOSS_TARGET - 50, LOAD_LOSS_TARGET + 50, color='red', alpha=0.10)
    ax.axhline(LOAD_LOSS_TARGET, color='red', lw=1.5, ls='--', label=f'{LOAD_LOSS_TARGET} N danger')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    for a_x in a_x_lines:
        fz_roll      = calc_dF_z_roll(DEFAULTS['m'], h_val, phi_dot_vals)
        lam          = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], v_val, DEFAULTS['L'])
        fzr_roll, _  = calc_dF_z_axle(lam, fz_roll)
        fz_acc       = calc_dF_z_accel(DEFAULTS['m'], h_val, a_x, DEFAULTS['L'])
        total        = fzr_roll + fz_acc
        lw = 2.5 if a_x == 0 else 1.8
        ax.plot(phi_dot_vals, total, color=cmap_a(norm_a(a_x)), lw=lw,
                label=f"a_x = {a_x:+.0f} m/s^2")
    ax.set_xlabel("phi_dot  (roll rate)  [rad/s]", fontsize=11)
    ax.set_ylabel("dF_z_rear_TOTAL  [N]", fontsize=11)
    ax.set_title(title)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.35)
    ax.set_ylim(-700, 500)

_plot_combined_lines(axes5[0], DEFAULTS['h'], DEFAULTS['v'],
                     f"Nominal  (h={DEFAULTS['h']} m, v={DEFAULTS['v']:.0f} m/s)")
_plot_combined_lines(axes5[1], 0.80, DEFAULTS['v'],
                     f"High CoM  (h=0.80 m, v={DEFAULTS['v']:.0f} m/s)  — note steeper curves")

#plt.savefig("/mnt/user-data/outputs/fig5_combined_lines.png", dpi=150, bbox_inches='tight')
#print("Saved fig5_combined_lines.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Console: threshold table including acceleration compensation
# ─────────────────────────────────────────────────────────────────────────────

print("\n── phi_dot threshold for dF_z_rear_TOTAL <= -250 N  (nominal h, v) ──")
print(f"{'a_x  [m/s^2]':>14} {'phi_dot_threshold  [rad/s]':>28}  {'accel compensation  [N]':>26}")
print("-" * 72)
for a_x in a_x_lines:
    fz_roll      = calc_dF_z_roll(DEFAULTS['m'], DEFAULTS['h'], phi_dot_vals)
    lam          = calc_lambda_rear(phi_dot_vals, DEFAULTS['theta_dot'], DEFAULTS['v'], DEFAULTS['L'])
    fzr_roll, _  = calc_dF_z_axle(lam, fz_roll)
    fz_acc       = calc_dF_z_accel(DEFAULTS['m'], DEFAULTS['h'], a_x, DEFAULTS['L'])
    total        = fzr_roll + fz_acc
    crossings    = phi_dot_vals[total <= LOAD_LOSS_TARGET]
    thr = f"{crossings[0]:.3f}" if len(crossings) else "never"
    print(f"{a_x:>14.1f} {thr:>28}  {fz_acc:>+26.1f}")

plt.show()