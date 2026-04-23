"""
Mozzi-Achsen Simulation für Motorrad-Fahrdynamik
=================================================
Simuliert zwei Manöver:
  1. Sinusförmiger Slalom
  2. Kurveneinlenken mit Gegenlenkung (counter-steer)

Berechnet und animiert:
  - Fahrweg des Schwerpunkts
  - Mozzi-Trace (Schnittpunkt der Mozzi-Achse mit der Fahrebene)
  - Drehzentrum (klassischer Kurvenradius)
  - Neigungswinkel Φ der Mozzi-Achse

Koordinatensystem: SAE
  x  vorwärts (Fahrtrichtung)
  y  rechts
  z  nach unten (nicht dargestellt)

Mozzi-Achse (vereinfacht, ohne Trägheitsterme):
  y_mozzi = (ψ̇ · V) / (ψ̇² + φ̇²)    [lateral, im Fahrzeugkoord.]
  Φ       = arctan(ψ̇ / φ̇)             [Neigungswinkel ggü. Fahrebene]
  R       = V / |ψ̇|                    [Kurvenradius / Drehzentrum]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow
import warnings
warnings.filterwarnings("ignore")

# ── Fahrzeugparameter ──────────────────────────────────────────────────────────
V        = 15.0   # Vorwärtsgeschwindigkeit [m/s]
WHEELBASE = 1.4   # Radstand [m]
H_CG     = 0.60   # Schwerpunkthöhe [m]

# ── Simulationsparameter ───────────────────────────────────────────────────────
DT   = 0.01        # Zeitschritt [s]
T_END = 8.0        # Simulationsdauer [s]
t    = np.arange(0, T_END, DT)

# ── Farben (konsistent mit Widget) ────────────────────────────────────────────
C_PATH  = '#378ADD'
C_MOZZI = '#EF9F27'
C_TURN  = '#1D9E75'
C_MOTO  = '#D85A30'
C_PHI   = '#7F77DD'
BG      = '#F4F2ED'
DARK    = '#2C2C2A'


def simulate_slalom(t, V=15.0, freq=0.5, roll_max_deg=30.0):
    """
    Sinusförmiger Slalom.
    Rollrate: φ̇ = φ_max · ω · cos(ω·t)
    Gierrate: ψ̇ = V · tan(φ) / L  (kinematisches Fahrradmodell)
    """
    omega   = 2 * np.pi * freq
    roll_max = np.deg2rad(roll_max_deg)

    phi_dot = roll_max * omega * np.cos(omega * t)   # [rad/s]
    phi     = roll_max * np.sin(omega * t)            # [rad]
    psi_dot = V * np.tan(phi) / WHEELBASE             # [rad/s]

    return phi_dot, psi_dot, phi


def simulate_turn_entry(t, V=15.0, roll_max_deg=30.0):
    """
    Kurveneinlenken mit Gegenlenkung (counter-steer).
    Phase 1: Einlenk-Impuls (negative Gierrate, positive Rollrate)
    Phase 2: Kurvenfahrt (positive Gierrate, Rollrate → 0)
    """
    roll_max = np.deg2rad(roll_max_deg)
    phi_dot  = np.zeros_like(t)
    psi_dot  = np.zeros_like(t)

    t1, t2, t3, t4 = 0.5, 1.5, 2.5, 8.0   # Phasengrenzen [s]

    for i, ti in enumerate(t):
        if ti < t1:
            # Gerade vorher
            phi_dot[i] = 0.0
            psi_dot[i] = 0.0
        elif ti < t2:
            # Gegenlenkung: kurz nach links einlenken (counter-steer)
            s = (ti - t1) / (t2 - t1)
            phi_dot[i] = roll_max * np.pi * np.sin(np.pi * s)
            psi_dot[i] = -roll_max * 0.6 * np.sin(np.pi * s)
        elif ti < t3:
            # Übergang: Rollrate klingt ab, Gierrate dreht auf positiv
            s = (ti - t2) / (t3 - t2)
            phi_dot[i] = roll_max * 0.5 * (1.0 - s) * np.cos(np.pi * s)
            psi_dot[i] = roll_max * 0.8 * s
        else:
            # Stationäre Rechtskurve: Rollrate ≈ 0, konstante Gierrate
            phi_dot[i] = 0.0
            psi_dot[i] = V * np.tan(roll_max * 0.7) / WHEELBASE

    phi = np.cumsum(phi_dot) * DT
    return phi_dot, psi_dot, phi


def compute_trajectory(psi_dot, V, dt):
    """Integriert die Fahrzeugposition aus der Gierrate."""
    N       = len(psi_dot)
    x       = np.zeros(N)
    y_pos   = np.zeros(N)
    heading = np.zeros(N)
    heading[0] = np.pi / 2   # Startrichtung: vorwärts = +y

    for i in range(1, N):
        heading[i] = heading[i-1] + psi_dot[i-1] * dt
        x[i]       = x[i-1] + np.cos(heading[i-1]) * V * dt
        y_pos[i]   = y_pos[i-1] + np.sin(heading[i-1]) * V * dt

    return x, y_pos, heading


def compute_mozzi(phi_dot, psi_dot, V, x, y_pos, heading):
    """
    Berechnet Mozzi-Trace und Drehzentrum in Weltkoordinaten.

    Mozzi-Trace:  Schnittpunkt der Mozzi-Achse mit der Fahrebene
    y_mozzi (Fahrzeugkoord.) = ψ̇·V / (ψ̇² + φ̇²)

    Drehzentrum:  klassischer Kurvenradius R = V/|ψ̇|, senkrecht zur Fahrtrichtung
    """
    denom     = psi_dot**2 + phi_dot**2
    eps       = 1e-6

    # Lateraler Offset in Fahrzeugkoordinaten
    with np.errstate(divide='ignore', invalid='ignore'):
        y_moz_veh = np.where(denom > eps, psi_dot * V / denom, np.nan)
        R_turn    = np.where(np.abs(psi_dot) > eps, V / np.abs(psi_dot), np.nan)

    # Clip auf sinnvollen Bereich (Ausreißer bei ψ̇ ≈ 0 begrenzen)
    y_moz_clamp = np.clip(y_moz_veh, -40, 40)
    R_clamp     = np.clip(R_turn, 0, 40)

    # Senkrechter Versatz: vom Fahrzeug-Heading 90° nach links/rechts
    perp_x = -np.sin(heading)
    perp_y =  np.cos(heading)

    mozzi_x = x + perp_x * y_moz_clamp
    mozzi_y = y_pos + perp_y * y_moz_clamp

    sign_turn = np.sign(psi_dot + eps)
    turn_x    = x + perp_x * sign_turn * R_clamp
    turn_y    = y_pos + perp_y * sign_turn * R_clamp

    # Mozzi ungültig machen wo ψ̇ ≈ 0 (Trace → ∞)
    valid_mozzi = np.abs(y_moz_veh) < 38
    mozzi_x[~valid_mozzi] = np.nan
    mozzi_y[~valid_mozzi] = np.nan

    valid_turn = R_turn < 38
    turn_x[~valid_turn] = np.nan
    turn_y[~valid_turn] = np.nan

    # Neigungswinkel Φ
    with np.errstate(divide='ignore', invalid='ignore'):
        Phi = np.where(np.abs(phi_dot) > eps,
                       np.rad2deg(np.arctan2(psi_dot, phi_dot)),
                       np.sign(psi_dot) * 90.0)

    return mozzi_x, mozzi_y, turn_x, turn_y, Phi, y_moz_veh, R_turn


# ── Beide Manöver simulieren ──────────────────────────────────────────────────
phi_dot_sl, psi_dot_sl, phi_sl = simulate_slalom(t)
x_sl, y_sl, hdg_sl = compute_trajectory(psi_dot_sl, V, DT)
mx_sl, my_sl, tx_sl, ty_sl, Phi_sl, ymv_sl, R_sl = compute_mozzi(
    phi_dot_sl, psi_dot_sl, V, x_sl, y_sl, hdg_sl)

phi_dot_te, psi_dot_te, phi_te = simulate_turn_entry(t)
x_te, y_te, hdg_te = compute_trajectory(psi_dot_te, V, DT)
mx_te, my_te, tx_te, ty_te, Phi_te, ymv_te, R_te = compute_mozzi(
    phi_dot_te, psi_dot_te, V, x_te, y_te, hdg_te)


# ── Hilfsfunktionen für Plot ──────────────────────────────────────────────────
def make_moto_patch(cx, cy, heading, size=0.8):
    """Gibt Koordinaten für ein einfaches Motorrad-Symbol zurück."""
    dx = np.cos(heading) * size
    dy = np.sin(heading) * size
    nx = -np.sin(heading) * size * 0.35
    ny =  np.cos(heading) * size * 0.35
    body = np.array([
        [cx - dx - nx, cy - dy - ny],
        [cx - dx + nx, cy - dy + ny],
        [cx + dx + nx, cy + dy + ny],
        [cx + dx - nx, cy + dy - ny],
    ])
    return body


# ── Animation ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor=BG)
fig.suptitle('Mozzi-Achse: Motorrad-Fahrdynamik', fontsize=14,
             fontweight='bold', color=DARK, y=0.98)

gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.35,
              left=0.07, right=0.97, top=0.93, bottom=0.06)

ax_sl  = fig.add_subplot(gs[0:2, 0])  # Slalom-Draufsicht
ax_te  = fig.add_subplot(gs[0:2, 1])  # Kurveneinlenken-Draufsicht
ax_phi = fig.add_subplot(gs[2, 0])    # Φ(t) für Slalom
ax_rt  = fig.add_subplot(gs[2, 1])    # Raten φ̇ und ψ̇

for ax in [ax_sl, ax_te, ax_phi, ax_rt]:
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=8, colors=DARK)
    for spine in ax.spines.values():
        spine.set_edgecolor('#D3D1C7')
        spine.set_linewidth(0.7)

# Slalom-Achse
ax_sl.set_title('Slalom-Manöver', fontsize=10, color=DARK, pad=4)
ax_sl.set_xlabel('y [m]', fontsize=8, color=DARK)
ax_sl.set_ylabel('x [m]', fontsize=8, color=DARK)
ax_sl.set_aspect('equal')
ax_sl.grid(True, color='#D3D1C7', linewidth=0.4, linestyle='--')

# Kurveneinlenken-Achse
ax_te.set_title('Kurveneinlenken (Gegenlenkung)', fontsize=10, color=DARK, pad=4)
ax_te.set_xlabel('y [m]', fontsize=8, color=DARK)
ax_te.set_ylabel('x [m]', fontsize=8, color=DARK)
ax_te.set_aspect('equal')
ax_te.grid(True, color='#D3D1C7', linewidth=0.4, linestyle='--')

# Φ-Winkel
ax_phi.set_title('Neigungswinkel Φ der Mozzi-Achse', fontsize=10, color=DARK, pad=4)
ax_phi.set_xlabel('t [s]', fontsize=8, color=DARK)
ax_phi.set_ylabel('Φ [°]', fontsize=8, color=DARK)
ax_phi.axhline(0, color='#B4B2A9', linewidth=0.6, linestyle='--')
ax_phi.grid(True, color='#D3D1C7', linewidth=0.4, linestyle='--')

# Raten
ax_rt.set_title('Roll- und Gierrate (Slalom)', fontsize=10, color=DARK, pad=4)
ax_rt.set_xlabel('t [s]', fontsize=8, color=DARK)
ax_rt.set_ylabel('[°/s]  /  [rad/s]', fontsize=8, color=DARK)
ax_rt.axhline(0, color='#B4B2A9', linewidth=0.6, linestyle='--')
ax_rt.grid(True, color='#D3D1C7', linewidth=0.4, linestyle='--')

# Statische Zeitreihen (vollständig, verblasst)
ax_phi.plot(t, Phi_sl, color=C_PHI,   alpha=0.18, linewidth=1.2)
ax_phi.plot(t, Phi_te, color=C_MOTO,  alpha=0.18, linewidth=1.2, linestyle='--')
ax_rt.plot(t, np.rad2deg(phi_dot_sl), color=C_PATH,  alpha=0.18, linewidth=1.2)
ax_rt.plot(t, np.rad2deg(psi_dot_sl), color=C_MOZZI, alpha=0.18, linewidth=1.2)

# Legende (Φ-Plot)
ax_phi.plot([], [], color=C_PHI,  linewidth=1.5, label='Slalom')
ax_phi.plot([], [], color=C_MOTO, linewidth=1.5, linestyle='--', label='Kurveneinlenken')
ax_phi.legend(fontsize=7, loc='upper right', framealpha=0.6)

# Legende (Raten)
ax_rt.plot([], [], color=C_PATH,  linewidth=1.5, label='φ̇ (Rollrate °/s)')
ax_rt.plot([], [], color=C_MOZZI, linewidth=1.5, label='ψ̇ (Gierrate °/s)')
ax_rt.legend(fontsize=7, loc='upper right', framealpha=0.6)

# Legende für Draufsichten
for ax in [ax_sl, ax_te]:
    ax.plot([], [], color=C_PATH,  lw=1.5, label='Fahrweg')
    ax.plot([], [], color=C_MOZZI, lw=1.5, label='Mozzi-Trace')
    ax.plot([], [], color=C_TURN,  lw=1.2, linestyle='--', label='Drehzentrum')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.6)

# Animierte Linien ──────────────────────────────────────────────────────────────
TRAIL = 200   # Anzahl Punkte im "Schweif"

line_sl_path,   = ax_sl.plot([], [], color=C_PATH,  lw=1.4, alpha=0.9)
line_sl_mozzi,  = ax_sl.plot([], [], color=C_MOZZI, lw=1.8, alpha=0.9)
line_sl_turn,   = ax_sl.plot([], [], color=C_TURN,  lw=1.2, alpha=0.7, linestyle='--')
moto_sl         = ax_sl.fill([], [], color=C_MOTO, zorder=5)[0]
dot_sl_mozzi    = ax_sl.plot([], [], 'o', color=C_MOZZI, ms=6, zorder=6)[0]
dot_sl_turn     = ax_sl.plot([], [], 'o', color=C_TURN,  ms=5, zorder=6,
                               markerfacecolor='none', markeredgewidth=1.5)[0]
conn_sl_mozzi   = ax_sl.plot([], [], color=C_MOZZI, lw=0.8, linestyle=':', alpha=0.6)[0]
conn_sl_turn    = ax_sl.plot([], [], color=C_TURN,  lw=0.8, linestyle=':', alpha=0.5)[0]

line_te_path,   = ax_te.plot([], [], color=C_PATH,  lw=1.4, alpha=0.9)
line_te_mozzi,  = ax_te.plot([], [], color=C_MOZZI, lw=1.8, alpha=0.9)
line_te_turn,   = ax_te.plot([], [], color=C_TURN,  lw=1.2, alpha=0.7, linestyle='--')
moto_te         = ax_te.fill([], [], color=C_MOTO, zorder=5)[0]
dot_te_mozzi    = ax_te.plot([], [], 'o', color=C_MOZZI, ms=6, zorder=6)[0]
dot_te_turn     = ax_te.plot([], [], 'o', color=C_TURN,  ms=5, zorder=6,
                               markerfacecolor='none', markeredgewidth=1.5)[0]
conn_te_mozzi   = ax_te.plot([], [], color=C_MOZZI, lw=0.8, linestyle=':', alpha=0.6)[0]
conn_te_turn    = ax_te.plot([], [], color=C_TURN,  lw=0.8, linestyle=':', alpha=0.5)[0]

line_phi_sl,    = ax_phi.plot([], [], color=C_PHI,  lw=1.8)
line_phi_te,    = ax_phi.plot([], [], color=C_MOTO, lw=1.8, linestyle='--')
line_rt_roll,   = ax_rt.plot([], [], color=C_PATH,  lw=1.8)
line_rt_yaw,    = ax_rt.plot([], [], color=C_MOZZI, lw=1.8)

# Zeitmarker
vline_phi = ax_phi.axvline(0, color=DARK, linewidth=0.8, alpha=0.5)
vline_rt  = ax_rt.axvline( 0, color=DARK, linewidth=0.8, alpha=0.5)

# Textfelder für Live-Werte
txt_sl = ax_sl.text(0.02, 0.97, '', transform=ax_sl.transAxes,
                    fontsize=8, va='top', color=DARK,
                    bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec='#D3D1C7', lw=0.5))
txt_te = ax_te.text(0.02, 0.97, '', transform=ax_te.transAxes,
                    fontsize=8, va='top', color=DARK,
                    bbox=dict(boxstyle='round,pad=0.3', fc=BG, ec='#D3D1C7', lw=0.5))

# Achsen-Limits vorberechnen ──────────────────────────────────────────────────
def axis_limits(x, y, margin=0.15):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xr, yr = xmax - xmin, ymax - ymin
    r = max(xr, yr) * (1 + margin) / 2
    xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
    return xc - r, xc + r, yc - r, yc + r

xl_sl = axis_limits(
    np.concatenate([y_sl, my_sl[~np.isnan(my_sl)]]),
    np.concatenate([x_sl, mx_sl[~np.isnan(mx_sl)]]))
xl_te = axis_limits(
    np.concatenate([y_te, my_te[~np.isnan(my_te)]]),
    np.concatenate([x_te, mx_te[~np.isnan(mx_te)]]))

ax_sl.set_xlim(xl_sl[0], xl_sl[1]); ax_sl.set_ylim(xl_sl[2], xl_sl[3])
ax_te.set_xlim(xl_te[0], xl_te[1]); ax_te.set_ylim(xl_te[2], xl_te[3])
ax_phi.set_xlim(0, T_END); ax_phi.set_ylim(
    np.nanmin([Phi_sl, Phi_te]) - 5, np.nanmax([Phi_sl, Phi_te]) + 5)
ax_rt.set_xlim(0, T_END)
ax_rt.set_ylim(min(np.rad2deg(phi_dot_sl).min(), np.rad2deg(psi_dot_sl).min()) - 5,
               max(np.rad2deg(phi_dot_sl).max(), np.rad2deg(psi_dot_sl).max()) + 5)


SKIP = 2   # Jeden 2. Frame animieren (Geschwindigkeit)

def update(frame):
    i = min(frame * SKIP, len(t) - 1)
    s = max(0, i - TRAIL)

    # ── Slalom ──
    line_sl_path.set_data(y_sl[s:i], x_sl[s:i])
    line_sl_mozzi.set_data(my_sl[s:i], mx_sl[s:i])
    line_sl_turn.set_data( ty_sl[s:i], tx_sl[s:i])

    body_sl = make_moto_patch(y_sl[i], x_sl[i], hdg_sl[i] - np.pi/2)
    moto_sl.set_xy(body_sl)

    if not np.isnan(my_sl[i]):
        dot_sl_mozzi.set_data([my_sl[i]], [mx_sl[i]])
        conn_sl_mozzi.set_data([y_sl[i], my_sl[i]], [x_sl[i], mx_sl[i]])
    else:
        dot_sl_mozzi.set_data([], [])
        conn_sl_mozzi.set_data([], [])

    if not np.isnan(ty_sl[i]):
        dot_sl_turn.set_data([ty_sl[i]], [tx_sl[i]])
        conn_sl_turn.set_data([y_sl[i], ty_sl[i]], [x_sl[i], tx_sl[i]])
    else:
        dot_sl_turn.set_data([], [])
        conn_sl_turn.set_data([], [])

    phi_str = f'φ̇ = {np.rad2deg(phi_dot_sl[i]):+.1f}°/s\nψ̇ = {psi_dot_sl[i]:+.3f} rad/s\nΦ = {Phi_sl[i]:+.1f}°'
    txt_sl.set_text(phi_str)

    # ── Kurveneinlenken ──
    line_te_path.set_data(y_te[s:i], x_te[s:i])
    line_te_mozzi.set_data(my_te[s:i], mx_te[s:i])
    line_te_turn.set_data( ty_te[s:i], tx_te[s:i])

    body_te = make_moto_patch(y_te[i], x_te[i], hdg_te[i] - np.pi/2)
    moto_te.set_xy(body_te)

    if not np.isnan(my_te[i]):
        dot_te_mozzi.set_data([my_te[i]], [mx_te[i]])
        conn_te_mozzi.set_data([y_te[i], my_te[i]], [x_te[i], mx_te[i]])
    else:
        dot_te_mozzi.set_data([], [])
        conn_te_mozzi.set_data([], [])

    if not np.isnan(ty_te[i]):
        dot_te_turn.set_data([ty_te[i]], [tx_te[i]])
        conn_te_turn.set_data([y_te[i], ty_te[i]], [x_te[i], tx_te[i]])
    else:
        dot_te_turn.set_data([], [])
        conn_te_turn.set_data([], [])

    phi_te_str = f'φ̇ = {np.rad2deg(phi_dot_te[i]):+.1f}°/s\nψ̇ = {psi_dot_te[i]:+.3f} rad/s\nΦ = {Phi_te[i]:+.1f}°'
    txt_te.set_text(phi_te_str)

    # ── Zeitreihen ──
    line_phi_sl.set_data(t[:i], Phi_sl[:i])
    line_phi_te.set_data(t[:i], Phi_te[:i])
    line_rt_roll.set_data(t[:i], np.rad2deg(phi_dot_sl[:i]))
    line_rt_yaw.set_data( t[:i], np.rad2deg(psi_dot_sl[:i]))
    vline_phi.set_xdata([t[i]])
    vline_rt.set_xdata( [t[i]])

    return (line_sl_path, line_sl_mozzi, line_sl_turn, moto_sl,
            dot_sl_mozzi, dot_sl_turn, conn_sl_mozzi, conn_sl_turn,
            line_te_path, line_te_mozzi, line_te_turn, moto_te,
            dot_te_mozzi, dot_te_turn, conn_te_mozzi, conn_te_turn,
            line_phi_sl, line_phi_te, line_rt_roll, line_rt_yaw,
            vline_phi, vline_rt, txt_sl, txt_te)


N_FRAMES = len(t) // SKIP

ani = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    interval=20,
    blit=True,
    repeat=True
)

plt.savefig('mozzi_preview.png', dpi=130,
            bbox_inches='tight', facecolor=BG)
print("Vorschau gespeichert.")

plt.show()