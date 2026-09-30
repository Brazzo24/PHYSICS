"""
Mozzi-Achse mit Trägheitstensor + Pacejka-Reifenmodell
=======================================================

Physikalisches Modell
---------------------
Zustandsvektor:  s = [x, y, ψ, φ, φ̇, ψ̇]

Rollgleichung  (aus dem Trägheitstensor, SAE-Koordinaten):
  Ix · φ̈  + Ixz · ψ̈  =  m·g·h·φ  −  Fy_f·(h−e)  −  Fy_r·h

Giergleichung:
  Ixz · φ̈  + Iz · ψ̈  =  Fy_f · a  −  Fy_r · b

Gelöst nach [φ̈, ψ̈]:
  det I = Ix·Iz − Ixz²
  φ̈ = (Iz · M_roll + Ixz · M_yaw) / det_I
  ψ̈ = (Ixz · M_roll + Ix  · M_yaw) / det_I

Pacejka Magic Formula (Querkraft aus Schräglaufwinkel):
  Fy(α) = D · sin(C · arctan(B·α − E·(B·α − arctan(B·α))))

Schräglaufwinkel (linearisiert):
  α_f = δ − (V·φ + a·ψ̇) / V
  α_r =   − (V·φ − b·ψ̇) / V

Lenkstabilisierung (PD-Regler auf φ, simuliert Fahrerregelung):
  δ = Kp·φ + Kd·φ̇ + δ_ext(t)

Mozzi-Achse (kinematisch + Trägheitstensor):
  y_mozzi  = ψ̇·V / (ψ̇² + φ̇²)               [lateraler Offset, Fahrzeugkoord.]
  Φ_kin    = arctan(ψ̇ / φ̇)                   [kinematischer Neigungswinkel]
  Φ_I      = arctan([Ix·(p−b) − Ixz·h] /      [Trägheitsbasierter Neigungswinkel]
                    [Iz·h − Ixz·(p−b)])         (p = x-Koord. Vorderrad-Aufstandspunkt)

Integration: klassisches RK4, dt = 0.005 s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
#  FAHRZEUG- UND REIFENPARAMETER
# ══════════════════════════════════════════════════════════════════════
class P:
    # --- Geometrie ---
    L   = 1.40    # Radstand                    [m]
    a   = 0.65    # Abstand CG → Vorderachse    [m]
    b   = 0.75    # Abstand CG → Hinterachse    [m]
    h   = 0.60    # Schwerpunkthöhe              [m]
    e   = 0.04    # Nachlauf                     [m]

    # --- Masse ---
    m   = 220.0   # Gesamtmasse                 [kg]
    g   = 9.81    # Erdbeschleunigung            [m/s²]

    # --- Trägheitstensor (SAE, am Schwerpunkt) ---
    Ix  =  35.0   # Rollträgheit                [kg·m²]
    Iz  =  50.0   # Gierträgheit                [kg·m²]
    Ixz =  -5.0   # Deviationsmoment (negativ!) [kg·m²]

    # --- Pacejka-Koeffizienten ---
    #   Vorderrad (etwas weicher, weniger Last)
    B_f = 10.5;  C_f = 1.35;  D_f = 1900.0;  E_f = 0.10
    #   Hinterrad (höhere Last, mehr Grip)
    B_r = 11.0;  C_r = 1.30;  D_r = 2400.0;  E_r = 0.05

    # --- Fahrgeschwindigkeit ---
    V   = 15.0    # [m/s]  ≈ 54 km/h

    # --- PD-Stabilisierungsregler (modelliert Fahrerinput) ---
    Kp  =  4.0    # Rollwinkel-Verstärkung      [rad/rad]
    Kd  =  1.2    # Rollraten-Verstärkung       [rad·s/rad]


# ══════════════════════════════════════════════════════════════════════
#  PACEJKA MAGIC FORMULA
# ══════════════════════════════════════════════════════════════════════
def pacejka(alpha, B, C, D, E):
    """Querkraft Fy [N] aus Schräglaufwinkel alpha [rad]."""
    Ba = B * alpha
    return D * np.sin(C * np.arctan(Ba - E * (Ba - np.arctan(Ba))))


# ══════════════════════════════════════════════════════════════════════
#  ODE-RECHTE SEITE
# ══════════════════════════════════════════════════════════════════════
def ode(t, s, delta_ext):
    """
    s = [x, y, psi, phi, phid, psid]
    delta_ext: externer Lenkwinkelbefehl (Manöver-Input) [rad]
    """
    x, y, psi, phi, phid, psid = s
    V = P.V

    # Fahrerregler + externer Befehl
    delta = P.Kp * phi + P.Kd * phid + delta_ext

    # Schräglaufwinkel (linearisiert)
    Vy  = V * phi
    af  = delta - (Vy + P.a * psid) / V
    ar  =       - (Vy - P.b * psid) / V

    # Pacejka-Reifenkräfte
    Fy_f = pacejka(af, P.B_f, P.C_f, P.D_f, P.E_f)
    Fy_r = pacejka(ar, P.B_r, P.C_r, P.D_r, P.E_r)

    # Momente
    M_roll = P.m * P.g * P.h * phi - Fy_f * (P.h - P.e) - Fy_r * P.h
    M_yaw  = Fy_f * P.a            - Fy_r * P.b

    # Trägheitstensor auflösen
    det_I  = P.Ix * P.Iz - P.Ixz**2
    phi_dd = (P.Iz * M_roll  + P.Ixz * M_yaw) / det_I
    psi_dd = (P.Ixz * M_roll + P.Ix  * M_yaw) / det_I

    return np.array([V * np.cos(psi),   # ẋ
                     V * np.sin(psi),   # ẏ
                     psid,              # ψ̇
                     phid,              # φ̇
                     phi_dd,            # φ̈
                     psi_dd])           # ψ̈


# ══════════════════════════════════════════════════════════════════════
#  RK4-SIMULATION
# ══════════════════════════════════════════════════════════════════════
def simulate(steer_fn, T=8.0, dt=0.005):
    """
    Integriert das ODE-System mit klassischem RK4.
    Gibt ein Dict mit allen Zeitreihen zurück.
    """
    N   = int(T / dt) + 1
    out = {k: np.zeros(N) for k in
           ['t', 'x', 'y', 'psi', 'phi', 'phid', 'psid',
            'delta', 'af', 'ar', 'Fyf', 'Fyr']}

    s = np.array([0., 0., np.pi / 2, 0., 0., 0.])

    for i in range(N):
        t        = i * dt
        de       = steer_fn(t)
        phi, phid, psid = s[3], s[4], s[5]
        V        = P.V

        # Abgeleitete Größen für Output
        delta    = P.Kp * phi + P.Kd * phid + de
        Vy       = V * phi
        af       = delta - (Vy + P.a * psid) / V
        ar       =       - (Vy - P.b * psid) / V

        out['t'][i]     = t
        out['x'][i]     = s[0];  out['y'][i]     = s[1]
        out['psi'][i]   = s[2];  out['phi'][i]   = phi
        out['phid'][i]  = phid;  out['psid'][i]  = psid
        out['delta'][i] = delta; out['af'][i]    = af
        out['ar'][i]    = ar
        out['Fyf'][i]   = pacejka(af, P.B_f, P.C_f, P.D_f, P.E_f)
        out['Fyr'][i]   = pacejka(ar, P.B_r, P.C_r, P.D_r, P.E_r)

        if i < N - 1:
            de2  = steer_fn(t + 0.5 * dt)
            de4  = steer_fn(t + dt)
            k1   = ode(t,           s,            de)
            k2   = ode(t + 0.5*dt,  s + 0.5*dt*k1, de2)
            k3   = ode(t + 0.5*dt,  s + 0.5*dt*k2, de2)
            k4   = ode(t + dt,      s + dt * k3,    de4)
            s    = s + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

    return out


# ══════════════════════════════════════════════════════════════════════
#  MOZZI-GRÖSSEN
# ══════════════════════════════════════════════════════════════════════
def compute_mozzi(d):
    """
    Berechnet für jeden Zeitschritt:
      - Mozzi-Trace (Weltkoordinaten)
      - Drehzentrum (Weltkoordinaten)
      - Φ_kin  (kinematischer Neigungswinkel)
      - Φ_I    (trägheitsbasierter Neigungswinkel, Formel aus Cossalter)
    """
    ph  = d['phid']; ps = d['psid']
    V   = P.V;       psi = d['psi']
    eps = 1e-8

    # ── Mozzi y-Koordinate (Fahrzeugkoord.) ───────────────────────
    denom  = ph**2 + ps**2
    y_veh  = np.where(denom > eps, ps * V / denom, np.nan)
    y_clip = np.where(np.abs(y_veh) < 35, y_veh, np.nan)

    perp_x = -np.sin(psi);  perp_y = np.cos(psi)
    moz_x  = d['x'] + perp_x * np.nan_to_num(y_clip)
    moz_y  = d['y'] + perp_y * np.nan_to_num(y_clip)
    moz_x[np.isnan(y_clip)] = np.nan
    moz_y[np.isnan(y_clip)] = np.nan

    # ── Drehzentrum ───────────────────────────────────────────────
    R      = np.where(np.abs(ps) > eps, V / np.abs(ps), np.nan)
    R_clip = np.where(R < 35, R, np.nan)
    sgn    = np.sign(ps + eps)
    tc_x   = d['x'] + perp_x * sgn * np.nan_to_num(R_clip)
    tc_y   = d['y'] + perp_y * sgn * np.nan_to_num(R_clip)
    tc_x[np.isnan(R_clip)] = np.nan
    tc_y[np.isnan(R_clip)] = np.nan

    # ── Φ kinematisch ─────────────────────────────────────────────
    Phi_kin = np.where(np.abs(ph) > eps,
                       np.degrees(np.arctan2(ps, ph)),
                       np.sign(ps) * 90.0)

    # ── Φ Trägheitstensor (Formel aus Textauszug, Kap. 8.12) ──────
    # p = x-Koordinate des Vorderrad-Aufstandspunkts im Fahrzeugkoord.-System
    # p − b = a + b + b − b = L   (vereinfacht: frontales Aufstandspunkt-Offset)
    p_front  = P.b + P.L
    num_I    = P.Ix  * (p_front - P.b) - P.Ixz * P.h
    den_I    = P.Iz  * P.h             - P.Ixz  * (p_front - P.b)
    Phi_I    = np.full_like(Phi_kin,
                            np.degrees(np.arctan2(num_I, den_I)))

    return moz_x, moz_y, tc_x, tc_y, Phi_kin, Phi_I


# ══════════════════════════════════════════════════════════════════════
#  LENKEINGABEN
# ══════════════════════════════════════════════════════════════════════
def steer_slalom(t):
    """Sinusförmiger Slalom-Befehl [rad]."""
    return 0.032 * np.sin(2 * np.pi * 0.42 * t)


def steer_turn_entry(t):
    """
    Gegenlenkmanöver:
      Phase 1 (0.4–1.0 s): kurzer negativer Impuls (Gegenlenkung links)
      Phase 2 (1.0–1.9 s): positiver Eintrag (Einlenken rechts)
      Phase 3 (>1.9 s):    langsam abklingend → stationäre Rechtskurve
    """
    if   t < 0.4: return 0.0
    elif t < 1.0: return -0.055 * np.sin(np.pi * (t - 0.4) / 0.6)
    elif t < 1.9: return  0.075 * np.sin(np.pi * (t - 1.0) / 0.9)
    else:         return max(0.010, 0.038 * (1.0 - (t - 1.9) * 0.015))


# ══════════════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════════════
print("Simuliere Slalom ...")
sl = simulate(steer_slalom)

print("Simuliere Kurveneinlenken ...")
te = simulate(steer_turn_entry)

moz_x_sl, moz_y_sl, tc_x_sl, tc_y_sl, Phi_kin_sl, Phi_I_sl = compute_mozzi(sl)
moz_x_te, moz_y_te, tc_x_te, tc_y_te, Phi_kin_te, Phi_I_te = compute_mozzi(te)

t_s = sl['t']
N   = len(t_s)

# Pacejka-Kurven (statisch)
alpha_range = np.linspace(-0.25, 0.25, 400)
Fy_f_curve  = np.array([pacejka(a, P.B_f, P.C_f, P.D_f, P.E_f) for a in alpha_range])
Fy_r_curve  = np.array([pacejka(a, P.B_r, P.C_r, P.D_r, P.E_r) for a in alpha_range])

print(f"φ_Slalom:  {np.degrees(sl['phi']).min():.2f}° … {np.degrees(sl['phi']).max():.2f}°")
print(f"φ_Kurve:   {np.degrees(te['phi']).min():.2f}° … {np.degrees(te['phi']).max():.2f}°")
print(f"Φ_I = {Phi_I_sl[0]:.1f}° (konstant, hängt nur von Ix, Iz, Ixz, h, a, b ab)")


# ══════════════════════════════════════════════════════════════════════
#  PLOT-AUFBAU
# ══════════════════════════════════════════════════════════════════════
BG   = '#F4F2ED'
DARK = '#2C2C2A'
MID  = '#888780'
CP   = '#378ADD'   # Fahrweg
CM   = '#EF9F27'   # Mozzi-Trace
CT   = '#1D9E75'   # Drehzentrum
CMO  = '#D85A30'   # Motorrad / Kurve
CPhi = '#7F77DD'   # Phi kinematisch
CPhI = '#EF9F27'   # Phi Trägheit
CFyf = '#D85A30'   # Vorderrad-Kraft
CFyr = '#1D9E75'   # Hinterrad-Kraft

fig = plt.figure(figsize=(16, 11), facecolor=BG)
fig.suptitle(
    f'Mozzi-Achse  ·  Trägheitstensor (Ix={P.Ix}, Iz={P.Iz}, Ixz={P.Ixz} kg·m²)'
    f'  +  Pacejka Magic Formula',
    fontsize=11.5, fontweight='bold', color=DARK, y=0.988)

gs = GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.38,
              left=0.06, right=0.98, top=0.95, bottom=0.05)

ax_sl   = fig.add_subplot(gs[0:2, 0])
ax_te   = fig.add_subplot(gs[0:2, 1])
ax_pac  = fig.add_subplot(gs[0:2, 2])
ax_phi  = fig.add_subplot(gs[2, 0:2])
ax_rt   = fig.add_subplot(gs[3, 0])
ax_fy   = fig.add_subplot(gs[3, 1])
ax_lean = fig.add_subplot(gs[3, 2])

for ax in fig.get_axes():
    ax.set_facecolor(BG)
    ax.tick_params(labelsize=8, colors=DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor('#D3D1C7'); sp.set_linewidth(0.6)
    ax.grid(True, color='#D3D1C7', linewidth=0.35, linestyle='--', alpha=0.8)

ax_sl.set_aspect('equal');  ax_te.set_aspect('equal')
ax_sl.set_title('Slalom  (Draufsicht)', fontsize=9, color=DARK, pad=3)
ax_te.set_title('Kurveneinlenken — Gegenlenkung  (Draufsicht)', fontsize=9, color=DARK, pad=3)
ax_pac.set_title('Pacejka Magic Formula  +  Betriebspunkte', fontsize=9, color=DARK, pad=3)
ax_phi.set_title('Neigungswinkel Φ der Mozzi-Achse — kinematisch vs. Trägheitstensor',
                 fontsize=9, color=DARK, pad=3)
ax_rt.set_title('Roll- u. Gierrate  (Slalom)', fontsize=9, color=DARK, pad=3)
ax_fy.set_title('Reifenquerkräfte Fy  (Slalom)', fontsize=9, color=DARK, pad=3)
ax_lean.set_title('Rollwinkel φ', fontsize=9, color=DARK, pad=3)

for ax, xl, yl in [
    (ax_sl,   'y [m]',  'x [m]'),
    (ax_te,   'y [m]',  'x [m]'),
    (ax_pac,  'α [°]',  'Fy [N]'),
    (ax_phi,  't [s]',  'Φ [°]'),
    (ax_rt,   't [s]',  '[°/s]'),
    (ax_fy,   't [s]',  'Fy [N]'),
    (ax_lean, 't [s]',  'φ [°]'),
]:
    ax.set_xlabel(xl, fontsize=8, color=MID)
    ax.set_ylabel(yl, fontsize=8, color=MID)

# ── Statische Inhalte ─────────────────────────────────────────────────
ax_pac.plot(np.degrees(alpha_range), Fy_f_curve,
            color=CFyf, lw=1.8, label='Vorderrad')
ax_pac.plot(np.degrees(alpha_range), Fy_r_curve,
            color=CFyr, lw=1.8, ls='--', label='Hinterrad')
ax_pac.axhline(0, color=MID, lw=0.5); ax_pac.axvline(0, color=MID, lw=0.5)
ax_pac.legend(fontsize=7, framealpha=0.7)

for ax in [ax_phi, ax_rt, ax_fy, ax_lean]:
    ax.axhline(0, color=MID, lw=0.4)

# ── Legenden ──────────────────────────────────────────────────────────
for ax in [ax_sl, ax_te]:
    for col, lb in [(CP, 'Fahrweg'), (CM, 'Mozzi-Trace'), (CT, 'Drehzentrum')]:
        ax.plot([], [], color=col, lw=1.5, label=lb)
    ax.legend(fontsize=7, framealpha=0.65, loc='upper right')

ax_phi.plot([], [], color=CPhi, lw=1.4, label='Φ_kin  Slalom')
ax_phi.plot([], [], color=CPhI, lw=1.4, ls='--',
            label=f'Φ_I  Slalom  (Ixz={P.Ixz} → Φ_I≈{Phi_I_sl[0]:.1f}°)')
ax_phi.plot([], [], color=CMO,  lw=1.4, label='Φ_kin  Kurveneinlenken')
ax_phi.plot([], [], color=CT,   lw=1.4, ls='--', label='Φ_I  Kurveneinlenken')
ax_phi.legend(fontsize=7, framealpha=0.7, loc='upper right', ncol=2)

ax_rt.plot([], [], color=CP, lw=1.4, label='φ̇  [°/s]')
ax_rt.plot([], [], color=CM, lw=1.4, ls='--', label='ψ̇  [°/s]')
ax_rt.legend(fontsize=7, framealpha=0.7)

ax_fy.plot([], [], color=CFyf, lw=1.4, label='Fy_f  Vorderrad')
ax_fy.plot([], [], color=CFyr, lw=1.4, ls='--', label='Fy_r  Hinterrad')
ax_fy.legend(fontsize=7, framealpha=0.7)

ax_lean.plot([], [], color=CP,  lw=1.4, label='Slalom')
ax_lean.plot([], [], color=CMO, lw=1.4, ls='--', label='Kurveneinlenken')
ax_lean.legend(fontsize=7, framealpha=0.7)

# ── Achslimits ────────────────────────────────────────────────────────
def sq_lim(xs, ys, mar=0.18):
    xa = np.concatenate([v[~np.isnan(v)] for v in xs])
    ya = np.concatenate([v[~np.isnan(v)] for v in ys])
    cx = (xa.min() + xa.max()) / 2;  rx = (xa.max() - xa.min()) / 2
    cy = (ya.min() + ya.max()) / 2;  ry = (ya.max() - ya.min()) / 2
    r  = max(rx, ry) * (1 + mar)
    return cx-r, cx+r, cy-r, cy+r

xl1 = sq_lim([sl['y'], moz_y_sl], [sl['x'], moz_x_sl])
xl2 = sq_lim([te['y'], moz_y_te], [te['x'], moz_x_te])
ax_sl.set_xlim(xl1[0], xl1[1]); ax_sl.set_ylim(xl1[2], xl1[3])
ax_te.set_xlim(xl2[0], xl2[1]); ax_te.set_ylim(xl2[2], xl2[3])

ax_phi.set_xlim(0, 8)
all_phi = np.concatenate([Phi_kin_sl, Phi_I_sl, Phi_kin_te, Phi_I_te])
ax_phi.set_ylim(np.nanpercentile(all_phi, 1) - 3,
                np.nanpercentile(all_phi, 99) + 3)

ax_rt.set_xlim(0, 8)
rd = np.concatenate([np.degrees(sl['phid']), np.degrees(sl['psid'])])
ax_rt.set_ylim(rd.min() - 2, rd.max() + 2)

ax_fy.set_xlim(0, 8)
fy_all = np.concatenate([sl['Fyf'], sl['Fyr']])
ax_fy.set_ylim(fy_all.min() - 30, fy_all.max() + 30)

ax_lean.set_xlim(0, 8)
lean_all = np.concatenate([np.degrees(sl['phi']), np.degrees(te['phi'])])
ax_lean.set_ylim(lean_all.min() - 0.2, lean_all.max() + 0.2)


# ══════════════════════════════════════════════════════════════════════
#  ANIMIERTE OBJEKTE
# ══════════════════════════════════════════════════════════════════════
TRAIL = 220

def make_map_objects(ax):
    lpath,  = ax.plot([], [], color=CP,  lw=1.3, alpha=0.9)
    lmozzi, = ax.plot([], [], color=CM,  lw=1.8, alpha=0.9)
    lturn,  = ax.plot([], [], color=CT,  lw=1.0, alpha=0.7, ls='--')
    moto    = ax.fill([], [], color=CMO, zorder=5)[0]
    dot_m,  = ax.plot([], [], 'o', color=CM,  ms=5.5, zorder=6)
    dot_t,  = ax.plot([], [], 'o', color=CT,  ms=4.5, zorder=6,
                      mfc='none', mew=1.4)
    conn_m, = ax.plot([], [], color=CM, lw=0.7, ls=':', alpha=0.5)
    conn_t, = ax.plot([], [], color=CT, lw=0.7, ls=':', alpha=0.4)
    txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                  fontsize=7.5, va='top', color=DARK, family='monospace',
                  bbox=dict(boxstyle='round,pad=0.3', fc=BG,
                            ec='#D3D1C7', lw=0.4))
    return lpath, lmozzi, lturn, moto, dot_m, dot_t, conn_m, conn_t, txt

sl_objs = make_map_objects(ax_sl)
te_objs = make_map_objects(ax_te)

dot_pac_f, = ax_pac.plot([], [], 'o', color=CFyf, ms=7, zorder=6)
dot_pac_r, = ax_pac.plot([], [], 's', color=CFyr, ms=6, zorder=6)
txt_pac = ax_pac.text(0.03, 0.97, '', transform=ax_pac.transAxes,
                      fontsize=7.5, va='top', color=DARK,
                      bbox=dict(boxstyle='round,pad=0.3', fc=BG,
                                ec='#D3D1C7', lw=0.4))

lph_ks, = ax_phi.plot([], [], color=CPhi, lw=1.6)
lph_is, = ax_phi.plot([], [], color=CPhI, lw=1.6, ls='--')
lph_kt, = ax_phi.plot([], [], color=CMO,  lw=1.6)
lph_it, = ax_phi.plot([], [], color=CT,   lw=1.6, ls='--')
l_roll, = ax_rt.plot([], [], color=CP,   lw=1.6)
l_yaw,  = ax_rt.plot([], [], color=CM,   lw=1.6, ls='--')
l_fyf,  = ax_fy.plot([], [], color=CFyf, lw=1.6)
l_fyr,  = ax_fy.plot([], [], color=CFyr, lw=1.6, ls='--')
l_lsl,  = ax_lean.plot([], [], color=CP,  lw=1.6)
l_lte,  = ax_lean.plot([], [], color=CMO, lw=1.6, ls='--')
vlines  = [ax.axvline(0, color=DARK, lw=0.7, alpha=0.4)
           for ax in [ax_phi, ax_rt, ax_fy, ax_lean]]


def moto_polygon(cx, cy, hdg, sz=0.5):
    """Einfaches Motorrad-Rechteck in Weltkoordinaten."""
    dx = np.cos(hdg) * sz;    dy = np.sin(hdg) * sz
    nx = -np.sin(hdg) * sz * 0.30
    ny =  np.cos(hdg) * sz * 0.30
    return np.array([
        [cx - dx - nx, cy - dy - ny],
        [cx - dx + nx, cy - dy + ny],
        [cx + dx + nx, cy + dy + ny],
        [cx + dx - nx, cy + dy - ny],
    ])


def update_map(objs, i, d, mox, moy, tcx, tcy, label):
    lp, lm, lt, mo, dm, dt_, cm_, ct_, txt_ = objs
    s = max(0, i - TRAIL)

    lp.set_data(d['y'][s:i], d['x'][s:i])

    def filt(a_arr, b_arr):
        aw = [a_arr[j] for j in range(s, i) if not np.isnan(a_arr[j])]
        bw = [b_arr[j] for j in range(s, i) if not np.isnan(b_arr[j])]
        return aw, bw

    lm.set_data(*filt(moy, mox))
    lt.set_data(*filt(tcy, tcx))
    mo.set_xy(moto_polygon(d['y'][i], d['x'][i], d['psi'][i] - np.pi / 2))

    if not np.isnan(mox[i]):
        dm.set_data([moy[i]], [mox[i]])
        cm_.set_data([d['y'][i], moy[i]], [d['x'][i], mox[i]])
    else:
        dm.set_data([], []);  cm_.set_data([], [])

    if not np.isnan(tcx[i]):
        dt_.set_data([tcy[i]], [tcx[i]])
        ct_.set_data([d['y'][i], tcy[i]], [d['x'][i], tcx[i]])
    else:
        dt_.set_data([], []);  ct_.set_data([], [])

    txt_.set_text(
        f'{label}\n'
        f'φ   = {np.degrees(d["phi"][i]):+5.2f}°\n'
        f'φ̇   = {np.degrees(d["phid"][i]):+5.1f}°/s\n'
        f'ψ̇   = {np.degrees(d["psid"][i]):+5.2f}°/s\n'
        f'δ   = {np.degrees(d["delta"][i]):+5.2f}°\n'
        f'αf  = {np.degrees(d["af"][i]):+5.2f}°\n'
        f'Fyf = {d["Fyf"][i]:+.0f} N'
    )


SKIP = 2   # Jeden 2. Frame (→ flüssig bei 60-fps-Daten)

def update(frame):
    i = min(frame * SKIP, N - 1)

    update_map(sl_objs, i, sl, moz_x_sl, moz_y_sl, tc_x_sl, tc_y_sl, 'Slalom')
    update_map(te_objs, i, te, moz_x_te, moz_y_te, tc_x_te, tc_y_te, 'Kurveneinlenken')

    # Pacejka-Betriebspunkt
    dot_pac_f.set_data([np.degrees(sl['af'][i])],  [sl['Fyf'][i]])
    dot_pac_r.set_data([np.degrees(sl['ar'][i])],  [sl['Fyr'][i]])
    txt_pac.set_text(
        f'αf = {np.degrees(sl["af"][i]):+.2f}°\n'
        f'Fyf= {sl["Fyf"][i]:+.0f} N\n'
        f'αr = {np.degrees(sl["ar"][i]):+.2f}°\n'
        f'Fyr= {sl["Fyr"][i]:+.0f} N'
    )

    # Zeitreihen
    lph_ks.set_data(t_s[:i], Phi_kin_sl[:i])
    lph_is.set_data(t_s[:i], Phi_I_sl[:i])
    lph_kt.set_data(t_s[:i], Phi_kin_te[:i])
    lph_it.set_data(t_s[:i], Phi_I_te[:i])
    l_roll.set_data(t_s[:i], np.degrees(sl['phid'][:i]))
    l_yaw.set_data( t_s[:i], np.degrees(sl['psid'][:i]))
    l_fyf.set_data( t_s[:i], sl['Fyf'][:i])
    l_fyr.set_data( t_s[:i], sl['Fyr'][:i])
    l_lsl.set_data( t_s[:i], np.degrees(sl['phi'][:i]))
    l_lte.set_data( t_s[:i], np.degrees(te['phi'][:i]))

    for vl in vlines:
        vl.set_xdata([t_s[i]])

    return (*sl_objs, *te_objs,
            dot_pac_f, dot_pac_r, txt_pac,
            lph_ks, lph_is, lph_kt, lph_it,
            l_roll, l_yaw, l_fyf, l_fyr, l_lsl, l_lte,
            *vlines)


ani = animation.FuncAnimation(
    fig, update,
    frames=N // SKIP,
    interval=18,      # ms zwischen Frames
    blit=True,
    repeat=True
)

plt.show()
# Optional: als GIF speichern (dauert ~1-2 Minuten):
# ani.save('mozzi_pacejka.gif', writer=animation.PillowWriter(fps=30), dpi=90)
