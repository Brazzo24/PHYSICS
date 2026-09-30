"""
Corner Exit Slip Analysis
=========================
Simulates the causal chain that causes rear tyre slip loss during
aggressive corner exit from high lean angle.

Scenario: apex exit at 60° lean, 70 km/h, full throttle application.
The rider applies a high roll rate to stand the bike up quickly.

Three coupled mechanisms are modelled:

  ① Inertial Fz unloading (dominant, fastest):
       ΔFz_r = −(Ix·φ̈ + Ixz·ψ̈) / b
     The roll acceleration φ̈ leads φ̇ by ~90°, so the load drop
     arrives BEFORE the peak roll rate.

  ② Mozzi axis lateral contact patch velocity:
       y_mozzi = ψ̇·V / (ψ̇² + φ̇²)
       v_lat   = φ̇ · y_mozzi
       α_r     = arctan(v_lat / V)
     Produces a lateral slip angle at the rear contact patch.

  ③ Tyre relaxation lag:
       dα_actual/ds = (α_demand − α_actual) / σ
     First-order lag in distance (σ = relaxation length).
     Converts the Mozzi disturbance into a delayed friction ellipse
     intrusion — this is WHY slip appears AFTER the φ̇ peak.

  Friction ellipse:
       Fx_avail = Fz · √(μx² − (Fy / (Fz·μy))²)
     Lateral demand from ② reduces available longitudinal force.

Usage:
    python apex_exit_analysis.py

Tune the parameters in the class P below and re-run.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════
#  PARAMETERS  ← tune these
# ══════════════════════════════════════════════════════════════════════
class P:
    # --- Geometry ---
    a   = 0.65    # CG → front axle   [m]
    b   = 0.75    # CG → rear axle    [m]
    h   = 0.60    # CG height         [m]
    L   = 1.40    # wheelbase         [m]

    # --- Mass / inertia ---
    m   = 220.0   # total mass        [kg]
    g   = 9.81    # gravity           [m/s²]
    Ix  =  35.0   # roll inertia      [kg·m²]   ← try 20–60
    Iz  =  50.0   # yaw inertia       [kg·m²]
    Ixz =  -5.0   # deviation moment  [kg·m²]   ← try 0 to see effect

    # --- Corner exit conditions ---
    phi0_deg  = 60.0   # initial lean angle  [°]   ← try 30–70
    V         = 70/3.6  # speed               [m/s]

    # --- Roll rate profile (Gaussian pulse) ---
    phi_dot_peak_deg = 120.0  # peak roll rate  [°/s]  ← try 40–200
    t_peak           = 0.18   # time of peak    [s]
    sigma_t          = 0.10   # pulse width     [s]    ← try 0.05–0.25

    # --- Tyre ---
    sigma_r  = 0.30    # rear relaxation length  [m]   ← try 0.15–0.60
    mu_x     = 1.05    # peak longitudinal friction
    mu_y     = 1.05    # peak lateral friction
    B_r      = 11.0    # Pacejka B (rear)
    C_r      = 1.30    # Pacejka C
    E_r      = 0.05    # Pacejka E

    # --- Throttle ---
    Fx_demand = 1800.0  # constant thrust demand  [N]  ← try 500–2500

    # --- Simulation ---
    dt = 0.001   # time step  [s]
    T  = 1.1     # duration   [s]


# ══════════════════════════════════════════════════════════════════════
#  SIMULATION
# ══════════════════════════════════════════════════════════════════════
def pacejka_norm(slip, B, C, E):
    """Normalised Pacejka (output in [-1, 1], multiply by Fz·μ for force)."""
    Bs = B * slip
    return np.sin(C * np.arctan(Bs - E * (Bs - np.arctan(Bs))))


def run():
    t  = np.arange(0, P.T, P.dt)
    N  = len(t)

    # ── Roll rate & acceleration ──────────────────────────────────────
    phi_dot_peak = np.radians(P.phi_dot_peak_deg)
    phi_dot  = phi_dot_peak * np.exp(-0.5 * ((t - P.t_peak) / P.sigma_t)**2)
    phi_ddot = np.gradient(phi_dot, P.dt)

    # ── Lean angle (integrate from initial value) ─────────────────────
    phi0 = np.radians(P.phi0_deg)
    phi  = np.clip(phi0 - np.cumsum(phi_dot) * P.dt, 0, phi0)

    # ── Yaw rate & acceleration (kinematic: ψ̇ = V·tan(φ)/L) ──────────
    psi_dot  = P.V * np.tan(phi) / P.L
    psi_ddot = np.gradient(psi_dot, P.dt)

    # ── Mechanism ①: Inertial Fz unloading ────────────────────────────
    # ΔFz_r = −(Ix·φ̈ + Ixz·ψ̈) / b
    dFz_inertial = -(P.Ix * phi_ddot + P.Ixz * psi_ddot) / P.b

    # Quasi-static rear vertical load (centrifugal + weight share)
    Fz_r_qs    = P.m * P.g * np.cos(phi) * P.a / P.L
    Fz_r_total = np.maximum(Fz_r_qs + dFz_inertial, 50.0)

    # ── Mechanism ②: Mozzi lateral velocity at rear contact ───────────
    # y_mozzi = ψ̇·V / (ψ̇² + φ̇²)
    eps   = 1e-8
    denom = psi_dot**2 + phi_dot**2
    y_mozzi  = np.where(denom > eps, psi_dot * P.V / denom, 0.0)
    y_mozzi  = np.clip(y_mozzi, -20, 20)

    v_lat     = phi_dot * y_mozzi                  # lateral velocity [m/s]
    alpha_src = np.arctan(v_lat / (P.V + eps))     # source slip angle [rad]

    # ── Mechanism ③: Tyre relaxation (first-order lag in distance) ────
    # dα/ds = (α_source − α_actual) / σ
    ds = P.V * P.dt
    alpha_actual = np.zeros(N)
    for i in range(1, N):
        alpha_actual[i] = (alpha_actual[i-1]
                           + (ds / P.sigma_r) * (alpha_src[i] - alpha_actual[i-1]))

    # ── Pacejka lateral force ─────────────────────────────────────────
    Fy_norm = pacejka_norm(alpha_actual, P.B_r, P.C_r, P.E_r)
    Fy      = Fy_norm * Fz_r_total * P.mu_y

    # ── Friction ellipse → available longitudinal force ───────────────
    Fx_avail = Fz_r_total * np.sqrt(
        np.maximum(0, P.mu_x**2 - (Fy / (Fz_r_total * P.mu_y + eps))**2))
    Fx_actual = np.minimum(P.Fx_demand, Fx_avail)
    slip_deficit = np.maximum(0, P.Fx_demand - Fx_avail)

    return dict(t=t,
                phi=phi, phi_dot=phi_dot, phi_ddot=phi_ddot,
                psi_dot=psi_dot, psi_ddot=psi_ddot,
                dFz_inertial=dFz_inertial,
                Fz_r_qs=Fz_r_qs, Fz_r_total=Fz_r_total,
                y_mozzi=y_mozzi, v_lat=v_lat,
                alpha_src=alpha_src, alpha_actual=alpha_actual,
                Fy=Fy, Fx_avail=Fx_avail,
                Fx_actual=Fx_actual, slip_deficit=slip_deficit)


# ══════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════
def plot(d):
    t_ms = d['t'] * 1000   # convert to ms for x-axis

    # ── Colour palette ────────────────────────────────────────────────
    BG   = '#F4F2ED'
    DARK = '#2C2C2A'
    MID  = '#888780'
    C_ROLL    = '#378ADD'
    C_ROLLACC = '#7F77DD'
    C_FZ      = '#D85A30'
    C_MOZZI   = '#EF9F27'
    C_SLIP    = '#E24B4A'
    C_FX      = '#1D9E75'

    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    fig.suptitle(
        f'Corner exit  ·  φ₀={P.phi0_deg}°,  V={P.V*3.6:.0f} km/h,  '
        f'φ̇_peak={P.phi_dot_peak_deg}°/s,  Ix={P.Ix} kg·m²,  Ixz={P.Ixz} kg·m²',
        fontsize=11.5, fontweight='bold', color=DARK, y=0.99)

    gs = mgridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.48, wspace=0.40,
                            left=0.07, right=0.97,
                            top=0.94, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, 0])   # roll rate & accel
    ax2 = fig.add_subplot(gs[0, 1])   # lean angle
    ax3 = fig.add_subplot(gs[0, 2])   # Mozzi y & v_lat
    ax4 = fig.add_subplot(gs[1, 0])   # Fz_r
    ax5 = fig.add_subplot(gs[1, 1])   # slip angle
    ax6 = fig.add_subplot(gs[1, 2])   # friction ellipse snapshot
    ax7 = fig.add_subplot(gs[2, 0:2]) # thrust + deficit
    ax8 = fig.add_subplot(gs[2, 2])   # normalised timeline

    for ax in fig.get_axes():
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=8, colors=DARK)
        for sp in ax.spines.values():
            sp.set_edgecolor('#D3D1C7'); sp.set_linewidth(0.6)
        ax.grid(True, color='#D3D1C7', lw=0.35, ls='--', alpha=0.8)

    def lbl(ax, title, yl=''):
        ax.set_title(title, fontsize=9, color=DARK, pad=3)
        ax.set_xlabel('t [ms]', fontsize=8, color=MID)
        if yl: ax.set_ylabel(yl, fontsize=8, color=MID)

    # ── ax1: φ̇ and φ̈ ─────────────────────────────────────────────────
    lbl(ax1, 'Rollrate φ̇  &  Rollbeschleunigung φ̈')
    ax1r = ax1.twinx()
    l1, = ax1.plot(t_ms, np.degrees(d['phi_dot']),
                   color=C_ROLL, lw=1.8, label='φ̇ [°/s]')
    l2, = ax1r.plot(t_ms, np.degrees(d['phi_ddot']),
                    color=C_ROLLACC, lw=1.4, ls='--', label='φ̈ [°/s²]')
    ax1.set_ylabel('φ̇ [°/s]', fontsize=8, color=C_ROLL)
    ax1r.set_ylabel('φ̈ [°/s²]', fontsize=8, color=C_ROLLACC)
    ax1r.tick_params(labelsize=8, colors=C_ROLLACC)
    ax1r.set_facecolor(BG)
    # Timing markers
    i_phidd_peak = np.argmax(np.abs(d['phi_ddot']))
    i_phid_peak  = np.argmax(d['phi_dot'])
    for ax_m, i, col, txt in [
        (ax1, i_phidd_peak, C_ROLLACC, f'φ̈ peak\n{t_ms[i_phidd_peak]:.0f}ms'),
        (ax1, i_phid_peak,  C_ROLL,    f'φ̇ peak\n{t_ms[i_phid_peak]:.0f}ms'),
    ]:
        ax_m.axvline(t_ms[i], color=col, lw=0.8, ls=':', alpha=0.7)
        ax_m.text(t_ms[i]+5, np.degrees(d['phi_dot']).max()*0.6,
                  txt, fontsize=7, color=col)
    ax1.legend(handles=[l1, l2], fontsize=7, loc='upper right', framealpha=0.7)

    # ── ax2: lean angle ───────────────────────────────────────────────
    lbl(ax2, 'Schräglagenwinkel φ', 'φ [°]')
    ax2.plot(t_ms, np.degrees(d['phi']), color=DARK, lw=1.8)
    ax2.axhline(P.phi0_deg, color=MID, lw=0.6, ls='--')
    ax2.fill_between(t_ms, np.degrees(d['phi']), 0, alpha=0.08, color=C_ROLL)

    # ── ax3: Mozzi y and v_lat ────────────────────────────────────────
    lbl(ax3, 'Mozzi-Offset  &  Lateralgeschwindigkeit am Kontaktpunkt')
    ax3r = ax3.twinx()
    ax3.plot(t_ms, d['y_mozzi'], color=C_MOZZI, lw=1.8, label='y_mozzi [m]')
    ax3r.plot(t_ms, d['v_lat'], color=C_SLIP, lw=1.4, ls='--',
              label='v_lat [m/s]')
    ax3.set_ylabel('y_mozzi [m]', fontsize=8, color=C_MOZZI)
    ax3r.set_ylabel('v_lat [m/s]', fontsize=8, color=C_SLIP)
    ax3r.tick_params(labelsize=8, colors=C_SLIP)
    ax3r.set_facecolor(BG)
    ax3.legend(fontsize=7, loc='upper right', framealpha=0.7)

    # ── ax4: Fz_r ─────────────────────────────────────────────────────
    lbl(ax4, 'Hinterrad-Vertikalkraft Fz_r', 'Fz [N]')
    ax4.plot(t_ms, d['Fz_r_qs'],    color=MID,  lw=1.2, ls='--',
             label='Fz_r quasi-statisch')
    ax4.plot(t_ms, d['Fz_r_total'], color=C_FZ, lw=1.8,
             label='Fz_r  gesamt (mit Trägheitsterm)')
    ax4.fill_between(t_ms, d['Fz_r_qs'], d['Fz_r_total'],
                     where=d['Fz_r_total'] < d['Fz_r_qs'],
                     color=C_FZ, alpha=0.15, label='Trägheits-Entlastung')
    i_fz_min = np.argmin(d['Fz_r_total'])
    ax4.annotate(f'Min: {d["Fz_r_total"][i_fz_min]:.0f} N\n@{t_ms[i_fz_min]:.0f} ms',
                 xy=(t_ms[i_fz_min], d['Fz_r_total'][i_fz_min]),
                 xytext=(t_ms[i_fz_min]+60, d['Fz_r_total'][i_fz_min]+100),
                 arrowprops=dict(arrowstyle='->', color=C_FZ, lw=0.8),
                 fontsize=7, color=C_FZ)
    ax4.legend(fontsize=7, framealpha=0.7)

    # ── ax5: slip angle ───────────────────────────────────────────────
    lag_ms = 3 * P.sigma_r / P.V * 1000
    lbl(ax5, f'Schräglaufwinkel α_r  (Reifenverzögerung σ={P.sigma_r} m  →  ~{lag_ms:.0f} ms)',
        'α [°]')
    ax5.plot(t_ms, np.degrees(d['alpha_src']),    color=C_MOZZI, lw=1.2,
             ls='--', alpha=0.6, label='α_mozzi (Quelle)')
    ax5.plot(t_ms, np.degrees(d['alpha_actual']), color=C_SLIP,  lw=1.8,
             label=f'α_r gefiltert (σ={P.sigma_r} m)')
    ax5.fill_between(t_ms, 0, np.degrees(d['alpha_actual']),
                     alpha=0.12, color=C_SLIP)
    ax5.axvline(t_ms[i_phid_peak] + lag_ms, color=C_SLIP, lw=0.8, ls=':', alpha=0.8)
    ax5.legend(fontsize=7, framealpha=0.7)

    # ── ax6: friction ellipse (snapshot at t=250 ms) ──────────────────
    lbl(ax6, 'Reibungsellipse  (Snapshot t=250 ms)', 'Fy / (μ·Fz)')
    ax6.set_xlabel('Fx / (μ·Fz)', fontsize=8, color=MID)
    theta_e = np.linspace(0, 2*np.pi, 300)
    ax6.plot(np.cos(theta_e), np.sin(theta_e), color=MID, lw=0.8, ls='--', alpha=0.5)
    i250    = min(int(0.25 / P.dt), len(d['t']) - 1)
    Fz_i    = d['Fz_r_total'][i250]
    Fy_i    = d['Fy'][i250]
    Fxa_i   = d['Fx_avail'][i250]
    Fxd_i   = min(P.Fx_demand, Fz_i * P.mu_x)
    Fy_n    = Fy_i / (Fz_i * P.mu_y)
    Fxa_n   = Fxa_i / (Fz_i * P.mu_x)
    Fxd_n   = Fxd_i / (Fz_i * P.mu_x)
    ax6.scatter([Fxa_n], [Fy_n], color=C_SLIP, s=70, zorder=5,
                label=f'Ist-Punkt\nFx={Fxa_i:.0f}N')
    ax6.scatter([Fxd_n], [Fy_n], color=C_FX,   s=50, marker='x',
                zorder=5, label=f'Bedarf\nFx={P.Fx_demand:.0f}N')
    ax6.annotate('', xy=(Fxa_n, Fy_n), xytext=(Fxd_n, Fy_n),
                 arrowprops=dict(arrowstyle='<->', color=C_SLIP, lw=1.2))
    ax6.text((Fxa_n+Fxd_n)/2, Fy_n + 0.05, 'Schlupf', fontsize=7,
             color=C_SLIP, ha='center')
    ax6.set_xlim(-0.15, 1.35); ax6.set_ylim(-0.55, 1.35)
    ax6.axhline(0, color=MID, lw=0.4); ax6.axvline(0, color=MID, lw=0.4)
    ax6.legend(fontsize=7, framealpha=0.7)

    # ── ax7: thrust timeline ──────────────────────────────────────────
    lbl(ax7, f'Schubkraft  —  Bedarf vs. Verfügbar  +  Schlupfdefizit', 'F [N]')
    ax7.axhline(P.Fx_demand, color=MID, lw=1.0, ls='--',
                label=f'Schubkraft-Bedarf ({P.Fx_demand:.0f} N)')
    ax7.plot(t_ms, d['Fx_avail'],  color=C_FX,  lw=1.8,
             label='Verfügbare Schubkraft')
    ax7.fill_between(t_ms, P.Fx_demand, d['Fx_avail'],
                     where=d['Fx_avail'] < P.Fx_demand,
                     color=C_SLIP, alpha=0.25, label='Schlupfdefizit')
    ax7r = ax7.twinx()
    ax7r.plot(t_ms, d['slip_deficit'], color=C_SLIP, lw=1.4, ls='--',
              label='Schlupfdefizit [N]')
    ax7r.set_ylabel('Schlupfdefizit [N]', fontsize=8, color=C_SLIP)
    ax7r.tick_params(labelsize=8, colors=C_SLIP)
    ax7r.set_facecolor(BG)
    # Event markers
    for tm, col, lbl_txt in [
        (t_ms[i_phidd_peak], C_ROLLACC, '① φ̈ peak'),
        (t_ms[i_phid_peak],  C_ROLL,    '② φ̇ peak'),
        (t_ms[i_phid_peak] + lag_ms, C_SLIP, f'③ Schlupf peak\n(+{lag_ms:.0f} ms)'),
    ]:
        ax7.axvline(tm, color=col, lw=0.8, ls=':', alpha=0.7)
        ax7.text(tm + 4, P.Fx_demand * 0.25, lbl_txt,
                 fontsize=7, color=col, rotation=90, va='bottom')
    ax7.legend(fontsize=7, framealpha=0.7, loc='lower right')

    # ── ax8: normalised timeline ──────────────────────────────────────
    lbl(ax8, 'Zeitliche Abfolge (normiert auf [0,1])')

    def n01(x):
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-12)

    ax8.plot(t_ms, n01(d['phi_ddot'].clip(0)),     color=C_ROLLACC, lw=1.6,
             label='① φ̈  (Rollbeschl.)')
    ax8.plot(t_ms, n01(d['phi_dot']),               color=C_ROLL,    lw=1.6,
             ls='--', label='② φ̇  (Rollrate)')
    ax8.plot(t_ms, n01(-d['dFz_inertial'].clip(max=0)), color=C_FZ, lw=1.6,
             ls='-.', label='③ ΔFz  (Entlastung)')
    ax8.plot(t_ms, n01(np.abs(d['alpha_actual'])), color=C_SLIP, lw=1.6,
             ls=':', label='④ α_r  (gefiltert)')
    ax8.plot(t_ms, n01(d['slip_deficit']),           color=C_SLIP, lw=2.4,
             alpha=0.45, label='⑤ Schlupfdefizit')
    for tm, col, num in [
        (t_ms[i_phidd_peak],          C_ROLLACC, '①'),
        (t_ms[i_phid_peak],           C_ROLL,    '②'),
        (t_ms[i_phid_peak] + lag_ms,  C_SLIP,    '③④⑤'),
    ]:
        ax8.axvline(tm, color=col, lw=0.7, ls=':', alpha=0.8)
        ax8.text(tm + 4, 1.05, num, fontsize=8, color=col, fontweight='bold')
    ax8.set_ylim(-0.1, 1.2)
    ax8.legend(fontsize=6.5, framealpha=0.7, loc='center right')

    # ── Print key numbers ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  φ̇  peak  :  {P.phi_dot_peak_deg:.0f} °/s   at {t_ms[i_phid_peak]:.0f} ms")
    print(f"  φ̈  peak  :  {np.degrees(np.abs(d['phi_ddot'])).max():.0f} °/s²  at {t_ms[i_phidd_peak]:.0f} ms")
    print(f"  ΔFz_r min:  {d['dFz_inertial'].min():.0f} N  (inertial)")
    print(f"  Fz_r min :  {d['Fz_r_total'].min():.0f} N  at {t_ms[np.argmin(d['Fz_r_total'])]:.0f} ms")
    print(f"  v_lat max:  {np.abs(d['v_lat']).max():.2f} m/s")
    print(f"  α_r max  :  {np.degrees(np.abs(d['alpha_actual'])).max():.2f}°")
    print(f"  Tyre lag :  {lag_ms:.0f} ms  (3·σ/V)")
    print(f"  Fx_avail min: {d['Fx_avail'].min():.0f} N")
    print(f"  Max slip deficit: {d['slip_deficit'].max():.0f} N")
    print(f"{'='*55}\n")

    plt.savefig('apex_exit_analysis.png', dpi=150, bbox_inches='tight',
                facecolor=BG)
    print("Plot saved as  apex_exit_analysis.png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    data = run()
    plot(data)
