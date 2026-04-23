"""
Anti-squat simulation for a motorbike chain drive
Based on: Cossalter, "Motorcycle Dynamics", 2nd ed.

Anti-squat (AS%) is the ratio of the chain-induced anti-squat force
to the inertial load transfer under acceleration.

Coordinate system: x = forward (positive), y = upward (positive)
Origin at rear contact patch.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Bike parameters (typical sportbike, SI units) ────────────────────────────

params = dict(
    wheelbase       = 1.410,   # L  [m]
    CoM_height      = 0.620,   # h  [m]  centre of mass height
    CoM_x           = 0.600,   # x_G [m]  CoM longitudinal position from rear axle
    rear_axle_height= 0.310,   # r_r [m]  rear wheel radius
    front_axle_height=0.310,   # r_f [m]  front wheel radius

    # Sprocket radii
    r_rear_sprocket = 0.095,   # [m]  rear (wheel) sprocket
    r_front_sprocket= 0.030,   # [m]  front (countershaft) sprocket

    # Swingarm pivot position (from rear axle, in bike frame)
    swingarm_pivot_x= -0.450,  # negative = forward from rear axle
    swingarm_pivot_y=  0.080,  # above rear axle centre

    # Front sprocket position (from swingarm pivot)
    front_sprocket_dx= 0.000,  # nearly on the pivot axis (simplification)
    front_sprocket_dy= 0.000,  # can offset to model real geometry

    mass            = 200.0,   # m  [kg]  total (rider + bike)
    g               = 9.81,    # [m/s²]
)

# ── Core geometry functions ───────────────────────────────────────────────────

def swingarm_length(p):
    """Distance from swingarm pivot to rear axle."""
    px, py = p['swingarm_pivot_x'], p['swingarm_pivot_y']
    return np.sqrt(px**2 + py**2)

def swingarm_angle(p):
    """Static swingarm angle (rad), measured from horizontal."""
    px, py = p['swingarm_pivot_x'], p['swingarm_pivot_y']
    return np.arctan2(-py, -px)   # pivot is forward+above rear axle

def front_sprocket_pos(p):
    """
    Front (countershaft) sprocket centre in bike frame,
    relative to rear contact patch.
    """
    SA_len = swingarm_length(p)
    SA_ang = swingarm_angle(p)

    # Swingarm pivot in rear-axle frame
    piv_x = -SA_len * np.cos(SA_ang)
    piv_y = p['rear_axle_height'] + SA_len * np.sin(SA_ang)

    # Front sprocket offset from pivot (engine layout)
    fs_x = piv_x + p['front_sprocket_dx']
    fs_y = piv_y + p['front_sprocket_dy']
    return fs_x, fs_y

def compute_anti_squat(p):
    """
    Compute anti-squat percentage using the instant-centre method.

    The instant centre (IC) is the intersection of:
      1. The swingarm line (extended)
      2. The upper chain run line (extended)

    Anti-squat% is derived from the height of the intersection of the
    IC–rear-contact-patch line at the CoM x-position, compared to the
    CoM height.

    Returns AS% (scalar).
    """
    # Rear axle position (origin of our world frame = rear contact patch)
    rear_axle   = np.array([0.0, p['rear_axle_height']])

    # Swingarm pivot
    SA_len = swingarm_length(p)
    SA_ang = swingarm_angle(p)
    pivot = rear_axle + np.array([-SA_len * np.cos(SA_ang),
                                   SA_len * np.sin(SA_ang)])

    # Front (countershaft) sprocket centre
    fs_x, fs_y = front_sprocket_pos(p)
    front_spr = np.array([fs_x, fs_y])

    # Rear sprocket centre
    rear_spr = rear_axle  # sprocket concentric with axle (simplification)

    # ── Chain line (upper run) ──────────────────────────────────────────────
    # Tangent line between the two sprocket circles.
    # For the upper run (tension side), we need the external tangent.
    # Direction vector of centre-to-centre:
    d = rear_spr - front_spr
    dist = np.linalg.norm(d)
    d_hat = d / dist

    # For external tangent the angle offset due to different radii:
    r1, r2 = p['r_front_sprocket'], p['r_rear_sprocket']
    sin_phi = (r2 - r1) / dist
    cos_phi = np.sqrt(max(0.0, 1 - sin_phi**2))

    # Normal to the chain direction (perpendicular, upper side)
    # Rotate d_hat by angle phi around z:
    chain_dir = np.array([ d_hat[0]*cos_phi - d_hat[1]*sin_phi,
                            d_hat[0]*sin_phi + d_hat[1]*cos_phi])

    # Chain line passes through front sprocket top tangent point
    perp = np.array([-chain_dir[1], chain_dir[0]])
    chain_point = front_spr + r1 * perp

    # ── Swingarm line ───────────────────────────────────────────────────────
    sa_dir = rear_axle - pivot

    # ── Instant centre = intersection of the two lines ──────────────────────
    # Line 1: pivot + t * sa_dir
    # Line 2: chain_point + s * chain_dir
    # Solve: pivot + t*sa_dir = chain_point + s*chain_dir
    A = np.array([[sa_dir[0], -chain_dir[0]],
                  [sa_dir[1], -chain_dir[1]]])
    b = chain_point - pivot
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]

    if abs(det) < 1e-12:
        return np.nan   # parallel lines → no finite IC

    t = (b[0]*A[1,1] - b[1]*A[0,1]) / det
    IC = pivot + t * sa_dir

    # ── Anti-squat line: from rear contact patch through IC ─────────────────
    # y at CoM x-position
    cp = np.array([0.0, 0.0])                          # rear contact patch
    ic_dir = IC - cp
    if abs(ic_dir[0]) < 1e-12:
        return np.nan

    # CoM is at (x_G forward from rear axle, h above ground)
    # In our frame x is negative going forward, so:
    x_G = -p['CoM_x']   # forward is negative x in our frame
    h_G  = p['CoM_height']

    # Height of anti-squat line at CoM x-position
    h_as = cp[1] + ic_dir[1] / ic_dir[0] * (x_G - cp[0])

    AS_pct = (h_as / h_G) * 100.0
    return AS_pct


def compute_geometry_snapshot(p):
    """Return all key positions for plotting."""
    rear_axle   = np.array([0.0, p['rear_axle_height']])
    SA_len = swingarm_length(p)
    SA_ang = swingarm_angle(p)
    pivot = rear_axle + np.array([-SA_len * np.cos(SA_ang),
                                   SA_len * np.sin(SA_ang)])
    fs_x, fs_y = front_sprocket_pos(p)
    front_spr = np.array([fs_x, fs_y])
    rear_spr  = rear_axle

    d = rear_spr - front_spr
    dist = np.linalg.norm(d)
    d_hat = d / dist
    r1, r2 = p['r_front_sprocket'], p['r_rear_sprocket']
    sin_phi = (r2 - r1) / dist
    cos_phi = np.sqrt(max(0.0, 1 - sin_phi**2))
    chain_dir = np.array([ d_hat[0]*cos_phi - d_hat[1]*sin_phi,
                            d_hat[0]*sin_phi + d_hat[1]*cos_phi])
    perp = np.array([-chain_dir[1], chain_dir[0]])
    chain_pt_f = front_spr + r1 * perp
    chain_pt_r = rear_spr  + r2 * perp

    sa_dir = rear_axle - pivot
    A = np.array([[sa_dir[0], -chain_dir[0]],
                  [sa_dir[1], -chain_dir[1]]])
    b_vec = chain_pt_f - pivot
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) > 1e-12:
        t = (b_vec[0]*A[1,1] - b_vec[1]*A[0,1]) / det
        IC = pivot + t * sa_dir
    else:
        IC = None

    CoM = np.array([-p['CoM_x'], p['CoM_height']])

    return dict(
        rear_axle=rear_axle, pivot=pivot,
        front_spr=front_spr, rear_spr=rear_spr,
        chain_pt_f=chain_pt_f, chain_pt_r=chain_pt_r,
        chain_dir=chain_dir, IC=IC, CoM=CoM,
        r1=r1, r2=r2
    )


# ── Parametric sweeps ─────────────────────────────────────────────────────────

def sweep_CoM_height(p, h_range=(0.45, 0.85), n=80):
    heights = np.linspace(*h_range, n)
    AS = []
    for h in heights:
        pp = {**p, 'CoM_height': h}
        AS.append(compute_anti_squat(pp))
    return heights, np.array(AS)

def sweep_rear_sprocket(p, r_range=(0.060, 0.130), n=80):
    radii = np.linspace(*r_range, n)
    AS = []
    for r in radii:
        pp = {**p, 'r_rear_sprocket': r}
        AS.append(compute_anti_squat(pp))
    return radii, np.array(AS)

def sweep_pivot_height(p, dy_range=(-0.05, 0.12), n=80):
    offsets = np.linspace(*dy_range, n)
    AS = []
    for dy in offsets:
        pp = {**p, 'swingarm_pivot_y': dy}
        AS.append(compute_anti_squat(pp))
    return offsets, np.array(AS)

def sweep_swingarm_angle_via_length(p, angle_range=(-6, 14), n=80):
    """Vary swingarm angle by changing pivot y while keeping pivot x fixed."""
    angles_deg = np.linspace(*angle_range, n)
    AS = []
    pivot_x = p['swingarm_pivot_x']
    sa_len = swingarm_length(p)
    for ang in angles_deg:
        ang_r = np.radians(ang)
        new_py = -sa_len * np.sin(ang_r)
        pp = {**p, 'swingarm_pivot_y': new_py}
        AS.append(compute_anti_squat(pp))
    return angles_deg, np.array(AS)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all():
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0f0f0f')
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    C_BLUE   = '#4a9eff'
    C_AMBER  = '#f5a623'
    C_TEAL   = '#3ecfb0'
    C_CORAL  = '#ff6b6b'
    C_GRAY   = '#888888'
    C_GREEN  = '#5ecf6e'
    C_BG     = '#0f0f0f'
    C_PANEL  = '#1a1a1a'
    C_GRID   = '#2a2a2a'
    C_TEXT   = '#d0d0d0'
    C_TEXT2  = '#888888'

    AS_nominal = compute_anti_squat(params)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(C_PANEL)
        ax.set_title(title, color=C_TEXT, fontsize=11, pad=8)
        ax.set_xlabel(xlabel, color=C_TEXT2, fontsize=9)
        ax.set_ylabel(ylabel, color=C_TEXT2, fontsize=9)
        ax.tick_params(colors=C_TEXT2, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C_GRID)
        ax.grid(color=C_GRID, linewidth=0.5, linestyle='--')
        ax.axhline(100, color=C_GRAY, lw=1.0, linestyle=':', alpha=0.7,
                   label='100% (neutral)')
        ax.axhline(AS_nominal, color=C_AMBER, lw=0.8, linestyle='--', alpha=0.5)

    # ── Panel 1: Geometry diagram ─────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(C_PANEL)
    ax0.set_title('Anti-squat geometry', color=C_TEXT, fontsize=11, pad=8)
    ax0.set_aspect('equal')
    ax0.tick_params(colors=C_TEXT2, labelsize=8)
    for sp in ax0.spines.values(): sp.set_edgecolor(C_GRID)
    ax0.grid(color=C_GRID, linewidth=0.4, linestyle='--', alpha=0.5)
    ax0.set_xlabel('x [m] (forward →)', color=C_TEXT2, fontsize=9)
    ax0.set_ylabel('y [m]', color=C_TEXT2, fontsize=9)

    geo = compute_geometry_snapshot(params)
    ra  = geo['rear_axle']
    piv = geo['pivot']
    fs  = geo['front_spr']
    rs  = geo['rear_spr']
    IC  = geo['IC']
    CoM = geo['CoM']

    # Ground
    ax0.axhline(0, color=C_GRAY, lw=1.0, alpha=0.5)

    # Rear wheel
    rear_wh = plt.Circle((ra[0], ra[1]), params['rear_axle_height'],
                          fill=False, color=C_GRAY, lw=1.2)
    ax0.add_patch(rear_wh)

    # Swingarm
    ax0.plot([ra[0], piv[0]], [ra[1], piv[1]],
             color=C_BLUE, lw=2.5, solid_capstyle='round', label='Swingarm')
    ax0.plot(*piv, 'o', color=C_BLUE, ms=8, zorder=5, label='SA pivot')

    # Sprockets
    for centre, r, col, lbl in [
        (fs, params['r_front_sprocket'], C_AMBER, 'Front sprocket'),
        (rs, params['r_rear_sprocket'],  C_AMBER, 'Rear sprocket'),
    ]:
        circ = plt.Circle(centre, r, fill=False, color=col, lw=1.2)
        ax0.add_patch(circ)
        ax0.plot(*centre, '+', color=col, ms=8)

    # Chain upper run
    ax0.plot([geo['chain_pt_f'][0], geo['chain_pt_r'][0]],
             [geo['chain_pt_f'][1], geo['chain_pt_r'][1]],
             color=C_AMBER, lw=1.8, linestyle='--', label='Chain (tension)')

    # IC
    if IC is not None:
        ax0.plot(*IC, 's', color=C_TEAL, ms=8, zorder=6, label='Instant centre')
        # Anti-squat line: contact patch → IC → CoM
        cp = np.array([0.0, 0.0])
        xs = [cp[0], IC[0], CoM[0]-0.1]
        ys = [cp[1], IC[1],
              cp[1] + (IC[1]-cp[1])/(IC[0]-cp[0]) * (CoM[0]-0.1 - cp[0])]
        ax0.plot(xs, ys, color=C_GREEN, lw=1.2, linestyle=':', alpha=0.8,
                 label='AS line')

    # CoM
    ax0.plot(*CoM, '*', color=C_CORAL, ms=14, zorder=7, label='CoM')
    ax0.annotate(f'AS = {AS_nominal:.1f}%', xy=CoM,
                 xytext=(CoM[0]+0.05, CoM[1]+0.06),
                 color=C_CORAL, fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=C_CORAL, lw=0.8))

    ax0.set_xlim(-0.70, 0.25)
    ax0.set_ylim(-0.05, 0.80)
    ax0.legend(fontsize=7, loc='upper right',
               facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── Panel 2: AS% vs CoM height ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1, 'AS% vs CoM height', 'CoM height h [m]', 'Anti-squat [%]')
    h_vals, AS_h = sweep_CoM_height(params)
    ax1.plot(h_vals, AS_h, color=C_CORAL, lw=2)
    ax1.axvline(params['CoM_height'], color=C_AMBER, lw=1, linestyle='--',
                alpha=0.8, label='Nominal')
    ax1.plot(params['CoM_height'], AS_nominal, 'o', color=C_AMBER, ms=6, zorder=5)
    ax1.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax1.set_ylim(0, 200)

    # ── Panel 3: AS% vs rear sprocket radius ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, 'AS% vs rear sprocket radius',
             'Rear sprocket radius [m]', 'Anti-squat [%]')
    r_vals, AS_r = sweep_rear_sprocket(params)
    teeth_approx = r_vals / params['r_rear_sprocket'] * 42  # ~42T nominal
    ax2.plot(r_vals, AS_r, color=C_TEAL, lw=2)
    ax2.axvline(params['r_rear_sprocket'], color=C_AMBER, lw=1, linestyle='--',
                alpha=0.8, label=f"Nominal ({params['r_rear_sprocket']*1000:.0f} mm)")
    ax2.plot(params['r_rear_sprocket'], AS_nominal, 'o', color=C_AMBER, ms=6, zorder=5)
    ax2.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax2.set_ylim(0, 200)

    # Secondary x axis: approximate tooth count
    ax2b = ax2.twiny()
    ax2b.set_xlim(ax2.get_xlim())
    t_ticks = np.array([32, 36, 40, 44, 48, 52])
    r_ticks  = t_ticks / 42 * params['r_rear_sprocket']
    ax2b.set_xticks(r_ticks)
    ax2b.set_xticklabels([f'{t}T' for t in t_ticks], fontsize=7, color=C_TEXT2)
    ax2b.tick_params(colors=C_TEXT2)
    ax2b.spines['top'].set_edgecolor(C_GRID)

    # ── Panel 4: AS% vs swingarm angle ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'AS% vs swingarm angle',
             'Swingarm angle [°] (positive = nose up)', 'Anti-squat [%]')
    ang_vals, AS_ang = sweep_swingarm_angle_via_length(params)
    nom_ang = np.degrees(swingarm_angle(params))
    ax3.plot(ang_vals, AS_ang, color=C_BLUE, lw=2)
    ax3.axvline(nom_ang, color=C_AMBER, lw=1, linestyle='--', alpha=0.8,
                label=f'Nominal ({nom_ang:.1f}°)')
    ax3.plot(nom_ang, AS_nominal, 'o', color=C_AMBER, ms=6, zorder=5)
    ax3.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
    ax3.set_ylim(0, 200)

    # ── Panel 5: 2D heat-map — sprocket radius × CoM height ──────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(C_PANEL)
    ax4.set_title('AS% heat map: sprocket radius × CoM height',
                  color=C_TEXT, fontsize=10, pad=8)
    ax4.set_xlabel('Rear sprocket radius [m]', color=C_TEXT2, fontsize=9)
    ax4.set_ylabel('CoM height [m]', color=C_TEXT2, fontsize=9)
    ax4.tick_params(colors=C_TEXT2, labelsize=8)
    for sp in ax4.spines.values(): sp.set_edgecolor(C_GRID)

    r_grid  = np.linspace(0.065, 0.125, 40)
    h_grid  = np.linspace(0.45,  0.82,  40)
    AS_map  = np.zeros((len(h_grid), len(r_grid)))
    for i, h in enumerate(h_grid):
        for j, r in enumerate(r_grid):
            pp = {**params, 'CoM_height': h, 'r_rear_sprocket': r}
            AS_map[i, j] = compute_anti_squat(pp)

    im = ax4.pcolormesh(r_grid, h_grid, AS_map,
                        cmap='RdYlGn', vmin=20, vmax=180, shading='auto')
    cb = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cb.set_label('AS%', color=C_TEXT2, fontsize=9)
    cb.ax.tick_params(colors=C_TEXT2, labelsize=8)
    cb.ax.yaxis.set_tick_params(color=C_TEXT2)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=C_TEXT2)

    # 100% contour
    cs = ax4.contour(r_grid, h_grid, AS_map, levels=[100],
                     colors=['white'], linewidths=1.5, linestyles='--')
    ax4.clabel(cs, fmt='100%%', colors='white', fontsize=8)

    ax4.plot(params['r_rear_sprocket'], params['CoM_height'],
             '*', color=C_AMBER, ms=12, zorder=5, label='Nominal')
    ax4.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── Panel 6: Sensitivity bar chart ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(C_PANEL)
    ax5.set_title('Local sensitivity (ΔAS% per ±10% param change)',
                  color=C_TEXT, fontsize=10, pad=8)
    ax5.tick_params(colors=C_TEXT2, labelsize=8)
    for sp in ax5.spines.values(): sp.set_edgecolor(C_GRID)
    ax5.grid(color=C_GRID, linewidth=0.5, linestyle='--', axis='x')

    sens_params = {
        'CoM height':         ('CoM_height',         0.10),
        'Rear sprocket r':    ('r_rear_sprocket',     0.10),
        'Front sprocket r':   ('r_front_sprocket',    0.10),
        'Pivot height':       ('swingarm_pivot_y',    0.10),
        'CoM fore/aft':       ('CoM_x',               0.10),
        'Wheelbase':          ('wheelbase',            0.10),
    }
    names = []
    deltas = []
    for label, (key, frac) in sens_params.items():
        nominal_val = params[key]
        dp = compute_anti_squat({**params, key: nominal_val*(1+frac)})
        dm = compute_anti_squat({**params, key: nominal_val*(1-frac)})
        delta = (dp - dm) / 2
        names.append(label)
        deltas.append(delta)

    colors_bar = [C_CORAL if d < 0 else C_TEAL for d in deltas]
    bars = ax5.barh(names, deltas, color=colors_bar, edgecolor='none', height=0.55)
    ax5.axvline(0, color=C_GRAY, lw=1.0)
    ax5.set_xlabel('ΔAS% per ±10% parameter change', color=C_TEXT2, fontsize=9)
    for bar, val in zip(bars, deltas):
        x = val + (1.5 if val >= 0 else -1.5)
        ax5.text(x, bar.get_y() + bar.get_height()/2,
                 f'{val:+.1f}%', va='center', ha='left' if val >= 0 else 'right',
                 color=C_TEXT2, fontsize=8)

    # ── Title & annotation ────────────────────────────────────────────────────
    fig.suptitle(
        f'Motorbike Anti-Squat Analysis  |  Nominal AS = {AS_nominal:.1f}%\n'
        f'(Cossalter instant-centre method)',
        color=C_TEXT, fontsize=13, y=0.98
    )

    plt.savefig('/mnt/user-data/outputs/anti_squat_simulation.png',
                dpi=160, bbox_inches='tight', facecolor=C_BG)
    print(f"Nominal anti-squat: {AS_nominal:.2f}%")
    print("Saved: anti_squat_simulation.png")
    plt.show()


if __name__ == '__main__':
    # Quick console report
    print("=" * 55)
    print("  ANTI-SQUAT ANALYSIS  (Cossalter instant-centre)")
    print("=" * 55)
    AS = compute_anti_squat(params)
    print(f"  Nominal AS%          : {AS:.2f}%")
    print(f"  Swingarm angle       : {np.degrees(swingarm_angle(params)):.2f}°")
    print(f"  Swingarm length      : {swingarm_length(params)*1000:.1f} mm")
    geo = compute_geometry_snapshot(params)
    if geo['IC'] is not None:
        print(f"  Instant centre (x,y) : ({geo['IC'][0]:.3f}, {geo['IC'][1]:.3f}) m")
    print("=" * 55)
    print()

    # What sprocket radius gives 100%?
    from scipy.optimize import brentq
    def residual(r):
        return compute_anti_squat({**params, 'r_rear_sprocket': r}) - 100.0
    try:
        r_100 = brentq(residual, 0.060, 0.140)
        print(f"  Rear sprocket for 100% AS: {r_100*1000:.1f} mm")
    except Exception:
        print("  (100% AS not achievable in the given sprocket range)")
    print()

    plot_all()
