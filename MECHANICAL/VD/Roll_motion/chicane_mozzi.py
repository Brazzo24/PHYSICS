"""
Motorbike Chicane Simulation — with Mozzi Axis Analysis
========================================================
Model:
  - Nonlinear lean-tracking ODE (roll dynamics)
  - Pacejka Magic Formula for lateral tyre forces
  - Kinematic bicycle model for yaw / path
  - Mozzi axis computed at every integration step
  - Vertical load transfer derived from instantaneous screw geometry

State vector:  [phi, phi_dot, psi, X, Y]
  phi      : roll (lean) angle [rad]
  phi_dot  : roll rate         [rad/s]
  psi      : yaw (heading)     [rad]
  X, Y     : global position   [m]
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

# ══════════════════════════════════════════════════════════════════
# 1.  VEHICLE PARAMETERS
# ══════════════════════════════════════════════════════════════════
class BikeParams:
    m           = 220.0   # total mass  [kg]
    h_com       = 0.58    # CoM height  [m]
    wheelbase   = 1.40    # L           [m]
    I_roll      = 30.0    # roll moment of inertia [kg·m²]
    g           = 9.81    # [m/s²]
    v_fwd       = 14.0    # constant forward speed [m/s]

    # Pacejka Magic Formula coefficients (lateral, typical sport bike tyre)
    B  = 10.0   # stiffness factor
    C  = 1.9    # shape factor
    D  = 1.0    # peak factor  (× F_z gives peak force)
    E  = -1.0   # curvature factor

P = BikeParams()


# ══════════════════════════════════════════════════════════════════
# 2.  PACEJKA MAGIC FORMULA
# ══════════════════════════════════════════════════════════════════
def pacejka_Fy(alpha, F_z, B=P.B, C=P.C, D=P.D, E=P.E):
    """
    Lateral force via Magic Formula.
    alpha : slip angle [rad]
    F_z   : normal (vertical) load [N]
    """
    Fz0   = P.m * P.g / 2.0          # reference load per axle
    mu    = D * (F_z / Fz0)          # load-dependent peak
    phi_p = (1.0 - E) * alpha + (E / B) * np.arctan(B * alpha)
    return mu * F_z * np.sin(C * np.arctan(B * phi_p))


# ══════════════════════════════════════════════════════════════════
# 3.  MOZZI AXIS  (screw axis of instantaneous rigid-body motion)
# ══════════════════════════════════════════════════════════════════
def mozzi_axis(omega, v_P, r_P=np.zeros(3)):
    """
    Returns (pitch h, point r_axis on axis, translational velocity v_par).
    If |ω| ≈ 0 (pure translation) returns (None, None, None).
    """
    omega = np.asarray(omega, float)
    v_P   = np.asarray(v_P,   float)
    r_P   = np.asarray(r_P,   float)

    w2 = float(omega @ omega)
    if w2 < 1e-10:
        return None, None, None

    h      = float(omega @ v_P) / w2
    r_axis = r_P + np.cross(omega, v_P) / w2
    v_par  = h * omega
    return h, r_axis, v_par


def bike_screw_state(phi, phi_dot, psi_dot, phi_ddot=0.0):
    """
    Build ω and v_CoM for the current bike state, then compute Mozzi axis.

    The body frame has:
      x̂ : forward  (along heading)
      ŷ : left
      ẑ : up (body)

    Angular velocity (in world/navigation frame, linearised around small lean):
      ω = phi_dot * x̂_body  +  psi_dot * ẑ_world
    """
    # Angular velocity vector (world frame, navigation frame approx.)
    omega = np.array([phi_dot * np.cos(phi),
                      phi_dot * np.sin(phi),
                      psi_dot])

    # CoM position in a body-fixed frame centred at contact-patch midpoint
    r_com = np.array([0.0, -P.h_com * np.sin(phi), P.h_com * np.cos(phi)])

    # CoM velocity (world frame)
    #   forward component + vertical component from lean rate
    v_com = np.array([P.v_fwd,
                      0.0,
                      phi_dot * P.h_com * np.cos(phi)])

    h_sc, r_ax, v_par = mozzi_axis(omega, v_com, r_com)
    return h_sc, r_ax, v_par, omega, v_com, r_com


# ══════════════════════════════════════════════════════════════════
# 4.  VERTICAL LOAD TRANSFER FROM MOZZI AXIS GEOMETRY
# ══════════════════════════════════════════════════════════════════
def vertical_loads(phi, phi_dot, psi_dot):
    """
    Compute front/rear vertical tyre loads using d'Alembert + Mozzi geometry.

    Two effects:
      (a) Longitudinal load transfer from pitch of Mozzi axis
          (couples roll rate to front/rear weight shift)
      (b) Centripetal load transfer (yaw → lateral centrifugal → small
          pitch correction through CoM offset)
    """
    h_sc, r_ax, v_par, omega, v_com, r_com = bike_screw_state(phi, phi_dot, psi_dot)

    # Static loads
    F_static = P.m * P.g / 2.0   # per axle

    # (a) Inertial force at CoM from roll acceleration (phi_ddot≈0 in ODE)
    #     Centrifugal from roll rate alone
    a_roll_vert = phi_dot**2 * P.h_com * np.cos(phi)

    # (b) Yaw-induced centripetal acceleration (lateral → tiny pitch component
    #     through CoM longitudinal offset, here assumed centred → 0)
    #     Real bikes have CoM slightly behind mid-wheelbase; keep it simple.

    # Pitch angle of Mozzi axis in the longitudinal-vertical plane
    if r_ax is not None:
        # angle of axis point relative to wheel-ground midpoint
        dz = r_ax[2]          # vertical offset of axis point
        dx = r_ax[0] + 1e-9   # longitudinal offset (avoid div/0)
        axis_pitch = np.arctan2(dz, dx)
    else:
        axis_pitch = 0.0

    # Load transfer: inertial moment projected onto wheelbase
    delta_F = P.m * a_roll_vert * np.tan(axis_pitch) if abs(axis_pitch) > 1e-6 else 0.0
    delta_F = np.clip(delta_F, -F_static * 0.8, F_static * 0.8)

    # Gravity component along lean
    F_grav_lat = P.m * P.g * np.sin(phi)   # lateral gravity (causes leaning moment)

    F_front = F_static + delta_F
    F_rear  = F_static - delta_F

    # Ensure physical positivity
    F_front = max(F_front, 10.0)
    F_rear  = max(F_rear,  10.0)

    return F_front, F_rear


# ══════════════════════════════════════════════════════════════════
# 5.  CHICANE STEERING INPUT
# ══════════════════════════════════════════════════════════════════
def steering_demand(t):
    """
    Desired lean angle schedule through a chicane.
    Phase 1 (0–1.5 s) : lean left  → right corner
    Phase 2 (1.5–3 s) : transition → left corner
    Phase 3 (3–4.5 s) : lean right → hold
    """
    phi_max = np.radians(32.0)
    if t < 1.5:
        return  phi_max * np.sin(np.pi * t / 1.5)
    elif t < 3.0:
        return -phi_max * np.sin(np.pi * (t - 1.5) / 1.5)
    else:
        return  phi_max * 0.5 * np.sin(np.pi * (t - 3.0) / 1.5)


# ══════════════════════════════════════════════════════════════════
# 6.  ODE RIGHT-HAND SIDE
# ══════════════════════════════════════════════════════════════════
def ode_rhs(t, y):
    phi, phi_dot, psi, X, Y = y

    # — Desired lean from chicane schedule —
    phi_ref     = steering_demand(t)
    phi_err     = phi_ref - phi
    phi_dot_ref = 0.0

    # PD lean controller → roll torque
    Kp, Kd   = 120.0, 25.0
    tau_roll  = Kp * phi_err + Kd * (phi_dot_ref - phi_dot)

    # — Vertical loads (Mozzi-informed) —
    F_front, F_rear = vertical_loads(phi, phi_dot, psi / (P.v_fwd + 1e-9))

    # — Slip angle (kinematic: yaw rate × wheelbase / speed) —
    psi_dot   = P.v_fwd * np.tan(phi) / (P.h_com + 1e-9)   # from lean balance
    alpha     = np.arctan2(psi_dot * P.wheelbase, P.v_fwd)

    # — Pacejka lateral forces —
    Fy_f = pacejka_Fy( alpha, F_front)
    Fy_r = pacejka_Fy(-alpha, F_rear)

    # — Roll dynamics (d'Alembert, linearised gravity term) —
    #   I·phi_ddot = tau_ctrl + m·g·h·sin(phi) - lateral force moment
    lateral_moment = (Fy_f + Fy_r) * P.h_com * np.cos(phi)
    gravity_moment =  P.m * P.g * P.h_com * np.sin(phi)
    phi_ddot = (tau_roll + gravity_moment - lateral_moment) / P.I_roll

    # — Kinematic yaw rate from lean (steady-state approximation) —
    psi_dot_kin = P.v_fwd * np.tan(phi) / P.wheelbase

    # — Global position —
    X_dot = P.v_fwd * np.cos(psi)
    Y_dot = P.v_fwd * np.sin(psi)

    return [phi_dot, phi_ddot, psi_dot_kin, X_dot, Y_dot]


# ══════════════════════════════════════════════════════════════════
# 7.  INTEGRATE
# ══════════════════════════════════════════════════════════════════
t_span = (0.0, 4.5)
t_eval = np.linspace(*t_span, 900)
y0     = [0.01, 0.0, 0.0, 0.0, 0.0]   # [phi, phi_dot, psi, X, Y]

sol = solve_ivp(ode_rhs, t_span, y0, t_eval=t_eval,
                method='RK45', rtol=1e-7, atol=1e-9)

t        = sol.t
phi_t    = sol.y[0]
phid_t   = sol.y[1]
psi_t    = sol.y[2]
X_t      = sol.y[3]
Y_t      = sol.y[4]


# ══════════════════════════════════════════════════════════════════
# 8.  POST-PROCESS  — Mozzi axis & forces at every time step
# ══════════════════════════════════════════════════════════════════
pitches  = np.zeros(len(t))
ax_x     = np.zeros(len(t))
ax_y     = np.zeros(len(t))
ax_z     = np.zeros(len(t))
F_fronts = np.zeros(len(t))
F_rears  = np.zeros(len(t))
Fy_fs    = np.zeros(len(t))
Fy_rs    = np.zeros(len(t))
psi_dots = np.zeros(len(t))

for i in range(len(t)):
    phi   = phi_t[i]
    phid  = phid_t[i]
    psid  = P.v_fwd * np.tan(phi) / P.wheelbase   # kinematic yaw rate

    psi_dots[i] = psid
    h_sc, r_ax, v_par, omega, v_com, r_com = bike_screw_state(phi, phid, psid)

    if h_sc is not None:
        pitches[i] = h_sc
        ax_x[i]    = r_ax[0]
        ax_y[i]    = r_ax[1]
        ax_z[i]    = r_ax[2]

    Ff, Fr          = vertical_loads(phi, phid, psid)
    F_fronts[i]     = Ff
    F_rears[i]      = Fr

    alpha_i         = np.arctan2(psid * P.wheelbase, P.v_fwd)
    Fy_fs[i]        = pacejka_Fy( alpha_i, Ff)
    Fy_rs[i]        = pacejka_Fy(-alpha_i, Fr)


# ══════════════════════════════════════════════════════════════════
# 9.  FIGURE 1 — Main Simulation Dashboard
# ══════════════════════════════════════════════════════════════════
phi_ref_t = np.array([steering_demand(ti) for ti in t])

fig1 = plt.figure(figsize=(14, 11))
fig1.suptitle("Motorbike Chicane — ODE Simulation with Mozzi Axis",
              fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.38)

# (0,0)–(0,1): Lean angle tracking
ax1 = fig1.add_subplot(gs[0, :2])
ax1.plot(t, np.degrees(phi_t),     'royalblue', lw=2,   label='φ (actual)')
ax1.plot(t, np.degrees(phi_ref_t), 'k--',       lw=1.5, label='φ_ref (demand)')
ax1.set_ylabel("Lean angle [°]"); ax1.set_title("Roll Angle Tracking")
ax1.legend(); ax1.grid(True, alpha=0.3)

# (0,2): Track path
ax2 = fig1.add_subplot(gs[0, 2])
sc = ax2.scatter(X_t, Y_t, c=np.degrees(phi_t), cmap='RdBu_r',
                 s=4, vmin=-35, vmax=35)
plt.colorbar(sc, ax=ax2, label='Lean [°]')
ax2.set_aspect('equal'); ax2.set_xlabel("X [m]"); ax2.set_ylabel("Y [m]")
ax2.set_title("Track Path\n(colour = lean)")
ax2.grid(True, alpha=0.3)

# (1,0): Mozzi pitch
ax3 = fig1.add_subplot(gs[1, 0])
ax3.plot(t, pitches, 'crimson', lw=2)
ax3.axhline(0, color='k', lw=0.8, ls='--')
ax3.fill_between(t, pitches, 0, alpha=0.15, color='crimson')
ax3.set_ylabel("Pitch h [m/rad]"); ax3.set_title("Mozzi Screw Pitch")
ax3.grid(True, alpha=0.3)

# (1,1): Mozzi axis point position
ax4 = fig1.add_subplot(gs[1, 1])
ax4.plot(t, ax_x, label='x',  color='steelblue',  lw=1.8)
ax4.plot(t, ax_y, label='y',  color='darkorange',  lw=1.8)
ax4.plot(t, ax_z, label='z',  color='seagreen',    lw=1.8)
ax4.set_ylabel("Position [m]"); ax4.set_title("Mozzi Axis Point (body frame)")
ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

# (1,2): Yaw rate
ax5 = fig1.add_subplot(gs[1, 2])
ax5.plot(t, np.degrees(psi_dots), color='purple', lw=2)
ax5.set_ylabel("ψ̇  [°/s]"); ax5.set_title("Yaw Rate")
ax5.grid(True, alpha=0.3)

# (2,0)–(2,1): Vertical tire loads
ax6 = fig1.add_subplot(gs[2, :2])
ax6.plot(t, F_fronts, label='F_z front', color='navy',  lw=2)
ax6.plot(t, F_rears,  label='F_z rear',  color='tomato',lw=2)
ax6.axhline(P.m*P.g/2, color='k', lw=0.8, ls='--', label='Static per axle')
ax6.set_ylabel("Force [N]"); ax6.set_xlabel("Time [s]")
ax6.set_title("Vertical Tyre Loads  (Mozzi-derived load transfer)")
ax6.legend(); ax6.grid(True, alpha=0.3)

# (2,2): Pacejka lateral forces
ax7 = fig1.add_subplot(gs[2, 2])
ax7.plot(t, Fy_fs, label='Fy front', color='teal',   lw=1.8)
ax7.plot(t, Fy_rs, label='Fy rear',  color='coral',  lw=1.8)
ax7.set_ylabel("Force [N]"); ax7.set_xlabel("Time [s]")
ax7.set_title("Pacejka Lateral Forces")
ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)

plt.show()


# ══════════════════════════════════════════════════════════════════
# 10. FIGURE 2 — 3D Mozzi Axis Evolution
#     Shows the axis in 3D space at selected instants
# ══════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(13, 5))
fig2.suptitle("Mozzi Axis in 3D — Snapshots Through the Chicane",
              fontsize=13, fontweight='bold')

N = len(t)
snapshots = [
    (0,              "t=0.0 s\n(start)"),
    (N//6,           "t≈0.75 s\n(peak lean L)"),
    (N//3,           "t≈1.5 s\n(transition)"),
    (N//2,           "t≈2.25 s\n(peak lean R)"),
    (min(2*N//3, N-1),"t≈3.0 s\n(exit)"),
]
colors_snap = ['royalblue','seagreen','crimson','darkorange','purple']

for idx, (si, label) in enumerate(snapshots):
    axi = fig2.add_subplot(1, 5, idx+1, projection='3d')

    phi  = phi_t[si]
    phid = phid_t[si]
    psid = P.v_fwd * np.tan(phi) / P.wheelbase

    h_sc, r_ax, v_par, omega, v_com, r_com = bike_screw_state(phi, phid, psid)

    # Draw bike frame (simplified: CoM and wheel contact points)
    r_f = np.array([ P.wheelbase/2, 0, 0])
    r_r = np.array([-P.wheelbase/2, 0, 0])

    # Lean the bike: rotate CoM around x-axis
    axi.scatter(*r_com, color=colors_snap[idx], s=60, zorder=5, label='CoM')
    axi.plot([r_r[0], r_f[0]], [r_r[1], r_f[1]], [r_r[2], r_f[2]],
             'k-', lw=2, label='Wheelbase')
    axi.plot([r_com[0], r_com[0]],
             [r_com[1], r_com[1]],
             [0,        r_com[2]],
             'k:', lw=1, alpha=0.5)

    # Draw Mozzi axis
    if r_ax is not None and omega is not None:
        omega_u = omega / (np.linalg.norm(omega) + 1e-12)
        s       = np.linspace(-0.8, 0.8, 40)
        pts     = r_ax[:, None] + omega_u[:, None] * s
        axi.plot(pts[0], pts[1], pts[2],
                 color=colors_snap[idx], lw=2.5, label='Mozzi axis')
        # Arrow for omega direction
        axi.quiver(*r_ax, *(0.5*omega_u),
                   color=colors_snap[idx], linewidth=2)

    # Ground plane
    xx, yy = np.meshgrid([-0.8, 0.8], [-0.8, 0.8])
    axi.plot_surface(xx, yy, np.zeros_like(xx),
                     alpha=0.08, color='grey')

    h_val = h_sc if h_sc is not None else 0.0
    axi.set_title(f"{label}\nh={h_val:.2f} m/rad\nφ={np.degrees(phi):.1f}°",
                  fontsize=8)
    axi.set_xlim(-1, 1); axi.set_ylim(-1, 1); axi.set_zlim(-0.2, 1.0)
    axi.set_xlabel('x', fontsize=7); axi.set_ylabel('y', fontsize=7)
    axi.set_zlabel('z', fontsize=7)
    axi.tick_params(labelsize=6)
    axi.view_init(elev=20, azim=-60)

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# 11. FIGURE 3 — Pacejka Friction Ellipse coloured by Mozzi pitch
# ══════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle("Pacejka Tyre Forces vs. Mozzi Pitch — Friction Ellipse View",
              fontsize=13, fontweight='bold')

for axi, (Fz_arr, Fy_arr, label) in zip(
        axes3,
        [(F_fronts, Fy_fs, "Front Tyre"),
         (F_rears,  Fy_rs, "Rear Tyre")]):

    sc = axi.scatter(Fz_arr, Fy_arr, c=pitches,
                     cmap='coolwarm', s=6,
                     vmin=-np.percentile(np.abs(pitches),95),
                     vmax= np.percentile(np.abs(pitches),95))
    plt.colorbar(sc, ax=axi, label='Mozzi pitch h [m/rad]')
    axi.set_xlabel("F_z  Vertical Load [N]")
    axi.set_ylabel("F_y  Lateral Force [N]")
    axi.set_title(f"{label}\nForce Space coloured by Mozzi Pitch")
    axi.grid(True, alpha=0.3)
    # Annotate start/end
    axi.scatter(Fz_arr[0],  Fy_arr[0],  marker='o', s=80,
                color='lime', zorder=5, label='start')
    axi.scatter(Fz_arr[-1], Fy_arr[-1], marker='s', s=80,
                color='black', zorder=5, label='end')
    axi.legend(fontsize=8)

plt.tight_layout()
plt.show()

# plt.close()
print("Friction ellipse plot saved.")

