import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────
# Bike parameters
# ─────────────────────────────────────────────
wheelbase   = 1.40   # m
h_com       = 0.60   # m  (CoM height)
m           = 220.0  # kg (bike + rider)
g           = 9.81   # m/s²

# Front/rear contact patch positions
r_front = np.array([wheelbase/2,  0.0, 0.0])
r_rear  = np.array([-wheelbase/2, 0.0, 0.0])
r_com   = np.array([0.0,          0.0, h_com])


def mozzi_axis(omega, v_P, r_P=np.zeros(3)):
    omega = np.asarray(omega, float)
    v_P   = np.asarray(v_P,   float)
    r_P   = np.asarray(r_P,   float)
    w2 = omega @ omega
    if w2 < 1e-12:
        return None, None, None
    h      = (omega @ v_P) / w2
    r_axis = r_P + np.cross(omega, v_P) / w2
    v_par  = h * omega
    return h, r_axis, v_par


def bike_mozzi(phi_dot, psi_dot, v_fwd, lean_angle_deg=0.0):
    """
    Compute Mozzi axis for a motorbike cornering scenario.

    phi_dot        : roll rate  [rad/s]
    psi_dot        : yaw rate   [rad/s]
    v_fwd          : forward speed [m/s]
    lean_angle_deg : static lean (phi) [deg]
    """
    phi = np.radians(lean_angle_deg)

    # Angular velocity in body frame (roll about x, yaw about z)
    omega = np.array([phi_dot,
                      0.0,
                      psi_dot])

    # CoM velocity: forward speed + lateral drift from lean
    # (simplified planar model)
    v_com = np.array([v_fwd * np.cos(phi),
                      v_fwd * np.sin(phi) * 0.0,   # no sideslip assumed
                      phi_dot * h_com * np.cos(phi)])

    h_screw, r_axis, v_par = mozzi_axis(omega, v_com, r_com)
    return h_screw, r_axis, v_par, omega, v_com


# ─────────────────────────────────────────────
# Scenario: entering a left-hand corner
# ─────────────────────────────────────────────
phi_dot = 0.3    # rolling in [rad/s]
psi_dot = 0.4    # yawing left [rad/s]
v_fwd   = 12.0   # m/s
lean    = 25.0   # deg

h_sc, r_ax, v_par, omega, v_com = bike_mozzi(phi_dot, psi_dot, v_fwd, lean)

print("=== Mozzi Axis — Cornering Motorbike ===")
print(f"  Screw pitch h     = {h_sc:.4f} m/rad")
print(f"  Point on axis     = {np.round(r_ax, 4)} m")
print(f"  v_parallel (along axis) = {np.round(v_par, 4)} m/s")
print(f"  Angular velocity  = {np.round(omega, 4)} rad/s")

def tire_load_transfer(r_axis, omega, v_fwd, phi_dot, psi_dot, lean_deg):
    """
    Estimate vertical load transfer front/rear from Mozzi axis geometry.
    Uses d'Alembert: inertial forces at CoM projected onto vertical.
    """
    phi = np.radians(lean_deg)

    # Centripetal acceleration (yaw-induced lateral)
    R_turn   = v_fwd / (psi_dot + 1e-9)   # turn radius
    a_lat    = v_fwd**2 / R_turn           # lateral centripetal acc

    # Roll angular acceleration contribution (simplified: steady state → 0)
    a_vert   = phi_dot**2 * h_com          # centrifugal from roll

    # Static distribution (assume CoM at centre of wheelbase)
    F_static = m * g / 2.0                 # per axle

    # Load transfer due to braking/acceleration (none here) and
    # longitudinal inertia (none in steady state).
    # Lateral load transfer shifts side-to-side, not front/rear —
    # but pitch of Mozzi axis encodes front/rear coupling:
    pitch_angle = np.arctan2(r_axis[2], r_axis[0])  # axis tilt in xz plane

    delta_F_longitudinal = m * a_vert * np.sin(pitch_angle)

    F_front = F_static + delta_F_longitudinal
    F_rear  = F_static - delta_F_longitudinal

    return F_front, F_rear, R_turn, a_lat


F_f, F_r, R, a_l = tire_load_transfer(r_ax, omega, v_fwd,
                                       phi_dot, psi_dot, lean)

print("\n=== Vertical Tire Forces ===")
print(f"  Turn radius       = {R:.2f} m")
print(f"  Lateral acc       = {a_l:.3f} m/s²")
print(f"  F_front (vertical)= {F_f:.1f} N")
print(f"  F_rear  (vertical)= {F_r:.1f} N")
print(f"  Total             = {F_f+F_r:.1f} N  (check: mg = {m*g:.1f} N)")


# ─────────────────────────────────────────────
# Chicane simulation: time-varying phi_dot, psi_dot
# ─────────────────────────────────────────────
t       = np.linspace(0, 4, 400)
v_fwd   = 12.0   # constant forward speed

# Sinusoidal roll/yaw inputs simulating chicane
phi_dot_t  =  0.5 * np.sin(2 * np.pi * t / 4.0)
psi_dot_t  =  0.4 * np.sin(2 * np.pi * t / 4.0)
lean_t     = 30.0 * np.sin(2 * np.pi * t / 4.0)   # deg

# Collect Mozzi axis data
pitches, axis_x, axis_z = [], [], []
F_fronts, F_rears       = [], []

for i in range(len(t)):
    try:
        h_sc, r_ax, v_par, omega, v_com = bike_mozzi(
            phi_dot_t[i], psi_dot_t[i], v_fwd, lean_t[i])
        if r_ax is None:
            continue
        F_f, F_r, _, _ = tire_load_transfer(
            r_ax, omega, v_fwd, phi_dot_t[i], psi_dot_t[i], lean_t[i])

        pitches.append(h_sc)
        axis_x.append(r_ax[0])
        axis_z.append(r_ax[2])
        F_fronts.append(F_f)
        F_rears.append(F_r)
    except Exception:
        pass

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle("Mozzi Axis & Tire Forces Through a Chicane", fontsize=13, fontweight='bold')

axes[0].plot(t[:len(pitches)], pitches, color='crimson', lw=2)
axes[0].axhline(0, color='k', lw=0.8, ls='--')
axes[0].set_ylabel("Screw Pitch h [m/rad]")
axes[0].set_title("Pitch of Mozzi Axis  (0 = pure rotation)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t[:len(axis_x)], axis_x, label="x-position", color='steelblue', lw=2)
axes[1].plot(t[:len(axis_z)], axis_z, label="z-position", color='darkorange', lw=2)
axes[1].set_ylabel("Axis position [m]")
axes[1].set_title("Position of point on Mozzi Axis (body frame)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t[:len(F_fronts)], F_fronts, label="F_front", color='navy',   lw=2)
axes[2].plot(t[:len(F_rears)],  F_rears,  label="F_rear",  color='tomato', lw=2)
axes[2].axhline(m*g/2, color='k', lw=0.8, ls='--', label="Static (each axle)")
axes[2].set_ylabel("Vertical Force [N]")
axes[2].set_xlabel("Time [s]")
axes[2].set_title("Vertical Tire Loads")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()