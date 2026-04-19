import numpy as np
import matplotlib.pyplot as plt

def mozzi_axis(omega, v_P, r_P=np.zeros(3)):
    """
    Compute the Mozzi (instantaneous screw) axis.
    
    Parameters
    ----------
    omega : angular velocity vector [rad/s]
    v_P   : velocity of reference point P [m/s]
    r_P   : position of reference point P [m]
    
    Returns
    -------
    pitch     : screw pitch h [m/rad]
    r_axis    : a point on the Mozzi axis [m]
    v_parallel: translational velocity along the axis [m/s]
    """
    omega = np.array(omega, dtype=float)
    v_P   = np.array(v_P,   dtype=float)
    r_P   = np.array(r_P,   dtype=float)

    omega_sq = np.dot(omega, omega)
    if omega_sq < 1e-12:
        raise ValueError("ω ≈ 0: pure translation, Mozzi axis is at infinity.")

    h        = np.dot(omega, v_P) / omega_sq
    r_axis   = r_P + np.cross(omega, v_P) / omega_sq
    v_par    = h * omega

    return h, r_axis, v_par


# --- Example 1: Pure Rotation ---
omega = [0, 0, 5]   # rad/s about z
v_P   = [0, 0, 0]   # reference point at origin, stationary

h, r_axis, v_par = mozzi_axis(omega, v_P)

print("=== Example 1: Pure Rotation ===")
print(f"  Pitch h       = {h:.4f} m/rad  (0 → pure rotation)")
print(f"  Point on axis = {r_axis}")
print(f"  v_parallel    = {v_par}")

# --- Example 2: General Screw Motion (like a propeller shaft) ---
omega = [0, 0, 3]      # rotating about z [rad/s]
v_P   = [2, -1, 6]     # reference point has a general velocity [m/s]
r_P   = [1,  0, 0]     # reference point is offset from origin

h, r_axis, v_par = mozzi_axis(omega, v_P, r_P)

print("=== Example 2: Screw Motion ===")
print(f"  Pitch h       = {h:.4f} m/rad")
print(f"  Point on axis = {np.round(r_axis, 4)}")
print(f"  v_parallel    = {np.round(v_par, 4)} m/s")
print(f"  Axis direction (unit ω) = {omega / np.linalg.norm(omega)}")

def plot_mozzi(omega, v_P, r_P=np.zeros(3), ax=None, label=""):
    h, r_axis, v_par = mozzi_axis(omega, v_P, r_P)
    omega_u = np.array(omega) / np.linalg.norm(omega)

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax  = fig.add_subplot(111, projection='3d')

    # Draw Mozzi axis as a line
    t   = np.linspace(-2, 2, 50)
    pts = r_axis[:, None] + omega_u[:, None] * t
    ax.plot(*pts, 'r-', lw=2.5, label=f"Mozzi axis {label}")

    # Reference point and its velocity
    ax.scatter(*r_P, color='blue', s=60, zorder=5, label=f"Ref point P {label}")
    ax.quiver(*r_P, *v_P, color='blue', length=0.4,
              normalize=True, label=f"v_P {label}")

    # Angular velocity
    ax.quiver(*r_axis, *omega_u, color='green', length=1.0,
              normalize=False, label=f"ω {label}")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Mozzi Axis  |  pitch h = {h:.3f} m/rad")
    ax.legend(fontsize=8)
    return ax

fig = plt.figure(figsize=(7, 6))
ax  = fig.add_subplot(111, projection='3d')
plot_mozzi([0, 0, 3], [2, -1, 6], r_P=[1, 0, 0], ax=ax)
plt.tight_layout()
plt.show()