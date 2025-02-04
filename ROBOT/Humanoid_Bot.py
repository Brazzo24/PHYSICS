import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def rotation_z(theta):
    """Return a 4x4 homogeneous rotation matrix about the Z-axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])

def rotation_y(theta):
    """Return a 4x4 homogeneous rotation matrix about the Y-axis."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ])

def translation(tx, ty, tz):
    """Return a 4x4 homogeneous translation matrix."""
    T = np.eye(4)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T

def apply_transform(T, point3):
    """
    Apply a 4x4 homogeneous transform T to a 3D point (x, y, z).
    Returns (x', y', z').
    """
    homo_point = np.array([point3[0], point3[1], point3[2], 1.0])
    transformed = T @ homo_point
    return transformed[:3]

# -------------------------------------------------
# Model Parameters
# -------------------------------------------------
# Torso as the global frame
T_torso_global = np.eye(4)

# Shoulder offset from the torso
T_shoulder_offset = translation(0.2, 0.3, 0.0)

# Arm length (from shoulder to elbow)
T_arm_length = translation(0.3, 0, 0)

# -------------------------------------------------
# Helper Function: Compute Torso->Elbow Transform
#   given a shoulder rotation about Z and Y
# -------------------------------------------------
def get_elbow_transform(shoulder_z_angle_deg, shoulder_y_angle_deg):
    """
    Returns the 4x4 transform from Torso to Elbow 
    given shoulder angles (in degrees).
    """
    # Convert degrees to radians
    theta_z = np.radians(shoulder_z_angle_deg)
    theta_y = np.radians(shoulder_y_angle_deg)
    
    # Rotation about Z
    Rz = rotation_z(theta_z)
    # Rotation about Y
    Ry = rotation_y(theta_y)
    
    # T_torso_to_shoulder = offset * Rz
    T_torso_to_shoulder = T_shoulder_offset @ Rz
    
    # T_torso_to_elbow = T_torso_to_shoulder * Ry * T_arm_length
    T_torso_to_elbow = T_torso_to_shoulder @ Ry @ T_arm_length
    return T_torso_to_elbow

# -------------------------------------------------
# Setup Matplotlib Figure
# -------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# We'll plot just a single 3D line for Torso->Shoulder->Elbow.
# It will have 3 points: [TorsoOrigin, Shoulder, Elbow].
line, = ax.plot([], [], [], 'o-', lw=2)

# Fix the 3D view limits so it doesn't jump around during animation
ax.set_xlim([-0.1, 0.6])
ax.set_ylim([-0.1, 0.6])
ax.set_zlim([-0.1, 0.6])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Simple Torso-Shoulder-Elbow Animation')

# -------------------------------------------------
# Animation Functions
# -------------------------------------------------
def init():
    """Initialize the line data (empty)"""
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    """
    For each frame, we'll vary the shoulder angles.
    Here, let's make 'frame' represent the Z-rotation in degrees,
    and we might also do a small Y-rotation for fun.
    """
    # Vary angles over time
    shoulder_z_angle = frame  # degrees
    shoulder_y_angle = 45 + 20 * np.sin(np.radians(frame))  # add a little oscillation

    # Compute transforms
    T_elbow = get_elbow_transform(shoulder_z_angle, shoulder_y_angle)
    
    # Extract points:
    torso_origin    = apply_transform(T_torso_global, [0, 0, 0])
    shoulder_origin = apply_transform(T_shoulder_offset @ rotation_z(np.radians(shoulder_z_angle)), [0, 0, 0])
    elbow_origin    = apply_transform(T_elbow, [0, 0, 0])
    
    # Build arrays for line plotting
    xs = [torso_origin[0], shoulder_origin[0], elbow_origin[0]]
    ys = [torso_origin[1], shoulder_origin[1], elbow_origin[1]]
    zs = [torso_origin[2], shoulder_origin[2], elbow_origin[2]]
    
    # Update the line data
    line.set_data(xs, ys)
    line.set_3d_properties(zs)
    
    return line,

# -------------------------------------------------
# Create the animation
# -------------------------------------------------
frames = np.arange(0, 91, 2)  # e.g. 0 to 90 degrees in steps of 2
anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=200)

plt.show()