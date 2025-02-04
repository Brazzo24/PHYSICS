import numpy as np
from my_rotations import rotation_z
from visualization import plot_vectors

# Define input vector and rotation angle
A = np.array([1, 2, 3])
alpha = 45  # Rotation angle in degrees

# Apply the rotation
A_new = rotation_z(A, alpha)

# Visualize the vectors
plot_vectors(
    vectors=[A, A_new],
    labels=["Original Vector", "Rotated Vector"],
    colors=["blue", "red"],
    title=f"Rotation by {alpha} degrees around Z-axis"
)