# Re-import necessary libraries after kernel reset
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# System parameters
J = np.array([
    0.10,  # Crank
    0.015, # Inlet cam gear end
    0.005, # Inlet cam lobe end
    0.015, # Outlet cam gear end
    0.005, # Outlet cam lobe end
    0.50   # Dyno
])

g1 = 2.0
g2 = -1.0

K_c = 1e4
C_c = 50.0
K_g1 = 5e4
C_g1 = 10.0
K_g2 = 5e4
C_g2 = 10.0
K_cam = 2e4
C_cam = 5.0

n_dof = len(J)
M = np.diag(J)
K = np.zeros((n_dof, n_dof))
C = np.zeros((n_dof, n_dof))

def add_connection(i, j, k_val, c_val, ratio=1.0):
    K[i,i] += k_val
    K[j,j] += k_val * ratio**2
    K[i,j] -= k_val * ratio
    K[j,i] -= k_val * ratio
    C[i,i] += c_val
    C[j,j] += c_val * ratio**2
    C[i,j] -= c_val * ratio
    C[j,i] -= c_val * ratio

# Reconstruct the system
add_connection(0, 5, K_c, C_c)
add_connection(0, 1, K_g1, C_g1, g1)
add_connection(1, 2, K_cam, C_cam)
add_connection(1, 3, K_g2, C_g2, g2)
add_connection(3, 4, K_cam, C_cam)

# Modal analysis
eigvals, eigvecs = la.eig(K, M)
eigvals = np.real(eigvals)
eigvecs = np.real(eigvecs)

idx = np.argsort(eigvals)
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

frequencies_hz = np.sqrt(eigvals[1:]) / (2 * np.pi)
mode_shapes = eigvecs[:, 1:]

mode_shapes_norm = mode_shapes / np.max(np.abs(mode_shapes), axis=0)

# Plot mode shapes
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(mode_shapes_norm.shape[1]):
    ax.plot(range(n_dof), mode_shapes_norm[:, i], marker='o', label=f'Mode {i+1} ({frequencies_hz[i]:.1f} Hz)')

ax.set_xticks(range(n_dof))
ax.set_xticklabels([
    'Crank',
    'Inlet Cam Gear',
    'Inlet Cam Lobe',
    'Outlet Cam Gear',
    'Outlet Cam Lobe',
    'Dyno'
])
ax.set_ylabel('Normalized Mode Shape')
ax.set_title('Torsional Mode Shapes of Branched Camtrain System')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
