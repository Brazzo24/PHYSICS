# Multi-segment camshafts: let's use 3 segments per camshaft (gear-side, mid, lobe-side)
# Total DOFs:
# 0 - Crank
# 1, 2, 3 - Inlet camshaft (gear, mid, lobe)
# 4, 5, 6 - Outlet camshaft (gear, mid, lobe)
# 7 - Dyno
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Define parameters
n_segments = 3
J_crank = 0.10
J_dyno = 0.50
J_cam_seg = 0.005  # evenly distributed
K_cam_seg = 3e4
C_cam_seg = 5.0

# Inertia vector: crank + 3 inlet + 3 outlet + dyno
J = np.array(
    [J_crank] +
    [J_cam_seg] * n_segments +  # inlet camshaft
    [J_cam_seg] * n_segments +  # outlet camshaft
    [J_dyno]
)

n_dof = len(J)
M = np.diag(J)
K = np.zeros((n_dof, n_dof))
C = np.zeros((n_dof, n_dof))

# Gear ratios
g1 = 2.0
g2 = -1.0
K_g1 = 5e4
C_g1 = 10.0
K_g2 = 5e4
C_g2 = 10.0
K_c = 1e4
C_c = 50.0

def add_connection(i, j, k_val, c_val, ratio=1.0):
    K[i,i] += k_val
    K[j,j] += k_val * ratio**2
    K[i,j] -= k_val * ratio
    K[j,i] -= k_val * ratio
    C[i,i] += c_val
    C[j,j] += c_val * ratio**2
    C[i,j] -= c_val * ratio
    C[j,i] -= c_val * ratio

# Index mapping
i_crank = 0
i_inlet = [1, 2, 3]
i_outlet = [4, 5, 6]
i_dyno = 7

# Coupling: crank ↔ dyno
add_connection(i_crank, i_dyno, K_c, C_c)

# Crank ↔ Inlet cam gear end (via gear)
add_connection(i_crank, i_inlet[0], K_g1, C_g1, g1)

# Inlet cam segments (1-2, 2-3)
add_connection(i_inlet[0], i_inlet[1], K_cam_seg, C_cam_seg)
add_connection(i_inlet[1], i_inlet[2], K_cam_seg, C_cam_seg)

# Inlet cam gear end ↔ Outlet cam gear end (via gear)
add_connection(i_inlet[0], i_outlet[0], K_g2, C_g2, g2)

# Outlet cam segments (4-5, 5-6)
add_connection(i_outlet[0], i_outlet[1], K_cam_seg, C_cam_seg)
add_connection(i_outlet[1], i_outlet[2], K_cam_seg, C_cam_seg)

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
labels = ['Crank'] + \
         [f'Inlet {i}' for i in ['Gear', 'Mid', 'Lobe']] + \
         [f'Outlet {i}' for i in ['Gear', 'Mid', 'Lobe']] + \
         ['Dyno']

fig, ax = plt.subplots(figsize=(12, 6))
for i in range(min(5, mode_shapes_norm.shape[1])):  # show first 5 modes
    ax.plot(range(n_dof), mode_shapes_norm[:, i], marker='o', label=f'Mode {i+1} ({frequencies_hz[i]:.1f} Hz)')

ax.set_xticks(range(n_dof))
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel('Normalized Mode Shape')
ax.set_title('Mode Shapes for Multi-Segment Camtrain Model')
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
