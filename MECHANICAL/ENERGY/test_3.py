import numpy as np
import scipy.linalg
import pandas as pd
import matplotlib.pyplot as plt

# System Parameters
m1, m2 = 2.0, 1.5  # Masses (kg)
k1, k2 = 1000, 800  # Stiffness (N/m)
c1, c2 = 5, 3       # Damping (Ns/m)

# Mass matrix
M = np.array([[m1, 0],
              [0, m2]])

# Stiffness matrix
K = np.array([[k1 + k2, -k2],
              [-k2, k2]])

# Damping matrix
C = np.array([[c1 + c2, -c2],
              [-c2, c2]])

# Solve the generalized eigenvalue problem for modal analysis
eigvals, eigvecs = scipy.linalg.eigh(K, M)

# Natural frequencies (Hz)
omega_n = np.sqrt(eigvals) / (2 * np.pi)

# Ensure that mode shapes are correctly treated as a 2D matrix
modes = eigvecs.reshape((M.shape[0], -1))

# Compute modal mass for each mode individually
modal_masses = np.array([modes[:, i].T @ M @ modes[:, i] for i in range(modes.shape[1])])

# Compute kinetic and potential energy
KE_modes = 0.5 * modal_masses * omega_n**2
PE_modes = 0.5 * np.diag(modes.T @ K @ modes)

# Compute power dissipation due to damping
P_damping = np.diag(modes.T @ C @ modes)

# Create a DataFrame for results
df_modes = pd.DataFrame({
    "Mode": [1, 2],
    "Natural Frequency (Hz)": omega_n,
    "Kinetic Energy (J)": KE_modes,
    "Potential Energy (J)": PE_modes,
    "Power Dissipation (W)": P_damping
})

# Print results as text table
print(df_modes.to_string(index=False))

# Visualization - Energy Distribution
fig, ax = plt.subplots(figsize=(8, 5))
modes = df_modes["Mode"]
kinetic_energy = df_modes["Kinetic Energy (J)"]
potential_energy = df_modes["Potential Energy (J)"]

ax.bar(modes - 0.1, kinetic_energy, width=0.2, label="Kinetic Energy", align='center')
ax.bar(modes + 0.1, potential_energy, width=0.2, label="Potential Energy", align='center')

ax.set_xlabel("Mode Number")
ax.set_ylabel("Energy (J)")
ax.set_title("Corrected Energy Distribution in Modal Analysis")
ax.legend()
ax.grid(True)

plt.show()

# Ensure modes is a NumPy array and properly structured as a 2D matrix
modes = np.array(eigvecs)  # Convert back to NumPy array if needed

if modes.ndim == 1:
    modes = modes.reshape((-1, 1))  # Reshape if necessary

# Normalize mode shapes for visualization
mode_shapes = modes / np.max(np.abs(modes), axis=0)

# Plot mode shapes
fig, ax = plt.subplots(figsize=(8, 5))
x_positions = np.array([0, 1])  # Positions of masses

for i in range(mode_shapes.shape[1]):
    ax.plot(x_positions, mode_shapes[:, i], marker='o', linestyle='-', label=f'Mode {i+1}')

ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel("Mass Position Index")
ax.set_ylabel("Normalized Displacement")
ax.set_title("Mode Shapes of the 2DOF System")
ax.legend()
ax.grid(True)

plt.show()