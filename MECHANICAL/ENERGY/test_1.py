# Rewriting the code to use Pandas for visualization instead of ace_tools

# Re-import necessary libraries
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

# Mode shapes (normalized)
modes = eigvecs

# Compute modal kinetic and potential energy
KE_modes = 0.5 * M @ modes**2
PE_modes = 0.5 * K @ modes**2

# Compute power dissipation due to damping
P_damping = C @ modes

# Create a DataFrame for results
df_modes = pd.DataFrame({
    "Mode": [1, 2],
    "Natural Frequency (Hz)": omega_n,
    "Kinetic Energy (J)": np.sum(KE_modes, axis=0),
    "Potential Energy (J)": np.sum(PE_modes, axis=0),
    "Power Dissipation (W)": np.sum(P_damping, axis=0)
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
ax.set_title("Energy Distribution in Modal Analysis")
ax.legend()
ax.grid(True)

plt.show()