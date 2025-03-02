# Re-import necessary libraries due to execution state reset
import numpy as np
import scipy.linalg
import pandas as pd
import matplotlib.pyplot as plt

# Vehicle Suspension System - Energy and Power Analysis

# System Parameters for a quarter-car model (2DOF system)
m_s = 300  # Sprung mass (kg) - chassis
m_u = 50   # Unsprung mass (kg) - wheel assembly
k_s = 15000  # Suspension stiffness (N/m)
k_t = 150000  # Tire stiffness (N/m)
c_s = 1000  # Suspension damping (Ns/m)

# Mass matrix
M_suspension = np.array([[m_s, 0],
                         [0, m_u]])

# Stiffness matrix
K_suspension = np.array([[k_s, -k_s],
                         [-k_s, k_s + k_t]])

# Damping matrix
C_suspension = np.array([[c_s, -c_s],
                         [-c_s, c_s]])

# Compute natural frequencies and mode shapes
eigvals_suspension, eigvecs_suspension = scipy.linalg.eigh(K_suspension, M_suspension)

# Natural frequencies (Hz)
omega_n_suspension = np.sqrt(eigvals_suspension) / (2 * np.pi)

# Modal kinetic and potential energy
KE_suspension = 0.5 * M_suspension @ eigvecs_suspension**2
PE_suspension = 0.5 * K_suspension @ eigvecs_suspension**2

# Power dissipation due to damping
P_damping_suspension = C_suspension @ eigvecs_suspension

# Prepare results
df_suspension_modes = pd.DataFrame({
    "Mode": [1, 2],
    "Natural Frequency (Hz)": omega_n_suspension,
    "Kinetic Energy (J)": np.sum(KE_suspension, axis=0),
    "Potential Energy (J)": np.sum(PE_suspension, axis=0),
    "Power Dissipation (W)": np.sum(P_damping_suspension, axis=0)
})

# Print results as a text table
print(df_suspension_modes.to_string(index=False))

# Visualization - Bar Plot for Energy Distribution
fig, ax = plt.subplots(figsize=(8, 5))
modes = df_suspension_modes["Mode"]
kinetic_energy = df_suspension_modes["Kinetic Energy (J)"]
potential_energy = df_suspension_modes["Potential Energy (J)"]

ax.bar(modes - 0.1, kinetic_energy, width=0.2, label="Kinetic Energy", align='center')
ax.bar(modes + 0.1, potential_energy, width=0.2, label="Potential Energy", align='center')

ax.set_xlabel("Mode Number")
ax.set_ylabel("Energy (J)")
ax.set_title("Energy Distribution in Suspension Modes")
ax.legend()
ax.grid(True)

plt.show()