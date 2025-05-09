import numpy as np
import matplotlib.pyplot as plt
from FDcalculations import free_vibration_analysis_free_chain, modal_energy_analysis

# Load baseline model (copy your values or import)
m_base = np.array([1.21e-2, 5e-3, 3.95e-4, 7.92e-4,
                   1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
                   2.73e-1, 2.69e+1])
k_base = np.array([1.0e4, 2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
                   2.72e4, 4.97e3, 7.73e2, 8.57e2])

# Sweep parameters
k_factors = np.linspace(0.4, 1.2, 10)   # multiply last 2 stiffnesses
m8_factors = np.linspace(0.6, 1.2, 10)  # scale inertia at DOF 8

freq_map = np.zeros((len(k_factors), len(m8_factors)))
ke_map = np.zeros_like(freq_map)

for i, kfac in enumerate(k_factors):
    for j, mfac in enumerate(m8_factors):
        m = m_base.copy()
        k = k_base.copy()

        m[9] *= mfac           # DOF 8 is index 9
        k[8] *= kfac           # Spring 7 → index 8
        k[9] *= kfac           # Spring 8 → index 9

        f_n, eigvecs, M, K = free_vibration_analysis_free_chain(m, k)
        modal_energies = modal_energy_analysis(m, k, f_n, eigvecs, M)

        freq_map[i, j] = f_n[2]  # Mode 3
        ke_map[i, j] = modal_energies[2]['T_dof'][9]  # KE at DOF 8

# Plot Frequency Heatmap
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(m8_factors, k_factors, freq_map, levels=20, cmap='viridis')
plt.colorbar(label='Mode 3 Frequency [Hz]')
plt.xlabel('Inertia Scaling (DOF 8)')
plt.ylabel('Stiffness Scaling (Springs 7 & 8)')
plt.title('Mode 3 Frequency Shift')

# Plot Kinetic Energy Heatmap
plt.subplot(1, 2, 2)
plt.contourf(m8_factors, k_factors, ke_map, levels=20, cmap='magma')
plt.colorbar(label='KE at DOF 8 [J]')
plt.xlabel('Inertia Scaling (DOF 8)')
plt.ylabel('Stiffness Scaling (Springs 7 & 8)')
plt.title('Kinetic Energy Concentration')

plt.tight_layout()
plt.show()
