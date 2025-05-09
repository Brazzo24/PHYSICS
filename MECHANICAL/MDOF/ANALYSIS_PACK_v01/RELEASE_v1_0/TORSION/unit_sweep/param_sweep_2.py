import numpy as np
import matplotlib.pyplot as plt
from FDcalculations import free_vibration_analysis_free_chain, modal_energy_analysis

# Baseline model
m_base = np.array([1.21e-2, 5e-3, 3.95e-4, 7.92e-4,
                   1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
                   2.73e-1, 2.69e+1])
k_base = np.array([1.0e4, 2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
                   2.72e4, 4.97e3, 7.73e2, 8.57e2])

# Sweep ranges
k_tail_factors = np.linspace(0.8, 1.5, 10)  # for k[8], k[9]
m_tail_factors = np.linspace(0.8, 1.1, 10)  # for m[9] (DOF 8)

mode2_freq = np.zeros((len(k_tail_factors), len(m_tail_factors)))
mode3_freq = np.zeros_like(mode2_freq)
ke_mode2_dof8 = np.zeros_like(mode2_freq)
ke_mode3_dof8 = np.zeros_like(mode2_freq)

for i, kfac in enumerate(k_tail_factors):
    for j, mfac in enumerate(m_tail_factors):
        m = m_base.copy()
        k = k_base.copy()

        m[9] *= mfac            # DOF 8
        k[8] *= kfac            # Spring 7 → index 8
        k[9] *= kfac            # Spring 8 → index 9

        f_n, eigvecs, M, K = free_vibration_analysis_free_chain(m, k)
        modal_energies = modal_energy_analysis(m, k, f_n, eigvecs, M)

        mode2_freq[i, j] = f_n[1]  # Mode 2
        mode3_freq[i, j] = f_n[2]  # Mode 3
        ke_mode2_dof8[i, j] = modal_energies[1]['T_dof'][9]  # DOF 8
        ke_mode3_dof8[i, j] = modal_energies[2]['T_dof'][9]  # DOF 8

# --- Plotting ---
k_labels = np.round(k_tail_factors, 2)
m_labels = np.round(m_tail_factors, 2)

plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.contourf(m_tail_factors, k_tail_factors, mode2_freq, levels=20, cmap='viridis')
plt.colorbar(label='Mode 2 Frequency [Hz]')
plt.title('Mode 2 Frequency')
plt.xlabel('Inertia Factor (DOF 8)')
plt.ylabel('Stiffness Factor (Springs 8-9)')

plt.subplot(2, 2, 2)
plt.contourf(m_tail_factors, k_tail_factors, ke_mode2_dof8, levels=20, cmap='magma')
plt.colorbar(label='KE at DOF 8 [J]')
plt.title('Mode 2: KE Concentration at DOF 8')
plt.xlabel('Inertia Factor (DOF 8)')
plt.ylabel('Stiffness Factor (Springs 8-9)')

plt.subplot(2, 2, 3)
plt.contourf(m_tail_factors, k_tail_factors, mode3_freq, levels=20, cmap='plasma')
plt.colorbar(label='Mode 3 Frequency [Hz]')
plt.title('Mode 3 Frequency')
plt.xlabel('Inertia Factor (DOF 8)')
plt.ylabel('Stiffness Factor (Springs 8-9)')

plt.subplot(2, 2, 4)
plt.contourf(m_tail_factors, k_tail_factors, ke_mode3_dof8, levels=20, cmap='cividis')
plt.colorbar(label='KE at DOF 8 [J]')
plt.title('Mode 3: KE Concentration at DOF 8')
plt.xlabel('Inertia Factor (DOF 8)')
plt.ylabel('Stiffness Factor (Springs 8-9)')

plt.tight_layout()
plt.show()
