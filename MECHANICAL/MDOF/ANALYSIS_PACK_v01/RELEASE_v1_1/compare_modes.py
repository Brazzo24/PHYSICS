# compare_modes.py

import numpy as np
import matplotlib.pyplot as plt
import os

def compare_modal_energy(modal_energies1, modal_energies2, limit=5, N=None):
    """
    Compares the modal energy distributions (kinetic and potential) for two runs.
    
    Parameters:
      modal_energies1, modal_energies2: Lists of dictionaries (one per mode) from modal_energy_analysis.
      limit: Number of modes to compare (default 5).
      N: Number of DOFs. If None, it is inferred from the first run's kinetic energy array.
    """
    num_modes = min(limit, len(modal_energies1), len(modal_energies2))
    if N is None:
        N = len(modal_energies1[0]['T_dof'])
    
    for mode in range(num_modes):
        mode_data1 = modal_energies1[mode]
        mode_data2 = modal_energies2[mode]
        mode_idx = mode_data1['mode']
        freq_hz = mode_data1['omega_rad_s'] / (2 * np.pi)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # --- Kinetic Energy Comparison ---
        indices = np.arange(1, N+1)
        width = 0.35  # Bar width for side-by-side comparison
        axs[0].bar(indices - width/2, mode_data1['T_dof'], width, label='Run 1', alpha=0.7)
        axs[0].bar(indices + width/2, mode_data2['T_dof'], width, label='Run 2', alpha=0.7)
        axs[0].set_xlabel('DOF Index')
        axs[0].set_ylabel('Kinetic Energy [J]')
        axs[0].set_title(f'Kinetic Energy, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        axs[0].legend()
        axs[0].grid(True)
        
        # --- Potential Energy Comparison ---
        # Potential energy is defined for springs, so there are N-1 indices.
        indices_springs = np.arange(1, N)
        axs[1].bar(indices_springs - width/2, mode_data1['V_springs'], width, label='Run 1', alpha=0.7)
        axs[1].bar(indices_springs + width/2, mode_data2['V_springs'], width, label='Run 2', alpha=0.7)
        axs[1].set_xlabel('Spring Index')
        axs[1].set_ylabel('Potential Energy [J]')
        axs[1].set_title(f'Potential Energy, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        axs[1].legend()
        axs[1].grid(True)
        
        plt.tight_layout()
        os.makedirs("PLOTS", exist_ok=True)
        plt.savefig(f"PLOTS/modal_energy_comparison_mode_{mode_idx}.png")
        plt.show()
