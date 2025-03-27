# comparison.py (or add this function to your existing comparison module)

import matplotlib.pyplot as plt
import numpy as np
import os

def compare_forced_response_overview(forced_response1, forced_response2, f_vals, m):
    """
    Creates a 2x3 plot comparing the forced response from two runs.
    
    Parameters:
      forced_response1, forced_response2: dictionaries returned by forced_response_postprocessing
      f_vals: frequency vector used in the simulations.
      m: mass array (used for the inertial reactive power plot).
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # --- Subplot (0,0): Input Power ---
    # For each run, compute the excitation velocity and power.
    V_exc1 = 1j * (2 * np.pi * f_vals) * forced_response1['X_vals'][0, :]
    P_exc1 = 1.0 * np.conjugate(V_exc1)
    
    V_exc2 = 1j * (2 * np.pi * f_vals) * forced_response2['X_vals'][0, :]
    P_exc2 = 1.0 * np.conjugate(V_exc2)
    
    axs[0, 0].plot(f_vals, np.real(P_exc1), label='Run 1 Active')
    axs[0, 0].plot(f_vals, np.imag(P_exc1), label='Run 1 Reactive')
    axs[0, 0].plot(f_vals, np.real(P_exc2), '--', label='Run 2 Active')
    axs[0, 0].plot(f_vals, np.imag(P_exc2), '--', label='Run 2 Reactive')
    axs[0, 0].set_xlabel('Frequency [Hz]')
    axs[0, 0].set_ylabel('Power [W/VAR]')
    axs[0, 0].set_title('Input Power')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # --- Subplot (0,1): Displacement Response ---
    N = forced_response1['X_vals'].shape[0]
    for i in range(N):
        axs[0, 1].plot(f_vals, np.abs(forced_response1['X_vals'][i, :]),
                       label=f'Run 1: |x_{i}|', alpha=0.7)
        axs[0, 1].plot(f_vals, np.abs(forced_response2['X_vals'][i, :]),
                       '--', label=f'Run 2: |x_{i}|', alpha=0.7)
    axs[0, 1].set_xlabel('Frequency [Hz]')
    axs[0, 1].set_ylabel('Displacement [rad]')
    axs[0, 1].set_title('Displacement Response')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # --- Subplot (0,2): Acceleration Response ---
    for i in range(N):
        axs[0, 2].plot(f_vals, np.abs(forced_response1['A_vals'][i, :]),
                       label=f'Run 1: |a_{i}|', alpha=0.7)
        axs[0, 2].plot(f_vals, np.abs(forced_response2['A_vals'][i, :]),
                       '--', label=f'Run 2: |a_{i}|', alpha=0.7)
    axs[0, 2].set_xlabel('Frequency [Hz]')
    axs[0, 2].set_ylabel('Acceleration [rad/s²]')
    axs[0, 2].set_title('Acceleration Response')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # --- Subplot (1,0): Active Power in Dampers ---
    num_dampers = forced_response1['P_damp'].shape[0]
    for j in range(num_dampers):
        axs[1, 0].plot(f_vals, np.real(forced_response1['P_damp'][j, :]),
                       label=f'Run 1: Damper {j}', alpha=0.7)
        axs[1, 0].plot(f_vals, np.real(forced_response2['P_damp'][j, :]),
                       '--', label=f'Run 2: Damper {j}', alpha=0.7)
    axs[1, 0].set_xlabel('Frequency [Hz]')
    axs[1, 0].set_ylabel('Active Power [W]')
    axs[1, 0].set_title('Active Power in Dampers')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # --- Subplot (1,1): Reactive Power in Springs ---
    num_springs = forced_response1['P_spring'].shape[0]
    for j in range(num_springs):
        axs[1, 1].plot(f_vals, np.imag(forced_response1['P_spring'][j, :]),
                       label=f'Run 1: Spring {j}', alpha=0.7)
        axs[1, 1].plot(f_vals, np.imag(forced_response2['P_spring'][j, :]),
                       '--', label=f'Run 2: Spring {j}', alpha=0.7)
    axs[1, 1].set_xlabel('Frequency [Hz]')
    axs[1, 1].set_ylabel('Reactive Power [VAR]')
    axs[1, 1].set_title('Reactive Power in Springs')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # --- Subplot (1,2): Inertial Reactive Power per Mass ---
    for i in range(len(m)):
        axs[1, 2].plot(f_vals, forced_response1['Q_mass'][:, i],
                       label=f'Run 1: Mass {i}', alpha=0.7)
        axs[1, 2].plot(f_vals, forced_response2['Q_mass'][:, i],
                       '--', label=f'Run 2: Mass {i}', alpha=0.7)
    axs[1, 2].set_xlabel('Frequency [Hz]')
    axs[1, 2].set_ylabel('Reactive Power [VAR]')
    axs[1, 2].set_title('Inertial Reactive Power per Mass')
    axs[1, 2].legend()
    axs[1, 2].grid(True)
    
    plt.tight_layout()
    # Automatically create the PLOTS directory and save the figure.
    os.makedirs("PLOTS", exist_ok=True)
    plt.savefig("PLOTS/forced_response_comparison_overview.png")
    plt.show()
