import matplotlib.pyplot as plt
import numpy as np
import os

def save_current_figure(filename):
    """Creates the PLOTS directory (if it doesn't exist) and saves the current figure."""
    plot_dir = "PLOTS"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

def plot_forced_response_overview(f_vals, X_vals, V_vals, A_vals, P_damp, P_spring, Q_mass, F_bound, m):
    N = X_vals.shape[0]
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # --- Input Power Plot ---
    V_exc = 1j * (2 * np.pi * f_vals) * X_vals[0, :]
    P_exc = 1.0 * np.conjugate(V_exc)  # F_ext[0] assumed 1.0
    axs[0, 0].plot(f_vals, np.real(P_exc), label='Active Power (Excitation)')
    axs[0, 0].plot(f_vals, np.imag(P_exc), label='Reactive Power (Excitation)')
    axs[0, 0].set_xlabel('Frequency [Hz]')
    axs[0, 0].set_ylabel('Power [W][VAR]')
    axs[0, 0].set_title('Input Power')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # # --- Displacement Response ---
    # for i in range(N):
    #     axs[0, 1].plot(f_vals, np.abs(X_vals[i, :]), label=f'|x_{i}|')
    # axs[0, 1].set_xlabel('Frequency [Hz]')
    # axs[0, 1].set_ylabel('Displacement [rad]')
    # axs[0, 1].set_title('Displacement Response')
    # axs[0, 1].legend()
    # axs[0, 1].grid(True)

    # --- Velocity Response ---
    for i in range(N):
        axs[0, 1].plot(f_vals, np.abs(V_vals[i, :]), label=f'|v_{i}|')
    axs[0, 1].set_xlabel('Frequency [Hz]')
    axs[0, 1].set_ylabel('Velocity [rad/s]')
    axs[0, 1].set_title('Velocity Response')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # --- Acceleration Response ---
    for i in range(N):
        axs[0, 2].plot(f_vals, np.abs(A_vals[i, :]), label=f'|a_{i}|')
    axs[0, 2].set_xlabel('Frequency [Hz]')
    axs[0, 2].set_ylabel('Acceleration [rad/s^2]')
    axs[0, 2].set_title('Acceleration Response')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # --- Active Power in Dampers ---
    for j in range(P_damp.shape[0]):
        axs[1, 0].plot(f_vals, np.real(P_damp[j, :]), label=f'Damper {j}')
    axs[1, 0].set_xlabel('Frequency [Hz]')
    axs[1, 0].set_ylabel('Active Power [W]')
    axs[1, 0].set_title('Active Power in Dampers')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # --- Reactive Power in Springs ---
    for j in range(P_spring.shape[0]):
        axs[1, 1].plot(f_vals, np.imag(P_spring[j, :]), label=f'Spring {j}')
    axs[1, 1].set_xlabel('Frequency [Hz]')
    axs[1, 1].set_ylabel('Reactive Power [VAR]')
    axs[1, 1].set_title('Reactive Power in Springs')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # --- Inertial Reactive Power per Mass ---
    for i in range(len(m)):
        axs[1, 2].plot(f_vals, Q_mass[:, i], label=f'Mass {i}')
    axs[1, 2].set_xlabel('Frequency [Hz]')
    axs[1, 2].set_ylabel('Reactive Power [VAR]')
    axs[1, 2].set_title('Inertial Reactive Power per Mass')
    axs[1, 2].legend()
    axs[1, 2].grid(True)
    
    plt.tight_layout()
    save_current_figure("forced_response_overview.png")
    plt.show()

def plot_phase_relationships(f_vals, phase_vals):
    """
    Plot the phase angle (in degrees) for each DOF's displacement 
    relative to the forcing (which is assumed real).
    """
    N = phase_vals.shape[0] - 1
    plt.figure(figsize=(9, 5))
    for i in range(N):
        phase_degrees = np.rad2deg(phase_vals[i, :])
        plt.plot(f_vals, phase_degrees, label=f'Phase DOF {i}')
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [degrees]')
    plt.title('Phase Relationship of Each DOF relative to Excitation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_current_figure("phase_relationships.png")
    plt.show()

def plot_phase_relationships_dof_reference(f_vals, phase_vals):
    """
    Plot the phase of each DOF relative to DOF 0 (i.e., the difference in angles).
    """
    N = phase_vals.shape[0] - 1
    reference_phase = phase_vals[0, :]  # DOF 0
    plt.figure(figsize=(9, 5))
    for i in range(N):
        phase_diff = phase_vals[i, :] - reference_phase
        phase_degrees = np.rad2deg(phase_diff)
        plt.plot(f_vals, phase_degrees, label=f'Phase DOF {i} - DOF 0')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase difference [degrees]')
    plt.title('Phase Relationship of Each DOF relative to DOF 0')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_current_figure("phase_relationships_dof_reference.png")
    plt.show()

def plot_poles_overview(poles):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(poles), np.imag(poles), s=50)
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title('Poles (State-Space Eigenvalues)')
    plt.grid(True)
    plt.tight_layout()
    save_current_figure("poles_overview.png")
    plt.show()

def plot_modal_energy_overview(modal_energies, limit, N):
    for mode_data in modal_energies[:min(limit, len(modal_energies))]:
        mode_idx = mode_data['mode']
        freq_hz = mode_data['omega_rad_s'] / (2 * np.pi)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # Kinetic Energy Distribution
        axs[0].bar(np.arange(N), mode_data['T_dof'], alpha=0.7)
        axs[0].set_xlabel('DOF Index')
        axs[0].set_ylabel('Kinetic Energy [J]')
        axs[0].set_title(f'Kinetic Energy, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        axs[0].grid(True)
        # Potential Energy Distribution
        axs[1].bar(np.arange(N - 1), mode_data['V_springs'], alpha=0.7)
        axs[1].set_xlabel('Spring Index')
        axs[1].set_ylabel('Potential Energy [J]')
        axs[1].set_title(f'Potential Energy, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        axs[1].grid(True)
        plt.tight_layout()
        save_current_figure(f"modal_energy_overview_mode_{mode_idx}.png")
        plt.show()
