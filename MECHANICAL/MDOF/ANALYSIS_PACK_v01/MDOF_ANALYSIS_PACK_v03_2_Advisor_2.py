import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig

# --------------------
# CONFIGURATION FLAGS
# --------------------
# Computation Flags
COMPUTE_FORCED_RESPONSE = True
COMPUTE_FREE_VIBRATION_ANALYSIS = True
COMPUTE_MODAL_ENERGY_ANALYSIS = True

# Plotting Flags (these now control whether an overview function is called)
PLOT_OVERVIEW_FORCED_RESPONSE = True
PLOT_EXCITATION_POWER = True
PLOT_POLES = True
PLOT_MODAL_ENERGY = True
PLOT_PHASE_ANGLES = True

# Limit for the number of modes to plot in modal energy overview
mode_plot_limit = 5

###############################################################################
# SYSTEM MATRIX BUILDING FUNCTIONS (Free-Chain, No Ground Connection)
###############################################################################
def build_free_chain_matrices(m, c_inter, k_inter):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N - 1):
        # Damping: affects DOF i and i+1
        C[i, i]     += c_inter[i]
        C[i, i+1]   -= c_inter[i]
        C[i+1, i]   -= c_inter[i]
        C[i+1, i+1] += c_inter[i]
        # Stiffness: affects DOF i and i+1
        K[i, i]     += k_inter[i]
        K[i, i+1]   -= k_inter[i]
        K[i+1, i]   -= k_inter[i]
        K[i+1, i+1] += k_inter[i]
    return M, C, K

# ADVISOR FUNCTION FOR OPTIMAL DAMPER PLACEMENT
def recommend_damper_location(f_vals, A_vals, P_damp, modal_energies, m, mode_frequencies, mode_number):
    print("\n======= Damper Placement Recommendation =======")
    target_mode_freq = mode_frequencies[mode_number - 1]
    print(f"Analyzing for Mode {mode_number} at {target_mode_freq:.2f} Hz")

    # --- 1) Modal Kinetic Energy distribution
    modal_kinetic_energy = modal_energies[mode_number - 1]['T_dof']
    print("\nModal Kinetic Energy Distribution (DOF-wise):")
    for i, val in enumerate(modal_kinetic_energy):
        print(f"DOF {i}: {val:.3e} J")

    # --- 2) Approximate relative velocity differences from mode shape
    phi_norm = modal_energies[mode_number - 1]['phi_norm']
    delta_velocity_estimates = np.abs(np.diff(phi_norm))

    print("\nRelative velocity differences (proxy for damper effectiveness):")
    for i, val in enumerate(delta_velocity_estimates):
        print(f"Between DOF {i} and {i+1}: {val:.3e}")

    # --- 3) Check forced response damping power around the mode frequency
    freq_idx = np.argmin(np.abs(f_vals - target_mode_freq))
    print("\nDamping power contributions at mode frequency:")
    for i in range(P_damp.shape[0]):
        power_at_mode = np.real(P_damp[i, freq_idx])
        print(f"Damper between DOF {i} and {i+1}: {power_at_mode:.3e} W")

    # --- 4) Combined ranking suggestion
    combined_score = delta_velocity_estimates * np.real(P_damp[:, freq_idx])

    best_location = np.argmax(combined_score)

    print("\n===== Recommended Damper Location =====")
    print(f"Place damper between DOF {best_location} and {best_location + 1}.")
    print(f"Reason: Highest combined score of modal velocity difference and damping power.")
    
    # Optional: Visualization
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(combined_score)), combined_score, alpha=0.7)
    plt.xlabel('Connection Index (between DOFs)')
    plt.ylabel('Combined Score')
    plt.title(f'Combined Damper Placement Score for Mode {mode_number} ({target_mode_freq:.2f} Hz)')
    plt.grid(True)
    plt.show()

###############################################################################
# AUGMENTED SYSTEM FORCED RESPONSE (Enforcing a Boundary Condition)
###############################################################################
def build_augmented_system(D, F_ext):
    N = D.shape[0]
    A_aug = np.zeros((N+1, N+1), dtype=complex)
    b_aug = np.zeros(N+1, dtype=complex)
    A_aug[0:N, 0:N] = D
    A_aug[N-1, N] = -1.0
    b_aug[0:N] = F_ext
    A_aug[N, N-1] = 1.0  # Enforce x_{N-1} = 0
    return A_aug, b_aug

def compute_forced_response_free_chain(m, c_inter, k_inter, f_vals, F_ext):
    N = len(m)
    num_points = len(f_vals)
    X = np.zeros((N, num_points), dtype=complex)
    F_bound = np.zeros(num_points, dtype=complex)
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    
    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        D = K + 1j*w*C - (w**2)*M
        A_aug, b_aug = build_augmented_system(D, F_ext)
        sol = np.linalg.solve(A_aug, b_aug)
        X[:, i] = sol[0:N]
        F_bound[i] = sol[N]
    return X, F_bound

###############################################################################
# POLE CALCULATION (State-Space Eigenvalues)
###############################################################################
def compute_poles_free_chain(m, c_inter, k_inter):
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    N = len(m)
    zero_block = np.zeros((N, N))
    I_block = np.eye(N)
    Minv = np.linalg.inv(M)
    
    A_upper = np.hstack((zero_block, I_block))
    A_lower = np.hstack((-Minv @ K, -Minv @ C))
    A = np.vstack((A_upper, A_lower))
    
    poles, _ = eig(A)
    return poles

###############################################################################
# FREE-VIBRATION ANALYSIS AND MODAL ENERGY DISTRIBUTION
###############################################################################
def free_vibration_analysis_free_chain(m, k_inter):
    N = len(m)
    M, _, K = build_free_chain_matrices(m, np.zeros(N-1), k_inter)
    eigvals, eigvecs = eigh(K, M)
    omega_n = np.sqrt(eigvals)
    f_n = omega_n / (2*np.pi)
    return f_n, eigvecs, M, K

def modal_energy_analysis(m, k_inter, f_n, eigvecs, M):
    N = len(m)
    modal_energies = []
    for i in range(eigvecs.shape[1]):
        phi = eigvecs[:, i]
        norm_factor = np.sqrt(np.real(np.conjugate(phi).T @ M @ phi))
        phi_norm = phi / norm_factor
        omega_i = 2*np.pi*f_n[i]
        T_dof = 1.0 * m * (omega_i * np.abs(phi_norm))**2
        V_springs = np.zeros(N-1)
        for s in range(N-1):
            V_springs[s] = 0.5 * k_inter[s] * (np.abs(phi_norm[s+1] - phi_norm[s]))**2
        modal_energies.append({
            'mode': i+1,
            'omega_rad_s': omega_i,
            'T_total': np.sum(T_dof),
            'V_total': np.sum(V_springs),
            'T_dof': T_dof,
            'V_springs': V_springs,
            'phi_norm': phi_norm
        })
    return modal_energies

###############################################################################
# FORCED RESPONSE POST-PROCESSING
###############################################################################
def forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext):
    """
    Computes the forced response using the augmented formulation and then derives:
      - Displacement (X_vals)
      - Acceleration (A_vals)
      - Active power in dampers (P_damp)
      - Reactive power in springs (P_spring)
      - Inertial reactive power (Q_mass)
      - Reaction force at boundary (F_bound)
      - NEW: Phase angles of displacement (phase_vals)
    """
    N = len(m)
    num_points = len(f_vals)
    
    X_vals = np.zeros((N, num_points), dtype=complex)
    A_vals = np.zeros((N, num_points), dtype=complex)
    P_damp = np.zeros((N-1, num_points), dtype=complex)
    P_spring = np.zeros((N-1, num_points), dtype=complex)
    Q_mass = np.zeros((num_points, N))
    F_bound_array = np.zeros(num_points, dtype=complex)
    
    # Build free-chain matrices
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    
    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        # Dynamic stiffness for free chain
        D = K + 1j*w*C - (w**2)*M
        # Augmented system to enforce x_{N-1} = 0
        A_aug, b_aug = build_augmented_system(D, F_ext)
        sol = np.linalg.solve(A_aug, b_aug)
        
        X = sol[0:N]  # Displacement solution
        F_bound_array[i] = sol[N]
        X_vals[:, i] = X
        
        # Acceleration (a = -w^2*x)
        A_vals[:, i] = -w**2 * X
        
        # Velocity for power calculations: V = j*w*x
        V = 1j * w * X
        
        # Damper and spring power
        for j in range(1, N):
            X_rel = X[j] - X[j-1]
            V_rel = V[j] - V[j-1]
            # Active power in dampers
            P_damp[j-1, i] = c_inter[j-1] * (V_rel * np.conjugate(V_rel))
            # Reactive power in springs
            P_spring[j-1, i] = k_inter[j-1] * (X_rel * np.conjugate(V_rel))
        
        # Inertial reactive power per mass (peak amplitude convention)
        Q_mass[i, :] = 1.0 * w * m * (np.abs(V)**2)
    
    # NEW: Phase angle of each DOF's displacement (relative to a real forcing)
    phase_vals = np.angle(X_vals)  # shape (N, num_points)    

    return {
        'X_vals': X_vals,
        'A_vals': A_vals,
        'P_damp': P_damp,
        'P_spring': P_spring,
        'Q_mass': Q_mass,
        'F_bound': F_bound_array,
        'phase_vals': phase_vals,  # ADDED
    }


###############################################################################
# PLOTTING FUNCTIONS (Overview Figures)
###############################################################################
def plot_forced_response_overview(f_vals, X_vals, A_vals, P_damp, P_spring, Q_mass, F_bound, m):
    N = X_vals.shape[0]
    # N = mode_plot_limit
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Displacement Response
    # Excitation Power
    V_exc = 1j * (2 * np.pi * f_vals) * X_vals[0, :]
    P_exc = 1.0 * np.conjugate(V_exc)  # F_ext[0] assumed 1.0
    axs[0, 0].plot(f_vals, np.real(P_exc), label='Active Power (Excitation)')
    axs[0, 0].plot(f_vals, np.imag(P_exc), label='Reactive Power (Excitation)')
    axs[0, 0].set_xlabel('Frequency [Hz]')
    axs[0, 0].set_ylabel('Power [W][VAR]')
    axs[0, 0].set_title('Input Power')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    for i in range(N):
        axs[0, 1].plot(f_vals, np.abs(X_vals[i, :]), label=f'|x_{i}|')
    axs[0, 1].set_xlabel('Frequency [Hz]')
    axs[0, 1].set_ylabel('Displacement [rad]')
    axs[0, 1].set_title('Displacement Response')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Acceleration Response
    for i in range(N):
        axs[0, 2].plot(f_vals, np.abs(A_vals[i, :]), label=f'|a_{i}|')
    axs[0, 2].set_xlabel('Frequency [Hz]')
    axs[0, 2].set_ylabel('Acceleration [rad/s^2]')
    axs[0, 2].set_title('Acceleration Response')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # Active Power in Dampers
    for j in range(P_damp.shape[0]):
        axs[1, 0].plot(f_vals, np.real(P_damp[j, :]), label=f'Damper {j}')
    axs[1, 0].set_xlabel('Frequency [Hz]')
    axs[1, 0].set_ylabel('Active Power [W]')
    axs[1, 0].set_title('Active Power in Dampers')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Reactive Power in Springs
    for j in range(P_spring.shape[0]):
        axs[1, 1].plot(f_vals, np.imag(P_spring[j, :]), label=f'Spring {j}')
    axs[1, 1].set_xlabel('Frequency [Hz]')
    axs[1, 1].set_ylabel('Reactive Power [VAR]')
    axs[1, 1].set_title('Reactive Power in Springs')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Inertial Reactive Power per Mass
    for i in range(len(m)):
        axs[1, 2].plot(f_vals, Q_mass[:, i], label=f'Mass {i}')
    axs[1, 2].set_xlabel('Frequency [Hz]')
    axs[1, 2].set_ylabel('Reactive Power [VAR]')
    axs[1, 2].set_title('Inertial Reactive Power per Mass')
    axs[1, 2].legend()
    axs[1, 2].grid(True)
    
    # # Boundary Reaction Force
    # axs[2, 1].plot(f_vals, np.abs(F_bound), 'r', label='|F_bound|')
    # axs[2, 1].set_xlabel('Frequency [Hz]')
    # axs[2, 1].set_ylabel('Reaction Force [N]')
    # axs[2, 1].set_title('Boundary Reaction Force')
    # axs[2, 1].legend()
    # axs[2, 1].grid(True)
    



    plt.tight_layout()
    plt.show()

def plot_phase_relationships(f_vals, phase_vals):
    """
    Plot the phase angle (in degrees) for each DOF's displacement 
    relative to the forcing (which is assumed real).
    """
    N = phase_vals.shape[0] - 1
    # N = mode_plot_limit

    plt.figure(figsize=(9, 5))
    for i in range(N):
        # Convert to degrees for easier interpretation
        phase_degrees = np.rad2deg(phase_vals[i, :])
        plt.plot(f_vals, phase_degrees, label=f'Phase DOF {i}')
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [degrees]')
    plt.title('Phase Relationship of Each DOF relative to Excitation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_phase_relationships_dof_reference(f_vals, phase_vals):
    """
    Plot the phase of each DOF relative to DOF 0 (i.e., the difference in angles).
    """
    N = phase_vals.shape[0] - 1
    reference_phase = phase_vals[0, :]  # DOF 0
    plt.figure(figsize=(9, 5))
    for i in range(N):
        # Phase difference: (phase_i - phase_0)
        phase_diff = phase_vals[i, :] - reference_phase
        phase_degrees = np.rad2deg(phase_diff)
        plt.plot(f_vals, phase_degrees, label=f'Phase DOF {i} - DOF 0')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase difference [degrees]')
    plt.title('Phase Relationship of Each DOF relative to DOF 0')
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot_excitation_power(f_vals, X_vals):
#     V_exc = 1j * (2 * np.pi * f_vals) * X_vals[0, :]
#     P_exc = 1.0 * np.conjugate(V_exc)  # F_ext[0] assumed 1.0
#     plt.figure(figsize=(9, 5))
#     plt.plot(f_vals, np.real(P_exc), label='Active Power (Excitation)')
#     plt.plot(f_vals, np.imag(P_exc), label='Reactive Power (Excitation)')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Power')
#     plt.title('Excitation Signal Power at DOF 0')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot_poles_overview(poles):
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(poles), np.imag(poles), color='b', s=50)
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.title('Poles (State-Space Eigenvalues)')
    plt.grid(True)
    plt.show()

def plot_modal_energy_overview(modal_energies, limit, N):
    for mode_data in modal_energies[:min(limit, len(modal_energies))]:
        mode_idx = mode_data['mode']
        freq_hz = mode_data['omega_rad_s'] / (2 * np.pi)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        # Kinetic Energy Distribution
        axs[0].bar(np.arange(1, N+1), mode_data['T_dof'], color='g', alpha=0.7)
        axs[0].set_xlabel('DOF Index')
        axs[0].set_ylabel('Kinetic Energy [J]')
        axs[0].set_title(f'Kinetic Energy, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        axs[0].grid(True)
        # Potential Energy Distribution
        axs[1].bar(np.arange(1, N), mode_data['V_springs'], color='r', alpha=0.7)
        axs[1].set_xlabel('Spring Index')
        axs[1].set_ylabel('Potential Energy [J]')
        axs[1].set_title(f'Potential Energy, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        axs[1].grid(True)
        plt.tight_layout()
        plt.show()



###############################################################################
# MAIN SCRIPT
###############################################################################
def main():
    # ------------------------------
    # User-defined System Parameters
    # ------------------------------
    # Example: Reduced Crankshaft Model
    m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
                  1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
                  2.73e-1, 2.69e+1])  # kgm^2


    # ([Crankshaft, CRCS, PG, Clutch 1, Clutch 2, Input, Output, Hub, Wheel, Road])
    # ([])
    c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nm.s/rad

    k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
                        2.72e4, 4.97e3, 7.73e2, 8.57e2]) # Nm/rad
    

    # CUSTOM INPUT
    # m = np.array([2.0e-0, 1.0e-0, 100])  # kg*m^2

    # c_inter = np.array([20.0, 800.0
    #                 ])  # Nm/rad.s

    # k_inter = np.array([1.0e5, 3.5e4])

    N = len(m)
    
    # External force vector: force applied at DOF 0.
    F_ext = np.zeros(N, dtype=complex)
    F_ext[0] = 1.0
    
    # Frequency range for forced response analysis.
    f_min, f_max = 0.1, 400.0
    num_points = 10000
    f_vals = np.linspace(f_min, f_max, num_points)
    
    # ------------------------------
    # Forced Response and Post-Processing
    # ------------------------------
    if COMPUTE_FORCED_RESPONSE:
        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)
        X_vals = response['X_vals']
        A_vals = response['A_vals']
        P_damp = response['P_damp']
        P_spring = response['P_spring']
        Q_mass = response['Q_mass']
        F_bound = response['F_bound']
        phase_vals = response['phase_vals']
        
        if PLOT_OVERVIEW_FORCED_RESPONSE:
            plot_forced_response_overview(f_vals, X_vals, A_vals, P_damp, P_spring, Q_mass, F_bound, m)
        
        # if PLOT_EXCITATION_POWER:
        #     plot_excitation_power(f_vals, X_vals)

        if PLOT_PHASE_ANGLES:
            plot_phase_relationships(f_vals, phase_vals)
    
    # ------------------------------
    # Free-Vibration Analysis and Modal Energy Distribution
    # ------------------------------
    if COMPUTE_FREE_VIBRATION_ANALYSIS or COMPUTE_MODAL_ENERGY_ANALYSIS:
        f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k_inter)
        print("\nUndamped Natural Frequencies (Hz):")
        for i, fn in enumerate(f_n):
            print(f"  Mode {i+1}: {fn:.3f} Hz")
            
        if COMPUTE_MODAL_ENERGY_ANALYSIS:
            modal_energies = modal_energy_analysis(m, k_inter, f_n, eigvecs, M_free)
            if PLOT_MODAL_ENERGY:
                plot_modal_energy_overview(modal_energies, mode_plot_limit, N)
    
    # ------------------------------
    # Poles (State-Space Eigenvalues)
    # ------------------------------

    if PLOT_POLES:
        poles = compute_poles_free_chain(m, c_inter, k_inter)
        plot_poles_overview(poles)

    
    if COMPUTE_MODAL_ENERGY_ANALYSIS and COMPUTE_FORCED_RESPONSE:
        user_mode_number = int(input("\nEnter the mode number you are concerned about: "))
        recommend_damper_location(f_vals, A_vals, P_damp, modal_energies, m, f_n, user_mode_number)
        print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
