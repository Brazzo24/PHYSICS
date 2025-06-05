import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig

# --------------------
# CONFIGURATION FLAGSimport numpy as np
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
PLOT_PHASE_ANGLES = False
PLOT_EXCITATION_POWER = True
PLOT_POLES = False
PLOT_MODAL_ENERGY = False


# Limit for the number of modes to plot in modal energy overview
mode_plot_limit = 5

###############################################################################
# SYSTEM MATRIX BUILDING FUNCTIONS (Free-Chain, No Ground Connection)
###############################################################################
def build_free_chain_matrices(m, c_inter, k_inter, C_ext=None, K_ext=None):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(len(k_inter)):
        C[i, i]     += c_inter[i]
        C[i, i+1]   -= c_inter[i]
        C[i+1, i]   -= c_inter[i]
        C[i+1, i+1] += c_inter[i]

        K[i, i]     += k_inter[i]
        K[i, i+1]   -= k_inter[i]
        K[i+1, i]   -= k_inter[i]
        K[i+1, i+1] += k_inter[i]

    if C_ext is not None:
        C += C_ext
    if K_ext is not None:
        K += K_ext

    return M, C, K



###############################################################################
# AUGMENTED SYSTEM FORCED RESPONSE (Enforcing a Boundary Condition)
###############################################################################
def build_augmented_system(D, F_ext, fixed_dof):
    N = D.shape[0]
    A_aug = np.zeros((N+1, N+1), dtype=complex)
    b_aug = np.zeros(N+1, dtype=complex)
    A_aug[0:N, 0:N] = D
    A_aug[fixed_dof, N] = -1.0
    b_aug[0:N] = F_ext
    A_aug[N, fixed_dof] = 1.0  # Enforce x_{N-1} = 0
    return A_aug, b_aug

def augment_with_branch(m, c_inter, k_inter, connection, m_branch, c_branch, k_branch, c_to_ground=0.0):
    i, j = connection  # i = existing DOF, j = new DOF index

    m_aug = np.append(m, m_branch)
    c_aug = np.append(c_inter, 0.0)  # New DOF
    k_aug = np.append(k_inter, 0.0)

    N = len(m_aug)
    C_ext = np.zeros((N, N))
    K_ext = np.zeros((N, N))

    # Add spring-damper between DOF i and new DOF j (existing branch)
    C_ext[i, i] += c_branch
    C_ext[i, j] -= c_branch
    C_ext[j, i] -= c_branch
    C_ext[j, j] += c_branch

    K_ext[i, i] += k_branch
    K_ext[i, j] -= k_branch
    K_ext[j, i] -= k_branch
    K_ext[j, j] += k_branch

    # Add damper from DOF j to ground (optional energy sink)
    if c_to_ground > 0.0:
        C_ext[j, j] += c_to_ground

    return m_aug, c_aug, k_aug, C_ext, K_ext



def compute_forced_response_free_chain(m, c_inter, k_inter, f_vals, F_ext):
    N = len(m)
    num_points = len(f_vals)
    X = np.zeros((N, num_points), dtype=complex)
    F_bound = np.zeros(num_points, dtype=complex)
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    
    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        D = K + 1j*w*C - (w**2)*M
        A_aug, b_aug = build_augmented_system(D, F_ext, fixed_dof=9)
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
def free_vibration_analysis_free_chain(m, k_inter, C_ext=None, K_ext=None):
    N = len(m)
    M, _, K = build_free_chain_matrices(m, np.zeros(N-1), k_inter)

    # If external stiffness matrix (e.g., from a branch) is provided, add it
    if K_ext is not None:
        K += K_ext

    eigvals, eigvecs = eigh(K, M)
    omega_n = np.sqrt(np.maximum(eigvals, 0))
    f_n = omega_n / (2 * np.pi)
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
def forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext, C_ext=None, K_ext=None):
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
    V_vals = np.zeros((N, num_points), dtype=complex)
    P_damp = np.zeros((N-1, num_points), dtype=complex)
    P_spring = np.zeros((N-1, num_points), dtype=complex)
    Q_mass = np.zeros((num_points, N))
    F_bound_array = np.zeros(num_points, dtype=complex)
    
    # Build free-chain matrices
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter, C_ext=C_ext, K_ext=K_ext)
    
    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        # Dynamic stiffness for free chain
        D = K + 1j*w*C - (w**2)*M
        # Augmented system to enforce x_{N-1} = 0
        A_aug, b_aug = build_augmented_system(D, F_ext, fixed_dof=9)  # or 9, or any grounded point

        sol = np.linalg.solve(A_aug, b_aug)
        
        X = sol[0:N]  # Displacement solution
        F_bound_array[i] = sol[N]
        X_vals[:, i] = X
        
        # Acceleration (a = -w^2*x)
        A_vals[:, i] = -w**2 * X
        
        # Velocity (v = w * X)
        V_vals[:, i] = w * X

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
        'V_vals': V_vals,
        'P_damp': P_damp,
        'P_spring': P_spring,
        'Q_mass': Q_mass,
        'F_bound': F_bound_array,
        'phase_vals': phase_vals,  # ADDED
    }



def compute_mode_interaction_force(mode_idx, dof_primary, dof_secondary, eigvecs, N, mode='cancel'):
    """
    Compute a force vector to either cancel or reinforce a specific mode using two excitations.

    Parameters:
    - mode_idx: index of the mode to target (0-based)
    - dof_primary: primary DOF for initial excitation
    - dof_secondary: secondary DOF for interaction (cancellation or reinforcement)
    - eigvecs: eigenvector matrix from modal analysis
    - N: total number of DOFs
    - mode: 'cancel' or 'reinforce'

    Returns:
    - F_ext: complex force vector for interaction
    """
    phi_mode = eigvecs[:, mode_idx]
    phi_mode /= np.max(np.abs(phi_mode))  # Normalize mode shape

    F_ext = np.zeros(N, dtype=complex)
    F_ext[dof_primary] = 1.0 + 0j

    ratio = phi_mode[dof_primary] / phi_mode[dof_secondary]
    if mode == 'cancel':
        F_ext[dof_secondary] = -ratio
    elif mode == 'reinforce':
        F_ext[dof_secondary] = +ratio
    else:
        raise ValueError("Mode must be 'cancel' or 'reinforce'")

    print(f"\nForce Vector to {mode.capitalize()} Mode {mode_idx + 1}:")
    print(f"  DOF {dof_primary}: {F_ext[dof_primary]}")
    print(f"  DOF {dof_secondary}: {F_ext[dof_secondary]}")
    return F_ext

def compute_power_flow(f_vals, X_vals, k_inter, m, omega_idx=None):
    """
    Computes energy flow across each spring at a specific frequency index.

    Parameters:
        f_vals (array): Frequency values [Hz]
        X_vals (array): Complex displacement response, shape (N_dof, N_freq)
        k_inter (array): Stiffness between DOFs, length N-1
        m (array): Mass/inertia at each DOF
        omega_idx (int): Optional, which frequency index to use (default = peak near 24Hz)
    
    Returns:
        P_spring_complex: Power flow (complex) per spring connection [Nm/s]
        power_map: Dictionary with real and imaginary components
    """
    N = X_vals.shape[0]
    if omega_idx is None:
        # Auto-detect peak near 24 Hz
        f_target = 24
        omega_idx = np.argmin(np.abs(f_vals - f_target))
    
    omega = 2 * np.pi * f_vals[omega_idx]
    X = X_vals[:, omega_idx]

    P_spring_complex = np.zeros(N - 1, dtype=complex)

    for i in range(N - 1):
        delta_theta = X[i] - X[i + 1]
        delta_omega_theta = 1j * omega * delta_theta
        P_spring_complex[i] = 0.5 * k_inter[i] * (delta_theta * np.conj(delta_omega_theta))

    power_map = {
        'real': np.real(P_spring_complex),   # energy exchange (elastic)
        'imag': np.imag(P_spring_complex),   # reactive flow
        'abs': np.abs(P_spring_complex)
    }

    return P_spring_complex, power_map

def sweep_energy_vs_stiffness_damping(m, k_inter, c_base, dof_idx, mode_idx, k_range, c_levels):
    results = []

    for c_scale in c_levels:
        energies = []
        for k_scale in k_range:
            # Scale stiffness and damping
            k_scaled = k_inter * k_scale
            c_scaled = c_base * c_scale

            # Recompute eigenvalues
            f_n, eigvecs, M, K = free_vibration_analysis_free_chain(m, k_scaled)

            # Normalize mode shape
            phi = eigvecs[:, mode_idx]
            phi /= np.max(np.abs(phi))

            # Modal Energy: simplified to potential + kinetic at DOF
            E_kin = 0.5 * m[dof_idx] * np.abs(phi[dof_idx])**2
            E_pot = 0.5 * (np.abs(phi[dof_idx])**2) * np.sum(k_scaled) / len(k_scaled)  # approx
            E_total = E_kin + E_pot
            energies.append(E_total)

        results.append((c_scale, energies))

    return results

def plot_energy_sweep(k_range, results):
    plt.figure()
    for c_scale, energies in results:
        label = f'c × {c_scale:.1f}'
        plt.plot(k_range, energies, label=label)
    plt.xlabel("Stiffness Scaling Factor")
    plt.ylabel("Energy at DOF")
    plt.title("Energy vs. Stiffness for Different Damping Levels")
    plt.legend()
    plt.grid(True)
    plt.show()



# --------------------
# Computation Flags
COMPUTE_FORCED_RESPONSE = True
COMPUTE_FREE_VIBRATION_ANALYSIS = True
COMPUTE_MODAL_ENERGY_ANALYSIS = True

# Plotting Flags (these now control whether an overview function is called)
PLOT_OVERVIEW_FORCED_RESPONSE = True
PLOT_PHASE_ANGLES = False
PLOT_EXCITATION_POWER = True
PLOT_POLES = False
PLOT_MODAL_ENERGY = False


# Limit for the number of modes to plot in modal energy overview
mode_plot_limit = 5

###############################################################################
# SYSTEM MATRIX BUILDING FUNCTIONS (Free-Chain, No Ground Connection)
###############################################################################
def build_free_chain_matrices(m, c_inter, k_inter, C_ext=None, K_ext=None):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(len(k_inter)):
        C[i, i]     += c_inter[i]
        C[i, i+1]   -= c_inter[i]
        C[i+1, i]   -= c_inter[i]
        C[i+1, i+1] += c_inter[i]

        K[i, i]     += k_inter[i]
        K[i, i+1]   -= k_inter[i]
        K[i+1, i]   -= k_inter[i]
        K[i+1, i+1] += k_inter[i]

    if C_ext is not None:
        C += C_ext
    if K_ext is not None:
        K += K_ext

    return M, C, K



###############################################################################
# AUGMENTED SYSTEM FORCED RESPONSE (Enforcing a Boundary Condition)
###############################################################################
def build_augmented_system(D, F_ext, fixed_dof):
    N = D.shape[0]
    A_aug = np.zeros((N+1, N+1), dtype=complex)
    b_aug = np.zeros(N+1, dtype=complex)
    A_aug[0:N, 0:N] = D
    A_aug[fixed_dof, N] = -1.0
    b_aug[0:N] = F_ext
    A_aug[N, fixed_dof] = 1.0  # Enforce x_{N-1} = 0
    return A_aug, b_aug

def augment_with_branch(m, c_inter, k_inter, connection, m_branch, c_branch, k_branch, c_to_ground=0.0):
    i, j = connection  # i = existing DOF, j = new DOF index

    m_aug = np.append(m, m_branch)
    c_aug = np.append(c_inter, 0.0)  # New DOF
    k_aug = np.append(k_inter, 0.0)

    N = len(m_aug)
    C_ext = np.zeros((N, N))
    K_ext = np.zeros((N, N))

    # Add spring-damper between DOF i and new DOF j (existing branch)
    C_ext[i, i] += c_branch
    C_ext[i, j] -= c_branch
    C_ext[j, i] -= c_branch
    C_ext[j, j] += c_branch

    K_ext[i, i] += k_branch
    K_ext[i, j] -= k_branch
    K_ext[j, i] -= k_branch
    K_ext[j, j] += k_branch

    # Add damper from DOF j to ground (optional energy sink)
    if c_to_ground > 0.0:
        C_ext[j, j] += c_to_ground

    return m_aug, c_aug, k_aug, C_ext, K_ext



def compute_forced_response_free_chain(m, c_inter, k_inter, f_vals, F_ext):
    N = len(m)
    num_points = len(f_vals)
    X = np.zeros((N, num_points), dtype=complex)
    F_bound = np.zeros(num_points, dtype=complex)
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    
    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        D = K + 1j*w*C - (w**2)*M
        A_aug, b_aug = build_augmented_system(D, F_ext, fixed_dof=9)
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
def free_vibration_analysis_free_chain(m, k_inter, C_ext=None, K_ext=None):
    N = len(m)
    M, _, K = build_free_chain_matrices(m, np.zeros(N-1), k_inter)

    # If external stiffness matrix (e.g., from a branch) is provided, add it
    if K_ext is not None:
        K += K_ext

    eigvals, eigvecs = eigh(K, M)
    omega_n = np.sqrt(np.maximum(eigvals, 0))
    f_n = omega_n / (2 * np.pi)
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
def forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext, C_ext=None, K_ext=None):
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
    V_vals = np.zeros((N, num_points), dtype=complex)
    P_damp = np.zeros((N-1, num_points), dtype=complex)
    P_spring = np.zeros((N-1, num_points), dtype=complex)
    Q_mass = np.zeros((num_points, N))
    F_bound_array = np.zeros(num_points, dtype=complex)
    
    # Build free-chain matrices
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter, C_ext=C_ext, K_ext=K_ext)
    
    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        # Dynamic stiffness for free chain
        D = K + 1j*w*C - (w**2)*M
        # Augmented system to enforce x_{N-1} = 0
        A_aug, b_aug = build_augmented_system(D, F_ext, fixed_dof=9)  # or 9, or any grounded point

        sol = np.linalg.solve(A_aug, b_aug)
        
        X = sol[0:N]  # Displacement solution
        F_bound_array[i] = sol[N]
        X_vals[:, i] = X
        
        # Acceleration (a = -w^2*x)
        A_vals[:, i] = -w**2 * X
        
        # Velocity (v = w * X)
        V_vals[:, i] = w * X

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
        'V_vals': V_vals,
        'P_damp': P_damp,
        'P_spring': P_spring,
        'Q_mass': Q_mass,
        'F_bound': F_bound_array,
        'phase_vals': phase_vals,  # ADDED
    }



def compute_mode_interaction_force(mode_idx, dof_primary, dof_secondary, eigvecs, N, mode='cancel'):
    """
    Compute a force vector to either cancel or reinforce a specific mode using two excitations.

    Parameters:
    - mode_idx: index of the mode to target (0-based)
    - dof_primary: primary DOF for initial excitation
    - dof_secondary: secondary DOF for interaction (cancellation or reinforcement)
    - eigvecs: eigenvector matrix from modal analysis
    - N: total number of DOFs
    - mode: 'cancel' or 'reinforce'

    Returns:
    - F_ext: complex force vector for interaction
    """
    phi_mode = eigvecs[:, mode_idx]
    phi_mode /= np.max(np.abs(phi_mode))  # Normalize mode shape

    F_ext = np.zeros(N, dtype=complex)
    F_ext[dof_primary] = 1.0 + 0j

    ratio = phi_mode[dof_primary] / phi_mode[dof_secondary]
    if mode == 'cancel':
        F_ext[dof_secondary] = -ratio
    elif mode == 'reinforce':
        F_ext[dof_secondary] = +ratio
    else:
        raise ValueError("Mode must be 'cancel' or 'reinforce'")

    print(f"\nForce Vector to {mode.capitalize()} Mode {mode_idx + 1}:")
    print(f"  DOF {dof_primary}: {F_ext[dof_primary]}")
    print(f"  DOF {dof_secondary}: {F_ext[dof_secondary]}")
    return F_ext

def compute_power_flow(f_vals, X_vals, k_inter, m, omega_idx=None):
    """
    Computes energy flow across each spring at a specific frequency index.

    Parameters:
        f_vals (array): Frequency values [Hz]
        X_vals (array): Complex displacement response, shape (N_dof, N_freq)
        k_inter (array): Stiffness between DOFs, length N-1
        m (array): Mass/inertia at each DOF
        omega_idx (int): Optional, which frequency index to use (default = peak near 24Hz)
    
    Returns:
        P_spring_complex: Power flow (complex) per spring connection [Nm/s]
        power_map: Dictionary with real and imaginary components
    """
    N = X_vals.shape[0]
    if omega_idx is None:
        # Auto-detect peak near 24 Hz
        f_target = 24
        omega_idx = np.argmin(np.abs(f_vals - f_target))
    
    omega = 2 * np.pi * f_vals[omega_idx]
    X = X_vals[:, omega_idx]

    P_spring_complex = np.zeros(N - 1, dtype=complex)

    for i in range(N - 1):
        delta_theta = X[i] - X[i + 1]
        delta_omega_theta = 1j * omega * delta_theta
        P_spring_complex[i] = 0.5 * k_inter[i] * (delta_theta * np.conj(delta_omega_theta))

    power_map = {
        'real': np.real(P_spring_complex),   # energy exchange (elastic)
        'imag': np.imag(P_spring_complex),   # reactive flow
        'abs': np.abs(P_spring_complex)
    }

    return P_spring_complex, power_map

def sweep_energy_vs_stiffness_damping(m, k_inter, c_base, dof_idx, mode_idx, k_range, c_levels):
    results = []

    for c_scale in c_levels:
        energies = []
        for k_scale in k_range:
            # Scale stiffness and damping
            k_scaled = k_inter * k_scale
            c_scaled = c_base * c_scale

            # Recompute eigenvalues
            f_n, eigvecs, M, K = free_vibration_analysis_free_chain(m, k_scaled)

            # Normalize mode shape
            phi = eigvecs[:, mode_idx]
            phi /= np.max(np.abs(phi))

            # Modal Energy: simplified to potential + kinetic at DOF
            E_kin = 0.5 * m[dof_idx] * np.abs(phi[dof_idx])**2
            E_pot = 0.5 * (np.abs(phi[dof_idx])**2) * np.sum(k_scaled) / len(k_scaled)  # approx
            E_total = E_kin + E_pot
            energies.append(E_total)

        results.append((c_scale, energies))

    return results

def plot_energy_sweep(k_range, results):
    plt.figure()
    for c_scale, energies in results:
        label = f'c × {c_scale:.1f}'
        plt.plot(k_range, energies, label=label)
    plt.xlabel("Stiffness Scaling Factor")
    plt.ylabel("Energy at DOF")
    plt.title("Energy vs. Stiffness for Different Damping Levels")
    plt.legend()
    plt.grid(True)
    plt.show()


