import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig

###############################################################################
# USER INPUT SECTION
###############################################################################
m = np.array([1.0, 1.0, 2.0])      # masses [kg]
c = np.array([25.0, 2.0, 3.0])     # damping [Ns/m]
k = np.array([2000.0, 12000.0, 6000.0])  # stiffness [N/m]

f_min = 0.1    # Hz
f_max = 35.0   # Hz
num_points = 1000

###############################################################################
# HELPER FUNCTIONS
###############################################################################

def build_chain_matrices(m, c, k):
    """
    Build full NxN mass, damping, and stiffness matrices for a chain-like system.
    For an N-DOF system:
      - element 0 => ground-to-mass0
      - element i => mass (i-1) to mass i for i=1..N-1
    Returns M_full, C_full, K_full as (N,N) matrices.
    """
    N = len(m)
    M_full = np.diag(m)
    C_full = np.zeros((N, N))
    K_full = np.zeros((N, N))
    
    if N == 1:
        # single-DOF
        C_full[0, 0] = c[0]
        K_full[0, 0] = k[0]
    else:
        # DOF 0
        C_full[0, 0] = c[0] + c[1]
        C_full[0, 1] = -c[1]
        C_full[1, 0] = -c[1]
        K_full[0, 0] = k[0] + k[1]
        K_full[0, 1] = -k[1]
        K_full[1, 0] = -k[1]
        # Intermediate DOFs
        for i in range(1, N-1):
            C_full[i, i]   = c[i] + c[i+1]
            C_full[i, i+1] = -c[i+1]
            C_full[i+1, i] = -c[i+1]
            
            K_full[i, i]   = k[i] + k[i+1]
            K_full[i, i+1] = -k[i+1]
            K_full[i+1, i] = -k[i+1]
        # Last DOF
        C_full[N-1, N-1] = c[N-1]
        K_full[N-1, N-1] = k[N-1]
    
    return M_full, C_full, K_full


def compute_poles(m, c, k):
    """
    Compute poles (state-space eigenvalues) for M x'' + C x' + K x = 0
    by building A = [[0, I], [-M^-1 K, -M^-1 C]] and computing eig(A).
    """
    N = len(m)
    M_full, C_full, K_full = build_chain_matrices(m, c, k)
    
    zero_block = np.zeros((N, N))
    I_block = np.eye(N)
    Minv = np.linalg.inv(M_full)
    
    A_upper = np.hstack((zero_block, I_block))
    A_lower = np.hstack((-Minv @ K_full, -Minv @ C_full))
    A = np.vstack((A_upper, A_lower))
    
    poles, _ = eig(A)
    return poles


def compute_NDOF_response_base_excitation(m, c, k, w):
    """
    Frequency-domain displacement response X for an NDOF chain system
    under base excitation with unit base velocity.
    Returns the complex displacement vector X (length N).
    """
    N = len(m)
    X_base = 1/(1j*w)  # base displacement from unit velocity
    A = np.zeros((N, N), dtype=complex)
    
    # Build dynamic stiffness
    if N == 1:
        A[0, 0] = k[0] - w**2*m[0] + 1j*w*c[0]
    else:
        A[0, 0] = (k[0] - w**2*m[0] + 1j*w*c[0]) + (k[1] + 1j*w*c[1])
        A[0, 1] = -(k[1] + 1j*w*c[1])
        for i in range(1, N-1):
            A[i, i-1] = -(k[i] + 1j*w*c[i])
            A[i, i]   = (k[i] + 1j*w*c[i]) + (k[i+1] + 1j*w*c[i+1]) - w**2*m[i]
            A[i, i+1] = -(k[i+1] + 1j*w*c[i+1])
        A[N-1, N-2] = -(k[N-1] + 1j*w*c[N-1])
        A[N-1, N-1] = k[N-1] - w**2*m[N-1] + 1j*w*c[N-1]
    
    # Forcing vector from base motion
    F = np.zeros(N, dtype=complex)
    F[0] = -(k[0] + 1j*w*c[0]) * X_base
    
    X = np.linalg.solve(A, F)
    return X


def free_vibration_analysis(m, k):
    """
    Undamped free-vibration analysis for an N-DOF chain system.
    Returns:
      f_n: array of natural frequencies (Hz),
      eigvecs: matrix of eigenvectors (each column is a mode shape),
      M_free, K_free: the mass and stiffness matrices.
    """
    N = len(m)
    M_free = np.diag(m)
    K_free = np.zeros((N, N))
    
    if N == 1:
        K_free[0, 0] = k[0]
    else:
        K_free[0, 0] = k[0] + k[1]
        K_free[0, 1] = -k[1]
        K_free[1, 0] = -k[1]
        for i in range(1, N-1):
            K_free[i, i]   = k[i] + k[i+1]
            K_free[i, i+1] = -k[i+1]
            K_free[i+1, i] = -k[i+1]
        K_free[N-1, N-1] = k[N-1]
    
    eigvals, eigvecs = eigh(K_free, M_free)
    omega_n = np.sqrt(eigvals)   # rad/s
    f_n = omega_n / (2*np.pi)    # Hz
    return f_n, eigvecs, M_free, K_free


def modal_energy_analysis(m, k, f_n, eigvecs, M_free):
    """
    For each eigenmode, mass-normalize and compute:
      - Kinetic energy distribution per DOF,
      - Potential energy distribution per spring.
    """
    N = len(m)
    modal_energies = []
    for i in range(eigvecs.shape[1]):
        phi = eigvecs[:, i]
        # Normalize so that phi^T M phi = 1
        norm_factor = np.sqrt(np.real(np.conjugate(phi).T @ M_free @ phi))
        phi_norm = phi / norm_factor
        
        omega_i = 2*np.pi*f_n[i]  # rad/s
        
        # Kinetic energies
        T_dof = 0.5 * m * (omega_i * np.abs(phi_norm))**2
        T_total = np.sum(T_dof)
        
        # Potential energies
        V_springs = np.zeros(N)
        # Spring 0 => base-to-mass0
        V_springs[0] = 0.5*k[0]*(np.abs(phi_norm[0]))**2
        # Springs i => mass i-1 to i
        for s in range(1, N):
            V_springs[s] = 0.5*k[s]*(np.abs(phi_norm[s] - phi_norm[s-1]))**2
        V_total = np.sum(V_springs)
        
        modal_energies.append({
            'mode': i+1,
            'omega_rad_s': omega_i,
            'T_total': T_total,
            'V_total': V_total,
            'T_dof': T_dof,
            'V_springs': V_springs,
            'phi_norm': phi_norm
        })
    return modal_energies



###############################################################################
# MAIN SCRIPT
###############################################################################
def main():
    """ Runs all analyses and produces structured plots. """
    f_vals = np.linspace(f_min, f_max, num_points)
    w_vals = 2 * np.pi * f_vals
    N = len(m)

    # Prepare storage
    X_vals = np.zeros((N, num_points), dtype=complex)
    A_vals = np.zeros((N, num_points), dtype=complex)
    T_vals = np.zeros((N, num_points))

    for i, w in enumerate(w_vals):
        X = compute_NDOF_response_base_excitation(m, c, k, w)
        X_vals[:, i] = X
        A_vals[:, i] = -w**2 * X
        T_vals[:, i] = w * np.abs(X)  # Transmissibility

    # --- PLOT: Acceleration & Transmissibility ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Acceleration Response
    for i in range(N):
        axes[0].plot(f_vals, np.abs(A_vals[i, :]), label=f'|A{i+1}|')
    axes[0].set_xlabel('Frequency [Hz]')
    axes[0].set_ylabel('Acceleration [m/s²]')
    axes[0].set_title('Acceleration Response of Each Mass')
    axes[0].legend()
    axes[0].grid(True)

    # Transmissibility
    for i in range(N):
        axes[1].plot(f_vals, T_vals[i, :], label=f'Transmissibility - Mass {i+1}')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Transmissibility')
    axes[1].set_title('Transmissibility of Each Mass')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # --- Compute Power Dissipation ---
    P_damping = np.zeros((num_points, N), dtype=complex)
    P_spring = np.zeros((num_points, N), dtype=complex)

    for i, w in enumerate(w_vals):
        X = compute_NDOF_response_base_excitation(m, c, k, w)
        V = 1j * w * X
        X_base = 1 / (1j * w)
        V_base = 1.0

        for j in range(N):
            if j == 0:
                V_rel = V[0] - V_base
                X_rel = X[0] - X_base
            else:
                V_rel = V[j] - V[j-1]
                X_rel = X[j] - X[j-1]

            P_damping[i, j] = c[j] * V_rel * np.conj(V_rel)  # Active Power
            P_spring[i, j] = k[j] * X_rel * np.conj(V_rel)  # Reactive Power

    # --- PLOT: Power Dissipation ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Active Power (Dampers)
    for j in range(N):
        axes[0].plot(f_vals, np.real(P_damping[:, j]), label=f"Damper {j+1}")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Active Power [W]")
    axes[0].set_title("Active Power Dissipated in Damping Elements")
    axes[0].legend()
    axes[0].grid(True)

    # Reactive Power (Springs)
    for j in range(N):
        axes[1].plot(f_vals, np.imag(P_spring[:, j]), label=f"Spring {j+1}")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Reactive Power [VAR]")
    axes[1].set_title("Reactive Power Stored in Springs")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 4) Free-vibration analysis
    f_n, eigvecs, M_free, K_free = free_vibration_analysis(m, k)
    print("\nUndamped Natural Frequencies (Hz):")
    for i, fn in enumerate(f_n):
        print(f"  Mode {i+1}: {fn:.3f} Hz")
    
    # 5) Modal energy analysis + bar plots
    modal_energies = modal_energy_analysis(m, k, f_n, eigvecs, M_free)
    for mode_data in modal_energies:
        mode_idx = mode_data['mode']
        freq_hz = mode_data['omega_rad_s']/(2*np.pi)
        print(f"\nMode {mode_idx} at {freq_hz:.3f} Hz")
        print(f"  Total Kinetic Energy: {mode_data['T_total']:.4f} J")
        print(f"  Total Potential Energy: {mode_data['V_total']:.4f} J")
        print(f"  T_dof distribution: {mode_data['T_dof']}")
        print(f"  V_springs distribution: {mode_data['V_springs']}")
    
    # Optional: bar plots for each mode
    for mode_data in modal_energies:
        mode_idx = mode_data['mode']
        freq_hz = mode_data['omega_rad_s']/(2*np.pi)
        
        # Kinetic
        plt.figure(figsize=(7,4))
        dof_indices = np.arange(N)+1
        plt.bar(dof_indices, mode_data['T_dof'], color='g', alpha=0.7)
        plt.xlabel('DOF Index')
        plt.ylabel('Kinetic Energy [J]')
        plt.title(f'Kinetic Energy Distribution, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        plt.grid(True)
        plt.show()
        
        # Potential
        plt.figure(figsize=(7,4))
        spring_indices = np.arange(N)+1
        plt.bar(spring_indices, mode_data['V_springs'], color='r', alpha=0.7)
        plt.xlabel('Spring Index')
        plt.ylabel('Potential Energy [J]')
        plt.title(f'Potential Energy Distribution, Mode {mode_idx} ({freq_hz:.2f} Hz)')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
