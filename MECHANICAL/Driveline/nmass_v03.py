import numpy as np
import matplotlib.pyplot as plt

# For sparse N-DOF system
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# For modal analysis
from scipy.linalg import eigh

###############################################################################
# 1. Basic Utility: Dynamic Stiffness
###############################################################################
def dynamic_stiffness(m, c, k, w):
    """
    Computes the complex dynamic stiffness for a single DOF:
        K_d(w) = k - w^2 * m + j * w * c
    """
    return k - (w**2)*m + 1j*w*c

###############################################################################
# 3. Generic N-DOF System (Sparse Representation)
###############################################################################
def compute_NDOF_response(N, masses, dampings, stiffnesses, w_vals):
    """
    Computes the frequency response for an N-DOF mass-spring-damper system
    using sparse matrices (for efficiency) and a base excitation at the first mass.
    Returns X_vals, shape = (N, n_freqs).
    """
    # Base velocity excitation in frequency domain
    X_base = 1.0 / (1j * w_vals)

    # Storage for responses
    X_vals = np.zeros((N, len(w_vals)), dtype=complex)

    # Frequency-by-frequency assembly
    for idx, w in enumerate(w_vals):
        # Sparse representation of the dynamic stiffness matrix
        main_diag = np.zeros(N, dtype=complex)
        off_diag  = np.zeros(N-1, dtype=complex)  # for both upper/lower

        for i in range(N):
            Kd = dynamic_stiffness(masses[i], dampings[i], stiffnesses[i], w)
            main_diag[i] = Kd
            if i > 0:
                off_diag[i-1] = -stiffnesses[i]

        # Build the sparse matrix with diagonals
        M_sparse = diags(
            [off_diag, main_diag, off_diag],
            offsets=[-1, 0, 1],
            format="csr"
        )

        # External force (only on mass 0 for base motion)
        F = -(1j*w*dampings[0] + stiffnesses[0]) * X_base[idx]
        RHS = np.zeros(N, dtype=complex)
        RHS[0] = F

        # Solve
        X_vals[:, idx] = spsolve(M_sparse, RHS)

    return X_vals

###############################################################################
# 4. Modal Analysis (Compute Natural Frequencies & Mode Shapes)
###############################################################################
def compute_modal_analysis(N, masses, dampings, stiffnesses):
    """
    Computes the undamped modal analysis for an N-DOF system:
    M_matrix (diagonal of masses)
    K_matrix (built from stiffnesses, including couplings)
    Returns the sorted natural frequencies (Hz) and corresponding mode shapes.
    """
    # Construct diagonal Mass matrix
    M_matrix = np.diag(masses)

    # Construct Stiffness matrix
    K_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        K_matrix[i, i] += stiffnesses[i]
        if i > 0:
            K_matrix[i, i]     += stiffnesses[i]   # coupling from neighbor
            K_matrix[i, i-1]    = -stiffnesses[i]
            K_matrix[i-1, i]    = -stiffnesses[i]

    # Generalized eigenvalue problem:  K * Phi = lambda * M * Phi
    eigvals, eigvecs = eigh(K_matrix, M_matrix)

    # Convert eigenvalues to frequencies (rad/s -> Hz)
    # Some small negative eigenvalues can appear from numerical round-off; take abs before sqrt.
    natural_frequencies = np.sqrt(np.abs(eigvals)) / (2.0 * np.pi)

    return natural_frequencies, eigvecs

###############################################################################
# 5. Modal Superposition
###############################################################################
def compute_modal_response(N, w_vals, natural_frequencies, mode_shapes, masses, stiffnesses):
    """
    Simple demonstration of modal superposition. For each mode i, compute
    its contribution to the total response. This example lumps the excitation
    at the first DOF (mode_shapes[0, i]) to scale each modal coordinate.
    """
    num_modes = N  # using all modes

    # Accumulate contributions from each mode
    modal_contributions = np.zeros((N, len(w_vals)), dtype=complex)

    for i in range(num_modes):
        phi_i = mode_shapes[:, i]
        lambda_i = (2.0 * np.pi * natural_frequencies[i])**2

        for j, w in enumerate(w_vals):
            # This line is a simplified approach to "modal coordinate" under unit base motion
            # The factor (phi_i[0]) is from applying force at DOF #0
            modal_contributions[:, j] += phi_i * (phi_i[0] / (lambda_i - w**2))

    # Sum across modes -> total response
    reconstructed_response = np.sum(modal_contributions, axis=0)
    return reconstructed_response, modal_contributions

###############################################################################
# MAIN SCRIPT / DEMO
###############################################################################
if __name__ == "__main__":


    # Frequency range
    f_min, f_max = 0.1, 35.0
    num_points = 1000
    f_vals = np.linspace(f_min, f_max, num_points)
    w_vals = 2.0 * np.pi * f_vals


    # ---------------------------------------------------------
    # 3) GENERAL N-DOF SYSTEM
    # ---------------------------------------------------------
    N_dof = 14
    # Example: linearly increasing mass, damping, stiffness
    masses = np.array([1.59e-6, 3.0e-3, 1.28e-4, 5.18e-5, 1.02e-3, 1.24e-3, 
                       1.22e-3, 1.96e-3, 1.22e-3, 1.22e-3, 1.0e-3,
                       1.77e-5, 5.46e-5, 2.77e-6]) # kgm^2
    
    dampings = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                         0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                         0.05, 0.05]) # Nm/rad.s

    # stiffnesses = np.array([4160.0, 1430.0, 1660.0, 90200.0, 37000.0,
    #                         8420.0, 8120.0, 8420.0, 8420.0,
    #                         8120.0, 8420.0, 33600.0, 21600.0, 17300.0]) # Nm/deg
    
    stiffnesses = np.array([2.38e5, 8.16e4, 9.48e4, 5.16e6, 2.12e6,
                        4.81e5, 4.64e5, 4.81e5, 4.81e5,
                        4.64e5, 4.81e5, 1.92e6, 1.24e6, 9.87e5]) # Nm/rad


    # Compute the response for each freq
    X_vals_NDOF = compute_NDOF_response(N_dof, masses, dampings, stiffnesses, w_vals)
    A_vals_NDOF = -w_vals**2 * X_vals_NDOF  # shape = (N_dof, num_points)

    # Plot the acceleration magnitudes for all DOFs
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(N_dof):
        ax.plot(f_vals, np.abs(A_vals_NDOF[i]), label=f"Mass {i+1}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.set_title(f"{N_dof}-DOF System Acceleration Response")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True)
    plt.show()

    # ---------------------------------------------------------
    # 4) MODAL ANALYSIS
    # ---------------------------------------------------------
    natural_frequencies, mode_shapes = compute_modal_analysis(N_dof, masses, dampings, stiffnesses)
    print("Natural Frequencies (Hz):")
    for i, fn in enumerate(natural_frequencies):
        print(f" Mode {i+1}: {fn:.2f} Hz")

    # Plot mode shapes
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(N_dof):
        ax.plot(range(1, N_dof+1), mode_shapes[:, i], marker='o', label=f"Mode {i+1}")
    ax.set_xlabel("Mass Index")
    ax.set_ylabel("Mode Shape Amplitude")
    ax.set_title("Mode Shapes of N-DOF System")
    ax.legend()
    ax.grid(True)
    plt.show()

    # ---------------------------------------------------------
    # 5) MODAL SUPERPOSITION EXAMPLE
    # ---------------------------------------------------------
    X_modal_reconstructed, modal_contributions = compute_modal_response(
        N_dof, w_vals, natural_frequencies, mode_shapes, masses, stiffnesses
    )
    A_modal_reconstructed = -w_vals**2 * X_modal_reconstructed

    # Compare with the full direct solution's response at DOF #0
    A_full_response = A_vals_NDOF[0, :]  # first mass only

    # Plot comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(f_vals, np.abs(A_full_response), label="Full System (1st mass)")
    ax.plot(f_vals, np.abs(A_modal_reconstructed), linestyle='--', label="Modal Approx (1st mass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.set_title("Modal Superposition vs. Full System")
    ax.legend()
    ax.grid(True)
    plt.show()

    # Plot each mode's contribution
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(N_dof):
        ax.plot(f_vals, np.abs(modal_contributions[i]), label=f"Mode {i+1}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Modal Contribution Amplitude")
    ax.set_title("Individual Modal Contributions (DOF #0)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True)
    plt.show()

    # ---------------------------------------------------------
    # 6) OPTIMIZATION EXAMPLES
    #    (Damping, Stiffness, Mass Redistribution)
    # ---------------------------------------------------------
    # ... The user’s original code demonstrates how to:
    #  - Identify critical modes
    #  - Increase damping at DOFs with large participation
    #  - Increase stiffness where certain modes dominate
    #  - Redistribute mass at critical DOFs
    #  - Compare new responses and energy dissipation
    #
    # The same pattern can be repeated:
    # 1) Modify "dampings" or "stiffnesses" or "masses"
    # 2) Re-run compute_NDOF_response
    # 3) Compare acceleration and damping power
    #
    # Below you can place all the "optimized damping" / "hybrid approach" code
    # as shown in your original script.

    print("\n[INFO] Code execution finished. Add optimization sections as desired.\n")
