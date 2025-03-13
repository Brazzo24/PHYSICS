import numpy as np
from scipy.linalg import eigh, eig

# (Assuming the functions from the previous code are imported or defined in the same file)
# For example:
# - build_free_chain_matrices(m, c_inter, k_inter)
# - build_augmented_system(D, F_ext)
# - compute_forced_response_free_chain(m, c_inter, k_inter, f_vals, F_ext)

def test_free_vibration_single_dof_verbose():
    print("=== Test: Free Vibration for a Single-DOF System (Verbose) ===")
    m = np.array([1.0])
    # For a 1-DOF system, we directly set a stiffness value.
    k_expected = 100.0
    M = np.diag(m)
    K = np.array([[k_expected]])
    
    print("Mass matrix M:")
    print(M)
    print("Stiffness matrix K:")
    print(K)
    
    # Solve eigenvalue problem: K phi = lambda M phi
    eigvals, eigvecs = eigh(K, M)
    print("Eigenvalues:", eigvals)
    print("Eigenvectors (mode shapes):")
    print(eigvecs)
    
    omega_n = np.sqrt(eigvals)
    f_n = omega_n / (2 * np.pi)
    expected_f = np.sqrt(k_expected / m[0]) / (2 * np.pi)
    
    print("Computed natural frequency (Hz):", f_n)
    print("Expected natural frequency (Hz):", expected_f)
    assert np.allclose(f_n, expected_f, rtol=1e-5), (
        f"Single DOF free-vibration test failed: computed f_n={f_n} vs expected {expected_f}"
    )
    print("Test passed.\n")

def test_augmented_system_constraint_verbose():
    print("=== Test: Augmented System Constraint for a 2-DOF Free-Chain (Verbose) ===")
    m = np.array([1.0, 2.0])
    c_inter = np.array([0.5])
    k_inter = np.array([100.0])
    
    # External force vector: apply a force at DOF 0 only.
    F_ext = np.array([1.0, 0.0], dtype=complex)
    f_val = 5.0  # frequency in Hz
    f_vals = np.array([f_val])
    
    # Build free-chain matrices
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    print("Free-chain mass matrix M:")
    print(M)
    print("Free-chain damping matrix C:")
    print(C)
    print("Free-chain stiffness matrix K:")
    print(K)
    
    # Compute dynamic stiffness matrix at f=5 Hz
    w = 2 * np.pi * f_val
    D = K + 1j * w * C - (w**2) * M
    print("\nDynamic stiffness matrix D at f = 5 Hz:")
    print(D)
    
    # Build augmented system to enforce x_{N-1}=0 (constraint on DOF 1)
    A_aug, b_aug = build_augmented_system(D, F_ext)
    print("\nAugmented system matrix A_aug:")
    print(A_aug)
    print("Augmented right-hand side vector b_aug:")
    print(b_aug)
    
    # Solve augmented system
    sol = np.linalg.solve(A_aug, b_aug)
    X = sol[0:len(m)]
    F_bound = sol[len(m)]
    print("\nSolution vector (displacements and reaction):")
    print(sol)
    print("Extracted displacements X:")
    print(X)
    print("Extracted reaction force F_bound:")
    print(F_bound)
    
    # Check that the displacement at the constrained DOF (last DOF) is zero.
    assert np.allclose(X[-1], 0, atol=1e-6), (
        f"Augmented system constraint not met: DOF {len(m)-1} displacement = {X[-1]}"
    )
    print("Test passed.\n")

def run_tests_verbose():
    test_free_vibration_single_dof_verbose()
    test_augmented_system_constraint_verbose()
    print("All verbose tests passed.")

if __name__ == "__main__":
    run_tests_verbose()
