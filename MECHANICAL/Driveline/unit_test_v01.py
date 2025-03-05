import unittest
import numpy as np

# --- Assume that these functions are defined in your code base ---
# For example, you might have:
# from my_dynamic_module import dynamic_stiffness, compute_NDOF_response, compute_modal_analysis, compute_modal_response
#
# For the sake of this test, we assume they are defined in the same file.

def dynamic_stiffness(m, c, k, w):
    return k - (w**2)*m + 1j*w*c

def compute_NDOF_response(N, masses, dampings, stiffnesses, w_vals):
    X_vals = np.zeros((N, len(w_vals)), dtype=complex)
    X_base = 1.0 / (1j * w_vals)  # base velocity excitation converted to displacement
    for idx, w in enumerate(w_vals):
        main_diag = np.zeros(N, dtype=complex)
        off_diag  = np.zeros(N-1, dtype=complex)
        for i in range(N):
            Kd = dynamic_stiffness(masses[i], dampings[i], stiffnesses[i], w)
            main_diag[i] = Kd
            if i > 0:
                off_diag[i-1] = -stiffnesses[i]
        # Build the sparse matrix using full (dense) representation for simplicity in the test
        M_matrix = np.diag(main_diag)
        for i in range(1, N):
            M_matrix[i, i-1] = off_diag[i-1]
            M_matrix[i-1, i] = off_diag[i-1]
        F = -(1j * w * dampings[0] + stiffnesses[0]) * X_base[idx]
        RHS = np.zeros(N, dtype=complex)
        RHS[0] = F
        X_vals[:, idx] = np.linalg.solve(M_matrix, RHS)
    return X_vals

def compute_modal_analysis(N, masses, dampings, stiffnesses):
    M_matrix = np.diag(masses)
    K_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        K_matrix[i, i] += stiffnesses[i]
        if i > 0:
            K_matrix[i, i] += stiffnesses[i]
            K_matrix[i, i-1] = -stiffnesses[i]
            K_matrix[i-1, i] = -stiffnesses[i]
    # Solve the generalized eigenvalue problem: K*phi = lambda*M*phi
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M_matrix) @ K_matrix)
    # Sort the eigenvalues (and corresponding eigenvectors) in ascending order
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    natural_frequencies = np.sqrt(np.abs(eigvals)) / (2 * np.pi)
    return natural_frequencies, eigvecs

def compute_modal_response(N, w_vals, natural_frequencies, mode_shapes, masses, stiffnesses):
    num_modes = N  # using all modes for reconstruction
    # This array will hold the contribution from each mode to each DOF at each frequency
    modal_contributions = np.zeros((N, len(w_vals)), dtype=complex)
    for i in range(num_modes):
        phi_i = mode_shapes[:, i]
        lambda_i = (2 * np.pi * natural_frequencies[i])**2
        for j, w in enumerate(w_vals):
            # Here the factor (phi_i[0]) represents the effect of the base excitation applied at DOF 0.
            modal_contributions[:, j] += phi_i * (phi_i[0] / (lambda_i - w**2))
    # For a direct comparison with the full response at DOF 0, we consider the reconstructed response there.
    reconstructed_response = modal_contributions[0, :]  # response at first DOF
    return reconstructed_response, modal_contributions
# --- End of assumed functions ---

class TestDynamicSystem(unittest.TestCase):
    
    def test_modal_analysis_2dof(self):
        # Define a simple 2-DOF system (serial arrangement)
        m1, m2 = 1.0, 1.0
        c1, c2 = 0.0, 0.0
        k1, k2 = 100.0, 100.0
        masses = np.array([m1, m2])
        dampings = np.array([c1, c2])
        stiffnesses = np.array([k1, k2])
        
        natural_frequencies, mode_shapes = compute_modal_analysis(2, masses, dampings, stiffnesses)
        # Expected natural frequencies are approximately 0.98 Hz and 2.58 Hz.
        self.assertAlmostEqual(natural_frequencies[0], 0.98, places=2)
        self.assertAlmostEqual(natural_frequencies[1], 2.58, places=2)
    
    def test_modal_superposition_2dof(self):
        # Define a 2-DOF system (no damping)
        m1, m2 = 1.0, 1.0
        c1, c2 = 0.0, 0.0
        k1, k2 = 100.0, 100.0
        masses = np.array([m1, m2])
        dampings = np.array([c1, c2])
        stiffnesses = np.array([k1, k2])
        
        # Test at a single frequency f = 1 Hz
        f_test = 1.0  # Hz
        w_test = 2 * np.pi * f_test
        w_vals = np.array([w_test])
        
        # Full system response (using our sparse solver; here implemented densely for the test)
        X_vals = compute_NDOF_response(2, masses, dampings, stiffnesses, w_vals)
        full_response_dof0 = X_vals[0, 0]  # displacement at DOF 0
        
        # Modal analysis and modal superposition reconstruction
        natural_frequencies, mode_shapes = compute_modal_analysis(2, masses, dampings, stiffnesses)
        modal_response, modal_contributions = compute_modal_response(2, w_vals, natural_frequencies, mode_shapes, masses, stiffnesses)
        modal_response_dof0 = modal_response[0]  # response at DOF 0
        
        # Compare full response vs. modal superposition result (both are complex values)
        self.assertAlmostEqual(np.abs(full_response_dof0), np.abs(modal_response_dof0), places=4)
        self.assertAlmostEqual(np.angle(full_response_dof0), np.angle(modal_response_dof0), places=4)

if __name__ == '__main__':
    unittest.main()
