import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig

# ----------------------
# CONFIGURATION FLAGS
# ----------------------
COMPUTE_FORCED_RESPONSE = True
COMPUTE_FREE_VIBRATION_ANALYSIS = True
COMPUTE_MODAL_ENERGY_ANALYSIS = True

PLOT_OVERVIEW_FORCED_RESPONSE = True
PLOT_PHASE_ANGLES = True
PLOT_EXCITATION_POWER = True
PLOT_POLES = False
PLOT_MODAL_ENERGY = True

MODE_PLOT_LIMIT = 5  # For modal energy overview

# ----------------------
# MATRIX ASSEMBLY
# ----------------------
def build_free_chain_matrices(masses, damping, stiffness):
    N = len(masses)
    M = np.diag(masses)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N - 1):
        for mat in [C, K]:
            mat[i, i]     += [damping, stiffness][mat is K][i]
            mat[i, i+1]   -= [damping, stiffness][mat is K][i]
            mat[i+1, i]   -= [damping, stiffness][mat is K][i]
            mat[i+1, i+1] += [damping, stiffness][mat is K][i]
    return M, C, K

# ----------------------
# FORCED RESPONSE SETUP
# ----------------------
def build_augmented_system(D, F_ext):
    N = D.shape[0]
    A_aug = np.zeros((N+1, N+1), dtype=complex)
    b_aug = np.zeros(N+1, dtype=complex)
    A_aug[:N, :N] = D
    A_aug[N-1, N] = -1.0
    A_aug[N, N-1] = 1.0
    b_aug[:N] = F_ext
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
        X[:, i], F_bound[i] = sol[:N], sol[N]

    return X, F_bound

# ----------------------
# EIGENVALUE ANALYSIS
# ----------------------
def compute_poles_free_chain(m, c_inter, k_inter):
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    N = len(m)
    zero_block, I_block = np.zeros((N, N)), np.eye(N)
    Minv = np.linalg.inv(M)
    A = np.vstack((
        np.hstack((zero_block, I_block)),
        np.hstack((-Minv @ K, -Minv @ C))
    ))
    poles, _ = eig(A)
    return poles

# ----------------------
# FREE VIBRATION & MODAL ENERGY
# ----------------------
def free_vibration_analysis_free_chain(m, k_inter):
    N = len(m)
    M, _, K = build_free_chain_matrices(m, np.zeros(N-1), k_inter)
    eigvals, eigvecs = eigh(K, M)
    f_n = np.sqrt(np.maximum(eigvals, 0)) / (2 * np.pi)
    return f_n, eigvecs, M, K

def modal_energy_analysis(m, k_inter, f_n, eigvecs, M):
    modal_energies = []
    for i in range(eigvecs.shape[1]):
        phi = eigvecs[:, i]
        norm_phi = phi / np.sqrt(np.real(np.conj(phi) @ M @ phi))
        omega_i = 2 * np.pi * f_n[i]
        T_dof = m * (omega_i * np.abs(norm_phi))**2
        V_springs = 0.5 * k_inter * (np.diff(norm_phi)**2)
        modal_energies.append({
            'mode': i+1,
            'omega_rad_s': omega_i,
            'T_total': T_dof.sum(),
            'V_total': V_springs.sum(),
            'T_dof': T_dof,
            'V_springs': V_springs,
            'phi_norm': norm_phi
        })
    return modal_energies

# ----------------------
# FORCED RESPONSE POSTPROCESSING
# ----------------------
def forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext):
    N, num_points = len(m), len(f_vals)
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)
    results = {
        'X_vals': np.zeros((N, num_points), dtype=complex),
        'A_vals': np.zeros((N, num_points), dtype=complex),
        'V_vals': np.zeros((N, num_points), dtype=complex),
        'P_damp': np.zeros((N-1, num_points), dtype=complex),
        'P_spring': np.zeros((N-1, num_points), dtype=complex),
        'Q_mass': np.zeros((num_points, N)),
        'F_bound': np.zeros(num_points, dtype=complex),
        'phase_vals': np.zeros((N, num_points))
    }

    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        D = K + 1j * w * C - (w ** 2) * M
        A_aug, b_aug = build_augmented_system(D, F_ext)
        sol = np.linalg.solve(A_aug, b_aug)
        X = sol[:N]
        V = 1j * w * X
        A = -w ** 2 * X

        results['X_vals'][:, i] = X
        results['V_vals'][:, i] = V
        results['A_vals'][:, i] = A
        results['F_bound'][i] = sol[N]
        results['Q_mass'][i, :] = w * m * np.abs(V)**2
        results['phase_vals'][:, i] = np.angle(X)

        for j in range(1, N):
            X_rel = X[j] - X[j-1]
            V_rel = V[j] - V[j-1]
            results['P_damp'][j-1, i] = c_inter[j-1] * (V_rel * np.conj(V_rel))
            results['P_spring'][j-1, i] = k_inter[j-1] * (X_rel * np.conj(V_rel))

    return results
