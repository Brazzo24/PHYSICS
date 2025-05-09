def build_full_matrices(m, c_inter, k_inter):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N - 1):
        C[i, i] += c_inter[i]
        C[i, i+1] -= c_inter[i]
        C[i+1, i] -= c_inter[i]
        C[i+1, i+1] += c_inter[i]

        K[i, i] += k_inter[i]
        K[i, i+1] -= k_inter[i]
        K[i+1, i] -= k_inter[i]
        K[i+1, i+1] += k_inter[i]
    return M, C, K

from scipy.linalg import eigh
def run_modal_analysis(M, K, m_array, k_array, mode_index=3):
    eigvals, eigvecs = eigh(K, M)
    freqs_hz = np.sqrt(eigvals) / (2 * np.pi)

    phi = eigvecs[:, mode_index]
    omega = 2 * np.pi * freqs_hz[mode_index]
    phi = phi / np.max(np.abs(phi))

    KE_dof = 0.5 * m_array * (phi * omega)**2
    PE_spring = 0.5 * k_array * np.diff(phi)**2

    return freqs_hz[mode_index], KE_dof, PE_spring
