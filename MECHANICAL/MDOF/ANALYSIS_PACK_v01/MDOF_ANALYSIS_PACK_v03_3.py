import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from excitation_generator import create_engine_excitation

# --------------------
# CONFIGURATION FLAGS
# --------------------
COMPUTE_FORCED_RESPONSE = True
COMPUTE_FREE_VIBRATION_ANALYSIS = True
COMPUTE_MODAL_ENERGY_ANALYSIS = True

PLOT_OVERVIEW_FORCED_RESPONSE = True
PLOT_EXCITATION_POWER = True
PLOT_POLES = True
PLOT_MODAL_ENERGY = True
PLOT_PHASE_ANGLES = True

mode_plot_limit = 5

# ------------------------------
# SYSTEM MATRIX BUILDING FUNCTIONS
# ------------------------------
def build_free_chain_matrices(m, c_inter, k_inter):
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

# ------------------------------
# AUGMENTED SYSTEM
# ------------------------------
def build_augmented_system(D, F_ext):
    N = D.shape[0]
    A_aug = np.zeros((N+1, N+1), dtype=complex)
    b_aug = np.zeros(N+1, dtype=complex)
    A_aug[0:N, 0:N] = D
    A_aug[N-1, N] = -1.0
    b_aug[0:N] = F_ext
    A_aug[N, N-1] = 1.0
    return A_aug, b_aug

# ------------------------------
# FORCED RESPONSE POST-PROCESSING
# ------------------------------
def forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext_matrix):
    N = len(m)
    num_points = len(f_vals)
    X_vals = np.zeros((N, num_points), dtype=complex)
    A_vals = np.zeros((N, num_points), dtype=complex)
    P_damp = np.zeros((N-1, num_points), dtype=complex)
    P_spring = np.zeros((N-1, num_points), dtype=complex)
    Q_mass = np.zeros((num_points, N))
    F_bound_array = np.zeros(num_points, dtype=complex)
    
    M, C, K = build_free_chain_matrices(m, c_inter, k_inter)

    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        D = K + 1j*w*C - (w**2)*M
        A_aug, b_aug = build_augmented_system(D, F_ext_matrix[:, i])
        sol = np.linalg.solve(A_aug, b_aug)
        X = sol[0:N]
        F_bound_array[i] = sol[N]
        X_vals[:, i] = X
        A_vals[:, i] = -w**2 * X
        V = 1j * w * X
        for j in range(1, N):
            X_rel = X[j] - X[j-1]
            V_rel = V[j] - V[j-1]
            P_damp[j-1, i] = c_inter[j-1] * (V_rel * np.conj(V_rel))
            P_spring[j-1, i] = k_inter[j-1] * (X_rel * np.conj(V_rel))
        Q_mass[i, :] = 1.0 * w * m * (np.abs(V)**2)

    phase_vals = np.angle(X_vals)
    return {
        'X_vals': X_vals,
        'A_vals': A_vals,
        'P_damp': P_damp,
        'P_spring': P_spring,
        'Q_mass': Q_mass,
        'F_bound': F_bound_array,
        'phase_vals': phase_vals,
    }

# ------------------------------
# MAIN SCRIPT
# ------------------------------
def main():
    m = np.array([1.21e-2, 3.95e-4, 7.92e-4, 1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3, 2.73e-1, 2.69e+1])
    c_inter = np.array([0.05]*9)
    k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5, 2.72e4, 4.97e3, 7.73e2, 8.57e2])

    f_min, f_max = 0.1, 200.0
    num_points = 10000
    f_vals = np.linspace(f_min, f_max, num_points)

    engine_speed_rpm = 6000
    harmonics = [
    (1, 100, 0),
    (2, 50, 0),
    (3, 20, 0),
    (4, 10, 0)
    ]


    excitation_array = create_engine_excitation(f_vals, harmonics, engine_speed_rpm)
    
    F_ext_matrix = np.zeros((len(m), num_points), dtype=complex)
    F_ext_matrix[0, :] = excitation_array

    plt.figure(figsize=(9, 5))
    plt.plot(f_vals, np.abs(F_ext_matrix[0, :]), label="Excitation Amplitude at DOF 0")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Nm)")
    plt.title("Frequency-Dependent Input Excitation")
    plt.grid()
    plt.legend()
    plt.show()

    if COMPUTE_FORCED_RESPONSE:
        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext_matrix)
        X_vals = response['X_vals']
        A_vals = response['A_vals']
        P_damp = response['P_damp']
        P_spring = response['P_spring']
        Q_mass = response['Q_mass']
        F_bound = response['F_bound']
        phase_vals = response['phase_vals']

        if PLOT_OVERVIEW_FORCED_RESPONSE:
            print("Overview plot function can be added here!")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
