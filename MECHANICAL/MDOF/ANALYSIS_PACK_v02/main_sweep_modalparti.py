import numpy as np
import matplotlib.pyplot as plt
from FDcalculations_ import *

def compute_modal_participation(X_vals, eigvecs, M):
    N_dof, N_freq = X_vals.shape
    N_modes = eigvecs.shape[1]
    Q_mode = np.zeros((N_freq, N_modes))

    for i in range(N_freq):
        X_f = X_vals[:, i]
        q_f = eigvecs.T @ M @ X_f
        Q_mode[i, :] = np.abs(q_f)**2

    return Q_mode

def sweep_modal_participation_vs_k_c(m, k_inter, c_inter, f_vals, F_ext,
                                     mode_idx,
                                     k_factors=np.linspace(0.5, 2.0, 10),
                                     c_factors=np.linspace(0.5, 1.5, 5),
                                     k_indices=None,
                                     c_indices=None):
    result_values = []
    power_values = []

    for c_scale in c_factors:
        for k_scale in k_factors:
            k_scaled = k_inter.copy()
            c_scaled = c_inter.copy()

            if k_indices is not None:
                for idx in k_indices:
                    k_scaled[idx] = k_inter[idx] * k_scale
            else:
                k_scaled *= k_scale

            if c_indices is not None:
                for idx in c_indices:
                    c_scaled[idx] = c_inter[idx] * c_scale
            else:
                c_scaled *= c_scale

            response = forced_response_postprocessing(m, c_scaled, k_scaled, f_vals, F_ext)
            X_vals = response['X_vals']
            f_n, eigvecs, M_modal, _ = free_vibration_analysis_free_chain(m, k_scaled)

            Q_mode = compute_modal_participation(X_vals, eigvecs, M_modal)
            participation = np.max(Q_mode[:, mode_idx])
            result_values.append(participation)

            _, power_map = compute_power_flow(f_vals, X_vals, k_scaled, m)
            power_values.append(np.max(power_map['abs']))

    return k_factors, c_factors, result_values, power_values

def plot_modal_participation_surface(k_factors, c_factors, result_values, mode_idx, zlabel="Modal Participation Energy"):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    K, C = np.meshgrid(k_factors, c_factors)
    Z = np.array(result_values).reshape(len(c_factors), len(k_factors))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(K, C, Z, cmap=cm.viridis, edgecolor='k', alpha=0.9)

    ax.set_xlabel("Stiffness Scaling Factor")
    ax.set_ylabel("Damping Scaling Factor")
    ax.set_zlabel(zlabel)
    ax.set_title(f"Surface Plot for Mode {mode_idx + 1}")
    plt.tight_layout()
    plt.show()

def main():
    m = np.array([1.21e-2, 3.95e-4, 7.92e-4, 1.02e-3, 1.42e-3,
                  1.12e-4, 1.22e-3, 1.35e-3, 7.73e-2, 2.69e+1])
    k_inter = np.array([2.34e4, 1.62e5, 1.1e3, 1.10e5, 1.10e5,
                        2.72e4, 4.97e3, 2.73e2, 8.57e2])
    c_inter = np.array([0.05] * 9)

    f_vals = np.linspace(0.1, 40.0, 5000)
    N = len(m)
    F_ext = np.zeros(N, dtype=complex)
    F_ext[5] = 1.0 + 0j

    mode_idx = 2  # mode 2 (0-based index)

    k_factors = np.linspace(0.5, 10.0, 20)
    c_factors = np.linspace(0.5, 4.0, 20)
    k_target_indices = [7]
    c_target_indices = [7]

    k_vals, c_vals, modal_results, power_results = sweep_modal_participation_vs_k_c(
        m, k_inter, c_inter, f_vals, F_ext,
        mode_idx,
        k_factors=k_factors,
        c_factors=c_factors,
        k_indices=k_target_indices,
        c_indices=c_target_indices
    )

    plot_modal_participation_surface(k_vals, c_vals, modal_results, mode_idx, zlabel="Modal Participation Energy")
    plot_modal_participation_surface(k_vals, c_vals, power_results, mode_idx, zlabel="Max Power Flow Magnitude")

if __name__ == "__main__":
    main()
