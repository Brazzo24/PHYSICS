# main_sweep.py

import numpy as np
import matplotlib.pyplot as plt
from FDcalculations_ import *

# Optional flags
PowerFlow = True
EbyKC = False
PowerSweep = True

def plot_3d_power_surface(k_vals, c_vals, power_values, mode_idx=0):
    """
    Plot 3D surface of power response vs. stiffness and damping scale.
    """
    K, C = np.meshgrid(k_vals, c_vals)
    Z = np.array(power_values).reshape(len(c_vals), len(k_vals))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K, C, Z, cmap='viridis', edgecolor='k', alpha=0.9)

    ax.set_xlabel("Stiffness Scale (k)")
    ax.set_ylabel("Damping Scale (c)")
    ax.set_zlabel("Peak Inertial Reactive Power [VAR]")
    ax.set_title(f"Mode {mode_idx+1}: Power Response Surface")

    plt.tight_layout()
    plt.show()

def extract_peak_inertial_power(f_vals, Q_mass, mass_idx, freq_bins):
    peaks = []
    for fmin, fmax in freq_bins:
        indices = np.where((f_vals >= fmin) & (f_vals <= fmax))[0]
        if len(indices) == 0:
            peaks.append(0.0)
            continue
        power_slice = Q_mass[indices, mass_idx]  # FIXED INDEXING
        peak_val = np.max(np.abs(power_slice))
        peaks.append(peak_val)
    return peaks



def sweep_power_vs_k_c(m, k_inter, c_inter, f_vals, F_ext, mass_idx, freq_bins, 
                       k_factors=np.linspace(0.5, 2.0, 4), 
                       c_factors=np.linspace(0.5, 1.0, 2),
                       k_indices=None,
                       c_indices=None):
    
    result_matrix = {i: [] for i in range(len(freq_bins))}

    for c_scale in c_factors:
        for k_scale in k_factors:
            # Copy base arrays
            k_scaled = k_inter.copy()
            c_scaled = c_inter.copy()

            # Apply scaling only to specified indices
            if k_indices is not None:
                for idx in k_indices:
                    k_scaled[idx] = k_inter[idx] * k_scale
            else:
                k_scaled *= k_scale  # fallback (old behavior)

            if c_indices is not None:
                for idx in c_indices:
                    c_scaled[idx] = c_inter[idx] * c_scale
            else:
                c_scaled *= c_scale  # fallback

            # Get response and extract peaks
            response = forced_response_postprocessing(m, c_scaled, k_scaled, f_vals, F_ext)
            Q_mass = response['Q_mass']
            peaks = extract_peak_inertial_power(f_vals, Q_mass, mass_idx, freq_bins)

            for i, val in enumerate(peaks):
                result_matrix[i].append(val)
    
    return k_factors, c_factors, result_matrix


def plot_sweep_results(k_factors, c_factors, result_matrix, freq_bins):
    for mode_idx, bin_label in enumerate(freq_bins):
        plt.figure()
        for j, c_scale in enumerate(c_factors):
            start = j * len(k_factors)
            stop = start + len(k_factors)
            plt.plot(k_factors, result_matrix[mode_idx][start:stop], label=f"c × {c_scale:.2f}")
        plt.xlabel("Stiffness Scaling Factor")
        plt.ylabel("Peak Inertial Reactive Power [VAR]")
        plt.title(f"Mode {mode_idx+1}: Max Power in Bin {bin_label[0]}-{bin_label[1]} Hz")
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    m = np.array([1.21e-2, 3.95e-4, 7.92e-4, 1.02e-3, 1.42e-3,
                  1.12e-4, 1.22e-3, 1.35e-3, 7.73e-2, 2.69e+1])
    k_inter = np.array([2.34e4, 1.62e5, 1.1e3, 1.10e5, 1.10e5,
                        2.72e4, 4.97e3, 2.73e2, 8.57e2])
    c_inter = np.array([0.05]*9)

    N = len(m)
    f_vals = np.linspace(0.1, 40.0, 1000)
    dof_primary = 5
    F_ext = np.zeros(N, dtype=complex)
    F_ext[dof_primary] = 1.0 + 0j

    if PowerSweep:
        mass_idx = 8
        freq_bins = [(10, 19), (20, 40)]

        # Only sweep spring and damper between DOF 7 and 8
        k_target_indices = [7]
        c_target_indices = [7]

        k_vals, c_vals, results = sweep_power_vs_k_c(
            m, k_inter, c_inter, f_vals, F_ext,
            mass_idx, freq_bins,
            k_factors=np.linspace(0.5, 10.0, 20),
            c_factors=np.linspace(0.5, 10.0, 20),
            k_indices=k_target_indices,
            c_indices=c_target_indices
        )

        plot_sweep_results(k_vals, c_vals, results, freq_bins)

        # (Optional) summary print/export remains unchanged

    # Plot Mode 1 (index 0) in 3D
    plot_3d_power_surface(k_vals, c_vals, results[0], mode_idx=0)
    plot_3d_power_surface(k_vals, c_vals, results[1], mode_idx=1)


if __name__ == "__main__":
    main()
