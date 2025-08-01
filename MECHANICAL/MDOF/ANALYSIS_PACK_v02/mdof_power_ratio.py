import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def compute_power_response(m, c_inter, k_inter, f_vals, F_ext, C_ext=None, K_ext=None, grounded_dof=None):
    N = len(m)
    nf = len(f_vals)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))

    for i in range(len(c_inter)):
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

    if grounded_dof is None:
        grounded_dof = N - 1

    X = np.zeros((N, nf), dtype=complex)
    V = np.zeros((N, nf), dtype=complex)
    A = np.zeros((N, nf), dtype=complex)
    P_damp = np.zeros((N-1, nf), dtype=complex)
    P_spring = np.zeros((N-1, nf), dtype=complex)
    Q_mass = np.zeros((N, nf))
    F_bound = np.zeros(nf, dtype=complex)
    P_active_total = np.zeros(nf)
    Q_reactive_total = np.zeros(nf)
    power_ratio = np.zeros(nf)

    for i, f in enumerate(f_vals):
        w = 2 * np.pi * f
        D = K + 1j * w * C - w**2 * M

        A_aug = np.zeros((N+1, N+1), dtype=complex)
        b_aug = np.zeros(N+1, dtype=complex)
        A_aug[:N, :N] = D
        A_aug[grounded_dof, N] = -1.0
        A_aug[N, grounded_dof] = 1.0
        b_aug[:N] = F_ext

        sol = np.linalg.solve(A_aug, b_aug)
        x = sol[:N]
        X[:, i] = x
        F_bound[i] = sol[N]

        v = 1j * w * x
        V[:, i] = v
        A[:, i] = -w**2 * x
        Q_mass[:, i] = w * m * np.abs(v)**2

        P_active_sum = 0
        Q_reactive_sum = 0

        for j in range(N-1):
            dx = x[j+1] - x[j]
            dv = v[j+1] - v[j]
            P_damp[j, i] = c_inter[j] * (dv * np.conj(dv))
            P_spring[j, i] = k_inter[j] * (dx * np.conj(dv))
            P_active_sum += P_damp[j, i].real
            Q_reactive_sum += P_spring[j, i].imag

        Q_reactive_sum += np.sum(Q_mass[:, i])
        P_active_total[i] = P_active_sum
        Q_reactive_total[i] = Q_reactive_sum
        power_ratio[i] = P_active_sum / Q_reactive_sum if Q_reactive_sum != 0 else 0.0

    return {
        'X': X,
        'V': V,
        'A': A,
        'P_damp': P_damp,
        'P_spring': P_spring,
        'Q_mass': Q_mass,
        'F_bound': F_bound,
        'f_vals': f_vals,
        'P_active': P_active_total,
        'Q_reactive': Q_reactive_total,
        'ratio': power_ratio,
        'M': M,
        'K': K
    }

def plot_forced_response_overview(f_vals, X_vals, V_vals, A_vals, P_damp, P_spring, Q_mass, F_bound, m, P_active, Q_reactive, ratio, mode_freq=None):
    N = X_vals.shape[0]
    fig, axs = plt.subplots(3, 3, figsize=(16, 12))

    # Input Power
    V_exc = 1j * (2 * np.pi * f_vals) * X_vals[0, :]
    P_exc = 1.0 * np.conjugate(V_exc)
    axs[0, 0].plot(f_vals, np.real(P_exc), label='Active Power (Excitation)')
    axs[0, 0].plot(f_vals, np.imag(P_exc), label='Reactive Power (Excitation)')
    axs[0, 0].set_title("Input Power")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    for i in range(N):
        axs[0, 1].plot(f_vals, np.abs(V_vals[i, :]), label=f'|v_{i}|')
    axs[0, 1].set_title("Velocity Response")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    for i in range(N):
        axs[0, 2].plot(f_vals, np.abs(A_vals[i, :]), label=f'|a_{i}|')
    axs[0, 2].set_title("Acceleration Response")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    for j in range(P_damp.shape[0]):
        axs[1, 0].plot(f_vals, np.real(P_damp[j, :]), label=f'Damper {j}')
    axs[1, 0].set_title("Active Power in Dampers")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    for j in range(P_spring.shape[0]):
        axs[1, 1].plot(f_vals, np.imag(P_spring[j, :]), label=f'Spring {j}')
    axs[1, 1].set_title("Reactive Power in Springs")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    for i in range(len(m)):
        axs[1, 2].plot(f_vals, Q_mass[i, :], label=f'Mass {i}')
    axs[1, 2].set_title("Inertial Reactive Power per Mass")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    axs[2, 0].plot(f_vals, P_active, label='Total Active Power')
    axs[2, 1].plot(f_vals, Q_reactive, label='Total Reactive Power')
    axs[2, 2].plot(f_vals, ratio, label='Power Ratio (Active/Reactive)')
    for ax in axs[2]:
        if mode_freq:
            ax.axvline(mode_freq, color='r', linestyle='--', label='Target Mode')
        ax.legend()
        ax.grid(True)

    axs[2, 0].set_title("System Active Power")
    axs[2, 1].set_title("System Reactive Power")
    axs[2, 2].set_title("Power Ratio")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example MDOF System
    m = np.array([1.0, 1.0, 1.0, 1.0])
    c_inter = np.array([0.1, 0.1, 0.1])
    k_inter = np.array([100.0, 100.0, 100.0])
    f_vals = np.linspace(1, 60, 1000)

    # Compute eigenmodes
    M = np.diag(m)
    K = np.zeros((4, 4))
    for i in range(len(k_inter)):
        K[i, i] += k_inter[i]
        K[i+1, i+1] += k_inter[i]
        K[i, i+1] -= k_inter[i]
        K[i+1, i] -= k_inter[i]

    eigvals, eigvecs = eigh(K, M)
    freqs_natural = np.sqrt(eigvals) / (2 * np.pi)
    mode_index = 1  # second mode
    mode_shape = eigvecs[:, mode_index]
    mode_shape /= np.max(np.abs(mode_shape))
    freq_mode = freqs_natural[mode_index]

    F_ext = mode_shape.astype(complex)

    result = compute_power_response(m, c_inter, k_inter, f_vals, F_ext)

    plot_forced_response_overview(
        result['f_vals'],
        result['X'],
        result['V'],
        result['A'],
        result['P_damp'],
        result['P_spring'],
        result['Q_mass'],
        result['F_bound'],
        m,
        result['P_active'],
        result['Q_reactive'],
        result['ratio'],
        mode_freq=freq_mode
    )
