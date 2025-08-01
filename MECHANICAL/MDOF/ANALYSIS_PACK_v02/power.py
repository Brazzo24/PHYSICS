import numpy as np
import matplotlib.pyplot as plt

def compute_power_response(m, c_inter, k_inter, f_vals, F_ext, C_ext=None, K_ext=None, grounded_dof=None):
    """
    Computes complex power flow components and system response.

    Parameters:
        m           : (N,) array of masses/inertias
        c_inter     : (N-1,) damping between adjacent DOFs
        k_inter     : (N-1,) stiffness between adjacent DOFs
        f_vals      : (nf,) array of frequency values [Hz]
        F_ext       : (N,) complex array, force input
        C_ext       : optional (N,N) external damping matrix (e.g., branch or grounding)
        K_ext       : optional (N,N) external stiffness matrix
        grounded_dof: index of the DOF to ground for well-posed augmented system

    Returns:
        result_dict with keys:
            'X'         : (N, nf) displacement
            'V'         : (N, nf) velocity
            'A'         : (N, nf) acceleration
            'P_damp'    : (N-1, nf) active power in dampers
            'P_spring'  : (N-1, nf) reactive power in springs
            'Q_mass'    : (N, nf) inertial reactive power
            'F_bound'   : (nf,) constraint reaction force at grounded DOF
            'P_active'  : (nf,) total system active power
            'Q_reactive': (nf,) total system reactive power
            'ratio'     : (nf,) active/reactive ratio
            'f_vals'    : frequency vector
    """
    m = np.asarray(m)
    c_inter = np.asarray(c_inter)
    k_inter = np.asarray(k_inter)

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
    ratio = np.zeros(nf)

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

        for j in range(N-1):
            dx = x[j+1] - x[j]
            dv = v[j+1] - v[j]
            P_damp[j, i] = c_inter[j] * (dv * np.conj(dv))
            P_spring[j, i] = k_inter[j] * (dx * np.conj(dv))

        P_active_total[i] = np.sum(np.real(P_damp[:, i]))
        Q_reactive_total[i] = np.sum(np.imag(P_spring[:, i])) + np.sum(Q_mass[:, i])

        ratio[i] = P_active_total[i] / Q_reactive_total[i] if Q_reactive_total[i] != 0 else 0.0

    return {
        'X': X,
        'V': V,
        'A': A,
        'P_damp': P_damp,
        'P_spring': P_spring,
        'Q_mass': Q_mass,
        'F_bound': F_bound,
        'P_active': P_active_total,
        'Q_reactive': Q_reactive_total,
        'ratio': ratio,
        'f_vals': f_vals
    }

def plot_forced_response_overview(f_vals, X_vals, V_vals, A_vals, P_damp, P_spring, Q_mass, F_bound, m, P_active, Q_reactive, ratio):
    N = X_vals.shape[0]
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))

    # Input Power
    V_exc = 1j * (2 * np.pi * f_vals) * X_vals[0, :]
    P_exc = 1.0 * np.conjugate(V_exc)
    axs[0, 0].plot(f_vals, np.real(P_exc), label='Active Power (Excitation)')
    axs[0, 0].plot(f_vals, np.imag(P_exc), label='Reactive Power (Excitation)')
    axs[0, 0].set_xlabel('Frequency [Hz]')
    axs[0, 0].set_ylabel('Power [W][VAR]')
    axs[0, 0].set_title('Input Power')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Velocity Response
    for i in range(N):
        axs[0, 1].plot(f_vals, np.abs(V_vals[i, :]), label=f'|v_{i}|')
    axs[0, 1].set_xlabel('Frequency [Hz]')
    axs[0, 1].set_ylabel('Velocity [rad/s]')
    axs[0, 1].set_title('Velocity Response')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Acceleration Response
    for i in range(N):
        axs[0, 2].plot(f_vals, np.abs(A_vals[i, :]), label=f'|a_{i}|')
    axs[0, 2].set_xlabel('Frequency [Hz]')
    axs[0, 2].set_ylabel('Acceleration [rad/s²]')
    axs[0, 2].set_title('Acceleration Response')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Power in Dampers
    for j in range(P_damp.shape[0]):
        axs[1, 0].plot(f_vals, np.real(P_damp[j, :]), label=f'Damper {j}')
    axs[1, 0].set_xlabel('Frequency [Hz]')
    axs[1, 0].set_ylabel('Active Power [W]')
    axs[1, 0].set_title('Active Power in Dampers')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Power in Springs
    for j in range(P_spring.shape[0]):
        axs[1, 1].plot(f_vals, np.imag(P_spring[j, :]), label=f'Spring {j}')
    axs[1, 1].set_xlabel('Frequency [Hz]')
    axs[1, 1].set_ylabel('Reactive Power [VAR]')
    axs[1, 1].set_title('Reactive Power in Springs')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Inertial Reactive Power
    for i in range(len(m)):
        axs[1, 2].plot(f_vals, Q_mass[i, :], label=f'Mass {i}')
    axs[1, 2].set_xlabel('Frequency [Hz]')
    axs[1, 2].set_ylabel('Reactive Power [VAR]')
    axs[1, 2].set_title('Inertial Reactive Power per Mass')
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Energy Balance Plot
    axs[2, 0].plot(f_vals, P_active, label='Total Active Power [W]')
    axs[2, 0].plot(f_vals, Q_reactive, label='Total Reactive Power [VAR]')
    axs[2, 0].plot(f_vals, ratio, label='Power Ratio (Active/Reactive)')
    axs[2, 0].set_xlabel('Frequency [Hz]')
    axs[2, 0].set_ylabel('Power / Ratio')
    axs[2, 0].set_title('System Energy Balance')
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    for ax in axs[2, 1:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    m = np.array([1.0, 1.0, 1.0])
    c_inter = np.array([0.2, 0.2])
    k_inter = np.array([1000.0, 500.0])
    f_vals = np.linspace(1, 50, 500)
    F_ext = np.array([1.0, 0.0, 0.0], dtype=complex)

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
        result['ratio']
    )
