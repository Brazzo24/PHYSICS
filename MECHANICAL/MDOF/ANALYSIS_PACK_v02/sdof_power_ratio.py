import numpy as np
import matplotlib.pyplot as plt

def compute_sdoff_power_sweep(k, damping_ratio, m_vals):
    P_active = []
    Q_reactive = []
    ratio = []

    for m in m_vals:
        c = 2 * damping_ratio * np.sqrt(k * m)
        omega_n = np.sqrt(k / m)
        omega_d = omega_n * np.sqrt(1 - damping_ratio**2)
        w = omega_d

        # Frequency response at damped natural frequency
        D = k + 1j * w * c - w**2 * m
        x = 1.0 / D  # Assume excitation force = 1.0 (unit amplitude)
        v = 1j * w * x

        # Powers
        p_active = c * (v * np.conj(v)).real
        dx = x  # in single DOF, dx is same as x
        p_spring = k * (dx * np.conj(v)).imag
        q_mass = w * m * np.abs(v)**2

        total_reactive = p_spring + q_mass
        power_ratio = p_active / total_reactive if total_reactive != 0 else 0.0

        P_active.append(p_active)
        Q_reactive.append(total_reactive)
        ratio.append(power_ratio)

    return np.array(P_active), np.array(Q_reactive), np.array(ratio)

def plot_sdoff_power_results(m_vals, P_active, Q_reactive, ratio):
    plt.figure(figsize=(10, 6))
    plt.plot(m_vals, P_active, label='Active Power [W]')
    plt.plot(m_vals, Q_reactive, label='Reactive Power [VAR]')
    plt.plot(m_vals, ratio, label='Power Ratio (Active/Reactive)')
    plt.xlabel('Mass [kg]')
    plt.ylabel('Power / Ratio')
    plt.title('Single DOF System: Power Analysis vs. Mass')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # User settings
    k = 100.0  # N/m
    damping_ratio = 0.05  # constant damping ratio
    m_vals = np.linspace(0.1, 10.0, 200)  # sweep mass

    P_active, Q_reactive, ratio = compute_sdoff_power_sweep(k, damping_ratio, m_vals)
    plot_sdoff_power_results(m_vals, P_active, Q_reactive, ratio)
