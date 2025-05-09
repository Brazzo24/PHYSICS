import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------------------------
# Setup: Simulation Parameters and Constants
# -------------------------------------------------
m_dmf = 5e-3
k_dmf = 1.0e4
c_dmf = 0.2

# Original mass, damping, stiffness arrays
m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
              1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
              2.73e-1, 2.69e+1])
c_inter = np.array([0.05] * 9)
k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
                    2.72e4, 4.97e3, 7.73e2, 8.57e2])

# Insert DMF between DOF0 and rest
m = np.insert(m, 1, m_dmf)
c_inter = np.insert(c_inter, 0, c_dmf)
k_inter = np.insert(k_inter, 0, k_dmf)

# DOF and system size
N = len(m)

# Engine torque signal
angles = np.linspace(0, 720, 1000)
harmonic_coeffs  = [50, 30, 15, 10, 5]
harmonic_freqs   = [1,  2,  3,  4,  5]
harmonic_phases  = [0, 45, 90, 135, 180]

def generate_fourier_torque(crank_angle, coeffs, freqs, phases):
    torque = np.zeros_like(crank_angle, dtype=float)
    for amp, freq, phase in zip(coeffs, freqs, phases):
        torque += amp * np.sin(np.radians(freq * crank_angle + phase))
    return torque

synthetic_torque = generate_fourier_torque(angles, harmonic_coeffs, harmonic_freqs, harmonic_phases)

def convert_crank_angle_to_time(crank_angles, engine_speed_rpm):
    engine_speed_rps = engine_speed_rpm / 60.0
    time_per_rev = 1.0 / engine_speed_rps
    time_per_deg = time_per_rev / 360.0
    return crank_angles * time_per_deg

engine_speed_rpm = 18000
time_signal = convert_crank_angle_to_time(angles, engine_speed_rpm)

# Controller
w_target = 10.0
Kp = 200.0

# Sweep parameters
Fz_values = [0, 1000, 2000, 3000, 4000]  # N
k0_last = k_inter[-1]                   # base stiffness
alpha = 0.05                            # Nm/rad per N

def build_full_matrices(m, c_inter, k_inter):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N-1):
        C[i, i]     += c_inter[i]
        C[i, i+1]   -= c_inter[i]
        C[i+1, i]   -= c_inter[i]
        C[i+1, i+1] += c_inter[i]
        K[i, i]     += k_inter[i]
        K[i, i+1]   -= k_inter[i]
        K[i+1, i]   -= k_inter[i]
        K[i+1, i+1] += k_inter[i]
    return M, C, K

def engine_driveline_ode_factory(F_z):
    def engine_driveline_ode(t, y, M, C, K):
        x = y[:N]
        v = y[N:]
        T_engine = np.interp(t, time_signal, synthetic_torque)
        T_fb = -Kp * (v[-1] - w_target)
        f = np.zeros(N)
        f[0] = T_engine
        f[-1] = T_fb
        spring_force = K @ x

        # Modify last stiffness element
        k_last = k0_last + alpha * F_z
        dx = x[-1] - x[-2]
        f_k = k_last * dx
        spring_force[-2] += f_k - k0_last * dx
        spring_force[-1] -= f_k - k0_last * dx

        a = np.linalg.solve(M, f - C @ v - spring_force)
        return np.concatenate((v, a))
    return engine_driveline_ode

def compute_fft(signal, t):
    N = len(signal)
    dt = t[1] - t[0]
    window = np.hanning(N)
    signal_win = signal * window
    fft_result = np.fft.fft(signal_win)
    freqs = np.fft.fftfreq(N, d=dt)
    pos_idx = np.where(freqs >= 0)
    return freqs[pos_idx], np.abs(fft_result[pos_idx])

# -------------------------------------------------
# Run Simulations for Each F_z Value
# -------------------------------------------------
fft_results = {}
M, C, K = build_full_matrices(m, c_inter, k_inter)
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
x0 = np.zeros(N)
v0 = np.zeros(N)
y0 = np.concatenate((x0, v0))

for F_z in Fz_values:
    ode_fun = engine_driveline_ode_factory(F_z)
    sol = solve_ivp(
        fun=lambda t, y: ode_fun(t, y, M, C, K),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )

    x_sol = sol.y[:N, :]
    v_sol = sol.y[N:, :]
    a_sol = np.zeros_like(x_sol)

    for i in range(len(sol.t)):
        t_i = sol.t[i]
        f_t = np.zeros(N)
        f_t[0] = np.interp(t_i, time_signal, synthetic_torque)
        f_t[-1] = -Kp * (v_sol[-1, i] - w_target)

        dx = x_sol[-1, i] - x_sol[-2, i]
        k_last = k0_last + alpha * F_z
        f_t[-2] += (k_last - k0_last) * dx
        f_t[-1] -= (k_last - k0_last) * dx

        a_sol[:, i] = np.linalg.solve(M, f_t - C @ v_sol[:, i] - K @ x_sol[:, i])

    fft_results[F_z] = compute_fft(a_sol[-1, :], sol.t)

# -------------------------------------------------
# Plot FFT for DOF N-1 Acceleration Across F_z
# -------------------------------------------------
plt.figure(figsize=(10, 6))
for F_z in Fz_values:
    freqs, fft_mag = fft_results[F_z]
    plt.plot(freqs, fft_mag, label=f'Fz = {F_z} N')
plt.xlabel('Frequency [Hz]')
plt.ylabel('FFT Magnitude (DOF N-1 accel)')
plt.title('FFT of DOF N-1 Acceleration for Different Fz')
plt.xlim(0, 500)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
