# Re-import necessary libraries after kernel reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import stft

# System setup
m = np.array([1.21e-4, 3.95e-4, 7.92e-4,
              1.02e-3, 1.42e-3,
              3.35e-2, 1.09e+1])
N = len(m)
c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.5])
k_inter = np.array([2.34e4, 1.62e5, 1.62e5, 1.62e5, 4.73e2, 9.57e2])

# Build matrices
def build_full_matrices(m, c_inter, k_inter):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N - 1):
        C[i, i]     += c_inter[i]
        C[i + 1, i + 1] += c_inter[i]
        C[i, i + 1] -= c_inter[i]
        C[i + 1, i] -= c_inter[i]
        K[i, i]     += k_inter[i]
        K[i + 1, i + 1] += k_inter[i]
        K[i, i + 1] -= k_inter[i]
        K[i + 1, i] -= k_inter[i]
    return M, C, K

M, C, K = build_full_matrices(m, c_inter, k_inter)

# Fourier torque input
def generate_fourier_torque(crank_angle, coeffs, freqs, phases):
    torque = np.zeros_like(crank_angle)
    for amp, freq, phase in zip(coeffs, freqs, phases):
        torque += amp * np.sin(np.radians(freq * crank_angle + phase))
    return torque

angles = np.linspace(0, 720, 1000)
harmonic_coeffs = [50, 30, 15, 10, 5]
harmonic_freqs = [1, 2, 3, 4, 5]
harmonic_phases = [0, 45, 90, 135, 180]
synthetic_torque = generate_fourier_torque(angles, harmonic_coeffs, harmonic_freqs, harmonic_phases)

def convert_crank_angle_to_time(crank_angles, engine_speed_rpm):
    engine_speed_rps = engine_speed_rpm / 60.0
    time_per_rev = 1.0 / engine_speed_rps
    time_per_deg = time_per_rev / 360.0
    return crank_angles * time_per_deg

engine_speed_rpm = 3000
time_signal = convert_crank_angle_to_time(angles, engine_speed_rpm)

# ODE function
def torque_driven_ode(t, y, M, C, K):
    x = y[:N]
    v = y[N:]
    T_engine = np.interp(t, time_signal, synthetic_torque)
    f_ext = np.zeros(N)
    f_ext[0] = T_engine
    a = np.linalg.solve(M, f_ext - C @ v - K @ x)
    return np.concatenate((v, a))

# Time settings
# t_eval = np.linspace(0, 5, 3000)
# fs = 1 / (t_eval[1] - t_eval[0])
# y0 = np.zeros(2 * N)

# # Solve system
# sol = solve_ivp(
#     fun=lambda t, y: torque_driven_ode(t, y, M, C, K),
#     t_span=(0, 5),
#     y0=y0,
#     t_eval=t_eval,
#     method='RK45'
# )

# # STFT
# twist_5_6 = sol.y[5] - sol.y[6]
# f_stft, t_stft, Zxx = stft(twist_5_6, fs=fs, nperseg=256, noverlap=200, window='hann')

# # Plot STFT
# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.title('Twist DOF5–DOF6 (Torque-Driven System Response)')
# plt.colorbar(label='Amplitude')
# plt.tight_layout()
# plt.show()

##
# SHORTE SIMULATION WITH STIFFER SOLVER
###

# Shorter simulation and stiff solver
short_t_eval = np.linspace(0, 1.5, 2000)
fs_short = 1 / (short_t_eval[1] - short_t_eval[0])
y0 = np.zeros(2 * N)

# Solve with Radau (stiff) method
sol = solve_ivp(
    fun=lambda t, y: torque_driven_ode(t, y, M, C, K),
    t_span=(0, 1.5),
    y0=y0,
    t_eval=short_t_eval,
    method='Radau'
)

# STFT analysis of twist between DOF 5 and 6
twist_5_6 = sol.y[5] - sol.y[6]
f_stft, t_stft, Zxx = stft(twist_5_6, fs=fs_short, nperseg=256, noverlap=200, window='hann')

# Plot updated STFT result
plt.figure(figsize=(10, 6))
plt.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Twist DOF5–DOF6 (Torque-Driven System, Short Simulation)')
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()

# Interpolate dyno RPM over STFT time points
v_dyno = sol.y[-1]
rpm_dyno = v_dyno * 60 / (2 * np.pi)  # convert to RPM
rpm_interp = np.interp(t_stft, sol.t, rpm_dyno)

# Compute magnitude and remove low-frequency "floor" by thresholding
Z_mag = np.abs(Zxx)
Z_mag[Z_mag < 1e-4] = 0  # suppress near-zero noise artifacts

# Plot STFT result with RPM on x-axis
plt.figure(figsize=(10, 6))
plt.pcolormesh(rpm_interp, f_stft, Z_mag, shading='gouraud')
plt.xlabel('Dyno Speed [RPM]')
plt.ylabel('Frequency [Hz]')
plt.title('Twist DOF5–DOF6 vs. Dyno RPM (Torque-Driven STFT)')
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()
