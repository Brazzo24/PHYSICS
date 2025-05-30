# Rebuild the system with the velocity profile applied at the last DOF (Dyno side)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define inertia values (kg·m²)
m = np.array([1.21e-4, 3.95e-4, 7.92e-4,
              1.02e-3, 1.42e-3,
              3.35e-2, 1.09e+1])  # last DOF = Dyno
N = len(m)

# Damping and stiffness
c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
k_inter = np.array([2.34e4, 1.62e5, 1.62e5, 1.62e5, 4.73e2, 9.57e2])

# Build system matrices
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

# Velocity profile applied at last DOF (Dyno)
def velocity_profile(t):
    rpm = np.clip(1200 * t, 0, 6000)  # ramp from 0 to 3000 RPM over 5s
    return (rpm * 2 * np.pi) / 60.0  # convert to rad/s

# ODE function with velocity constraint at last DOF
# Apply a more moderate gain and switch to a stiff solver
def velocity_driven_ode_dyno(t, y, M, C, K):
    x = y[:N]
    v = y[N:]

    # Moderate gain to enforce velocity profile at dyno
    v_target = velocity_profile(t)
    T_dyno = 1000.0 * (v_target - v[-1])

    f_ext = np.zeros(N)
    f_ext[-1] = T_dyno

    a = np.linalg.solve(M, f_ext - C @ v - K @ x)
    return np.concatenate((v, a))

# Initial conditions and time setup
y0 = np.zeros(2 * N)
t_eval = np.linspace(0, 50, 6000)

# Re-simulate with improved solver and gain
sol = solve_ivp(
    fun=lambda t, y: velocity_driven_ode_dyno(t, y, M, C, K),
    t_span=(0, 50),
    y0=y0,
    t_eval=t_eval,
    method='Radau'  # Stiff solver
)

# Plot the results
v_sol = sol.y[N:, :]
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(sol.t, v_sol[i], label=f'DOF {i}')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.title('System Response to Velocity Run-Up at Dyno (Last DOF, Stiff Solver)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from scipy.signal import stft

# Compute crank speed in RPM
rpm_signal = v_sol[-1] * 60 / (2 * np.pi)  # convert dyno rad/s to RPM

# Use twist between last two elements (DOF 5 to DOF 6) as vibration signal
twist_5_6 = (sol.y[5] - sol.y[6])  # radians

# Sampling frequency
dt = t_eval[1] - t_eval[0]
fs = 1 / dt

# Compute STFT
f_stft, t_stft, Zxx = stft(twist_5_6, fs=fs, nperseg=256, noverlap=200, window='hann')

# Map STFT time to RPM from dyno
rpm_interp = np.interp(t_stft, sol.t, rpm_signal)

# Plot frequency vs. RPM heatmap (magnitude of STFT)
plt.figure(figsize=(10, 6))
plt.pcolormesh(rpm_interp, f_stft, np.abs(Zxx), shading='gouraud')
plt.xlabel('Dyno Speed [RPM]')
plt.ylabel('Frequency [Hz]')
plt.title('Shaft Vibration vs. Dyno RPM (STFT of DOF5-DOF6 Twist)')
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()


# Compute angular velocity fluctuation for m[0] and m[1]
v_0 = v_sol[0]  # DOF 0
v_1 = v_sol[1]  # DOF 1

# Subtract mean to highlight fluctuations
v_0_fluct = v_0 - np.mean(v_0)
v_1_fluct = v_1 - np.mean(v_1)

# STFT for DOF 0
f0, t0, Z0 = stft(v_0_fluct, fs=fs, nperseg=256, noverlap=200, window='hann')
rpm_interp_0 = np.interp(t0, sol.t, rpm_signal)

# STFT for DOF 1
f1, t1, Z1 = stft(v_1_fluct, fs=fs, nperseg=256, noverlap=200, window='hann')
rpm_interp_1 = np.interp(t1, sol.t, rpm_signal)

# Plot STFT of angular velocity fluctuation at DOF 0
plt.figure(figsize=(10, 6))
plt.pcolormesh(rpm_interp_0, f0, np.abs(Z0), shading='gouraud', vmin =0, vmax=0.02)
plt.xlabel('Dyno Speed [RPM]')
plt.ylabel('Frequency [Hz]')
plt.title('Angular Velocity Fluctuation at DOF 0 vs. Dyno RPM')
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()

# Plot STFT of angular velocity fluctuation at DOF 1
plt.figure(figsize=(10, 6))
plt.pcolormesh(rpm_interp_1, f1, np.abs(Z1), shading='gouraud', vmin =0, vmax=0.02)
plt.xlabel('Dyno Speed [RPM]')
plt.ylabel('Frequency [Hz]')
plt.title('Angular Velocity Fluctuation at DOF 1 vs. Dyno RPM')
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()
