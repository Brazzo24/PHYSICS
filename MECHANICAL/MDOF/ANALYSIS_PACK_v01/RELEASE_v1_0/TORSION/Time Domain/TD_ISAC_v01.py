import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) System Definition (same as before)
# -------------------------------------------------
# m = np.array([1.0, 1.0, 100.0])  # 3-DOF masses
# c_inter = np.array([0.5, 0.3])   # damping between (DOF0-DOF1) and (DOF1-DOF2)
# k_inter = np.array([1.0e3, 5.0e2])  # stiffness between (DOF0-DOF1) and (DOF1-DOF2)

# # ([Crankshaft, CRCS, PG, Clutch 1, Clutch 2, Input, Output, Hub, Wheel, Road])
# m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
#                 1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
#                 2.73e-1, 2.69e+1])  # kgm^2



# # ([Gear, Gear, Primary Damper, Clutch, Spline, GBX, Chain, RWD, Tyre])
# c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nm.s/rad

# k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
#                     2.72e4, 4.97e3, 7.73e2, 8.57e2]) # Nm/rad


# Original definition of m, c_inter, k_inter

# Add DMF between DOF0 and the rest
m_dmf = 5e-3
k_dmf = 1.0e4
c_dmf = 0.2

m = np.array([1.21e-2, 3.95e-4, 4.438e-4,
                9.044e-4, 7.173e-4, 2.639e-4, 7.257e-4, 3.88e-4,
                1.244e-2]) # kgm^2
c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nms/rad
k_inter = np.array([2.34e4, 1.62e5, 1.112e3, 1.10e5, 1.10e5,
                    2.72e4, 1.94e3, 9.127e1])   # Nm/rad

# Insert DMF at index 1
# m = np.insert(m, 1, m_dmf)
# c_inter = np.insert(c_inter, 0, c_dmf)
# k_inter = np.insert(k_inter, 0, k_dmf)


def build_full_matrices(m, c_inter, k_inter):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N-1):
        # Damping matrix
        C[i, i]     += c_inter[i]
        C[i, i+1]   -= c_inter[i]
        C[i+1, i]   -= c_inter[i]
        C[i+1, i+1] += c_inter[i]
        # Stiffness matrix
        K[i, i]     += k_inter[i]
        K[i, i+1]   -= k_inter[i]
        K[i+1, i]   -= k_inter[i]
        K[i+1, i+1] += k_inter[i]
    return M, C, K

# -------------------------------------------------
# 2) Generate Synthetic Engine Torque (same idea)
# -------------------------------------------------
def generate_fourier_torque(crank_angle, coeffs, freqs, phases):
    torque = np.zeros_like(crank_angle, dtype=float)
    for amp, freq, phase in zip(coeffs, freqs, phases):
        torque += amp * np.sin(np.radians(freq * crank_angle + phase))
    return torque

# Sinusoidal forcing on DOF N-2
amp_sine = 20.0        # [Nm] amplitude of sinusoidal forcing
freq_sine = 10.0       # [Hz] frequency of sinusoidal forcing
phase_sine = 0.0       # [deg] phase shift


angles = np.linspace(0, 720, 1000)  # 0-720 deg
harmonic_coeffs  = [50, 30, 15, 10, 5]
harmonic_freqs   = [1,  2,  3,  4,  5]
harmonic_phases  = [0, 45, 90, 135, 180]
synthetic_torque = generate_fourier_torque(angles, harmonic_coeffs, harmonic_freqs, harmonic_phases)

def convert_crank_angle_to_time(crank_angles, engine_speed_rpm):
    engine_speed_rps = engine_speed_rpm / 60.0
    time_per_rev = 1.0 / engine_speed_rps
    time_per_deg = time_per_rev / 360.0
    return crank_angles * time_per_deg

engine_speed_rpm = 9000
time_signal = convert_crank_angle_to_time(angles, engine_speed_rpm)

# -------------------------------------------------
# 3) Define Our Feedback Parameters
# -------------------------------------------------
w_target = 10.0   # [rad/s] target speed for DOF2
Kp = 200.0        # Proportional gain

# -------------------------------------------------
# 4) ODE Function with Feedback Controller
# -------------------------------------------------
# def engine_driveline_ode(t, y, M, C, K):
#     """
#     y = [x0, x1, x2, v0, v1, v2] for a 3-DOF system.
#     Forcing:
#       - DOF0: engine torque (interpolated from synthetic_torque)
#       - DOF1: no external torque
#       - DOF2: feedback torque to hold w_target
#     """
 
#     N = len(m)
#     x = y[:N]
#     v = y[N:]
    
#     T_engine = np.interp(t, time_signal, synthetic_torque)
#     T_fb = -Kp * (v[-1] - w_target)  # Apply feedback on last DOF

#     f = np.zeros(N)
#     f[0] = T_engine
#     f[-1] = T_fb

#     a = np.linalg.solve(M, f - C @ v - K @ x)
#     return np.concatenate((v, a))

def engine_driveline_ode(t, y, M, C, K):
    N = len(m)
    x = y[:N]
    v = y[N:]
    
    # Engine torque (DOF0)
    T_engine = np.interp(t, time_signal, synthetic_torque)
    
    # Feedback torque (DOF N-1)
    T_fb = -Kp * (v[-1] - w_target)
    
    # New sinusoidal forcing at DOF N-2
    #  T_sine = amp_sine * np.sin(2 * np.pi * freq_sine * t + np.radians(phase_sine))
    T_sine = 0

    # External force vector
    f = np.zeros(N)
    f[0] = T_engine
    f[N - 2] = T_sine
    f[-1] = T_fb

    a = np.linalg.solve(M, f - C @ v - K @ x)
    return np.concatenate((v, a))

# -------------------------------------------------
# 5) Run the Simulation
# -------------------------------------------------
M, C, K = build_full_matrices(m, c_inter, k_inter)
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
N = len(m)
x0 = np.zeros(N)
v0 = np.zeros(N)
y0 = np.concatenate((x0, v0))

sol = solve_ivp(
    fun=lambda t, y: engine_driveline_ode(t, y, M, C, K),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45'
)

x_sol = sol.y[:N, :]  # displacements for DOF0, DOF1, DOF2
v_sol = sol.y[N:, :]  # velocities for DOF0, DOF1, DOF2

# -------------------------------------------------
# 6) Post-Processing: Compute Acceleration, Spring Forces, and Damping Power
# -------------------------------------------------
num_steps = len(sol.t)
a_sol = np.zeros_like(x_sol)  # To store acceleration (3 x num_steps)

# For the springs: there are 2 springs (between DOF0-DOF1 and DOF1-DOF2)
spring_force = np.zeros((2, num_steps))
# For the dampers: there are 2 dampers
damper_force = np.zeros((2, num_steps))
damping_power = np.zeros((2, num_steps))

for i in range(num_steps):
    t_i = sol.t[i]
    # Compute acceleration at time t_i using the same formula as in the ODE:
    f_t = np.array([np.interp(t_i, time_signal, synthetic_torque), 0.0, -Kp * (v_sol[2, i] - w_target)])
    a_sol = np.zeros_like(x_sol)  
    spring_force = np.zeros((N - 1, num_steps))
    damper_force = np.zeros((N - 1, num_steps))
    damping_power = np.zeros((N - 1, num_steps))

    for i in range(num_steps):
        t_i = sol.t[i]
        f_t = np.zeros(N)
        f_t[0] = np.interp(t_i, time_signal, synthetic_torque)
        f_t[-1] = -Kp * (v_sol[-1, i] - w_target)
        a_sol[:, i] = np.linalg.solve(M, f_t - C @ v_sol[:, i] - K @ x_sol[:, i])
        
        for j in range(N - 1):
            spring_force[j, i] = k_inter[j] * (x_sol[j+1, i] - x_sol[j, i])
            damper_force[j, i] = c_inter[j] * (v_sol[j+1, i] - v_sol[j, i])
            damping_power[j, i] = c_inter[j] * (v_sol[j+1, i] - v_sol[j, i])**2

# -------------------------------------------------
# 7) Plot the Results
# -------------------------------------------------
# Plot acceleration for each DOF:
plt.figure(figsize=(10, 3 * N))
for i in range(N):
    plt.subplot(N, 1, i + 1)
    plt.plot(sol.t, a_sol[i, :], label=f'DOF{i} Acceleration')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [rad/s²]')
    plt.title(f'DOF{i} Acceleration Over Time')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

# Plot spring forces for each connected DOF pair:
plt.figure(figsize=(10, 3 * (N - 1)))
for i in range(N - 1):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(sol.t, spring_force[i, :], label=f'Spring Force (DOF{i}-DOF{i+1})')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [Nm]')
    plt.title(f'Spring Force Between DOF{i} and DOF{i+1}')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

# Plot damping power for each damper:
plt.figure(figsize=(10, 3 * (N - 1)))
for i in range(N - 1):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(sol.t, damping_power[i, :], label=f'Damping Power (DOF{i}-DOF{i+1})')
    plt.xlabel('Time [s]')
    plt.ylabel('Power [W]')
    plt.title(f'Damping Power Between DOF{i} and DOF{i+1}')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 8) FFT Analysis for Acceleration Signals with a Hanning Window
# -------------------------------------------------
def compute_fft(signal, t):
    """
    Compute the FFT of a 1-D signal using a Hanning window.
    
    Parameters:
        signal (array): 1D array (e.g., acceleration for a single DOF)
        t (array): time vector corresponding to the signal
        
    Returns:
        freqs (array): frequencies (only positive frequencies)
        fft_mag (array): magnitude of the FFT (absolute value)
    """
    N = len(signal)
    dt = t[1] - t[0]         # sampling time step
    window = np.hanning(N)   # Hanning window
    signal_win = signal * window
    fft_result = np.fft.fft(signal_win)
    freqs = np.fft.fftfreq(N, d=dt)
    
    # Only keep positive frequencies
    pos_idx = np.where(freqs >= 0)
    return freqs[pos_idx], np.abs(fft_result[pos_idx])

# Compute and plot FFT for each DOF acceleration signal
num_dofs = a_sol.shape[0]  # should be 3 in your case

plt.figure(figsize=(10, 3 * num_dofs))
for i in range(num_dofs):
    freqs, fft_mag = compute_fft(a_sol[i, :], sol.t)
    plt.subplot(num_dofs, 1, i+1)
    plt.plot(freqs, fft_mag, label=f'FFT of Acceleration DOF{i}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(f'FFT of Acceleration Signal for DOF{i}')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
