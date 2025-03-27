import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) System Definition (same as before)
# -------------------------------------------------
m = np.array([1.0, 1.0, 100.0])  # 3-DOF masses
c_inter = np.array([0.5, 0.3])   # damping between (DOF0-DOF1) and (DOF1-DOF2)
k_inter = np.array([1.0e3, 5.0e2])  # stiffness between (DOF0-DOF1) and (DOF1-DOF2)

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

engine_speed_rpm = 18000
time_signal = convert_crank_angle_to_time(angles, engine_speed_rpm)

# -------------------------------------------------
# 3) Define Our Feedback Parameters
# -------------------------------------------------
w_target = 10.0   # [rad/s] target speed for DOF2
Kp = 200.0        # Proportional gain

# -------------------------------------------------
# 4) ODE Function with Feedback Controller
# -------------------------------------------------
def engine_driveline_ode(t, y, M, C, K):
    """
    y = [x0, x1, x2, v0, v1, v2] for a 3-DOF system.
    Forcing:
      - DOF0: engine torque (interpolated from synthetic_torque)
      - DOF1: no external torque
      - DOF2: feedback torque to hold w_target
    """
    N = len(m)  # 3
    x = y[:N]
    v = y[N:]
    
    # Interpolate engine torque at DOF0:
    T_engine = np.interp(t, time_signal, synthetic_torque)
    
    # Feedback torque at DOF2:
    T_fb = -Kp * (v[2] - w_target)
    
    # Forcing vector:
    f = np.array([T_engine, 0.0, T_fb])
    
    # Compute acceleration:
    a = np.linalg.solve(M, f - C @ v - K @ x)
    
    return np.concatenate((v, a))

# -------------------------------------------------
# 5) Run the Simulation
# -------------------------------------------------
M, C, K = build_full_matrices(m, c_inter, k_inter)
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 2000)
x0 = np.zeros(3)
v0 = np.zeros(3)
y0 = np.concatenate((x0, v0))

sol = solve_ivp(
    fun=lambda t, y: engine_driveline_ode(t, y, M, C, K),
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='RK45'
)

x_sol = sol.y[:3, :]  # displacements for DOF0, DOF1, DOF2
v_sol = sol.y[3:, :]  # velocities for DOF0, DOF1, DOF2

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
    a_sol[:, i] = np.linalg.solve(M, f_t - C @ v_sol[:, i] - K @ x_sol[:, i])
    
    # Compute spring forces:
    # Spring 0 (between DOF0 and DOF1)
    spring_force[0, i] = k_inter[0] * (x_sol[1, i] - x_sol[0, i])
    # Spring 1 (between DOF1 and DOF2)
    spring_force[1, i] = k_inter[1] * (x_sol[2, i] - x_sol[1, i])
    
    # Compute damper forces:
    # Damper 0 (between DOF0 and DOF1)
    damper_force[0, i] = c_inter[0] * (v_sol[1, i] - v_sol[0, i])
    # Damper 1 (between DOF1 and DOF2)
    damper_force[1, i] = c_inter[1] * (v_sol[2, i] - v_sol[1, i])
    
    # Compute instantaneous damping power: P = c * (Δv)^2
    damping_power[0, i] = c_inter[0] * (v_sol[1, i] - v_sol[0, i])**2
    damping_power[1, i] = c_inter[1] * (v_sol[2, i] - v_sol[1, i])**2

# -------------------------------------------------
# 7) Plot the Results
# -------------------------------------------------
# Plot acceleration for each DOF:
plt.figure(figsize=(10, 4))
plt.plot(sol.t, a_sol[0, :], label='DOF0 Acceleration')
plt.plot(sol.t, a_sol[1, :], label='DOF1 Acceleration')
plt.plot(sol.t, a_sol[2, :], label='DOF2 Acceleration')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [rad/s²]')
plt.title('Acceleration of Each DOF')
plt.legend()
plt.grid(True)
plt.show()

# Plot spring forces
plt.figure(figsize=(10, 4))
plt.plot(sol.t, spring_force[0, :], label='Spring Force (DOF0-DOF1)')
plt.plot(sol.t, spring_force[1, :], label='Spring Force (DOF1-DOF2)')
plt.xlabel('Time [s]')
plt.ylabel('Force [Nm]')
plt.title('Spring Forces Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plot damping power
plt.figure(figsize=(10, 4))
plt.plot(sol.t, damping_power[0, :], label='Damping Power (DOF0-DOF1)')
plt.plot(sol.t, damping_power[1, :], label='Damping Power (DOF1-DOF2)')
plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
plt.title('Instantaneous Power Dissipated in Dampers')
plt.legend()
plt.grid(True)
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
