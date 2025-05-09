import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Toggle to enable non-linear spring at the last element
use_nonlinear = True
k_nl = 1.0e5  # [Nm/rad^3] Non-linear stiffness coefficient

# -------------------------------------------------
# 1) System Definition
# -------------------------------------------------
m_dmf = 5e-3
k_dmf = 1.0e4
c_dmf = 0.2

m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
              1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
              2.73e-1, 2.69e+1])
c_inter = np.array([0.05] * 9)
k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
                    2.72e4, 4.97e3, 7.73e2, 8.57e2])

# Insert DMF at index 1
m = np.insert(m, 1, m_dmf)
c_inter = np.insert(c_inter, 0, c_dmf)
k_inter = np.insert(k_inter, 0, k_dmf)

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

# -------------------------------------------------
# 2) Generate Synthetic Engine Torque
# -------------------------------------------------
def generate_fourier_torque(crank_angle, coeffs, freqs, phases):
    torque = np.zeros_like(crank_angle, dtype=float)
    for amp, freq, phase in zip(coeffs, freqs, phases):
        torque += amp * np.sin(np.radians(freq * crank_angle + phase))
    return torque

angles = np.linspace(0, 720, 1000)
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
# 3) Feedback Controller
# -------------------------------------------------
w_target = 10.0
Kp = 200.0

# -------------------------------------------------
# 4) ODE Function with Optional Nonlinear Spring
# -------------------------------------------------
def engine_driveline_ode(t, y, M, C, K):
    N = len(m)
    x = y[:N]
    v = y[N:]
    T_engine = np.interp(t, time_signal, synthetic_torque)
    T_fb = -Kp * (v[-1] - w_target)
    f = np.zeros(N)
    f[0] = T_engine
    f[-1] = T_fb
    spring_force = K @ x
    if use_nonlinear:
        dx = x[-1] - x[-2]
        k_linear = k_inter[-1]
        f_lin = k_linear * dx
        f_nl = k_linear * dx + k_nl * dx**3
        spring_force[-2] -= f_lin
        spring_force[-1] += f_lin
        spring_force[-2] += f_nl
        spring_force[-1] -= f_nl
    a = np.linalg.solve(M, f - C @ v - spring_force)
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

x_sol = sol.y[:N, :]
v_sol = sol.y[N:, :]

# -------------------------------------------------
# 6) Post-Processing
# -------------------------------------------------
num_steps = len(sol.t)
a_sol = np.zeros_like(x_sol)
spring_force = np.zeros((N - 1, num_steps))
damper_force = np.zeros((N - 1, num_steps))
damping_power = np.zeros((N - 1, num_steps))

for i in range(num_steps):
    t_i = sol.t[i]
    f_t = np.zeros(N)
    f_t[0] = np.interp(t_i, time_signal, synthetic_torque)
    f_t[-1] = -Kp * (v_sol[-1, i] - w_target)

    if use_nonlinear:
        dx = x_sol[-1, i] - x_sol[-2, i]
        k_linear = k_inter[-1]
        f_lin = k_linear * dx
        f_nl = k_linear * dx + k_nl * dx**3
        spring_force[-2, i] -= f_lin
        spring_force[-1, i] += f_lin
        spring_force[-2, i] += f_nl
        spring_force[-1, i] -= f_nl

    a_sol[:, i] = np.linalg.solve(M, f_t - C @ v_sol[:, i] - K @ x_sol[:, i])

    for j in range(N - 1):
        dx = x_sol[j+1, i] - x_sol[j, i]
        if use_nonlinear and j == N - 2:
            spring_force[j, i] = k_inter[j] * dx + k_nl * dx**3
        else:
            spring_force[j, i] = k_inter[j] * dx
        damper_force[j, i] = c_inter[j] * (v_sol[j+1, i] - v_sol[j, i])
        damping_power[j, i] = c_inter[j] * (v_sol[j+1, i] - v_sol[j, i])**2

# -------------------------------------------------
# 7) Plot Acceleration
# -------------------------------------------------
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

# -------------------------------------------------
# 8) FFT Analysis
# -------------------------------------------------
def compute_fft(signal, t):
    N = len(signal)
    dt = t[1] - t[0]
    window = np.hanning(N)
    signal_win = signal * window
    fft_result = np.fft.fft(signal_win)
    freqs = np.fft.fftfreq(N, d=dt)
    pos_idx = np.where(freqs >= 0)
    return freqs[pos_idx], np.abs(fft_result[pos_idx])

plt.figure(figsize=(10, 3 * N))
for i in range(N):
    freqs, fft_mag = compute_fft(a_sol[i, :], sol.t)
    plt.subplot(N, 1, i + 1)
    plt.plot(freqs, fft_mag, label=f'FFT of Acceleration DOF{i}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(f'FFT of Acceleration Signal for DOF{i}')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show()
