# Build a simplified version of your driveline model with 3 DOFs to check resonance behavior
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Apply both fixes: clip eigenvalues and optimize sweep speed in the 3-DOF model

# Redefine system for clarity
m_simple = np.array([0.01, 0.01, 0.01])           # inertias
k_simple = np.array([5000.0, 500.0])              # stiffnesses
c_simple = np.array([0.1, 0.05])                  # dampings
N_simple = len(m_simple)

# Build system matrices
def build_matrices(m, c, k):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N - 1):
        C[i, i] += c[i]
        C[i + 1, i + 1] += c[i]
        C[i, i + 1] -= c[i]
        C[i + 1, i] -= c[i]

        K[i, i] += k[i]
        K[i + 1, i + 1] += k[i]
        K[i, i + 1] -= k[i]
        K[i + 1, i] -= k[i]
    return M, C, K

M, C, K = build_matrices(m_simple, c_simple, k_simple)

# Modal analysis with fix for negative eigenvalues
eigvals, eigvecs = eig(K, M)
eigvals = np.real(eigvals)
eigvals = np.clip(eigvals, a_min=0, a_max=None)  # fix: remove negative numerical noise
natural_freqs_hz = np.sqrt(eigvals) / (2 * np.pi)
natural_freqs_hz_sorted = np.sort(natural_freqs_hz)

# Define ODE with harmonic input at DOF2
def odefunc_simple(t, y, M, C, K, freq):
    x = y[:N_simple]
    v = y[N_simple:]
    f = np.zeros(N_simple)
    f[-1] = np.sin(2 * np.pi * freq * t)
    a = np.linalg.solve(M, f - C @ v - K @ x)
    return np.concatenate((v, a))

# Optimized sweep
sweep_freqs = np.linspace(1, 100, 40)
rms_twist = []

for f_drive in sweep_freqs:
    y0 = np.zeros(2 * N_simple)
    t_eval = np.linspace(0, 1.0, 1000)  # shorter, faster
    sol = solve_ivp(lambda t, y: odefunc_simple(t, y, M, C, K, f_drive),
                    [0, 1.0], y0, t_eval=t_eval, method='RK45')
    twist = (sol.y[0, :] - sol.y[1, :]) * (180 / np.pi)
    twist_centered = twist - np.mean(twist)
    rms = np.sqrt(np.mean(twist_centered**2))
    rms_twist.append(rms)

# Plot resonance detection result
plt.figure(figsize=(10, 5))
plt.plot(sweep_freqs, rms_twist, label='RMS Twist (DOF0–DOF1)', color='blue')
for f in natural_freqs_hz_sorted[:3]:
    plt.axvline(f, color='red', linestyle='--', alpha=0.7, label=f'Mode @ {f:.1f} Hz')
plt.xlabel('Excitation Frequency [Hz]')
plt.ylabel('RMS Twist [deg]')
plt.title('Optimized 3-DOF Model: Resonance Detection')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

natural_freqs_hz_sorted[:3]

