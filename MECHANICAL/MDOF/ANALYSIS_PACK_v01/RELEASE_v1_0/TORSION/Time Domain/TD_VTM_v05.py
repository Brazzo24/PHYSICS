# Scale up to a 7-DOF system with a realistic resonance structure
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import eig
# Define inertias, damping, and stiffness for a 7-DOF system
#----------Cam Out, Cam in, Gear42, gear40, R2, R1, Dyno
m = np.array([1.21e-3, 3.95e-3, 7.92e-4,
              1.02e-3, 1.42e-3, 3.35e-2, 1.09e-1])  # [kg·m²]

c_inter = np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.05])  # [Nm·s/rad]

#-----------------gear,     gear,    gear,  spline, Shaft,  Reich
k_inter = np.array([2.34e4, 1.62e5, 1.62e5, 1.62e5, 4.73e5, 8.057e2])  # [Nm/rad]

N = len(m)

# Build matrices
def build_full_matrices(m, c_inter, k_inter):
    N = len(m)
    M = np.diag(m)
    C = np.zeros((N, N))
    K = np.zeros((N, N))
    for i in range(N - 1):
        C[i, i] += c_inter[i]
        C[i + 1, i + 1] += c_inter[i]
        C[i, i + 1] -= c_inter[i]
        C[i + 1, i] -= c_inter[i]

        K[i, i] += k_inter[i]
        K[i + 1, i + 1] += k_inter[i]
        K[i, i + 1] -= k_inter[i]
        K[i + 1, i] -= k_inter[i]
    return M, C, K

M, C, K = build_full_matrices(m, c_inter, k_inter)

# Modal analysis (sorted eigenvalues and eigenvectors)
eigvals, eigvecs = eig(K, M)
eigvals = np.real(eigvals)
eigvecs = np.real(eigvecs)
eigvals = np.clip(eigvals, a_min=0, a_max=None)

# Sort both frequencies and corresponding mode shapes
idx_sorted = np.argsort(eigvals)
eigvals = eigvals[idx_sorted]
eigvecs = eigvecs[:, idx_sorted]

natural_freqs_hz = np.sqrt(eigvals) / (2 * np.pi)
natural_freqs_hz_sorted = np.sort(natural_freqs_hz)

# ODE function with harmonic torque input at last DOF
def odefunc_scaled(t, y, M, C, K, freq):
    x = y[:N]
    v = y[N:]
    f = np.zeros(N)
    f[-1] = np.sin(2 * np.pi * freq * t)
    a = np.linalg.solve(M, f - C @ v - K @ x)
    return np.concatenate((v, a))

# Frequency sweep
# Further optimized: reduce to 20 frequency points, 0.5s simulation
sweep_freqs = np.linspace(10, 300, 50)
rms_twist = []
peak_to_peak_twist = []

for f_drive in sweep_freqs:
    y0 = np.zeros(2 * N)
    t_eval = np.linspace(0, 0.5, 600)
    sol = solve_ivp(lambda t, y: odefunc_scaled(t, y, M, C, K, f_drive),
                    [0, 0.5], y0, t_eval=t_eval, method='Radau')
    twist = (sol.y[0, :] - np.mean(sol.y[0, :])) * (180 / np.pi)
    twist_centered = twist - np.mean(twist)
    rms = np.sqrt(np.mean(twist_centered**2))
    rms_twist.append(rms)
    
    p2p = np.max(twist_centered) - np.min(twist_centered)
    peak_to_peak_twist.append(p2p)

# Normalize mode shapes for plotting
mode_shapes = eigvecs / np.max(np.abs(eigvecs), axis=0)


# Plot first 5 mode shapes
plt.figure(figsize=(10, 6))
for i in range(min(7, N)):
    plt.plot(np.arange(N), mode_shapes[:, i], marker='o', label=f'Mode {i+1} ({natural_freqs_hz[i]:.1f} Hz)')
plt.xticks(np.arange(N))
plt.xlabel('DOF Index')
plt.ylabel('Normalized Mode Shape')
plt.title('First 5 Mode Shapes')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print frequencies
print("Natural frequencies (Hz):")
for i, f in enumerate(natural_freqs_hz[:min(10, N)]):
    print(f"Mode {i+1}: {f:.2f} Hz")


# # Plot resonance curve
# plt.figure(figsize=(10, 5))
# plt.plot(sweep_freqs, rms_twist, label='RMS Twist (DOF0–DOF1)', color='blue')
# for f in natural_freqs_hz_sorted[:3]:
#     plt.axvline(f, color='red', linestyle='--', alpha=0.7, label=f'Mode @ {f:.1f} Hz')
# plt.xlabel('Excitation Frequency [Hz]')
# plt.ylabel('RMS Twist [deg]')
# plt.title('7-DOF Driveline Model: Resonance Detection (Fast Sweep)')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(sweep_freqs, rms_twist, 'o-', label='RMS Twist [deg]', color='blue')
plt.plot(sweep_freqs, peak_to_peak_twist, 's--', label='Peak-to-Peak Twist [deg]', color='red')
for f in natural_freqs_hz[:3]:
    plt.axvline(f, color='gray', linestyle='--', alpha=0.5, label=f'Mode @ {f:.1f} Hz')
plt.xlabel('Excitation Frequency [Hz]')
plt.ylabel('Twist Amplitude [deg]')
plt.title('7-DOF Driveline: RMS and Peak-to-Peak Swing Angle')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(0, 200)
plt.show()

natural_freqs_hz[:3]
