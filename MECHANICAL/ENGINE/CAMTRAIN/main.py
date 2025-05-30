import numpy as np
# Physical parameters (dummy values)
J_crank, J_dyn = 0.10, 0.50      # inertias (kg·m^2)
J_in, J_out   = 0.02, 0.02
K_c, C_c      = 1.0e4, 50.0      # coupling stiffness (N·m/rad) and damping
g1, g2        = 2.0, -1.0        # gear ratios (crank->cam, inlet->outlet)
K_g1 = K_g2   = 5.0e4           # gear mesh stiffnesses
C_g1 = C_g2   = 10.0            # gear mesh damping

# Assemble mass matrix (4x4 diagonal)
M = np.diag([J_crank, J_in, J_out, J_dyn])
# Initialize stiffness and damping matrices
K = np.zeros((4,4))
C = np.zeros((4,4))

# Rubber coupling between crank (0) and dyno (3)
i, j, r = 0, 3, 1.0
K[i,i] += K_c;  K[j,j] += K_c
K[i,j] -= K_c*r; K[j,i] -= K_c*r
C[i,i] += C_c;  C[j,j] += C_c
C[i,j] -= C_c*r; C[j,i] -= C_c*r

# Gear train between crank (0) and inlet cam (1)
i, j, r = 0, 1, g1
K[i,i] += K_g1;           K[j,j] += K_g1 * r**2
K[i,j] += -K_g1 * r;      K[j,i] += -K_g1 * r
C[i,i] += C_g1;           C[j,j] += C_g1 * r**2
C[i,j] += -C_g1 * r;      C[j,i] += -C_g1 * r

# Gear pair between inlet cam (1) and outlet cam (2)
i, j, r = 1, 2, g2
K[i,i] += K_g2;           K[j,j] += K_g2 * r**2     # (r^2 = 1)
K[i,j] += -K_g2 * r;      K[j,i] += -K_g2 * r       # g2 is -1
C[i,i] += C_g2;           C[j,j] += C_g2 * r**2
C[i,j] += -C_g2 * r;      C[j,i] += -C_g2 * r

print(M)
print(K)
print(C)

import scipy.linalg as la
# Solve generalized eigenvalue problem K x = λ M x
eigvals, eigvecs = la.eig(K, M)
eigvals = np.real(eigvals)         # discard tiny imaginary parts
eigvals = np.sort(eigvals)         # sort eigenvalues
# Exclude the zero eigenvalue (rigid mode), and take sqrt for ω (rad/s)
omega_n = np.sqrt(eigvals[1:])     # [1:] skips the zero mode
freqs_hz = omega_n / (2*np.pi)
print("Natural frequencies (Hz):", freqs_hz)


# Until this point, the Code create a system with mass/stiffness/damping-matrices and solves for the
# undamped natural frequencies using the numpy solvers for this task.



from math import sin
from scipy.integrate import solve_ivp

# ODE system: state y = [θ0, θ1, θ2, θ3, ω0, ω1, ω2, ω3]
def torsion_deriv(t, y):
    θ = y[:4]; ω = y[4:]                        # split angles and speeds
    # External torque on crank (index 0)
    T_mean = 50.0 * min(t/0.5, 1.0)             # ramp-up to 50 N·m by 0.5s
    T_pulse = 20.0 * sin(2 * θ[0])              # 2 pulses per rev
    T_ext = np.array([T_mean + T_pulse, 0, 0, 0])
    # Equation of motion: M·α = -K·θ - C·ω + T_ext
    torque = -K.dot(θ) - C.dot(ω) + T_ext       # net torque on each inertia
    α = np.linalg.solve(M, torque)             # angular accelerations
    return np.concatenate((ω, α))

# Initial state: all angles=0, angular velocities=0
y0 = np.zeros(8)
# Integrate from t=0 to 5 s
sol = solve_ivp(torsion_deriv, [0, 5], y0, max_step=5e-4, rtol=1e-6, atol=1e-8)

import matplotlib.pyplot as plt

t = sol.t
theta = sol.y[:4]  # angular positions of crank, inlet cam, outlet cam, dyno
theta_crank, theta_dyno = theta[0], theta[3]

# Torsional twist across rubber coupling (in degrees)
twist_coupling_deg = (theta_crank - theta_dyno) * (180 / np.pi)

plt.figure()
plt.plot(t, twist_coupling_deg)
plt.axvline(2.0, color='r', linestyle='--', label='~1500 RPM mark')
plt.xlabel('Time [s]')
plt.ylabel('Coupling Twist [deg]')
plt.title('Rubber Coupling Twist during Engine Run-Up')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


omega = sol.y[4:]  # angular velocities
omega_crank, omega_in, omega_out = omega[0], omega[1], omega[2]
theta_in, theta_out = theta[1], theta[2]

# Gear torque formulas
T_gear1 = K_g1 * (theta_crank - g1 * theta_in) + C_g1 * (omega_crank - g1 * omega[1])
T_gear2 = K_g2 * (theta_in - g2 * theta_out) + C_g2 * (omega_in - g2 * omega_out)

plt.figure()
plt.plot(t, T_gear1, label='Crank → Inlet Cam Gear')
plt.plot(t, T_gear2, '--', label='Inlet → Outlet Cam Gear')
plt.axvline(2.0, color='r', linestyle='--', label='~1500 RPM mark')
plt.xlabel('Time [s]')
plt.ylabel('Gear Torque [Nm]')
plt.title('Camtrain Gear Torques during Run-Up')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Frequency range (rad/s)
freqs_hz = np.linspace(10, 600, 1000)
omegas = 2 * np.pi * freqs_hz

# Excitation: 1 Nm harmonic torque at crankshaft (DOF 0)
T = np.zeros(4, dtype=complex)
T[0] = 1.0

# Allocate outputs
twist_coupling = []
torque_gear1 = []
torque_gear2 = []

for omega in omegas:
    # System dynamic stiffness matrix
    Z = -omega**2 * M + 1j * omega * C + K
    # Solve for steady-state response
    theta_resp = np.linalg.solve(Z, T)

    # Angular twist (crank - dyno)
    twist_cd = theta_resp[0] - theta_resp[3]
    twist_coupling.append(np.abs(twist_cd))

    # Relative twist and velocity (for torque calc)
    rel_theta_01 = theta_resp[0] - g1 * theta_resp[1]
    rel_omega_01 = 1j * omega * (theta_resp[0] - g1 * theta_resp[1])
    torque1 = K_g1 * rel_theta_01 + C_g1 * rel_omega_01
    torque_gear1.append(np.abs(torque1))

    rel_theta_12 = theta_resp[1] - g2 * theta_resp[2]
    rel_omega_12 = 1j * omega * (theta_resp[1] - g2 * theta_resp[2])
    torque2 = K_g2 * rel_theta_12 + C_g2 * rel_omega_12
    torque_gear2.append(np.abs(torque2))

twist_coupling = np.array(twist_coupling)
torque_gear1 = np.array(torque_gear1)
torque_gear2 = np.array(torque_gear2)

plt.figure(figsize=(10, 6))
plt.plot(freqs_hz, twist_coupling, label='Coupling Twist (Crank - Dyno)')
plt.plot(freqs_hz, torque_gear1, label='Torque in Gear 1 (Crank → Inlet Cam)')
plt.plot(freqs_hz, torque_gear2, label='Torque in Gear 2 (Inlet Cam → Outlet Cam)')

plt.xlabel('Excitation Frequency [Hz]')
plt.ylabel('Amplitude [rad or Nm]')
plt.title('Frequency Response of Camtrain System')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Eigenvalue analysis (from earlier)
eigvals, eigvecs = la.eig(K, M)
eigvals = np.real(eigvals)
eigvals = np.sort(eigvals)
omega_n = np.sqrt(eigvals[1:])  # skip rigid mode
natural_freqs = omega_n / (2 * np.pi)

plt.figure(figsize=(10, 6))
plt.plot(freqs_hz, twist_coupling, label='Coupling Twist (Crank - Dyno)')
plt.plot(freqs_hz, torque_gear1, label='Torque in Gear 1')
plt.plot(freqs_hz, torque_gear2, label='Torque in Gear 2')

# Highlight natural frequencies
for f in natural_freqs:
    plt.axvline(f, color='r', linestyle='--', alpha=0.6, label=f'Mode at {f:.1f} Hz')

plt.xlabel('Excitation Frequency [Hz]')
plt.ylabel('Amplitude [rad or Nm]')
plt.title('Frequency Response with Natural Frequencies')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

phase_crank = []
phase_inlet = []
phase_outlet = []
phase_dyno = []

for omega in omegas:
    Z = -omega**2 * M + 1j * omega * C + K
    theta_resp = np.linalg.solve(Z, T)

    phase_crank.append(np.angle(theta_resp[0], deg=True))
    phase_inlet.append(np.angle(theta_resp[1], deg=True))
    phase_outlet.append(np.angle(theta_resp[2], deg=True))
    phase_dyno.append(np.angle(theta_resp[3], deg=True))

plt.figure(figsize=(10, 6))
plt.plot(freqs_hz, phase_crank, label='Crank')
plt.plot(freqs_hz, phase_inlet, label='Inlet Cam')
plt.plot(freqs_hz, phase_outlet, label='Outlet Cam')
plt.plot(freqs_hz, phase_dyno, label='Dyno')

plt.xlabel('Excitation Frequency [Hz]')
plt.ylabel('Phase [degrees]')
plt.title('Phase Response of System DOFs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

Kc_values = [5e3, 1e4, 2e4]
twist_all = []

for K_c_test in Kc_values:
    # Regenerate K and C matrices with updated K_c
    K_mod = K.copy()
    C_mod = C.copy()
    # Reset crank-dyno coupling terms (DOFs 0 and 3)
    for i in [0, 3]:
        for j in [0, 3]:
            K_mod[i, j] -= K_c if i == j else (-K_c if i != j else 0)
            C_mod[i, j] -= C_c if i == j else (-C_c if i != j else 0)
    for i, j in [(0, 3), (3, 0)]:
        K_mod[i, j] += K_c_test
        C_mod[i, j] += C_c

    twist_kc = []
    for omega in omegas:
        Z = -omega**2 * M + 1j * omega * C_mod + K_mod
        theta_resp = np.linalg.solve(Z, T)
        twist_cd = theta_resp[0] - theta_resp[3]
        twist_kc.append(np.abs(twist_cd))
    twist_all.append(twist_kc)

# Plotting
plt.figure(figsize=(10, 6))
for i, Kc in enumerate(Kc_values):
    plt.plot(freqs_hz, twist_all[i], label=f'Kc = {Kc:.0e} Nm/rad')

for f in natural_freqs:
    plt.axvline(f, color='r', linestyle='--', alpha=0.3)

plt.xlabel('Excitation Frequency [Hz]')
plt.ylabel('Coupling Twist Amplitude [rad]')
plt.title('Effect of Coupling Stiffness on Frequency Response')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from scipy.integrate import simps

# Trim time window to where run-up is steady (e.g., from 2 to 5 seconds)
t_mask = (t >= 2.0) & (t <= 5.0)
t_window = t[t_mask]
torque_signal = T_gear1[t_mask]

# Mean torque
T_mean = np.mean(torque_signal)
# Peak torque (absolute)
T_peak = np.max(np.abs(torque_signal))
# RMS torque using numerical integration
T_rms = np.sqrt(simps(torque_signal**2, t_window) / (t_window[-1] - t_window[0]))

# Amplification metrics
peak_to_mean = T_peak / np.abs(T_mean) if T_mean != 0 else np.inf
crest_factor = T_peak / T_rms if T_rms != 0 else np.inf

# Print results
print(f"Torque Amplification (Crank → Inlet Cam Gear):")
print(f"  Mean Torque     = {T_mean:.3f} Nm")
print(f"  Peak Torque     = {T_peak:.3f} Nm")
print(f"  RMS Torque      = {T_rms:.3f} Nm")
print(f"  Peak-to-Mean    = {peak_to_mean:.2f}")
print(f"  Peak-to-RMS     = {crest_factor:.2f}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.integrate import solve_ivp

# Physical parameters (base values)
J_crank, J_dyn = 0.10, 0.50
J_in, J_out = 0.02, 0.02
g1, g2 = 2.0, -1.0
K_g1 = K_g2 = 5.0e4
C_g1 = C_g2 = 10.0

# Parameter sweep values for coupling stiffness and damping
Kc_values = [5e3, 1e4, 2e4]
Cc_values = [10.0, 50.0, 100.0]

# Frequency of excitation in run-up (2/rev)
def simulate_runup(K_c, C_c):
    # Assemble matrices
    M = np.diag([J_crank, J_in, J_out, J_dyn])
    K = np.zeros((4, 4))
    C = np.zeros((4, 4))
    
    def add_connection(i, j, k_val, c_val, ratio):
        K[i, i] += k_val
        K[j, j] += k_val * ratio ** 2
        K[i, j] -= k_val * ratio
        K[j, i] -= k_val * ratio
        C[i, i] += c_val
        C[j, j] += c_val * ratio ** 2
        C[i, j] -= c_val * ratio
        C[j, i] -= c_val * ratio

    add_connection(0, 3, K_c, C_c, 1.0)
    add_connection(0, 1, K_g1, C_g1, g1)
    add_connection(1, 2, K_g2, C_g2, g2)

    # Time domain simulation setup
    def torsion_deriv(t, y):
        theta = y[:4]
        omega = y[4:]
        T_mean = 50.0 * min(t / 0.5, 1.0)
        T_pulse = 20.0 * np.sin(2 * theta[0])
        T_ext = np.array([T_mean + T_pulse, 0, 0, 0])
        torque = -K @ theta - C @ omega + T_ext
        alpha = np.linalg.solve(M, torque)
        return np.concatenate((omega, alpha))

    y0 = np.zeros(8)
    t_span = (0, 5)
    sol = solve_ivp(torsion_deriv, t_span, y0, max_step=5e-4, rtol=1e-6, atol=1e-8)
    t = sol.t
    theta = sol.y[:4]
    omega = sol.y[4:]

    # Compute torque in gear 1: crank to inlet cam
    T_gear1 = K_g1 * (theta[0] - g1 * theta[1]) + C_g1 * (omega[0] - g1 * omega[1])

    # Evaluate metrics in a trimmed time window (2s to 5s)
    t_mask = (t >= 2.0) & (t <= 5.0)
    t_window = t[t_mask]
    torque_signal = T_gear1[t_mask]
    T_mean = np.mean(torque_signal)
    T_peak = np.max(np.abs(torque_signal))
    T_rms = np.sqrt(simps(torque_signal**2, t_window) / (t_window[-1] - t_window[0]))
    peak_to_mean = T_peak / np.abs(T_mean) if T_mean != 0 else np.inf
    crest_factor = T_peak / T_rms if T_rms != 0 else np.inf

    return {
        "K_c": K_c,
        "C_c": C_c,
        "T_mean": T_mean,
        "T_peak": T_peak,
        "T_rms": T_rms,
        "Peak/Mean": peak_to_mean,
        "Peak/RMS": crest_factor
    }

# Run simulations for all Kc and Cc combinations
results = []
for K_c in Kc_values:
    for C_c in Cc_values:
        res = simulate_runup(K_c, C_c)
        results.append(res)

# Create a summary table
import pandas as pd
df = pd.DataFrame(results)

# Re-import necessary libraries after kernel reset
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Re-create the data from previous results
data = [
    {"K_c": 5000.0, "C_c": 10.0, "Peak/Mean": 4.942674, "Peak/RMS": 2.077888},
    {"K_c": 5000.0, "C_c": 50.0, "Peak/Mean": 3.475830, "Peak/RMS": 2.081809},
    {"K_c": 5000.0, "C_c": 100.0, "Peak/Mean": 2.752878, "Peak/RMS": 2.079649},
    {"K_c": 10000.0, "C_c": 10.0, "Peak/Mean": 6.965949, "Peak/RMS": 2.286007},
    {"K_c": 10000.0, "C_c": 50.0, "Peak/Mean": 3.601051, "Peak/RMS": 2.100930},
]

df = pd.DataFrame(data)

# Pivot tables for heatmap visualization
pivot_peak_mean = df.pivot(index="K_c", columns="C_c", values="Peak/Mean")
pivot_crest = df.pivot(index="K_c", columns="C_c", values="Peak/RMS")

# Heatmap: Peak-to-Mean Ratio
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_peak_mean, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Peak-to-Mean Torque Amplification")
plt.xlabel("Coupling Damping (C_c) [Nm·s/rad]")
plt.ylabel("Coupling Stiffness (K_c) [Nm/rad]")
plt.tight_layout()
plt.show()

# Heatmap: Crest Factor (Peak-to-RMS)
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_crest, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Crest Factor (Peak-to-RMS)")
plt.xlabel("Coupling Damping (C_c) [Nm·s/rad]")
plt.ylabel("Coupling Stiffness (K_c) [Nm/rad]")
plt.tight_layout()
plt.show()
