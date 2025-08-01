import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Define system: masses, dampers, springs
m = np.array([1.0, 1.0, 1.0, 1.0])
k = np.array([100.0, 200.0, 150.0])  # springs between masses
c = np.array([0.5, 10.3, 0.4])        # dampers between masses
N = len(m)

# Assemble M, C, K matrices
M = np.diag(m)
K = np.zeros((N, N))
C = np.zeros((N, N))
for i in range(len(k)):
    K[i, i] += k[i]
    K[i+1, i+1] += k[i]
    K[i, i+1] -= k[i]
    K[i+1, i] -= k[i]
    C[i, i] += c[i]
    C[i+1, i+1] += c[i]
    C[i, i+1] -= c[i]
    C[i+1, i] -= c[i]

# Modal analysis (undamped)
eigvals, eigvecs = eigh(K, M)
wn = np.sqrt(eigvals)
Phi = eigvecs
M_modal = Phi.T @ M @ Phi
K_modal = Phi.T @ K @ Phi
C_modal = Phi.T @ C @ Phi  # modal damping, not necessarily diagonal

# Excite one modal coordinate only
mode_index = 1  # 2nd mode
f_n = wn[mode_index] / (2 * np.pi)
f_vals = np.linspace(f_n * 0.5, f_n * 1.5, 500)
omega_vals = 2 * np.pi * f_vals

# External force in modal space (only excite this mode)
Q_modal = np.zeros((N, len(f_vals)), dtype=complex)
Q_modal[mode_index, :] = 1.0

# Frequency response in modal coordinates
eta = np.zeros_like(Q_modal)
for i in range(N):
    for j, omega in enumerate(omega_vals):
        denom = -omega**2 * M_modal[i, i] + 1j * omega * C_modal[i, i] + K_modal[i, i]
        eta[i, j] = Q_modal[i, j] / denom

# Compute modal velocities
eta_dot = 1j * omega_vals * eta

# Compute powers
P_active_modal = np.real(Q_modal * np.conj(eta_dot))
Q_reactive_modal = omega_vals * (
    M_modal.diagonal()[:, None] * np.abs(eta_dot)**2 +
    K_modal.diagonal()[:, None] * np.abs(eta)**2
)
power_ratio_modal = np.zeros_like(P_active_modal)
mask = Q_reactive_modal > 0
power_ratio_modal[mask] = P_active_modal[mask] / Q_reactive_modal[mask]

# Plot for the selected mode
plt.figure(figsize=(10, 6))
plt.plot(f_vals, P_active_modal[mode_index], label='Active Power')
plt.plot(f_vals, Q_reactive_modal[mode_index], label='Reactive Power')
plt.plot(f_vals, power_ratio_modal[mode_index], label='Power Ratio')
plt.axvline(f_n, color='gray', linestyle='--', label='Natural Frequency')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.title(f'Modal Power Response for Mode {mode_index + 1}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
