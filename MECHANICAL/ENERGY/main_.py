import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

# --- System parameters ---
J1 = 0.05   # kg·m^2
J2 = 0.08   # kg·m^2
K1 = 1000.0 # N·m/rad
C1 = 5.0    # N·m·s/rad

J = np.diag([J1, J2])
K = np.array([[ K1, -K1],
              [-K1,  K1]], dtype=float)

# Proportional damping: C = alpha*J + beta*K
alpha = 2.0
beta  = 0.001

# External torque phasor on the first inertia only
T0 = np.array([10.0, 0.0], dtype=complex)

# Frequency of interest
omega = 30.0  # rad/s

def modal_analysis_undamped(J, K):
    """ Solve the undamped eigenvalue problem: (K - w^2 J) phi = 0. 
        Return sorted natural frequencies and J-normalized mode shapes.
    """
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(J) @ K)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Natural frequencies
    omegas = np.sqrt(eigvals)
    # J-normalize each mode
    for i in range(len(omegas)):
        phi_i = eigvecs[:,i]
        norm_i = np.sqrt(phi_i.conjugate().T @ J @ phi_i)
        eigvecs[:,i] = phi_i / norm_i
    return omegas, eigvecs

def build_modal_damping(omegas, alpha, beta):
    """ If damping is proportional, c_i = alpha + beta * omega_i^2 (for J-normalized modes). """
    return alpha + beta * (omegas**2)

def solve_modal_forced_response(omegas, phi, alpha, beta, J, K, T0, freq):
    """ Solve the frequency-domain modal response for each mode's phasor Q_i. """
    c_modal = build_modal_damping(omegas, alpha, beta)
    f_modal = phi.T @ T0  # transform torque phasor to modal coords
    Q = np.zeros_like(f_modal, dtype=complex)
    for i in range(len(omegas)):
        # j_i = 1 for J-normalized modes => k_i = omega_i^2
        j_i  = 1.0
        k_i  = omegas[i]**2
        c_i  = c_modal[i]
        w    = freq
        Z_i  = (k_i - w**2*j_i) + 1j*(w*c_i)
        Q[i] = f_modal[i] / Z_i
    return Q, f_modal

def compute_power_per_mode(Q, f_modal, freq):
    """ Real (average) power in each mode i: 0.5 * Re{ f_i * (j w Q_i)* }. """
    P = []
    for i in range(len(Q)):
        P_complex = 0.5 * f_modal[i] * np.conjugate(1j*freq*Q[i])
        P.append(P_complex.real)
    return np.array(P)

# 1) Perform undamped modal analysis
omegas, phi = modal_analysis_undamped(J, K)

# 2) Solve forced response in modal coordinates for the chosen frequency
Q, f_modal = solve_modal_forced_response(omegas, phi, alpha, beta, J, K, T0, omega)

# 3) Compute power per mode
P_modes = compute_power_per_mode(Q, f_modal, omega)
P_total = P_modes.sum()

print("Natural frequencies [rad/s]:", omegas)
print("Mode shapes (columns):\n", phi)
print("Modal displacement phasors Q:", Q)
print("Power per mode:", P_modes)
print("Total power input:", P_total)

modes = np.arange(len(P_modes)) + 1  # [1, 2, ...]
plt.figure(figsize=(6,4))
plt.bar(modes, P_modes, color='skyblue', edgecolor='k', alpha=0.7)
plt.axhline(y=0, color='k', linewidth=0.8)
for i, val in enumerate(P_modes):
    plt.text(modes[i], val+0.05*max(P_modes), f"{val:.2f} W", 
             ha='center', va='bottom', color='blue', fontweight='bold')

plt.title(f"Per-Mode Power at ω = {omega} rad/s")
plt.xlabel("Mode Number")
plt.ylabel("Power (W)")
plt.tight_layout()
plt.show()

# Sankey diagram: Input -> Mode 1 -> Damping 1,  Input -> Mode 2 -> Damping 2, etc.
# Each mode's "flow" is the real power P_modes[i].

plt.figure(figsize=(8,6))

sankey = Sankey(scale=1.0, format='%.2f', unit=' W')

# The total input is the sum of all P_modes
flows_in = [P_total]  # from the source
flows_out = -P_modes  # negative (outflows) since Sankey convention needs sign

sankey.add(flows=[ P_total,          # +ve: input from an external source
                  -P_modes[0],       # first mode
                  -P_modes[1]        # second mode
                 ],
           labels=['Torque Input', 
                   f'Mode 1\n({P_modes[0]:.2f} W)', 
                   f'Mode 2\n({P_modes[1]:.2f} W)'],
           orientations=[0, 0, 0],
           pathlengths=[0.25, 0.25, 0.25],
           trunklength=1.0
          )

# If we want each mode to further split into "damping" or something else, 
# we can add additional sankey "subflows." 
# For simplicity, let's just show that the mode's entire real power is dissipated.

diagram = sankey.finish()
plt.title("Power Flow via Sankey Diagram")
plt.show()

freqs = np.linspace(1, 60, 60)
P_mode_matrix = []

for w in freqs:
    Qw, fmodal_w = solve_modal_forced_response(omegas, phi, alpha, beta, J, K, T0, w)
    Pm = compute_power_per_mode(Qw, fmodal_w, w)
    P_mode_matrix.append(Pm)

P_mode_matrix = np.array(P_mode_matrix)  # shape: (n_freqs, n_modes)

plt.figure()
for i in range(P_mode_matrix.shape[1]):
    plt.plot(freqs, P_mode_matrix[:,i], label=f'Mode {i+1}')
plt.plot(freqs, np.sum(P_mode_matrix, axis=1), 'k--', label='Total')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Power (W)')
plt.title('Power per Mode vs. Frequency')
plt.legend()
plt.grid(True)
plt.show()