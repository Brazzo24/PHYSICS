import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Define system matrices
# ---------------------------------------------------------
m1, m2 = 1.0, 2.0
k1, k2, k3 = 50.0, 80.0, 50.0

# Mass matrix
M = np.array([[m1,   0.0],
              [0.0,  m2 ]])

# Stiffness matrix
K = np.array([[k1 + k2,   -k2   ],
              [   -k2  , k2 + k3]])

# Small damping to avoid infinite resonance
c1, c2 = 0.5, 0.5
C = np.array([[c1,  0.0],
              [0.0, c2 ]])

# ---------------------------------------------------------
# 2. For reference: undamped modal properties
# ---------------------------------------------------------
lam, Phi = np.linalg.eig(np.linalg.inv(M).dot(K))
omega_n = np.sqrt(lam)

# Sort ascending
idx = np.argsort(omega_n)
omega_n = omega_n[idx]
Phi = Phi[:, idx]

# Mass-normalize the modes: Phi^T M Phi = I
for i in range(Phi.shape[1]):
    mass_modal = Phi[:, i].T @ M @ Phi[:, i]
    Phi[:, i] = Phi[:, i] / np.sqrt(mass_modal)

print("Undamped natural frequencies [rad/s]:", omega_n)

# ---------------------------------------------------------
# 3. Define harmonic forcing (applied on DOF 1)
# ---------------------------------------------------------
F0 = np.array([1.0, 0.0])   # amplitude vector (complex)
omega_min, omega_max = 0.1, 100.0
omegas = np.linspace(omega_min, omega_max, 1000)

# ---------------------------------------------------------
# 4. Frequency sweep
# ---------------------------------------------------------
T_list_physical = []
U_list_physical = []
E_list_physical = []

T_list_modes = []
U_list_modes = []
E_list_modes = []

for w in omegas:
    # Dynamic stiffness: D(w) = (-w^2 M + j w C + K)
    D = -w**2 * M + 1j*w * C + K
    
    # Solve for steady-state X(w) in physical coords
    X = np.linalg.solve(D, F0)  # shape: (2, )
    
    # 4a) Time-averaged energies in PHYSICAL coordinates
    # Kinetic: T_avg = (1/4)* X^*(w)^T (w^2 M) X(w)
    T_physical = 0.25 * (X.conjugate().T @ (w**2 * M) @ X).real
    
    # Potential: U_avg = (1/4)* X^*(w)^T K X(w)
    U_physical = 0.25 * (X.conjugate().T @ K @ X).real
    
    E_physical = T_physical + U_physical
    
    T_list_physical.append(T_physical)
    U_list_physical.append(U_physical)
    E_list_physical.append(E_physical)
    
    # 4b) Transform to modal coords: Q(w) = Phi^T M X(w)
    #     (Phi is mass-normalized => Potential ~ omega_r^2 |Q_r|^2, Kinetic ~ w^2 |Q_r|^2)
    Q = Phi.T @ M @ X  # shape: (2, )
    
    # For each mode r:
    #   T_r_avg = 1/4 * w^2 * |Q_r|^2
    #   U_r_avg = 1/4 * omega_r^2 * |Q_r|^2
    # Summation across r gives total energy in modes
    T_modes = 0.0
    U_modes = 0.0
    for r in range(2):
        qr = Q[r]
        T_modes += 0.25 * w**2 * np.abs(qr)**2
        U_modes += 0.25 * omega_n[r]**2 * np.abs(qr)**2
    
    E_modes = T_modes + U_modes
    
    T_list_modes.append(T_modes)
    U_list_modes.append(U_modes)
    E_list_modes.append(E_modes)


T_list_physical = np.array(T_list_physical)
U_list_physical = np.array(U_list_physical)
E_list_physical = np.array(E_list_physical)

T_list_modes = np.array(T_list_modes)
U_list_modes = np.array(U_list_modes)
E_list_modes = np.array(E_list_modes)

# ---------------------------------------------------------
# 5. Check the difference: E_physical vs E_modes
# ---------------------------------------------------------
E_diff = E_list_physical - E_list_modes  # ideally ~ 0
max_diff = np.max(np.abs(E_diff))
print(f"Max |E_physical - E_modes| across the sweep: {max_diff:.2e}")

# ---------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(8,9), sharex=True)

axs[0].plot(omegas, T_list_physical, label="T_phys")
axs[0].plot(omegas, U_list_physical, label="U_phys")
axs[0].plot(omegas, E_list_physical, 'k--', label="E_phys")
axs[0].set_ylabel("Time-Averaged Energy (J)")
axs[0].set_title("Physical Coordinates")
axs[0].legend()
axs[0].grid(True)
axs[0].set_yscale('log')

axs[1].plot(omegas, T_list_modes, label="T_modes")
axs[1].plot(omegas, U_list_modes, label="U_modes")
axs[1].plot(omegas, E_list_modes, 'k--', label="E_modes")
axs[1].set_ylabel("Time-Averaged Energy (J)")
axs[1].set_title("Modal Coordinates")
axs[1].legend()
axs[1].grid(True)
axs[1].set_yscale('log')

axs[2].plot(omegas, E_diff, label="E_phys - E_modes")
axs[2].set_xlabel("Frequency (rad/s)")
axs[2].set_ylabel("Energy Difference (J)")
axs[2].set_title("Sanity Check: Difference in Total Energy")
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
