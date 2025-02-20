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
              [  -k2   , k2 + k3]])

# Small damping for each DOF (purely diagonal for simplicity)
c1, c2 = 0.5, 0.5
C = np.array([[c1,  0.0],
              [0.0, c2 ]])

# ---------------------------------------------------------
# 2. Modal properties (undamped modes for reference)
#    We also get the mode shapes to transform response
#    into modal coordinates.
# ---------------------------------------------------------
# Solve eigenvalue problem for M^{-1} K
lam, Phi = np.linalg.eig(np.linalg.inv(M).dot(K))
omega_n = np.sqrt(lam)

# Sort by ascending frequency
idx = np.argsort(omega_n)
omega_n = omega_n[idx]
Phi = Phi[:, idx]

# Mass-normalize the eigenvectors: Phi^T M Phi = I
for i in range(Phi.shape[1]):
    m_modal = Phi[:, i].T @ M @ Phi[:, i]
    Phi[:, i] = Phi[:, i] / np.sqrt(m_modal)

# Extract natural frequencies for reference
omega1, omega2 = omega_n[0], omega_n[1]
print(f"Undamped natural frequencies (rad/s): {omega1:.3f}, {omega2:.3f}")

# ---------------------------------------------------------
# 3. Define forcing
# ---------------------------------------------------------
# Let's assume a simple harmonic force on the first mass only:
F0 = np.array([1.0, 0.0])   # amplitude vector (complex)
# We'll sweep over a range of frequencies and solve for the response

# ---------------------------------------------------------
# 4. Frequency sweep
# ---------------------------------------------------------
omegas = np.linspace(0.1, 100, 1000)  # from 0.1 to 100 rad/s
E1_vals = []  # to store time-averaged energy in mode 1
E2_vals = []  # to store time-averaged energy in mode 2
E_total_vals = []

for w in omegas:
    # Dynamic stiffness matrix = (-w^2 M + j w C + K)
    # We'll use complex '1j' in NumPy for sqrt(-1).
    D = -w**2 * M + 1j*w * C + K
    
    # Solve for X(omega) in physical coordinates
    # X(omega) = D^{-1} * F0
    X = np.linalg.solve(D, F0)
    
    # Convert to modal coordinates: Q(omega) = Phi^T M X(omega)
    # (Phi is mass-normalized, so M_modal = I in the modal space)
    Q = Phi.T @ M @ X  # shape: (2, )

    # ----------------------------------------------------------------
    # Time-averaged energy in each mode
    # ----------------------------------------------------------------
    # If Q_r is the complex amplitude of mode r at frequency w,
    # the time-averaged potential energy in mode r is:
    #   U_r_avg = (1/4) * omega_r^2 * |Q_r|^2
    #
    # the time-averaged kinetic energy is:
    #   T_r_avg = (1/4) * w^2 * |Q_r|^2
    #
    # total mode energy = U_r_avg + T_r_avg
    #
    # For a 2-DOF system with r in {1, 2}:
    
    # Note: |Q_r|^2 = Q_r.conjugate() * Q_r  (scalar for each mode)
    
    # We'll extract Q_1 and Q_2 from Q:
    Q1 = Q[0]
    Q2 = Q[1]
    
    # Mode 1
    E1 = 0.25 * (omega1**2 * np.abs(Q1)**2 + w**2 * np.abs(Q1)**2)
    # Mode 2
    E2 = 0.25 * (omega2**2 * np.abs(Q2)**2 + w**2 * np.abs(Q2)**2)
    
    E_total = E1 + E2
    
    E1_vals.append(E1)
    E2_vals.append(E2)
    E_total_vals.append(E_total)

E1_vals = np.array(E1_vals)
E2_vals = np.array(E2_vals)
E_total_vals = np.array(E_total_vals)

# ---------------------------------------------------------
# 5. Compute fraction of energy in each mode (if desired)
# ---------------------------------------------------------
eta1 = E1_vals / E_total_vals
eta2 = E2_vals / E_total_vals

# ---------------------------------------------------------
# 6. Plot Results
# ---------------------------------------------------------
plt.figure(figsize=(9,6))

# a) Total energies vs frequency
plt.subplot(2, 1, 1)
plt.plot(omegas, E1_vals, label='Mode 1 Energy')
plt.plot(omegas, E2_vals, label='Mode 2 Energy')
plt.plot(omegas, E_total_vals, 'k--', label='Total Energy')
plt.yscale('log')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Time-Averaged Energy')
plt.title('Steady-State Modal Energy (Frequency Domain)')
plt.grid(True)
plt.legend()

# b) Modal energy fraction
plt.subplot(2, 1, 2)
plt.plot(omegas, eta1, label='Mode 1 Fraction')
plt.plot(omegas, eta2, label='Mode 2 Fraction')
plt.ylim([0,1])
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Energy Fraction')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
