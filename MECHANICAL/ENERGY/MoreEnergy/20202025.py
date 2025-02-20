import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Define the system matrices M, C, K
#    (In practice, you might import these from files or generate them procedurally.)
# =============================================================================

# Example dimensions: n=3 DOFs
n = 3

# Mass matrix (3x3)
M = np.array([[2.0, 0.0, 0.0],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.5]])

# Damping matrix (3x3)
C = np.array([[0.1, 0.0, 0.0],
              [0.0, 0.2, 0.0],
              [0.0, 0.0, 0.1]])

# Stiffness matrix (3x3)
K = np.array([[  50.0, -20.0,   0.0],
              [ -20.0,  40.0, -10.0],
              [   0.0, -10.0,  30.0]])

# =============================================================================
# 2. Define springs and connectivity
#    For potential energy calculation, we need to know which DOFs each spring connects.
#    Suppose we have springs with the same k-values as in K above.
#    But let's store them explicitly with (k, dof1, dof2).
#    (We only list the non-zero off-diagonal parts, typically.)
# =============================================================================

# Example: from K, we see that we have springs between:
#   DOF0 and DOF1 => stiffness = 20
#   DOF1 and DOF2 => stiffness = 10
#   DOF0 <-> DOF0 => 50 (but that is effectively the sum of all springs at DOF0)
#   DOF1 <-> DOF1 => 40
#   DOF2 <-> DOF2 => 30
#
# For clarity, let's define spring connections as if they are between DOFs:
springs = [
    (20.0, 0, 1),   # spring between DOF0 and DOF1
    (10.0, 1, 2)    # spring between DOF1 and DOF2
]

# Additionally, some systems treat each diagonal as a grounded spring (DOF i to ground).
# If you want to interpret the diagonal terms as a separate spring to ground:
#   e.g. (k=50.0, 0, 'ground')
# For demonstration, let’s do that too:
springs_to_ground = [
    (50.0, 0),  # DOF0 to ground
    (40.0, 1),  # DOF1 to ground
    (30.0, 2)   # DOF2 to ground
]

# =============================================================================
# 3. Define forcing and frequency range
# =============================================================================

# Example: a unit force applied at DOF0, zero elsewhere
F = np.array([1.0, 0.0, 0.0], dtype=complex)

# Frequency range (in rad/s)
omega_vals = np.linspace(0.1, 20, 200)  # from 0.1 to 20 rad/s

# Prepare storage for energies
n_freqs = len(omega_vals)

# Kinetic energy in each DOF vs. frequency (peak values)
T_kin = np.zeros((n_freqs, n))
# Potential energy in each spring vs. frequency (peak values)
V_springs = np.zeros((n_freqs, len(springs)))
V_ground  = np.zeros((n_freqs, len(springs_to_ground)))

# =============================================================================
# 4. Frequency-response loop
# =============================================================================

for i, omega in enumerate(omega_vals):
    # Dynamic stiffness matrix: H(omega) = ( -omega^2 M + i omega C + K )
    H = -omega**2 * M + 1j*omega * C + K
    
    # Solve for displacement (complex amplitudes)
    x_hat = np.linalg.solve(H, F)  # shape (n,)

    # -------------------------------------------------------------------------
    # 4a. Kinetic energy per DOF (peak)
    #
    #   T_j(peak) = 1/2 * m_j * (omega * |x_hat_j|)^2
    # -------------------------------------------------------------------------
    for j in range(n):
        m_j = M[j, j]  # or whichever is appropriate for your indexing
        disp_j = x_hat[j]
        T_kin[i, j] = 0.5 * m_j * (omega * np.abs(disp_j))**2

    # -------------------------------------------------------------------------
    # 4b. Potential energy per spring (peak)
    #
    #   V_spring(peak) = 1/2 * k * |x_p - x_q|^2
    #   For a spring to ground, V_ground = 1/2 * k * |x_j|^2
    # -------------------------------------------------------------------------
    # Springs between DOFs
    for s_idx, (k_s, dof_p, dof_q) in enumerate(springs):
        rel_disp = x_hat[dof_p] - x_hat[dof_q]
        V_springs[i, s_idx] = 0.5 * k_s * (np.abs(rel_disp))**2

    # Springs to ground
    for s_idx, (k_s, dof_j) in enumerate(springs_to_ground):
        V_ground[i, s_idx] = 0.5 * k_s * (np.abs(x_hat[dof_j]))**2

# =============================================================================
# 5. Post-process and plotting
# =============================================================================

# Total kinetic energy at each frequency (sum over DOFs)
T_kin_total = np.sum(T_kin, axis=1)

# Total potential energy from all springs
V_springs_total = np.sum(V_springs, axis=1)
V_ground_total  = np.sum(V_ground, axis=1)
V_total = V_springs_total + V_ground_total

# Plot total energies vs. omega
plt.figure(figsize=(8, 6))
plt.plot(omega_vals, T_kin_total, label='Total Kinetic Energy (peak)')
plt.plot(omega_vals, V_total, label='Total Potential Energy (peak)')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Energy [J] (arbitrary units)')
plt.title('Peak Kinetic & Potential Energies vs. Frequency')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# If you want to see distribution (fractions) among DOFs or among springs:
# -----------------------------------------------------------------------------

# For example, fraction of total KE in each DOF at a single frequency index
freq_index = 50  # pick some frequency index
omega_plot = omega_vals[freq_index]
fraction_KE = T_kin[freq_index, :] / T_kin_total[freq_index]

print(f"At omega = {omega_plot:.2f} rad/s:")
for j in range(n):
    print(f"  DOF {j} fraction of total KE = {fraction_KE[j]*100:.2f}%")

# Similarly for spring potential energies
fraction_PE_spr = V_springs[freq_index, :] / V_springs_total[freq_index] if V_springs_total[freq_index]!=0 else 0
fraction_PE_gnd = V_ground[freq_index, :] / V_ground_total[freq_index] if V_ground_total[freq_index]!=0 else 0

print("Spring potential energy fractions (between DOFs):")
for s_idx, (k_s, dof_p, dof_q) in enumerate(springs):
    print(f"  Spring {dof_p}-{dof_q} fraction of total spring PE = {fraction_PE_spr[s_idx]*100:.2f}%")

print("Spring potential energy fractions (to ground):")
for s_idx, (k_s, dof_j) in enumerate(springs_to_ground):
    print(f"  Spring {dof_j}-ground fraction of total ground PE = {fraction_PE_gnd[s_idx]*100:.2f}%")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# --- (A) Kinetic Energy Distribution ---
plt.subplot(1, 2, 1)
plt.bar(np.arange(n), fraction_KE * 100, color='steelblue')
plt.xlabel("DOF index")
plt.ylabel("Fraction of total KE [%]")
plt.title(f"Kinetic Energy at ω={omega_plot:.2f} rad/s")
plt.xticks(np.arange(n))

# --- (B) Potential Energy Distribution ---
# Combine the between-DOF springs + ground springs into one list
springs_labels = [f"{dof_p}-{dof_q}" for (k_s, dof_p, dof_q) in springs]
ground_labels  = [f"{dof_j}-gnd" for (k_s, dof_j) in springs_to_ground]

PE_spring_distribution = np.concatenate((fraction_PE_spr, fraction_PE_gnd))
PE_spring_labels       = springs_labels + ground_labels

plt.subplot(1, 2, 2)
plt.bar(np.arange(len(PE_spring_distribution)), PE_spring_distribution * 100, color='orange')
plt.ylabel("Fraction of total PE [%]")
plt.title("Potential Energy Distribution")
plt.xticks(np.arange(len(PE_spring_distribution)), PE_spring_labels, rotation=45)

plt.tight_layout()
plt.show()
