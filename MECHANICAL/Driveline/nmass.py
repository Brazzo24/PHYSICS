# Corrected 4-DOF system with proper mass coupling
# Function to compute dynamic stiffness

import numpy as np
import matplotlib.pyplot as plt


# Updated System Parameters (from image)
m1 = 1.0   # Lower mass [kg]
c1 = 10.0  # Lower damping [Ns/m] (converted from 0.01 Ns/mm)
k1 = 2000.0  # Lower stiffness [N/m] (converted from 2 N/mm)

m2 = 1.0   # Upper mass [kg]
c2 = 2.0   # Upper damping [Ns/m] (converted from 0.001 Ns/mm)
k2 = 12000.0  # Upper stiffness [N/m] (converted from 4 N/mm)

# Updated Frequency Range (0 to 35 Hz)
f_min = 0.1   # Avoid division by zero
f_max = 35.0

m3 = 1.5   # Third mass [kg]
c3 = 5.0   # Third damping [Ns/m]
k3 = 3000.0  # Third stiffness [N/m]

m4 = 0.8   # Fourth mass [kg]
c4 = 3.0   # Fourth damping [Ns/m]
k4 = 3500.0  # Fourth stiffness [N/m]

num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s


def dynamic_stiffness(m, c, k, w):
    """Computes the dynamic stiffness: K_d(w) = k - w^2 * m + j * w * c"""
    return k - (w**2) * m + 1j * w * c

def compute_4DOF_response(m1, c1, k1, m2, c2, k2, m3, c3, k3, m4, c4, k4, w):
    """Solves for X1, X2, X3, X4 given a base velocity excitation with correct mass coupling."""

    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)
    Kd3 = dynamic_stiffness(m3, c3, k3, w)
    Kd4 = dynamic_stiffness(m4, c4, k4, w)

    # Define base velocity excitation (1 m/s)
    V_base = 1.0  # Unit velocity excitation
    X_base = V_base / (1j * w)  # Convert velocity to displacement in frequency domain

    # Construct system matrix for 4-DOF system with correct coupling
    M = np.array([
        [Kd1 + k2, -k2, 0, 0],      # Mass 1 connected to Mass 2 via k2
        [-k2, Kd2 + k3, -k3, 0],    # Mass 2 connected to Mass 1 & 3
        [0, -k3, Kd3 + k4, -k4],    # Mass 3 connected to Mass 2 & 4
        [0, 0, -k4, Kd4]            # Mass 4 connected only to Mass 3
    ])

    # Right-hand side force vector due to base velocity excitation
    F_input = -(1j * w * c1 + k1) * X_base  # Force applied to the first mass only
    RHS = np.array([F_input, 0, 0, 0])

    # Solve for displacements X1, X2, X3, X4
    X1, X2, X3, X4 = np.linalg.solve(M, RHS)

    return X1, X2, X3, X4


# Compute response for each frequency with 4-DOF system (corrected coupling)
X1_vals_4DOF, X2_vals_4DOF, X3_vals_4DOF, X4_vals_4DOF = (
    np.zeros_like(w_vals, dtype=complex),
    np.zeros_like(w_vals, dtype=complex),
    np.zeros_like(w_vals, dtype=complex),
    np.zeros_like(w_vals, dtype=complex),
)

for i, w in enumerate(w_vals):
    X1_vals_4DOF[i], X2_vals_4DOF[i], X3_vals_4DOF[i], X4_vals_4DOF[i] = compute_4DOF_response(
        m1, c1, k1, m2, c2, k2, m3, c3, k3, m4, c4, k4, w
    )

# Compute acceleration responses
A1_vals_4DOF = -w_vals**2 * X1_vals_4DOF
A2_vals_4DOF = -w_vals**2 * X2_vals_4DOF
A3_vals_4DOF = -w_vals**2 * X3_vals_4DOF
A4_vals_4DOF = -w_vals**2 * X4_vals_4DOF

# Extract magnitudes
A1_mag_4DOF, A2_mag_4DOF, A3_mag_4DOF, A4_mag_4DOF = (
    np.abs(A1_vals_4DOF),
    np.abs(A2_vals_4DOF),
    np.abs(A3_vals_4DOF),
    np.abs(A4_vals_4DOF),
)

# Plot Mass 1 to Mass 4 Acceleration Magnitudes (with corrected coupling)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, A1_mag_4DOF, 'b', lw=2, label="Mass 1 Acceleration")
ax.plot(f_vals, A2_mag_4DOF, 'r', lw=2, label="Mass 2 Acceleration")
ax.plot(f_vals, A3_mag_4DOF, 'g', lw=2, label="Mass 3 Acceleration")
ax.plot(f_vals, A4_mag_4DOF, 'm', lw=2, label="Mass 4 Acceleration")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Acceleration |A(ω)| [m/s²]")
ax.set_title("Acceleration Responses of Corrected 4-DOF System")
ax.legend()
ax.grid()
plt.show()


"""

SPARSE MATRIX NDOF 

"""

# Import sparse matrix solver tools
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Define a scalable function for an N-DOF system using a sparse matrix approach
def compute_NDOF_response(N, masses, dampings, stiffnesses, w_vals):
    """Computes response for an N-DOF mass-spring-damper system using a sparse solver."""
    
    # Define base velocity excitation (1 m/s)
    V_base = 1.0  # Unit velocity excitation
    X_base = V_base / (1j * w_vals)  # Convert velocity to displacement in frequency domain

    # Initialize storage for responses
    X_vals = np.zeros((N, len(w_vals)), dtype=complex)
    
    # Iterate over all frequency values
    for idx, w in enumerate(w_vals):
        # Construct sparse dynamic stiffness matrix
        main_diag = np.zeros(N, dtype=complex)
        upper_diag = np.zeros(N-1, dtype=complex)
        lower_diag = np.zeros(N-1, dtype=complex)
        
        for i in range(N):
            # Compute dynamic stiffness for each mass
            Kd = dynamic_stiffness(masses[i], dampings[i], stiffnesses[i], w)
            main_diag[i] = Kd  # Main diagonal
            
            if i > 0:
                upper_diag[i-1] = -stiffnesses[i]  # Upper diagonal
                lower_diag[i-1] = -stiffnesses[i]  # Lower diagonal

        # Construct sparse matrix
        M_sparse = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format="csr")

        # Define force input (only applied to first mass)
        RHS = np.zeros(N, dtype=complex)
        RHS[0] = -(1j * w * dampings[0] + stiffnesses[0]) * X_base[idx]

        # Solve the sparse system
        X_vals[:, idx] = spsolve(M_sparse, RHS)

    return X_vals

# Define a larger 10-DOF system
N_dof = 10  # Number of masses
masses = np.linspace(1.0, 2.0, N_dof)  # Linearly increasing mass
dampings = np.linspace(5.0, 15.0, N_dof)  # Increasing damping
stiffnesses = np.linspace(2000, 6000, N_dof)  # Increasing stiffness

# Compute response using the sparse solver
X_vals_NDOF = compute_NDOF_response(N_dof, masses, dampings, stiffnesses, w_vals)

# Compute acceleration responses for each mass
A_vals_NDOF = -w_vals**2 * X_vals_NDOF

# Plot Acceleration Magnitudes for a Selection of Masses
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N_dof):
    ax.plot(f_vals, np.abs(A_vals_NDOF[i, :]), lw=1.5, label=f"Mass {i+1}")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Acceleration |A(ω)| [m/s²]")
ax.set_title(f"Acceleration Responses of {N_dof}-DOF System (Sparse Solver)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid()
plt.show()

"""

MODE SHAPE PLOT

"""

# Import necessary libraries for eigenvalue decomposition
from scipy.linalg import eigh

# Function to compute modal analysis for an N-DOF system
def compute_modal_analysis(N, masses, dampings, stiffnesses):
    """Computes modal frequencies and mode shapes for an N-DOF system."""
    
    # Construct mass and stiffness matrices
    M_matrix = np.diag(masses)  # Mass matrix (diagonal)
    K_matrix = np.zeros((N, N))  # Stiffness matrix

    for i in range(N):
        K_matrix[i, i] += stiffnesses[i]  # Main diagonal

        if i > 0:
            K_matrix[i, i] += stiffnesses[i]  # Additional contribution
            K_matrix[i, i-1] = -stiffnesses[i]  # Coupling term
            K_matrix[i-1, i] = -stiffnesses[i]  # Symmetric

    # Solve the generalized eigenvalue problem K * Φ = λ * M * Φ
    eigvals, eigvecs = eigh(K_matrix, M_matrix)

    # Convert eigenvalues to natural frequencies (Hz)
    natural_frequencies = np.sqrt(np.abs(eigvals)) / (2 * np.pi)  

    return natural_frequencies, eigvecs

# Compute modal analysis for the 10-DOF system
natural_frequencies, mode_shapes = compute_modal_analysis(N_dof, masses, dampings, stiffnesses)

# Plot Natural Frequencies and Mode Shapes
fig, ax = plt.subplots(figsize=(7, 5))

for i in range(N_dof):
    ax.plot(np.arange(1, N_dof + 1), mode_shapes[:, i], marker='o', linestyle="-", label=f"Mode {i+1}")

ax.set_xlabel("Mass Index")
ax.set_ylabel("Mode Shape Amplitude")
ax.set_title("Mode Shapes of 10-DOF System")
ax.legend()
ax.grid()
plt.show()

# Print natural frequencies
print("Natural Frequencies (Hz):")
for i, f in enumerate(natural_frequencies):
    print(f"Mode {i+1}: {f:.2f} Hz")

 
"""
 
 MODAL SUPERPOSITION
 
"""

# Function to compute modal superposition response
def compute_modal_response(N, w_vals, natural_frequencies, mode_shapes, masses, stiffnesses):
    """Computes the modal response approximation using the dominant modes."""

    # Number of modes to include in reconstruction
    num_modes = N  # Use all modes for accurate reconstruction
    
    # Initialize modal response storage
    modal_contributions = np.zeros((N, len(w_vals)), dtype=complex)
    
    # Iterate over modes to compute modal coordinates
    for i in range(num_modes):
        # Extract mode shape and eigenvalue (stiffness term in modal coordinates)
        phi_i = mode_shapes[:, i]
        lambda_i = (2 * np.pi * natural_frequencies[i]) ** 2  # Convert frequency to stiffness term

        # Compute modal coordinate q_i(w) = (Φ_i^T * F) / (λ_i - ω²)
        for j, w in enumerate(w_vals):
            modal_contributions[:, j] += (phi_i * (phi_i[0] / (lambda_i - w**2)))  # Excitation applied at first mass

    # Reconstruct total response using modal superposition
    reconstructed_response = np.sum(modal_contributions, axis=0)

    return reconstructed_response, modal_contributions

# Compute modal response reconstruction
X_modal_reconstructed, modal_contributions = compute_modal_response(
    N_dof, w_vals, natural_frequencies, mode_shapes, masses, stiffnesses
)

# Compute acceleration from modal response
A_modal_reconstructed = -w_vals**2 * X_modal_reconstructed

# Compute full system response for comparison
X_full_response = compute_NDOF_response(N_dof, masses, dampings, stiffnesses, w_vals)
A_full_response = -w_vals**2 * X_full_response[0, :]  # Only first mass for direct comparison

# Plot Comparison: Modal Approximation vs. Full Response
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, np.abs(A_full_response), 'b', lw=2, label="Full System Response")
ax.plot(f_vals, np.abs(A_modal_reconstructed), 'r', lw=2, linestyle="dashed", label="Modal Approximation")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Acceleration |A(ω)| [m/s²]")
ax.set_title("Modal Superposition Approximation vs. Full System Response")
ax.legend()
ax.grid()
plt.show()

# Plot Individual Mode Contributions
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(N_dof):
    ax.plot(f_vals, np.abs(modal_contributions[i, :]), lw=1.5, label=f"Mode {i+1}")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Modal Contribution |q_i(ω)|")
ax.set_title("Individual Modal Contributions to the Response")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid()
plt.show()


"""

OPTIMIZED DAMPING

"""

# Identify optimal damping placement based on mode shapes

# Find the masses with the highest displacement in the first 3 modes
num_critical_modes = 3
optimal_damping_indices = np.argmax(np.abs(mode_shapes[:, :num_critical_modes]), axis=0)

# Define new damping values with increased damping at critical locations
optimized_dampings = np.copy(dampings)
for idx in optimal_damping_indices:
    optimized_dampings[idx] *= 3  # Increase damping by a factor of 3

# Compute response with optimized damping
X_vals_optimized = compute_NDOF_response(N_dof, masses, optimized_dampings, stiffnesses, w_vals)
A_vals_optimized = -w_vals**2 * X_vals_optimized[0, :]  # Only first mass for comparison

# Plot Comparison: Original vs. Optimized Damping
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, np.abs(A_full_response), 'b', lw=2, label="Original Damping")
ax.plot(f_vals, np.abs(A_vals_optimized), 'r', lw=2, linestyle="dashed", label="Optimized Damping")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Acceleration |A(ω)| [m/s²]")
ax.set_title("Effect of Optimized Damping on System Response")
ax.legend()
ax.grid()
plt.show()

# Print optimized damping values
print("Optimal damping applied at mass indices:", optimal_damping_indices)
print("New damping values:", optimized_dampings)




# Compute Energy Dissipation in the Damping Elements

# Compute power dissipation (Active Power) for original and optimized damping
P_damping_original = np.zeros_like(w_vals, dtype=complex)
P_damping_optimized = np.zeros_like(w_vals, dtype=complex)

for i, w in enumerate(w_vals):
    X_original = X_full_response[:, i]  # Original displacement
    X_optimized = X_vals_optimized[:, i]  # Optimized displacement

    # Compute velocity phasors (V = jωX)
    V_original = 1j * w * X_original
    V_optimized = 1j * w * X_optimized

    # Compute power dissipated by damping elements (Active Power)
    P_damping_original[i] = np.sum(dampings * V_original * np.conj(V_original))  # Original damping power
    P_damping_optimized[i] = np.sum(optimized_dampings * V_optimized * np.conj(V_optimized))  # Optimized damping power

# Extract real power dissipation (Active Power)
P_damping_original_real = np.real(P_damping_original)
P_damping_optimized_real = np.real(P_damping_optimized)

# Plot Energy Dissipation Comparison
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, P_damping_original_real, 'b', lw=2, label="Original Damping Power")
ax.plot(f_vals, P_damping_optimized_real, 'r', lw=2, linestyle="dashed", label="Optimized Damping Power")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping Power Dissipation (W)")
ax.set_title("Energy Dissipation in Damping Elements")
ax.legend()
ax.grid()
plt.show()

# Compute total energy dissipated over frequency range
total_energy_original = np.trapz(P_damping_original_real, f_vals)
total_energy_optimized = np.trapz(P_damping_optimized_real, f_vals)

# Print total energy dissipation comparison
print(f"Total Energy Dissipated (Original Damping): {total_energy_original:.2f} Joules")
print(f"Total Energy Dissipated (Optimized Damping): {total_energy_optimized:.2f} Joules")
print(f"Percentage Improvement: {((total_energy_optimized - total_energy_original) / total_energy_original) * 100:.2f}%")


# **Step 1: Identify Critical Frequencies** (From previous modal analysis)
critical_frequencies = natural_frequencies[:3]  # Focus on the first three critical modes

# **Step 2: Modify Stiffness to Shift Natural Frequencies**

# Strategy: Increase stiffness where resonance is strong to push frequencies higher
optimized_stiffnesses = np.copy(stiffnesses)
for i in range(len(critical_frequencies)):
    # Increase stiffness of masses with high mode shape amplitude in critical modes
    idx = np.argmax(np.abs(mode_shapes[:, i]))  # Find the most affected mass in mode i
    optimized_stiffnesses[idx] *= 1.5  # Increase stiffness by 50%

# **Step 3: Recompute Modal Analysis with Optimized Stiffness**
optimized_natural_frequencies, optimized_mode_shapes = compute_modal_analysis(
    N_dof, masses, dampings, optimized_stiffnesses
)

# **Step 4: Re-apply Optimal Damping to New Mode Shapes**

# Identify the new best damping locations based on mode shapes
optimal_damping_indices_optimized = np.argmax(np.abs(optimized_mode_shapes[:, :num_critical_modes]), axis=0)

# Apply increased damping at these new critical locations
hybrid_dampings = np.copy(dampings)
for idx in optimal_damping_indices_optimized:
    hybrid_dampings[idx] *= 3  # Increase damping by a factor of 3

# **Step 5: Compute Response for Hybrid Optimized System**
X_vals_hybrid = compute_NDOF_response(N_dof, masses, hybrid_dampings, optimized_stiffnesses, w_vals)
A_vals_hybrid = -w_vals**2 * X_vals_hybrid[0, :]  # First mass for comparison

# **Step 6: Compare Results**
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, np.abs(A_full_response), 'b', lw=2, label="Original System")
ax.plot(f_vals, np.abs(A_vals_optimized), 'r', lw=2, linestyle="dashed", label="Damping Optimization")
ax.plot(f_vals, np.abs(A_vals_hybrid), 'g', lw=2, linestyle="dotted", label="Hybrid Optimization (Stiffness + Damping)")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Acceleration |A(ω)| [m/s²]")
ax.set_title("Comparison: Original vs. Damping Optimization vs. Hybrid Optimization")
ax.legend()
ax.grid()
plt.show()

# **Step 7: Evaluate Energy Dissipation in the Hybrid System**
P_damping_hybrid = np.zeros_like(w_vals, dtype=complex)

for i, w in enumerate(w_vals):
    X_hybrid = X_vals_hybrid[:, i]
    V_hybrid = 1j * w * X_hybrid
    P_damping_hybrid[i] = np.sum(hybrid_dampings * V_hybrid * np.conj(V_hybrid))

# Extract real power dissipation (Active Power)
P_damping_hybrid_real = np.real(P_damping_hybrid)

# Plot Energy Dissipation Comparison
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, P_damping_original_real, 'b', lw=2, label="Original Damping Power")
ax.plot(f_vals, P_damping_optimized_real, 'r', lw=2, linestyle="dashed", label="Optimized Damping Power")
ax.plot(f_vals, P_damping_hybrid_real, 'g', lw=2, linestyle="dotted", label="Hybrid Optimization Power")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping Power Dissipation (W)")
ax.set_title("Energy Dissipation in Hybrid Optimized System")
ax.legend()
ax.grid()
plt.show()

# Compute total energy dissipation over frequency range
total_energy_hybrid = np.trapz(P_damping_hybrid_real, f_vals)

# Print energy dissipation comparison
print(f"Total Energy Dissipated (Original Damping): {total_energy_original:.2f} Joules")
print(f"Total Energy Dissipated (Optimized Damping): {total_energy_optimized:.2f} Joules")
print(f"Total Energy Dissipated (Hybrid Optimization): {total_energy_hybrid:.2f} Joules")
print(f"Improvement over Original: {((total_energy_hybrid - total_energy_original) / total_energy_original) * 100:.2f}%")
print(f"Improvement over Damping-Only Optimization: {((total_energy_hybrid - total_energy_optimized) / total_energy_optimized) * 100:.2f}%")

"""

STUFF

"""

# **Mass Redistribution Strategy**

# Strategy: Increase mass where energy concentration is high (from mode shapes)
optimized_masses = np.copy(masses)
for i in range(len(critical_frequencies)):
    idx = np.argmax(np.abs(optimized_mode_shapes[:, i]))  # Find the most affected mass in mode i
    optimized_masses[idx] *= 1.5  # Increase mass by 50%

# **Recompute Modal Analysis with Mass Redistribution**
redistributed_natural_frequencies, redistributed_mode_shapes = compute_modal_analysis(
    N_dof, optimized_masses, hybrid_dampings, optimized_stiffnesses
)

# **Compute Response with Mass Redistribution**
X_vals_mass_redistributed = compute_NDOF_response(N_dof, optimized_masses, hybrid_dampings, optimized_stiffnesses, w_vals)
A_vals_mass_redistributed = -w_vals**2 * X_vals_mass_redistributed[0, :]  # First mass for comparison

# **Compare Results: Hybrid Optimization vs. Mass Redistribution**
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, np.abs(A_full_response), 'b', lw=2, label="Original System")
ax.plot(f_vals, np.abs(A_vals_hybrid), 'r', lw=2, linestyle="dashed", label="Hybrid Optimization (Stiffness + Damping)")
ax.plot(f_vals, np.abs(A_vals_mass_redistributed), 'g', lw=2, linestyle="dotted", label="Mass Redistribution Optimization")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Acceleration |A(ω)| [m/s²]")
ax.set_title("Comparison: Original vs. Hybrid Optimization vs. Mass Redistribution")
ax.legend()
ax.grid()
plt.show()

# **Compute Energy Dissipation for Mass Redistribution**
P_damping_mass_redistributed = np.zeros_like(w_vals, dtype=complex)

for i, w in enumerate(w_vals):
    X_mass_redistributed = X_vals_mass_redistributed[:, i]
    V_mass_redistributed = 1j * w * X_mass_redistributed
    P_damping_mass_redistributed[i] = np.sum(hybrid_dampings * V_mass_redistributed * np.conj(V_mass_redistributed))

# Extract real power dissipation (Active Power)
P_damping_mass_redistributed_real = np.real(P_damping_mass_redistributed)

# **Plot Energy Dissipation Comparison**
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, P_damping_original_real, 'b', lw=2, label="Original Damping Power")
ax.plot(f_vals, P_damping_hybrid_real, 'r', lw=2, linestyle="dashed", label="Hybrid Optimization Power")
ax.plot(f_vals, P_damping_mass_redistributed_real, 'g', lw=2, linestyle="dotted", label="Mass Redistribution Power")

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping Power Dissipation (W)")
ax.set_title("Energy Dissipation in Mass Redistribution Optimization")
ax.legend()
ax.grid()
plt.show()

# **Compute Total Energy Dissipation for Mass Redistribution**
total_energy_mass_redistributed = np.trapz(P_damping_mass_redistributed_real, f_vals)

# **Print Energy Dissipation Comparison**
print(f"Total Energy Dissipated (Original Damping): {total_energy_original:.2f} Joules")
print(f"Total Energy Dissipated (Hybrid Optimization): {total_energy_hybrid:.2f} Joules")
print(f"Total Energy Dissipated (Mass Redistribution): {total_energy_mass_redistributed:.2f} Joules")
print(f"Improvement over Original: {((total_energy_mass_redistributed - total_energy_original) / total_energy_original) * 100:.2f}%")
print(f"Improvement over Hybrid Optimization: {((total_energy_mass_redistributed - total_energy_hybrid) / total_energy_hybrid) * 100:.2f}%")