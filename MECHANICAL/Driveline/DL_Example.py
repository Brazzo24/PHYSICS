# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# **Step 1: Define the Torsional Multi-DOF System for the Vehicle Driveline**

# Define rotational inertias for key components (in kg·m²)
J_crankshaft = 0.15  # Crankshaft inertia
J_clutch = 0.08      # Clutch inertia
J_gearbox = 0.12     # Gearbox input inertia
J_driveshaft = 0.10  # Driveshaft inertia
J_wheel = 0.25       # Drive-wheel inertia

# Define torsional stiffness values (in N·m/rad)
K_crank_clutch = 15000  # Between crankshaft and clutch
K_clutch_gearbox = 12000  # Between clutch and gearbox
K_gearbox_driveshaft = 8000  # Between gearbox and driveshaft
K_driveshaft_wheel = 10000  # Between driveshaft and wheel

# Define torsional damping values (in N·m·s/rad)
C_crank_clutch = 10  # Damping at clutch
C_clutch_gearbox = 5  # Damping in transmission
C_gearbox_driveshaft = 2  # Minimal damping in driveshaft
C_driveshaft_wheel = 8  # Damping in the wheel-tire interface

# System parameters as arrays for further computation
J_values = np.array([J_crankshaft, J_clutch, J_gearbox, J_driveshaft, J_wheel])
K_values = np.array([K_crank_clutch, K_clutch_gearbox, K_gearbox_driveshaft, K_driveshaft_wheel])
C_values = np.array([C_crank_clutch, C_clutch_gearbox, C_gearbox_driveshaft, C_driveshaft_wheel])

# Number of DOFs (each inertia is a separate rotational DOF)
N_dof_driveline = len(J_values)

# Function to compute modal analysis for an N-DOF torsional system
def compute_modal_analysis(N, inertias, dampings, stiffnesses):
    """Computes modal frequencies and mode shapes for an N-DOF torsional system."""

    # Construct mass (inertia) and stiffness matrices
    M_matrix = np.diag(inertias)  # Inertia matrix (diagonal)
    K_matrix = np.zeros((N, N))  # Stiffness matrix

    for i in range(N):
        K_matrix[i, i] += stiffnesses[i] if i < N - 1 else 0  # Main diagonal
        if i > 0:
            K_matrix[i, i] += stiffnesses[i-1]  # Contribution from previous connection
            K_matrix[i, i-1] = -stiffnesses[i-1]  # Coupling term
            K_matrix[i-1, i] = -stiffnesses[i-1]  # Symmetric term

    # Solve the generalized eigenvalue problem K * Φ = λ * M * Φ
    eigvals, eigvecs = eigh(K_matrix, M_matrix)

    # Convert eigenvalues to natural frequencies (Hz)
    natural_frequencies = np.sqrt(np.abs(eigvals)) / (2 * np.pi)

    return natural_frequencies, eigvecs

# **Step 2: Compute Modal Analysis for the Driveline System**
natural_frequencies_driveline, mode_shapes_driveline = compute_modal_analysis(N_dof_driveline, J_values, C_values, K_values)

# **Plot Mode Shapes for the Driveline**
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(N_dof_driveline):
    ax.plot(np.arange(1, N_dof_driveline + 1), mode_shapes_driveline[:, i], marker='o', linestyle="-", label=f"Mode {i+1}")

ax.set_xlabel("Component Index (Crankshaft → Wheel)")
ax.set_ylabel("Mode Shape Amplitude")
ax.set_title("Mode Shapes of Vehicle Driveline")
ax.legend()
ax.grid()
plt.show()

# **Print Natural Frequencies of the Driveline**
print("Natural Frequencies of the Vehicle Driveline (Hz):")
for i, f in enumerate(natural_frequencies_driveline):
    print(f"Mode {i+1}: {f:.2f} Hz")


"""

TORSIONAL DAMPING EFFICIENCY

"""

# **Step 3: Compute Damping Efficiency and Energy Dissipation**

# Define frequency range for analysis (0 - 150 Hz)
f_min = 0.1   # Avoid division by zero
f_max = 150.0
num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s

# Function to compute response and energy dissipation with damping
def compute_damping_efficiency(N, w_vals, inertias, dampings, stiffnesses):
    """Computes the torsional damping efficiency by analyzing energy dissipation."""
    
    # Define storage for power dissipation
    P_damping = np.zeros(len(w_vals), dtype=complex)
    
    # Iterate over frequencies
    for idx, w in enumerate(w_vals):
        # Construct system matrices
        M_matrix = np.diag(inertias)
        K_matrix = np.zeros((N, N), dtype=complex)
        C_matrix = np.zeros((N, N), dtype=complex)

        for i in range(N):
            K_matrix[i, i] += stiffnesses[i] if i < N - 1 else 0
            C_matrix[i, i] += dampings[i] if i < N - 1 else 0

            if i > 0:
                K_matrix[i, i] += stiffnesses[i-1]
                K_matrix[i, i-1] = -stiffnesses[i-1]
                K_matrix[i-1, i] = -stiffnesses[i-1]

                C_matrix[i, i] += dampings[i-1]
                C_matrix[i, i-1] = -dampings[i-1]
                C_matrix[i-1, i] = -dampings[i-1]

        # Compute frequency-dependent system matrix
        Z_matrix = K_matrix - w**2 * M_matrix + 1j * w * C_matrix

        # Define input excitation (torque applied to the crankshaft)
        F_torque = np.zeros(N, dtype=complex)
        F_torque[0] = 1.0  # Unit torque applied at the crankshaft

        # Solve for angular displacements (θ values)
        theta_vals = np.linalg.solve(Z_matrix, F_torque)

        # Compute angular velocities (ωθ) for power dissipation
        omega_theta = 1j * w * theta_vals

        # Compute power dissipation in damping elements
        P_damping[idx] = np.sum(C_matrix @ omega_theta * np.conj(omega_theta))

    return np.real(P_damping)  # Return only real part (active power dissipation)

# Compute damping efficiency for the original system
P_damping_original = compute_damping_efficiency(N_dof_driveline, w_vals, J_values, C_values, K_values)

# **Step 4: Plot Damping Efficiency Results**
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, P_damping_original, 'b', lw=2, label="Original System Damping Power")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping Power Dissipation (W)")
ax.set_title("Torsional Damping Efficiency in the Driveline")
ax.legend()
ax.grid()
plt.show()

# Compute total energy dissipated over frequency range
total_energy_damping = np.trapz(P_damping_original, f_vals)

# **Print Energy Dissipation Analysis**
print(f"Total Energy Dissipated by Damping: {total_energy_damping:.2f} Joules")
print(f"Peak Damping Efficiency at {f_vals[np.argmax(P_damping_original)]:.2f} Hz")


"""

CLUTCH DAMPER

"""

# **Step 5: Introduce Additional Damping (e.g., Clutch Damper) and Evaluate Impact**

# Strategy: Add a torsional damper at the clutch (common in dual-mass flywheels)
additional_damping_clutch = 20  # Additional damping in N·m·s/rad

# Create a new damping array with the added clutch damper
C_values_clutch_damped = np.copy(C_values)
C_values_clutch_damped[1] += additional_damping_clutch  # Increase damping at clutch location

# Compute damping efficiency with the additional clutch damper
P_damping_clutch_damped = compute_damping_efficiency(N_dof_driveline, w_vals, J_values, C_values_clutch_damped, K_values)

# **Step 6: Compare Damping Efficiency Before and After Adding the Clutch Damper**
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, P_damping_original, 'b', lw=2, label="Original System Damping Power")
ax.plot(f_vals, P_damping_clutch_damped, 'r', lw=2, linestyle="dashed", label="With Clutch Damper")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping Power Dissipation (W)")
ax.set_title("Effect of Clutch Damper on Torsional Damping Efficiency")
ax.legend()
ax.grid()
plt.show()

# Compute total energy dissipated over frequency range for the new system
total_energy_damping_clutch = np.trapz(P_damping_clutch_damped, f_vals)

# **Print Energy Dissipation Comparison**
print(f"Total Energy Dissipated (Original Damping): {total_energy_damping:.2f} Joules")
print(f"Total Energy Dissipated (With Clutch Damper): {total_energy_damping_clutch:.2f} Joules")
print(f"Improvement in Energy Dissipation: {((total_energy_damping_clutch - total_energy_damping) / total_energy_damping) * 100:.2f}%")

"""

TUNED TORSIONAL DAMPER

"""

# **Step 7: Explore Different Types of Torsional Dampers**

# Strategy: Introduce a **Tuned Torsional Damper (TTD)** at the gearbox
# - A tuned torsional damper is designed to **target a specific vibration mode**.
# - We will add it at the **gearbox** because gearbox modes were dominant in mid-to-high frequencies.

# Define parameters for the tuned torsional damper (TTD)
J_ttd = 0.02  # Inertia of the torsional damper [kg·m²]
K_ttd = 5000  # Stiffness of the torsional damper [N·m/rad]
C_ttd = 15    # Damping of the torsional damper [N·m·s/rad]

# Expand the system to include the tuned torsional damper (TTD) at gearbox location
J_values_TTD = np.insert(J_values, 2, J_ttd)  # Add damper inertia at gearbox location
K_values_TTD = np.insert(K_values, 2, K_ttd)  # Add damper stiffness connection
C_values_TTD = np.insert(C_values, 2, C_ttd)  # Add damper damping

# Update number of DOFs with added damper
N_dof_TTD = len(J_values_TTD)

# Compute damping efficiency with the added torsional damper
P_damping_TTD = compute_damping_efficiency(N_dof_TTD, w_vals, J_values_TTD, C_values_TTD, K_values_TTD)

# **Step 8: Compare Different Damping Solutions**
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, P_damping_original, 'b', lw=2, label="Original Damping Power")
ax.plot(f_vals, P_damping_clutch_damped, 'r', lw=2, linestyle="dashed", label="With Clutch Damper")
ax.plot(f_vals, P_damping_TTD, 'g', lw=2, linestyle="dotted", label="With Tuned Torsional Damper (TTD)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping Power Dissipation (W)")
ax.set_title("Comparison of Different Torsional Dampers")
ax.legend()
ax.grid()
plt.show()

# Compute total energy dissipated over frequency range for the TTD system
total_energy_damping_TTD = np.trapz(P_damping_TTD, f_vals)

# **Print Energy Dissipation Comparison**
print(f"Total Energy Dissipated (Original Damping): {total_energy_damping:.2f} Joules")
print(f"Total Energy Dissipated (With Clutch Damper): {total_energy_damping_clutch:.2f} Joules")
print(f"Total Energy Dissipated (With Tuned Torsional Damper): {total_energy_damping_TTD:.2f} Joules")
print(f"Improvement over Original: {((total_energy_damping_TTD - total_energy_damping) / total_energy_damping) * 100:.2f}%")
print(f"Improvement over Clutch Damper: {((total_energy_damping_TTD - total_energy_damping_clutch) / total_energy_damping_clutch) * 100:.2f}%")

"""

HYBRID OPTIMIZATION

"""

# **Step 9: Hybrid Optimization for the Vehicle Driveline**
# Combining: **Stiffness Tuning, Mass Redistribution, and Optimized Damping**

# **Strategy 1: Shift Resonance Frequencies Using Stiffness Tuning**
# - Increase stiffness where resonance is strong to push frequencies higher.
# - Modify gearbox and driveshaft stiffness as they are critical in mid-to-high frequencies.

optimized_stiffnesses = np.copy(K_values_TTD)
optimized_stiffnesses[1] *= 1.5  # Increase clutch-to-gearbox stiffness
optimized_stiffnesses[3] *= 1.2  # Slightly increase driveshaft stiffness

# **Strategy 2: Redistribute Mass to Balance Mode Energy**
# - Adjust inertias at critical mode locations (mid-range modes).
optimized_masses = np.copy(J_values_TTD)
optimized_masses[2] *= 1.2  # Increase gearbox inertia
optimized_masses[3] *= 0.9  # Reduce driveshaft inertia slightly

# **Strategy 3: Apply Targeted Damping Placement**
# - Increase damping where mode shapes indicate high movement.
optimized_dampings = np.copy(C_values_TTD)
optimized_dampings[1] *= 2  # Double damping at clutch-gearbox connection
optimized_dampings[3] *= 1.5  # Increase damping at driveshaft

# **Compute Hybrid Optimized Response**
P_damping_hybrid = compute_damping_efficiency(N_dof_TTD, w_vals, optimized_masses, optimized_dampings, optimized_stiffnesses)

# **Step 10: Compare Hybrid Optimization vs. Previous Approaches**
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, P_damping_original, 'b', lw=2, label="Original Damping Power")
ax.plot(f_vals, P_damping_clutch_damped, 'r', lw=2, linestyle="dashed", label="With Clutch Damper")
ax.plot(f_vals, P_damping_TTD, 'g', lw=2, linestyle="dotted", label="With Tuned Torsional Damper")
ax.plot(f_vals, P_damping_hybrid, 'm', lw=2, linestyle="dashdot", label="Hybrid Optimization (Stiffness + Damping + Mass Redistribution)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Damping Power Dissipation (W)")
ax.set_title("Comparison of Hybrid Optimization vs. Previous Methods")
ax.legend()
ax.grid()
plt.show()

# Compute total energy dissipated over frequency range for the hybrid system
total_energy_damping_hybrid = np.trapz(P_damping_hybrid, f_vals)

# **Print Energy Dissipation Comparison**
print(f"Total Energy Dissipated (Original Damping): {total_energy_damping:.2f} Joules")
print(f"Total Energy Dissipated (With Clutch Damper): {total_energy_damping_clutch:.2f} Joules")
print(f"Total Energy Dissipated (With Tuned Torsional Damper): {total_energy_damping_TTD:.2f} Joules")
print(f"Total Energy Dissipated (Hybrid Optimization): {total_energy_damping_hybrid:.2f} Joules")
print(f"Improvement over Original: {((total_energy_damping_hybrid - total_energy_damping) / total_energy_damping) * 100:.2f}%")
print(f"Improvement over TTD Optimization: {((total_energy_damping_hybrid - total_energy_damping_TTD) / total_energy_damping_TTD) * 100:.2f}%")

"""

MODE SHAPE OVERLAY

"""

# **Recompute Modal Analysis for Hybrid Optimized System**

# Compute the new mode shapes after hybrid optimization
redistributed_natural_frequencies, redistributed_mode_shapes = compute_modal_analysis(
    N_dof_TTD, optimized_masses, optimized_dampings, optimized_stiffnesses
)

# **Ensure both mode shapes arrays have the same number of DOFs for comparison**
min_dof = min(mode_shapes_driveline.shape[0], redistributed_mode_shapes.shape[0])

# Trim or interpolate the mode shapes if necessary to match DOF count
mode_shapes_trimmed = mode_shapes_driveline[:min_dof, :min_dof]
redistributed_mode_shapes_trimmed = redistributed_mode_shapes[:min_dof, :min_dof]

# **Re-plot Mode Shape Comparison with Corrected Dimensions**
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(min_dof):
    ax.plot(np.arange(1, min_dof + 1), mode_shapes_trimmed[:, i], marker='o', linestyle="-", label=f"Mode {i+1} (Original)")
    ax.plot(np.arange(1, min_dof + 1), redistributed_mode_shapes_trimmed[:, i], marker='s', linestyle="dashed", label=f"Mode {i+1} (Hybrid)")

ax.set_xlabel("Component Index (Crankshaft → Wheel)")
ax.set_ylabel("Mode Shape Amplitude")
ax.set_title("Mode Shape Comparison Before and After Optimization")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid()
plt.show()


"""

STABILITY

"""

# **Corrected Stability Analysis: Compute Eigenvalues Properly**

# Construct the mass, stiffness, and damping matrices in the correct 2D format
M_matrix_opt = np.diag(optimized_masses)  # Mass matrix (diagonal)
K_matrix_opt = np.zeros((N_dof_TTD, N_dof_TTD), dtype=complex)  # Stiffness matrix
C_matrix_opt = np.zeros((N_dof_TTD, N_dof_TTD), dtype=complex)  # Damping matrix

for i in range(N_dof_TTD):
    K_matrix_opt[i, i] += optimized_stiffnesses[i] if i < N_dof_TTD - 1 else 0
    C_matrix_opt[i, i] += optimized_dampings[i] if i < N_dof_TTD - 1 else 0

    if i > 0:
        K_matrix_opt[i, i] += optimized_stiffnesses[i - 1]
        K_matrix_opt[i, i - 1] = -optimized_stiffnesses[i - 1]
        K_matrix_opt[i - 1, i] = -optimized_stiffnesses[i - 1]

        C_matrix_opt[i, i] += optimized_dampings[i - 1]
        C_matrix_opt[i, i - 1] = -optimized_dampings[i - 1]
        C_matrix_opt[i - 1, i] = -optimized_dampings[i - 1]

# Solve the complex eigenvalue problem (stability analysis)
eigenvalues_opt = np.linalg.eigvals(np.linalg.inv(M_matrix_opt) @ (K_matrix_opt - 1j * C_matrix_opt))

# Extract real and imaginary parts of eigenvalues
real_parts_opt = np.real(eigenvalues_opt)
imag_parts_opt = np.imag(eigenvalues_opt)

# **Plot Stability Diagram (Real vs. Imaginary Part of Eigenvalues)**
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(real_parts_opt, imag_parts_opt, c='b', marker='o', label="Optimized System Modes")
ax.axvline(0, color='r', linestyle="dashed", label="Stability Boundary (Re=0)")

ax.set_xlabel("Real Part (Damping Influence)")
ax.set_ylabel("Imaginary Part (Frequency Influence)")
ax.set_title("Stability Analysis: Real vs. Imaginary Parts of Eigenvalues")
ax.legend()
ax.grid()
plt.show()

"""

ACTIVE AND REACTIVE POWER

"""

# **Step 13: Compute Active & Reactive Power for the Driveline**

# Function to compute active and reactive power for the torsional system
def compute_active_reactive_power(N, w_vals, inertias, dampings, stiffnesses):
    """Computes active and reactive power in the torsional system."""

    # Initialize power storage
    P_active = np.zeros(len(w_vals), dtype=complex)
    P_reactive = np.zeros(len(w_vals), dtype=complex)

    # Iterate over frequencies
    for idx, w in enumerate(w_vals):
        # Construct system matrices
        M_matrix = np.diag(inertias)
        K_matrix = np.zeros((N, N), dtype=complex)
        C_matrix = np.zeros((N, N), dtype=complex)

        for i in range(N):
            K_matrix[i, i] += stiffnesses[i] if i < N - 1 else 0
            C_matrix[i, i] += dampings[i] if i < N - 1 else 0

            if i > 0:
                K_matrix[i, i] += stiffnesses[i-1]
                K_matrix[i, i-1] = -stiffnesses[i-1]
                K_matrix[i-1, i] = -stiffnesses[i-1]

                C_matrix[i, i] += dampings[i-1]
                C_matrix[i, i-1] = -dampings[i-1]
                C_matrix[i-1, i] = -dampings[i-1]

        # Compute frequency-dependent system matrix
        Z_matrix = K_matrix - w**2 * M_matrix + 1j * w * C_matrix

        # Define input excitation (torque applied to the crankshaft)
        F_torque = np.zeros(N, dtype=complex)
        F_torque[0] = 1.0  # Unit torque applied at the crankshaft

        # Solve for angular displacements (θ values)
        theta_vals = np.linalg.solve(Z_matrix, F_torque)

        # Compute angular velocities (ωθ)
        omega_theta = 1j * w * theta_vals

        # Compute active power dissipation (P_active = damping * |velocity|^2)
        P_active[idx] = np.sum(C_matrix @ omega_theta * np.conj(omega_theta))

        # Compute reactive power (P_reactive = stiffness * |displacement|^2)
        P_reactive[idx] = np.sum(K_matrix @ theta_vals * np.conj(theta_vals))

    return np.real(P_active), np.real(P_reactive)  # Extract only real parts

# Compute power for the original system
P_active_original, P_reactive_original = compute_active_reactive_power(
    N_dof_driveline, w_vals, J_values, C_values, K_values)

# Compute power for the hybrid-optimized system
P_active_hybrid, P_reactive_hybrid = compute_active_reactive_power(
    N_dof_TTD, w_vals, optimized_masses, optimized_dampings, optimized_stiffnesses)

# **Step 14: Plot Active & Reactive Power for the Driveline**
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

ax1.plot(f_vals, P_active_original, 'b', lw=2, label="Active Power (Original)")
ax1.plot(f_vals, P_active_hybrid, 'r', lw=2, linestyle="dashed", label="Active Power (Hybrid)")

ax2.plot(f_vals, P_reactive_original, 'g', lw=2, label="Reactive Power (Original)", linestyle="dotted")
ax2.plot(f_vals, P_reactive_hybrid, 'm', lw=2, linestyle="dashdot", label="Reactive Power (Hybrid)")

ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Active Power Dissipation (W)", color="b")
ax2.set_ylabel("Reactive Power (Vibrational Energy) (W)", color="g")

ax1.set_title("Active vs. Reactive Power in the Driveline System")
ax1.grid()
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.show()

# **Step 15: Compute Total Energy for Active & Reactive Power**
total_active_original = np.trapz(P_active_original, f_vals)
total_reactive_original = np.trapz(P_reactive_original, f_vals)

total_active_hybrid = np.trapz(P_active_hybrid, f_vals)
total_reactive_hybrid = np.trapz(P_reactive_hybrid, f_vals)

# **Print Power Analysis Results**
print(f"Total Active Power Dissipated (Original): {total_active_original:.2f} Joules")
print(f"Total Reactive Power (Stored Energy, Original): {total_reactive_original:.2f} Joules")
print(f"Total Active Power Dissipated (Hybrid): {total_active_hybrid:.2f} Joules")
print(f"Total Reactive Power (Stored Energy, Hybrid): {total_reactive_hybrid:.2f} Joules")

print(f"Improvement in Active Power Dissipation: {((total_active_hybrid - total_active_original) / total_active_original) * 100:.2f}%")
print(f"Reduction in Reactive Power (Energy Storage): {((total_reactive_original - total_reactive_hybrid) / total_reactive_original) * 100:.2f}%")