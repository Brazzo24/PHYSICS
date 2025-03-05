# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt

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
C_crank_clutch = 0.01  # Damping at clutch
C_clutch_gearbox = 0.05  # Damping in transmission
C_gearbox_driveshaft = 0.2  # Minimal damping in driveshaft
C_driveshaft_wheel = 0.8  # Damping in the wheel-tire interface

# System parameters as arrays for further computation
J_values = np.array([J_crankshaft, J_clutch, J_gearbox, J_driveshaft, J_wheel])
K_values = np.array([K_crank_clutch, K_clutch_gearbox, K_gearbox_driveshaft, K_driveshaft_wheel])
C_values = np.array([C_crank_clutch, C_clutch_gearbox, C_gearbox_driveshaft, C_driveshaft_wheel])

# Number of DOFs (each inertia is a separate rotational DOF)
N_dof_driveline = len(J_values)

# Define frequency range for analysis (0 - 150 Hz)
f_min = 0.1   # Avoid division by zero
f_max = 150.0
num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s


# **Step 2: Compute System Response with Base Excitation at the Drive-Wheel**
# Define base excitation at the last DOF (drive-wheel) with unit amplitude
F_base_excitation = np.zeros(N_dof_driveline, dtype=complex)
F_base_excitation[-1] = 1.0  # Unit amplitude at the drive-wheel

# Function to compute system response under base excitation
def compute_response_with_base_excitation(N, w_vals, inertias, dampings, stiffnesses, excitation):
    """Computes system response given a base excitation."""

    # Initialize response storage
    theta_vals = np.zeros((N, len(w_vals)), dtype=complex)

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

        # Solve for angular displacements (θ values) under base excitation
        theta_vals[:, idx] = np.linalg.solve(Z_matrix, excitation)

    return theta_vals

# Compute response for the driveline system with base excitation
theta_base_excitation = compute_response_with_base_excitation(
    N_dof_driveline, w_vals, J_values, C_values, K_values, F_base_excitation
)

# **Step 3: Compute Active & Reactive Power for Base Excitation Case**
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

        # Compute angular velocities (ωθ)
        theta_vals = np.linalg.solve(Z_matrix, F_base_excitation)
        omega_theta = 1j * w * theta_vals

        # Compute active power dissipation (P_active = damping * |velocity|^2)
        P_active[idx] = np.sum(C_matrix @ omega_theta * np.conj(omega_theta))

        # Compute reactive power (P_reactive = stiffness * |displacement|^2)
        P_reactive[idx] = np.sum(K_matrix @ theta_vals * np.conj(theta_vals))

    return np.real(P_active), np.real(P_reactive)  # Extract only real parts

# Compute active and reactive power with base excitation
P_active_base, P_reactive_base = compute_active_reactive_power(
    N_dof_driveline, w_vals, J_values, C_values, K_values
)

# **Step 4: Plot Active & Reactive Power with Base Excitation**
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

ax1.plot(f_vals, P_active_base, 'b', lw=2, label="Active Power (Base Excitation)")
ax2.plot(f_vals, P_reactive_base, 'r', lw=2, linestyle="dashed", label="Reactive Power (Base Excitation)")

ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Active Power Dissipation (W)", color="b")
ax2.set_ylabel("Reactive Power (Vibrational Energy) (W)", color="r")

ax1.set_title("Effect of Base Excitation on Active & Reactive Power")
ax1.grid()
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.show()

# **Step 5: Compute Total Energy for Base Excitation Case**
total_active_base = np.trapz(P_active_base, f_vals)
total_reactive_base = np.trapz(P_reactive_base, f_vals)

# **Print Updated Power Analysis Results**
print(f"Total Active Power Dissipated (Base Excitation): {total_active_base:.2f} Joules")
print(f"Total Reactive Power (Base Excitation): {total_reactive_base:.2f} Joules")