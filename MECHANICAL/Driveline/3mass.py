# Re-import necessary libraries since the execution state was reset
import numpy as np
import matplotlib.pyplot as plt

# Re-define system parameters
m1 = 1.0   # Lower mass [kg]
c1 = 10.0  # Lower damping [Ns/m]
k1 = 2000.0  # Lower stiffness [N/m]

m2 = 1.0   # Upper mass [kg]
c2 = 1.0   # Upper damping [Ns/m]
k2 = 4000.0  # Upper stiffness [N/m]

# Frequency range (0 to 35 Hz)
f_min = 0.1   # Avoid division by zero
f_max = 35.0
num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s

# Function to compute dynamic stiffness
def dynamic_stiffness(m, c, k, w):
    """Computes the dynamic stiffness: K_d(w) = k - w^2 * m + j * w * c"""
    return k - (w**2) * m + 1j * w * c

# Function to compute 2DOF response
def compute_2DOF_response(m1, c1, k1, m2, c2, k2, w):
    """Solves for X1 and X2 given a base velocity excitation"""
    
    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)
    
    # Define base velocity excitation (1 m/s)
    V_base = 1.0  # Unit velocity excitation
    X_base = V_base / (1j * w)  # Convert velocity to displacement in frequency domain

    # Construct system matrix
    M = np.array([[Kd1 + Kd2, -Kd2],
                  [-Kd2, Kd2 - w**2 * m2]])

    # Right-hand side force vector due to base velocity excitation
    F_input = -(1j * w * c1 + k1) * X_base  
    RHS = np.array([F_input, 0])

    # Solve for displacements X1 and X2
    X1, X2 = np.linalg.solve(M, RHS)

    return X1, X2

# Function to compute vertical load response
def vertical_load(X1, w, c1, k1):
    """Computes vertical force response at lower suspension"""
    return (k1 + 1j * w * c1) * (X1 - (1 / (1j * w)))

# Recompute the original 2DOF system response for comparison
X1_vals, X2_vals = np.zeros_like(w_vals, dtype=complex), np.zeros_like(w_vals, dtype=complex)

for i, w in enumerate(w_vals):
    X1_vals[i], X2_vals[i] = compute_2DOF_response(m1, c1, k1, m2, c2, k2, w)

# Compute acceleration responses for the original system
A1_vals = -w_vals**2 * X1_vals  # Lower mass acceleration
A2_vals = -w_vals**2 * X2_vals  # Upper mass acceleration

# Compute vertical load response for the original system
F_lower_vals = vertical_load(X1_vals, w_vals, c1, k1)

# Extract magnitudes for the original system
A1_mag, A2_mag = np.abs(A1_vals), np.abs(A2_vals)
F_mag = np.abs(F_lower_vals)

# Define TMD parameters (tuned for 10 Hz suppression)
m3 = 0.5  # Additional mass [kg]
f_tmd = 10.0  # Target suppression frequency [Hz]
k3 = (2 * np.pi * f_tmd) ** 2 * m3  # Tuned stiffness
c3 = 5.0  # Damping for TMD [Ns/m]

# Function to compute 3DOF response with TMD
def compute_3DOF_response(m1, c1, k1, m2, c2, k2, m3, c3, k3, w):
    """Solves for X1, X2, and X3 given a base velocity excitation with a Tuned Mass Damper (TMD)."""
    
    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)
    Kd3 = dynamic_stiffness(m3, c3, k3, w)
    
    # Define base velocity excitation (1 m/s)
    V_base = 1.0  # Unit velocity excitation
    X_base = V_base / (1j * w)  # Convert velocity to displacement in frequency domain

    # Construct system matrix for 3-DOF system
    M = np.array([
        [Kd1 + Kd2, -Kd2, 0],
        [-Kd2, Kd2 + Kd3, -Kd3],
        [0, -Kd3, Kd3]
    ])

    # Right-hand side force vector due to base velocity excitation
    F_input = -(1j * w * c1 + k1) * X_base  
    RHS = np.array([F_input, 0, 0])

    # Solve for displacements X1, X2, and X3
    X1, X2, X3 = np.linalg.solve(M, RHS)

    return X1, X2, X3

# Compute response for each frequency with TMD
X1_vals_TMD, X2_vals_TMD, X3_vals_TMD = np.zeros_like(w_vals, dtype=complex), np.zeros_like(w_vals, dtype=complex), np.zeros_like(w_vals, dtype=complex)

for i, w in enumerate(w_vals):
    X1_vals_TMD[i], X2_vals_TMD[i], X3_vals_TMD[i] = compute_3DOF_response(m1, c1, k1, m2, c2, k2, m3, c3, k3, w)

# Compute acceleration responses with TMD
A1_vals_TMD = -w_vals**2 * X1_vals_TMD  # Lower mass acceleration
A2_vals_TMD = -w_vals**2 * X2_vals_TMD  # Upper mass acceleration
A3_vals_TMD = -w_vals**2 * X3_vals_TMD  # TMD acceleration

# Compute vertical load response with TMD
F_lower_vals_TMD = vertical_load(X1_vals_TMD, w_vals, c1, k1)

# Extract magnitudes
A1_mag_TMD, A2_mag_TMD, A3_mag_TMD = np.abs(A1_vals_TMD), np.abs(A2_vals_TMD), np.abs(A3_vals_TMD)
F_mag_TMD = np.abs(F_lower_vals_TMD)

# Plot Mass 1 Acceleration with and without TMD
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(f_vals, A1_mag, 'b', lw=2, label="Without TMD")
ax1.plot(f_vals, A1_mag_TMD, 'r', lw=2, linestyle="dashed", label="With TMD")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Acceleration |A1(ω)| [m/s²]")
ax1.set_title("Mass 1 Acceleration - Effect of TMD")
ax1.legend()
ax1.grid()
plt.show()


"""
VARIOUS OPTIONS OF TMD

"""

# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt

# Re-define system parameters
m1 = 1.0   # Lower mass [kg]
c1 = 10.0  # Lower damping [Ns/m]
k1 = 2000.0  # Lower stiffness [N/m]

m2 = 1.0   # Upper mass [kg]
c2 = 1.0   # Upper damping [Ns/m]
k2 = 4000.0  # Upper stiffness [N/m]

# Frequency range (0 to 35 Hz)
f_min = 0.1   # Avoid division by zero
f_max = 35.0
num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s

# Function to compute dynamic stiffness
def dynamic_stiffness(m, c, k, w):
    """Computes the dynamic stiffness: K_d(w) = k - w^2 * m + j * w * c"""
    return k - (w**2) * m + 1j * w * c

# Function to compute 2DOF response
def compute_2DOF_response(m1, c1, k1, m2, c2, k2, w):
    """Solves for X1 and X2 given a base velocity excitation"""
    
    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)
    
    # Define base velocity excitation (1 m/s)
    V_base = 1.0  # Unit velocity excitation
    X_base = V_base / (1j * w)  # Convert velocity to displacement in frequency domain

    # Construct system matrix
    M = np.array([[Kd1 + Kd2, -Kd2],
                  [-Kd2, Kd2 - w**2 * m2]])

    # Right-hand side force vector due to base velocity excitation
    F_input = -(1j * w * c1 + k1) * X_base  
    RHS = np.array([F_input, 0])

    # Solve for displacements X1 and X2
    X1, X2 = np.linalg.solve(M, RHS)

    return X1, X2

# Define TMD parameters (tuned for 10 Hz suppression)
m3 = 0.5  # Additional mass [kg]
f_tmd = 10.0  # Target suppression frequency [Hz]
k3 = (2 * np.pi * f_tmd) ** 2 * m3  # Tuned stiffness
c3 = 5.0  # Damping for TMD [Ns/m]

# Function to compute 3DOF response with TMD
def compute_3DOF_response(m1, c1, k1, m2, c2, k2, m3, c3, k3, w):
    """Solves for X1, X2, and X3 given a base velocity excitation with a Tuned Mass Damper (TMD)."""
    
    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)
    Kd3 = dynamic_stiffness(m3, c3, k3, w)
    
    # Define base velocity excitation (1 m/s)
    V_base = 1.0  # Unit velocity excitation
    X_base = V_base / (1j * w)  # Convert velocity to displacement in frequency domain

    # Construct system matrix for 3-DOF system
    M = np.array([
        [Kd1 + Kd2, -Kd2, 0],
        [-Kd2, Kd2 + Kd3, -Kd3],
        [0, -Kd3, Kd3]
    ])

    # Right-hand side force vector due to base velocity excitation
    F_input = -(1j * w * c1 + k1) * X_base  
    RHS = np.array([F_input, 0, 0])

    # Solve for displacements X1, X2, and X3
    X1, X2, X3 = np.linalg.solve(M, RHS)

    return X1, X2, X3

# Compute complex power in the frequency domain with TMD

# Initialize storage for power values with TMD
P_damping1_TMD = np.zeros_like(w_vals, dtype=complex)  # Damping power for mass 1
P_damping2_TMD = np.zeros_like(w_vals, dtype=complex)  # Damping power for mass 2
P_damping3_TMD = np.zeros_like(w_vals, dtype=complex)  # Damping power for TMD

P_spring1_TMD = np.zeros_like(w_vals, dtype=complex)   # Spring power for spring 1
P_spring2_TMD = np.zeros_like(w_vals, dtype=complex)   # Spring power for spring 2
P_spring3_TMD = np.zeros_like(w_vals, dtype=complex)   # Spring power for TMD

# Compute power for each frequency
for i, w in enumerate(w_vals):
    X1, X2, X3 = compute_3DOF_response(m1, c1, k1, m2, c2, k2, m3, c3, k3, w)
    
    # Compute velocity phasors (V = jωX)
    V1 = 1j * w * X1
    V2 = 1j * w * X2
    V3 = 1j * w * X3

    # Compute power dissipated by damping (Active Power)
    P_damping1_TMD[i] = c1 * V1 * np.conj(V1)  # Power dissipated in damping 1
    P_damping2_TMD[i] = c2 * V2 * np.conj(V2)  # Power dissipated in damping 2
    P_damping3_TMD[i] = c3 * V3 * np.conj(V3)  # Power dissipated in TMD damping

    # Compute power stored and exchanged in springs (Reactive Power)
    P_spring1_TMD[i] = k1 * X1 * np.conj(V1)  # Power in spring 1
    P_spring2_TMD[i] = k2 * (X2 - X1) * np.conj(V2 - V1)  # Power in spring 2
    P_spring3_TMD[i] = k3 * (X3 - X2) * np.conj(V3 - V2)  # Power in TMD spring

# Extract real and imaginary parts (Active & Reactive Power)
P_damping1_real_TMD, P_damping1_imag_TMD = np.real(P_damping1_TMD), np.imag(P_damping1_TMD)
P_damping2_real_TMD, P_damping2_imag_TMD = np.real(P_damping2_TMD), np.imag(P_damping2_TMD)
P_damping3_real_TMD, P_damping3_imag_TMD = np.real(P_damping3_TMD), np.imag(P_damping3_TMD)

P_spring1_real_TMD, P_spring1_imag_TMD = np.real(P_spring1_TMD), np.imag(P_spring1_TMD)
P_spring2_real_TMD, P_spring2_imag_TMD = np.real(P_spring2_TMD), np.imag(P_spring2_TMD)
P_spring3_real_TMD, P_spring3_imag_TMD = np.real(P_spring3_TMD), np.imag(P_spring3_TMD)

# Plot Active Power Dissipation
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(f_vals, P_damping1_real_TMD, 'b', lw=2, label="Damping 1 Power (TMD)")
ax1.plot(f_vals, P_damping2_real_TMD, 'r', lw=2, label="Damping 2 Power (TMD)")
ax1.plot(f_vals, P_damping3_real_TMD, 'g', lw=2, label="TMD Damping Power")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Active Power (W)")
ax1.set_title("Active Power Dissipated with TMD")
ax1.legend()
ax1.grid()
plt.show()

# Plot Reactive Power (Energy Exchange in Springs) with and without TMD
fig, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(f_vals, P_spring1_imag_TMD, 'b', lw=2, linestyle="dashed", label="Reactive Power - Spring 1 (TMD)")
ax2.plot(f_vals, P_spring2_imag_TMD, 'r', lw=2, linestyle="dashed", label="Reactive Power - Spring 2 (TMD)")
ax2.plot(f_vals, P_spring3_imag_TMD, 'g', lw=2, linestyle="dashed", label="Reactive Power - TMD Spring")

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Reactive Power (VAR)")
ax2.set_title("Reactive Power Stored in Springs - Effect of TMD")
ax2.legend()
ax2.grid()
plt.show()