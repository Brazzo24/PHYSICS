import numpy as np
import matplotlib.pyplot as plt

"""
INPUT
"""
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

"""
FUNCTION
"""

def dynamic_stiffness(m, c, k, w):
    """Computes the dynamic stiffness: K_d(w) = k - w^2 * m + j * w * c"""
    return k - (w**2) * m + 1j * w * c

def compute_2DOF_response(m1, c1, k1, m2, c2, k2, w):
    """Solves for X1 and X2 given a base velocity excitation"""
    
    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)
    
    # Define base velocity excitation (1 m/s)
    V_base = 1.0  # Unit velocity excitation
    X_base = V_base / (1j * w)  # Convert velocity to displacement in frequency domain

    # Construct corrected system matrix
    M = np.array([[Kd1 + Kd2, -Kd2],
                  [-Kd2, Kd2 - w**2 * m2]])  # FIX: Added massΩ term for mass 2

    # Right-hand side force vector due to base velocity excitation
    F_input = -(1j * w * c1 + k1) * X_base  # Corrected imaginary unit
    RHS = np.array([F_input, 0])

    # Solve for displacements X1 and X2
    X1, X2 = np.linalg.solve(M, RHS)

    return X1, X2

def vertical_load(X1, w, c1, k1):
    """Computes vertical force response at lower suspension"""
    return (k1 + 1j * w * c1) * (X1 - (1 / (1j * w)))


num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s

# Compute response for each frequency
X1_vals, X2_vals = np.zeros_like(w_vals, dtype=complex), np.zeros_like(w_vals, dtype=complex)
for i, w in enumerate(w_vals):
    X1_vals[i], X2_vals[i] = compute_2DOF_response(m1, c1, k1, m2, c2, k2, w)

# Compute acceleration responses
A1_vals = -w_vals**2 * X1_vals  # Lower mass acceleration
A2_vals = -w_vals**2 * X2_vals  # Upper mass acceleration

# Compute vertical load response
F_lower_vals = vertical_load(X1_vals, w_vals, c1, k1)

# Extract magnitudes and phases
A1_mag, A1_phase = np.abs(A1_vals), np.angle(A1_vals, deg=True)
A2_mag, A2_phase = np.abs(A2_vals), np.angle(A2_vals, deg=True)
F_mag, F_phase = np.abs(F_lower_vals), np.angle(F_lower_vals, deg=True)

# Function to plot magnitude and phase on dual axes with optional y-axis max limit
def plot_response(f_vals, mag, phase, mag_label, phase_label, title, mag_color='b', phase_color='g', mag_ylim_max=None):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(f_vals, mag, mag_color, lw=2, label=mag_label)
    ax2.plot(f_vals, phase, phase_color, lw=1.5, linestyle="dashed", label=phase_label)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel(mag_label, color=mag_color)
    ax2.set_ylabel(phase_label, color=phase_color)
    ax1.set_title(title)
    ax1.grid()
    
    if mag_ylim_max is not None:
        ax1.set_ylim(0, mag_ylim_max)
    
    plt.show()

# Plot responses
plot_response(f_vals, A1_mag, A1_phase, 'Acceleration |A1(ω)| [m/s²]', 'Phase ∠A1(ω) [degrees]', 'Mass 1 Acceleration and Phase')
plot_response(f_vals, A2_mag, A2_phase, 'Acceleration |A2(ω)| [m/s²]', 'Phase ∠A2(ω) [degrees]', 'Mass 2 Acceleration and Phase')
# plot_response(f_vals, F_mag, F_phase, 'Vertical Load |F(ω)| [N]', 'Phase ∠F(ω) [degrees]', 'Vertical Load and Phase', mag_color='r', phase_color='g', mag_ylim_max=max(F_mag)*1.1 if F_mag.size > 0 else None)
plot_response(f_vals, F_mag, F_phase, 'Vertical Load |F(ω)| [N]', 'Phase ∠F(ω) [degrees]', 'Vertical Load and Phase', mag_color='r', phase_color='g', mag_ylim_max=1000)


"""
STABILITY ANALYSIS

"""
# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt

# Re-define functions after execution state reset
def dynamic_stiffness(m, c, k, w):
    """Computes the dynamic stiffness: K_d(w) = k - w^2 * m + j * w * c"""
    return k - (w**2) * m + 1j * w * c

# Function to compute eigenvalues of the system matrix
def compute_eigenvalues(m1, c1, k1, m2, c2, k2, w):
    """Computes the eigenvalues of the system matrix to analyze stability."""

    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)

    # Construct system matrix
    M = np.array([[Kd1 + Kd2, -Kd2],
                  [-Kd2, Kd2 - w**2 * m2]])

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(M)
    
    return eigenvalues

# Frequency Range (0 to 35 Hz)
f_min = 0.1   # Avoid division by zero
f_max = 35.0
num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s

# Compute eigenvalues over frequency range
real_parts = np.zeros((2, num_points))  # Store real parts of eigenvalues

for i, w in enumerate(w_vals):
    eigs = compute_eigenvalues(m1, c1, k1, m2, c2, k2, w)
    real_parts[:, i] = np.real(eigs)  # Store real parts

# Plot real parts of eigenvalues to check stability
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, real_parts[0, :], 'b', lw=2, label="Real part of eigenvalue 1")
ax.plot(f_vals, real_parts[1, :], 'r', lw=2, label="Real part of eigenvalue 2")

ax.axhline(0, color='k', linestyle='dashed', linewidth=1)  # Stability threshold
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Real Part of Eigenvalues')
ax.set_title('Stability Analysis: Real Parts of Eigenvalues')
ax.legend()
ax.grid()

plt.show()

# Check if any real part becomes positive (unstable)
#unstable_frequencies = f_vals[np.any(real_parts > 0, axis=0)]
#if len(unstable_frequencies) > 0:
#    print(f"⚠️ Unstable frequencies detected at: {unstable_frequencies} Hz")
#else:
#    print("✅ System remains stable across all analyzed frequencies.")

"""
STABILITY PART 2：

    MODE BEHAVIOUR


"""
# Compute imaginary parts of eigenvalues over frequency range
imag_parts = np.zeros((2, num_points))  # Store imaginary parts of eigenvalues

for i, w in enumerate(w_vals):
    eigs = compute_eigenvalues(m1, c1, k1, m2, c2, k2, w)
    imag_parts[:, i] = np.imag(eigs)  # Store imaginary parts

# Plot imaginary parts of eigenvalues to observe mode behavior
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(f_vals, imag_parts[0, :], 'b', lw=2, label="Imaginary part of eigenvalue 1")
ax.plot(f_vals, imag_parts[1, :], 'r', lw=2, label="Imaginary part of eigenvalue 2")

ax.axhline(0, color='k', linestyle='dashed', linewidth=1)  # Reference line at zero
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Imaginary Part of Eigenvalues')
ax.set_title('Mode Behavior: Imaginary Parts of Eigenvalues')
ax.legend()
ax.grid()

plt.show()

"""
STABILITY PART 3:

    ALL IN ONE PLOT

"""

# Plot both real and imaginary parts of eigenvalues to compare stability and mode behavior
fig, ax1 = plt.subplots(figsize=(7, 5))

# Plot real parts (stability indicator)
ax1.plot(f_vals, real_parts[0, :], 'b', lw=2, label="Real part of eigenvalue 1")
ax1.plot(f_vals, real_parts[1, :], 'r', lw=2, label="Real part of eigenvalue 2")
ax1.axhline(0, color='k', linestyle='dashed', linewidth=1)  # Stability threshold

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Real Part of Eigenvalues', color='b')
ax1.set_title('Stability & Mode Behavior: Real & Imaginary Parts of Eigenvalues')
ax1.grid()

# Create secondary axis for imaginary parts
ax2 = ax1.twinx()
ax2.plot(f_vals, imag_parts[0, :], 'b', lw=2, linestyle="dashed", label="Imaginary part of eigenvalue 1")
ax2.plot(f_vals, imag_parts[1, :], 'r', lw=2, linestyle="dashed", label="Imaginary part of eigenvalue 2")
ax2.set_ylabel('Imaginary Part of Eigenvalues', color='r')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

"""
ENERGY DISTRIBUTION

"""

# Function to compute mode shapes (eigenvectors) at given frequencies
def compute_mode_shapes(m1, c1, k1, m2, c2, k2, w):
    """Computes the mode shapes (eigenvectors) of the system matrix."""
    
    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)

    # Construct system matrix
    M = np.array([[Kd1 + Kd2, -Kd2],
                  [-Kd2, Kd2 - w**2 * m2]])

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(M)
    
    return eigenvalues, eigenvectors

# Select key frequencies (resonance and instability regions)
key_frequencies = [2, 5, 10, 15, 25]  # Hz
key_w_values = [2 * np.pi * f for f in key_frequencies]  # Convert to rad/s

# Compute mode shapes at key frequencies
mode_shapes = {}
for f, w in zip(key_frequencies, key_w_values):
    eigvals, eigvecs = compute_mode_shapes(m1, c1, k1, m2, c2, k2, w)
    mode_shapes[f] = eigvecs


# Recompute kinetic energy per mass properly over the full frequency range

KE_mass1_ratios_fixed = []
KE_mass2_ratios_fixed = []

for i, w in enumerate(w_vals):
    eigvals, eigvecs = compute_mode_shapes(m1, c1, k1, m2, c2, k2, w)

    # Compute kinetic energy distribution per mass correctly using velocities
    v1 = 1j * w * eigvecs[0, 0]  # Velocity of Mass 1 in Mode 1
    v2 = 1j * w * eigvecs[1, 0]  # Velocity of Mass 2 in Mode 1

    KE_mass1 = 0.5 * m1 * np.abs(v1)**2  # KE of Mass 1
    KE_mass2 = 0.5 * m2 * np.abs(v2)**2  # KE of Mass 2

    KE_total = KE_mass1 + KE_mass2
    KE_mass1_ratios_fixed.append(KE_mass1 / KE_total)
    KE_mass2_ratios_fixed.append(KE_mass2 / KE_total)

KE_mass1_ratios_fixed = np.array(KE_mass1_ratios_fixed)
KE_mass2_ratios_fixed = np.array(KE_mass2_ratios_fixed)

# Plot corrected kinetic energy distribution per mass
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(f_vals, KE_mass1_ratios_fixed, 'b', lw=2, label="KE in Mass 1 (Fixed)")
ax1.plot(f_vals, KE_mass2_ratios_fixed, 'r', lw=2, label="KE in Mass 2 (Fixed)")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Kinetic Energy Ratio")
ax1.set_title("Corrected Kinetic Energy Distribution: Mass 1 vs. Mass 2")
ax1.legend()
ax1.grid()


# Recompute potential energy distribution per spring over the full frequency range

PE_spring1_ratios_fixed = []
PE_spring2_ratios_fixed = []

for i, w in enumerate(w_vals):
    eigvals, eigvecs = compute_mode_shapes(m1, c1, k1, m2, c2, k2, w)

    # Compute potential energy distribution per spring using mode shapes
    PE_spring1 = 0.5 * k1 * np.abs(eigvecs[0, 0])**2  # PE in Spring 1
    PE_spring2 = 0.5 * k2 * np.abs(eigvecs[1, 0] - eigvecs[0, 0])**2  # PE in Spring 2

    PE_total = PE_spring1 + PE_spring2
    PE_spring1_ratios_fixed.append(PE_spring1 / PE_total)
    PE_spring2_ratios_fixed.append(PE_spring2 / PE_total)

PE_spring1_ratios_fixed = np.array(PE_spring1_ratios_fixed)
PE_spring2_ratios_fixed = np.array(PE_spring2_ratios_fixed)

# Plot corrected potential energy distribution per spring
fig, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(f_vals, PE_spring1_ratios_fixed, 'b', lw=2, label="PE in Spring 1 (Fixed)")
ax2.plot(f_vals, PE_spring2_ratios_fixed, 'r', lw=2, label="PE in Spring 2 (Fixed)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Potential Energy Ratio")
ax2.set_title("Corrected Potential Energy Distribution: Spring 1 vs. Spring 2")
ax2.legend()
ax2.grid()
plt.show()

# Overlay kinetic and potential energy distributions for a full energy exchange visualization

fig, ax1 = plt.subplots(figsize=(7, 5))

# Plot kinetic energy ratios
ax1.plot(f_vals, KE_mass1_ratios_fixed, 'b', lw=2, linestyle='-', label="KE in Mass 1")
ax1.plot(f_vals, KE_mass2_ratios_fixed, 'r', lw=2, linestyle='-', label="KE in Mass 2")

ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Kinetic Energy Ratio")
ax1.set_title("Kinetic vs. Potential Energy Distribution")
ax1.legend(loc='upper left')
ax1.grid()

# Create secondary y-axis for potential energy
ax2 = ax1.twinx()

# Plot potential energy ratios
ax2.plot(f_vals, PE_spring1_ratios_fixed, 'b', lw=2, linestyle='dashed', label="PE in Spring 1")
ax2.plot(f_vals, PE_spring2_ratios_fixed, 'r', lw=2, linestyle='dashed', label="PE in Spring 2")
ax2.set_ylabel("Potential Energy Ratio")

# Add second legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

# Define a range of damping values to analyze stability improvement
c1_values = np.linspace(10, 200, 10)  # Increasing damping for Mass 1
c2_values = np.linspace(1, 50, 10)    # Increasing damping for Mass 2

# Store results for real parts of eigenvalues
real_parts_damping_effect = np.zeros((len(c1_values), len(w_vals), 2))

# Sweep through different damping values
for i, c1_test in enumerate(c1_values):
    for j, w in enumerate(w_vals):
        eigs = compute_eigenvalues(m1, c1_test, k1, m2, c2, k2, w)
        real_parts_damping_effect[i, j, :] = np.real(eigs)  # Store real parts

"""
POWER ANALYSIS:

    ACTIVE AND REACTIVE POWER

"""


# Recompute complex power in the frequency domain

# Initialize storage for power values
P_damping1 = np.zeros_like(w_vals, dtype=complex)  # Damping power for mass 1
P_damping2 = np.zeros_like(w_vals, dtype=complex)  # Damping power for mass 2
P_spring1 = np.zeros_like(w_vals, dtype=complex)   # Spring power for spring 1
P_spring2 = np.zeros_like(w_vals, dtype=complex)   # Spring power for spring 2

# Compute power for each frequency
for i, w in enumerate(w_vals):
    X1, X2 = compute_2DOF_response(m1, c1, k1, m2, c2, k2, w)
    
    # Compute velocity phasors (V = jωX)
    V1 = 1j * w * X1
    V2 = 1j * w * X2

    # Compute power dissipated by damping (Active Power)
    P_damping1[i] = c1 * V1 * np.conj(V1)  # Power dissipated in damping 1
    P_damping2[i] = c2 * V2 * np.conj(V2)  # Power dissipated in damping 2

    # Compute power stored and exchanged in springs (Reactive Power)
    P_spring1[i] = k1 * X1 * np.conj(V1)  # Power in spring 1
    P_spring2[i] = k2 * (X2 - X1) * np.conj(V2 - V1)  # Power in spring 2

# Extract real and imaginary parts (Active & Reactive Power)
P_damping1_real, P_damping1_imag = np.real(P_damping1), np.imag(P_damping1)
P_damping2_real, P_damping2_imag = np.real(P_damping2), np.imag(P_damping2)

P_spring1_real, P_spring1_imag = np.real(P_spring1), np.imag(P_spring1)
P_spring2_real, P_spring2_imag = np.real(P_spring2), np.imag(P_spring2)

# Plot Active Power (Damping Losses)
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(f_vals, P_damping1_real, 'b', lw=2, label="Active Power - Damping 1")
ax1.plot(f_vals, P_damping2_real, 'r', lw=2, label="Active Power - Damping 2")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("Active Power (W)")
ax1.set_title("Active Power Dissipated in Damping Elements")
ax1.legend()
ax1.grid()
plt.show()

# Plot Reactive Power (Energy Exchange in Springs)
fig, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(f_vals, P_spring1_imag, 'b', lw=2, linestyle="dashed", label="Reactive Power - Spring 1")
ax2.plot(f_vals, P_spring2_imag, 'r', lw=2, linestyle="dashed", label="Reactive Power - Spring 2")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Reactive Power (VAR)")
ax2.set_title("Reactive Power Stored in Springs")
ax2.legend()
ax2.grid()
plt.show()


"""
SUMMARY REPORT

"""

# Function to compute natural frequencies by solving det(M) = 0
def compute_natural_frequencies(m1, k1, m2, k2):
    """Computes the natural frequencies of the system"""
    M = np.array([[k1 + k2, -k2],
                  [-k2, k2]])
    masses = np.array([[m1, 0],
                       [0, m2]])
    
    eigvals = np.linalg.eigvals(np.linalg.inv(masses) @ M)
    freqs = np.sqrt(np.abs(eigvals)) / (2 * np.pi)  # Convert to Hz
    return np.sort(freqs)

# Compute natural frequencies
natural_frequencies = compute_natural_frequencies(m1, k1, m2, k2)

# Identify unstable regions based on eigenvalues
unstable_frequencies = f_vals[np.any(real_parts_damping_effect[-1, :, :] > 0, axis=1)]
stability_summary = "Fully Stable" if len(unstable_frequencies) == 0 else f"Unstable in {len(unstable_frequencies)} frequency points"

# Compute maximum and minimum power dissipation
max_P_damping1 = np.max(P_damping1_real)
min_P_damping1 = np.min(P_damping1_real)
max_P_damping2 = np.max(P_damping2_real)
min_P_damping2 = np.min(P_damping2_real)

max_P_spring1 = np.max(P_spring1_imag)
min_P_spring1 = np.min(P_spring1_imag)
max_P_spring2 = np.max(P_spring2_imag)
min_P_spring2 = np.min(P_spring2_imag)

# Print Summary Report
summary_report = f"""
===== SYSTEM SUMMARY REPORT =====
📌 Natural Frequencies (Hz): {natural_frequencies[0]:.2f}, {natural_frequencies[1]:.2f}

📌 Stability Summary: {stability_summary}
  - Unstable Frequencies: {unstable_frequencies[:10]} ... (truncated for readability)

📌 Maximum & Minimum Power Dissipation:
  - Damping 1: Max {max_P_damping1:.2f} W, Min {min_P_damping1:.2f} W
  - Damping 2: Max {max_P_damping2:.2f} W, Min {min_P_damping2:.2f} W

📌 Maximum & Minimum Reactive Power (Energy Exchange):
  - Spring 1: Max {max_P_spring1:.2f} VAR, Min {min_P_spring1:.2f} VAR
  - Spring 2: Max {max_P_spring2:.2f} VAR, Min {min_P_spring2:.2f} VAR

✅ Optimization Recommendations:
  - Increase damping to stabilize unstable frequencies.
  - Adjust stiffness to shift resonance peaks.
  - Consider a Tuned Mass Damper (TMD) for high-frequency resonance suppression.
  - Add a secondary damper to improve energy dissipation.

=================================
"""

print(summary_report)