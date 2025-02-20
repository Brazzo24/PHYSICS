import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# System parameters (user-defined)
m1 = 1.0      # Primary mass (kg)
k1 = 1000.0   # Primary stiffness (N/m)
c1 = 2.0      # Primary damping (Ns/m)

m2 = 0.2      # Absorber mass (kg) (Increased for better peak visibility)
k2 = 200.0    # Absorber stiffness (N/m) (Adjusted for tuning)
c2 = 1.0      # Absorber damping (Ns/m) (Reduced to maintain sharp peaks)

# Natural frequencies
omega_n1 = np.sqrt(k1 / m1)
omega_n2 = np.sqrt(k2 / m2)

# Frequency range as a ratio of primary natural frequency
r_values = np.linspace(0.4, 2.5, 500)
frequencies = r_values * omega_n1

# Arrays to store results
X1_vals = []  # Amplitude ratio of primary system
X2_vals = []  # Amplitude ratio of absorber
KE_vals = []  # Kinetic energy stored
PE_vals = []  # Potential energy stored
Dissipation_vals = []  # Energy dissipated

# Loop over frequency range
for omega in frequencies:
    # Dynamic stiffness matrix
    A = np.array([[k1 - m1 * omega**2 + 1j * c1 * omega, -k2 + 1j * c2 * omega],
                  [-k2 + 1j * c2 * omega, k2 - m2 * omega**2 + 1j * c2 * omega]])
    
    F = np.array([1.0, 0.0])  # Force applied to primary mass only
    X = solve(A, F)  # Solve for displacements
    
    X1, X2 = X[0], X[1]
    
    # Compute energy terms
    KE1 = 0.5 * m1 * (omega**2) * np.abs(X1)**2
    KE2 = 0.5 * m2 * (omega**2) * np.abs(X2)**2
    PE1 = 0.5 * k1 * np.abs(X1)**2
    PE2 = 0.5 * k2 * np.abs(X2)**2
    Dissipation = 0.5 * (c1 * omega * np.abs(X1)**2 + c2 * omega * np.abs(X2)**2)
    
    # Store values
    X1_vals.append(np.abs(X1))
    X2_vals.append(np.abs(X2))
    KE_vals.append(KE1 + KE2)
    PE_vals.append(PE1 + PE2)
    Dissipation_vals.append(Dissipation)

# Convert lists to arrays
X1_vals = np.array(X1_vals)
X2_vals = np.array(X2_vals)
KE_vals = np.array(KE_vals)
PE_vals = np.array(PE_vals)
Dissipation_vals = np.array(Dissipation_vals)

# Plot Amplitude Ratio
plt.figure(figsize=(10, 5))
plt.plot(r_values, X1_vals, label='Primary System Amplitude Ratio')
plt.plot(r_values, X2_vals, label='Absorber Amplitude Ratio', linestyle='dashed')
plt.axvline(omega_n2 / omega_n1, color='red', linestyle='dotted', label='Absorber Natural Frequency')
plt.xlabel('Frequency Ratio (ω/ω_n1)')
plt.ylabel('Amplitude Ratio')
plt.title('Frequency Response of Damped Vibration Absorber')
plt.legend()
plt.grid()
plt.show()

# Plot Energy Analysis
plt.figure(figsize=(10, 5))
plt.plot(r_values, KE_vals, label='Kinetic Energy')
plt.plot(r_values, PE_vals, label='Potential Energy')
plt.plot(r_values, Dissipation_vals, label='Dissipated Energy', linestyle='dashed')
plt.xlabel('Frequency Ratio (ω/ω_n1)')
plt.ylabel('Energy')
plt.title('Energy Analysis of Damped Vibration Absorber')
plt.legend()
plt.grid()
plt.show()
