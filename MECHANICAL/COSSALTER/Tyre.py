import numpy as np
import matplotlib.pyplot as plt

# Pacejka parameters (example values)
B = 10.0    # Stiffness factor
C = 1.9     # Shape factor
D = 1.0     # Peak factor
E = 0.97    # Curvature factor
Sh = 0.0    # Horizontal shift
K = 0.1     # Camber stiffness

def pacejka_magic_formula(alpha, gamma):
    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    gamma_rad = np.deg2rad(gamma)
    
    # Effective slip angle
    alpha_eff = alpha_rad + Sh + K * gamma_rad
    
    # Calculate lateral force
    Fy = D * np.sin(C * np.arctan(B * alpha_eff - E * (B * alpha_eff - np.arctan(B * alpha_eff))))
    return Fy

# Slip angle range from -15 to 15 degrees
alpha_values = np.linspace(-15, 15, 300)
gamma_value = 5  # Camber angle in degrees

# Calculate lateral forces
Fy_values = [pacejka_magic_formula(alpha, gamma_value) for alpha in alpha_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, Fy_values)
plt.title('Pacejka Magic Formula - Lateral Force vs Slip Angle')
plt.xlabel('Slip Angle (degrees)')
plt.ylabel('Lateral Force (normalized)')
plt.grid(True)
plt.show()
