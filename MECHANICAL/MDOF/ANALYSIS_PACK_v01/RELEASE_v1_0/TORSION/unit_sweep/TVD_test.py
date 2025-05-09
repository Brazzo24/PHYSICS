import numpy as np
import matplotlib.pyplot as plt

# System parameters (adjust these)
mu = 0.1     # Mass ratio m2/m1
f = 0.9      # Tuning ratio omega_2 / omega_a
zeta = 0.05  # Damping ratio of absorber

# Frequency ratio range g = omega / omega_a
g = np.linspace(0.5, 2.0, 500)

# Numerator for A1 and A2
numerator_A1 = (2 * f * zeta * g)**2 + (g**2 - f**2)**2
numerator_A2 = (2 * f * zeta * g)**2 + f**4

# Common denominator
term1 = (2 * f * zeta * g)**2 * g**2 * (1 + mu * g**2)
term2 = (mu * f**2 * g**2 - (g**2 - 1) * (g**2 - f**2))**2
denominator = np.sqrt(term1 + term2)

# Relative amplitude expressions
A1 = numerator_A1 / denominator
A2 = numerator_A2 / denominator

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(g, A1, label=r'$A_1(g)$ (equipment mass)')
plt.plot(g, A2, label=r'$A_2(g)$ (absorber mass)', linestyle='--')
plt.axvline(1.0, color='gray', linestyle=':', label=r'$g=1$ (resonance of primary)')
plt.axvline(f, color='orange', linestyle=':', label=r'$g=f$ (resonance of absorber)')
plt.xlabel(r'Frequency Ratio $g = \omega / \omega_a$')
plt.ylabel('Relative Amplitude')
plt.title('Amplitude Response of Tuned Torsional Vibration Damper')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Given mass ratio
mu = 0.05  # try with 0.1 or 0.2 too

# Optimal tuning ratio
f_opt = 1 / (1 + mu)

# Optimal damping ratio
zeta_opt = np.sqrt(3 * mu / (8 * (1 + mu)**3))

# Peak frequency location (both A and B)
g_AB = ((2 + mu) + np.sqrt((2 + mu) * mu)) / ((1 + mu) * (2 + mu))

# Amplitude at peaks A = B
A_AB = np.sqrt(1 + 2 / mu)

# Print results
print(f"Mass ratio µ:        {mu}")
print(f"Optimal f:           {f_opt:.4f}")
print(f"Optimal ζ:           {zeta_opt:.4f}")
print(f"Peak location g_AB:  {g_AB:.4f}")
print(f"Peak amplitude A_AB: {A_AB:.4f}")

# Plot amplitude curve for optimal values
g = np.linspace(0.5, 2.0, 500)

# Numerator and denominator for A1
num = (2 * f_opt * zeta_opt * g)**2 + (g**2 - f_opt**2)**2
den1 = (2 * f_opt * zeta_opt * g)**2 * g**2 * (1 + mu * g**2)
den2 = (mu * f_opt**2 * g**2 - (g**2 - 1)*(g**2 - f_opt**2))**2
A1 = num / np.sqrt(den1 + den2)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(g, A1, label='Relative amplitude $A_1(g)$')
plt.axhline(A_AB, linestyle='--', color='red', label=f'Peak Amplitude = {A_AB:.2f}')
plt.axvline(g_AB, linestyle=':', color='orange', label=f'Peak Location g = {g_AB:.2f}')
plt.axvline(1.0, linestyle=':', color='gray', label='Main system resonance (g=1)')
plt.xlabel('Frequency Ratio $g = \\omega / \\omega_a$')
plt.ylabel('Relative Amplitude $A_1(g)$')
plt.title('Optimal Tuned Torsional Vibration Damper (TVD) Response')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
