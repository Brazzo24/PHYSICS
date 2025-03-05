import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System Parameters for Balanced vs. Localized Energy
m1, m2 = 2.0, 1.5  # Masses (kg)
k1_balanced, k2_balanced = 1000, 800  # Soft Stiffness (Balanced Energy)
k1_local, k2_local = 5000, 4000  # Stiff Stiffness (Localized Energy Peaks)
c_balanced, c_local = 50, 10  # Damping (High for balanced, low for localized)

# Initial Conditions: Small Displacement and Zero Velocity
x0 = [0.05, 0.0, 0.05, 0.0]  # [x1, v1, x2, v2]

# Time Span
t_span = (0, 5)
t_eval = np.linspace(0, 5, 500)

# Equations of Motion for 2DOF System
def system_dynamics(t, y, m1, m2, k1, k2, c):
    x1, v1, x2, v2 = y
    dx1dt = v1
    dx2dt = v2
    
    dv1dt = (-k1 * x1 + k2 * (x2 - x1) - c * v1) / m1
    dv2dt = (-k2 * (x2 - x1) - c * v2) / m2

    return [dx1dt, dv1dt, dx2dt, dv2dt]

# Solve ODEs for Balanced Energy Case
sol_balanced = solve_ivp(system_dynamics, t_span, x0, t_eval=t_eval, args=(m1, m2, k1_balanced, k2_balanced, c_balanced))

# Solve ODEs for Localized Energy Peaks Case
sol_local = solve_ivp(system_dynamics, t_span, x0, t_eval=t_eval, args=(m1, m2, k1_local, k2_local, c_local))

# Compute Energy
def compute_energy(sol, m1, m2, k1, k2):
    x1, v1, x2, v2 = sol.y
    KE = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    PE = 0.5 * k1 * x1**2 + 0.5 * k2 * (x2 - x1)**2
    return KE, PE

KE_balanced, PE_balanced = compute_energy(sol_balanced, m1, m2, k1_balanced, k2_balanced)
KE_local, PE_local = compute_energy(sol_local, m1, m2, k1_local, k2_local)

# Plot Results
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot Displacement Comparison
axs[0].plot(sol_balanced.t, sol_balanced.y[0], label="Mass 1 - Balanced", linestyle="--")
axs[0].plot(sol_balanced.t, sol_balanced.y[2], label="Mass 2 - Balanced", linestyle=":")
axs[0].plot(sol_local.t, sol_local.y[0], label="Mass 1 - Localized", linestyle="-")
axs[0].plot(sol_local.t, sol_local.y[2], label="Mass 2 - Localized", linestyle="-.")
axs[0].set_ylabel("Displacement (m)")
axs[0].set_title("Displacement Over Time")
axs[0].legend()
axs[0].grid(True)

# Plot Energy Comparison
axs[1].plot(sol_balanced.t, KE_balanced, label="Kinetic Energy - Balanced", linestyle="--")
axs[1].plot(sol_balanced.t, PE_balanced, label="Potential Energy - Balanced", linestyle=":")
axs[1].plot(sol_local.t, KE_local, label="Kinetic Energy - Localized", linestyle="-")
axs[1].plot(sol_local.t, PE_local, label="Potential Energy - Localized", linestyle="-.")
axs[1].set_ylabel("Energy (J)")
axs[1].set_title("Energy Distribution Over Time")
axs[1].legend()
axs[1].grid(True)

# Plot Energy Release (Potential Energy Dissipation)
axs[2].plot(sol_balanced.t, PE_balanced - KE_balanced, label="Energy Release - Balanced", linestyle="--")
axs[2].plot(sol_local.t, PE_local - KE_local, label="Energy Release - Localized", linestyle="-")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Energy Release (J)")
axs[2].set_title("Energy Release Over Time")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()