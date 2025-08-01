import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Example nonlinear stiffness (piecewise linear + hardening region)
def nonlinear_k(dtheta):
    k1, k2, theta_switch = 500, 2000, 0.05  # Nm/rad, Nm/rad, rad
    return np.where(np.abs(dtheta) < theta_switch,
                    k1 * dtheta,
                    np.sign(dtheta) * (k1 * theta_switch + k2 * (np.abs(dtheta) - theta_switch)))

# Parameters
J1 = 0.2  # kg.m^2
J2 = 0.1  # kg.m^2
c = 5     # Nm.s/rad

def DMF_ODE(t, y):
    theta1, omega1, theta2, omega2 = y
    dtheta = theta1 - theta2
    domega = omega1 - omega2
    # Input torque: e.g., harmonic
    Tin = 50 * np.sin(2 * np.pi * 5 * t)  # 5 Hz, 50 Nm amplitude
    Tout = 0  # Free
    k_val = nonlinear_k(dtheta)
    c_val = c * domega
    d2theta1 = (Tin - k_val - c_val) / J1
    d2theta2 = (k_val + c_val - Tout) / J2
    return [omega1, d2theta1, omega2, d2theta2]

# Initial conditions
y0 = [0, 0, 0, 0]

# Time vector
t_span = (0, 2)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

sol = solve_ivp(DMF_ODE, t_span, y0, t_eval=t_eval, method='RK45')

# Plotting
plt.figure(figsize=(10,6))
plt.plot(sol.t, sol.y[0], label='theta1 (engine)')
plt.plot(sol.t, sol.y[2], label='theta2 (gearbox)')
plt.plot(sol.t, sol.y[0] - sol.y[2], label='Twist (theta1 - theta2)')
plt.xlabel('Time [s]')
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid(True)
plt.title('Dual Mass Flywheel Simulation (Nonlinear Stiffness)')
plt.show()
