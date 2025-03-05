import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##### FULL VEHICLE MODEL (4DOF) #####
# System Parameters
m_chassis = 600  # Sprung mass (kg)
m_wheel_f, m_wheel_r = 50, 50  # Unsprung masses (kg)
k_s_f, k_s_r = 20000, 20000  # Suspension stiffness (N/m)
k_t_f, k_t_r = 150000, 150000  # Tire stiffness (N/m)
c_s_f, c_s_r = 1500, 1500  # Suspension damping (Ns/m)
I_pitch = 250  # Pitch moment of inertia (kg·m²)
L_f, L_r = 1.2, 1.2  # Distance from CoG to front/rear wheels (m)

# Initial Conditions
x0_vehicle = [0.05, 0.0, 0.05, 0.0, 0.02, 0.0, 0.02, 0.0]  
# [Chassis bounce, bounce velocity, Pitch angle, Pitch angular velocity, Wheel F disp, Wheel F vel, Wheel R disp, Wheel R vel]

def vehicle_dynamics(t, y, m_chassis, I_pitch, m_wheel_f, m_wheel_r, k_s_f, k_s_r, k_t_f, k_t_r, c_s_f, c_s_r, L_f, L_r):
    x_c, v_c, theta, omega, x_w_f, v_w_f, x_w_r, v_w_r = y
    dx_cdt, dthetadt = v_c, omega
    dv_cdt = (-k_s_f * (x_c + L_f * theta - x_w_f) - k_s_r * (x_c - L_r * theta - x_w_r) - c_s_f * (v_c + L_f * omega - v_w_f) - c_s_r * (v_c - L_r * omega - v_w_r)) / m_chassis
    domegadt = (-k_s_f * L_f * (x_c + L_f * theta - x_w_f) + k_s_r * L_r * (x_c - L_r * theta - x_w_r) - c_s_f * L_f * (v_c + L_f * omega - v_w_f) + c_s_r * L_r * (v_c - L_r * omega - v_w_r)) / I_pitch
    dx_w_fdt, dx_w_rdt = v_w_f, v_w_r
    dv_w_fdt = (k_s_f * (x_c + L_f * theta - x_w_f) + c_s_f * (v_c + L_f * omega - v_w_f) - k_t_f * x_w_f) / m_wheel_f
    dv_w_rdt = (k_s_r * (x_c - L_r * theta - x_w_r) + c_s_r * (v_c - L_r * omega - v_w_r) - k_t_r * x_w_r) / m_wheel_r
    return [dx_cdt, dv_cdt, dthetadt, domegadt, dx_w_fdt, dv_w_fdt, dx_w_rdt, dv_w_rdt]

# Solve ODEs
sol_vehicle = solve_ivp(vehicle_dynamics, (0, 3), x0_vehicle, t_eval=np.linspace(0, 3, 500), args=(m_chassis, I_pitch, m_wheel_f, m_wheel_r, k_s_f, k_s_r, k_t_f, k_t_r, c_s_f, c_s_r, L_f, L_r))

# Visualization for Vehicle Model
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
axs[0].plot(sol_vehicle.t, sol_vehicle.y[0], label="Chassis Bounce")
axs[0].plot(sol_vehicle.t, sol_vehicle.y[2], label="Pitch Angle")
axs[0].set_ylabel("Displacement / Angle")
axs[0].set_title("Chassis Bounce & Pitch Over Time")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(sol_vehicle.t, sol_vehicle.y[4], label="Front Wheel")
axs[1].plot(sol_vehicle.t, sol_vehicle.y[6], label="Rear Wheel")
axs[1].set_ylabel("Vertical Displacement")
axs[1].set_title("Wheel Vertical Displacement")
axs[1].legend()
axs[1].grid(True)

##### MULTI-CYLINDER ENGINE MODEL (4DOF) #####
# System Parameters
I1, I2, I3, I4 = 0.1, 0.08, 0.09, 0.07  # Rotational inertias (kg·m²)
k_t12, k_t23, k_t34 = 5000, 6000, 5500  # Torsional stiffness (N·m/rad)
c_t12, c_t23, c_t34 = 50, 40, 45  # Torsional damping (N·m·s/rad)

# Initial Conditions
theta0_engine = [0.02, 0.0, -0.015, 0.0, 0.01, 0.0, -0.01, 0.0]  

def engine_dynamics(t, y, I1, I2, I3, I4, k_t12, k_t23, k_t34, c_t12, c_t23, c_t34):
    theta1, omega1, theta2, omega2, theta3, omega3, theta4, omega4 = y
    dtheta1dt, dtheta2dt, dtheta3dt, dtheta4dt = omega1, omega2, omega3, omega4
    domega1dt = (-k_t12 * (theta1 - theta2) - c_t12 * (omega1 - omega2)) / I1
    domega2dt = (k_t12 * (theta1 - theta2) - k_t23 * (theta2 - theta3) + c_t12 * (omega1 - omega2) - c_t23 * (omega2 - omega3)) / I2
    domega3dt = (k_t23 * (theta2 - theta3) - k_t34 * (theta3 - theta4) + c_t23 * (omega2 - omega3) - c_t34 * (omega3 - omega4)) / I3
    domega4dt = (k_t34 * (theta3 - theta4) + c_t34 * (omega3 - omega4)) / I4
    return [dtheta1dt, domega1dt, dtheta2dt, domega2dt, dtheta3dt, domega3dt, dtheta4dt, domega4dt]

sol_engine = solve_ivp(engine_dynamics, (0, 2), theta0_engine, t_eval=np.linspace(0, 2, 500), args=(I1, I2, I3, I4, k_t12, k_t23, k_t34, c_t12, c_t23, c_t34))
