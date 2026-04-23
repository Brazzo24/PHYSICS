import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Motorcycle Parameters (Simplified approximations) ---
m = 250.0        # Mass of bike + rider (kg)
V = 20.0         # Constant forward speed (m/s) -> ~72 km/h
L = 1.45         # Wheelbase (m)
a = 0.7          # Distance from CoG to front axle (m)
b = L - a        # Distance from CoG to rear axle (m)
h = 0.6          # Height of CoG (m)
I_z = 30.0       # Yaw moment of inertia (kg*m^2)
I_x = 15.0       # Roll moment of inertia (kg*m^2)
g = 9.81         # Gravity (m/s^2)

# Tire stiffness parameters (Linearized)
C_af = 15000.0   # Front cornering stiffness (N/rad)
C_ar = 18000.0   # Rear cornering stiffness (N/rad)
C_gf = 1200.0    # Front camber stiffness (N/rad)
C_gr = 1500.0    # Rear camber stiffness (N/rad)

# --- Chicane Steering Input ---
def steering_input(t):
    # A sine wave representing a quick left-right chicane
    # Steering starts at t=1, lasts for 2 seconds
    if 1.0 <= t <= 3.0:
        # Note: To lean left, a rider briefly steers right (counter-steering).
        # This sine wave represents the steering angle delta.
        return 0.02 * np.sin(np.pi * (t - 1.0)) 
    return 0.0

# --- State-Space Equations of Motion ---
def motorcycle_dynamics(t, state):
    # Unpack state vector
    v_y, r, phi, p = state  # Lateral velocity, Yaw rate, Roll angle, Roll rate
    
    delta = steering_input(t)
    
    # Calculate slip angles (small angle approximation)
    alpha_f = delta - ((v_y + a * r) / V)
    alpha_r = -((v_y - b * r) / V)
    
    # Calculate lateral tire forces (Slip + Camber thrust)
    F_yf = C_af * alpha_f + C_gf * phi
    F_yr = C_ar * alpha_r + C_gr * phi
    
    # 1. Lateral acceleration equation: m(v_y_dot + V*r) = F_yf + F_yr
    v_y_dot = (F_yf + F_yr) / m - V * r
    
    # 2. Yaw acceleration equation: I_z * r_dot = a*F_yf - b*F_yr
    r_dot = (a * F_yf - b * F_yr) / I_z
    
    # 3. Roll acceleration equation (Simplified inverted pendulum)
    # I_x * p_dot = m*g*h*phi - (F_yf + F_yr)*h
    # Note: A real motorcycle requires steer torque modeling for accurate roll,
    # but this captures the basic pendulum physics reacting to lateral force.
    p_dot = (m * g * h * phi - (F_yf + F_yr) * h) / I_x
    
    return [v_y_dot, r_dot, p, p_dot]

# --- Simulation Setup ---
t_span = (0, 5)            # Simulate for 5 seconds
t_eval = np.linspace(0, 5, 500)
initial_state = [0.0, 0.0, 0.0, 0.0]  # Going straight initially

# Run the integration
solution = solve_ivp(motorcycle_dynamics, t_span, initial_state, t_eval=t_eval, method='RK45')

# Extract results
time = solution.t
v_y = solution.y[0]
yaw_rate = solution.y[1]
roll_angle = solution.y[2]
roll_rate = solution.y[3]

# Calculate Tire Loads over time
# Static weight distribution
F_zf_static = m * g * (b / L)
F_zr_static = m * g * (a / L)

# Dynamic lateral forces (recalculated for plotting)
delta_array = np.array([steering_input(t) for t in time])
alpha_f = delta_array - ((v_y + a * yaw_rate) / V)
alpha_r = -((v_y - b * yaw_rate) / V)
F_yf = C_af * alpha_f + C_gf * roll_angle
F_yr = C_ar * alpha_r + C_gr * roll_angle

# --- Plotting the Results ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# 1. Roll and Yaw Rates
axs[0].plot(time, np.degrees(roll_rate), label='Roll Rate (deg/s)', color='blue')
axs[0].plot(time, np.degrees(yaw_rate), label='Yaw Rate (deg/s)', color='red')
axs[0].set_title('Instantaneous Roll Rate & Yaw Rate during Chicane')
axs[0].set_ylabel('Rate (deg/s)')
axs[0].legend()
axs[0].grid(True)

# 2. Roll Angle
axs[1].plot(time, np.degrees(roll_angle), label='Roll Angle (deg)', color='purple')
axs[1].plot(time, np.degrees(delta_array), label='Steering Input (deg)', color='orange', linestyle='--')
axs[1].set_title('Motorcycle Lean (Roll) Angle & Steering Input')
axs[1].set_ylabel('Angle (deg)')
axs[1].legend()
axs[1].grid(True)

# 3. Lateral Tire Forces
axs[2].plot(time, F_yf, label='Front Lateral Force (N)', color='green')
axs[2].plot(time, F_yr, label='Rear Lateral Force (N)', color='darkred')
axs[2].set_title('Instantaneous Lateral Tire Loads')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Force (N)')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Print static vertical loads for reference
print(f"Static Front Vertical Tire Load: {F_zf_static:.1f} N")
print(f"Static Rear Vertical Tire Load: {F_zr_static:.1f} N")