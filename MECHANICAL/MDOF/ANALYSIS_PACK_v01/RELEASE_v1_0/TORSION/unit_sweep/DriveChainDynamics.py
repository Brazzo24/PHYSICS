import math
import numpy as np
import matplotlib.pyplot as plt

# --- Input Parameters ---
# Geometry (in meters)
pivot = (0.0, 0.0)              # Swingarm pivot coordinates (x, y)
front_sprocket = (-0.5, 0.3)    # Front sprocket (chain) coordinates (x, y)
initial_axle = (0.5, -0.2)      # Rear axle coordinates at nominal suspension position
wheel_radius = 0.3             # Rear wheel radius
CoG_height = 0.55              # Center of gravity height (for weight transfer, if needed)

# Dynamic inputs
engine_torque = 100.0          # Engine drive torque (Nm)
gear_ratio = 3.0               # Overall gear ratio (unitless)
rear_sprocket_radius = 0.15    # Rear sprocket radius (m)

# Compute chain tension from engine torque and gearing
chain_tension = engine_torque * gear_ratio / rear_sprocket_radius  # in Newtons

# --- Static Force and Moment Calculation at nominal position ---
# Rear axle coordinates (use initial nominal position)
axle_x, axle_y = initial_axle
# Chain unit direction vector (from rear axle toward front sprocket)
dx = front_sprocket[0] - axle_x
dy = front_sprocket[1] - axle_y
chain_length = math.hypot(dx, dy)
ux = dx / chain_length
uy = dy / chain_length
# Chain force components
Fx = chain_tension * ux   # horizontal component of chain force
Fy = chain_tension * uy   # vertical component of chain force
# Moment about swingarm pivot due to chain force (r x F)
r_x = axle_x - pivot[0]
r_y = axle_y - pivot[1]
moment_pivot = r_x * Fy - r_y * Fx  # positive => tends to extend (lift) the suspension
# Vertical force at rear tire due to chain (positive = upward force on tire, reducing load)
vertical_force_tire = -Fy

# Output the computed values at nominal position
print("Chain tension:", round(chain_tension, 1), "N")
print("Chain force vector: Fx =", round(Fx,1), "N, Fy =", round(Fy,1), "N")
print("Moment about pivot from chain:", round(moment_pivot,1), "N·m")
print("Vertical force at rear tire due to chain:", round(vertical_force_tire,1), "N (positive=lift)")

# --- Effect of Suspension Travel (two sample positions) ---
# Define two extreme swingarm angles for comparison
low_angle_deg = 30.0   # swingarm angled 30° downward (extended)
high_angle_deg = 10.0  # swingarm 10° downward (nearly flat, compressed)
swingarm_length = math.hypot(initial_axle[0]-pivot[0], initial_axle[1]-pivot[1])  # pivot-to-axle distance
for angle_deg in [low_angle_deg, high_angle_deg]:
    ang = math.radians(angle_deg)
    # Compute axle position for this swingarm angle (pivot at origin)
    axle_x = swingarm_length * math.cos(ang)
    axle_y = - swingarm_length * math.sin(ang)
    # Recompute chain direction and forces at this position
    dx = front_sprocket[0] - axle_x
    dy = front_sprocket[1] - axle_y
    L = math.hypot(dx, dy)
    ux, uy = dx/L, dy/L
    Fx = chain_tension * ux
    Fy = chain_tension * uy
    moment = (axle_x - pivot[0])*Fy - (axle_y - pivot[1])*Fx
    vertical_force = -Fy
    state = "Extended" if angle_deg > high_angle_deg else "Compressed"
    print(f"\nSuspension {state} (swingarm angle {angle_deg}°):")
    print("  Chain angle from horizontal:", round(math.degrees(math.asin(uy)), 1), "°")
    print("  Chain vertical force on axle:", round(Fy,1), "N (positive=upward pull)")
    print("  Moment about pivot:", round(moment,1), "N·m")
    print("  Vertical force at tire from chain:", round(vertical_force,1), "N")

# --- Dynamic Simulation (oscillating chain tension) ---
# Time array (seconds)
t = np.linspace(0.0, 2.0, 201)        # simulate 2 seconds
# Engine torque oscillating (e.g., ±30 Nm around 100 Nm)
torque_mean = 100.0                  # Nm
torque_amp = 30.0                    # Nm amplitude
frequency = 1.0                      # Hz
omega = 2 * math.pi * frequency
# Compute chain tension over time from oscillating engine torque
chain_tension_time = (torque_mean + torque_amp * np.sin(omega * t)) * gear_ratio / rear_sprocket_radius
# (If simulating suspension oscillation, we could also vary axle_y or swingarm angle here)
# For simplicity, keep geometry fixed at initial_axle for force calculation
dx = front_sprocket[0] - initial_axle[0]
dy = front_sprocket[1] - initial_axle[1]
L = math.hypot(dx, dy)
ux, uy = dx/L, dy/L  # (constant because geometry fixed in this simulation)
vertical_force_time = - chain_tension_time * uy  # vertical chain force at tire over time

# Plot the results of the dynamic simulation
plt.figure(figsize=(6,4))
plt.subplot(2,1,1)
plt.plot(t, chain_tension_time, 'r')
plt.ylabel('Chain Tension (N)')
plt.title('Drive Torque Oscillation: Chain Tension and Rear Tire Force')
plt.subplot(2,1,2)
plt.plot(t, vertical_force_time, 'b')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Vertical Force on Tire (N)')
plt.tight_layout()
plt.show()
