import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Engine parameters (in meters)
bore = 81e-3             # Bore diameter in meters
stroke = 41e-3           # Stroke in meters
crank_radius = 24.25e-3  # Crank radius in meters
conrod_length = 90.1e-3  # Connecting rod length in meters
piston_area = np.pi * (bore / 2)**2  # Piston area in m²

# Mass properties
piston_mass = 0.3  # Approximate piston mass in kg
conrod_mass = 0.5  # Approximate connecting rod mass in kg
rotating_mass_ratio = 0.3  # Percentage of conrod mass considered rotating

# Engine speed (RPM)
rpm = 8000
omega = (2 * np.pi * rpm) / 60  # Convert RPM to angular velocity (rad/s)

# Define crank angle range (0-720°)
angles = np.linspace(0, 720, 1000)  # Crank angles in degrees

# Dummy pressure curve (simulate pressure vs. crank angle)
def generate_dummy_pressure(crank_angle):
    """Simulate a pressure curve based on a sum of sinusoids."""
    return np.maximum(0, np.sin(np.radians(crank_angle))**2) * 10  # Pressure in bar

# Create a DataFrame to simulate gas pressure input from a CSV
df = pd.DataFrame({"Crank Angle (deg)": angles})
df["Pressure (bar)"] = generate_dummy_pressure(df["Crank Angle (deg)"])
df["Pressure (Pa)"] = df["Pressure (bar)"] * 1e5  # Convert bar to Pascals

# Define cylinder crank angles and offsets
cylinders = {
    1: 70,    # Cylinder 1 fires at 70° (crank pin offset added)
    4: 180,   # Cylinder 4 fires at 180°
    2: 360,   # Cylinder 2 fires at 360°
    3: 610    # Cylinder 3 fires at 610° (540° + 70° offset)
}
offsets = {
    1: 0,     # Cylinder 1 is at 0° offset
    2: 0,     # Cylinder 2 is at 0° offset
    3: 90,    # Cylinder 3 is 90° offset (bank angle)
    4: 90     # Cylinder 4 is 90° offset (bank angle)
}

# Function to compute inertial force
def compute_inertial_force(crank_angle, offset):
    """Computes the inertial force due to reciprocating and rotating masses."""
    theta_rad = np.radians(crank_angle - offset)  # Adjusted crank angle
    
    # Piston acceleration using crank-slider mechanism
    a_p = crank_radius * omega**2 * (np.cos(theta_rad) + (crank_radius / conrod_length) * np.cos(2 * theta_rad))
    
    # Inertial force
    force_inertia = piston_mass * a_p  # Reciprocating force
    
    return force_inertia

# Function to compute piston force and torque including inertia
def compute_piston_force_and_torque(crank_angle, pressure, offset):
    """Computes the piston force and torque with inertia for a given crank angle and offset."""
    theta_rad = np.radians(crank_angle - offset)  # Adjusted crank angle
    
    # Piston position using crank-slider equation
    x = crank_radius * (1 - np.cos(theta_rad)) + np.sqrt(conrod_length**2 - (crank_radius * np.sin(theta_rad))**2)
    
    # Connecting rod angle beta
    beta = np.arcsin((crank_radius * np.sin(theta_rad)) / conrod_length)
    
    # Force on the piston from gas pressure
    force_piston = pressure * piston_area  # F = P * A
    
    # Inertial force
    force_inertia = compute_inertial_force(crank_angle, offset)
    
    # Total force on the crankshaft (piston + inertia)
    force_crank = (force_piston - force_inertia) * np.cos(beta)
    
    # Torque on crankshaft
    torque = force_crank * crank_radius
    
    return torque

# Compute total torque by summing contributions from all cylinders
df["Torque (Nm)"] = 0  # Initialize torque column

for cyl, firing_angle in cylinders.items():
    df[f"Torque_Cyl{cyl}"] = compute_piston_force_and_torque(df["Crank Angle (deg)"] - firing_angle, df["Pressure (Pa)"], offsets[cyl])
    df["Torque (Nm)"] += df[f"Torque_Cyl{cyl}"]

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(df["Crank Angle (deg)"], df["Torque (Nm)"], label="Total Torque", linewidth=2)
for cyl in cylinders.keys():
    plt.plot(df["Crank Angle (deg)"], df[f"Torque_Cyl{cyl}"], linestyle="--", label=f"Torque - Cylinder {cyl}")

plt.xlabel("Crank Angle (deg)")
plt.ylabel("Torque (Nm)")
plt.title("V4 Engine Torque with 70° Crank Pin Offset")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid()
plt.show()