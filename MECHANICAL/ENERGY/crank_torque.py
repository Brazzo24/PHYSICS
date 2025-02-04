import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Engine parameters (in meters)
bore = 81e-3          # Bore diameter in meters
stroke = 41e-3        # Stroke in meters
crank_radius = 24.25e-3  # Crank radius in meters
conrod_length = 90.1e-3  # Connecting rod length in meters
piston_area = np.pi * (bore / 2)**2  # Piston area in m²

# Load Gas Pressure Curve (Crank Angle vs. Pressure)
# For now, we create a dummy dataset
angles = np.linspace(0, 720, 1000)  # Crank angles in degrees
dummy_pressure = np.maximum(0, np.sin(np.radians(angles))**2) * 10  # Dummy pressure in bar

# Create a DataFrame to simulate input from a CSV file
df = pd.DataFrame({"Crank Angle (deg)": angles, "Pressure (bar)": dummy_pressure})

# Convert pressure from bar to Pascals (1 bar = 1e5 Pa)
df["Pressure (Pa)"] = df["Pressure (bar)"] * 1e5

# Function to compute piston position and crank angle relationships
def compute_piston_force_and_torque(crank_angle, pressure):
    """Computes the piston force and torque on the crankshaft for a given crank angle and pressure."""
    theta_rad = np.radians(crank_angle)  # Convert crank angle to radians
    
    # Position of the piston using the crank-slider equation
    x = crank_radius * (1 - np.cos(theta_rad)) + np.sqrt(conrod_length**2 - (crank_radius * np.sin(theta_rad))**2)
    
    # Instantaneous connecting rod angle (beta)
    beta = np.arcsin((crank_radius * np.sin(theta_rad)) / conrod_length)
    
    # Force on the piston
    force_piston = pressure * piston_area  # F = P * A
    
    # Force on the crankshaft (projecting force via connecting rod angle)
    force_crank = force_piston * np.cos(beta)  # Effective force on crank
    
    # Torque on the crankshaft
    torque = force_crank * crank_radius
    
    return torque

# Compute torque for each crank angle
df["Torque (Nm)"] = compute_piston_force_and_torque(df["Crank Angle (deg)"], df["Pressure (Pa)"])

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(df["Crank Angle (deg)"], df["Torque (Nm)"], label="Torque")
plt.xlabel("Crank Angle (deg)")
plt.ylabel("Torque (Nm)")
plt.title("Torque Output Over 720° Engine Cycle")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid()
plt.show()