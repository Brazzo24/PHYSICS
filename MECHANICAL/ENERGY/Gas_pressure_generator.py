import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define crank angle range (0-720°)
angles = np.linspace(0, 720, 1000)  # Crank angles in degrees

# Dummy pressure curve function
def generate_dummy_pressure(crank_angle, peak_pressure=110, expansion_factor=0.5):
    """
    Generate a synthetic gas pressure curve for an engine.
    
    Parameters:
        crank_angle (array): Crank angles in degrees
        peak_pressure (float): Maximum pressure in bar (default 50 bar)
        expansion_factor (float): Shape control for expansion phase (default 0.8)
    
    Returns:
        array: Simulated pressure values in bar
    """
    # Simulated pressure using a Wiebe-function-inspired approach
    pressure = np.exp(-((crank_angle - 360) / (180 * expansion_factor))**2) * peak_pressure
    pressure[crank_angle < 340] *= 0.5  # Before combustion
    pressure[crank_angle > 400] *= 0.5  # Expansion phase
    
    return pressure

# Generate pressure curve
pressure_curve = generate_dummy_pressure(angles)

# Create DataFrame
df = pd.DataFrame({"Crank Angle (deg)": angles, "Pressure (bar)": pressure_curve})

# Save to CSV
csv_filename = "dummy_gas_pressure.csv"
df.to_csv(csv_filename, index=False)
print(f"Dummy gas pressure data saved as {csv_filename}")

# Plot the generated pressure curve
plt.figure(figsize=(10, 5))
plt.plot(angles, pressure_curve, label="Gas Pressure Curve", linewidth=2, color='r')
plt.xlabel("Crank Angle (deg)")
plt.ylabel("Pressure (bar)")
plt.title("Generated Gas Pressure Curve for Custom Engine")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid()
plt.show()