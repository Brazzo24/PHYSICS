import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd

# === 1. Define anchor points here ===
# Example: slip vs force
slip_points = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
force_points = np.array([-4000, -2500, 0, 2500, 4000])

# === 2. Interpolation ===
# Sort points just in case
sort_idx = np.argsort(slip_points)
slip_points = slip_points[sort_idx]
force_points = force_points[sort_idx]

# Create smooth spline
xnew = np.linspace(slip_points.min(), slip_points.max(), 200)
spline = make_interp_spline(slip_points, force_points, k=3)
ynew = spline(xnew)

# === 3. Plot for visual check ===
plt.plot(slip_points, force_points, 'ro', label='Anchor Points')
plt.plot(xnew, ynew, 'b-', label='Spline Curve')
plt.xlabel('Slip')
plt.ylabel('Force')
plt.title('Pacejka-like Interpolated Curve')
plt.legend()
plt.grid()
plt.show()

# === 4. Generate and export lookup table ===
n_lookup = 50  # You can change this for more/less resolution
slip_lookup = np.linspace(slip_points.min(), slip_points.max(), n_lookup)
force_lookup = spline(slip_lookup)
lookup_table = pd.DataFrame({'Slip': slip_lookup, 'Force': force_lookup})
lookup_table.to_csv('pacejka_lookup_table.csv', index=False)
print("Lookup table saved as 'pacejka_lookup_table.csv'.")
print(lookup_table.head())
