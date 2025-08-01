import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd

# 1. User clicks to define points
points = []

def onclick(event):
    if event.inaxes:
        points.append((event.xdata, event.ydata))
        plt.plot(event.xdata, event.ydata, 'ro')
        plt.draw()

fig, ax = plt.subplots()
ax.set_title("Click to define points. Close the window when done.")
ax.set_xlabel("Slip ratio / Slip angle")
ax.set_ylabel("Force (N)")

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)

if len(points) < 3:
    print("Please select at least 3 points for interpolation.")
else:
    # 2. Sort and interpolate
    points = sorted(points)
    x, y = zip(*points)
    x = np.array(x)
    y = np.array(y)
    xnew = np.linspace(min(x), max(x), 100)

    # Spline interpolation for smooth curve
    spline = make_interp_spline(x, y, k=3)
    ynew = spline(xnew)

    # 3. Plot the result
    plt.figure()
    plt.plot(x, y, 'ro', label='Defined Points')
    plt.plot(xnew, ynew, 'b-', label='Spline Curve')
    plt.title('Interpolated Pacejka-like Curve')
    plt.xlabel('Slip')
    plt.ylabel('Force')
    plt.legend()
    plt.show()

    # 4. Generate lookup table
    n_lookup = 50  # Number of points in lookup table
    x_lookup = np.linspace(min(x), max(x), n_lookup)
    y_lookup = spline(x_lookup)
    table = pd.DataFrame({'Slip': x_lookup, 'Force': y_lookup})

    # 5. Export as CSV
    table.to_csv('pacejka_lookup_table.csv', index=False)
    print("Lookup table saved as 'pacejka_lookup_table.csv'.")
    print(table.head())
