import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set parameters
m = 1.0   # mass (kg)
k = 50.0  # spring constant (N/m)
c = 2.0   # damping coefficient (N·s/m)

# Times of interest
t_start = 0.0
t_end = 2.0
num_points = 1000
t_eval = np.linspace(t_start, t_end, num_points)