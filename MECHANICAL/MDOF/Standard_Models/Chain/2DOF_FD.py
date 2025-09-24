###############################################################################
# 3. Code Imports and Utilities
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Utility function: dynamic_stiffness
def dynamic_stiffness(m, c, k, w):
    """
    Computes the dynamic stiffness:
    K_d(w) = k - w^2*m + j*w*c
    """
    return k - (w**2)*m + 1j*w*c

"""
INPUT
"""
# Updated System Parameters
m1 = 1.0   # Lower mass [kg]
c1 = 10.0  # Lower damping [Ns/m]
k1 = 2000.0  # Lower stiffness [N/m]

m2 = 1.0   # Upper mass [kg]
c2 = 2.0   # Upper damping [Ns/m]
k2 = 12000.0  # Upper stiffness [N/m]

# Updated Frequency Range (0 to 35 Hz)
f_min = 0.1   # Avoid division by zero
f_max = 35.0

def compute_2DOF_response(m1, c1, k1, m2, c2, k2, w):
    """Solves for X1 and X2 given a base velocity excitation (1 m/s)."""
    
    # Dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)  # For mass 1
    Kd2 = dynamic_stiffness(m2, c2, k2, w)  # For mass 2
    
    # Base velocity excitation: V_base = 1.0 m/s
    # => base displacement in frequency domain: X_base = 1 / (j w)
    V_base = 1.0
    X_base = V_base / (1j * w)  # [m]
    
    # System matrix (2x2) for X1, X2
    # IMPORTANT: Avoid double subtracting w^2*m2 on the bottom-right entry
    M = np.array([
        [(k1 - w**2*m1 + 1j*w*c1) + (k2 + 1j*w*c2),   -(k2 + 1j*w*c2)],
        [-(k2 + 1j*w*c2),                            (k2 - w**2*m2 + 1j*w*c2)]
    ])
    
    # Right-hand side: from base motion
    # Force transmitted by the lower spring/damper is (k1 + j w c1)*(X1 - X_base).
    # We move that to the RHS => F_input = -(k1 + j w c1)*X_base
    F_input = -(k1 + 1j*w*c1) * X_base
    RHS = np.array([F_input, 0])
    
    # Solve for X1 and X2
    X1, X2 = np.linalg.solve(M, RHS)
    
    return X1, X2

def vertical_load(X1, w, c1, k1):
    """Computes vertical force response at lower suspension."""
    return (k1 + 1j * w * c1) * (X1 - (1 / (1j * w)))

# Frequency array
num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals  # Convert Hz to rad/s

# Arrays to store frequency responses
X1_vals = np.zeros_like(w_vals, dtype=complex)
X2_vals = np.zeros_like(w_vals, dtype=complex)

for i, w in enumerate(w_vals):
    X1_vals[i], X2_vals[i] = compute_2DOF_response(m1, c1, k1, m2, c2, k2, w)

###############################################################################
# 1. Acceleration of the upper mass (A2)
###############################################################################
# A2 = -w^2 * X2 (frequency domain relation)
A2_vals = -w_vals**2 * X2_vals

###############################################################################
# 2. Transmissibility (ratio of mass motion to base motion)
###############################################################################
# Base displacement amplitude = |X_base| = 1 / w
# Transmissibility T2 = |X2| / |X_base| = w * |X2|
# Similarly T1 = w * |X1|
X_base_mag = 1 / w_vals
X1_mag = np.abs(X1_vals)
X2_mag = np.abs(X2_vals)

T1 = X1_mag / X_base_mag  # or simply w_vals * X1_mag
T2 = X2_mag / X_base_mag  # or simply w_vals * X2_mag

###############################################################################
# 3. Force in the upper spring/damper
###############################################################################
# F_upper = (k2 + j w c2)*(X2 - X1)
F_upper_vals = (k2 + 1j*w_vals*c2) * (X2_vals - X1_vals)
F_upper_mag = np.abs(F_upper_vals)

###############################################################################
# (Optional) We can still compute the vertical load in the lower suspension:
###############################################################################
F_lower_vals = vertical_load(X1_vals, w_vals, c1, k1)
F_lower_mag = np.abs(F_lower_vals)

###############################################################################
# Plotting
###############################################################################
plt.figure(figsize=(10, 6))
plt.plot(f_vals, np.abs(A2_vals), label='|A2| (Upper Mass Acceleration)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Acceleration Magnitude [m/s^2]')
plt.title('Upper Mass Acceleration Response')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(f_vals, T1, label='T1 = |X1|/|X_base|')
plt.plot(f_vals, T2, label='T2 = |X2|/|X_base|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Transmissibility')
plt.title('Transmissibility of Mass 1 and Mass 2')
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(f_vals, F_upper_mag, label='|F_upper|')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Force Magnitude [N]')
plt.title('Force in Upper Spring/Damper')
plt.grid(True)
plt.legend()

# (Optional) If you want to plot the lower load as well:
plt.figure(figsize=(10, 6))
plt.plot(f_vals, F_lower_mag, label='|F_lower| (Lower Suspension Load)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Force Magnitude [N]')
plt.title('Force in Lower Spring/Damper')
plt.grid(True)
plt.legend()

plt.show()
