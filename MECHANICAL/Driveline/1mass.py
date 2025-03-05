import numpy as np
import matplotlib.pyplot as plt

def dynamic_stiffness(m, c, k, w):
    """ Computes the dynamic stiffness for a single-DOF system: 
        K_d(w) = k - w^2 * m + j * w * c
    """
    return k - (w**2) * m + 1j * w * c

def transfer_function_base_excitation(m, c, k, w):
    """ Computes the displacement response to base velocity excitation:
        H(w) = jω (c + k / (jω)) / K_d(w)
    """
    K_d = dynamic_stiffness(m, c, k, w)
    numerator = 1j * w * (c + k / (1j * w))  # jω (c + k/jω)
    return numerator / K_d

def mass_acceleration(H_x, w):
    """ Computes acceleration response: A(w) = -ω² * X(w) """
    return -w**2 * H_x

def vertical_load(H_x, w, c, k):
    """ Computes vertical force response: 
        F(w) = (k + jωc) * (X(w) - V_base / jω)
    """
    return (k + 1j * w * c) * (H_x - (1 / (1j * w)))

# System parameters
m = 1.0      # Mass [kg]
c = 0.2      # Damping coefficient [N s/m]
k = 50.0     # Stiffness [N/m]

# Frequency range
w_min = 0.1   # Avoid division by zero
w_max = 20.0
num_points = 1000
w_vals = np.linspace(w_min, w_max, num_points)

# Compute frequency response function
H_x_vals = np.array([transfer_function_base_excitation(m, c, k, w) for w in w_vals])
H_a_vals = mass_acceleration(H_x_vals, w_vals)  # Acceleration response
H_f_vals = vertical_load(H_x_vals, w_vals, c, k)  # Vertical force response

# Extract magnitude and phase
A_mag = np.abs(H_a_vals)  # Acceleration magnitude
A_phase = np.angle(H_a_vals)  # Acceleration phase

F_mag = np.abs(H_f_vals)  # Force magnitude
F_phase = np.angle(H_f_vals)  # Force phase

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# Acceleration plot
ax1.plot(w_vals, A_mag, 'b', lw=2)
ax1.set_ylabel('Mass Acceleration |A(ω)| [m/s²]')
ax1.set_title('Mass Acceleration and Vertical Load Response')

# Force plot
ax2.plot(w_vals, F_mag, 'r', lw=2)
ax2.set_xlabel('Frequency (rad/s)')
ax2.set_ylabel('Vertical Load |F(ω)| [N]')

plt.tight_layout()
plt.show()
