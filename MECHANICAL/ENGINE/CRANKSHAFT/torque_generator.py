import numpy as np
import matplotlib.pyplot as plt

# Define crank angle range (0-720°)
angles = np.linspace(0, 720, 1000)  # Crank angles in degrees

# Define Fourier-based torque synthesis function
def generate_fourier_torque(crank_angle, coeffs, freqs, phases):
    """
    Generate a synthetic torque curve using a sum of sinusoids (Fourier series approach).
    
    Parameters:
        crank_angle (array): Crank angles in degrees
        coeffs (list): Amplitudes of the sinusoidal components
        freqs (list): Frequency multipliers (harmonics)
        phases (list): Phase shifts in degrees

    Returns:
        array: Simulated torque values in Nm
    """
    torque = np.zeros_like(crank_angle, dtype=float)
    
    for amp, freq, phase in zip(coeffs, freqs, phases):
        torque += amp * np.sin(np.radians(freq * crank_angle + phase))
    
    return torque

# Define harmonic components (customizable)
harmonic_coeffs = [50, 30, 15, 10, 5]  # Amplitudes (Nm)
harmonic_freqs = [1, 2, 3, 4, 5]  # Frequency multipliers (1x, 2x, 3x of base frequency)
harmonic_phases = [0, 45, 90, 135, 180]  # Phase shifts in degrees

# Generate the artificial torque curve
synthetic_torque = generate_fourier_torque(angles, harmonic_coeffs, harmonic_freqs, harmonic_phases)

# Plot the synthetic torque signal
plt.figure(figsize=(10, 5))
plt.plot(angles, synthetic_torque, label="Artificial Torque Signal", linewidth=2, color='g')
plt.xlabel("Crank Angle (deg)")
plt.ylabel("Torque (Nm)")
plt.title("Artificially Generated Torque Signal (Sinusoidal)")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid()
plt.show()