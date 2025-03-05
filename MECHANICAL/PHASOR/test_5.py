import cmath
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

class MDOFSystem:
    """
    Represents a multi-degree-of-freedom (MDOF) mechanical system.
    """
    def __init__(self, mass_matrix, damping_matrix, stiffness_matrix, force_magnitude, force_phase, frequency):
        self.M = np.array(mass_matrix, dtype=np.float64)
        self.C = np.array(damping_matrix, dtype=np.float64)
        self.K = np.array(stiffness_matrix, dtype=np.float64)
        self.F = cmath.rect(force_magnitude, np.radians(force_phase))  # Convert to rectangular form
        self.frequency = frequency
        self.eigenvalues, self.eigenvectors = scipy.linalg.eig(self.K, self.M)
        self.natural_frequencies = np.sqrt(np.real(self.eigenvalues))  # Extract natural frequencies
    
    def mode_shape_impedance(self, mode_index):
        """Computes the modal impedance for a given mode shape."""
        omega = 2 * np.pi * self.frequency
        phi = self.eigenvectors[:, mode_index]  # Extract mode shape
        modal_mass = phi.T @ self.M @ phi
        modal_damping = phi.T @ self.C @ phi
        modal_stiffness = phi.T @ self.K @ phi
        
        # Compute modal impedance: Z = c + j(ωm - k/ω)
        return modal_damping + 1j * (omega * modal_mass - modal_stiffness / omega)
    
    def mode_velocity(self, mode_index):
        """Computes the velocity phasor for a given mode."""
        Z = self.mode_shape_impedance(mode_index)
        return self.F / Z
    
    def mode_complex_power(self, mode_index):
        """Computes mechanical complex power (S = P + jQ) for a given mode."""
        v = self.mode_velocity(mode_index)
        S = self.F * v.conjugate()
        return {"Complex Power (S)": S, "Real Power (P)": S.real, "Reactive Power (Q)": S.imag}
    
    def plot_mode_phasor(self, mode_index):
        """Plots the phasor diagram of force, velocity, and power for a given mode."""
        v = self.mode_velocity(mode_index)
        S = self.mode_complex_power(mode_index)["Complex Power (S)"]
        P = S.real  # Real Power
        Q = S.imag  # Reactive Power
        
        plt.figure(figsize=(6, 6))
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        
        # Plot Force Phasor
        plt.arrow(0, 0, self.F.real, self.F.imag, head_width=5, head_length=5, color='r', label='Force (F)')
        
        # Plot Velocity Phasor
        plt.arrow(0, 0, v.real * 20, v.imag * 20, head_width=5, head_length=5, color='b', label='Velocity (v)')
        
        # Plot Complex Power Phasor
        plt.arrow(0, 0, P / 5, 0, head_width=5, head_length=5, color='g', label='Real Power (P)')
        plt.arrow(P / 5, 0, 0, Q / 5, head_width=5, head_length=5, color='m', label='Reactive Power (Q)')
        
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.title(f"Mode {mode_index + 1} Phasor Diagram")
        plt.xlabel("Real Axis")
        plt.ylabel("Imaginary Axis")
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Define a 2DOF system (Mass, Damping, Stiffness Matrices)
    M = [[2, 0], [0, 1]]  # Mass Matrix
    C = [[1.0, 0], [0, 0.5]]  # Damping Matrix
    K = [[50, -20], [-20, 30]]  # Stiffness Matrix
    
    force_magnitude = 100  # N
    force_phase = 30  # degrees
    frequency = 10  # Hz
    
    # Create Mechanical System Object
    system = MDOFSystem(M, C, K, force_magnitude, force_phase, frequency)
    
    # Display results for each mode
    for mode in range(len(system.natural_frequencies)):
        print(f"Mode {mode + 1}:")
        print("Natural Frequency:", system.natural_frequencies[mode], "Hz")
        print("Modal Impedance:", system.mode_shape_impedance(mode))
        print("Velocity Phasor:", system.mode_velocity(mode))
        print("Complex Power Analysis:", system.mode_complex_power(mode))
        
        # Plot phasor diagram for each mode
        system.plot_mode_phasor(mode)
