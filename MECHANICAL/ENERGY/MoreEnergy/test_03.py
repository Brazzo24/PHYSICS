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
    
    def mode_energy_distribution(self, mode_index):
        """Computes the kinetic energy per mass and potential energy per spring for a given mode."""
        omega = 2 * np.pi * self.frequency
        phi = self.eigenvectors[:, mode_index]
        kinetic_energy = 0.5 * np.diag(self.M) * (np.abs(phi) ** 2) * (omega ** 2)
        
        # Compute potential energy per spring
        potential_energy = []
        for i in range(len(self.K) - 1):
            relative_displacement = np.abs(phi[i] - phi[i + 1])
            potential_energy.append(0.5 * self.K[i, i + 1] * relative_displacement ** 2)
        
        return kinetic_energy, potential_energy
    
    def plot_mode_energy_distribution(self, mode_index):
        """Visualizes the kinetic energy per mass and potential energy per spring for a given mode."""
        kinetic_energy, potential_energy = self.mode_energy_distribution(mode_index)
        masses = [f"Mass {i+1}" for i in range(len(kinetic_energy))]
        springs = [f"Spring {i+1}-{i+2}" for i in range(len(potential_energy))]
        
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        
        # Kinetic Energy per Mass
        ax[0].bar(masses, kinetic_energy, color='blue', label='Kinetic Energy')
        ax[0].set_ylabel("Energy (J)")
        ax[0].set_title(f"Kinetic Energy per Mass for Mode {mode_index + 1}")
        ax[0].grid(True, linestyle='--', alpha=0.6)
        ax[0].legend()
        
        # Potential Energy per Spring
        ax[1].bar(springs, potential_energy, color='orange', label='Potential Energy')
        ax[1].set_ylabel("Energy (J)")
        ax[1].set_title(f"Potential Energy per Spring for Mode {mode_index + 1}")
        ax[1].grid(True, linestyle='--', alpha=0.6)
        ax[1].legend()
        
        plt.tight_layout()
        plt.show()
    
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
    # Define a 3DOF system (Mass, Damping, Stiffness Matrices) in series
    M = [[2, 0, 0], [0, 1.5, 0], [0, 0, 1]]  # Mass Matrix
    C = [[0.1, -0.05, 0], [-0.05, 0.1, -0.05], [0, -0.05, 0.05]]  # Damping Matrix
    K = [[500, -200, 0], [-200, 400, -200], [0, -200, 300]]  # Stiffness Matrix
    
    force_magnitude = 100  # N
    force_phase = 30  # degrees
    frequency = 100  # Hz
    
    # Create Mechanical System Object
    system = MDOFSystem(M, C, K, force_magnitude, force_phase, frequency)
    
    # Display results for each mode
    for mode in range(len(system.natural_frequencies)):
        # print(f"Mode {mode + 1}:")
        system.plot_mode_phasor(mode)
        system.plot_mode_energy_distribution(mode)
