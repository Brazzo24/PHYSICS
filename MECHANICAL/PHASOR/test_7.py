import cmath
import numpy as np
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt

class MDOFSystem:
    """Base class for an MDOF mechanical system, handling matrices and eigenanalysis."""
    def __init__(self, mass_matrix, damping_matrix, stiffness_matrix):
        self.M = np.array(mass_matrix, dtype=np.float64)
        self.C = np.array(damping_matrix, dtype=np.float64)
        self.K = np.array(stiffness_matrix, dtype=np.float64)
        
        # Compute mode shapes and natural frequencies
        self.eigenvalues, self.eigenvectors = scipy.linalg.eig(self.K, self.M)
        self.natural_frequencies = np.sqrt(np.real(self.eigenvalues))

class MDOFFrequencyResponse(MDOFSystem):
    """Handles frequency-domain analysis using phasors and impedance."""
    def __init__(self, mass_matrix, damping_matrix, stiffness_matrix, force_magnitude, force_phase, frequency):
        super().__init__(mass_matrix, damping_matrix, stiffness_matrix)
        self.F = cmath.rect(force_magnitude, np.radians(force_phase))
        self.frequency = frequency
    
    def mode_shape_impedance(self, mode_index):
        omega = 2 * np.pi * self.frequency
        phi = self.eigenvectors[:, mode_index]
        modal_mass = phi.T @ self.M @ phi
        modal_damping = phi.T @ self.C @ phi
        modal_stiffness = phi.T @ self.K @ phi
        return modal_damping + 1j * (omega * modal_mass - modal_stiffness / omega)
    
    def mode_velocity(self, mode_index):
        return self.F / self.mode_shape_impedance(mode_index)
    
    def mode_complex_power(self, mode_index):
        v = self.mode_velocity(mode_index)
        S = self.F * v.conjugate()
        return {"Complex Power (S)": S, "Real Power (P)": S.real, "Reactive Power (Q)": S.imag}

class MDOFTimeResponse(MDOFSystem):
    """Handles time-domain analysis using numerical integration."""
    def __init__(self, mass_matrix, damping_matrix, stiffness_matrix):
        super().__init__(mass_matrix, damping_matrix, stiffness_matrix)
    
    def equations_of_motion(self, t, state, force_func):
        """State-space representation of MDOF system."""
        n = len(self.M)
        x = state[:n]
        v = state[n:]
        
        # Compute accelerations
        F_t = force_func(t)  # External force at time t
        a = np.linalg.solve(self.M, F_t - self.C @ v - self.K @ x)
        return np.concatenate((v, a))
    
    def solve_time_response(self, x0, v0, t_span, force_func):
        """Solves the system's time response for given initial conditions and force function."""
        n = len(self.M)
        initial_state = np.concatenate((x0, v0))
        sol = scipy.integrate.solve_ivp(self.equations_of_motion, t_span, initial_state, args=(force_func,), t_eval=np.linspace(t_span[0], t_span[1], 1000))
        return sol.t, sol.y[:n], sol.y[n:]
    
    def plot_time_response(self, t, x, v):
        """Plots displacement and velocity over time for each mass."""
        plt.figure(figsize=(10, 6))
        for i in range(x.shape[0]):
            plt.plot(t, x[i], label=f"Mass {i+1} Displacement")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.title("Time Response: Displacement")
        plt.legend()
        plt.grid()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        for i in range(v.shape[0]):
            plt.plot(t, v[i], label=f"Mass {i+1} Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Time Response: Velocity")
        plt.legend()
        plt.grid()
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Define a 3DOF system (Mass, Damping, Stiffness Matrices) in series
    M = [[2, 0, 0], [0, 1.5, 0], [0, 0, 1]]
    C = [[0.1, -0.05, 0], [-0.05, 0.1, -0.05], [0, -0.05, 0.05]]
    K = [[500, -200, 0], [-200, 400, -200], [0, -200, 300]]
    
    # Frequency Response Analysis
    freq_analysis = MDOFFrequencyResponse(M, C, K, force_magnitude=100, force_phase=30, frequency=100)
    for mode in range(len(freq_analysis.natural_frequencies)):
        print(f"Mode {mode + 1}:")
        print("Complex Power Analysis:", freq_analysis.mode_complex_power(mode))
    
    # Time Response Analysis
    def force_function(t):
        return np.array([0, 0, 100 * np.sin(2 * np.pi * 1 * t)])  # Sinusoidal force on last mass
    
    time_analysis = MDOFTimeResponse(M, C, K)
    t, x, v = time_analysis.solve_time_response(x0=[0, 0, 0], v0=[0, 0, 0], t_span=(0, 10), force_func=force_function)
    time_analysis.plot_time_response(t, x, v)
