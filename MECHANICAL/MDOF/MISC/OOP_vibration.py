import numpy as np
import matplotlib.pyplot as plt

class MDOFSystem:
    def __init__(self, mass_list, damping_list, springs, springs_to_ground=None):
        self.mass_list = mass_list
        self.damping_list = damping_list
        self.springs = springs  # List of (k, i, j)
        self.springs_to_ground = springs_to_ground if springs_to_ground else []
        self.n = len(mass_list)

        self.M = self._assemble_mass_matrix()
        self.C = self._assemble_damping_matrix()
        self.K = self._assemble_stiffness_matrix()

    def _assemble_mass_matrix(self):
        return np.diag(self.mass_list)

    def _assemble_damping_matrix(self):
        return np.diag(self.damping_list)

    def _assemble_stiffness_matrix(self):
        K = np.zeros((self.n, self.n))
        for k, i, j in self.springs:
            K[i, i] += k
            K[j, j] += k
            K[i, j] -= k
            K[j, i] -= k
        for k, i in self.springs_to_ground:
            K[i, i] += k
        print(K)
        return K
    
    

    def solve_frequency_response(self, F, omega_vals):
        responses = []
        for omega in omega_vals:
            H = -omega**2 * self.M + 1j * omega * self.C + self.K
            x_hat = np.linalg.solve(H, F)
            responses.append(x_hat)
        return np.array(responses)


class EnergyAnalyzer:
    def __init__(self, system: MDOFSystem, omega_vals, x_hat_all):
        self.system = system
        self.omega_vals = omega_vals
        self.x_hat_all = x_hat_all
        self.n_freqs = len(omega_vals)
        self.n_dof = system.n
        self.T_kin = np.zeros((self.n_freqs, self.n_dof))
        self.V_springs = np.zeros((self.n_freqs, len(system.springs)))
        self.V_ground = np.zeros((self.n_freqs, len(system.springs_to_ground)))
        self._compute_energies()

    def _compute_energies(self):
        M = self.system.M
        for i, omega in enumerate(self.omega_vals):
            x_hat = self.x_hat_all[i]
            for j in range(self.n_dof):
                self.T_kin[i, j] = 0.5 * M[j, j] * (omega * np.abs(x_hat[j]))**2
            for s_idx, (k, i_dof, j_dof) in enumerate(self.system.springs):
                rel_disp = x_hat[i_dof] - x_hat[j_dof]
                self.V_springs[i, s_idx] = 0.5 * k * np.abs(rel_disp)**2
            for s_idx, (k, j_dof) in enumerate(self.system.springs_to_ground):
                self.V_ground[i, s_idx] = 0.5 * k * np.abs(x_hat[j_dof])**2

    def total_kinetic_energy(self):
        return np.sum(self.T_kin, axis=1)

    def total_potential_energy(self):
        return np.sum(self.V_springs, axis=1) + np.sum(self.V_ground, axis=1)

    def energy_distribution_at(self, target_frequency):
        idx = np.argmin(np.abs(self.omega_vals - target_frequency))
        omega = self.omega_vals[idx]
        T_total = np.sum(self.T_kin[idx, :])
        V_total = np.sum(self.V_springs[idx, :]) + np.sum(self.V_ground[idx, :])
        ke_frac = self.T_kin[idx, :] / T_total if T_total > 0 else np.zeros_like(self.T_kin[idx, :])
        pe_spring_frac = self.V_springs[idx, :] / V_total if V_total > 0 else np.zeros_like(self.V_springs[idx, :])
        pe_ground_frac = self.V_ground[idx, :] / V_total if V_total > 0 else np.zeros_like(self.V_ground[idx, :])
        return omega, ke_frac, pe_spring_frac, pe_ground_frac


class Visualizer:
    @staticmethod
    def plot_total_energy(omega_vals, T_total, V_total):
        plt.figure(figsize=(10, 6))
        plt.plot(omega_vals, T_total, label="Total Kinetic Energy (peak)")
        plt.plot(omega_vals, V_total, label="Total Potential Energy (peak)")
        plt.xlabel("Frequency [rad/s]")
        plt.ylabel("Energy [J]")
        plt.title("Peak Energies vs. Frequency")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_energy_distribution(ke_frac, pe_spring_frac, pe_ground_frac, system: MDOFSystem, omega):
        n_dof = len(ke_frac)
        spring_labels = [f"{i}-{j}" for (_, i, j) in system.springs]
        ground_labels = [f"{j}-gnd" for (_, j) in system.springs_to_ground]
        pe_all = np.concatenate([pe_spring_frac, pe_ground_frac])
        pe_labels = spring_labels + ground_labels

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(np.arange(n_dof), ke_frac * 100)
        plt.xlabel("DOF")
        plt.ylabel("KE fraction [%]")
        plt.title(f"Kinetic Energy at ω = {omega:.2f} rad/s")

        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(pe_all)), pe_all * 100, color="orange")
        plt.xticks(np.arange(len(pe_all)), pe_labels, rotation=45)
        plt.ylabel("PE fraction [%]")
        plt.title("Potential Energy Distribution")

        plt.tight_layout()
        plt.show()


# Step 1: Define system inputs
mass_list = [2.0, 1.0, 1.5, 1.0]          # Mass per DOF
damping_list = [0.1, 0.2, 0.1, 0.1]      # Damping per DOF

# Define springs as (stiffness, DOF_i, DOF_j)
springs = [
    (20.0, 0, 1),
    (10.0, 1, 2),
    (15.0, 0, 3)
]

# Define springs to ground as (stiffness, DOF_j)
springs_to_ground = []  # Could add e.g., (50.0, 0) for a spring at DOF 0 to ground

# Step 2: Create the system object
system = MDOFSystem(mass_list, damping_list, springs, springs_to_ground)

# Step 3: Define excitation and frequency range
F = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # Excite DOF 0
omega_vals = np.linspace(0.1, 20, 300)        # Frequency range (rad/s)

# Step 4: Solve frequency response
x_hat_all = system.solve_frequency_response(F, omega_vals)

# Step 5: Analyze energy
analyzer = EnergyAnalyzer(system, omega_vals, x_hat_all)
T_total = analyzer.total_kinetic_energy()
V_total = analyzer.total_potential_energy()

# Step 6: Plot total energy vs frequency
Visualizer.plot_total_energy(omega_vals, T_total, V_total)

# Step 7: Plot energy distribution at a specific frequency
target_freq = 5.0  # rad/s
omega_eval, ke_frac, pe_spr_frac, pe_gnd_frac = analyzer.energy_distribution_at(target_freq)
Visualizer.plot_energy_distribution(ke_frac, pe_spr_frac, pe_gnd_frac, system, omega_eval)
