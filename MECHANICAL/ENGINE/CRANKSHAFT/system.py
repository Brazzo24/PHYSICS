import numpy as np
from scipy.linalg import eigh

print("system.py loaded")

class TorsionalSystem:
    def __init__(self, ignore_rigid_body_modes=True, rbm_tolerance=1e-12):
        """
        Initialize an empty system.
        """
        self._num_inertias = 0
        self._inertia_dict = {}
        self._springs = []

        self._eigsq = None
        self._modes = None
        self._omega = None

        self.ignore_rigid_body_modes = ignore_rigid_body_modes
        self.rbm_tolerance = rbm_tolerance

    def debug_print_system(self):
        """Prints a structured representation of all inertias and spring connections."""
        print("\n Debugging Torsional System:")
        print(f"Total Inertias: {self._num_inertias}")
        print("\n Inertia Values:")
        for node, inertia in self._inertia_dict.items():
            print(f"  - Node {node}: Inertia = {inertia:.6f} kg·m²")

        print("\n Spring Connections:")
        for (nodeA, nodeB, stiffness) in self._springs:
            if nodeA == 0:
                print(f"  - Ground 🌍 → Node {nodeB} (Stiffness = {stiffness:.2f} Nm/rad)")
            else:
                print(f"  - Node {nodeA} ↔ Node {nodeB} (Stiffness = {stiffness:.2f} Nm/rad)")

        print("\n System check completed!\n")

    def solve_eigenmodes(self):
        """
        Solve the generalized eigenvalue problem:
            K phi = lambda I phi
        where lambda = omega^2.
        
        - Stores results internally in self._eigsq, self._omega, self._modes
        - If ignore_rigid_body_modes is True, remove eigenvalues below threshold
        """

        I_mat, K_mat = self.build_matrices()

        if self._num_inertias < 1:
            raise ValueError("No inertias added. The system is empty.")

        # Solve the generalized eigenvalue problem using SciPy
        eigsq_full, modes_full = eigh(K_mat, I_mat)

        # Convert eigenvalues to natural frequencies
        omega_full = np.sqrt(np.maximum(eigsq_full, 0.0))

        if self.ignore_rigid_body_modes:
            valid_indices = np.where(eigsq_full > self.rbm_tolerance)[0]
            self._eigsq = eigsq_full[valid_indices]
            self._omega = omega_full[valid_indices]
            self._modes = modes_full[:, valid_indices]
        else:
            self._eigsq = eigsq_full
            self._omega = omega_full
            self._modes = modes_full

        return self._eigsq, self._modes

    def add_inertia(self, inertia_value):
        node = self._num_inertias + 1
        self._inertia_dict[node] = inertia_value
        self._num_inertias += 1
        return node

    def add_spring(self, nodeA, nodeB, stiffness):
        self._springs.append((nodeA, nodeB, stiffness))

    def build_matrices(self):
        """
        Builds the inertia (I) and stiffness (K) matrices.
        """
        n = self._num_inertias
        I_mat = np.zeros((n, n))
        
        # Assign inertia values
        for node, inertia_val in self._inertia_dict.items():
            I_mat[node-1, node-1] = inertia_val

        K_mat = np.zeros((n, n))

        for connection in self._springs:
            nodeA, nodeB, k = connection  # Ensure k is properly unpacked
            k = float(k)  # Convert to scalar (avoids the sequence error)

            if nodeA == 0 and nodeB != 0:
                B_idx = nodeB - 1
                K_mat[B_idx, B_idx] += k
            elif nodeB == 0 and nodeA != 0:
                A_idx = nodeA - 1
                K_mat[A_idx, A_idx] += k
            else:
                A_idx, B_idx = nodeA - 1, nodeB - 1
                K_mat[A_idx, A_idx] += k
                K_mat[B_idx, B_idx] += k
                K_mat[A_idx, B_idx] -= k
                K_mat[B_idx, A_idx] -= k

        return I_mat, K_mat

    def compute_energy_equilibrium(self, theta, theta_dot):
        I_mat, K_mat = self.build_matrices()
        ke = 0.5 * np.sum((I_mat @ theta_dot.T) * theta_dot.T, axis=0)
        pe = 0.5 * np.sum((K_mat @ theta.T) * theta.T, axis=0)
        return ke, pe
    
    def compute_energy_distributions(self):
        """
        For each mode, compute:
          1) Fraction of kinetic energy in each inertia.
          2) Fraction of potential energy in each spring.
        
        Returns
        -------
        kinetic_fractions : list of arrays
            kinetic_fractions[r] is an array of length num_modes_kept
            giving the fraction of kinetic energy in each inertia 
            for mode r.
        
        potential_fractions : list of arrays
            potential_fractions[r] is an array of length = number_of_springs
            giving the fraction of potential energy in each spring
            for mode r.
        
        Must call solve_eigenmodes() first so we have self._modes.
        """
        if self._modes is None:
            raise ValueError("Eigenmodes not yet computed. Call solve_eigenmodes() first.")

        # Build matrices again to get the diagonal inertias, etc.
        I_mat, K_mat = self.build_matrices()
        
        # n is the total # inertias, but note we may have removed some modes
        n = self._num_inertias  
        # actual number of modes kept after removing rigid-body modes
        n_modes_kept = self._modes.shape[1]  
        
        # Inertia array
        inertia_list = np.diag(I_mat)

        # We'll store fractions in lists of length n_modes_kept
        kinetic_fractions = []
        potential_fractions = []

        def angle(phi, node):
            """Return the angle of the given mode shape at the given node.
               node=0 => ground => 0 deflection."""
            if node == 0:
                return 0.0
            else:
                return phi[node - 1]

        # Loop over each retained mode
        for r in range(n_modes_kept):
            phi_r = self._modes[:, r]

            # 1) Kinetic energy distribution
            KE_r_total = 0.0
            for i in range(n):
                KE_r_total += inertia_list[i] * phi_r[i]**2

            if KE_r_total > 0:
                KE_fracs = [
                    (inertia_list[i] * phi_r[i]**2) / KE_r_total
                    for i in range(n)
                ]
            else:
                KE_fracs = [0.0]*n

            # 2) Potential energy distribution
            spring_energy = []
            PE_r_total = 0.0
            for (nodeA, nodeB, k_val) in self._springs:
                diff = angle(phi_r, nodeA) - angle(phi_r, nodeB)
                U_m = k_val * (diff**2)  # ignoring the 1/2 factor for fraction
                spring_energy.append(U_m)
                PE_r_total += U_m

            if PE_r_total > 0:
                PE_fracs = [val / PE_r_total for val in spring_energy]
            else:
                PE_fracs = [0.0]*len(spring_energy)

            kinetic_fractions.append(KE_fracs)
            potential_fractions.append(PE_fracs)

        return kinetic_fractions, potential_fractions