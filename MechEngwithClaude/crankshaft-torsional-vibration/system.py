import numpy as np
from scipy.linalg import eigh

class TorsionalSystem:
    def __init__(self, ignore_rigid_body_modes=True, rbm_tolerance=1e-12):
        """
        Initialize an empty system.
        
        Parameters
        ----------
        ignore_rigid_body_modes : bool
            If True, any eigenvalues below rbm_tolerance are treated as 
            rigid-body modes and will be removed from the final results.
        
        rbm_tolerance : float
            Eigenvalue threshold below which we consider an eigenvalue 'zero'
            (i.e., a rigid-body mode). 
        """
        self._num_inertias = 0
        self._inertia_dict = {}  # maps node -> inertia value
        self._springs = []       # list of (nodeA, nodeB, stiffness)

        # For storing eigen-solutions:
        self._eigsq = None  # array of eigenvalues (omega^2)
        self._modes = None  # matrix whose columns are eigenvectors
        self._omega = None  # array of natural frequencies = sqrt(eigsq)

        # New attributes controlling how to handle rigid-body modes:
        self.ignore_rigid_body_modes = ignore_rigid_body_modes
        self.rbm_tolerance = rbm_tolerance

    def add_inertia(self, inertia_value):
        """
        Adds a new inertia to the system, returning its node number.
        Inertia is stored at the next available node.
        """
        new_node = self._num_inertias + 1  # next node
        self._inertia_dict[new_node] = inertia_value
        self._num_inertias += 1
        return new_node

    def add_spring(self, nodeA, nodeB, stiffness):
        """
        Adds a spring between nodeA and nodeB with the given stiffness.
        nodeA or nodeB can be 0 for ground.
        """
        self._springs.append((nodeA, nodeB, stiffness))

    def build_matrices(self):
        """
        Builds and returns the inertia (mass) matrix and the stiffness matrix
        for the current system.

        Returns
        -------
        I_mat : (num_inertias x num_inertias) numpy array
        K_mat : (num_inertias x num_inertias) numpy array
        """
        # We have self._num_inertias dynamic nodes (1..num_inertias).
        n = self._num_inertias
        
        # Build the inertia matrix (diagonal). 
        I_mat = np.zeros((n, n))
        for node, inertia_val in self._inertia_dict.items():
            row_col = node - 1  # 1-based node => 0-based index
            I_mat[row_col, row_col] = inertia_val

        # Build the stiffness matrix
        K_mat = np.zeros((n, n))
        for (nodeA, nodeB, k_val) in self._springs:
            if nodeA == 0 and nodeB != 0:
                # ground to inertia nodeB
                B_idx = nodeB - 1
                K_mat[B_idx, B_idx] += k_val
            elif nodeB == 0 and nodeA != 0:
                # ground to inertia nodeA
                A_idx = nodeA - 1
                K_mat[A_idx, A_idx] += k_val
            else:
                # inertia-inertia
                A_idx = nodeA - 1
                B_idx = nodeB - 1
                K_mat[A_idx, A_idx] += k_val
                K_mat[B_idx, B_idx] += k_val
                K_mat[A_idx, B_idx] -= k_val
                K_mat[B_idx, A_idx] -= k_val

        return I_mat, K_mat

    def solve_eigenmodes(self):
        """
        Solve the generalized eigenvalue problem:
            K phi = lambda I phi
        where lambda = omega^2.
        
        - Stores results internally:
            self._eigsq = array of eigenvalues (omega^2)
            self._omega = array of natural frequencies (omega)
            self._modes = matrix of eigenvectors (columns)
        - If ignore_rigid_body_modes is True, we remove any eigenvalues 
          below self.rbm_tolerance as 'rigid-body' modes.
        """
        I_mat, K_mat = self.build_matrices()
        
        if self._num_inertias < 1:
            raise ValueError("No inertias added. The system is empty.")
        
        # Solve the generalized eigenvalue problem using SciPy
        eigsq_full, modes_full = eigh(K_mat, I_mat)

        # Convert eigenvalues to natural frequencies
        omega_full = np.sqrt(np.maximum(eigsq_full, 0.0))  # clip negative (floating errors)

        if self.ignore_rigid_body_modes:
            # Find indices of valid (non-rigid-body) eigenvalues:
            # Rigid-body => eigenvalue < rbm_tolerance
            valid_indices = np.where(eigsq_full > self.rbm_tolerance)[0]
            # Filter them
            self._eigsq = eigsq_full[valid_indices]
            self._omega = omega_full[valid_indices]
            self._modes = modes_full[:, valid_indices]
        else:
            # Keep all
            self._eigsq = eigsq_full
            self._omega = omega_full
            self._modes = modes_full

        return self._eigsq, self._modes

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

    def __repr__(self):
        """
        Simple string representation for debugging.
        """
        return (f"TorsionalSystem(\n"
                f"  #inertias = {self._num_inertias},\n"
                f"  inertias = {self._inertia_dict},\n"
                f"  springs  = {self._springs},\n"
                f"  ignoring RBMs? = {self.ignore_rigid_body_modes},\n"
                f"  tolerance = {self.rbm_tolerance}\n"
                f")")

