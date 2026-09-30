import numpy as np

class TorsionalVibrationSystem:
    def __init__(self):
        """
        Initialize the torsional vibration system with empty lists
        for moments of inertia and stiffness values.
        """
        self.inertias = []  # List to store moments of inertia
        self.stiffnesses = []  # List to store stiffnesses between masses

    def add_mass(self, inertia, stiffness=None):
        """
        Add a mass to the system.

        :param inertia: Moment of inertia for the new mass.
        :param stiffness: Stiffness between the previous and this mass (None for the last mass).
        """
        self.inertias.append(inertia)
        if stiffness is not None:
            self.stiffnesses.append(stiffness)

    def calculate_natural_frequencies(self):
        """
        Calculate the natural frequencies of the torsional vibration system.

        :return: Natural frequencies of the system.
        """
        if len(self.inertias) < 2 or len(self.stiffnesses) < 1:
            raise ValueError("The system must have at least 2 masses and 1 stiffness.")

        # Create the stiffness matrix (K) and inertia matrix (I)
        n = len(self.inertias)
        K = np.zeros((n, n))  # Stiffness matrix
        I = np.diag(self.inertias)  # Inertia matrix (diagonal)

        # Fill stiffness matrix
        for i in range(len(self.stiffnesses)):
            K[i, i] += self.stiffnesses[i]
            K[i + 1, i + 1] += self.stiffnesses[i]
            K[i, i + 1] -= self.stiffnesses[i]
            K[i + 1, i] -= self.stiffnesses[i]

        # Solve the eigenvalue problem (K * phi = omega^2 * I * phi)
        eigenvalues, _ = np.linalg.eig(np.linalg.inv(I).dot(K))

        # Natural frequencies are square roots of eigenvalues
        natural_frequencies = np.sqrt(np.abs(eigenvalues))
        natural_frequencies = np.sort(natural_frequencies)  # Sort frequencies
        return natural_frequencies

    def __str__(self):
        """
        String representation of the system.
        """
        return (f"Torsional Vibration System:\n"
                f"  Inertias: {self.inertias}\n"
                f"  Stiffnesses: {self.stiffnesses}\n")

# Example Usage
if __name__ == "__main__":
    # Create the torsional vibration system
    system = TorsionalVibrationSystem()

    # Add masses and stiffnesses
    system.add_mass(inertia=10)  # First mass, no stiffness yet
    system.add_mass(inertia=15, stiffness=200)
    system.add_mass(inertia=20, stiffness=300)
    system.add_mass(inertia=25, stiffness=400)

    # Print the system
    print(system)

    # Calculate and print natural frequencies
    frequencies = system.calculate_natural_frequencies()
    print(f"Natural Frequencies: {frequencies}")
