from system import TorsionalSystem
from postprocessing import plot_energy_distributions
from load_file import load_input

# -------------------------------
# Load Data from Input File
# -------------------------------
inertias, stiffnesses = load_input("Input.csv")

# -------------------------------
# Initialize Torsional System
# -------------------------------
system = TorsionalSystem(ignore_rigid_body_modes=True, rbm_tolerance=1e-12)

# Add inertias dynamically
nodes = []  # Store node IDs
for inertia in inertias:
    node = system.add_inertia(inertia)
    nodes.append(node)

# Add springs dynamically
for (nodeA, nodeB, stiffness) in stiffnesses:
    system.add_spring(nodeA, nodeB, stiffness)

# -------------------------------
# Solve Eigenmodes
# -------------------------------
eigsq, modes = system.solve_eigenmodes()

print("Eigenvalues (omega^2) after removing rigid-body:", eigsq)
print("Mode shapes (each column) =\n", modes)
print("Natural frequencies (rad/s):", system._omega)

# -------------------------------
# Compute Energy Distributions
# -------------------------------
ke_frac, pe_frac = system.compute_energy_distributions()
print("\nKinetic fractions per mode:", ke_frac)
print("Potential fractions per mode:", pe_frac)

# -------------------------------
# Plot Energy Distributions
# -------------------------------
plot_energy_distributions(system)