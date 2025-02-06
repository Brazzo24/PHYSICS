from system import TorsionalSystem
from postprocessing import plot_energy_distributions, plot_torsional_system
from load_file import get_manual_input, load_input
#import numpy as np

# -------------------------------
# Configuration: Set to True for manual input, False for file input
# -------------------------------
# Configuration
USE_MANUAL_INPUT = True  # Change to False to use file input

# Load Data
if USE_MANUAL_INPUT:
    inertias, stiffnesses = get_manual_input()
else:
    inertias, stiffnesses = load_input("Input.csv")

# Debugging printout
print("Inertias:", inertias)
print("Stiffnesses:", stiffnesses)

# User option: set True if the first inertia should be connected to ground
connect_to_ground = False # Change to False if you do NOT want a ground connection

# -------------------------------
# Initialize Torsional System
# -------------------------------
system = TorsionalSystem(ignore_rigid_body_modes=True, rbm_tolerance=1e-13)

# Add inertias dynamically
nodes = []  # Store node IDs
for inertia in inertias:
    node = system.add_inertia(inertia)
    nodes.append(node)

# Add springs dynamically
# for (nodeA, nodeB, stiffness) in stiffnesses:
#    system.add_spring(nodeA, nodeB, stiffness)

if connect_to_ground:
    # connect first inertia to ground
    system.add_spring(0, nodes[0], stiffnesses[0][2])
    stiffness_index_offset = 1 # Offset index for sequential connections
else:
    stiffness_index_offset = 0 # No offset if no ground connection

# connect sequential inertia
for i in range(len(nodes) - 1):
    _, _, stiffness_value = stiffnesses[i + stiffness_index_offset] # extract stiffness only
    system.add_spring(nodes[i], nodes[i + 1], stiffness_value)

# -------------------------------
# Solve Eigenmodes
# -------------------------------
eigsq, modes = system.solve_eigenmodes()

# print("Eigenvalues (omega^2) after removing rigid-body:", eigsq)
# print("Mode shapes (each column) =\n", modes)
# print("Natural frequencies (rad/s):", system._omega)

# -------------------------------
# Compute Energy Distributions
# -------------------------------
ke_frac, pe_frac = system.compute_energy_distributions()
# print("\nKinetic fractions per mode:", ke_frac)
# print("Potential fractions per mode:", pe_frac)

# -------------------------------
# Plot Energy Distributions
# -------------------------------
plot_energy_distributions(system)

# -------------------------------
# Plot Transient Solution
# -------------------------------
# plot_time_response(system)

# Call the debug function
system.debug_print_system()
# Call the visualization function
plot_torsional_system(system)

#Hello there, Git! :)