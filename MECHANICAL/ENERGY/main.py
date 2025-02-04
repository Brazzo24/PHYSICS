import numpy as np

def solve_torsional_frequency_domain(J, C, K, T0, omega):
    """
    Solve the steady-state phasor response for a multi-inertia torsional system.

    Parameters:
    -----------
    J : (n, n) array_like
        Diagonal or full inertia matrix.
    C : (n, n) array_like
        Damping matrix.
    K : (n, n) array_like
        Stiffness matrix.
    T0 : (n,) array_like or complex
        Complex torque phasor vector (one entry might be non-zero for input, etc.).
    omega : float
        Excitation (angular) frequency in rad/s.

    Returns:
    --------
    Theta : (n,) complex ndarray
        Complex phasor of angular displacements (theta_1, ..., theta_n).
    """
    # Convert inputs to numpy arrays if not already
    J = np.array(J, dtype=complex)
    C = np.array(C, dtype=complex)
    K = np.array(K, dtype=complex)
    T0 = np.array(T0, dtype=complex)

    # Dynamic stiffness matrix:  K - omega^2 * J + j*omega * C
    # Note: j in Python is represented by 1j
    Z = K - omega**2 * J + 1j * omega * C

    # Solve for Theta (phasor of angular displacement)
    # Z * Theta = T0  ->  Theta = inv(Z) * T0
    Theta = np.linalg.solve(Z, T0)

    return Theta

def compute_shaft_torque_phasors(Theta, K_vals, C_vals, omega):
    """
    Given the phasor solutions Theta_i, compute the phasor of the shaft torques.
    For example, for a system with n=3 masses, we have 2 shafts: (1-2), (2-3).

    K_vals, C_vals: lists or arrays of torsional spring/damper constants
                    [K_1, K_2, ...], [C_1, C_2, ...]

    Returns:
    --------
    tau_phasors : list of complex
        The phasors for each shaft torque.
    """
    n_shafts = len(K_vals)
    tau_phasors = []

    for i in range(n_shafts):
        # The shaft i connects inertia i and i+1 (Python index 0->1, etc.)
        k_i = K_vals[i]
        c_i = C_vals[i]

        # Impedance of shaft i at freq omega
        Z_i = k_i + 1j * omega * c_i

        # Torsional torque in the shaft is Z_i * (Theta_{i+1} - Theta_i)
        shaft_torque = Z_i * (Theta[i+1] - Theta[i])
        tau_phasors.append(shaft_torque)

    return tau_phasors

def average_power_flow(tau_phasor, theta_phasor, omega):
    """
    Compute the average (real) power associated with a given torque phasor
    and angular velocity phasor = j*omega*theta_phasor.

    P_avg = 0.5 * Re{ (Torque_phasor) * (AngularVel_phasor)* }
    """
    angular_vel_phasor = 1j * omega * theta_phasor
    P_complex = 0.5 * tau_phasor * np.conjugate(angular_vel_phasor)
    return np.real(P_complex)


import numpy as np

# Example system parameters (SI units)
J1, J2, J3 = 0.05, 0.08, 0.02   # kg·m^2
K1, K2 = 1000, 600             # N·m/rad
C1, C2 = 5, 2                  # N·m·s/rad

# Construct J, C, K matrices (3x3)
J = np.diag([J1, J2, J3])

C = np.array([
    [ C1, -C1,   0 ],
    [-C1, C1+C2, -C2],
    [  0,   -C2,  C2]
], dtype=float)

K = np.array([
    [ K1,  -K1,   0 ],
    [-K1, K1+K2, -K2],
    [  0,   -K2,  K2]
], dtype=float)

# Suppose the input torque is applied only to J1:
# T_in(t) = real{ T0 * e^(jωt) }, with magnitude e.g., 10 N·m.
# No torque on J2, J3 for this example
T0_vec = np.array([10.0, 0.0, 0.0], dtype=complex)

# Choose an excitation frequency
omega = 10.0  # rad/s

# 1) Solve for the displacement phasors
Theta = solve_torsional_frequency_domain(J, C, K, T0_vec, omega)

print("Displacement phasors [Theta1, Theta2, Theta3]:")
print(Theta)

# 2) Compute shaft torques
K_vals = [K1, K2]
C_vals = [C1, C2]
tau_phasors = compute_shaft_torque_phasors(Theta, K_vals, C_vals, omega)
print("\nShaft torque phasors (shaft1, shaft2):")
print(tau_phasors)

# 3) Compute average power flows
#    For example, the power input from the torque on J1 is:
P_in_J1 = average_power_flow(T0_vec[0], Theta[0], omega)
print("\nAverage power input at J1: {:.4f} W".format(P_in_J1))

#    We can also see how much power flows through each shaft
#    from side i to side i+1, typically using tau_phasor of that shaft
#    and the angular velocity on one side.
#    We can interpret the sign of the result for direction of flow.
shaft1_power = average_power_flow(tau_phasors[0], Theta[0], omega)
shaft2_power = average_power_flow(tau_phasors[1], Theta[1], omega)

print("Average power flow across shaft 1 (from J1 side): {:.4f} W".format(shaft1_power))
print("Average power flow across shaft 2 (from J2 side): {:.4f} W".format(shaft2_power))

#    We can also compute the total damping losses if we want to confirm power balance.
#    Damping losses occur in each shaft damper:
#    P_damp_i = 0.5 * c_i * (omega * |Theta_{i+1} - Theta_i|)^2
#    or via phasor real power approach similarly.