import numpy as np
import matplotlib.pyplot as plt

def solve_2dof_phasors(M, C, K, F0, omega):
    """
    Solve the steady-state response for a 2-DoF system
    with forcing F(t) = Re{F0 * exp(i omega t)}.

    M, C, K are 2x2 numpy arrays.
    F0 is a 2-element forcing vector (phasor).
    omega is the driving frequency.

    Returns X, the 2-element complex displacement vector,
    and V, the 2-element complex velocity vector.
    """
    # The dynamic stiffness matrix:
    A = -omega**2 * M + 1j*omega*C + K

    # Solve for displacement phasors
    X = np.linalg.solve(A, F0)

    # Velocity phasors
    V = 1j * omega * X

    return X, V

def compute_forces_2dof(M, C, K, X, omega, F0):
    """
    Compute the inertial, damping, spring, and external forces
    as 2-element complex vectors.
    """
    F_inertial = - M * (omega**2) @ X   # matrix multiply
    F_damping  = -1j * omega * C @ X
    F_spring   = - K @ X
    # external is just the given phasor
    F_external = F0

    return F_inertial, F_damping, F_spring, F_external

def average_power_2dof(F0, V):
    """
    Average power input by the external force in multi-DoF:
    P_avg = 1/2 * Re( F0^H * conj(V) )
    where F0, V are 2-element complex vectors (phasors).
    """
    return 0.5 * np.real(np.vdot(F0, V))  # vdot does conjugate of 1st arg by default

def example_2dof():
    # Example parameters
    m1, m2 = 1.0, 1.0
    c1, c2 = 0.2, 0.2
    k1, k2 = 10.0, 15.0
    omega = 5.0
    # Forcing only on mass 1
    F0_mag = 1.0
    phi = 0.0

    # Construct the 2x2 matrices
    M = np.array([[m1,   0 ],
                  [ 0,  m2 ]])
    C = np.array([[c1+c2, -c2 ],
                  [-c2,    c2 ]])
    K = np.array([[k1+k2, -k2 ],
                  [-k2,    k2 ]])

    # Force phasor
    F0 = np.array([ F0_mag*np.exp(1j*phi),  0.0 ])

    # Solve for displacement
    X, V = solve_2dof_phasors(M, C, K, F0, omega)

    print("Displacement phasors [X1, X2]:", X)
    print("Velocity phasors     [V1, V2]:", V)

    # Compute forces
    F_inertial, F_damping, F_spring, F_external = compute_forces_2dof(M, C, K, X, omega, F0)

    print("Inertial:",  F_inertial)
    print("Damping:",   F_damping)
    print("Spring:",    F_spring)
    print("External:",  F_external)
    print("Sum of forces (should be ~0):", F_inertial + F_damping + F_spring + F_external)

    # Average power from external force
    P_avg = average_power_2dof(F_external, V)
    print("Average power input by the external force =", P_avg)

if __name__ == "__main__":
    example_2dof()


