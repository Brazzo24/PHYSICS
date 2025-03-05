import numpy as np
import matplotlib.pyplot as plt

def solve_2dof_phasors(M, C, K, F0, omega):
    """
    Solve the steady-state response for a 2-DoF system with forcing
    F(t) = Re{ F0 * exp(i omega t) } at a single frequency omega.

    :param M, C, K: 2x2 numpy arrays (mass, damping, stiffness matrices).
    :param F0: 2-element complex forcing vector (phasor).
    :param omega: scalar frequency.
    :return: X (2-element complex displacement phasor).
    """
    # Dynamic stiffness matrix:
    A = -omega**2 * M + 1j*omega*C + K
    # Solve A * X = F0
    X = np.linalg.solve(A, F0)
    return X

def compute_forces_2dof(M, C, K, X, omega, F0):
    """
    Compute the phasors of inertial, damping, spring, and external forces
    for each mass (2-element vectors).

    :param M, C, K: 2x2 numpy arrays
    :param X: 2-element displacement phasor
    :param omega: driving frequency
    :param F0: 2-element forcing phasor
    :return: (F_inertial, F_damping, F_spring, F_external),
             each a 2-element complex vector
    """
    # Inertial force = -M * (omega^2) * X
    F_inertial = - M * (omega**2) @ X
    # Damping force = -i omega C X
    F_damping  = -1j * omega * C @ X
    # Spring force = -K X
    F_spring   = - K @ X
    # External force (phasor) = F0
    F_external = F0
    return F_inertial, F_damping, F_spring, F_external

def plot_forces_in_complex_plane(forces, labels, title="Forces in Complex Plane", margin=0.2):
    """
    Plot multiple complex vectors (forces) as arrows from origin in an Argand diagram.

    :param forces: list of complex numbers
    :param labels: list of string labels
    :param margin: float, additional margin as a percentage of the largest absolute value
    """
    fig, ax = plt.subplots(figsize=(6,6))

    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    for i, (F, lbl) in enumerate(zip(forces, labels)):
        x, y = F.real, F.imag
        ax.quiver(0, 0, x, y, angles='xy', scale_units='xy',
                  scale=1, color=colors[i % len(colors)],
                  label=f"{lbl}: {F:.2e}")

    # Determine the largest absolute value
    reals = [f.real for f in forces]
    imags = [f.imag for f in forces]
    x_min, x_max = min(reals), max(reals)
    y_min, y_max = min(imags), max(imags)

    max_abs = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))

    # Apply margin
    max_abs_with_margin = max_abs * (1 + margin)

    # Set equal limits for x and y axes with margin
    ax.set_xlim(-max_abs_with_margin, max_abs_with_margin)
    ax.set_ylim(-max_abs_with_margin, max_abs_with_margin)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()

def bode_2dof(M, C, K, F0, w_array):
    """
    Sweep over frequencies in w_array, solve for X(omega),
    and then plot amplitude & phase for each DOF (x1, x2).
    """
    amp_1 = np.zeros_like(w_array, dtype=float)
    amp_2 = np.zeros_like(w_array, dtype=float)
    phase_1 = np.zeros_like(w_array, dtype=float)
    phase_2 = np.zeros_like(w_array, dtype=float)
    
    for i, w in enumerate(w_array):
        X = solve_2dof_phasors(M, C, K, F0, w)
        x1, x2 = X[0], X[1]
        amp_1[i]   = np.abs(x1)
        amp_2[i]   = np.abs(x2)
        phase_1[i] = np.angle(x1)  # radians
        phase_2[i] = np.angle(x2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7,8), sharex=True)

    # --- Amplitude plots ---
    ax1.plot(w_array, amp_1, label="|X1|", lw=2)
    ax1.plot(w_array, amp_2, label="|X2|", lw=2, ls='--')
    ax1.set_ylabel("Amplitude")
    ax1.set_title("2-DoF System: Amplitude vs. Frequency")
    ax1.grid(True)
    ax1.legend()

    # --- Phase plots (in degrees) ---
    ax2.plot(w_array, np.degrees(phase_1), label="Phase X1", lw=2)
    ax2.plot(w_array, np.degrees(phase_2), label="Phase X2", lw=2, ls='--')
    ax2.set_xlabel("Frequency (rad/s)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title("2-DoF System: Phase vs. Frequency")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    # ---------------------------------------------------------
    # Example system parameters (2-DoF)
    # ---------------------------------------------------------
    m1, m2 = 1.0, 1.0
    c1, c2 = 0.2, 0.2
    k1, k2 = 10.0, 15.0

    M = np.array([[m1,   0 ],
                  [ 0,  m2 ]])
    C = np.array([[c1+c2, -c2 ],
                  [-c2,    c2 ]])
    K = np.array([[k1+k2, -k2 ],
                  [-k2,    k2 ]])

    # Suppose we force only mass 1 with amplitude F0_mag
    F0_mag = 1.0
    phi = 0.0  # phase offset
    F0 = np.array([ F0_mag*np.exp(1j*phi),  0.0 ])  # 2-element forcing

    # ---------------------------------------------------------
    # 1) Choose a single frequency, solve, and plot forces
    # ---------------------------------------------------------
    omega_single = 3.0  # pick a single frequency (rad/s)

    # Solve for displacement phasors
    X = solve_2dof_phasors(M, C, K, F0, omega_single)
    print(f"At frequency w={omega_single:.2f}, displacement phasors X = {X}")

    # Compute the 2-element force vectors
    F_inertial, F_damping, F_spring, F_external = compute_forces_2dof(M, C, K, X, omega_single, F0)
    sum_forces = F_inertial + F_damping + F_spring + F_external
    print(f"Sum of forces = {sum_forces} (should be ~0)")

    # --- Argand plot for each mass separately ---
    # mass 1 experiences F_inertial[0], F_damping[0], F_spring[0], F_external[0]
    forces_m1 = [F_inertial[0], F_damping[0], F_spring[0], F_external[0]]
    labels_m1 = ["Inertial(m1)", "Damping(m1)", "Spring(m1)", "External(m1)"]
    plot_forces_in_complex_plane(forces_m1, labels_m1,
        title=f"Mass 1 Forces at w={omega_single:.2f}")

    # mass 2 experiences F_inertial[1], F_damping[1], F_spring[1], F_external[1]
    forces_m2 = [F_inertial[1], F_damping[1], F_spring[1], F_external[1]]
    labels_m2 = ["Inertial(m2)", "Damping(m2)", "Spring(m2)", "External(m2)"]
    plot_forces_in_complex_plane(forces_m2, labels_m2,
        title=f"Mass 2 Forces at w={omega_single:.2f}")

    # ---------------------------------------------------------
    # 2) Sweep frequencies for a Bode-like plot
    # ---------------------------------------------------------
    w_array = np.linspace(0.1, 10, 200)
    bode_2dof(M, C, K, F0, w_array)

if __name__ == "__main__":
    main()
