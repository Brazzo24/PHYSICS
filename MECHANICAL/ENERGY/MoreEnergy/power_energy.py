import numpy as np
import matplotlib.pyplot as plt

def driven_damped_oscillator_forces(m, c, k, F0, omega, phi=0.0):
    """
    Returns the complex phasors for the four forces (inertial, damping,
    spring, and external) at steady state, given system parameters.

    F(t) = Re{ F0 * exp[i*(omega*t + phi)] } = F0 cos(omega*t + phi)

    :param m: Mass
    :param c: Damping coefficient
    :param k: Spring constant
    :param F0: Amplitude of driving force
    :param omega: Driving angular frequency
    :param phi: Phase offset of the driving force, in radians
    :return: (F_inertial, F_damping, F_spring, F_external, X)
             where each is a complex phasor, and X is the displacement phasor.
    """
    # Driving force phasor:
    F_phasor = F0 * np.exp(1j * phi)

    # Steady-state displacement phasor X:
    # X = F_ext / (k - m*omega^2 + i c omega)
    X = F_phasor / (k - m * omega**2 + 1j * c * omega)

    # Inertial force: -m ω^2 X
    F_inertial = -m * omega**2 * X

    # Damping force: -i c ω X
    F_damping  = -1j * c * omega * X

    # Spring force: -k X
    F_spring   = -k * X

    # External forcing phasor = F_phasor
    F_external = F_phasor

    return F_inertial, F_damping, F_spring, F_external, X

def average_power(m, c, k, F0, omega, phi=0.0):
    """
    Compute the time-averaged power delivered by the driving force
    in steady-state at frequency omega and phase offset phi.

    The force phasor is F0 * exp(i * phi).
    The velocity phasor is i * omega * X.
    We compute P_avg = (1/2) * Re{ F * conj(velocity) }.
    """
    F_phasor = F0 * np.exp(1j * phi)
    X = F_phasor / (k - m * omega**2 + 1j * c * omega)
    V = 1j * omega * X  # velocity phasor

    # Average power:
    #   P_avg = (1/2) * Re[ F_phasor * conj(V) ]
    P_avg = 0.5 * np.real(F_phasor * np.conjugate(V))
    return P_avg

def plot_forces_in_complex_plane(forces, labels, title="Forces in the Complex Plane"):
    """
    Plot the given complex phasors in a 2D Argand diagram (real vs. imaginary).
    forces: list or tuple of complex numbers
    labels: corresponding list of string labels
    """
    fig, ax = plt.subplots(figsize=(6,6))

    # Plot each force as a vector from the origin
    colors = ['r','g','b','m','c','y','k']
    for i, (F, lbl) in enumerate(zip(forces, labels)):
        x, y = F.real, F.imag
        ax.quiver(0, 0, x, y, angles='xy', scale_units='xy',
                  scale=1, color=colors[i % len(colors)],
                  label=f"{lbl}: {F:.2e}")

    # Make sure the axes are scaled to see the arrows comfortably
    reals = [f.real for f in forces]
    imags = [f.imag for f in forces]
    x_min, x_max = min(reals), max(reals)
    y_min, y_max = min(imags), max(imags)
    # Add some margin so the arrows aren't squashed
    margin_x = 0.1*(x_max - x_min + 1e-15)
    margin_y = 0.1*(y_max - y_min + 1e-15)
    ax.set_xlim([x_min - margin_x, x_max + margin_x])
    ax.set_ylim([y_min - margin_y, y_max + margin_y])

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('Real axis')
    ax.set_ylabel('Imag axis')
    ax.set_title(title)
    ax.legend(loc='best')
    plt.grid(True)
    plt.show()

def main():
    # --- System parameters ---
    m = 1.0         # mass
    c = 0.5         # damping coefficient
    k = 10.0        # spring constant
    F0 = 1.0        # amplitude of driving force
    omega = 3.0     # driving angular frequency
    phi = np.pi     # phase offset of driving force in radians (e.g., 180°)

    # Get the force phasors + displacement amplitude with phase offset
    F_inertial, F_damping, F_spring, F_external, X = driven_damped_oscillator_forces(m, c, k, F0, omega, phi)

    # Plot in the complex plane
    forces = [F_inertial, F_damping, F_spring, F_external]
    labels = ["Inertial", "Damping", "Spring", "External"]
    plot_forces_in_complex_plane(forces, labels, 
        title=f"Forces at ω={omega} rad/s, phi={phi} rad\n(m={m}, c={c}, k={k}, F0={F0})")

    # Print sum of all forces (should be ~ zero in steady-state)
    sum_forces = sum(forces)
    print("Sum of all force phasors = ", sum_forces)

    # Compute and print average power delivered by the external force
    P = average_power(m, c, k, F0, omega, phi)
    print(f"\nPhase offset phi = {phi} rad")
    print(f"Displacement amplitude |X| = {abs(X):.3f}")
    print(f"Average power input by the driving force = {P:.5f} (in whatever units)")

if __name__ == "__main__":
    main()
