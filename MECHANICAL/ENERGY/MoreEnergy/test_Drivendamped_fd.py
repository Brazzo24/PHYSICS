import numpy as np
import matplotlib.pyplot as plt

def driven_damped_oscillator_forces(m, c, k, F0, omega):
    """
    Returns the complex phasors for the four forces (inertial, damping,
    spring, and external) at steady state, given system parameters.
    """
    # Steady-state amplitude (complex)
    X = F0 / (k - m*omega**2 + 1j*c*omega)

    # Inertial force: -m ω^2 X e^{i ω t}
    F_inertial = -m * omega**2 * X

    # Damping force: -i c ω X e^{i ω t}
    F_damping  = -1j * c * omega * X

    # Spring force: -k X e^{i ω t}
    F_spring   = -k * X

    # External forcing: F0 e^{i ω t} (phasor amplitude = F0)
    F_external = F0

    return F_inertial, F_damping, F_spring, F_external

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
        # Real part is x, Imag part is y
        x, y = F.real, F.imag
        # Use quiver to plot a vector from (0,0) to (x,y)
        ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1,
                  color=colors[i % len(colors)], label=f"{lbl}: {F:.2e}")

    # Draw axes lines
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)

    # ax.set_aspect('equal', 'box')
    lims = 5
    ax.set_xlabel('Real axis')
    ax.set_ylabel('Imag axis')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_ylim(-lims, lims)
    ax.set_xlim(-lims, lims)
    plt.grid(True)
    plt.show()

def main():
    # --- System parameters ---
    m = 1.0         # mass
    c = 1.0         # damping coefficient
    k = 10.0        # spring constant
    F0 = 10.0        # amplitude of driving force
    omega = 1.0     # driving angular frequency

    # Compute the force phasors
    F_inertial, F_damping, F_spring, F_external = driven_damped_oscillator_forces(m, c, k, F0, omega)

    # Put them in a list
    forces = [F_inertial, F_damping, F_spring, F_external]
    labels = ["Inertial", "Damping", "Spring", "External"]

    # Plot in the complex plane
    plot_forces_in_complex_plane(forces, labels, 
        title=f"Forces at ω={omega} rad/s (m={m}, c={c}, k={k}, F0={F0})")

if __name__ == "__main__":
    main()
