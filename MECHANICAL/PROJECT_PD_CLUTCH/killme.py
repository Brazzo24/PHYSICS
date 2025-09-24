import numpy as np
import matplotlib.pyplot as plt

class TwoDOFSystem:
    def __init__(self, m1, m2, c1, c2, k1, k2):
        self.M = np.array([[m1,  0 ],
                           [ 0, m2]])
        self.C = np.array([[c1+c2, -c2 ],
                           [-c2,     c2]])
        self.K = np.array([[k1+k2, -k2 ],
                           [-k2,     k2]])

    def solve_response(self, F0, omega):
        A = -omega**2 * self.M + 1j * omega * self.C + self.K
        X = np.linalg.solve(A, F0)
        return X

    def compute_forces(self, X, omega, F0):
        F_inertial = -self.M @ (omega**2 * X)
        F_damping  = -1j * omega * self.C @ X
        F_spring   = -self.K @ X
        F_external = F0
        return F_inertial, F_damping, F_spring, F_external

    def bode_plot(self, F0, w_array):
        amp = np.zeros((2, len(w_array)))
        phase = np.zeros((2, len(w_array)))
        
        for i, w in enumerate(w_array):
            X = self.solve_response(F0, w)
            amp[:, i] = np.abs(X)
            phase[:, i] = np.angle(X)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        
        ax1.plot(w_array, amp[0], label="|X1|")
        ax1.plot(w_array, amp[1], label="|X2|", linestyle="--")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Amplitude vs Frequency")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(w_array, np.degrees(phase[0]), label="Phase X1")
        ax2.plot(w_array, np.degrees(phase[1]), label="Phase X2", linestyle="--")
        ax2.set_xlabel("Frequency (rad/s)")
        ax2.set_ylabel("Phase (deg)")
        ax2.set_title("Phase vs Frequency")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_force_phasors(forces, labels, title, margin=0.2):
        fig, ax = plt.subplots(figsize=(6,6))
        colors = ['r', 'g', 'b', 'm']
        
        for i, (f, lbl) in enumerate(zip(forces, labels)):
            ax.quiver(0, 0, f.real, f.imag, angles='xy', scale_units='xy', scale=1,
                      color=colors[i % len(colors)], label=f"{lbl}: {f:.2e}")
        
        reals = [f.real for f in forces]
        imags = [f.imag for f in forces]
        max_val = max(map(abs, reals + imags)) * (1 + margin)

        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_aspect('equal')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imag')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.show()


# ----------------- Example Usage -------------------

def main():
    # Define system
    system = TwoDOFSystem(m1=1.0, m2=1.0, c1=0.2, c2=0.2, k1=10.0, k2=15.0)

    # Forcing parameters
    omega = 3.0
    F0 = np.array([1.0 + 0j, 0.0])

    # Solve for phasors
    X = system.solve_response(F0, omega)
    print(f"X at omega={omega:.2f} rad/s: {X}")

    # Compute forces
    F_inertial, F_damping, F_spring, F_external = system.compute_forces(X, omega, F0)
    total_force = F_inertial + F_damping + F_spring + F_external
    print(f"Total (should be ~0): {total_force}")

    # Plot for mass 1
    system.plot_force_phasors(
        [F_inertial[0], F_damping[0], F_spring[0], F_external[0]],
        ["Inertial (m1)", "Damping (m1)", "Spring (m1)", "External (m1)"],
        title=f"Forces on Mass 1 at ω = {omega:.2f}"
    )

    # Plot for mass 2
    system.plot_force_phasors(
        [F_inertial[1], F_damping[1], F_spring[1], F_external[1]],
        ["Inertial (m2)", "Damping (m2)", "Spring (m2)", "External (m2)"],
        title=f"Forces on Mass 2 at ω = {omega:.2f}"
    )

    # Bode plot
    w_array = np.linspace(0.1, 10, 200)
    system.bode_plot(F0, w_array)


if __name__ == "__main__":
    main()
