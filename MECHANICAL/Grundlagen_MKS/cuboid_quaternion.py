import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def skew(v):
    """Return the skew-symmetric matrix of a 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def quat_kinematics_matrix(q):
    """
    Returns the L matrix so that dq/dt = 0.5 * L(q) * omega
    """
    e0, e1, e2, e3 = q
    L = np.array([
        [-e1, -e2, -e3],
        [ e0,  e3, -e2],
        [-e3,  e0,  e1],
        [ e2, -e1,  e0]
    ])
    return L

def rigid_body_ode(t, x, Theta, grav):
    # State unpacking
    r = x[0:3]       # position
    q = x[3:7]       # quaternion (Euler parameters)
    v = x[7:10]      # velocity
    omega = x[10:13] # angular velocity

    # Quaternion normalization (to prevent drift)
    q = q / np.linalg.norm(q)

    # Linear acceleration
    v_dot = grav  # gravity, no other forces

    # Quaternion derivative
    L = quat_kinematics_matrix(q)
    q_dot = 0.5 * L @ omega

    # Angular acceleration (Euler's equations)
    omega_dot = np.linalg.solve(
        Theta,
        -np.cross(omega, Theta @ omega)
    )

    # Derivative of position is velocity
    r_dot = v

    # Stack all derivatives
    dx = np.zeros_like(x)
    dx[0:3] = r_dot
    dx[3:7] = q_dot
    dx[7:10] = v_dot
    dx[10:13] = omega_dot
    return dx

# Main simulation block
if __name__ == '__main__':
    # Physical parameters
    a = 0.1
    b = 0.05
    c = 0.01
    rho = 700
    grav = np.array([0, 0, -9.81])

    mass = rho * a * b * c
    Theta = (mass/12) * np.diag([b**2 + c**2, c**2 + a**2, a**2 + b**2])

    # Initial state
    r0 = np.array([0, 0, 0])
    q0 = np.array([1, 0, 0, 0])  # quaternion (no rotation)
    v0 = np.array([0, 0, 7])     # thrown upward
    omega0 = np.array([0, 25, 0])
    omega0 = omega0 + np.max(np.abs(omega0)) / 100  # small disturbance

    x0 = np.concatenate([r0, q0, v0, omega0])

    # Time span and evaluation points
    tspan = (0, 1.5)
    t_eval = np.linspace(tspan[0], tspan[1], 300)

    # Integrate ODE
    sol = solve_ivp(
        fun=lambda t, x: rigid_body_ode(t, x, Theta, grav),
        t_span=tspan,
        y0=x0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-12,
        method='RK45'
    )

    # Extract angular velocities
    wx = sol.y[10]
    wy = sol.y[11]
    wz = sol.y[12]

    # Plot angular velocities
    plt.figure(figsize=(8,4))
    plt.plot(sol.t, wx, label=r'$\omega_x$')
    plt.plot(sol.t, wy, label=r'$\omega_y$')
    plt.plot(sol.t, wz, label=r'$\omega_z$')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [rad/s]')
    plt.legend()
    plt.title('Cuboid Angular Velocities in Free Flight')
    plt.tight_layout()
    plt.show()
