import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def euler_dot(phi, w):
    """
    Computes time-derivative of ZYX Euler angles from angular velocity vector w.
    phi: [phi1, phi2, phi3] Euler angles (ZYX order)
    w: [wx, wy, wz] angular velocity in body frame
    """
    phi1, phi2, phi3 = phi
    s2 = np.sin(phi2)
    c2 = np.cos(phi2)
    t2 = np.tan(phi2)
    s3 = np.sin(phi3)
    c3 = np.cos(phi3)
    
    # ZYX Euler angles: [yaw, pitch, roll] or [phi1, phi2, phi3]
    # Angular velocity to Euler angle rate mapping
    E = np.array([
        [1, s3*t2, c3*t2],
        [0, c3,    -s3],
        [0, s3/c2, c3/c2]
    ])
    return E @ w

def quader_fk(t, x, theta, grav):
    """
    State-space derivative for rigid cuboid in gravity.
    x: [r (3), phi (3), v (3), w (3)]
    theta: unused (placeholder for external forces or parameters)
    grav: additional gravity vector (set to 0 for only gravity)
    """
    # Unpack state
    r = x[0:3]       # position
    phi = x[3:6]     # Euler angles (ZYX)
    v = x[6:9]       # linear velocity
    w = x[9:12]      # angular velocity

    # Physical parameters
    m = 1.3
    J = np.diag([0.11, 0.21, 0.16])

    # Forces and torques
    Fp = np.array([0, 0, -m*9.81]) + (grav if isinstance(grav, np.ndarray) else 0)

    # Kinematics and dynamics
    phi_p = euler_dot(phi, w)
    v_p = Fp / m
    w_p = np.linalg.solve(J, -np.cross(w, J @ w))

    # State derivative
    x_p = np.concatenate((v, phi_p, v_p, w_p))
    return x_p

def sys_A0k_pri(epr):
    """
    Computes rotation matrix from Euler parameters (quaternions)
    epr: [e0, e1, e2, e3]
    """
    e0, e1, e2, e3 = epr
    A0k = np.array([
        [e0**2 + e1**2 - e2**2 - e3**2,     2*(e1*e2 - e0*e3),         2*(e1*e3 + e0*e2)],
        [2*(e1*e2 + e0*e3),     e0**2 - e1**2 + e2**2 - e3**2,         2*(e2*e3 - e0*e1)],
        [2*(e1*e3 - e0*e2),     2*(e2*e3 + e0*e1),     e0**2 - e1**2 - e2**2 + e3**2]
    ])
    return A0k

if __name__ == "__main__":
    # Initial state
    a, b, c = 0.3, 0.6, 0.1  # dimensions (not used here, but could be for more advanced features)
    m = 1.3
    J = np.diag([0.11, 0.21, 0.16])

    x0 = np.zeros(12)
    # Position (first 3) and velocity (6:9) start at zero
    # Euler angles (3:6) start at zero
    # Angular velocity (9:12) start at zero
    # The following sets the orientation in y-axis (as per the Matlab script, but for Euler angles it's just zeros)
    # If you want to use quaternions instead, adapt accordingly.

    tspan = (0, 1.5)
    t_eval = np.linspace(*tspan, 150)

    # Run simulation
    sol = solve_ivp(
        lambda t, x: quader_fk(t, x, np.zeros(3), 0),
        tspan,
        x0,
        t_eval=t_eval,
        rtol=1e-6, atol=1e-9
    )

    # Plotting x-position vs time
    plt.figure(figsize=(8, 4))
    plt.plot(sol.t, sol.y[0], 'o-')
    plt.xlabel('Time [s]')
    plt.ylabel('x-Position [m]')
    plt.title('x-Position of the Cuboid in Free Fall')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
