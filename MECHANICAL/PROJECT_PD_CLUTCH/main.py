"""
Code to calculate the Primary Damper (non-linear) in transients.
Also include a Clutch model with slipping function.

"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Utilities: load/build curve
# =========================
def load_curve_from_csv(csv_path):
    """
    CSV must contain: angle_deg, torque_Nm
    Angle is relative twist (deg), torque is resisting torque (N·m).
    """
    df = pd.read_csv(csv_path)
    if not {"angle_deg", "torque_Nm"} <= set(df.columns):
        raise ValueError("CSV must have columns: angle_deg, torque_Nm")
    # Sort by angle and drop duplicates
    df = df.sort_values("angle_deg").drop_duplicates("angle_deg")
    return df["angle_deg"].to_numpy(), df["torque_Nm"].to_numpy()

def make_demo_curve():
    """
    Demo torque-angle curve (deg → N·m), slightly softer near zero.
    Replace with your data for real runs.
    """
    angles_deg = np.array([
        -5.0,-4.0,-3.0,-2.0,-1.5,-1.0,-0.5,-0.2,-0.1, 0.0,
         0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0
    ])
    torque = []
    for a in angles_deg:
        if abs(a) < 0.2:
            k_soft = 80.0
            torque.append(k_soft * a)
        else:
            k1 = 90.0 if abs(a) < 2.0 else 100.0
            torque.append(k1 * a)
    return angles_deg, np.array(torque)

# =========================
# Nonlinear link (spring + viscous)
# =========================
class NonlinearSpringDamperLookup:
    """
    Torque = f(theta_rel) + c * theta_rel_dot
      - theta_rel [rad], theta_rel_dot [rad/s]
      - The lookup table is torque vs angle in *degrees*
    """
    def __init__(self, angle_deg, torque_nm, damping_viscous=0.0, extrapolate=True):
        order = np.argsort(angle_deg)
        self.angle_deg = np.asarray(angle_deg, float)[order]
        self.torque_nm = np.asarray(torque_nm, float)[order]
        self.c = float(damping_viscous)  # N·m·s/rad
        self._interp = PchipInterpolator(self.angle_deg, self.torque_nm,
                                         extrapolate=extrapolate)

    def spring_torque(self, theta_rel_rad):
        theta_deg = np.degrees(theta_rel_rad)
        return float(self._interp(theta_deg))

    def torque(self, theta_rel_rad, theta_rel_rate_rad_s):
        return self.spring_torque(theta_rel_rad) + self.c * float(theta_rel_rate_rad_s)

# =========================
# Other elements
# =========================
class LinearDamperToGround:
    """Viscous damper to ground at a node: tau = -c * omega_i"""
    def __init__(self, node, c_visc):
        self.node = int(node)
        self.c = float(c_visc)

    def torques(self, thetas, omegas):
        return {self.node: -self.c * omegas[self.node]}

class LinkElement:
    """
    Connects node i -> j with a torque model providing t(theta_rel, theta_rel_dot).
    IMPORTANT SIGN (fixed): tau_i = +t, tau_j = -t  (restoring)
    """
    def __init__(self, i, j, torque_model: NonlinearSpringDamperLookup):
        self.i = int(i)
        self.j = int(j)
        self.model = torque_model

    def torques(self, thetas, omegas):
        theta_rel = thetas[self.j] - thetas[self.i]
        theta_rel_rate = omegas[self.j] - omegas[self.i]
        t = self.model.torque(theta_rel, theta_rel_rate)
        # Restoring pair:
        return {self.i: +t, self.j: -t}

# =========================
# System
# =========================
class DrivelineSystem:
    """
    N-node torsional chain (or graph if you add more links).
    State y = [theta_0..theta_N-1, omega_0..omega_N-1]
    """
    def __init__(self, inertias):
        self.J = np.asarray(inertias, float)
        self.N = len(self.J)
        self.elements = []          # link elements
        self.ground_dampers = []    # viscous to ground
        # per-node external torque functions f_i(t) -> Nm
        self.t_ext = [lambda t, i=i: 0.0 for i in range(self.N)]

    def add_element(self, element):
        self.elements.append(element)

    def add_ground_damper(self, damper):
        self.ground_dampers.append(damper)

    def set_external_torque(self, node, func):
        self.t_ext[node] = func

    def rhs(self, t, y):
        thetas = y[:self.N]
        omegas = y[self.N:]
        tau = np.zeros(self.N)

        # Links
        for e in self.elements:
            contrib = e.torques(thetas, omegas)
            for k, v in contrib.items():
                tau[k] += v

        # Ground dampers
        for d in self.ground_dampers:
            contrib = d.torques(thetas, omegas)
            for k, v in contrib.items():
                tau[k] += v

        # External
        for i in range(self.N):
            tau[i] += float(self.t_ext[i](t))

        domega = tau / self.J
        dtheta = omegas
        return np.hstack([dtheta, domega])

    def simulate(self, t_span, y0, method="RK45", max_step=1e-3, rtol=1e-7, atol=1e-9):
        return solve_ivp(self.rhs, t_span, y0, method=method,
                         max_step=max_step, rtol=rtol, atol=atol)

# =========================
# Demo / sanity check
# =========================
if __name__ == "__main__":
    # --- Choose your curve source ---
    CSV_PATH = None  # e.g. CSV_PATH = "damper_curve.csv"
    if CSV_PATH:
        ang_deg, tor_nm = load_curve_from_csv(CSV_PATH)
    else:
        ang_deg, tor_nm = make_demo_curve()  # synthetic curve

    # Nonlinear link (add viscous damping across the link if you like)
    link_model = NonlinearSpringDamperLookup(ang_deg, tor_nm, damping_viscous=6.0)

    # Two inertias (engine side, wheel side)
    J1, J2 = 0.12, 0.18  # kg·m²
    sys = DrivelineSystem([J1, J2])
    sys.add_element(LinkElement(0, 1, link_model))
    sys.add_ground_damper(LinearDamperToGround(node=1, c_visc=2.5))  # optional wheel drag

    # External torque: engine step for 0.25 s
    def engine_torque(t):
        return 50.0 if t < 0.25 else 0.0
    sys.set_external_torque(0, engine_torque)

    # Initial state
    theta0 = np.zeros(sys.N)
    omega0 = np.zeros(sys.N)
    y0 = np.hstack([theta0, omega0])

    # Integrate
    t_end = 5.0
    sol = sys.simulate((0.0, t_end), y0)

    # Extract
    t = sol.t
    thetas = sol.y[:sys.N, :]
    omegas = sol.y[sys.N:, :]
    theta_rel = thetas[1] - thetas[0]
    omega_rel = omegas[1] - omegas[0]
    tau_link = np.array([link_model.torque(th, om) for th, om in zip(theta_rel, omega_rel)])

    # =========================
    # Plots
    # =========================
    plt.figure()
    plt.plot(t, thetas[0], label="theta_0")
    plt.plot(t, thetas[1], label="theta_1")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.legend()
    plt.title("Node angles")

    plt.figure()
    plt.plot(t, omegas[0], label="omega_0")
    plt.plot(t, omegas[1], label="omega_1")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend()
    plt.title("Node speeds")

    plt.figure()
    plt.plot(t, theta_rel, label="theta_rel = theta_1 - theta_0")
    plt.xlabel("Time [s]")
    plt.ylabel("Relative angle [rad]")
    plt.legend()
    plt.title("Relative twist across link")

    plt.figure()
    plt.plot(t, tau_link, label="tau_link (spring+damp)")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N·m]")
    plt.legend()
    plt.title("Torque across nonlinear link")

    # Show the torque-angle curve used
    plt.figure()
    plt.plot(ang_deg, tor_nm, "o-", label="Lookup curve")
    plt.xlabel("Angle [deg]")
    plt.ylabel("Torque [N·m]")
    plt.legend()
    plt.title("Damper torque–angle lookup")
    plt.show()