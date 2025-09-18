import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# Curve helpers
# =========================
def load_curve_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if not {"angle_deg", "torque_Nm"} <= set(df.columns):
        raise ValueError("CSV must have columns: angle_deg, torque_Nm")
    df = df.sort_values("angle_deg").drop_duplicates("angle_deg")
    return df["angle_deg"].to_numpy(), df["torque_Nm"].to_numpy()

def make_demo_curve():
    angles_deg = np.array([-5,-4,-3,-2,-1.5,-1,-0.5,-0.2,-0.1,0,0.1,0.2,0.5,1,1.5,2,3,4,5], float)
    torque = []
    for a in angles_deg:
        torque.append(80.0*a if abs(a)<0.2 else (90.0 if abs(a)<2 else 100.0)*a)
    return angles_deg, np.array(torque, float)

# =========================
# Nonlinear link: spring(θ) + viscous c*θdot  (lookup in degrees)
# =========================
class NonlinearSpringDamperLookup:
    def __init__(self, angle_deg, torque_nm, damping_viscous=0.0, extrapolate=True):
        order = np.argsort(angle_deg)
        self.angle_deg = np.asarray(angle_deg, float)[order]
        self.torque_nm = np.asarray(torque_nm, float)[order]
        self.c = float(damping_viscous)  # N·m·s/rad
        self._interp = PchipInterpolator(self.angle_deg, self.torque_nm, extrapolate=extrapolate)

    def spring_torque(self, theta_rel_rad):
        return float(self._interp(np.degrees(theta_rel_rad)))

    def torque(self, theta_rel_rad, theta_rel_rate_rad_s):
        return self.spring_torque(theta_rel_rad) + self.c * float(theta_rel_rate_rad_s)

# =========================
# Elements
# =========================
class LinearDamperToGround:
    def __init__(self, node, c_visc):
        self.node = int(node)
        self.c = float(c_visc)
    def torques(self, t, thetas, omegas):
        return {self.node: -self.c * omegas[self.node]}

class LinkElement:
    """
    Connects node i -> j. Restoring sign convention:
      tau_i = +t(theta_rel), tau_j = -t(theta_rel),
      theta_rel = theta_j - theta_i.
    engage(t) in [0..1] scales the link (clutch ramp).
    """
    def __init__(self, i, j, torque_model: NonlinearSpringDamperLookup, engage=lambda t: 1.0):
        self.i = int(i); self.j = int(j); self.model = torque_model
        self.engage = engage

    def torques(self, t, thetas, omegas):
        theta_rel = thetas[self.j] - thetas[self.i]
        theta_rel_rate = omegas[self.j] - omegas[self.i]
        g = float(self.engage(t))
        tval = g * self.model.torque(theta_rel, theta_rel_rate)
        return {self.i: +tval, self.j: -tval}

# =========================
# System (supports time-varying inertias and state-dependent torques)
# =========================
class DrivelineSystem:
    def __init__(self, inertias):
        self.J_base = np.asarray(inertias, float)  # base (can be time-varying via J_fun)
        self.N = len(self.J_base)
        self.elements = []
        self.ground_dampers = []
        self.input_torque_funcs = []             # functions f(t)-> {node: tau}
        self.state_torque_funcs = []             # functions f(t,thetas,omegas)-> {node: tau}
        self.J_fun = None                        # optional (t,thetas,omegas)-> array J

    def add_element(self, e): self.elements.append(e)
    def add_ground_damper(self, d): self.ground_dampers.append(d)
    def add_input_torque(self, f): self.input_torque_funcs.append(f)
    def add_state_torque(self, f): self.state_torque_funcs.append(f)
    def set_time_varying_inertias(self, f): self.J_fun = f

    def rhs(self, t, y):
        thetas, omegas = y[:self.N], y[self.N:]
        tau = np.zeros(self.N)

        for e in self.elements:
            for k, v in e.torques(t, thetas, omegas).items(): tau[k] += v
        for d in self.ground_dampers:
            for k, v in d.torques(t, thetas, omegas).items(): tau[k] += v
        for f in self.input_torque_funcs:
            for k, v in f(t).items(): tau[k] += v
        for f in self.state_torque_funcs:
            for k, v in f(t, thetas, omegas).items(): tau[k] += v

        J = self.J_base if self.J_fun is None else np.asarray(self.J_fun(t, thetas, omegas), float)
        return np.hstack([omegas, tau / J])

    def simulate(self, t_span, y0, method="RK45", max_step=1e-3, rtol=1e-7, atol=1e-9):
        return solve_ivp(self.rhs, t_span, y0, method=method, max_step=max_step, rtol=rtol, atol=atol)

# =========================
# Vehicle side helpers (reflected to gearbox input, i.e., node 1)
# =========================
def J_vehicle_reflected(n_ratio, m_vehicle, R_wheel, J_wheel):
    # Input (gearbox) sees (J_wheel + mR^2) * n^2
    return (J_wheel + m_vehicle*R_wheel**2) * n_ratio**2

def road_load_input_torque(n_ratio, R_wheel, m_vehicle, CdA=0.6, rho=1.225, Crr=0.015, grade_deg=0.0):
    """
    Returns a state-dependent torque function applied at node 1 (gearbox input).
    Sign convention: resists motion.
    """
    def f(t, thetas, omegas):
        n = float(n_ratio(t))  # current ratio omega_input/omega_wheel
        omega_input = omegas[1]
        omega_w = omega_input / n
        v = omega_w * R_wheel

        sgn_v = 0.0 if abs(v) < 1e-6 else np.sign(v)
        F_roll  = Crr * m_vehicle * 9.81 * np.cos(np.radians(grade_deg)) * sgn_v
        F_aero  = 0.5 * rho * CdA * v * abs(v) * sgn_v
        F_grade = m_vehicle * 9.81 * np.sin(np.radians(grade_deg))  # + uphill (resist), - downhill
        F_long = F_roll + F_aero + F_grade

        T_wheel = -F_long * R_wheel
        T_input = T_wheel / n
        return {1: T_input}
    return f

def wheel_brake_input_torque(n_ratio, brake_wheel_Nm):
    """
    brake_wheel_Nm(t) returns a positive magnitude [N·m] at the WHEEL.
    We apply it opposite wheel rotation and reflect to input (node 1).
    """
    def f(t, thetas, omegas):
        n = float(n_ratio(t))
        Tb_mag = float(brake_wheel_Nm(t))
        omega_input = omegas[1]
        omega_w = omega_input / n
        sgn = 0.0 if abs(omega_w) < 1e-6 else -np.sign(omega_w)  # oppose motion
        T_wheel = sgn * Tb_mag
        return {1: T_wheel / n}
    return f

# =========================
# Convenience: piecewise gear ratio and clutch ramp
# =========================
def piecewise_constant_ratio(breakpoints):
    """
    breakpoints: list of (t_start, ratio) sorted by t_start.
    Returns n(t) function.
    """
    bp = sorted(breakpoints, key=lambda x: x[0])
    def n_of_t(t):
        r = bp[0][1]
        for ts, r_here in bp:
            if t >= ts: r = r_here
            else: break
        return r
    return n_of_t

def clutch_ramp_profile(stages):
    """
    stages: list of segments controlling link engagement g(t) in [0..1]
      Each entry: (t_start, t_end, g_start, g_end) with linear interpolation.
      Omitted time => constant 1.0 elsewhere.
    Example for a shift with open gap:
      stages = [
        (2.00, 2.08, 1.0, 0.0),  # ramp out
        (2.08, 2.15, 0.0, 0.0),  # open
        (2.15, 2.25, 0.0, 1.0),  # ramp in
      ]
    """
    def g(t):
        val = 1.0
        for t0, t1, g0, g1 in stages:
            if t < t0: continue
            if t0 <= t <= t1:
                s = (t - t0) / max(t1 - t0, 1e-12)
                return g0 + s*(g1 - g0)
            val = g1  # past segment end: hold last
        return val
    return g

# =========================
# Example usage with a single shift
# =========================
if __name__ == "__main__":
    # --- Damper curve (swap for CSV if you have one) ---
    ang_deg, tor_nm = make_demo_curve()
    link_model = NonlinearSpringDamperLookup(ang_deg, tor_nm, damping_viscous=6.0)

    # --- Vehicle / driveline params ---
    J_engine = 0.12        # engine rotating inertia [kg m^2]
    J_wheel  = 0.8         # wheel inertia [kg m^2]
    m_vehicle = 200.0      # kg
    R_wheel = 0.30         # m

    # --- Gear schedule: overall ratio AFTER the damper (node1 -> wheel) ---
    #   n = omega_input / omega_wheel
    #   1st gear to 2nd gear at t=2.0 s
    gear_bp = [
        (0.0, 10.0),   # 1st
        (2.0,  6.0),   # 2nd (becomes active at t>=2.0 s)
    ]
    n_ratio = piecewise_constant_ratio(gear_bp)

    # --- Clutch ramp during the shift (optional but recommended) ---
    stages = [
        (2.00, 2.08, 1.0, 0.0),  # ramp out
        (2.08, 2.15, 0.0, 0.0),  # fully open
        (2.15, 2.30, 0.0, 1.0),  # ramp in
    ]
    g_clutch = clutch_ramp_profile(stages)

    # --- Build system (engine node 0, gearbox input node 1) ---
    # J_base will be updated in time via J_fun.
    sys = DrivelineSystem([J_engine, 1.0])
    # time-varying inertia at node 1 (vehicle reflected to input side):
    def J_timevarying(t, thetas, omegas):
        return np.array([J_engine, J_vehicle_reflected(n_ratio(t), m_vehicle, R_wheel, J_wheel)])
    sys.set_time_varying_inertias(J_timevarying)

    # Nonlinear clutch/damper between engine and gearbox input
    sys.add_element(LinkElement(0, 1, link_model, engage=g_clutch))

    # Optional small drag on vehicle side
    sys.add_ground_damper(LinearDamperToGround(node=1, c_visc=1.0))

    # Road load + brake (state-dependent torques at node 1)
    sys.add_state_torque(road_load_input_torque(n_ratio, R_wheel, m_vehicle, CdA=0.6, Crr=0.015, grade_deg=0.0))
    # Example: no service brake in this run
    sys.add_state_torque(wheel_brake_input_torque(n_ratio, brake_wheel_Nm=lambda t: 0.0))

    # Engine torque (node 0): step then hold
    def engine_torque(t):
        return 70.0 if t < 2.0 else 40.0
    sys.add_input_torque(lambda t: {0: engine_torque(t)})

    # Initial state (start at 20 km/h)
    v0 = 20/3.6
    omega_input0 = (v0 / R_wheel) * n_ratio(0.0)   # gearbox input speed
    omega_engine0 = omega_input0                   # initially locked (no twist)
    y0 = np.array([0.0, 0.0, omega_engine0, omega_input0])

    # Integrate
    t_end = 6.0
    sol = sys.simulate((0.0, t_end), y0, max_step=5e-4)
    t = sol.t
    thetas = sol.y[:2, :]
    omegas = sol.y[2:, :]
    theta_rel = thetas[1] - thetas[0]
    omega_rel = omegas[1] - omegas[0]

    # Link torque time history
    tau_link = np.array([link_model.torque(th, om) * g_clutch(tt) for tt, th, om in zip(t, theta_rel, omega_rel)])

    # Build piecewise ratio signal for plotting
    n_plot = np.array([n_ratio(tt) for tt in t])

    # Plots
    plt.figure(); plt.title("Speeds")
    plt.plot(t, omegas[0], label="omega_engine [rad/s]")
    plt.plot(t, omegas[1], label="omega_input [rad/s]")
    plt.plot(t, omegas[1]/n_plot, label="omega_wheel [rad/s]")
    plt.xlabel("t [s]"); plt.ylabel("rad/s"); plt.legend()

    plt.figure(); plt.title("Vehicle speed")
    v = (omegas[1]/n_plot) * R_wheel
    plt.plot(t, v*3.6); plt.xlabel("t [s]"); plt.ylabel("km/h")

    plt.figure(); plt.title("Relative twist & link torque")
    plt.plot(t, theta_rel, label="theta_rel [rad]")
    plt.plot(t, tau_link, label="tau_link [N·m]")
    plt.xlabel("t [s]"); plt.legend()

    plt.figure(); plt.title("Gear ratio & clutch engagement")
    plt.plot(t, n_plot, label="gear ratio n(t)")
    g_plot = np.array([g_clutch(tt) for tt in t])
    plt.plot(t, g_plot, label="clutch engage g(t)")
    plt.xlabel("t [s]"); plt.legend()

    plt.show()
