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
        if abs(a) < 0.2:
            torque.append(80.0 * a)            # softer near zero
        else:
            torque.append((90.0 if abs(a) < 2 else 100.0) * a)
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
# Other elements
# =========================
class LinearDamperToGround:
    def __init__(self, node, c_visc):
        self.node = int(node)
        self.c = float(c_visc)
    def torques(self, thetas, omegas):
        return {self.node: -self.c * omegas[self.node]}

class LinkElement:
    """
    Connects node i -> j. Restoring sign convention:
      tau_i = +t(theta_rel), tau_j = -t(theta_rel),
      with theta_rel = theta_j - theta_i.
    """
    def __init__(self, i, j, torque_model: NonlinearSpringDamperLookup):
        self.i = int(i); self.j = int(j); self.model = torque_model
    def torques(self, thetas, omegas):
        theta_rel = thetas[self.j] - thetas[self.i]
        theta_rel_rate = omegas[self.j] - omegas[self.i]
        t = self.model.torque(theta_rel, theta_rel_rate)
        return {self.i: +t, self.j: -t}

# =========================
# System integrator
# =========================
class DrivelineSystem:
    def __init__(self, inertias):
        self.J = np.asarray(inertias, float)
        self.N = len(self.J)
        self.elements = []
        self.ground_dampers = []
        self.t_ext = [lambda t, i=i: 0.0 for i in range(self.N)]  # external torques

    def add_element(self, e): self.elements.append(e)
    def add_ground_damper(self, d): self.ground_dampers.append(d)
    def set_external_torque(self, node, func): self.t_ext[node] = func

    def rhs(self, t, y):
        thetas, omegas = y[:self.N], y[self.N:]
        tau = np.zeros(self.N)
        for e in self.elements:
            for k, v in e.torques(thetas, omegas).items(): tau[k] += v
        for d in self.ground_dampers:
            for k, v in d.torques(thetas, omegas).items(): tau[k] += v
        for i in range(self.N): tau[i] += float(self.t_ext[i](t))
        return np.hstack([omegas, tau / self.J])

    def simulate(self, t_span, y0, method="RK45", max_step=1e-3, rtol=1e-7, atol=1e-9):
        return solve_ivp(self.rhs, t_span, y0, method=method, max_step=max_step, rtol=rtol, atol=atol)

# =========================
# Vehicle model (reflected to engine side)
# =========================
def vehicle_equivalent_inertia_to_engine_side(m_vehicle, R_wheel, J_wheel, gear_ratio):
    """
    Reflect wheel + vehicle mass to the engine side.
    gear_ratio n = omega_engine / omega_wheel.
    J_vehicle_wheel = m * R^2 (translational mass to rotational at wheel).
    Reflected inertia to engine side: J_ref = (J_wheel + m R^2) * n^2
    """
    J_wheel_side = J_wheel + m_vehicle * R_wheel**2
    return (J_wheel_side) * gear_ratio**2

def road_load_torque_engine_side(omega_engine, gear_ratio, R_wheel, m_vehicle,
                                 CdA=0.6, rho=1.225, Crr=0.015, grade_deg=0.0):
    """
    Compute resistive torque on engine side from aero, rolling, and grade.
    Sign convention: torque resists motion.
    """
    n = gear_ratio
    omega_w = omega_engine / n
    v = omega_w * R_wheel  # m/s
    sgn_v = 0.0 if abs(v) < 1e-6 else np.sign(v)

    F_roll  = Crr * m_vehicle * 9.81 * np.cos(np.radians(grade_deg)) * sgn_v
    F_aero  = 0.5 * rho * CdA * v * abs(v) * sgn_v  # v*|v| ensures the sign resists motion
    F_grade = m_vehicle * 9.81 * np.sin(np.radians(grade_deg))  # + uphill (resist), - downhill (assist)

    F_long = F_roll + F_aero + F_grade
    T_wheel = -F_long * R_wheel               # negative = resisting
    T_engine_side = T_wheel / n               # reflect through ratio
    return T_engine_side

# =========================
# Scenarios
# =========================
def run_scenario(name, engine_torque_fn, brake_torque_wheel_fn,
                 sys, link_model, n_ratio, R_wheel, m_vehicle,
                 CdA=0.6, Crr=0.015, grade_deg=0.0,
                 t_end=6.0, v0=0.0):
    """
    - engine_torque_fn(t) on node 0 (engine side)
    - brake_torque_wheel_fn(t) -> wheel torque (positive = braking). Reflected to engine side.
    - Road load computed from omega and reflected automatically.
    """
    # External torque at node 0 = engine + road + brakes(reflected)
    def T0(t, cache={"last_omega0":0.0}):
        # We need current omega0; solve_ivp does not give it here, so we approximate
        # by reading from a closure set by an event. Instead, we will attach road load later
        # via a time-varying function that uses a global ω0 (updated in the integration loop).
        return 0.0

    sys.set_external_torque(0, T0)
    sys.set_external_torque(1, lambda t: 0.0)  # no direct torque on vehicle node; everything at node 0 after reflection

    # initial state from v0
    omega_w0 = v0 / R_wheel
    omega_e0 = n_ratio * omega_w0
    y0 = np.array([0.0, 0.0, omega_e0, omega_e0])  # [theta0, theta1, omega0, omega1]
    # small: set thetas equal so initial twist is zero

    # Integrate with manual stepping so we can compute road/brake torques using current ω0
    dt = 1e-3
    t = np.arange(0.0, t_end+dt, dt)
    y = np.zeros((4, t.size))
    y[:,0] = y0

    # Energetics (optional)
    K = np.zeros(t.size)
    P_damp_link = np.zeros(t.size)  # instantaneous viscous power in link

    for k in range(t.size-1):
        th0, th1, om0, om1 = y[:,k]
        # Reflect road + brake torques
        T_road = road_load_torque_engine_side(om0, n_ratio, R_wheel, m_vehicle,
                                              CdA=CdA, Crr=Crr, grade_deg=grade_deg)
        T_brake_eng = (brake_torque_wheel_fn(t[k]) / n_ratio) if brake_torque_wheel_fn else 0.0
        T_engine = engine_torque_fn(t[k])
        # Apply net external on node 0 now that we know omega0
        sys.t_ext[0] = lambda _t, Tr=T_road, Tb=T_brake_eng, Te=T_engine: Te + Tr - Tb

        # One RK4 step (explicit) to keep it simple here
        def f(tt, yy): return sys.rhs(tt, yy)
        k1 = f(t[k], y[:,k])
        k2 = f(t[k]+0.5*dt, y[:,k] + 0.5*dt*k1)
        k3 = f(t[k]+0.5*dt, y[:,k] + 0.5*dt*k2)
        k4 = f(t[k]+dt,     y[:,k] + dt*k3)
        y[:,k+1] = y[:,k] + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

        # Energetics
        th0, th1, om0, om1 = y[:,k+1]
        theta_rel = th1 - th0
        omega_rel = om1 - om0
        tau_spring = link_model.spring_torque(theta_rel)
        # power in viscous part (always dissipative): c * ω_rel^2
        P_damp_link[k+1] = link_model.c * (omega_rel**2)
        # kinetic energy
        K[k+1] = 0.5*sys.J[0]*om0**2 + 0.5*sys.J[1]*om1**2

    thetas = y[:2,:]; omegas = y[2:,:]
    theta_rel = thetas[1] - thetas[0]
    omega_rel = omegas[1] - omegas[0]
    tau_link = np.array([link_model.torque(th, om) for th, om in zip(theta_rel, omega_rel)])
    v = (omegas[0] / n_ratio) * R_wheel

    # Plots
    plt.figure(); plt.title(f"{name}: speeds")
    plt.plot(t, omegas[0], label="omega_engine [rad/s]")
    plt.plot(t, omegas[0]/n_ratio, label="omega_wheel [rad/s]")
    plt.xlabel("t [s]"); plt.ylabel("rad/s"); plt.legend()

    plt.figure(); plt.title(f"{name}: vehicle speed")
    plt.plot(t, v*3.6); plt.xlabel("t [s]"); plt.ylabel("km/h")

    plt.figure(); plt.title(f"{name}: relative twist & link torque")
    plt.plot(t, theta_rel, label="theta_rel [rad]")
    plt.plot(t, tau_link, label="tau_link [N·m]")
    plt.xlabel("t [s]"); plt.legend()

    plt.figure(); plt.title(f"{name}: kinetic energy & viscous dissipation rate")
    plt.plot(t, K, label="Kinetic energy [J]")
    plt.plot(t, P_damp_link, label="Link viscous power [W]")
    plt.xlabel("t [s]"); plt.legend()

    return t, thetas, omegas, tau_link, v, K, P_damp_link


# =========================
# Build system with vehicle equivalent inertia (engine side)
# =========================
# Curve (swap for your CSV)
ang_deg, tor_nm = make_demo_curve()
link_model = NonlinearSpringDamperLookup(ang_deg, tor_nm, damping_viscous=6.0)

# Driveline + vehicle
J_engine_side = 0.12           # engine-side rotating inertia [kg m^2]
J_wheel = 0.8                  # actual wheel rotational inertia [kg m^2]
m_vehicle = 200.0              # vehicle mass [kg]
R_wheel = 0.3                  # wheel effective radius [m]
gear_ratio = 8.0               # omega_engine / omega_wheel (overall)

J_vehicle_eq = vehicle_equivalent_inertia_to_engine_side(m_vehicle, R_wheel, J_wheel, gear_ratio)
J2 = J_vehicle_eq              # node 1 inertia = vehicle side reflected to engine side

sys = DrivelineSystem([J_engine_side, J2])
sys.add_element(LinkElement(0, 1, link_model))
# Optionally small losses on vehicle side (already reflected): keep modest
sys.add_ground_damper(LinearDamperToGround(node=1, c_visc=1.0))

# =========================
# Example scenarios
# =========================
# 1) Acceleration: throttle step to +70 Nm for 2 s, then hold 20 Nm
def T_engine_accel(t): return 70.0 if t < 2.0 else 20.0
def T_brake_zero(t): return 0.0

# 2) Deceleration: throttle closes (engine drag -20 Nm), plus 40 Nm brake at wheel from 1.0–3.0 s
def T_engine_decel(t): return -20.0
def T_brake_decel_wheel(t): return 40.0 if (1.0 <= t <= 3.0) else 0.0

# Start from 20 km/h
v0 = 20/3.6

run_scenario("ACCEL", T_engine_accel, T_brake_zero,
             sys, link_model, gear_ratio, R_wheel, m_vehicle,
             CdA=0.6, Crr=0.015, grade_deg=0.0, t_end=6.0, v0=v0)

run_scenario("DECEL", T_engine_decel, T_brake_decel_wheel,
             sys, link_model, gear_ratio, R_wheel, m_vehicle,
             CdA=0.6, Crr=0.015, grade_deg=0.0, t_end=6.0, v0=v0)

plt.show()
