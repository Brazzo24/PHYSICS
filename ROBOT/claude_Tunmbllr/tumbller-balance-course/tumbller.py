"""tumbller.py - model, firmware-faithful controller and simulator for the
ELEGOO Tumbller self-balancing robot.  Shared by all notebooks of the course.

Conventions (used everywhere)
-----------------------------
x      wheel-axle position along the ground            [m]   (+ = forward)
theta  body tilt from the vertical                     [rad] (+ = leaning forward)
phi    wheel rotation angle = x / r  (no slip)         [rad]
u      motor command in PWM counts, -255 ... +255      (+ = drive wheels forward)
tau    total torque of both motors on the wheels       [Nm]

State vector  s = [x, x_dot, theta, theta_dot]

The module has four layers:
    1. Params / Gains            - the numbers
    2. dynamics() / linearize()  - the physics (notebook 1)
    3. KalmanAngle / FirmwareController - a line-by-line Python port of the
       Elegoo firmware control law in BalanceCar.h + KalmanFilter.cpp
    4. simulate() / animate()    - plant + sensors + firmware in one loop
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import matplotlib.pyplot as plt

G = 9.81
RAD2DEG = 180.0 / math.pi

# ----------------------------------------------------------------- style ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 1.8, "font.size": 9,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "axes.spines.top": False, "axes.spines.right": False,
        "animation.embed_limit": 40,
    })
    np.set_printoptions(precision=4, suppress=True, linewidth=110)


def tidy(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


# ------------------------------------------------------------ parameters ----
@dataclass(frozen=True)
class Params:
    """Physical parameters.  Source of each default value:

    [D] datasheet in the Elegoo tutorial folder (GA37-520 motor, MPU-6050)
    [F] read from the firmware
    [E] ESTIMATE - replace with your own measurement (notebook 4)
    """
    # --- body: everything that tilts (chassis, motor housings, battery, boards)
    m_body: float = 0.80        # [E] kg
    l: float = 0.040            # [E] m   axle -> body centre of mass
    I_body: float = 2.0e-3      # [E] kg m^2 about the body CoM (pitch axis)
    # --- wheels (both together)
    m_wheels: float = 0.070     # [E] kg
    r: float = 0.0335           # [E] m   wheel radius (67 mm wheel)
    I_wheels: float = 4.0e-5    # [E] kg m^2 ~ 1/2 m r^2
    # --- motors (both together), referred to the wheel side of the gearbox
    gear: float = 30.0          # [D] gear ratio
    I_rotor: float = 7.2e-4     # [E] kg m^2 = 2 * gear^2 * J_rotor (J ~ 4e-7)
    v_rated: float = 12.0       # [D] V
    stall_rated: float = 0.49   # [D] Nm per motor at v_rated (5 kg cm)
    w0_rated: float = 37.7      # [D] rad/s no-load at v_rated (360 rpm)
    v_batt: float = 8.0         # [E] V   2 Li-ion cells, 7.0 ... 8.4 V
    tau_coulomb: float = 0.0    # [E] Nm  dry friction in gearbox (0 = ideal)
    # --- sensors
    counts_per_rev: float = 780.0   # [E] encoder edges per WHEEL rev seen by the firmware
    h_imu: float = 0.060        # [E] m   IMU height above the axle
    # --- firmware timing
    dt: float = 0.005           # [F] s   balance loop period (MsTimer2, 5 ms)
    speed_div: int = 8          # [F] speed loop runs every 8th tick (40 ms)

    # derived motor constants: tau = k_m * u - b_m * (wheel speed rel. to body)
    @property
    def stall_total(self) -> float:
        """Stall torque of both motors at battery voltage and full PWM [Nm]."""
        return 2.0 * self.stall_rated * self.v_batt / self.v_rated

    @property
    def w0(self) -> float:
        """No-load wheel speed at battery voltage and full PWM [rad/s]."""
        return self.w0_rated * self.v_batt / self.v_rated

    @property
    def k_m(self) -> float:
        """Torque per PWM count [Nm/count]."""
        return self.stall_total / 255.0

    @property
    def b_m(self) -> float:
        """Back-EMF damping [Nm s/rad]: slope of the torque-speed line."""
        return self.stall_total / self.w0

    @property
    def N(self) -> float:
        """Encoder counts per radian of wheel rotation."""
        return self.counts_per_rev / (2.0 * math.pi)

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def summary(self) -> str:
        return (
            f"body   m = {self.m_body:.3f} kg   l = {self.l*1e3:.0f} mm   I = {self.I_body*1e3:.2f} g m^2\n"
            f"wheels m = {self.m_wheels:.3f} kg   r = {self.r*1e3:.1f} mm   I = {self.I_wheels*1e7:.0f} g cm^2\n"
            f"rotor inertia at wheel  I_r = {self.I_rotor*1e3:.2f} g m^2  "
            f"(= {self.I_rotor/self.r**2:.2f} kg equivalent mass!)\n"
            f"motors @ {self.v_batt:.1f} V: stall {self.stall_total:.2f} Nm, "
            f"no-load {self.w0:.1f} rad/s = {self.w0*self.r:.2f} m/s\n"
            f"k_m = {self.k_m*1e3:.2f} mNm/count   b_m = {self.b_m*1e3:.1f} mNm s/rad\n"
            f"encoder {self.counts_per_rev:.0f} counts/wheel rev   loop {self.dt*1e3:.0f} ms"
        )


@dataclass(frozen=True)
class Gains:
    """Controller gains in FIRMWARE units - the numbers at the top of BalanceCar.h."""
    kp_balance: float = 55.0    # PWM per degree
    kd_balance: float = 0.75    # PWM per (degree/s)
    kp_speed: float = 10.0      # PWM per (encoder count per 40 ms)
    ki_speed: float = 0.26      # PWM per accumulated count

    def with_(self, **kw) -> "Gains":
        return replace(self, **kw)


STOCK = Gains()


# ------------------------------------------------------------- physics ------
def mass_matrix(p: Params, theta: float = 0.0):
    """Entries of the 2x2 mass matrix for q = [x, theta]."""
    m11 = p.m_body + p.m_wheels + (p.I_wheels + p.I_rotor) / p.r**2
    m12 = p.m_body * p.l * math.cos(theta) - p.I_rotor / p.r
    m22 = p.I_body + p.m_body * p.l**2 + p.I_rotor
    return m11, m12, m22


def motor_torque(p: Params, u: float, w_rel: float) -> float:
    """Total motor torque on the wheels [Nm].

    u      PWM command, saturated to +-255
    w_rel  wheel speed relative to the body = x_dot/r - theta_dot  [rad/s]
    """
    u = max(-255.0, min(255.0, u))
    tau = p.k_m * u - p.b_m * w_rel
    if p.tau_coulomb:
        tau -= p.tau_coulomb * math.tanh(w_rel / 0.5)
    return tau


def dynamics(p: Params, s, u: float, f_push: float = 0.0):
    """Nonlinear equations of motion.  Returns ds/dt for s = [x, v, theta, omega].

        m11 x'' + m12 th'' - m l sin(th) th'^2 =  tau / r + F
        m12 x'' + m22 th'' - m g l sin(th)     = -tau     + F l cos(th)
    """
    x, v, th, om = s
    sn, cs = math.sin(th), math.cos(th)
    m11, m12, m22 = mass_matrix(p, th)
    tau = motor_torque(p, u, v / p.r - om)
    f1 = tau / p.r + p.m_body * p.l * sn * om * om + f_push
    f2 = -tau + p.m_body * G * p.l * sn + f_push * p.l * cs
    det = m11 * m22 - m12 * m12
    a = (m22 * f1 - m12 * f2) / det
    al = (m11 * f2 - m12 * f1) / det
    return (v, a, om, al)


def linearize(p: Params):
    """Linear model about upright:  s' = A s + B u   (u in PWM counts)."""
    m11, m12, m22 = mass_matrix(p, 0.0)
    Minv = np.linalg.inv(np.array([[m11, m12], [m12, m22]]))
    b, r = p.b_m, p.r
    #               x   v        theta                 omega
    F = np.array([[0, -b / r**2, 0,                    b / r],
                  [0,  b / r,    p.m_body * G * p.l,  -b]])
    E = np.array([[p.k_m / r], [-p.k_m]])
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[2, 3] = 1.0
    A[[1, 3], :] = Minv @ F
    B = np.zeros((4, 1))
    B[[1, 3], :] = Minv @ E
    return A, B


def discretize(A, B, dt):
    """Exact zero-order-hold discretisation:  s[k+1] = Ad s[k] + Bd u[k]."""
    from scipy.linalg import expm
    n = A.shape[0]
    M = np.zeros((n + B.shape[1], n + B.shape[1]))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


# ---------------------------------------------- firmware <-> physics --------
def firmware_to_state_feedback(p: Params, g: Gains):
    """Express the firmware control law as u = K @ [x, v, theta, omega].

    The firmware computes (BalanceCar.h)
        u = kp_b*angle_deg + kd_b*gyro_dps + kp_s*speed_counts + ki_s*sum(speed_counts)
    where speed_counts are encoder counts per 40 ms and the encoder sits on the
    motor, so it measures wheel rotation RELATIVE TO THE BODY (x/r - theta).
    """
    Ts = p.dt * p.speed_div
    N = p.N
    kx = g.ki_speed * N / p.r
    kv = g.kp_speed * N * Ts / p.r
    kth = g.kp_balance * RAD2DEG - g.ki_speed * N
    kom = g.kd_balance * RAD2DEG - g.kp_speed * N * Ts
    return np.array([kx, kv, kth, kom])


def state_feedback_to_firmware(p: Params, K) -> Gains:
    """Inverse of firmware_to_state_feedback (K as in u = +K @ s)."""
    Ts = p.dt * p.speed_div
    N = p.N
    kx, kv, kth, kom = [float(k) for k in np.ravel(K)]
    ki_s = kx * p.r / N
    kp_s = kv * p.r / (N * Ts)
    return Gains(kp_balance=(kth + ki_s * N) / RAD2DEG,
                 kd_balance=(kom + kp_s * N * Ts) / RAD2DEG,
                 kp_speed=kp_s, ki_speed=ki_s)


def lqr(p: Params, Q=None, R=1.0):
    """Continuous-time LQR.  Returns K such that u = +K @ s is the optimal law
    (sign already flipped to match the firmware convention)."""
    from scipy.linalg import solve_continuous_are
    A, B = linearize(p)
    if Q is None:
        Q = np.diag([200.0, 1.0, 2000.0, 1.0])
    Rm = np.atleast_2d(R)
    P = solve_continuous_are(A, B, Q, Rm)
    return -np.linalg.solve(Rm, B.T @ P).ravel()


def lifted_closed_loop(p: Params, g: Gains, speed_filter: float = 0.7):
    """Exact discrete closed loop of plant + firmware with ideal sensors.

    The firmware is multi-rate: balance terms every 5 ms, speed terms every
    40 ms.  Stepping the system one 40 ms frame at a time gives a constant
    matrix Phi with  z[k+1] = Phi z[k],  z = [x, v, theta, omega, speed_f, integral].
    The loop is stable when all eigenvalues of Phi lie inside the unit circle
    (one eigenvalue is exactly 1: absolute position x is never fed back).
    """
    A, B = linearize(p)
    Ad, Bd = discretize(A, B, p.dt)
    Kb = np.array([[0.0, 0.0, g.kp_balance * RAD2DEG, g.kd_balance * RAD2DEG]])
    Ai = Ad + Bd @ Kb                       # one tick with the balance law only
    # propagate over one frame: s_end = Sx s_0 + Sc c   (c = speed-loop output)
    Sx, Sc = np.eye(4), np.zeros((4, 1))
    for _ in range(p.speed_div):
        Sc = Ai @ Sc + Bd
        Sx = Ai @ Sx
    rel = np.array([[1.0 / p.r, 0.0, -1.0, 0.0]]) * p.N     # state -> counts
    cx, cc = rel @ (Sx - np.eye(4)), rel @ Sc               # counts in frame
    a = speed_filter
    c_of_z = np.array([[0, 0, 0, 0, g.kp_speed, g.ki_speed]])
    Phi = np.zeros((6, 6))
    Phi[:4, :4] = Sx
    Phi[:4, :] += Sc @ c_of_z
    row_cnt = np.hstack([cx, np.zeros((1, 2))]) + cc @ c_of_z
    Phi[4, :] = (1 - a) * row_cnt
    Phi[4, 4] += a
    Phi[5, :] = Phi[4, :]
    Phi[5, 5] += 1.0
    return Phi


def stability_margin(p: Params, g: Gains) -> float:
    """Largest |eigenvalue| of the lifted loop, ignoring the trivial one at 1.
    < 1 means stable; smaller means faster decay (per 40 ms frame)."""
    ev = np.linalg.eigvals(lifted_closed_loop(p, g))
    ev = np.delete(ev, np.argmin(np.abs(ev - 1.0)))
    return float(np.max(np.abs(ev)))


# ------------------------------------------------- firmware port ------------
class KalmanAngle:
    """Port of KalmanFilter::Kalman_Filter (KalmanFilter.cpp).

    Two states: tilt angle [deg] and gyro bias [deg/s].
    Predict with the gyro, correct with the accelerometer angle.
    """

    def __init__(self, Q_angle=0.001, Q_gyro=0.005, R_angle=0.5):
        self.Q_angle, self.Q_gyro, self.R_angle = Q_angle, Q_gyro, R_angle
        self.angle = 0.0
        self.q_bias = 0.0
        self.P = [[1.0, 0.0], [0.0, 1.0]]
        self.K0 = 0.0

    def update(self, angle_m: float, gyro_m: float, dt: float) -> float:
        P = self.P
        self.angle += (gyro_m - self.q_bias) * dt            # predict
        err = angle_m - self.angle
        P[0][0] += (self.Q_angle - P[0][1] - P[1][0]) * dt
        P[0][1] += -P[1][1] * dt
        P[1][0] += -P[1][1] * dt
        P[1][1] += self.Q_gyro * dt
        E = self.R_angle + P[0][0]                           # correct
        K0, K1 = P[0][0] / E, P[1][0] / E
        t0, t1 = P[0][0], P[0][1]
        P[0][0] -= K0 * t0
        P[0][1] -= K0 * t1
        P[1][0] -= K1 * t0
        P[1][1] -= K1 * t1
        self.angle += K0 * err
        self.q_bias += K1 * err
        self.K0 = K0
        return self.angle


class FirmwareController:
    """Port of balanceCar() in BalanceCar.h - same variable names, same order.

    Call tick() once per 5 ms with what the hardware would deliver:
    accelerometer angle [deg], gyro rate [deg/s], encoder edges since last tick.
    Returns the PWM command (-255 ... 255).
    """

    def __init__(self, gains: Gains = STOCK, p: Params = Params(),
                 angle_zero=0.0, speed_enabled=True, fall_angle=22.0):
        self.g, self.p = gains, p
        self.kf = KalmanAngle()
        self.angle_zero = angle_zero
        self.speed_enabled = speed_enabled
        self.fall_angle = fall_angle
        self.setting_car_speed = 0.0
        self.encoder_pulse_num_speed = 0.0
        self.speed_control_period_count = 0
        self.speed_filter = 0.0
        self.car_speed_integral = 0.0
        self.speed_control_output = 0.0
        self.balance_control_output = 0.0
        self.pwm = 0.0
        self.stopped = False

    def tick(self, angle_acc_deg, gyro_dps, encoder_edges):
        g = self.g
        # encoder has ONE channel -> no direction; firmware borrows the PWM sign
        self.encoder_pulse_num_speed += -encoder_edges if self.pwm < 0 else encoder_edges
        angle = self.kf.update(angle_acc_deg, gyro_dps, self.p.dt)
        self.balance_control_output = (g.kp_balance * (angle - self.angle_zero)
                                       + g.kd_balance * gyro_dps)
        self.speed_control_period_count += 1
        if self.speed_control_period_count >= self.p.speed_div:
            self.speed_control_period_count = 0
            car_speed = self.encoder_pulse_num_speed
            self.encoder_pulse_num_speed = 0.0
            self.speed_filter = self.speed_filter * 0.7 + car_speed * 0.3
            self.car_speed_integral += self.speed_filter
            self.car_speed_integral += -self.setting_car_speed
            self.car_speed_integral = max(-3000.0, min(3000.0, self.car_speed_integral))
            self.speed_control_output = (-g.kp_speed * self.speed_filter
                                         - g.ki_speed * self.car_speed_integral)
        if not self.speed_enabled:
            self.speed_control_output = 0.0
        pwm = self.balance_control_output - self.speed_control_output
        pwm = max(-255.0, min(255.0, pwm))
        if abs(angle) > self.fall_angle:          # firmware: motion_mode = STOP
            self.stopped = True
        if self.stopped:
            pwm = 0.0
            self.car_speed_integral = 0.0
        self.pwm = pwm
        return pwm


# ------------------------------------------------------- simulation ---------
@dataclass
class SensorModel:
    """What the MPU-6050 and the encoders add to the truth."""
    ideal: bool = False
    acc_noise_g: float = 0.004       # accelerometer noise per sample [g]
    gyro_noise_dps: float = 0.10     # gyro noise per sample [deg/s]
    gyro_bias_dps: float = 0.0       # uncorrected gyro offset [deg/s]
    mount_offset_deg: float = 0.0    # IMU not square to the body / CoM not on the axis
    quantize_encoder: bool = True
    encoder_true_sign: bool = False  # True = pretend we had a quadrature encoder


class Result:
    """Simulation log - arrays sampled at the 5 ms control rate."""

    def __init__(self, p, g, log):
        self.p, self.g = p, g
        for k, v in log.items():
            setattr(self, k, np.asarray(v))
        self.theta_deg = self.theta * RAD2DEG
        self.fell = bool(self.fell_flag.any())

    def __repr__(self):
        return (f"<Result {self.t[-1]:.1f} s  fell={self.fell}  "
                f"max|theta|={np.max(np.abs(self.theta_deg)):.1f} deg  "
                f"x_end={self.x[-1]*100:.1f} cm>")


def _rk4(p, s, u, f, h):
    k1 = dynamics(p, s, u, f)
    k2 = dynamics(p, tuple(si + 0.5 * h * ki for si, ki in zip(s, k1)), u, f)
    k3 = dynamics(p, tuple(si + 0.5 * h * ki for si, ki in zip(s, k2)), u, f)
    k4 = dynamics(p, tuple(si + h * ki for si, ki in zip(s, k3)), u, f)
    return tuple(si + h / 6.0 * (a + 2 * b + 2 * c + d)
                 for si, a, b, c, d in zip(s, k1, k2, k3, k4))


def simulate(p: Params = Params(), gains: Gains = STOCK, t_end=6.0,
             theta0_deg=3.0, omega0_dps=0.0,
             push=None, speed_cmd=None, sensors: SensorModel | None = None,
             controller="firmware", K=None, angle_zero=0.0,
             speed_enabled=True, fall_angle=22.0, kf_warmup=5.0,
             substeps=5, seed=1) -> Result:
    """Closed-loop simulation: nonlinear plant + sensors + controller.

    push        (t_start, duration, force_N) horizontal push at CoM height
    speed_cmd   function t -> setting_car_speed (encoder counts per 40 ms)
    angle_zero  the firmware's angle_zero [deg]; a number or a function of t
    kf_warmup   seconds the robot is held still before t = 0 so the Kalman
                filter has converged (the firmware waits 2 s after power-up)
    controller  "firmware"  - FirmwareController with `gains`
                "state"     - ideal state feedback u = K @ s, held for 5 ms
                "none"      - motors unpowered (u = 0)
    """
    sensors = sensors or SensorModel()
    rng = np.random.default_rng(seed)
    ctrl = FirmwareController(gains, p, angle_zero(0.0) if callable(angle_zero) else angle_zero,
                              speed_enabled, fall_angle)
    ctrl.kf.angle = theta0_deg
    held = (theta0_deg, 0.0) if sensors.ideal else (
        theta0_deg + sensors.mount_offset_deg, sensors.gyro_bias_dps)
    for _ in range(int(kf_warmup / p.dt)):        # robot held still, filter settles
        ctrl.kf.update(held[0], held[1], p.dt)
    n = int(round(t_end / p.dt))
    h = p.dt / substeps
    s = (0.0, 0.0, theta0_deg / RAD2DEG, omega0_dps / RAD2DEG)
    rel_prev = s[0] / p.r - s[2]
    edge_prev = math.floor(rel_prev * p.N)
    keys = ("t x v theta omega u angle_est speed_filter integral balance_out "
            "speed_out angle_acc fell_flag").split()
    log = {k: [] for k in keys}
    u, acc_prev, fell = 0.0, (0.0, 0.0), False
    for i in range(n):
        t = i * p.dt
        x, v, th, om = s
        # ---------------- sensors ----------------
        a, al = acc_prev                               # accelerations at IMU
        sn, cs = math.sin(th), math.cos(th)
        hi = p.h_imu
        ax_w = a + hi * (al * cs - om * om * sn)
        az_w = hi * (-al * sn - om * om * cs) + G      # specific force, world
        f_fwd = ax_w * cs - az_w * sn
        f_up = ax_w * sn + az_w * cs
        if sensors.ideal:
            angle_acc, gyro = th * RAD2DEG, om * RAD2DEG
        else:
            f_fwd += rng.normal(0, sensors.acc_noise_g * G)
            f_up += rng.normal(0, sensors.acc_noise_g * G)
            angle_acc = math.atan2(-f_fwd, f_up) * RAD2DEG + sensors.mount_offset_deg
            gyro = om * RAD2DEG + sensors.gyro_bias_dps + rng.normal(0, sensors.gyro_noise_dps)
            gyro = round(gyro * 131.0) / 131.0         # 16-bit gyro LSB
        rel = x / p.r - th
        if sensors.quantize_encoder and not sensors.ideal:
            edge = math.floor(rel * p.N)
            edges = edge - edge_prev
            edge_prev = edge
        else:
            edges = (rel - rel_prev) * p.N
        rel_prev = rel
        if not (sensors.encoder_true_sign or sensors.ideal):
            edges = abs(edges)                         # single channel: magnitude only
        elif ctrl.pwm < 0:
            edges = -edges                             # undo the firmware's sign guess
        # ---------------- controller ----------------
        if speed_cmd is not None:
            ctrl.setting_car_speed = speed_cmd(t)
        if callable(angle_zero):
            ctrl.angle_zero = angle_zero(t)
        if controller == "firmware":
            u = ctrl.tick(angle_acc, gyro, edges)
        elif controller == "state":
            ctrl.kf.update(angle_acc, gyro, p.dt)
            u = float(np.dot(K, s))
            ctrl.speed_filter = (v / p.r - om) * p.N * p.dt * p.speed_div   # for the log only
            u = max(-255.0, min(255.0, u))
            if abs(th * RAD2DEG) > fall_angle:
                ctrl.stopped = True
            if ctrl.stopped:
                u = 0.0
        else:
            ctrl.kf.update(angle_acc, gyro, p.dt)
            u = 0.0
        if abs(th) > math.radians(75):
            fell = True
        for k_, val in zip(keys, (t, x, v, th, om, u, ctrl.kf.angle, ctrl.speed_filter,
                                  ctrl.car_speed_integral, ctrl.balance_control_output,
                                  ctrl.speed_control_output, angle_acc,
                                  fell or ctrl.stopped)):
            log[k_].append(val)
        if fell:                                       # lying on the ground
            continue
        # ---------------- plant ----------------
        f = push[2] if (push and push[0] <= t < push[0] + push[1]) else 0.0
        for _ in range(substeps):
            s = _rk4(p, s, u, f, h)
        d = dynamics(p, s, u, f)
        acc_prev = (d[1], d[3])
    return Result(p, gains, log)


def load_log(path):
    """Load a CSV recorded from the robot (tools/tumbller_logger.py).

    Returns a dict of arrays keyed by column name; 't' is added in seconds,
    starting at zero.  Comment lines (#) and broken lines are skipped.
    'events' lists (t, text) for every gain change the robot reported."""
    rows, names, events = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# ms="):             # "# ms=12345 kp_balance=..."
                head, _, rest = line[5:].partition(" ")
                events.append((float(head), rest))
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if names is None and not parts[0].lstrip("-").replace(".", "").isdigit():
                names = parts
                continue
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    width = max(set(map(len, rows)), key=[len(r) for r in rows].count)
    data = np.array([r for r in rows if len(r) == width])
    names = names or [f"c{i}" for i in range(width)]
    out = {n: data[:, i] for i, n in enumerate(names)}
    out["t"] = (out[names[0]] - out[names[0]][0]) / 1000.0
    out["events"] = [((ms - out[names[0]][0]) / 1000.0, text) for ms, text in events]
    return out


# ------------------------------------------------------------ plotting ------
def plot_run(res: Result, title=None, axs=None, label=None, show_est=False):
    """Four standard panels: tilt, position, PWM, speed-loop internals."""
    new = axs is None
    if new:
        fig, axs = plt.subplots(1, 4, figsize=(12.5, 2.7))
    axs[0].plot(res.t, res.theta_deg, label=label)
    if show_est:
        axs[0].plot(res.t, res.angle_est, lw=1, ls="--", color=INK2, label="Kalman estimate")
        axs[0].legend()
    axs[1].plot(res.t, res.x * 100, label=label)
    axs[2].plot(res.t, res.u, lw=1.0, label=label)
    axs[3].plot(res.t, res.speed_filter, label=label)
    if new:
        tidy(axs[0], title or "tilt angle", "t [s]", "theta [deg]")
        tidy(axs[1], "position", "t [s]", "x [cm]")
        tidy(axs[2], "motor command", "t [s]", "PWM [counts]")
        tidy(axs[3], "filtered speed", "t [s]", "counts / 40 ms")
        axs[2].set_ylim(-275, 275)
        axs[2].axhline(255, color=GRID, lw=1)
        axs[2].axhline(-255, color=GRID, lw=1)
        plt.tight_layout()
    return axs


def draw_robot(ax, p: Params, x, theta, color=SERIES[0], alpha=1.0):
    """Side view of the robot at position x [m], tilt theta [rad]."""
    r = p.r
    hgt, wid = 0.19, 0.045
    c, s = math.cos(theta), math.sin(theta)
    corners = np.array([[-wid / 2, -0.015], [wid / 2, -0.015],
                        [wid / 2, hgt - r], [-wid / 2, hgt - r], [-wid / 2, -0.015]])
    R = np.array([[c, s], [-s, c]])
    pts = corners @ R.T + np.array([x, r])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.6, alpha=alpha)
    ax.add_patch(plt.Circle((x, r), r, fc="white", ec=INK, lw=1.4, alpha=alpha, zorder=3))
    phi = x / r
    ax.plot([x, x + r * math.sin(phi)], [r, r + r * math.cos(phi)], color=INK, lw=1.2,
            alpha=alpha, zorder=4)
    ax.plot([x + p.l * s], [r + p.l * c], "o", ms=5, color=SERIES[1], alpha=alpha, zorder=5)


def filmstrip(res: Result, times, ax=None, spread=0.16, title=None):
    """Static 'comic strip' of a run: the robot drawn at the given times.

    Poses are drawn side by side (not at their true x) so they never overlap;
    the true position is written under each pose."""
    if ax is None:
        _, ax = plt.subplots(figsize=(min(12.5, 1.5 * len(times) + 1), 2.5))
    for k, tt in enumerate(times):
        i = min(int(round(tt / res.p.dt)), len(res.t) - 1)
        x0 = k * spread
        draw_robot(ax, res.p, x0, res.theta[i])
        ax.plot([x0, x0 + res.u[i] / 255 * 0.06], [0.215, 0.215], color=SERIES[7], lw=2.5)
        ax.text(x0, -0.012, f"t = {res.t[i]:.2f} s\n{res.theta_deg[i]:+.1f}°\nx = {res.x[i]*100:+.1f} cm",
                ha="center", va="top", fontsize=7.5, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_ylim(-0.085, 0.23)
    if title:
        ax.set_title(title + "   (red bar = motor command)", loc="left")
    return ax


def animate(res: Result, every=6, window=0.45, title=""):
    """Side-view animation of a simulation result.  In a notebook:

        from IPython.display import HTML
        HTML(tb.animate(res).to_jshtml())
    """
    from matplotlib.animation import FuncAnimation
    p = res.p
    idx = np.arange(0, len(res.t), every)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9), dpi=80,
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_ylim(-0.02, 0.26)
    ax.set_yticks([])
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    ax.axhline(0, color=INK2, lw=1)
    ticks = [ax.plot([k * 0.1, k * 0.1], [0, -0.008], color=INK2, lw=1)[0]
             for k in range(-60, 61)]
    body, = ax.plot([], [], color=SERIES[0], lw=2.2)
    wheel = plt.Circle((0, p.r), p.r, fc="white", ec=INK, lw=1.5, zorder=3)
    ax.add_patch(wheel)
    spoke, = ax.plot([], [], color=INK, lw=1.3, zorder=4)
    com, = ax.plot([], [], "o", ms=6, color=SERIES[1], zorder=5)
    arrow, = ax.plot([], [], color=SERIES[7], lw=2.5, zorder=6)
    txt = ax.text(0.02, 0.93, "", transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlabel("x [m]")
    ax.set_title(title, loc="left")
    ax2.plot(res.t, res.theta_deg, color=SERIES[0], label="tilt [deg]")
    ax2.plot(res.t, res.u / 25.5, color=SERIES[1], lw=1, label="PWM / 25.5")
    ax2.legend(loc="upper right")
    ax2.set_xlabel("t [s]")
    cursor = ax2.axvline(0, color=INK2, lw=1)
    hgt, wid = 0.19, 0.045
    corners = np.array([[-wid / 2, -0.015], [wid / 2, -0.015],
                        [wid / 2, hgt - p.r], [-wid / 2, hgt - p.r], [-wid / 2, -0.015]])
    plt.tight_layout()

    def frame(j):
        i = idx[j]
        x, th = res.x[i], res.theta[i]
        c, s = math.cos(th), math.sin(th)
        pts = corners @ np.array([[c, s], [-s, c]]).T + np.array([x, p.r])
        body.set_data(pts[:, 0], pts[:, 1])
        wheel.center = (x, p.r)
        phi = x / p.r
        spoke.set_data([x, x + p.r * math.sin(phi)], [p.r, p.r + p.r * math.cos(phi)])
        com.set_data([x + p.l * s], [p.r + p.l * c])
        arrow.set_data([x, x + res.u[i] / 255 * 0.12], [0.235, 0.235])
        ax.set_xlim(x - window, x + window)
        txt.set_text(f"t = {res.t[i]:4.2f} s   tilt = {res.theta_deg[i]:+5.1f} deg   PWM = {res.u[i]:+4.0f}")
        cursor.set_xdata([res.t[i], res.t[i]])
        return body, wheel, spoke, com, arrow, txt, cursor

    anim = FuncAnimation(fig, frame, frames=len(idx), interval=every * p.dt * 1000, blit=False)
    plt.close(fig)
    return anim
