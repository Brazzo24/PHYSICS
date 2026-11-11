"""
driveline.py - small torsional driveline / cranktrain model for the
"Sweeps, Sensitivity, Calibration, Surrogates & Optimisation" notebook series.

Everything is reduced to crankshaft speed (ratios are folded into the
inertias and stiffnesses), so the model is a simple torsional chain:

    [0] crank + alternator --k_prim-- [1] primary gear + clutch basket
        --k_damper, c_damper-- [2] clutch hub + gearbox input shaft
        --k_shaft-- [3] gearbox output + sprocket
        --k_cush, c_cush-- [4] rear wheel + dyno

Two models live here:

1. ``simulate``     linear, frequency domain, ~1 ms per run-up.
                    Used for sweeps, sensitivity, calibration (notebooks 01-03, 05, 06).
2. ``nl_simulate``  nonlinear, time domain (two-stage clutch damper with
                    friction hysteresis), ~1-3 s per run-up.
                    Stands in for an "expensive" Dymola / MBS model (notebook 04).

Only numpy is required.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Operating grid used throughout the notebooks
# --------------------------------------------------------------------------
ORDERS = np.array([0.5, 1.0, 1.5, 2.5, 3.0])       # engine orders we track
RPM = np.arange(2000.0, 10001.0, 100.0)            # run-up grid [1/min]
CHANNELS = ("crank", "gbx")                        # speed pick-ups on the bench


# --------------------------------------------------------------------------
# 1. Engine excitation from crank geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Engine:
    """Two-cylinder four-stroke; numbers are generic, not a specific engine."""
    bore: float = 0.090            # m
    stroke: float = 0.0686         # m
    conrod: float = 0.129          # m
    eps: float = 13.0              # compression ratio
    m_osc: float = 0.33            # oscillating mass per cylinder, kg
    firing: tuple = (0.0, 270.0)   # crank-angle offsets of the firing TDCs, deg
    gamma: float = 1.32            # polytropic exponent
    n_grid: int = 1440             # points per 720 deg (0.5 deg resolution)

    @property
    def r(self):
        return self.stroke / 2

    @property
    def area(self):
        return math.pi * self.bore ** 2 / 4

    @property
    def v_disp(self):
        return self.area * self.stroke


def crank_kinematics(eng: Engine, phi):
    """Piston position s (from crank centre) and its derivatives w.r.t. crank angle."""
    r, l = eng.r, eng.conrod
    sin, cos = np.sin(phi), np.cos(phi)
    root = np.sqrt(l ** 2 - (r * sin) ** 2)
    s = r * cos + root
    ds = -r * sin - r ** 2 * sin * cos / root
    d2s = (-r * cos - r ** 2 * (cos ** 2 - sin ** 2) / root
           - r ** 4 * sin ** 2 * cos ** 2 / root ** 3)
    return s, ds, d2s


def cylinder_pressure(eng: Engine, load: float = 1.0):
    """Synthetic single-zone pressure trace over one 720 deg cycle.

    phi runs from -360 deg to +360 deg, firing TDC at 0. Closed-cycle part is
    integrated from  dp/dphi = -gamma p V'/V + (gamma-1)/V dQ/dphi  with a
    Wiebe heat release.  ``load`` in (0, 1] scales intake pressure (throttle).
    """
    n = eng.n_grid
    phi = np.linspace(-2 * np.pi, 2 * np.pi, n, endpoint=False)
    s, ds, _ = crank_kinematics(eng, phi)
    v_c = eng.v_disp / (eng.eps - 1)
    vol = v_c + eng.area * (eng.r + eng.conrod - s)
    dvol = -eng.area * ds

    p_in = 0.30e5 + 0.65e5 * load
    p_ex = 1.05e5
    m_air = p_in * (v_c + eng.v_disp) / (287.0 * 330.0)
    q_tot = m_air / 14.7 * 43e6 * 0.74            # heat that ends up in the gas

    phi0, dphi_b, a_w, m_w = np.deg2rad(-18.0), np.deg2rad(58.0), 6.9, 2.0

    def dq(ph):
        x = (ph - phi0) / dphi_b
        if x <= 0 or x >= 1.3:
            return 0.0
        return q_tot * a_w * (m_w + 1) / dphi_b * x ** m_w * math.exp(-a_w * x ** (m_w + 1))

    def rhs(ph, pp):
        _, ds_, _ = crank_kinematics(eng, ph)
        s_ = crank_kinematics(eng, ph)[0]
        v_ = v_c + eng.area * (eng.r + eng.conrod - s_)
        dv_ = -eng.area * ds_
        return -eng.gamma * pp * dv_ / v_ + (eng.gamma - 1) / v_ * dq(ph)

    p = np.full(n, p_in)
    h = phi[1] - phi[0]
    i0 = n // 4            # -180 deg : intake valve closes (idealised)
    i1 = 3 * n // 4        # +180 deg
    for i in range(i0, i1):
        k1 = rhs(phi[i], p[i])
        k2 = rhs(phi[i] + h, p[i] + h * k1)
        p[i + 1] = p[i] + 0.5 * h * (k1 + k2)
    # exhaust: blend from the expansion line to exhaust back-pressure (blow-down)
    blend = np.clip((np.rad2deg(phi) - 130.0) / 60.0, 0, 1)
    blend = blend ** 2 * (3 - 2 * blend)
    p_exp = np.where(np.arange(n) <= i1, p, p_ex)
    p = (1 - blend) * p_exp + blend * p_ex
    del vol, dvol
    return phi, p


def torque_harmonics(eng: Engine, load: float = 1.0, orders=ORDERS):
    """Complex torque harmonics of the whole engine.

    Returns (G, I, T_mean):  total harmonic at crank speed Omega [rad/s] is
        T_hat(order) = gas_scale * G(order) + Omega**2 * I(order)
    G: gas torque (independent of speed in this simple model),
    I: oscillating-mass torque per Omega^2.  T_mean: mean indicated torque [Nm].
    """
    phi, p = cylinder_pressure(eng, load)
    _, ds, d2s = crank_kinematics(eng, phi)
    t_gas = -(p - 1.0e5) * eng.area * ds
    t_in = -eng.m_osc * d2s * ds              # times Omega^2
    n = len(phi)
    fg = np.fft.rfft(t_gas) * 2 / n
    fi = np.fft.rfft(t_in) * 2 / n
    orders = np.asarray(orders, float)
    k = np.rint(orders * 2).astype(int)        # FFT bin: order = k/2
    G = np.zeros(len(orders), complex)
    I = np.zeros(len(orders), complex)
    for d in eng.firing:
        ph = np.exp(-1j * orders * np.deg2rad(d))
        G += fg[k] * ph
        I += fi[k] * ph
    t_mean = len(eng.firing) * t_gas.mean()
    return G, I, t_mean


def torque_time_series(eng: Engine, load: float, rpm: float, gas_scale: float = 1.0):
    """Engine torque over one 720 deg cycle on the 0.5 deg grid (for the time-domain model)."""
    phi, p = cylinder_pressure(eng, load)
    _, ds, d2s = crank_kinematics(eng, phi)
    om = rpm * np.pi / 30
    t_one = -gas_scale * (p - 1.0e5) * eng.area * ds - eng.m_osc * om ** 2 * d2s * ds
    n = len(phi)
    tot = np.zeros(n)
    for d in eng.firing:
        tot += np.roll(t_one, int(round(d / 720.0 * n)))
    return tot


# --------------------------------------------------------------------------
# 2. Parameters
# --------------------------------------------------------------------------
# name, nominal ("CAD / data-sheet" value), lower & upper bound as multipliers, unit
PARAM_TABLE = [
    ("k_prim",    6.0e4, 0.50, 2.00, "Nm/rad"),    # crank web + primary gear mesh
    ("k_damper",  1.6e3, 0.50, 2.00, "Nm/rad"),    # clutch torsional damper
    ("k_shaft",   9.0e3, 0.50, 2.00, "Nm/rad"),    # gearbox shafts + meshes
    ("k_cush",    2.2e2, 0.50, 2.00, "Nm/rad"),    # chain + cush drive + tyre
    ("c_damper",  0.25,  0.20, 5.00, "Nms/rad"),   # clutch damper damping
    ("c_cush",    0.25,  0.20, 5.00, "Nms/rad"),   # cush drive damping
    ("J_gbx",     4.0e-3, 0.70, 1.40, "kgm2"),     # clutch hub + gearbox rotating parts
    ("gas_scale", 1.00,  0.80, 1.25, "-"),         # scale on gas-torque excitation
]
NAMES = [r[0] for r in PARAM_TABLE]
UNITS = {r[0]: r[4] for r in PARAM_TABLE}
NOMINAL = np.array([r[1] for r in PARAM_TABLE])
LOWER = NOMINAL * np.array([r[2] for r in PARAM_TABLE])
UPPER = NOMINAL * np.array([r[3] for r in PARAM_TABLE])
N_PAR = len(NAMES)

# things we treat as known (weighed / from CAD)
FIXED = dict(J_crank=2.0e-2, J_basket=6.0e-3, J_out=3.0e-3, J_wheel=0.25,
             eta_struct=0.02,      # loss factor of the "stiff" steel elements
             d_crank=0.02)         # bearing/oil drag on the crank, Nms/rad


def as_dict(x):
    """Parameter vector (physical units, order of NAMES) -> dict."""
    return dict(zip(NAMES, np.asarray(x, float)))


def to_unit(x):
    """Physical values -> [0, 1] cube, logarithmic between the bounds."""
    return (np.log(x) - np.log(LOWER)) / (np.log(UPPER) - np.log(LOWER))


def from_unit(u):
    """[0, 1] cube -> physical values (logarithmic scaling)."""
    return np.exp(np.log(LOWER) + np.asarray(u) * (np.log(UPPER) - np.log(LOWER)))


# --------------------------------------------------------------------------
# 3. Linear frequency-domain model
# --------------------------------------------------------------------------
_ENGINE = Engine()
_HARM_CACHE: dict = {}


def _harmonics(load, orders):
    key = (round(float(load), 6), tuple(np.asarray(orders, float)))
    if key not in _HARM_CACHE:
        _HARM_CACHE[key] = torque_harmonics(_ENGINE, load, orders)
    return _HARM_CACHE[key]


def _matrices(p, fixed, structure):
    """Inertia vector and list of springs (i, j, k, c_viscous, eta_hysteretic)."""
    J = [fixed["J_crank"], fixed["J_basket"], p["J_gbx"], fixed["J_out"], fixed["J_wheel"]]
    eta = fixed["eta_struct"]
    springs = [(0, 1, p["k_prim"], 0.0, eta),
               (1, 2, p["k_damper"], p["c_damper"], 0.0),
               (2, 3, p["k_shaft"], 0.0, eta),
               (3, 4, p["k_cush"], p["c_cush"], 0.0)]
    if structure:                       # optional "reality is more complicated" effects
        if "damper_eta" in structure:   # hysteretic instead of viscous clutch damper
            springs[1] = (1, 2, p["k_damper"], 0.0, structure["damper_eta"])
        if "J_alt" in structure:        # alternator rotor on a flexible stub shaft
            J[0] -= structure["J_alt"]
            J.append(structure["J_alt"])
            springs.append((0, len(J) - 1, structure["k_alt"], 0.0, structure.get("eta_alt", 0.05)))
    return np.array(J), springs


def natural_frequencies(x=NOMINAL, fixed=FIXED):
    """Undamped natural frequencies [Hz] and mode shapes of the free-free chain."""
    p = as_dict(x)
    J, springs = _matrices(p, fixed, None)
    n = len(J)
    K = np.zeros((n, n))
    for i, j, k, _, _ in springs:
        K[i, i] += k; K[j, j] += k; K[i, j] -= k; K[j, i] -= k
    Jm = 1 / np.sqrt(J)
    w2, v = np.linalg.eigh(Jm[:, None] * K * Jm[None, :])
    f = np.sqrt(np.clip(w2, 0, None)) / (2 * np.pi)
    shapes = Jm[:, None] * v
    shapes = shapes / np.abs(shapes).max(axis=0)
    return f[1:], shapes[:, 1:]


def simulate(x=NOMINAL, rpm=RPM, orders=ORDERS, load=1.0, fixed=FIXED,
             structure=None, full_output=False):
    """Order amplitudes of the steady-state torsional vibration over a run-up.

    Parameters
    ----------
    x : array (N_PAR,)  parameters in physical units, order of ``NAMES``
    rpm, orders : operating grid
    load : engine load 0..1

    Returns
    -------
    amp : array (2, n_orders, n_rpm)
        Speed-fluctuation amplitude [rad/s] at the crank (channel 0) and the
        gearbox input shaft (channel 1) -- what a pair of encoders would see.
    If ``full_output`` a dict with additional signals is returned as second value.
    """
    p = as_dict(x)
    rpm = np.atleast_1d(np.asarray(rpm, float))
    orders = np.asarray(orders, float)
    J, springs = _matrices(p, fixed, structure)
    n = len(J)
    G, I, t_mean = _harmonics(load, orders)

    om_e = rpm * np.pi / 30                                  # crank speed
    w = orders[:, None] * om_e[None, :]                      # (n_ord, n_rpm)
    Kc = np.zeros((n, n), complex)                           # K incl. hysteretic part
    C = np.zeros((n, n))
    for i, j, k, c, eta in springs:
        kc = k * (1 + 1j * eta)
        Kc[i, i] += kc; Kc[j, j] += kc; Kc[i, j] -= kc; Kc[j, i] -= kc
        C[i, i] += c; C[j, j] += c; C[i, j] -= c; C[j, i] -= c
    C[0, 0] += fixed["d_crank"]
    Z = (Kc[None, None] + 1j * w[..., None, None] * C[None, None]
         - (w ** 2)[..., None, None] * np.diag(J)[None, None])
    F = np.zeros(w.shape + (n,), complex)
    F[..., 0] = p["gas_scale"] * G[:, None] + om_e[None, :] ** 2 * I[:, None]
    th = np.linalg.solve(Z, F[..., None])[..., 0]            # (n_ord, n_rpm, n)
    vel = 1j * w[..., None] * th
    amp = np.stack([np.abs(vel[..., 0]), np.abs(vel[..., 2])])
    if not full_output:
        return amp
    k_d, c_d = springs[1][2], springs[1][3]
    eta_d = springs[1][4]
    t_damper = (k_d * (1 + 1j * eta_d) + 1j * w * c_d) * (th[..., 1] - th[..., 2])
    k_s = springs[2][2]
    t_shaft = k_s * (th[..., 2] - th[..., 3])
    extra = dict(theta=th, t_damper=np.abs(t_damper), t_shaft=np.abs(t_shaft),
                 acc_gbx=np.abs(w ** 2 * th[..., 2]), t_mean=t_mean * p["gas_scale"],
                 excitation=np.abs(F[..., 0]))
    return amp, extra


# --------------------------------------------------------------------------
# 4. Test-bench data (synthetic) - see make_bench_data.py for how it is produced
# --------------------------------------------------------------------------
def load_bench(name="A", folder=None):
    """Load a synthetic bench data set written by make_bench_data.py.

    Returns dict with rpm, orders, amp (2, n_ord, n_rpm), load.
    """
    import os
    folder = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    d = np.load(os.path.join(folder, f"bench_{name}.npz"))
    return {k: d[k] for k in d.files}


def residuals(x, bench, floor=0.05, **kw):
    """Log-amplitude residual vector between model and bench (flattened).

    r = ln(model + floor) - ln(bench + floor);  ``floor`` [rad/s] keeps tiny
    amplitudes (below the encoder noise floor) from dominating.
    """
    sim = simulate(x, rpm=bench["rpm"], orders=bench["orders"], load=float(bench["load"]), **kw)
    return (np.log(sim + floor) - np.log(bench["amp"] + floor)).ravel()


def cost(x, bench, **kw):
    """Scalar cost: root-mean-square log-amplitude error (0.1 ~ 10 % amplitude error)."""
    r = residuals(x, bench, **kw)
    return float(np.sqrt(np.mean(r ** 2)))


# --------------------------------------------------------------------------
# 5. Nonlinear time-domain model (the "expensive" one)
# --------------------------------------------------------------------------
NL_TABLE = [
    # name, nominal, lower, upper, unit  (absolute bounds here: these are DESIGN variables)
    ("k2",      1.6e3, 0.8e3, 3.2e3, "Nm/rad"),   # main-stage stiffness of the clutch damper
    ("phi1",    0.030, 0.005, 0.080, "rad"),      # end of the soft pre-stage
    ("T_hyst",  3.0,   0.5,   12.0,  "Nm"),       # friction hysteresis torque
    ("J_add",   0.0,   0.0,   4.0e-3, "kgm2"),    # extra inertia on the gearbox input side
]
NL_NAMES = [r[0] for r in NL_TABLE]
NL_NOMINAL = np.array([r[1] for r in NL_TABLE])
NL_LOWER = np.array([r[2] for r in NL_TABLE])
NL_UPPER = np.array([r[3] for r in NL_TABLE])
NL_RPM = np.arange(2000.0, 6001.0, 250.0)          # 17 speeds, where rattle matters


def nl_point(design, rpm, load=0.3, x=NOMINAL, fixed=FIXED, k1=250.0,
             t_settle=0.8, n_meas=6, return_series=False):
    """Steady-state time simulation of ONE operating point with the nonlinear damper.

    The dyno holds the wheel at constant speed, crank+basket are lumped (k_prim
    is treated as rigid), the clutch damper is two-stage with friction:

        T_d = k1*d                         for |d| <  phi1
            = sign(d)*(k1*phi1 + k2*(|d|-phi1))   otherwise
            + c_damper*d_dot + T_hyst*tanh(d_dot/0.5)

    Fixed-step RK4, 0.5 deg crank angle per step.  Returns dict of KPIs:
      acc_rms  rms angular acceleration of the gearbox input shaft [rad/s^2] ("rattle index")
      t_peak   peak clutch-damper torque [Nm]
      d_max    max damper wind-up angle [rad]
    """
    k2, phi1, t_h, j_add = [float(v) for v in design]
    p = as_dict(x)
    om = rpm * math.pi / 30
    tq = torque_time_series(_ENGINE, load, rpm, p["gas_scale"])
    n_tab = len(tq)
    t_mean = float(tq.mean())

    j0 = fixed["J_crank"] + fixed["J_basket"]
    j2 = p["J_gbx"] + j_add
    j3 = fixed["J_out"]
    c_d = p["c_damper"]
    k_s, c_s = p["k_shaft"], fixed["eta_struct"] * p["k_shaft"] / (2 * math.pi * 250.0)
    k_c, c_c = p["k_cush"], p["c_cush"]
    d_cr = fixed["d_crank"]
    tanh = math.tanh

    def damper(d, dd):
        ad = abs(d)
        if ad < phi1:
            f = k1 * d
        else:
            f = math.copysign(k1 * phi1 + k2 * (ad - phi1), d)
        return f + c_d * dd + t_h * tanh(dd / 0.5)

    def rhs(te, a0, a2, a3, w0, w2, w3):
        # angles are deviations from the rigid rotation Omega*t, speeds likewise
        td = damper(a0 - a2, w0 - w2)
        ts = k_s * (a2 - a3) + c_s * (w2 - w3)
        tc = k_c * a3 + c_c * w3
        return (w0, w2, w3,
                (te - td - d_cr * w0) / j0,
                (td - ts) / j2,
                (ts - tc) / j3), td

    # static wind-up under mean torque as initial condition
    a3 = t_mean / k_c
    a2 = a3 + t_mean / k_s
    wind = t_mean / k1 if t_mean < k1 * phi1 else phi1 + (t_mean - k1 * phi1) / k2
    a0 = a2 + wind
    y = [a0, a2, a3, 0.0, 0.0, 0.0]

    dt = (4 * math.pi / n_tab) / om
    cyc_t = 4 * math.pi / om
    n_settle = max(8, int(math.ceil(t_settle / cyc_t)))
    n_steps = (n_settle + n_meas) * n_tab // 2      # 2 table points per RK4 step
    i_meas = n_settle * n_tab // 2
    dt2 = 2 * dt
    acc = np.empty(n_steps - i_meas)
    tdl = np.empty(n_steps - i_meas)
    dl = np.empty(n_steps - i_meas)
    for i in range(n_steps):
        ia = (2 * i) % n_tab
        ta, tb, tc_ = tq[ia], tq[(ia + 1) % n_tab], tq[(ia + 2) % n_tab]
        k1_, td0 = rhs(ta, *y)
        k2_, _ = rhs(tb, *[y[j] + 0.5 * dt2 * k1_[j] for j in range(6)])
        k3_, _ = rhs(tb, *[y[j] + 0.5 * dt2 * k2_[j] for j in range(6)])
        k4_, _ = rhs(tc_, *[y[j] + dt2 * k3_[j] for j in range(6)])
        if i >= i_meas:
            acc[i - i_meas] = k1_[4]
            tdl[i - i_meas] = td0
            dl[i - i_meas] = y[0] - y[1]
        y = [y[j] + dt2 / 6 * (k1_[j] + 2 * k2_[j] + 2 * k3_[j] + k4_[j]) for j in range(6)]
    out = dict(acc_rms=float(np.sqrt(np.mean((acc - acc.mean()) ** 2))),
               t_peak=float(np.max(np.abs(tdl))), d_max=float(np.max(np.abs(dl))),
               t_mean=t_mean)
    if return_series:
        out.update(acc=acc, t_damper=tdl, d=dl, dt=dt2)
    return out


def nl_simulate(design, rpm=NL_RPM, load=0.3, **kw):
    """Run the nonlinear model over a set of speeds. Returns dict of arrays over rpm."""
    res = [nl_point(design, r, load=load, **kw) for r in np.atleast_1d(rpm)]
    return {k: np.array([r[k] for r in res]) for k in ("acc_rms", "t_peak", "d_max")}


def nl_kpi(design, rpm=NL_RPM, load=0.3, **kw):
    """The two numbers we care about from one expensive run-up:

    rattle : mean over the run-up of the gearbox-input angular-acceleration rms
             [rad/s^2] -- lower = less gear-rattle excitation
    windup : largest clutch-damper wind-up angle [rad] -- must stay below the
             damper's travel limit
    """
    res = nl_simulate(design, rpm, load, **kw)
    return np.array([np.mean(res["acc_rms"]), np.max(res["d_max"])])
