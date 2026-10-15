"""enginecalc -- small, transparent helpers for engine hand calculations.

Everything here is deliberately simple: closed-form crank-slider kinematics,
a single-zone Wiebe pressure model, crank-drive forces and a few strength
formulas. The point is that every line can be checked against the notebook
that introduces it -- this is a learning tool, not a design code.

Sign conventions (used in every notebook)
-----------------------------------------
theta : crank angle [rad], 0 = firing TDC, positive in the direction of rotation.
        A four-stroke cycle runs theta in [-2*pi, 2*pi):
        intake [-360, -180), compression [-180, 0), expansion [0, 180),
        exhaust [180, 360) degrees.
x     : piston displacement FROM TDC, positive towards the crank [m].
F     : axial forces positive when they push the piston towards the crank.
beta  : connecting rod angle, sin(beta) = lambda * sin(theta).

Units: SI throughout (m, kg, s, N, Pa, J). Notebooks convert for display only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Plot style (same look as the orthomount math primer)
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.labelsize": 9, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.9,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "font.size": 9,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "axes.spines.top": False, "axes.spines.right": False,
    })
    np.set_printoptions(precision=4, suppress=True, linewidth=110)


def tidy(ax, title=None, xlabel=None, ylabel=None, deg_ticks=False):
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if deg_ticks:
        lo, hi = ax.get_xlim()
        ax.set_xticks(np.arange(np.ceil(lo / 90) * 90, hi + 1, 90))
    return ax


def trapz(y, x):
    """np.trapezoid on NumPy >= 2, np.trapz on older versions."""
    f = getattr(np, "trapezoid", None) or np.trapz
    return f(y, x)


def rpm_to_omega(rpm: float) -> float:
    return rpm * 2.0 * np.pi / 60.0


# ---------------------------------------------------------------------------
#  Engine description
# ---------------------------------------------------------------------------
@dataclass
class Engine:
    """Geometry and masses of ONE cylinder's crank drive.

    The default values describe an *illustrative* 1000 cc, four-cylinder
    race engine (250 cc per cylinder, very over-square). They are round,
    plausible numbers for learning -- not data of any real engine.
    """
    bore: float = 0.081            # [m]
    stroke: float = 0.0485         # [m]
    rod_length: float = 0.105      # centre-to-centre [m]
    compression_ratio: float = 13.5
    n_cyl: int = 4

    # masses
    m_piston_group: float = 0.28   # piston + rings + pin + clips [kg]
    m_rod: float = 0.30            # complete con-rod incl. bolts, bearing shells [kg]
    rod_cg_from_big_end: float = 0.035   # distance big-end centre -> rod CG [m]
    J_rod_cg: float = 6.0e-4       # rod mass moment of inertia about its CG [kg m^2]
    m_rod_cap: float = 0.07        # big-end cap + bolts (for cap loading) [kg]

    # gas-side
    p_crankcase: float = 1.0e5     # [Pa]

    # -------- derived geometry
    @property
    def r(self) -> float:
        """Crank radius [m]."""
        return 0.5 * self.stroke

    @property
    def lam(self) -> float:
        """Crank-radius-to-rod-length ratio lambda = r / l [-]."""
        return self.r / self.rod_length

    @property
    def A_piston(self) -> float:
        return 0.25 * np.pi * self.bore ** 2

    @property
    def V_d(self) -> float:
        """Swept volume of one cylinder [m^3]."""
        return self.A_piston * self.stroke

    @property
    def V_c(self) -> float:
        """Clearance (combustion chamber) volume [m^3]."""
        return self.V_d / (self.compression_ratio - 1.0)

    # -------- con-rod mass split
    @property
    def m_rod_osc(self) -> float:
        """Rod mass lumped at the small end (statically equivalent two-mass model)."""
        return self.m_rod * self.rod_cg_from_big_end / self.rod_length

    @property
    def m_rod_rot(self) -> float:
        """Rod mass lumped at the big end."""
        return self.m_rod - self.m_rod_osc

    @property
    def m_osc(self) -> float:
        """Total reciprocating mass [kg]."""
        return self.m_piston_group + self.m_rod_osc

    def summary(self) -> str:
        rows = [
            ("bore x stroke", f"{self.bore*1e3:.1f} x {self.stroke*1e3:.1f} mm"),
            ("bore/stroke", f"{self.bore/self.stroke:.2f}"),
            ("rod length", f"{self.rod_length*1e3:.1f} mm"),
            ("lambda = r/l", f"{self.lam:.3f}"),
            ("swept volume / cyl", f"{self.V_d*1e6:.1f} cm^3"),
            ("total displacement", f"{self.V_d*self.n_cyl*1e6:.0f} cm^3"),
            ("compression ratio", f"{self.compression_ratio:.1f}"),
            ("piston group", f"{self.m_piston_group*1e3:.0f} g"),
            ("rod (osc / rot)", f"{self.m_rod*1e3:.0f} g  ({self.m_rod_osc*1e3:.0f} / {self.m_rod_rot*1e3:.0f} g)"),
            ("reciprocating mass", f"{self.m_osc*1e3:.0f} g"),
        ]
        w = max(len(a) for a, _ in rows)
        return "\n".join(f"{a:<{w}}  {b}" for a, b in rows)


# ---------------------------------------------------------------------------
#  Kinematics (notebook 01)
# ---------------------------------------------------------------------------
def rod_angle(eng: Engine, theta):
    """Con-rod angle beta [rad] (positive on the same side as the crank pin)."""
    return np.arcsin(eng.lam * np.sin(theta))


def piston_kinematics(eng: Engine, theta, omega: float = 1.0):
    """Exact piston displacement, velocity and acceleration.

    Returns x [m], v [m/s], a [m/s^2] for constant crank speed omega [rad/s].
    x is measured from TDC towards the crank. Derivation in notebook 01.
    """
    r, lam = eng.r, eng.lam
    s, c = np.sin(theta), np.cos(theta)
    D = np.sqrt(1.0 - (lam * s) ** 2)
    x = r * (1.0 - c) + eng.rod_length * (1.0 - D)
    dx = r * (s + lam * s * c / D)                                   # dx/dtheta
    ddx = r * (c + lam * ((c * c - s * s) / D + lam ** 2 * s * s * c * c / D ** 3))
    return x, dx * omega, ddx * omega ** 2


def piston_kinematics_series(eng: Engine, theta, omega: float = 1.0):
    """Classic 2nd-order approximation (first two terms of the series)."""
    r, lam = eng.r, eng.lam
    x = r * ((1 - np.cos(theta)) + 0.25 * lam * (1 - np.cos(2 * theta)))
    v = r * omega * (np.sin(theta) + 0.5 * lam * np.sin(2 * theta))
    a = r * omega ** 2 * (np.cos(theta) + lam * np.cos(2 * theta))
    return x, v, a


def rod_angular_kinematics(eng: Engine, theta, omega: float = 1.0):
    """Rod angular velocity and acceleration [rad/s], [rad/s^2]."""
    lam = eng.lam
    s, c = np.sin(theta), np.cos(theta)
    D = np.sqrt(1.0 - (lam * s) ** 2)
    beta_dot = lam * c / D * omega
    beta_ddot = -lam * s * (1 - lam ** 2) / D ** 3 * omega ** 2
    return beta_dot, beta_ddot


def cylinder_volume(eng: Engine, theta):
    x, _, _ = piston_kinematics(eng, theta)
    return eng.V_c + eng.A_piston * x


def dV_dtheta(eng: Engine, theta):
    _, dx, _ = piston_kinematics(eng, theta, omega=1.0)
    return eng.A_piston * dx


# ---------------------------------------------------------------------------
#  Thermodynamics (notebook 02)
# ---------------------------------------------------------------------------
R_AIR = 287.0          # [J/(kg K)]
LHV_GASOLINE = 43.0e6  # [J/kg]


def otto_efficiency(eps, gamma=1.4):
    return 1.0 - eps ** (1.0 - gamma)


def wiebe(theta, theta_soc, dtheta, a=5.0, m=2.0):
    """Wiebe mass-fraction-burned x_b(theta) and its derivative dx_b/dtheta."""
    y = np.clip((theta - theta_soc) / dtheta, 0.0, None)
    xb = 1.0 - np.exp(-a * y ** (m + 1))
    dxb = np.where(y > 0, a * (m + 1) / dtheta * y ** m * np.exp(-a * y ** (m + 1)), 0.0)
    return np.where(theta >= theta_soc, xb, 0.0), dxb


@dataclass
class Combustion:
    """Inputs of the single-zone Wiebe model (illustrative full-load values)."""
    p_ivc: float = 1.05e5       # trapped pressure at IVC (= BDC here) [Pa]
    T_ivc: float = 330.0        # trapped temperature [K]
    afr: float = 12.8           # air/fuel ratio (rich, full load) [-]
    lhv: float = LHV_GASOLINE   # [J/kg]
    eta_q: float = 0.75         # fraction of fuel energy released into the gas
                                # (lumps combustion efficiency AND wall heat loss)
    gamma: float = 1.30         # constant ratio of specific heats [-]
    soc_deg: float = -22.0      # start of combustion [deg, 0 = firing TDC]
    duration_deg: float = 50.0  # 0-99 % burn duration [deg]
    wiebe_a: float = 6.9        # a = 6.9 -> 99.9 % burned at end of duration
    wiebe_m: float = 2.0
    p_exhaust: float = 1.15e5   # exhaust-stroke pressure [Pa]
    p_intake: float = 0.95e5    # intake-stroke pressure [Pa]
    blowdown_deg: float = 40.0  # length of the EVO pressure blend [deg]


def trapped_mass(eng: Engine, comb: Combustion) -> float:
    V_bdc = eng.V_d + eng.V_c
    return comb.p_ivc * V_bdc / (R_AIR * comb.T_ivc)


def heat_per_cycle(eng: Engine, comb: Combustion) -> float:
    """Heat actually released into the gas per cycle and cylinder [J]."""
    m_air = trapped_mass(eng, comb)
    return m_air / comb.afr * comb.lhv * comb.eta_q


def pressure_trace(eng: Engine, comb: Combustion | None = None, n: int = 7201):
    """Cylinder pressure over one 720 deg cycle from a single-zone Wiebe model.

    Closed part (IVC = -180 deg .. EVO = +180 deg) integrates the first law
        dp/dtheta = (gamma-1)/V * dQ/dtheta - gamma * p / V * dV/dtheta
    The gas exchange strokes are crude constant-pressure levels with a short
    cosine 'blowdown' blend after EVO. Returns theta [rad], p [Pa], extras dict.
    """
    from scipy.integrate import solve_ivp

    comb = comb or Combustion()
    g = comb.gamma
    Q = heat_per_cycle(eng, comb)
    soc, dur = np.radians(comb.soc_deg), np.radians(comb.duration_deg)

    def rhs(th, y):
        p = y[0]
        V = cylinder_volume(eng, th)
        _, dxb = wiebe(th, soc, dur, comb.wiebe_a, comb.wiebe_m)
        return [(g - 1.0) / V * Q * dxb - g * p / V * dV_dtheta(eng, th)]

    theta = np.linspace(-2 * np.pi, 2 * np.pi, n)
    closed = (theta >= -np.pi) & (theta <= np.pi)
    th_c = theta[closed]
    sol = solve_ivp(rhs, (th_c[0], th_c[-1]), [comb.p_ivc], t_eval=th_c,
                    method="LSODA", rtol=1e-8, atol=1.0, max_step=np.radians(0.5))
    p = np.empty_like(theta)
    p[closed] = sol.y[0]
    p[theta < -np.pi] = comb.p_intake
    exh = theta > np.pi
    p_evo = sol.y[0][-1]
    blend = np.clip((theta[exh] - np.pi) / np.radians(comb.blowdown_deg), 0, 1)
    w = 0.5 * (1 + np.cos(np.pi * blend))
    p[exh] = comb.p_exhaust + (p_evo - comb.p_exhaust) * w

    V = cylinder_volume(eng, theta)
    xb, _ = wiebe(theta, soc, dur, comb.wiebe_a, comb.wiebe_m)
    W_ind = trapz(p, V)
    extras = dict(V=V, xb=xb, Q=Q, W_ind=W_ind, imep=W_ind / eng.V_d,
                  T=p * V / (trapped_mass(eng, comb) * R_AIR))
    return theta, p, extras


# ---------------------------------------------------------------------------
#  Crank-drive forces (notebook 03)
# ---------------------------------------------------------------------------
def crank_forces(eng: Engine, theta, p, omega: float):
    """All the classic crank-drive forces for one cylinder.

    Returns a dict of arrays (SI units):
      F_gas   gas force on the piston (net of crankcase pressure)
      F_inert inertia force of the reciprocating mass, -m_osc * a
      F_pist  F_gas + F_inert (axial, positive towards crank)
      F_rod   force along the rod (positive = compression)
      F_side  piston side (thrust) force on the liner
      F_tan   tangential force at the crank pin (positive = driving)
      F_rad   radial force at the crank pin from the rod (positive = towards crank centre)
      F_cent  centrifugal force of the rod's rotating mass (outwards, constant)
      T       torque on the crankshaft [N m]
    """
    _, _, a = piston_kinematics(eng, theta, omega)
    beta = rod_angle(eng, theta)
    F_gas = (p - eng.p_crankcase) * eng.A_piston
    F_inert = -eng.m_osc * a
    F = F_gas + F_inert
    F_rod = F / np.cos(beta)
    F_side = F * np.tan(beta)
    F_tan = F * np.sin(theta + beta) / np.cos(beta)
    F_rad = F * np.cos(theta + beta) / np.cos(beta)
    F_cent = eng.m_rod_rot * eng.r * omega ** 2
    return dict(F_gas=F_gas, F_inert=F_inert, F_pist=F, F_rod=F_rod,
                F_side=F_side, F_tan=F_tan, F_rad=F_rad, F_cent=F_cent,
                T=F_tan * eng.r, beta=beta)


def crankpin_load(eng: Engine, theta, forces):
    """Load ON the crank pin in a crank-fixed frame (radial, tangential).

    Returns (F_r_pin, F_t_pin): radial positive towards the crank centre, and
    the resultant magnitude. Rod centrifugal force pulls the pin outwards,
    so it subtracts from the radial component.
    """
    Fr = forces["F_rad"] - forces["F_cent"]
    Ft = forces["F_tan"]
    return Fr, Ft, np.hypot(Fr, Ft)


# ---------------------------------------------------------------------------
#  Strength helpers (notebook 04 onwards)
# ---------------------------------------------------------------------------
@dataclass
class Material:
    name: str
    E: float        # Young's modulus [Pa]
    Rp02: float     # 0.2 % proof stress [Pa]
    Rm: float       # ultimate tensile strength [Pa]
    sigma_w: float  # fully reversed (R=-1) fatigue strength, polished specimen [Pa]
    rho: float      # density [kg/m^3]


# Typical *textbook-level* values -- check the real material sheet before use.
MATERIALS = {
    "Ti-6Al-4V": Material("Ti-6Al-4V (annealed/STA)", 114e9, 880e6, 950e6, 510e6, 4430),
    "34CrNiMo6": Material("34CrNiMo6 QT", 210e9, 900e6, 1100e6, 500e6, 7850),
    "42CrMo4":   Material("42CrMo4 QT", 210e9, 750e6, 1000e6, 450e6, 7850),
    "AlSi12CuNiMg": Material("Al piston alloy, ~20 C", 80e9, 230e6, 300e6, 110e6, 2700),
    "2618A-T61": Material("2618A forged Al, ~20 C", 74e9, 330e6, 420e6, 125e6, 2760),
}


def i_section(B, H, t_flange, t_web):
    """Symmetric I-section with the web along H.

    Returns A, I_strong (bending about the axis perpendicular to H, i.e. the
    in-plane-of-rotation buckling axis when H lies in the plane of the crank),
    I_weak. Dimensions in m.
    """
    h_web = H - 2 * t_flange
    A = 2 * B * t_flange + h_web * t_web
    I_strong = (B * H ** 3 - (B - t_web) * h_web ** 3) / 12.0
    I_weak = (2 * t_flange * B ** 3 + h_web * t_web ** 3) / 12.0
    return A, I_strong, I_weak


def euler_buckling(E, I, L_eff):
    return np.pi ** 2 * E * I / L_eff ** 2


def johnson_buckling(E, I, A, L_eff, sigma_y):
    """Euler for slender columns, Johnson parabola for short ones. Returns P_cr [N]."""
    k = np.sqrt(I / A)
    slender = L_eff / k
    transition = np.sqrt(2 * np.pi ** 2 * E / sigma_y)
    if slender >= transition:
        return euler_buckling(E, I, L_eff)
    sig_cr = sigma_y - (sigma_y * slender / (2 * np.pi)) ** 2 / E
    return sig_cr * A


def goodman_safety(sigma_a, sigma_m, sigma_w, Rm):
    """Safety factor on the (modified) Goodman line: 1/S = sa/sw + sm/Rm."""
    return 1.0 / (sigma_a / sigma_w + max(sigma_m, 0.0) / Rm)


# ---------------------------------------------------------------------------
#  Drawing
# ---------------------------------------------------------------------------
def draw_crank_slider(ax, eng: Engine, theta: float, annotate: bool = True):
    """Schematic of the crank drive at crank angle theta (cylinder axis vertical).

    Crank centre at the origin, crank pin at (r sin(theta), r cos(theta)).
    Lengths are drawn in mm.
    """
    r, l = eng.r * 1e3, eng.rod_length * 1e3
    beta = float(rod_angle(eng, theta))
    pin = np.array([r * np.sin(theta), r * np.cos(theta)])
    wp = np.array([0.0, r * np.cos(theta) + l * np.cos(beta)])
    B = eng.bore * 1e3
    # liner
    for sx in (-1, 1):
        ax.plot([sx * B / 2] * 2, [l - r - 5, l + r + 45], color=GRID, lw=3)
    # piston
    ax.add_patch(plt.Rectangle((-B / 2 + 1, wp[1] - 12), B - 2, 34,
                               facecolor=SERIES[0], alpha=0.25, edgecolor=SERIES[0]))
    # crank circle, web, rod
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(t), r * np.sin(t), color=GRID, lw=1, ls="--")
    ax.plot([0, pin[0]], [0, pin[1]], color=SERIES[1], lw=4, solid_capstyle="round")
    ax.plot([pin[0], wp[0]], [pin[1], wp[1]], color=INK2, lw=5, solid_capstyle="round")
    for pt, c in ((np.zeros(2), INK), (pin, SERIES[1]), (wp, SERIES[0])):
        ax.plot(*pt, "o", color="white", mec=c, mew=2, ms=7, zorder=5)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    if annotate:
        ax.text(pin[0] / 2 + 3, pin[1] / 2, "r", color=SERIES[1], fontsize=10)
        mid = (pin + wp) / 2
        ax.text(mid[0] + 4, mid[1], "l", color=INK2, fontsize=10)
        ax.text(3, -12, r"$\theta$=%d°" % round(np.degrees(theta)), color=SERIES[1], fontsize=9)
        ax.text(wp[0] + 4, wp[1] - 6, r"$\beta$=%.1f°" % np.degrees(beta), color=INK2, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-B / 2 - 15, B / 2 + 15)
    ax.set_ylim(-r - 15, l + r + 45)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
