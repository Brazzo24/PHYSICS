"""Force and moment 'input blocks' of the Hammerstein model (thesis Sec. 4.5-4.7).

The thesis keeps all non-linear force elements outside the multibody
equations: they are evaluated from the current state (mostly by table
look-up) and placed in the input vector u_in.  This module provides those
blocks.  The numbers quoted from the thesis are marked [T]; all spring,
damper, linkage and chain data are ESTIMATES (the thesis only shows them as
plots, Figs. 4.8 and 4.9) -- marked [E].
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .model import make_inputs, IIN, NIN
from .params import BikeParams

# --------------------------------------------------------------------------
# 4.5.1  suspension
# --------------------------------------------------------------------------
@dataclass
class ForkModel:
    """Telescopic fork: force as a function of extension q1 and speed u1.

    As in the thesis the characteristics are tabulated at initialisation and
    looked up at run time.  Returned force is positive when it EXTENDS the fork.
    """
    travel: float = 0.120          # [E] m
    k: float = 18.0e3              # [E] N/m, both legs
    preload: float = 0.022         # [E] m
    k_top: float = 20.0e3          # [E] top-out spring over the last 15 mm
    top_len: float = 0.015
    k_bottom: float = 400.0e3      # [E] hydraulic/bump stop over the last 12 mm
    bottom_len: float = 0.012
    c_comp: tuple = (900.0, 400.0)   # [E] N s/m low / high speed slope
    c_reb: tuple = (1800.0, 800.0)   # [E]
    v_knee: float = 0.15           # [E] m/s
    n_table: int = 241

    def __post_init__(self):
        # extension table extends beyond the travel so the stops act as very
        # stiff springs ("limits imposed by increasing the spring rates")
        self.q_tab = np.linspace(-0.02, self.travel + 0.02, self.n_table)
        x = self.travel - self.q_tab                         # compression
        F = self.k * (x + self.preload)
        F -= self.k_top * np.clip(self.top_len - x, 0, None)
        F += self.k_bottom * np.clip(x - (self.travel - self.bottom_len), 0, None)
        F -= self.k_bottom * np.clip(-x, 0, None)            # hard stop, full extension
        self.F_spring_tab = F
        self.v_tab = np.linspace(-2.0, 2.0, 401)             # extension speed
        self.F_damp_tab = np.array([self._damp(v) for v in self.v_tab])

    def _damp(self, v):
        lo, hi = (self.c_reb if v > 0 else self.c_comp)      # v>0: rebound
        a = abs(v)
        mag = lo * a if a <= self.v_knee else lo * self.v_knee + hi * (a - self.v_knee)
        return -np.sign(v) * mag

    def spring(self, q1):
        return np.interp(q1, self.q_tab, self.F_spring_tab)

    def damper(self, u1):
        return np.interp(u1, self.v_tab, self.F_damp_tab)

    def force(self, q1, u1):
        return self.spring(q1) + self.damper(u1)


@dataclass
class RearSuspension:
    """Rear shock acting through a linkage: torque about the swing-arm pivot
    as a function of swing-arm angle q2 and rate u2 (positive torque
    increases q2, i.e. extends the suspension).

    'Lever ratio' as defined in the thesis: shock speed / tangential speed
    of the rear axle.
    """
    Ra: float
    q2_top: float
    q2_bottom: float
    ratio_top: float = 0.44        # [E] lever ratio at full extension
    ratio_bottom: float = 0.56     # [E] ... fully compressed (rising rate)
    k: float = 98.0e3              # [E] N/m shock spring
    preload: float = 0.009         # [E] m
    k_bump: float = 500.0e3        # [E] bump rubber over the last 10 mm of shock
    bump_len: float = 0.010
    k_top: float = 300.0e3         # [E] top-out
    c_comp: tuple = (6.0e3, 2.0e3)    # [E] N s/m at the shock, low/high speed
    c_reb: tuple = (14.0e3, 5.0e3)    # [E]
    v_knee: float = 0.05           # [E] m/s shock speed
    n_table: int = 201

    def __post_init__(self):
        ext = 0.04
        self.q_tab = np.linspace(self.q2_bottom - ext, self.q2_top + ext, self.n_table)
        s = (self.q2_top - self.q_tab) / (self.q2_top - self.q2_bottom)   # 0..1
        self.ratio_tab = self.ratio_top + (self.ratio_bottom - self.ratio_top) * np.clip(s, 0, 1)
        # shock compression = integral of ratio * Ra d(q2) from full extension
        dq = np.gradient(self.q_tab)
        comp = np.cumsum((self.ratio_tab * self.Ra * dq)[::-1])[::-1]
        comp -= np.interp(self.q2_top, self.q_tab, comp)
        self.comp_tab = comp                                    # >0 compressed
        self.stroke = float(np.interp(self.q2_bottom, self.q_tab, comp))
        F = self.k * (comp + self.preload)
        F += self.k_bump * np.clip(comp - (self.stroke - self.bump_len), 0, None)
        F += self.k_top * np.clip(comp, None, 0)                # top-out stop
        self.T_spring_tab = F * self.ratio_tab * self.Ra
        # 2-D damping torque table indexed by q2 and SHOCK speed (thesis p. 49)
        self.vs_tab = np.linspace(-1.0, 1.0, 401)               # +: compression
        Fd = np.array([self._damp(v) for v in self.vs_tab])
        self.T_damp_tab = np.outer(self.ratio_tab * self.Ra, Fd)

    def _damp(self, v):
        lo, hi = (self.c_comp if v > 0 else self.c_reb)
        a = abs(v)
        mag = lo * a if a <= self.v_knee else lo * self.v_knee + hi * (a - self.v_knee)
        return np.sign(v) * mag            # force resisting the motion (ext. +)

    def lever_ratio(self, q2):
        return np.interp(q2, self.q_tab, self.ratio_tab)

    def spring_torque(self, q2):
        return np.interp(q2, self.q_tab, self.T_spring_tab)

    def damper_torque(self, q2, u2):
        # procedure of thesis p. 49: lever ratio -> shock speed -> table
        v_shock = -u2 * self.Ra * self.lever_ratio(q2)
        iq = np.interp(q2, self.q_tab, np.arange(len(self.q_tab)))
        i0 = int(np.clip(np.floor(iq), 0, len(self.q_tab) - 2))
        w = iq - i0
        row = (1 - w) * self.T_damp_tab[i0] + w * self.T_damp_tab[i0 + 1]
        return float(np.interp(v_shock, self.vs_tab, row))

    def torque(self, q2, u2):
        return self.spring_torque(q2) + self.damper_torque(q2, u2)


# --------------------------------------------------------------------------
# 4.5.2  chain
# --------------------------------------------------------------------------
@dataclass
class Chain:
    """Final drive geometry in the plane of the bike (Fig. 4.10).

    Co-ordinates are in the main frame axes (a1 forward, a2 up) relative to
    the swing-arm pivot P.  [E] 16/42 sprockets, 525 chain.
    """
    Ra: float
    r_front: float = 0.0407        # [E] 16 T, 15.875 mm pitch
    r_rear: float = 0.1062         # [E] 42 T
    xs: float = 0.075              # [E] drive sprocket centre, ahead of P
    zs: float = 0.010              # [E] ... above P

    def _runs(self, q2):
        """Tangent points (front, rear) of the upper and lower chain run."""
        cf = np.array([self.xs, self.zs])
        cr = -self.Ra * np.array([np.cos(q2), np.sin(q2)])      # rear axle
        d = cf - cr
        L = np.linalg.norm(d)
        t = d / L                                                # rear -> front
        n = np.array([-t[1], t[0]])                              # left normal = up
        out = {}
        for name, sgn in (("upper", 1.0), ("lower", -1.0)):
            # external tangent: offset both circles to the same side
            s = (self.r_rear - self.r_front) / L                 # sin of run angle
            c = np.sqrt(1 - s * s)
            nn = sgn * c * n + s * t                             # outward normal
            out[name] = (cf + self.r_front * nn, cr + self.r_rear * nn)
        return out

    def lever_arms(self, q2):
        """Perpendicular distance from the pivot P to the upper / lower run
        with the sign of the torque (about a3) that a unit TENSION in that
        run applies to the swing-arm."""
        arms = {}
        for name, (pf, pr) in self._runs(q2).items():
            d = (pf - pr) / np.linalg.norm(pf - pr)              # force on rear sprocket
            arms[name] = pr[0] * d[1] - pr[1] * d[0]             # (r x F)_3
        return arms

    def swingarm_torque(self, q2, wheel_torque):
        """Contribution to Tsay.  ``wheel_torque`` is the axle torque Trwy
        (negative = driving, positive = engine braking)."""
        arms = self.lever_arms(q2)
        tension = abs(wheel_torque) / self.r_rear
        return tension * (arms["upper"] if wheel_torque < 0 else arms["lower"])


# --------------------------------------------------------------------------
# 4.5.5  drag (coast-down fit)
# --------------------------------------------------------------------------
@dataclass
class Drag:
    """Coast-down polynomial a = p v^2 + q v + r, eq. (4.2) [T]."""
    p: float = -932.95e-6
    q: float = -2.91e-3
    r: float = -189.67e-3
    mass: float = 270.0            # [T] bike + rider
    rho: float = 1.235             # [T] kg/m^3 (10 C, 1004 mbar)
    Ks_front: float = -11.11       # [T] N per (kg m^2 * m/s^2)
    Ks_rear: float = -10.02        # [T]

    @property
    def CdA(self):                 # eq. (4.6):  1/2 rho CdA = -m p
        return -2.0 * self.mass * self.p / self.rho

    def aero(self, v):             # eq. (4.5), magnitude
        return 0.5 * self.rho * self.CdA * v * abs(v)

    def mechanical(self, v):       # eq. (4.7), F_r = m q v + m r  (negative)
        return self.mass * (self.q * v + self.r)

    def wheel_drag_torques(self, v, r_front, r_rear):
        """T_df, T_dr below eq. (4.7).  Returned as POSITIVE axle torques in
        the sign convention of the model (positive = decelerates a forward
        rolling wheel)."""
        Fr = -self.mechanical(v)
        return 0.1 * Fr * r_front, 0.9 * Fr * r_rear

    def inertial_shear(self, a, I_front, I_rear):
        """Longitudinal tyre shear force due to wheel inertia, F = Ks I a."""
        return self.Ks_front * I_front * a, self.Ks_rear * I_rear * a


# --------------------------------------------------------------------------
# 4.5.3  crankshaft
# --------------------------------------------------------------------------
def crank_torques(w_A_body, omega_crank, domega_crank, I_crank):
    """Torque of a transverse crankshaft on the main frame (Tmfx, Tmfy, Tmfz).

    w_A_body : angular velocity of the main frame in its own axes (a1,a2,a3)
    omega_crank : crank speed about a3 (negative = same sense as the wheels)
    Reaction of spinning the rotor up (about a3) plus gyroscopic torque
    -w x h with h = I_crank * omega_crank * a3.
    """
    h = np.array([0.0, 0.0, I_crank * omega_crank])
    gyro = -np.cross(w_A_body, h)
    return gyro[0], -I_crank * domega_crank + gyro[2], gyro[1]


# --------------------------------------------------------------------------
# 4.6  rider body position
# --------------------------------------------------------------------------
def rider_offset(ay, az, Ty, Z, MM, MA):
    """Lateral offset A_y of the main-frame mass centre, eq. (4.11)."""
    return MM / MA * (Ty + Z * ay / az)


def surrogate_accels(q9, v, yaw_rate, g=9.81):
    """Surrogate IMU channels a'_y, a'_z of Section 4.6."""
    ay = 1.9 * q9
    ac = v * yaw_rate
    ar = g / np.cos(np.arctan2(ac, g))
    return ay, np.sqrt(np.maximum(ar**2 - ay**2, 0.0))


# --------------------------------------------------------------------------
# placeholder until the tyre model of chapter 5
# --------------------------------------------------------------------------
@dataclass
class TyreVertical:
    """Linear radial spring/damper for the tyre deflections q3, q4.
    PLACEHOLDER [E] -- the Magic Formula model is the subject of chapter 5."""
    kf: float = 170.0e3
    kr: float = 185.0e3
    cf: float = 250.0
    cr: float = 250.0

    def forces(self, q3, q4, u3, u4):
        return self.kf * q3 + self.cf * u3, self.kr * q4 + self.cr * u4


# --------------------------------------------------------------------------
# assembly of the input vector for the in-plane force elements
# --------------------------------------------------------------------------
@dataclass
class ForceSet:
    fork: ForkModel
    rear: RearSuspension
    tyre: TyreVertical
    g: float = 9.81

    @classmethod
    def default(cls, p: BikeParams):
        return cls(ForkModel(travel=p.fork_travel),
                   RearSuspension(p.Ra, p.q2_top, p.q2_bottom),
                   TyreVertical(), p.g)

    def inputs(self, t, k, damping=True):
        """u_in containing gravity, suspension and vertical tyre forces."""
        q, u = k["q"], k["u"]
        d = 1.0 if damping else 0.0
        Fzf, Fzr = self.tyre.forces(q[2], q[3], d * u[2], d * u[3])
        return make_inputs(
            g=self.g,
            Ffork=self.fork.spring(q[0]) + d * self.fork.damper(u[0]),
            Tsay=self.rear.spring_torque(q[1]) + d * self.rear.damper_torque(q[1], u[1]),
            Ffwz=Fzf, Frwz=Fzr)
