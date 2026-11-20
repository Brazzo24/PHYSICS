"""
geardyn.py -- small shared library for the gear-dynamics notebook series.

Contents
--------
GearPair            geometry, inertia, ISO-6336-style mesh stiffness, damping, time-varying k(s)
draw_pair           involute gear drawing with base circles and line of action
rigid_pair          rigid-mesh dynamics (notebook 01)
mesh_force_on_gear2 / shaft_bearing_reactions   statics of the reaction points
flank_force         unilateral spring-damper (contact with backlash)
Model               tiny MBS/FE assembler: dofs, masses, springs, bearings, gear meshes,
                    loads, fixed / prescribed dofs, optional modal reduction, energy bookkeeping
add_shaft           Euler-Bernoulli + torsion shaft finite elements (notebook 06)
simulate_dte        1-DOF dynamic-transmission-error model (notebook 03/04)
floquet_multipliers parametric stability of the linear time-varying mesh

Sign conventions (used in every notebook)
-----------------------------------------
* Gear 1 (pinion) at the origin, gear 2 at (a, 0). Shafts along +z.
* theta1 positive counter-clockwise, theta2 positive CLOCKWISE, so that in normal running
  both angles grow and theta2 = theta1 / i.
* Static transmission coordinate along the line of action (LOA):
      s = r_b1*theta1 - r_b2*theta2      (zero for perfect rigid rolling)
* Drive flank: gear 1 pushes gear 2 while turning CCW. Unit LOA vector of the force on gear 2
      n_d = ( sin a,  cos a )     reverse (coast) flank   n_r = ( sin a, -cos a )
* Mesh force F > 0 on the drive flank, F < 0 on the coast flank.
* Bearing reaction R = force the bearing exerts ON THE SHAFT (housing load is -R).
* SI units everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, field
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

# ---------------------------------------------------------------------------------------
# plotting style
# ---------------------------------------------------------------------------------------
C = dict(blue="#1f5fa8", red="#c0392b", green="#2e8b57", orange="#d97706",
         grey="#6b7280", purple="#7c3aed", black="#111827")


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.figsize": (8, 4.2), "figure.dpi": 110, "axes.grid": True,
        "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False, "lines.linewidth": 1.4,
        "axes.prop_cycle": mpl.cycler(color=[C["blue"], C["red"], C["green"],
                                             C["orange"], C["purple"], C["grey"]]),
    })


def inv(a):
    """involute function inv(a) = tan a - a"""
    return np.tan(a) - a


# ---------------------------------------------------------------------------------------
# the gear pair
# ---------------------------------------------------------------------------------------
@dataclass
class GearPair:
    """Spur gear pair. Illustrative numbers, not data of any real gearbox."""
    z1: int = 24
    z2: int = 48
    module: float = 2.5e-3
    alpha_deg: float = 20.0
    face_width: float = 15e-3
    addendum: float = 1.0          # h_a / m
    dedendum: float = 1.25         # h_f / m
    backlash: float = 0.0          # total normal backlash j_n measured along the LOA [m]
    c_prime: float = 14.0          # single tooth-pair stiffness, N/(mm*um)  (ISO 6336 default)
    zeta: float = 0.05             # mesh damping ratio (of the mesh mode)
    rho: float = 7850.0
    J1_extra: float = 0.0          # extra inertia rigidly attached to gear 1 [kg m^2]
    J2_extra: float = 0.0
    m1_extra: float = 0.0          # extra mass moving laterally with gear 1 [kg]
    m2_extra: float = 0.0
    tv_ratio: float = 1.6          # k(double contact) / k(single contact)
    tv_smooth: float = 0.03        # width of the single/double transition [base pitches]

    # ---- geometry ----------------------------------------------------------------------
    @property
    def alpha(self): return np.deg2rad(self.alpha_deg)
    @property
    def r1(self): return self.module * self.z1 / 2
    @property
    def r2(self): return self.module * self.z2 / 2
    @property
    def rb1(self): return self.r1 * np.cos(self.alpha)
    @property
    def rb2(self): return self.r2 * np.cos(self.alpha)
    @property
    def ra1(self): return self.r1 + self.addendum * self.module
    @property
    def ra2(self): return self.r2 + self.addendum * self.module
    @property
    def a(self): return self.r1 + self.r2
    @property
    def i(self): return self.z2 / self.z1
    @property
    def pb(self): return np.pi * self.module * np.cos(self.alpha)
    @property
    def path_length(self):
        return (np.sqrt(self.ra1**2 - self.rb1**2) + np.sqrt(self.ra2**2 - self.rb2**2)
                - self.a * np.sin(self.alpha))
    @property
    def eps_alpha(self): return self.path_length / self.pb

    # ---- mass properties (solid disc at pitch radius) ------------------------------------
    @property
    def m1(self): return self.rho * np.pi * self.r1**2 * self.face_width + self.m1_extra
    @property
    def m2(self): return self.rho * np.pi * self.r2**2 * self.face_width + self.m2_extra
    @property
    def J1(self):
        md = self.rho * np.pi * self.r1**2 * self.face_width
        return 0.5 * md * self.r1**2 + self.J1_extra
    @property
    def J2(self):
        md = self.rho * np.pi * self.r2**2 * self.face_width
        return 0.5 * md * self.r2**2 + self.J2_extra
    @property
    def m_eq(self):
        """equivalent mass of the relative motion along the LOA"""
        return 1.0 / (self.rb1**2 / self.J1 + self.rb2**2 / self.J2)

    # ---- stiffness and damping -------------------------------------------------------------
    @property
    def c_gamma(self):
        """ISO 6336 mean mesh stiffness per face width, N/(mm*um)"""
        return self.c_prime * (0.75 * self.eps_alpha + 0.25)
    @property
    def k_mesh(self):
        """mean mesh stiffness along the LOA [N/m]"""
        return self.c_gamma * (self.face_width * 1e3) * 1e6
    @property
    def c_mesh(self):
        return 2 * self.zeta * np.sqrt(self.k_mesh * self.m_eq)
    @property
    def f_n(self):
        """mesh natural frequency with both gears free [Hz]"""
        return np.sqrt(self.k_mesh / self.m_eq) / (2 * np.pi)

    def f_mesh(self, n1_rpm):
        """tooth-mesh frequency [Hz] for pinion speed n1 [rpm]"""
        return self.z1 * np.asarray(n1_rpm) / 60.0

    # ---- time-varying mesh stiffness k(s), s = mesh position in base pitches ------------------
    def _kparams(self):
        """(k_single, half-width h, smoothing sw, constant?) cached per parameter set"""
        key = (self.z1, self.z2, self.module, self.alpha_deg, self.face_width, self.addendum,
               self.c_prime, self.tv_ratio, self.tv_smooth)
        cache = self.__dict__.get("_kcache")
        if cache is not None and cache[0] == key:
            return cache[1]
        e = self.eps_alpha
        const = e >= 2.0 or self.tv_ratio == 1.0
        h = 0.5 * (min(e, 2.0) - 1.0); sw = max(self.tv_smooth, 1e-4)
        ks = self.k_mesh
        if not const:
            s = np.linspace(0, 1, 4001)[:-1]
            w, _ = self._box(s, h, sw)
            ks = self.k_mesh / np.mean(1 + (self.tv_ratio - 1) * w)
        p = (ks, h, sw, const)
        self.__dict__["_kcache"] = (key, p)
        return p

    @staticmethod
    def _box(s, h, sw):
        d = np.mod(s - h + 0.5, 1.0) - 0.5      # distance from the double-contact zone centre
        a, b = (d + h) / sw, (d - h) / sw
        w = 0.5 * (np.tanh(a) - np.tanh(b))
        dw = 0.5 / sw * (1 / np.cosh(a)**2 - 1 / np.cosh(b)**2)
        return w, dw

    @property
    def _k_single(self):
        return self._kparams()[0]

    def k_of_s(self, s):
        """mesh stiffness as a function of mesh position s (periodic, mean = k_mesh).
        Double contact for s mod 1 in [0, eps-1), single contact for the rest."""
        ks, h, sw, const = self._kparams()
        if np.isscalar(s):
            if const:
                return ks
            d = (s - h + 0.5) % 1.0 - 0.5
            return ks * (1 + (self.tv_ratio - 1) * 0.5 * (math.tanh((d + h) / sw) - math.tanh((d - h) / sw)))
        s = np.asarray(s, float)
        if const:
            return np.full_like(s, ks)
        w, _ = self._box(s, h, sw)
        return ks * (1 + (self.tv_ratio - 1) * w)

    def dk_ds(self, s):
        ks, h, sw, const = self._kparams()
        if const:
            return 0.0 * np.asarray(s, float)
        _, dw = self._box(np.asarray(s, float), h, sw)
        return ks * (self.tv_ratio - 1) * dw

    def phase(self, theta1):
        """mesh position in base pitches for pinion angle theta1"""
        return np.asarray(theta1) * self.z1 / (2 * np.pi)

    def with_(self, **kw):
        return replace(self, **kw)

    def summary(self):
        rows = [
            ("z1 / z2", f"{self.z1} / {self.z2}", "-"),
            ("ratio i = z2/z1", f"{self.i:.3f}", "-"),
            ("module", f"{self.module*1e3:.2f}", "mm"),
            ("pressure angle", f"{self.alpha_deg:.1f}", "deg"),
            ("pitch radii r1 / r2", f"{self.r1*1e3:.2f} / {self.r2*1e3:.2f}", "mm"),
            ("base radii rb1 / rb2", f"{self.rb1*1e3:.2f} / {self.rb2*1e3:.2f}", "mm"),
            ("centre distance a", f"{self.a*1e3:.2f}", "mm"),
            ("base pitch pb", f"{self.pb*1e3:.3f}", "mm"),
            ("contact ratio eps_alpha", f"{self.eps_alpha:.3f}", "-"),
            ("J1 / J2", f"{self.J1:.3e} / {self.J2:.3e}", "kg m^2"),
            ("m1 / m2", f"{self.m1:.3f} / {self.m2:.3f}", "kg"),
            ("equivalent mass m_eq", f"{self.m_eq:.4f}", "kg"),
            ("mesh stiffness k (mean)", f"{self.k_mesh/1e6:.1f}", "N/um"),
            ("mesh damping c", f"{self.c_mesh:.1f}", "N s/m"),
            ("mesh natural frequency", f"{self.f_n:.0f}", "Hz"),
            ("backlash j_n", f"{self.backlash*1e6:.0f}", "um"),
        ]
        w = max(len(r[0]) for r in rows)
        return "\n".join(f"{n:<{w}}  {v:>22}  {u}" for n, v, u in rows)


# ---------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------
def gear_outline(z, m, alpha_deg=20.0, ha=1.0, hf=1.25, n_flank=25):
    """closed (x, y) outline of an involute spur gear, tooth 0 centred on +x"""
    al = np.deg2rad(alpha_deg)
    r = m * z / 2; rb = r * np.cos(al); ra = r + ha * m; rf = r - hf * m
    psi = np.pi / (2 * z) + inv(al)             # half tooth angle at the base-circle line
    rr = np.linspace(max(rb, rf), ra, n_flank)
    ar = np.arccos(np.clip(rb / rr, -1, 1))
    ang = psi - inv(ar)                          # angular half-thickness at radius rr
    pts = []
    for k in range(z):
        c = 2 * np.pi * k / z
        # root -> right flank (negative side) -> tip -> left flank -> root
        th_r = c - ang; th_l = c + ang
        flank_r = np.c_[rr * np.cos(th_r), rr * np.sin(th_r)]
        flank_l = np.c_[rr * np.cos(th_l), rr * np.sin(th_l)][::-1]
        root_start = np.array([[rf * np.cos(c - ang[0]), rf * np.sin(c - ang[0])]])
        root_end = np.array([[rf * np.cos(c + ang[0]), rf * np.sin(c + ang[0])]])
        pts += [root_start, flank_r, flank_l, root_end]
        # root arc to next tooth
        t = np.linspace(c + ang[0], c + 2 * np.pi / z - ang[0], 6)
        pts.append(np.c_[rf * np.cos(t), rf * np.sin(t)])
    return np.vstack(pts)


def draw_pair(ax, gp: GearPair, theta1=0.0, show_loa=True, flank="drive", F=None,
              zoom=False, alpha_fill=0.15):
    """draw both gears in mesh, base circles, pitch point and line(s) of action"""
    o1 = gear_outline(gp.z1, gp.module, gp.alpha_deg, gp.addendum, gp.dedendum)
    o2 = gear_outline(gp.z2, gp.module, gp.alpha_deg, gp.addendum, gp.dedendum)

    def rot(p, t):
        c, s = np.cos(t), np.sin(t)
        return p @ np.array([[c, s], [-s, c]])

    # gear 2: a tooth gap of gear 2 must face +x of gear 1 tooth 0 at theta1 = 0
    t2 = np.pi + np.pi / gp.z2 - theta1 / gp.i
    p1 = rot(o1, theta1) * 1e3
    p2 = rot(o2, t2) * 1e3 + np.array([gp.a * 1e3, 0])
    ax.fill(p1[:, 0], p1[:, 1], color=C["blue"], alpha=alpha_fill, lw=0)
    ax.plot(p1[:, 0], p1[:, 1], color=C["blue"], lw=0.8)
    ax.fill(p2[:, 0], p2[:, 1], color=C["red"], alpha=alpha_fill, lw=0)
    ax.plot(p2[:, 0], p2[:, 1], color=C["red"], lw=0.8)
    t = np.linspace(0, 2 * np.pi, 400)
    for (cx, rb, rp, col) in [(0, gp.rb1, gp.r1, C["blue"]), (gp.a, gp.rb2, gp.r2, C["red"])]:
        ax.plot((cx + rb * np.cos(t)) * 1e3, rb * np.sin(t) * 1e3, "--", color=col, lw=0.7)
        ax.plot((cx + rp * np.cos(t)) * 1e3, rp * np.sin(t) * 1e3, ":", color=col, lw=0.7)
    al = gp.alpha
    if show_loa:
        flanks = ["drive", "coast"] if flank == "both" else [flank]
        for fl in flanks:
            sgn = 1 if fl == "drive" else -1
            # tangent points on the base circles
            T1 = gp.rb1 * np.array([np.cos(al), -sgn * np.sin(al)])
            T2 = np.array([gp.a, 0]) - gp.rb2 * np.array([np.cos(al), -sgn * np.sin(al)])
            ax.plot([T1[0] * 1e3, T2[0] * 1e3], [T1[1] * 1e3, T2[1] * 1e3],
                    color=C["green"] if fl == "drive" else C["orange"], lw=1.6,
                    label=f"line of action ({fl})")
            ax.plot([0, T1[0] * 1e3], [0, T1[1] * 1e3], color=C["grey"], lw=0.6)
            ax.plot([gp.a * 1e3, T2[0] * 1e3], [0, T2[1] * 1e3], color=C["grey"], lw=0.6)
    if F is not None:
        n = np.array([np.sin(al), np.cos(al) * np.sign(F)])
        P = np.array([gp.r1, 0]) * 1e3
        L = 0.35 * gp.r1 * 1e3
        ax.annotate("", xy=P + L * n, xytext=P, arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=2))
        ax.annotate("", xy=P - L * n, xytext=P, arrowprops=dict(arrowstyle="-|>", color=C["blue"], lw=2))
        ax.text(*(P + L * n * 1.1), "F on gear 2", color=C["red"], fontsize=8)
        ax.text(*(P - L * n * 1.35), "F on gear 1", color=C["blue"], fontsize=8)
    ax.plot([gp.r1 * 1e3], [0], "ko", ms=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
    if zoom:
        R = 3.5 * gp.module * 1e3
        ax.set_xlim(gp.r1 * 1e3 - R, gp.r1 * 1e3 + R); ax.set_ylim(-R, R)
    ax.grid(False)
    return ax


# ---------------------------------------------------------------------------------------
# rigid mesh (notebook 01)
# ---------------------------------------------------------------------------------------
def rigid_pair(gp: GearPair, T1, T2):
    """Rigid mesh, T1 drive torque on gear 1 (CCW), T2 load torque on gear 2 opposing its
    (CW) rotation. Returns (alpha1, alpha2, F) with F the normal mesh force (>0 drive flank)."""
    J_red = gp.J1 + gp.J2 / gp.i**2
    a1 = (np.asarray(T1) - np.asarray(T2) / gp.i) / J_red
    a2 = a1 / gp.i
    F = (np.asarray(T2) + gp.J2 * a2) / gp.rb2
    return a1, a2, F


def mesh_force_on_gear2(F, alpha):
    """(Fx, Fy) of the mesh force acting on gear 2 for signed normal force F"""
    F = np.asarray(F, float)
    return np.abs(F) * np.sin(alpha), F * np.cos(alpha)


def shaft_bearing_reactions(Fvec, P, zA, zB):
    """Statically determinate shaft on a locating bearing A and a floating bearing B.
    Fvec = (Fx, Fy, Fz) applied at P = (px, py, pz) (px, py relative to the shaft axis).
    Returns RA = (x, y, z), RB = (x, y, 0): forces the bearings exert ON THE SHAFT.
    The torque about z is carried by the shaft (drive/load), not by the bearings."""
    Fx, Fy, Fz = [np.asarray(v, float) for v in Fvec]
    px, py, pz = P
    L = zB - zA; h = pz - zA
    RBx = (px * Fz - h * Fx) / L
    RBy = (py * Fz - h * Fy) / L
    RAx = -Fx - RBx; RAy = -Fy - RBy; RAz = -Fz
    return np.array([RAx, RAy, RAz]), np.array([RBx, RBy, 0 * RBx])


# ---------------------------------------------------------------------------------------
# contact
# ---------------------------------------------------------------------------------------
def flank_force(pen, pen_rate, k, c):
    """unilateral spring-damper: force >= 0 only while pen > 0 (no 'sticky' tension)"""
    F = k * pen + c * pen_rate
    return np.where(pen > 0, np.maximum(F, 0.0), 0.0)


# ---------------------------------------------------------------------------------------
# Model: dofs + linear structure + nonlinear gear meshes
# ---------------------------------------------------------------------------------------
class GearMesh:
    """Mesh element between gear 1 dofs (x1, y1, th1) and gear 2 dofs (x2, y2, th2).
    th2 must be the CW-positive rotation of gear 2. x/y may be None (purely torsional)."""

    def __init__(self, model, gp, th1, th2, x1=None, y1=None, x2=None, y2=None,
                 stiffness="const", contact="backlash", gap=None, phase0=0.0, name="mesh"):
        self.gp, self.name = gp, name
        ix = model.index
        self.i_th1, self.i_th2 = ix(th1), ix(th2)
        self.lat = None if x1 is None else (ix(x1), ix(y1), ix(x2), ix(y2))
        self.stiffness = stiffness           # 'const' | 'tv'
        self.contact = contact               # 'linear' (bilateral spring) | 'backlash' (two flanks)
        self.gap = gp.backlash / 2 if gap is None else gap
        self.phase0 = phase0
        self.k0, self.c = gp.k_mesh, gp.c_mesh
        sa, ca = np.sin(gp.alpha), np.cos(gp.alpha)
        self.sa, self.ca = sa, ca

    # deflections of the two flanks (penetrations) and rates
    def kinematics(self, q, v):
        gp = self.gp
        s = gp.rb1 * q[self.i_th1] - gp.rb2 * q[self.i_th2]
        sd = gp.rb1 * v[self.i_th1] - gp.rb2 * v[self.i_th2]
        if self.lat is not None:
            a, b, c_, d = self.lat
            dx, dy = q[a] - q[c_], q[b] - q[d]
            dxd, dyd = v[a] - v[c_], v[b] - v[d]
        else:
            dx = dy = dxd = dyd = 0.0 * s
        pd = s + dx * self.sa + dy * self.ca - self.gap
        pr = -s + dx * self.sa - dy * self.ca - self.gap
        pdd = sd + dxd * self.sa + dyd * self.ca
        prd = -sd + dxd * self.sa - dyd * self.ca
        return s, pd, pr, pdd, prd

    def k_and_dk(self, th1):
        if self.stiffness == "tv":
            ph = self.gp.phase(th1) + self.phase0
            return self.gp.k_of_s(ph), self.gp.dk_ds(ph) * self.gp.z1 / (2 * np.pi)
        return self.k0 + 0 * th1, 0 * th1

    def evaluate(self, q, v):
        """returns dict with Fd, Fr (flank forces >= 0), F = Fd - Fr, strain energy,
        dissipated power and the generalized-force contributions"""
        k, dk = self.k_and_dk(q[self.i_th1])
        s, pd, pr, pdd, prd = self.kinematics(q, v)
        if self.contact == "linear":
            # bilateral spring on the drive LOA (tension allowed), no backlash
            dlt = pd + self.gap
            Fd = k * dlt + self.c * pdd
            Fr = 0 * Fd
            U = 0.5 * k * dlt**2
            P = self.c * pdd**2
            dUdth = 0.5 * dk * dlt**2
        else:
            Fd = flank_force(pd, pdd, k, self.c)
            Fr = flank_force(pr, prd, k, self.c)
            pdp, prp = np.maximum(pd, 0), np.maximum(pr, 0)
            U = 0.5 * k * (pdp**2 + prp**2)
            P = (Fd - k * pdp) * pdd + (Fr - k * prp) * prd
            dUdth = 0.5 * dk * (pdp**2 + prp**2)
        F = Fd - Fr
        gp = self.gp
        Q = {self.i_th1: -gp.rb1 * F - dUdth, self.i_th2: gp.rb2 * F}
        if self.lat is not None:
            a, b, c_, d = self.lat
            fx2 = (Fd + Fr) * self.sa             # force on gear 2
            fy2 = (Fd - Fr) * self.ca
            Q.update({a: -fx2, b: -fy2, c_: fx2, d: fy2})
        return dict(Fd=Fd, Fr=Fr, F=F, U=U, P=P, k=k, s=s, pd=pd, pr=pr, Q=Q)

    def g(self, n, flank="drive"):
        """gradient of the flank penetration w.r.t. the full dof vector (for linearization)"""
        g = np.zeros(n); gp = self.gp
        sg = 1 if flank == "drive" else -1
        g[self.i_th1] += sg * gp.rb1; g[self.i_th2] -= sg * gp.rb2
        if self.lat is not None:
            a, b, c_, d = self.lat
            g[a] += self.sa; g[c_] -= self.sa
            g[b] += sg * self.ca; g[d] -= sg * self.ca
        return g


@dataclass
class Bearing:
    name: str
    ix: int
    iy: int
    k: float
    c: float


class Model:
    """Small assembler for linear structures (masses, springs, shaft FE) coupled through
    nonlinear gear meshes. Equations: M a + C v + K q = Q_mesh(q, v, t) + Q_loads(t, q, v)."""

    def __init__(self):
        self.labels: list[str] = []
        self._M, self._K, self._C = [], [], []
        self.meshes: list[GearMesh] = []
        self.bearings: list[Bearing] = []
        self.loads = []
        self.fixed: set[int] = set()
        self.prescribed: dict[int, tuple] = {}
        self._compiled = None

    # -- building -----------------------------------------------------------------------------
    def add_dof(self, label):
        if label in self.labels:
            raise ValueError(f"dof {label} exists")
        self.labels.append(label); self._compiled = None
        return len(self.labels) - 1

    def index(self, label):
        return label if isinstance(label, (int, np.integer)) else self.labels.index(label)

    def add_mass(self, dof, m):
        i = self.index(dof); self._M.append((i, i, m)); self._compiled = None

    def add_matrix(self, dofs, K=None, M=None, Cm=None):
        idx = [self.index(d) for d in dofs]
        for store, A in ((self._K, K), (self._M, M), (self._C, Cm)):
            if A is None:
                continue
            for a, ia in enumerate(idx):
                for b, ib in enumerate(idx):
                    if A[a][b] != 0:
                        store.append((ia, ib, A[a][b]))
        self._compiled = None

    def add_spring(self, d1, d2, k, c=0.0):
        """spring (and damper) between two dofs, d2=None means ground"""
        if d2 is None:
            self.add_matrix([d1], K=[[k]], Cm=[[c]])
        else:
            A = np.array([[1, -1], [-1, 1]])
            self.add_matrix([d1, d2], K=k * A, Cm=c * A)

    def add_bearing(self, name, dx, dy, k, c=0.0):
        self.add_spring(dx, None, k, c); self.add_spring(dy, None, k, c)
        self.bearings.append(Bearing(name, self.index(dx), self.index(dy), k, c))

    def add_mesh(self, gp, th1, th2, **kw):
        m = GearMesh(self, gp, th1, th2, **kw); self.meshes.append(m); self._compiled = None
        return m

    def add_load(self, dof, func):
        """func(t, q, v) -> generalized force on dof (torque on a rotation dof)"""
        self.loads.append((self.index(dof), func)); self._compiled = None

    def clear_loads(self):
        self.loads = []; self._compiled = None

    def fix(self, *dofs):
        for d in dofs:
            self.fixed.add(self.index(d))
        self._compiled = None

    def release(self, *dofs):
        for d in dofs:
            self.fixed.discard(self.index(d)); self.prescribed.pop(self.index(d), None)
        self._compiled = None

    def prescribe(self, dof, f, fd, fdd):
        self.prescribed[self.index(dof)] = (f, fd, fdd); self._compiled = None

    # -- matrices ------------------------------------------------------------------------------
    @property
    def n(self): return len(self.labels)

    def matrices(self):
        n = self.n
        out = []
        for store in (self._M, self._K, self._C):
            A = np.zeros((n, n))
            for i, j, v in store:
                A[i, j] += v
            out.append(A)
        return out

    def mesh_stiffness_matrix(self, flank="drive"):
        n = self.n; Km = np.zeros((n, n))
        for m in self.meshes:
            g = m.g(n, flank); Km += m.k0 * np.outer(g, g)
        return Km

    def compile(self, f_cut=None):
        M, K, Cm = self.matrices()
        n = self.n
        cons = sorted(self.fixed | set(self.prescribed))
        f = np.array([i for i in range(n) if i not in cons], int)
        p = np.array([i for i in cons], int)
        Mff, Kff, Cff = M[np.ix_(f, f)], K[np.ix_(f, f)], Cm[np.ix_(f, f)]
        Psi = np.zeros((len(f), len(p)))
        if f_cut is None:
            Phi = np.eye(len(f)); freqs = None
        else:
            # Ritz basis: static constraint modes for the prescribed/fixed dofs (Psi, as in
            # Craig-Bampton) + normal modes below f_cut with those dofs clamped (Phi).
            # Phi is augmented with static 'attachment' shapes for the mesh-force directions
            # and the loaded dofs, so static deflections are represented exactly (residual
            # flexibility), then re-orthonormalized.
            Kall = K + self.mesh_stiffness_matrix()
            Kb = Kall[np.ix_(f, f)]
            w2, V = eigh(Kb, Mff)
            w = np.sqrt(np.clip(w2, 0, None))
            keep = w < 2 * np.pi * f_cut
            extra = [msh.g(n, fl)[f] for msh in self.meshes for fl in ("drive", "coast")]
            for i, _ in self.loads:
                if i in set(f):
                    e = np.zeros(n); e[i] = 1.0; extra.append(e[f])
            B = V[:, keep]
            if extra and keep.sum() < len(f):
                A = np.linalg.lstsq(Kb, np.array(extra).T, rcond=None)[0]
                A = A - B @ (B.T @ Mff @ A)                 # remove the part already in B
                nrm = np.sqrt(np.einsum("ij,ij->j", A, Mff @ A))
                A = A[:, nrm > 1e-12 * max(nrm.max(), 1e-300)]
                B = np.hstack([B, A])
            w2r, Wr = eigh(B.T @ Kb @ B, B.T @ Mff @ B)
            Phi = B @ Wr; freqs = np.sqrt(np.clip(w2r, 0, None)) / (2 * np.pi)
            if len(p):
                Psi = -np.linalg.solve(Kb, Kall[np.ix_(f, p)])
        Mr = Phi.T @ Mff @ Phi
        self._compiled = dict(M=M, K=K, C=Cm, f=f, p=p, Phi=Phi, Psi=Psi, freqs=freqs,
                              Minv=np.linalg.inv(Mr), Kff=Kff, Cff=Cff,
                              Mfp=M[np.ix_(f, p)] + Mff @ Psi, Kfp=K[np.ix_(f, p)],
                              Cfp=Cm[np.ix_(f, p)], Mff=Mff, f_cut=f_cut)
        return self._compiled

    def linear_modes(self, flank="drive", include_mesh=True):
        """natural frequencies [Hz] and mode shapes (full dof vectors) with the mesh
        linearized on one flank (mean stiffness). Fixed and prescribed dofs are clamped."""
        M, K, _ = self.matrices()
        if include_mesh:
            K = K + self.mesh_stiffness_matrix(flank)
        cons = self.fixed | set(self.prescribed)
        f = np.array([i for i in range(self.n) if i not in cons], int)
        w2, V = eigh(K[np.ix_(f, f)], M[np.ix_(f, f)])
        shapes = np.zeros((self.n, len(f))); shapes[f] = V
        return np.sqrt(np.clip(w2, 0, None)) / (2 * np.pi), shapes

    def static(self, Q, flank="drive"):
        """static equilibrium for a constant load vector Q (full size) with every mesh in
        contact on the given flank (mean stiffness). Rigid-body modes: minimum-norm solution."""
        M, K, _ = self.matrices()
        b = np.array(Q, float).copy()
        for m in self.meshes:
            g = m.g(self.n, flank)
            K = K + m.k0 * np.outer(g, g)
            if m.contact != "linear":
                b += m.k0 * m.gap * g
        cons = sorted(self.fixed | set(self.prescribed))
        f = np.array([i for i in range(self.n) if i not in cons], int)
        q = np.zeros(self.n)
        q[f] = np.linalg.lstsq(K[np.ix_(f, f)], b[f], rcond=None)[0]
        return q

    # -- dynamics ------------------------------------------------------------------------------
    def _full(self, t, eta, etad):
        cp = self._compiled
        q = np.zeros(self.n); v = np.zeros(self.n); a_p = np.zeros(len(cp["p"]))
        p = cp["p"]
        for j, i in enumerate(p):
            if i in self.prescribed:
                fn, fd, fdd = self.prescribed[i]
                q[i], v[i], a_p[j] = fn(t), fd(t), fdd(t)
        q[cp["f"]] = cp["Phi"] @ eta + cp["Psi"] @ q[p]
        v[cp["f"]] = cp["Phi"] @ etad + cp["Psi"] @ v[p]
        return q, v, a_p

    def rhs(self, t, y):
        cp = self._compiled; nr = cp["Phi"].shape[1]
        eta, etad = y[:nr], y[nr:2 * nr]
        q, v, a_p = self._full(t, eta, etad)
        Q = np.zeros(self.n); Pd = 0.0; Pw = 0.0
        for m in self.meshes:
            r = m.evaluate(q, v)
            for i, val in r["Q"].items():
                Q[i] += val
            Pd += r["P"]
        for i, fn in self.loads:
            val = fn(t, q, v); Q[i] += val; Pw += val * v[i]
        f, p = cp["f"], cp["p"]
        rhs = Q[f] - cp["Kff"] @ q[f] - cp["Cff"] @ v[f]
        if len(p):
            rhs = rhs - cp["Kfp"] @ q[p] - cp["Cfp"] @ v[p] - cp["Mfp"] @ a_p
        acc = cp["Minv"] @ (cp["Phi"].T @ rhs)
        Pd += v @ cp["C"] @ v
        return np.concatenate([etad, acc, [Pd, Pw]])

    def simulate(self, t_end, q0=None, v0=None, t_eval=None, n_out=4000, f_cut=None,
                 max_step=None, rtol=1e-8, atol=None, method="DOP853"):
        """integrate from t=0 to t_end. Returns a Result with full q, v and outputs."""
        if self._compiled is None or self._compiled["f_cut"] != f_cut:
            self.compile(f_cut)
        cp = self._compiled
        n = self.n
        q0 = np.zeros(n) if q0 is None else np.asarray(q0, float)
        v0 = np.zeros(n) if v0 is None else np.asarray(v0, float)
        f = cp["f"]; Phi = cp["Phi"]
        Pm = Phi.T @ cp["Mff"]                    # projection (Phi is M-orthonormal or I)
        if cp["f_cut"] is None:
            eta0, etad0 = q0[f], v0[f]
        else:
            q_p0, v_p0, _ = self._full(0.0, np.zeros(Phi.shape[1]), np.zeros(Phi.shape[1]))
            p = cp["p"]
            eta0 = Pm @ (q0[f] - cp["Psi"] @ q_p0[p])
            etad0 = Pm @ (v0[f] - cp["Psi"] @ v_p0[p])
        y0 = np.concatenate([eta0, etad0, [0.0, 0.0]])
        if t_eval is None:
            t_eval = np.linspace(0, t_end, n_out)
        if max_step is None:
            fr = cp["freqs"] if cp["freqs"] is not None else self.linear_modes()[0]
            max_step = 1 / (20 * max(fr.max(), 1.0))
        if atol is None:
            atol = 1e-12
        sol = solve_ivp(self.rhs, (0, t_end), y0, method=method, t_eval=t_eval,
                        max_step=max_step, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return Result(self, sol)


class Result:
    """post-processed simulation output"""

    def __init__(self, model: Model, sol):
        self.model, self.t = model, sol.t
        cp = model._compiled; nr = cp["Phi"].shape[1]
        nt = len(sol.t)
        self.q = np.zeros((model.n, nt)); self.v = np.zeros((model.n, nt))
        for k in range(nt):
            self.q[:, k], self.v[:, k], _ = model._full(sol.t[k], sol.y[:nr, k], sol.y[nr:2 * nr, k])
        self.E_diss, self.W_ext = sol.y[2 * nr], sol.y[2 * nr + 1]
        self.mesh = [m.evaluate(self.q, self.v) for m in model.meshes]
        self.bearing = {}
        for b in model.bearings:
            Rx = -(b.k * self.q[b.ix] + b.c * self.v[b.ix])
            Ry = -(b.k * self.q[b.iy] + b.c * self.v[b.iy])
            self.bearing[b.name] = np.vstack([Rx, Ry])
        M, K = cp["M"], cp["K"]
        self.T = 0.5 * np.einsum("it,ij,jt->t", self.v, M, self.v)
        self.V = 0.5 * np.einsum("it,ij,jt->t", self.q, K, self.q) + sum(r["U"] for r in self.mesh)

    def __getitem__(self, label):
        i = self.model.index(label)
        return self.q[i]

    def vel(self, label):
        return self.v[self.model.index(label)]

    @property
    def energy_residual(self):
        """(T + V + E_diss - W_ext) - initial value; zero for a correct integration
        (not meaningful with prescribed motions, whose work is not tracked)"""
        E = self.T + self.V + self.E_diss - self.W_ext
        return E - E[0]


def torsional_pair(gp: GearPair, **mesh_kw):
    """2-DOF model: dofs 'th1', 'th2' (CW positive) with inertias J1, J2 and one mesh"""
    m = Model()
    m.add_dof("th1"); m.add_dof("th2")
    m.add_mass("th1", gp.J1); m.add_mass("th2", gp.J2)
    m.add_mesh(gp, "th1", "th2", **mesh_kw)
    return m


def lateral_pair(gp: GearPair, kb1=3e8, kb2=3e8, cb1=0.0, cb2=0.0, **mesh_kw):
    """6-DOF model: x1, y1, th1, x2, y2, th2; each gear on an isotropic bearing spring
    (kb = combined stiffness of both bearings of that shaft, as seen at the gear)"""
    m = Model()
    for d in ["x1", "y1", "th1", "x2", "y2", "th2"]:
        m.add_dof(d)
    for d, v in zip(["x1", "y1", "th1", "x2", "y2", "th2"],
                    [gp.m1, gp.m1, gp.J1, gp.m2, gp.m2, gp.J2]):
        m.add_mass(d, v)
    m.add_bearing("B1", "x1", "y1", kb1, cb1)
    m.add_bearing("B2", "x2", "y2", kb2, cb2)
    m.add_mesh(gp, "th1", "th2", x1="x1", y1="y1", x2="x2", y2="y2", **mesh_kw)
    return m


def peaks(x):
    """indices of local maxima"""
    return np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1


# ---------------------------------------------------------------------------------------
# shaft finite elements (Euler-Bernoulli bending in two planes + torsion)
# ---------------------------------------------------------------------------------------
def beam_matrices(EI, rhoA, L):
    k = EI / L**3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L**2, -6 * L, 2 * L**2],
                              [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L**2, -6 * L, 4 * L**2]])
    m = rhoA * L / 420 * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L**2, 13 * L, -3 * L**2],
                                   [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L**2, -22 * L, 4 * L**2]])
    return k, m


def add_shaft(model: Model, name, length, d, n_el=8, stations=(), E=2.1e11, G=8.1e10,
              rho=7850.0, bend_scale=1.0, tors_scale=1.0, torsion_only=False, beta=0.0):
    """Add a uniform shaft along z in [0, length] with node dofs
       {name}.{k}.x, .y   lateral displacement     {name}.{k}.sx, .sy  slopes dx/dz, dy/dz
       {name}.{k}.th      torsion angle (sign convention chosen by the caller)
    `stations` are z positions that must coincide with nodes (gear, bearings, discs).
    bend_scale / tors_scale multiply EI / GJ (use e.g. 1e3 for a 'rigid' shaft).
    beta: stiffness-proportional (material) damping, C_e = beta * K_e.
    Returns dict(z=node positions, node=function z -> node index, dof=function(k, comp),
                 elements=[(kind, dof labels, K_e)] for energy bookkeeping)."""
    z = np.unique(np.round(np.concatenate([np.linspace(0, length, n_el + 1), stations]), 9))
    A = np.pi * d**2 / 4; I = np.pi * d**4 / 64; Jp = 2 * I
    elements = []
    for k in range(len(z)):
        comps = ("th",) if torsion_only else ("x", "y", "sx", "sy", "th")
        for c in comps:
            model.add_dof(f"{name}.{k}.{c}")
    for e in range(len(z) - 1):
        L = z[e + 1] - z[e]
        kt = G * Jp * tors_scale / L
        mt = rho * Jp * L / 6 * np.array([[2, 1], [1, 2]])
        Kt = kt * np.array([[1, -1], [-1, 1]]); dt = [f"{name}.{e}.th", f"{name}.{e+1}.th"]
        model.add_matrix(dt, K=Kt, M=mt, Cm=beta * Kt)
        elements.append(("torsion", dt, Kt))
        if not torsion_only:
            kb, mb = beam_matrices(E * I * bend_scale, rho * A, L)
            for w, s in (("x", "sx"), ("y", "sy")):
                db = [f"{name}.{e}.{w}", f"{name}.{e}.{s}", f"{name}.{e+1}.{w}", f"{name}.{e+1}.{s}"]
                model.add_matrix(db, K=kb, M=mb, Cm=beta * kb)
                elements.append(("bending", db, kb))

    def node(zz):
        return int(np.argmin(np.abs(z - zz)))

    def dof(k, comp):
        return f"{name}.{k}.{comp}"

    return dict(z=z, node=node, dof=dof, d=d, length=length, elements=elements, name=name)


def element_energy(model: Model, shaft, q, kind):
    """strain energy of all shaft elements of one kind ('torsion' | 'bending') for dof vector q"""
    E = 0.0
    for k, dofs, Ke in shaft["elements"]:
        if k == kind:
            idx = [model.index(d) for d in dofs]
            qe = q[idx]
            E = E + 0.5 * np.einsum("i...,ij,j...->...", qe, Ke, qe)
    return E


def add_disc(model: Model, shaft, z, m, Jp, Jd=None, torsion_only=False):
    """rigid disc (gear, flywheel) at axial position z of a shaft built with add_shaft"""
    k = shaft["node"](z); dof = shaft["dof"]
    model.add_mass(dof(k, "th"), Jp)
    if not torsion_only:
        model.add_mass(dof(k, "x"), m); model.add_mass(dof(k, "y"), m)
        Jd = 0.5 * Jp if Jd is None else Jd
        model.add_mass(dof(k, "sx"), Jd); model.add_mass(dof(k, "sy"), Jd)
    return k


# ---------------------------------------------------------------------------------------
# 1-DOF dynamic transmission error model (notebooks 03, 04)
# ---------------------------------------------------------------------------------------
def simulate_dte(gp: GearPair, n1_rpm, Fm, cycles=80, keep=20, pts=60, x0=None,
                 contact="linear", stiffness="tv", err=None, rtol=1e-8):
    """m_eq x'' + F_mesh(x, x', t) = Fm   with x = dynamic transmission error along the LOA
    (= r_b1 th1 - r_b2 th2) and the mesh position s = f_mesh * t (mean speed imposed).
    contact: 'linear' (spring may pull) or 'backlash' (two flanks, gap gp.backlash/2).
    err(s): optional static transmission error / profile error [m] (positive = excess on the
            drive flank), entering as F = k (x - err) ...
    Returns dict(t, x, xd, F, k, s, x_end) with the last `keep` mesh cycles summarised."""
    fm = gp.f_mesh(n1_rpm); Tm = 1 / fm
    g = gp.backlash / 2; c = gp.c_mesh; m = gp.m_eq

    def kk(s):
        return gp.k_of_s(s) if stiffness == "tv" else gp.k_mesh + 0 * s

    def force(t, x, xd):
        s = fm * t; k = kk(s); e = 0.0 if err is None else err(s)
        xe = x - e
        if contact == "linear":
            return k * xe + c * xd
        Fd = flank_force(xe - g, xd, k, c)
        Fr = flank_force(-xe - g, -xd, k, c)
        return Fd - Fr

    def rhs(t, y):
        return [y[1], (Fm - force(t, y[0], y[1])) / m]

    if x0 is None:
        x0 = [Fm / gp.k_mesh + (g if contact != "linear" else 0.0), 0.0]
    t_end = cycles * Tm
    t_eval = np.linspace(t_end - keep * Tm, t_end, keep * pts + 1)
    sol = solve_ivp(rhs, (0, t_end), x0, method="DOP853", t_eval=t_eval,
                    max_step=Tm / pts, rtol=rtol, atol=1e-14)
    x, xd = sol.y
    F = force(sol.t, x, xd)
    return dict(t=sol.t, x=x, xd=xd, F=F, s=fm * sol.t, x_end=sol.y[:, -1].copy(), fm=fm)


def floquet_multipliers(gp: GearPair, n1_rpm):
    """largest |Floquet multiplier| of m x'' + c x' + k(f_m t) x = 0 (>1: parametric instability)"""
    fm = gp.f_mesh(n1_rpm); Tm = 1 / fm; m, c = gp.m_eq, gp.c_mesh

    def rhs(t, y):
        k = gp.k_of_s(fm * t)
        return [y[1], -(c * y[1] + k * y[0]) / m, y[3], -(c * y[3] + k * y[2]) / m]

    sol = solve_ivp(rhs, (0, Tm), [1, 0, 0, 1], method="DOP853", rtol=1e-10, atol=1e-14,
                    max_step=Tm / 200)
    Mo = sol.y[:, -1].reshape(2, 2).T
    return np.max(np.abs(np.linalg.eigvals(Mo)))
