"""Model parameters.

IMPORTANT -- the thesis does NOT publish the geometric and inertial data of
the instrumented 2004 Honda CBR600RR.  Only a handful of numbers appear in
the text (total mass 270 kg, wheel spin inertias, tyre radii/profiles, drag
fit).  Everything else in :func:`cbr600rr_estimate` is an engineering
estimate built from public CBR600RR data (wheelbase, rake, trail) and typical
supersport values.  Replace them with measured data when available.

Symbols follow the thesis (Fig. 4.24, List of Notations).  All lengths in m,
angles in rad, SI units throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np

# order of the geometric parameters handed to the generated kinematics code
GEOM_ORDER = ["Ra", "Px", "Pz", "Py", "Hx", "Hz", "Qx", "Qz", "Bx", "Bz",
              "Cx", "Cz", "Fx", "Fz", "Rx", "Rz", "eps",
              "Rf", "rhof", "Rr", "rhor"]


@dataclass
class BikeParams:
    # ---- geometry (Fig. 4.24) -------------------------------------------
    Ra: float      # swing-arm length, rear axle -> pivot P (along b1)
    Px: float      # pivot P -> main frame mass centre A, along a1
    Pz: float      #                                         along a2
    Hx: float      # A -> steering head point H, along a1
    Hz: float      #                              along a2
    Qx: float      # H -> Q perpendicular to steering axis (fork offset), f1
    Qz: float      # H -> Q along steering axis f2 (negative: Q below H)
    Bx: float      # P -> swing-arm mass centre, along b1 (negative: rearwards)
    Bz: float      #                              along b2
    Cx: float      # front axle G -> fork-lower mass centre C, along f1
    Cz: float      #                                            along f2
    Fx: float      # H -> steering assembly mass centre F, along f1
    Fz: float      #                                        along f2
    Rx: float      # A -> aerodynamic centre of pressure R, along a1
    Rz: float      #                                         along a2
    eps: float     # steering head (caster) angle
    Rf: float      # front tyre unloaded radius (crown centre line)
    rhof: float    # front tyre crown (cross-section) radius
    Rr: float      # rear tyre unloaded radius
    rhor: float    # rear tyre crown radius
    Py: float = 0.0   # lateral offset of A (rider hang-off, Section 4.6: A_y)
    # ---- masses ----------------------------------------------------------
    MA: float = 0.0   # main frame, engine, rider
    MB: float = 0.0   # swing-arm
    MC: float = 0.0   # fork lower (unsprung)
    MD: float = 0.0   # rear wheel
    MF: float = 0.0   # steering assembly (fork upper, bars, forearms)
    MG: float = 0.0   # front wheel
    # ---- inertia tensors about the mass centres, in the body frames ------
    # (axis 1 fwd, 2 up, 3 right;  wheels: axis 3 is the spin axis)
    IA: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    IB: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    IC: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    ID: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    IF: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    IG: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    # ---- travel limits (used by the suspension force models) -------------
    fork_travel: float = 0.120      # q1 in [0, fork_travel]; 0 = bottomed
    q2_top: float = 0.0             # swing-arm angle at full extension
    q2_bottom: float = 0.0          # swing-arm angle fully compressed
    g: float = 9.81

    # ------------------------------------------------------------------
    def geom(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in GEOM_ORDER], dtype=float)

    @property
    def total_mass(self) -> float:
        return self.MA + self.MB + self.MC + self.MD + self.MF + self.MG

    def copy(self, **changes) -> "BikeParams":
        d = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in asdict(self).items()}
        d.update(changes)
        return BikeParams(**d)


def cbr600rr_estimate() -> tuple[BikeParams, dict]:
    """Estimated parameter set for a 2004 CBR600RR with rider (270 kg).

    Returns ``(params, design)`` where ``design`` holds the nominal static
    configuration the geometry was built around (q1, q2 at mu = 0 with
    undeflected tyres).

    From the thesis:   total mass 270 kg (Sec. 4.5.5); wheel spin inertias
                       0.484 / 0.638 kg m^2 (Sec. 4.5.5); tyre radii
                       303.2 / 320.4 mm and profile points (Sec. 5.5,
                       Table 5.6 -> crown radii by circle fit).
    Public data:       wheelbase 1.390 m, rake 24 deg, trail 95 mm,
                       120 mm fork travel.
    Everything else:   ESTIMATED.
    """
    W, eps, trail = 1.390, np.radians(24.0), 0.095
    Rf, Rr = 0.3032, 0.3204
    # crown radius of a circle through tread edge and crown (Table 5.6):
    # front half-width 59.5 mm, crown height 49.5 mm; rear 92.0 / 56.0 mm
    rhof = (59.5**2 + 49.5**2) / (2 * 49.5) * 1e-3     # 0.0605 m
    rhor = (92.0**2 + 56.0**2) / (2 * 56.0) * 1e-3     # 0.1036 m

    Ra = 0.560                 # swing-arm length                  (estimate)
    zP = 0.410                 # pivot height at static trim       (estimate)
    xA, zA = 0.700, 0.660      # frame+engine+rider mass centre    (estimate)
    zH = 0.860                 # height of steering head point H   (estimate)
    q1s = 0.085                # static fork extension (35 mm sag) (estimate)

    q2s = np.arcsin((zP - Rr) / Ra)
    xP = Ra * np.cos(q2s)
    xH = (W + trail) - zH * np.tan(eps)
    Qx = Rf * np.sin(eps) - trail * np.cos(eps)        # eq. (4.25) inverted
    e2 = np.array([-np.sin(eps), np.cos(eps)])
    Qz = float((np.array([W, Rf]) - np.array([xH, zH])) @ e2 + q1s)

    rear_travel = 0.130        # axle travel                       (estimate)
    rear_sag = 0.030
    q2_top = q2s + rear_sag / Ra
    q2_bottom = q2_top - rear_travel / Ra

    def diag(a, b, c):
        return np.diag([a, b, c]).astype(float)

    p = BikeParams(
        Ra=Ra, Px=xA - xP, Pz=zA - zP, Hx=xH - xA, Hz=zH - zA,
        Qx=Qx, Qz=Qz,
        Bx=-0.300, Bz=0.000,
        Cx=-0.010, Cz=0.120,
        Fx=0.020, Fz=-0.100,
        Rx=0.080, Rz=0.060,
        eps=eps, Rf=Rf, rhof=rhof, Rr=Rr, rhor=rhor,
        MA=219.0, MB=8.0, MC=7.0, MD=14.5, MF=10.0, MG=11.5,   # sum = 270
        IA=diag(13.0, 20.0, 28.0),
        IB=diag(0.05, 0.35, 0.35),
        IC=diag(0.10, 0.05, 0.10),
        ID=diag(0.36, 0.36, 0.638),       # spin inertia from the thesis
        IF=diag(0.25, 0.10, 0.20),
        IG=diag(0.27, 0.27, 0.484),       # spin inertia from the thesis
        fork_travel=0.120, q2_top=q2_top, q2_bottom=q2_bottom,
    )
    design = dict(q1=q1s, q2=q2s, wheelbase=W, trail=trail)
    return p, design
