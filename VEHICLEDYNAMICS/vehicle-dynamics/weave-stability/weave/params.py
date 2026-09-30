"""
weave.params
============
Parameter containers for the multibody weave model, plus constructors that
re-use the existing ``MotorcycleParams`` / ``build_inertia`` / Magic-Formula
helpers from the Pacejka Ch.11 stage 1-3 code (vendored in ``weave/_legacy``).

Frame convention (rear-frame axes, origin O = centre of the rear wheel):
    x forward, y left, z up (right-handed);  roll phi about +x
    (positive = lean RIGHT), yaw psi about +z, steer delta about the steering
    axis (positive = steer LEFT, i.e. positive rotation about the upward axis).
Wheel spin: angular velocity vector +y (= forward rolling).
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple
import numpy as np

from ._legacy.pacejka_moto_stage12 import MotorcycleParams, normal_loads, _Ky
from ._legacy.pacejka_moto_stage3 import (build_inertia, cornering_stiffness,
                                          camber_stiffness)

G = 9.81


def inertia_tensor(Ixx, Iyy, Izz, Ixz=0.0):
    """Inertia tensor about the COM in (x fwd, y left, z up) axes.
    ``Ixz`` is the tensor element (= -integral x z dm)."""
    return np.array([[Ixx, 0.0, Ixz], [0.0, Iyy, 0.0], [Ixz, 0.0, Izz]])


# ---------------------------------------------------------------------------
@dataclass
class Body:
    """Rigid body (or ideal rotor when mass = 0).

    parent        : 'rear' (moves with rear frame) or 'front' (steers)
    pos           : COM in rear-frame axes, reference configuration [m]
    inertia       : 3x3 about COM, body axes aligned with the parent frame
    spin_axis     : unit vector in parent axes about which the body spins
    spin_per_speed: spin rate / forward speed  [rad s^-1 / (m s^-1)]  (signed)
    """
    name: str
    mass: float
    pos: Tuple[float, float, float]
    inertia: np.ndarray
    parent: str = "rear"
    spin_axis: Optional[Tuple[float, float, float]] = None
    spin_per_speed: float = 0.0


@dataclass
class Geometry:
    wheelbase: float = 1.40      # contact point to contact point [m]
    rake: float = np.radians(27.0)   # steering axis from vertical [rad]
    trail: float = 0.10          # ground trail [m]
    r_rear: float = 0.305
    r_front: float = 0.305


@dataclass
class Tyre:
    """Linear tyre: Fy = -(C_alpha * slip_vel/u + C_gamma * gamma), first-order
    lag with relaxation length sigma (0 = no lag), force acts a distance
    ``trail`` behind the contact centre (pneumatic trail -> self-aligning)."""
    c_alpha: float = 20000.0
    c_gamma: float = 400.0
    sigma: float = 0.0
    trail: float = 0.045


# ---------------------------------------------------------------------------
@dataclass
class Rotor:
    """One rotating group of the powertrain (ideal gyro about its axis).

    stage : 'crank' | 'input' | 'output'
        crank  : crank-speed group   (speed = wheel * primary * gear * final)
        input  : clutch/input shaft  (speed = wheel * gear * final, reversed
                 by the primary gear pair relative to the crank)
        output : output shaft/sprocket (speed = wheel * final, same sense as wheel)
    I     : axial moment of inertia [kg m^2]
    sense : +1/-1 extra sign (e.g. -1 = counter-rotating balance shaft)
    axis  : rotation axis in rear-frame axes (default transverse, +y)
    """
    name: str
    stage: str
    I: float
    sense: float = 1.0
    axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    pos: Tuple[float, float, float] = (0.55, 0.0, 0.40)


@dataclass
class Powertrain:
    """Engine + transmission spinning inertias and gear ratios."""
    rotors: List[Rotor] = field(default_factory=list)
    primary: float = 1.80        # crank -> input shaft
    gears: Tuple[float, ...] = (2.62, 1.99, 1.63, 1.38, 1.21, 1.09)
    final: float = 2.90          # output shaft -> rear wheel (chain/sprocket)
    gear: int = 6                # 1-based
    crank_direction: float = 1.0  # +1 = with wheels, -1 = counter-rotating crank

    def stage_ratio(self, stage: str) -> float:
        """Signed speed ratio (rotor speed / rear wheel speed)."""
        G_ = self.gears[self.gear - 1]
        if stage == "crank":
            return self.crank_direction * self.primary * G_ * self.final
        if stage == "input":
            return -self.crank_direction * G_ * self.final
        if stage == "output":
            return self.final
        raise ValueError(stage)

    def overall_crank_ratio(self) -> float:
        return abs(self.stage_ratio("crank"))

    def bodies(self, r_rear: float) -> List[Body]:
        out = []
        for r in self.rotors:
            k = r.sense * self.stage_ratio(r.stage) / r_rear
            out.append(Body(name=r.name, mass=0.0, pos=r.pos,
                            inertia=r.I * np.outer(r.axis, r.axis),
                            parent="rear", spin_axis=r.axis,
                            spin_per_speed=k))
        return out

    def angular_momentum(self, u: float, r_rear: float) -> float:
        """Net transverse angular momentum of the powertrain at speed u."""
        return sum(r.I * r.sense * self.stage_ratio(r.stage) * u / r_rear
                   for r in self.rotors)

    def crank_rpm(self, u: float, r_rear: float) -> float:
        return abs(self.stage_ratio("crank")) * u / r_rear * 60 / (2 * np.pi)

    # presets ---------------------------------------------------------------
    @staticmethod
    def inline4_sportbike(**kw) -> "Powertrain":
        """Transverse inline-4, forward-rotating crank (typical superbike)."""
        rotors = [Rotor("crank+flywheel+alt", "crank", 0.0120),
                  Rotor("clutch+input shaft", "input", 0.0070),
                  Rotor("output shaft+sprocket", "output", 0.0035)]
        return Powertrain(rotors=rotors, **kw)

    @staticmethod
    def v_twin_cruiser(**kw) -> "Powertrain":
        """Heavy-flywheel twin (large crank inertia, low ratio)."""
        rotors = [Rotor("crank+flywheel", "crank", 0.055),
                  Rotor("clutch+input shaft", "input", 0.012),
                  Rotor("output shaft+sprocket", "output", 0.004)]
        kw.setdefault("primary", 1.65)
        kw.setdefault("gears", (2.4, 1.6, 1.2, 1.0, 0.85))
        kw.setdefault("final", 2.2)
        kw.setdefault("gear", 5)
        return Powertrain(rotors=rotors, **kw)


# ---------------------------------------------------------------------------
@dataclass
class BikeModel:
    """Everything the multibody model needs."""
    geom: Geometry
    bodies: List[Body]                  # rear frame, engine block, front frame, wheels ...
    tyre_front: Tyre = field(default_factory=Tyre)
    tyre_rear: Tyre = field(default_factory=Tyre)
    steer_damper: float = 0.0           # N m s / rad
    powertrain: Optional[Powertrain] = None
    name: str = "bike"

    def all_bodies(self) -> List[Body]:
        pt = self.powertrain.bodies(self.geom.r_rear) if self.powertrain else []
        return list(self.bodies) + pt

    # -- convenience --------------------------------------------------------
    def total_mass(self) -> float:
        return sum(b.mass for b in self.bodies)

    def with_powertrain(self, pt: Optional[Powertrain]) -> "BikeModel":
        return replace(self, powertrain=pt)

    def static_loads(self):
        """Static vertical wheel loads from the mass distribution (upright)."""
        m = self.total_mass()
        xc = sum(b.mass * b.pos[0] for b in self.bodies) / m
        w = self.geom.wheelbase
        Fz_f = m * G * xc / w
        return Fz_f, m * G - Fz_f


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------
def steering_axis(geom: Geometry):
    """Point on the ground where the steering axis meets it (rear-frame axes,
    origin at rear axle centre) and the axis direction (pointing up/back).
    Positive trail: the axis meets the ground AHEAD of the front contact point."""
    G_s = np.array([geom.wheelbase + geom.trail, 0.0, -geom.r_rear])
    s = np.array([-np.sin(geom.rake), 0.0, np.cos(geom.rake)])
    return G_s, s


def from_legacy(p: Optional[MotorcycleParams] = None,
                powertrain: Optional[Powertrain] = None,
                m_engine: float = 55.0,
                engine_pos: Tuple[float, float] = (0.55, 0.42),
                m_wheel_front: float = 8.0,
                m_wheel_rear: float = 10.0,
                Iyy_rear: float = 45.0,
                Izz_rear: float = 12.0,
                sigma: float = 0.25,
                steer_damper: float = 15.0,
                **inertia_kw) -> BikeModel:
    """
    Build a :class:`BikeModel` from the existing ``MotorcycleParams`` /
    ``build_inertia`` (stage 1-3 code).  What is taken over unchanged:
    wheelbase, rake, trail, wheel radii, all masses, CoG heights, wheel spin
    inertias, front-fork inertia about the steering axis, roll inertia and
    roll/yaw product of inertia of the mainframe+rider group, tyre
    Magic-Formula linearisation (cornering / camber stiffness, pneumatic trail),
    steering-damper coefficient.

    Added assumptions (not in the legacy code): pitch/yaw inertia of the rear
    group (``Iyy_rear``, ``Izz_rear``), wheel masses, an explicit engine block
    (mass + position, carved out of the mainframe mass), tyre relaxation
    length ``sigma``.  The legacy longitudinal reference point A is placed at
    x = bc from the rear axle.
    """
    p = p or MotorcycleParams()
    ip = build_inertia(p, **inertia_kw)
    geom = Geometry(wheelbase=p.l, rake=p.lam, trail=p.tc,
                    r_rear=p.r2, r_front=p.r1)
    G_s, s = steering_axis(geom)

    # rear group (mainframe+rider), legacy: CoG at reference point A, height hm0
    m_rear_grp = p.mm + p.mr
    x_A = p.bc
    z_A = ip.hm0 - geom.r_rear                          # rel. rear axle centre
    I_rear_grp = inertia_tensor(ip.Imx0, Iyy_rear + Izz_rear * 0, Izz_rear,
                                ip.Imxz0)
    I_rear_grp[1, 1] = Iyy_rear

    # carve out rear wheel and engine block; keep the group COM where legacy has it
    m_rest = m_rear_grp - m_wheel_rear - m_engine
    ex, ez = engine_pos[0], engine_pos[1] - geom.r_rear
    # group COM = (m_rest*x_rest + m_engine*ex + m_wheel*0)/m_rear_grp
    x_rest = (m_rear_grp * x_A - m_engine * ex) / m_rest
    z_rest = (m_rear_grp * z_A - m_engine * ez) / m_rest
    bodies = [
        Body("rear frame+rider", m_rest, (x_rest, 0.0, z_rest), I_rear_grp),
        Body("engine block", m_engine, (ex, 0.0, ez),
             inertia_tensor(1.2, 1.5, 1.2)),
        Body("rear wheel", m_wheel_rear, (0.0, 0.0, 0.0),
             inertia_tensor(ip.Iwy2 / 2, ip.Iwy2, ip.Iwy2 / 2),
             spin_axis=(0.0, 1.0, 0.0), spin_per_speed=1.0 / geom.r_rear),
    ]

    # front assembly (fork+bars) : mass, COM on/near steering axis
    m_front = p.mf + p.ms - m_wheel_front
    h_f = (p.mf * p.hf + p.ms * p.hs) / (p.mf + p.ms)   # above ground
    e_f = (p.mf * p.ef + p.ms * p.es) / (p.mf + p.ms)   # ahead of the axis
    t = (h_f - 0.0) / np.cos(geom.rake)
    on_axis = G_s + s * t
    n_f = np.array([np.cos(geom.rake), 0.0, np.sin(geom.rake)])
    com_f = on_axis + e_f * n_f
    # tensor with prescribed inertia about the steering axis = If_steer
    shape = np.diag([1.0, 1.0, 0.6])
    d2 = e_f**2
    k = (ip.If_steer - m_front * d2) / (s @ shape @ s)
    bodies.append(Body("front assembly", m_front, tuple(com_f),
                       k * shape, parent="front"))
    C_F0 = np.array([geom.wheelbase, 0.0, geom.r_front - geom.r_rear])
    bodies.append(Body("front wheel", m_wheel_front, tuple(C_F0),
                       inertia_tensor(ip.Iwy1 / 2, ip.Iwy1, ip.Iwy1 / 2),
                       parent="front", spin_axis=(0.0, 1.0, 0.0),
                       spin_per_speed=1.0 / geom.r_front))

    model = BikeModel(geom=geom, bodies=bodies, powertrain=powertrain,
                      steer_damper=steer_damper, name="legacy-based sportbike")
    # tyres: legacy Magic-Formula linearisation at the static loads of *this* mass distribution
    Fz_f, Fz_r = model.static_loads()
    model.tyre_front = Tyre(cornering_stiffness(Fz_f, p), camber_stiffness(Fz_f, p),
                            sigma, p.t0)
    model.tyre_rear = Tyre(cornering_stiffness(Fz_r, p), camber_stiffness(Fz_r, p),
                           sigma, 0.85 * p.t0)
    return model


def benchmark_bicycle(stiff_tyre: float = 1e8) -> BikeModel:
    """
    Meijaard et al. (2007) 'Linearized dynamics equations for the balance and
    steer of a bicycle: a benchmark and review', Proc. R. Soc. A 463, Table 1.
    Used to validate the derivation: the no-slip limit (very stiff tyres, no
    camber thrust, no lag) must reproduce the published critical speeds
    v_weave = 4.292 m/s and v_capsize = 6.024 m/s.
    The paper uses z pointing down; converted here to z up.
    """
    # rake = steer-axis tilt from the vertical (paper: lambda = pi/10 = 18 deg)
    geom = Geometry(wheelbase=1.02, rake=np.pi / 10,
                    trail=0.08, r_rear=0.3, r_front=0.35)
    r_R, r_F = geom.r_rear, geom.r_front
    bodies = [
        Body("rear frame B", 85.0, (0.3, 0.0, 0.9 - r_R),
             inertia_tensor(9.2, 11.0, 2.8, -2.4)),
        Body("rear wheel R", 2.0, (0.0, 0.0, 0.0),
             inertia_tensor(0.0603, 0.12, 0.0603),
             spin_axis=(0, 1, 0), spin_per_speed=1.0 / r_R),
        Body("front frame H", 4.0, (0.9, 0.0, 0.7 - r_R),
             inertia_tensor(0.05892, 0.06, 0.00708, +0.00756), parent="front"),
        Body("front wheel F", 3.0, (geom.wheelbase, 0.0, r_F - r_R),
             inertia_tensor(0.1405, 0.28, 0.1405), parent="front",
             spin_axis=(0, 1, 0), spin_per_speed=1.0 / r_F),
    ]
    tyre = Tyre(c_alpha=stiff_tyre, c_gamma=0.0, sigma=0.0, trail=0.0)
    return BikeModel(geom, bodies, tyre, tyre, 0.0, None, "Meijaard benchmark")
