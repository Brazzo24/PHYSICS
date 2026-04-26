"""
Parameter containers.

All geometric and inertial data live here as frozen dataclasses so parameter
sweeps are trivially reproducible: copy, mutate one field, re-run.

Sign conventions (ISO-ish, adapted for a bike)
---------------------------------------------
* x forward, z up, y to the left.
* Wheelbase p > 0; caster angle epsilon > 0 measured from vertical.
* All distances in metres, masses in kg, angles in radians unless noted.
* CoG height `h` measured from the ground.
* `b` = horizontal distance from rear contact patch to CoG projection.
  Then (p - b) is the distance from front contact patch to CoG projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional
import math


# --------------------------------------------------------------------------- #
# Core inertial & geometric properties
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MassProperties:
    """Total-vehicle mass properties (rider included unless noted)."""
    mass: float                  # kg, total including rider
    h_cog: float                 # m, CoG height above ground (static, unladen bike upright)
    b: float                     # m, horizontal distance rear contact patch -> CoG projection
    I_roll: float = 30.0         # kg m^2, about roll axis through CoG
    I_pitch: float = 25.0        # kg m^2
    I_yaw: float = 25.0          # kg m^2
    m_unsprung_front: float = 12.0
    m_unsprung_rear: float = 18.0

    def weight_distribution_static(self, wheelbase: float) -> tuple[float, float]:
        """Return (front_fraction, rear_fraction) on level ground, bike upright."""
        front = self.b / wheelbase
        rear = 1.0 - front
        return front, rear


@dataclass(frozen=True)
class Geometry:
    """Primary chassis geometry."""
    wheelbase: float             # m
    caster_angle: float          # rad, rake from vertical at steering axis
    trail: float                 # m, mechanical trail
    front_wheel_radius: float = 0.305
    rear_wheel_radius: float = 0.31

    def __post_init__(self):
        if self.wheelbase <= 0:
            raise ValueError("wheelbase must be positive")
        if not (0 < self.caster_angle < math.pi / 2):
            raise ValueError("caster_angle (rad) out of plausible range")


# --------------------------------------------------------------------------- #
# Suspension kinematics — the core of anti-squat / anti-dive / Mozzi axis work
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SuspensionGeometry:
    """
    Suspension pickup points and linkage geometry needed to compute
    anti-squat, anti-dive, anti-lift, and the instant centres.

    All coordinates are in the side-view plane (x, z) with origin at the
    rear contact patch, x forward, z up.

    For the rear swingarm we assume a single-pivot or linkage-idealised
    swingarm: the swingarm pivot is specified directly, and the effective
    instant centre (IC) may differ if `rear_ic` is given explicitly
    (e.g. for a linkage-driven rear).

    For the front we support telescopic forks (default) and a fictitious
    front IC (`front_ic`) for hub-centre / girder / Hossack analyses.
    """
    # Rear
    swingarm_pivot: tuple[float, float]      # (x, z), m, from rear contact
    rear_ic: Optional[tuple[float, float]] = None  # override effective IC for linkage bikes

    # Front (telescopic fork default)
    fork_offset: float = 0.030               # m, fork offset at triple clamp
    front_ic: Optional[tuple[float, float]] = None  # for non-telescopic front ends

    # Sprocket / drive geometry — governs chain-force anti-squat
    sprocket_center: tuple[float, float] = (0.0, 0.30)   # (x, z) of countershaft sprocket
    countershaft_sprocket_radius: float = 0.035          # m
    rear_sprocket_radius: float = 0.095                  # m
    chain_angle_override: Optional[float] = None         # rad, bypass tangent geometry

    # Travel & stiffness (quasi-static load-transfer helpers)
    rear_travel: float = 0.130               # m
    front_travel: float = 0.130              # m
    k_rear_wheel: float = 90_000.0           # N/m, rear wheel-rate
    k_front_wheel: float = 22_000.0          # N/m, front wheel-rate (per fork leg * 2 lumped)


@dataclass(frozen=True)
class Powertrain:
    """Drivetrain parameters affecting load transfer via chain pull."""
    final_drive_ratio: float = 2.8           # rear_sprocket_teeth / countershaft_teeth
    primary_plus_gearbox: float = 2.0        # overall internal ratio in current gear
    driven_wheel: str = "rear"               # "rear" or, rarely, "front" (2WD experimental)
    # The countershaft sprocket position lives on SuspensionGeometry so chain
    # geometry stays a *geometric* concern, not a powertrain concern.


@dataclass(frozen=True)
class BrakeSetup:
    """Brake torque distribution at the wheels."""
    front_brake_fraction: float = 0.70       # 0..1 of total braking torque at front wheel
    # Rear fraction = 1 - front_brake_fraction


# --------------------------------------------------------------------------- #
# Top-level composite
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Motorcycle:
    """
    Composite container. Pass this into analyses to study one bike.

    Use `.with_changes(...)` for parameter sweeps — it returns a new
    Motorcycle with shallow-replaced fields and does NOT mutate the original.
    """
    mass: MassProperties
    geometry: Geometry
    suspension: SuspensionGeometry
    powertrain: Powertrain = field(default_factory=Powertrain)
    brakes: BrakeSetup = field(default_factory=BrakeSetup)
    name: str = "generic"

    def with_changes(self, **kwargs) -> "Motorcycle":
        """
        Return a modified copy. Supports dotted paths for nested fields:

            bike.with_changes(**{"suspension.swingarm_pivot": (0.45, 0.35)})

        or a direct top-level replacement:

            bike.with_changes(mass=new_mass_props)
        """
        top_level = {}
        nested: dict[str, dict] = {}
        for key, value in kwargs.items():
            if "." in key:
                parent, child = key.split(".", 1)
                nested.setdefault(parent, {})[child] = value
            else:
                top_level[key] = value

        for parent, child_kwargs in nested.items():
            current = getattr(self, parent)
            top_level[parent] = replace(current, **child_kwargs)

        return replace(self, **top_level)
