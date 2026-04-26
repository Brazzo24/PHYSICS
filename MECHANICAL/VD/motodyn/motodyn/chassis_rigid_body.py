"""
Chassis as a rigid body: roll / pitch / yaw equilibrium.

This module does NOT integrate anything — it gives you the moments and
forces on the sprung mass for a given operating point, so you can build
quasi-static studies or feed them into an ODE later.

The core function `chassis_equilibrium` returns the required lean angle
to balance lateral acceleration, plus the resulting pitch attitude from
longitudinal load transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..parameters import Motorcycle
from ..state import OperatingPoint
from . import load_transfer, suspension_kinematics as sk

G = 9.80665


@dataclass(frozen=True)
class ChassisEquilibrium:
    lean_required: float     # rad — φ such that the centrifugal + gravity resultant aligns
    pitch_angle: float       # rad — chassis pitch due to suspension deflection
    front_compression: float # m — fork compression (positive = compressed)
    rear_compression: float  # m — shock compression at the wheel (positive = compressed)
    N_front: float
    N_rear: float


def required_lean_angle(a_y: float) -> float:
    """
    Kinematic lean angle ignoring tyre width and camber thrust:

        φ = atan(a_y / g)
    """
    return math.atan2(a_y, G)


def chassis_equilibrium(bike: Motorcycle, op: OperatingPoint) -> ChassisEquilibrium:
    """
    Quasi-static equilibrium: wheel loads, suspension deflections, lean.

    Only the component of load transfer that goes *through the springs*
    causes compression — that's the whole point of anti-squat/anti-dive.
    """
    loads = load_transfer.longitudinal_load_transfer(bike, op)
    paths = sk.effective_load_paths(bike, op.a_x)

    k_f = bike.suspension.k_front_wheel
    k_r = bike.suspension.k_rear_wheel

    # Static preload compressions (from gravity) — also only the spring path matters,
    # and at rest the spring path carries the full static load.
    static = load_transfer.static_loads(bike)

    comp_f = (static.N_front + paths.through_spring_front) / k_f
    comp_r = (static.N_rear + paths.through_spring_rear) / k_r

    # Pitch from differential compression over the wheelbase
    # Positive pitch = nose up; nose lifts when rear compresses more than front
    dz_rear_minus_front = comp_r - comp_f
    pitch = math.atan2(dz_rear_minus_front, bike.geometry.wheelbase)

    return ChassisEquilibrium(
        lean_required=required_lean_angle(op.a_y),
        pitch_angle=pitch,
        front_compression=comp_f,
        rear_compression=comp_r,
        N_front=loads.N_front,
        N_rear=loads.N_rear,
    )
