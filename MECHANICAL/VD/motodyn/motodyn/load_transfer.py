"""
Quasi-static longitudinal (and basic lateral) load transfer.

References
----------
Cossalter, V. "Motorcycle Dynamics" (2nd ed.), ch. 2 & 4.
Foale, T. "Motorcycle Handling and Chassis Design".

Sign convention: a_x > 0 under acceleration, < 0 under braking.
Positive load transfer ΔN goes from front to rear on acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..parameters import Motorcycle
from ..state import OperatingPoint

G = 9.80665


@dataclass(frozen=True)
class WheelLoads:
    """Static or instantaneous normal loads at each contact patch, in newtons."""
    N_front: float
    N_rear: float

    @property
    def total(self) -> float: return self.N_front + self.N_rear

    @property
    def front_fraction(self) -> float:
        t = self.total
        return self.N_front / t if t > 0 else 0.0


def static_loads(bike: Motorcycle) -> WheelLoads:
    """Upright, stationary, level ground."""
    W = bike.mass.mass * G
    front_frac, rear_frac = bike.mass.weight_distribution_static(bike.geometry.wheelbase)
    return WheelLoads(N_front=W * front_frac, N_rear=W * rear_frac)


def longitudinal_load_transfer(
    bike: Motorcycle,
    op: OperatingPoint,
) -> WheelLoads:
    """
    Rigid-chassis, flat-ground, small-pitch load transfer.

        ΔN = m * a_x * h / p

    This is the textbook result: the suspension geometry (anti-squat /
    anti-dive) does not change total load transfer, only how it is shared
    between sprung/unsprung paths. That split is handled in
    `suspension_kinematics.effective_load_paths`.
    """
    m = bike.mass.mass
    h = bike.mass.h_cog
    p = bike.geometry.wheelbase

    static = static_loads(bike)
    delta = m * op.a_x * h / p       # positive under accel → shifts toward rear

    N_f = static.N_front - delta
    N_r = static.N_rear + delta

    # Clamp to zero — wheel lift is a valid regime, but negative normal force is not physical
    N_f = max(N_f, 0.0)
    N_r = max(N_r, 0.0)
    return WheelLoads(N_front=N_f, N_rear=N_r)


def wheelie_threshold_acceleration(bike: Motorcycle) -> float:
    """Longitudinal accel (m/s^2) at which front load → 0 on level ground."""
    p = bike.geometry.wheelbase
    h = bike.mass.h_cog
    _, rear_frac = bike.mass.weight_distribution_static(p)
    # At lift-off: m a h / p = m g * (1 - rear_frac?) -- use front static directly
    front_frac = 1 - rear_frac
    return G * front_frac * p / h


def stoppie_threshold_deceleration(bike: Motorcycle) -> float:
    """Deceleration (m/s^2, positive number) at which rear load → 0."""
    p = bike.geometry.wheelbase
    h = bike.mass.h_cog
    _, rear_frac = bike.mass.weight_distribution_static(p)
    return G * rear_frac * p / h
