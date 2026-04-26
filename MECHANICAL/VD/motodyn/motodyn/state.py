"""
State and operating-point definitions.

`OperatingPoint` is the quasi-static "what is the bike doing right now"
container — speed, longitudinal accel, roll angle, etc. It's the primary
input to the analysis functions.

`VehicleState` is a full dynamic state vector (positions + velocities)
prepared for ODE integration in a future extension. The quasi-static
analyses don't use it today, but parameter names line up so nothing has
to change when you add the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class OperatingPoint:
    """Instantaneous conditions used by quasi-static analyses."""
    speed: float = 0.0              # m/s, forward speed
    a_x: float = 0.0                # m/s^2, longitudinal accel (>0 accelerating)
    a_y: float = 0.0                # m/s^2, lateral accel at CoG in bike frame
    roll: float = 0.0               # rad, lean angle (positive to the right)
    pitch: float = 0.0              # rad, chassis pitch (positive nose up)
    throttle: float = 0.0           # 0..1
    brake: float = 0.0              # 0..1, total brake command

    @property
    def is_braking(self) -> bool:
        return self.a_x < -0.05     # small deadband


@dataclass
class VehicleState:
    """
    Full 8-DoF-style state vector (reserved for dynamic mode).

    q = [x, y, psi, roll, pitch, z_sprung, s_front, s_rear]
        global planar position, yaw, roll, pitch, heave, suspension strokes
    qd = time derivatives of q

    This is the handle that a future integrator will advance; for now it
    simply stores values and can be converted to/from an OperatingPoint.
    """
    q: np.ndarray = field(default_factory=lambda: np.zeros(8))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(8))

    @property
    def roll(self) -> float: return float(self.q[3])
    @property
    def pitch(self) -> float: return float(self.q[4])
    @property
    def roll_rate(self) -> float: return float(self.qd[3])
    @property
    def speed(self) -> float:
        # planar speed magnitude
        return float(np.hypot(self.qd[0], self.qd[1]))

    def to_operating_point(self, a_x: float = 0.0, a_y: float = 0.0) -> OperatingPoint:
        return OperatingPoint(
            speed=self.speed, a_x=a_x, a_y=a_y,
            roll=self.roll, pitch=self.pitch,
        )
