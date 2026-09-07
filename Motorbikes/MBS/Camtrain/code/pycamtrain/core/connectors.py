"""Connectors.

The Modelica connector concept is kept, but the semantics differ because this
framework works in minimal coordinates:

  * Modelica  -- a rotational flange carries a potential variable (phi) and a
                 flow variable (tau). Connecting two flanges generates
                 phi_a = phi_b and tau_a + tau_b = 0.
  * Here      -- a RotFlange *is* a reference to an existing degree of freedom.
                 There is no torque balance to generate: torques are applied as
                 Kane loads and the balance is an outcome of the method.

So a connection is not an equation to be flattened; it is a shared symbol.
That removes an entire class of structural-singularity errors (Section 11.3 of
the model package guide) at the cost of losing acausal reusability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sympy as sp


@dataclass
class RotFlange:
    """Rotational interface: an angle, a speed, and the frame that carries them.

    ``axis`` is the unit rotation axis expressed in ``frame``. ``body_frame``
    is the frame a torque load is applied to. Grounded flanges (housing) have
    ``body_frame is None`` and silently absorb reaction torques.
    """

    name: str
    angle: sp.Expr
    speed: sp.Expr
    frame: object
    axis: object
    body_frame: Optional[object] = None

    @property
    def grounded(self) -> bool:
        return self.body_frame is None

    def apply_torque(self, system, torque_expr) -> None:
        """Apply a torque about ``axis``. No-op for grounded flanges."""
        if self.grounded:
            return
        system.add_load((self.body_frame, torque_expr * self.axis))


@dataclass
class SpatialFrame:
    """MultiBody-style frame connector: a point plus an orientation.

    Present so that spatial components share one interface type with the
    rotational ones; the camtrain skeleton only uses it for housing anchors
    and for the spatial verification cases.
    """

    name: str
    point: object
    frame: object
    body_frame: Optional[object] = None

    def apply_force(self, system, force_vec) -> None:
        if self.body_frame is None:
            return
        system.add_load((self.point, force_vec))

    def apply_torque(self, system, torque_vec) -> None:
        if self.body_frame is None:
            return
        system.add_load((self.body_frame, torque_vec))
