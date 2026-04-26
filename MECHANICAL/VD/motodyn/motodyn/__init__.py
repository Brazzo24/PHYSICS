"""
motodyn — a modular motorcycle dynamics framework.

Design philosophy
-----------------
* Parameters (dataclasses) are decoupled from models (pure functions / small classes).
* Each physics subsystem is independently usable: you can study anti-squat without
  ever constructing a chassis or tyre model.
* Quasi-static by default; every model exposes a callable signature that is also
  compatible with an ODE state vector, so a future dynamic integrator can be
  layered on without rewriting the physics.

Top-level re-exports for convenience; submodules remain the canonical API.
"""

from .parameters import (
    MassProperties,
    Geometry,
    SuspensionGeometry,
    Powertrain,
    BrakeSetup,
    Motorcycle,
)
from .state import VehicleState, OperatingPoint

__all__ = [
    "MassProperties",
    "Geometry",
    "SuspensionGeometry",
    "Powertrain",
    "BrakeSetup",
    "Motorcycle",
    "VehicleState",
    "OperatingPoint",
]

__version__ = "0.1.0"
