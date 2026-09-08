"""pycamtrain -- a minimal-coordinate symbolic MBS framework.

A Python counterpart to the KTM_Camtrain_MBS Dymola package, built to compare
workflow, capability and performance between the two environments.

Stage 1 (current): framework skeleton, verified against closed-form solutions.
"""

__version__ = "0.1.0"

from .core import MBSystem, simulate
from .components import (CamtrainBank, CompliantSpurGearMesh, Housing,
                         KinematicDrive, PendulumBody, Shaft, SpeedSensor,
                         TorqueSource, TorsionalSpringDamper)
from .geometry import BankGeometry, MeshStage, layout_from_bearing_angles

__all__ = ["MBSystem", "simulate", "Housing", "Shaft", "TorsionalSpringDamper",
           "CompliantSpurGearMesh", "TorqueSource", "SpeedSensor", "PendulumBody",
           "KinematicDrive", "CamtrainBank", "BankGeometry", "MeshStage",
           "layout_from_bearing_angles"]
