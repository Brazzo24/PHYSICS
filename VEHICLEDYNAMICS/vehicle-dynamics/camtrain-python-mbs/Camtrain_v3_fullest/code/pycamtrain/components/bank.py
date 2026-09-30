"""CamtrainBank -- one camtrain bank as a reusable subsystem.

Ownership follows the central principle of Section 1: shared equipment lives at
the wrapper level, bank-specific shafts and meshes live inside the bank. The
bank therefore contains four downstream shafts, four compliant meshes, sensors
and diagnostics -- and deliberately does *not* contain the World, the physical
crankshaft or the crank actuator. It receives the crankshaft through a flange,
exactly as ``driveIn`` does in the Dymola package.

All geometry comes from a single BankGeometry (Section 5.1). Initial speeds are
derived from the drive speed and the ratio chain (Section 11.1), so the model
starts free of mesh shock.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import sympy as sp

from ..core.component import Component
from ..core.connectors import RotFlange
from ..geometry import SHAFT_NAMES, BankGeometry
from .rotational import CompliantSpurGearMesh, Shaft

# placeholder polar inertias [kg m^2]; replace with CAD values
DEFAULT_INERTIA = {
    "stepShaft": 3.0e-4,
    "idlerCamShaft": 4.5e-4,
    "intakeCamshaft": 6.0e-4,
    "exhaustCamshaft": 6.0e-4,
}


class CamtrainBank(Component):
    """Four shafts and four compliant meshes driven through ``drive_flange``.

    Parameters
    ----------
    geometry      : BankGeometry -- the only geometric input
    housing       : the stationary Housing all bearings attach to
    drive_flange  : crankshaft flange (external to the bank, Section 7)
    drive_speed   : reference crankshaft speed [rad/s], used only to derive
                    consistent initial speeds
    """

    def __init__(self, system, name, geometry: BankGeometry, housing,
                 drive_flange: RotFlange, drive_speed: float,
                 inertias: Optional[Dict[str, float]] = None,
                 mesh_stiffness: float = 1.0e8, mesh_damping: float = 2.0e3,
                 use_base_radius: bool = True, parent=None):
        super().__init__(system, name, parent)

        self.geometry = geometry
        self.drive_flange = drive_flange
        inertias = dict(DEFAULT_INERTIA if inertias is None else inertias)

        N = system.frame
        axis = geometry.axis

        # ---- shafts: rows 1..4 of the shaft-position matrix --------------
        self.shafts: Dict[str, Shaft] = {}
        for row in range(1, len(SHAFT_NAMES)):
            sname = SHAFT_NAMES[row]
            p = geometry.shaft_pos[row]
            position = float(p[0]) * N.x + float(p[1]) * N.y + float(p[2]) * N.z
            self.shafts[sname] = Shaft(
                system, sname, J=inertias[sname], parent=self, housing=housing,
                position=position, axis=axis,
                w0=drive_speed * geometry.total_ratio(row))

        # ---- meshes -------------------------------------------------------
        radius = geometry.base_radius if use_base_radius else geometry.pitch_radius
        self.meshes: List[CompliantSpurGearMesh] = []
        for st in geometry.stages:
            fa = drive_flange if st.i_a == 0 else self.shafts[SHAFT_NAMES[st.i_a]].flange
            fb = self.shafts[SHAFT_NAMES[st.i_b]].flange
            self.meshes.append(CompliantSpurGearMesh(
                system, st.name, fa, fb,
                r_a=radius(st.z_a), r_b=radius(st.z_b),
                c=mesh_stiffness, d=mesh_damping, parent=self))

        # ---- diagnostics (Section 13) -------------------------------------
        w_crank = drive_flange.speed

        # Six signed compatibility residuals per bank: the four stage residuals
        # come from the meshes themselves; these two close the chain against the
        # crankshaft (three reversals to intake, four to exhaust).
        for row, label in ((3, "crank_to_intake"), (4, "crank_to_exhaust")):
            w_down = self.shafts[SHAFT_NAMES[row]].flange.speed
            self.output(f"residual_{label}",
                        w_down - geometry.total_ratio(row) * w_crank)

        # torque the crankshaft must supply to this bank
        self.output("driveTorque", -self.meshes[0].system.outputs[
            f"{self.meshes[0].path}.torque_a"])

        # total power flowing into the bank through the first mesh
        self.output("drivePower",
                    -self.meshes[0].system.outputs[f"{self.meshes[0].path}.torque_a"]
                    * w_crank)

        # total mesh damping loss
        self.output("meshDampingPower", sum(
            system.outputs[f"{m.path}.damping_power"] for m in self.meshes))

        # kinetic energy of the bank; with drivePower and meshDampingPower this
        # closes an energy balance over the whole subsystem
        self.output("kineticEnergy", sum(
            0.5 * inertias[n] * self.shafts[n].flange.speed ** 2
            for n in self.shafts))

    # ------------------------------------------------------------------
    @property
    def camshaft_flanges(self) -> Dict[str, RotFlange]:
        """Attachment points for the measured cam loads of Stage 6."""
        return {n: self.shafts[n].flange
                for n in ("intakeCamshaft", "exhaustCamshaft")}

    def shaft_names(self) -> List[str]:
        return [SHAFT_NAMES[i] for i in range(1, len(SHAFT_NAMES))]

    def residual_names(self) -> List[str]:
        return ([f"{self.path}.{m.name}.residual" for m in self.meshes]
                + [f"{self.path}.residual_crank_to_intake",
                   f"{self.path}.residual_crank_to_exhaust"])
