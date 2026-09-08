"""Bank geometry -- one source of geometric truth (Section 5.1).

The shaft-position matrix is the only geometric input. Mesh centre vectors,
centre distances and lines of action are all *derived* from it, never entered
separately. That is Rule 3 of Section 16.3 made structural: there is no second
place where a centre distance could be typed in and drift.

Shaft rows, in order:
    0 crankshaft      (external -- the row exists so the geometry is complete)
    1 stepShaft
    2 idlerCamShaft
    3 intakeCamshaft
    4 exhaustCamshaft

Coordinate convention: shaft axes lie along x (transverse, parallel to the
crankshaft), so all shaft centres live in the y-z plane. The rear bank mirrors
the y coordinates (Section 5.3), which is a configuration assumption to be
replaced by CAD data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

SHAFT_NAMES = ["crankshaft", "stepShaft", "idlerCamShaft", "intakeCamshaft",
               "exhaustCamshaft"]


@dataclass(frozen=True)
class MeshStage:
    """One external spur-gear stage between two shafts."""

    name: str
    i_a: int          # driving shaft row
    i_b: int          # driven shaft row
    z_a: int          # driving tooth count
    z_b: int          # driven tooth count

    @property
    def ratio(self) -> float:
        """Nominal magnitude |w_b / w_a| = z_a / z_b."""
        return self.z_a / self.z_b


# The Section 5.2 ratio table
DEFAULT_STAGES: List[MeshStage] = [
    MeshStage("crank_to_step", 0, 1, 24, 40),        # 0.600000
    MeshStage("step_to_idler", 1, 2, 25, 51),        # 0.490196
    MeshStage("idler_to_intake", 2, 3, 51, 30),      # 1.700000
    MeshStage("intake_to_exhaust", 3, 4, 30, 30),    # 1.000000
]


@dataclass
class BankGeometry:
    """Shaft-position matrix plus tooth data for one camtrain bank."""

    shaft_pos: np.ndarray                       # (5, 3) [m]
    stages: List[MeshStage] = field(default_factory=lambda: list(DEFAULT_STAGES))
    module: float = 1.5e-3                      # [m]
    pressure_angle: float = np.deg2rad(20.0)
    axis: str = "x"

    def __post_init__(self):
        self.shaft_pos = np.asarray(self.shaft_pos, dtype=float)
        if self.shaft_pos.shape != (len(SHAFT_NAMES), 3):
            raise ValueError(f"shaft_pos must be {len(SHAFT_NAMES)}x3")

    # -- derived quantities -------------------------------------------
    def centre_vector(self, stage: MeshStage) -> np.ndarray:
        """r_ab = shaftPos[b] - shaftPos[a], exactly as Section 5.1 writes it."""
        return self.shaft_pos[stage.i_b] - self.shaft_pos[stage.i_a]

    def centre_distance(self, stage: MeshStage) -> float:
        return float(np.linalg.norm(self.centre_vector(stage)))

    def pitch_radius(self, z: int) -> float:
        return self.module * z / 2.0

    def base_radius(self, z: int) -> float:
        """Base radius. Mesh compliance acts along the line of action, so this
        is the correct lever arm for the compliant mesh, not the pitch radius.
        The ratio r_base_a / r_base_b is unchanged, so the ratio table holds."""
        return self.pitch_radius(z) * np.cos(self.pressure_angle)

    def nominal_centre_distance(self, stage: MeshStage) -> float:
        return self.pitch_radius(stage.z_a) + self.pitch_radius(stage.z_b)

    def total_ratio(self, i_shaft: int) -> float:
        """Signed speed ratio w_shaft / w_crank through the stage chain."""
        r = 1.0
        for st in self.stages:
            r *= -st.ratio
            if st.i_b == i_shaft:
                return r
        if i_shaft == 0:
            return 1.0
        raise ValueError(f"shaft row {i_shaft} not reachable through the stages")

    def signed_speeds(self, crank_speed: float) -> List[float]:
        return [crank_speed * self.total_ratio(i) for i in range(len(SHAFT_NAMES))]

    # -- validation ----------------------------------------------------
    def consistency(self) -> List[Tuple[str, float, float, float]]:
        """Compare geometric centre distances with the tooth-derived values.

        Returns (stage name, geometric [m], nominal [m], error [m]) per stage.
        A mismatch means the shaft-position matrix and the tooth counts
        disagree -- exactly the kind of silent error that a second, manually
        entered centre distance would hide.
        """
        out = []
        for st in self.stages:
            geo = self.centre_distance(st)
            nom = self.nominal_centre_distance(st)
            out.append((st.name, geo, nom, geo - nom))
        return out

    def check(self, tol: float = 50e-6) -> bool:
        return all(abs(e) <= tol for _, _, _, e in self.consistency())

    def report(self) -> str:
        lines = ["shaft-position matrix [mm]",
                 f"  {'shaft':<18}{'x':>10}{'y':>10}{'z':>10}"]
        for n, p in zip(SHAFT_NAMES, self.shaft_pos):
            lines.append(f"  {n:<18}{p[0]*1e3:>10.2f}{p[1]*1e3:>10.2f}{p[2]*1e3:>10.2f}")
        lines += ["", "geometry vs tooth data",
                  f"  {'stage':<20}{'geometric':>12}{'nominal':>12}{'error um':>12}"]
        for name, geo, nom, err in self.consistency():
            lines.append(f"  {name:<20}{geo*1e3:>12.4f}{nom*1e3:>12.4f}{err*1e6:>12.2f}")
        lines += ["", "ratio chain (Section 5.2)",
                  f"  {'stage':<20}{'|ratio|':>12}{'sign':>10}"]
        for st in self.stages:
            lines.append(f"  {st.name:<20}{st.ratio:>12.6f}{'reversal':>10}")
        return "\n".join(lines)

    def mirrored(self) -> "BankGeometry":
        """Rear bank: mirror the y coordinates (Section 5.3).

        This is a configuration assumption, not a universal rule. Replace with
        measured or CAD-derived rear-bank geometry when available. Both banks
        must keep the same physical crankshaft location, so row 0 is not
        mirrored.
        """
        pos = self.shaft_pos.copy()
        pos[1:, 1] *= -1.0
        return BankGeometry(pos, list(self.stages), self.module,
                            self.pressure_angle, self.axis)


def layout_from_bearing_angles(angles_deg, stages=None, module: float = 1.5e-3,
                               round_to: float = 1e-4) -> np.ndarray:
    """Build a plausible shaft-position matrix from a chain of bearing angles.

    Placeholder generator for the real CAD matrix. Angles are measured from
    +z toward +y, one per stage. Positions are rounded to ``round_to`` so the
    matrix looks like drawing data rather than an exact reconstruction -- which
    is what makes ``BankGeometry.consistency`` a real check instead of a
    tautology.
    """
    stages = list(DEFAULT_STAGES) if stages is None else list(stages)
    pos = np.zeros((len(SHAFT_NAMES), 3))
    for st, ang in zip(stages, np.deg2rad(angles_deg)):
        d = module * (st.z_a + st.z_b) / 2.0
        pos[st.i_b] = pos[st.i_a] + np.array([0.0, d * np.sin(ang), d * np.cos(ang)])
    return np.round(pos / round_to) * round_to
