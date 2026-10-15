"""Data model for crankshaft mass balancing.

Coordinate system (same as Dagna, Delpretre, Gastaldi, Appl. Sci. 2021, 11, 8997):
    y : crankshaft axis (positive rotation about +y, right-handed)
    z : reference "up" direction, crank angle theta measured from +z towards +x
    x : completes the right-handed system

All user-facing inputs are in mm, g, deg, rpm. Internally SI is used.

Angles
------
bank_deg  beta  : direction of the cylinder axis (crank centre -> cylinder head),
                  measured from +z towards +x.  Inline engine: 0.  V-engine: +/- V/2.
pin_deg   psi   : crankpin angle at theta = 0, same convention.
                  Cylinder i is at TDC when theta + psi_i = beta_i.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Cylinder:
    name: str
    bank_deg: float          # cylinder axis direction  [deg]
    pin_deg: float           # crankpin angle at theta=0 [deg]
    y_mm: float              # axial position of the conrod centre line [mm]
    m_osc_g: float           # reciprocating mass: piston+rings+pin+clips + conrod small-end share [g]
    m_rot_g: float           # rotating conrod share (big-end) [g]
    m_throw_g: float = 0.0   # crank-throw own unbalance (pin + webs, without CW) reduced to crank radius,
                             # share attributed to this cylinder [g]


@dataclass
class Balancer:
    """A set of counterweight planes rotating with order*omega in direction `sense`.

    The crankshaft counterweights are simply a Balancer with order=1, sense=+1, x=z=0.
    The solver returns one complex unbalance U_k (mass x radius, phase at theta=0) per plane.
    """
    name: str
    order: int               # 1 = primary, 2 = secondary
    sense: int               # +1 co-rotating with the crank, -1 counter-rotating
    enabled: bool = True
    planes_y_mm: list[float] = field(default_factory=lambda: [-100.0, 100.0])
    r_c_mm: float = 30.0     # CG radius of the counterweights on this shaft [mm]
    x_mm: float = 0.0        # shaft position in the x-z plane (only for plotting & roll moment My)
    z_mm: float = 0.0


@dataclass
class Engine:
    name: str
    r_mm: float              # crank radius (half stroke)
    l_mm: float              # conrod length (centre-centre)
    rpm: float
    cylinders: list[Cylinder]
    # Balancing setup
    crank_cw: Balancer = field(default_factory=lambda: Balancer("Crankshaft counterweights", 1, +1,
                                                                 planes_y_mm=[-50.0, 50.0], r_c_mm=30.0))
    fwd_primary_on_crank: float = 1.0   # share of the forward-rotating 1st-order reciprocating
                                        # component assigned to the crank counterweights (0..1).
                                        # The rest goes to the primary co-rotating shaft (if enabled).
    shafts: list[Balancer] = field(default_factory=list)
    notes: str = ""

    # ---- derived ---------------------------------------------------------
    @property
    def Lambda(self) -> float:
        return self.r_mm / self.l_mm

    @property
    def omega(self) -> float:
        import math
        return self.rpm * 2.0 * math.pi / 60.0

    def shaft(self, order: int, sense: int) -> Balancer | None:
        for s in self.shafts:
            if s.order == order and s.sense == sense:
                return s
        return None

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Engine":
        d = dict(d)
        d["cylinders"] = [Cylinder(**c) for c in d["cylinders"]]
        d["crank_cw"] = Balancer(**d["crank_cw"])
        d["shafts"] = [Balancer(**s) for s in d.get("shafts", [])]
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Engine":
        return cls.from_dict(json.loads(Path(path).read_text()))


def default_shafts(r_c_mm: float = 30.0, l_sh1: float = 200.0, l_sh2: float = 200.0,
                   x1: float = 100.0, x2: float = 200.0, z: float = 0.0,
                   enabled=(False, False, False, False)) -> list[Balancer]:
    """The four possible balance shafts (paper Fig. 3/4): 1x co/counter, 2x co/counter."""
    return [
        Balancer("Primary co-rotating (+1x)", 1, +1, enabled[0], [-l_sh1 / 2, l_sh1 / 2], r_c_mm, +x1, z),
        Balancer("Primary counter-rotating (-1x)", 1, -1, enabled[1], [-l_sh1 / 2, l_sh1 / 2], r_c_mm, -x1, z),
        Balancer("Secondary co-rotating (+2x)", 2, +1, enabled[2], [-l_sh2 / 2, l_sh2 / 2], r_c_mm, +x2, z),
        Balancer("Secondary counter-rotating (-2x)", 2, -1, enabled[3], [-l_sh2 / 2, l_sh2 / 2], r_c_mm, -x2, z),
    ]
