"""Data model for the crankshaft torsional root-cause study.

Angle conventions are identical to ``crank-balancing/crankbal`` (Dagna et al. 2021):
    crank angle theta measured from +z towards +x, positive rotation about the crank axis
    bank_deg  beta : cylinder axis direction
    pin_deg   psi  : crankpin angle at theta = 0
    cylinder i is at TDC when theta + psi_i = beta_i   ->  theta_TDC,i = beta_i - psi_i
A 4-stroke cylinder has two TDCs per 720 deg; ``fire_on_second_tdc`` picks which one fires:
    theta_fire,i = (beta_i - psi_i) mod 360 + 360 * fire_on_second_tdc

Units: inputs in mm, g, deg, bar, rpm, MPa; everything is converted to SI internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cylinder:
    name: str
    bank_deg: float
    pin_deg: float
    node: str                       # name of the torsional-chain node (crank throw) it drives
    fire_on_second_tdc: bool = False

    @property
    def tdc_deg(self) -> float:
        return (self.bank_deg - self.pin_deg) % 360.0

    @property
    def fire_deg(self) -> float:
        return self.tdc_deg + (360.0 if self.fire_on_second_tdc else 0.0)


@dataclass
class Engine:
    bore_mm: float
    r_mm: float                     # crank radius = stroke / 2
    l_mm: float                     # conrod length
    m_osc_g: float                  # reciprocating mass per cylinder (piston, rings, pin, rod small-end share)
    cylinders: list[Cylinder]
    p_crankcase_bar: float = 1.0


@dataclass
class PressureModel:
    """Single-zone closed-cycle model with Wiebe heat release, scaled to an IMEP target curve.

    If ``csv_path`` is given, measured/simulated traces are used instead (see README).
    """
    compression_ratio: float = 13.5
    gamma: float = 1.32
    p_intake_bar: float = 1.05      # ram-air plenum
    p_exhaust_bar: float = 1.15
    soc_deg: float = -20.0          # start of combustion, deg rel. firing TDC
    duration_deg: float = 50.0
    wiebe_a: float = 5.0
    wiebe_m: float = 2.0
    evo_deg: float = 130.0          # exhaust valve opening (blow-down to p_exhaust until 180)
    imep_curve: list[tuple[float, float]] = field(default_factory=list)   # (rpm, IMEP bar)
    csv_path: Optional[str] = None


@dataclass
class Spring:
    a: str
    b: str
    k_Nm_rad: float
    tunable: bool = True            # scaled when tuning the chain to the FE frequency


@dataclass
class Chain:
    """Lumped torsional model. Node inertias in kg m^2 (already reduced to crank speed)."""
    nodes: dict[str, float]
    springs: list[Spring]
    output_node: str                                 # where mean torque is reacted (static path)
    add_recip_inertia: bool = True                   # adds m_osc r^2 / 2 to each driven node
    tune_to_fe_hz: Optional[float] = None            # scale tunable springs so that ...
    tune_mode: int = 1                               # ... this mode (0 = rigid body) hits the FE value
    zeta: float = 0.015                              # modal damping ratio (all elastic modes)
    fe_modes_hz: list[float] = field(default_factory=list)   # overlay on the Campbell diagram


@dataclass
class Section:
    """Failure section = one spring of the chain (its torque is the section torque)."""
    spring_ab: tuple[str, str]
    d_outer_mm: float
    d_inner_mm: float = 0.0
    kt: float = 1.0                                   # torsional stress concentration at the fillet
    label: str = "failure section"


@dataclass
class Material:
    """Local fatigue properties at the failure fillet (shear), FKM-style."""
    name: str
    tau_w_MPa: float                # local fully-reversed shear fatigue strength at N_D (incl. surface/size)
    mean_sens_M: float              # mean stress sensitivity: tau_a,eq = tau_a + M * tau_m
    k_slope: float = 8.0            # S-N slope (FKM: 8 for shear, 5 for normal stress)
    N_D: float = 1.0e6
    haibach: bool = True            # Miner-Haibach: slope 2k-1 below tau_w (else elementary Miner)


@dataclass
class DutyCycle:
    rpm_bins: list[float]           # bin centres
    time_frac: list[float]          # time share per bin (normalised internally)
    observed_life_h: Optional[float] = None


@dataclass
class Study:
    engine: Engine
    pressure: PressureModel
    chain: Chain
    section: Section
    material: Material
    duty: DutyCycle
    rpm_min: float = 3000.0
    rpm_max: float = 21000.0
    rpm_step: float = 50.0
    rpm_operating: tuple[float, float] = (8000.0, 20000.0)
    orders_max: float = 12.0
    rpm_detail: float = 18000.0     # rpm for single-cycle plots
    highlight_order: float = 3.0
    out_dir: str = "results"
