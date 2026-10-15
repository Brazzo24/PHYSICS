"""All study inputs in one place.

Every value tagged  # EST  is an engineering estimate for a ~1000 cc racing V4
(90 deg vee, 180 deg crank, 70 deg split pins) and MUST be replaced with real data.
Values tagged  # PLACEHOLDER  only make the plots meaningful (e.g. observed life).
"""
from __future__ import annotations

import copy

import numpy as np

from torsion.config import (Chain, Cylinder, DutyCycle, Engine, Material, PressureModel,
                            Section, Spring, Study)

VEE = 90.0
SPLIT = 70.0


def engine(firing: str = "A") -> Engine:
    """Same layout as crankbal preset 'V4 90 deg - 180 deg crank, 70 deg split pins'.

    TDC angles (theta_TDC = beta - psi): A1 315, B1 335, A2 135, B2 155 deg.
    firing 'A' (EST): B2 155 - A1 315 - A2 495 - B1 695  -> intervals 160/180/200/180 deg
    firing 'B'      : A1 315 - B1 335 - A2 495 - B2 515  -> intervals 20/160/20/520 deg ("twin pulse")
    -> set fire_on_second_tdc to YOUR firing order.
    """
    second = {"A": dict(A1=False, B1=True, A2=True, B2=False),
              "B": dict(A1=False, B1=False, A2=True, B2=True)}[firing]
    cyl = [
        Cylinder("A1", -VEE / 2, 0.0, "throw1", second["A1"]),
        Cylinder("B1", +VEE / 2, 0.0 + SPLIT, "throw1", second["B1"]),
        Cylinder("A2", -VEE / 2, 180.0, "throw2", second["A2"]),
        Cylinder("B2", +VEE / 2, 180.0 + SPLIT, "throw2", second["B2"]),
    ]
    return Engine(bore_mm=81.0,          # EST (MotoGP bore limit)
                  r_mm=24.25,            # EST stroke 48.5 mm
                  l_mm=101.0,            # EST lambda ~ 0.24
                  m_osc_g=300.0,         # EST piston+rings+pin+clips+rod small-end share
                  cylinders=cyl)


def pressure() -> PressureModel:
    return PressureModel(
        compression_ratio=13.5, gamma=1.32, p_intake_bar=1.05, p_exhaust_bar=1.15,   # EST
        soc_deg=-20.0, duration_deg=50.0, wiebe_a=5.0, wiebe_m=2.0, evo_deg=130.0,  # EST
        imep_curve=[(3000, 11.0), (8000, 15.0), (13000, 17.5), (17000, 18.0),       # EST (bar)
                    (19000, 17.0), (21000, 15.5)],
        csv_path=None,   # e.g. "data/pressure.csv" -> measured traces override the model
    )


def chain(tune_hz: float | None = 1000.0) -> Chain:
    """Crank-only lumped model (free-free, like your FE free-free run).

    Inertias kg m^2, stiffnesses Nm/rad.  The tunable springs are scaled so that the first
    elastic mode matches the FE frequency -> only the *ratios* below matter after tuning.
    Better: replace by inertias/stiffnesses reduced from your FE/MBS crank model.
    """
    return Chain(
        nodes={"nose": 0.35e-3,       # EST cam-drive gear, pulse rotor, front web
               "throw1": 3.0e-3,      # EST webs + pin + counterweights + rod rotating share
               "throw2": 3.0e-3,      # EST
               "flywheel": 8.0e-3},   # EST flywheel + primary drive gear + generator rotor
        springs=[Spring("nose", "throw1", 1.0e5),       # EST
                 Spring("throw1", "throw2", 1.5e5),     # EST
                 Spring("throw2", "flywheel", 1.5e5)],  # EST  <- failure section
        output_node="flywheel",
        tune_to_fe_hz=tune_hz,       # your FE: first torsional mode ~1000 Hz
        tune_mode=1,
        zeta=0.015,                  # EST 1-2 % typical for an undamped crank in oil-film bearings
        fe_modes_hz=[1000.0],        # PLACEHOLDER: list all FE modes (bending ones too) for the Campbell
    )


def add_drivetrain(ch: Chain) -> Chain:
    """Primary drive + clutch + soft driveline, reflected to crank speed (all EST)."""
    ch = copy.deepcopy(ch)
    i_prim = 1.8                                   # EST crank : clutch
    ch.nodes["clutch"] = 6.0e-3 / i_prim**2        # EST clutch basket+hub+gear at clutch speed
    ch.nodes["vehicle"] = 0.5                      # EST gearbox+chain+wheel+bike, at crank speed
    ch.springs.append(Spring("flywheel", "clutch", 2.7e5, tunable=False))   # EST gear mesh c*r_b^2
    ch.springs.append(Spring("clutch", "vehicle", 2.0e3, tunable=False))    # EST driveline (chain, cush)
    ch.output_node = "vehicle"
    return ch


def section() -> Section:
    return Section(spring_ab=("throw2", "flywheel"),
                   d_outer_mm=38.0,     # EST journal / shaft diameter at the crack
                   d_inner_mm=0.0,      # EST bore (0 = solid)
                   kt=1.6,              # EST torsional Kt of the fillet (take from FE: local / nominal)
                   label="throw 2 - flywheel")


def material() -> Material:
    return Material(name="34CrNiMo6 QT, nitrided (EST)",
                    tau_w_MPa=300.0,     # EST local shear fatigue strength incl. surface treatment
                    mean_sens_M=0.17,    # EST FKM: M_tau = 0.577*(0.00035*Rm - 0.1), Rm ~ 1100 MPa
                    k_slope=8.0, N_D=1.0e6, haibach=True)


def duty() -> DutyCycle:
    """rpm histogram (time share). EST: race-lap like distribution - replace by logger data."""
    bins = np.arange(3000.0, 20001.0, 500.0)
    frac = (0.15 * np.exp(-0.5 * ((bins - 9000) / 2000) ** 2)
            + 0.55 * np.exp(-0.5 * ((bins - 15000) / 1800) ** 2)
            + 0.30 * np.exp(-0.5 * ((bins - 17800) / 900) ** 2))
    frac[bins > 19000] *= 0.3       # limiter region rarely reached
    return DutyCycle(rpm_bins=bins.tolist(), time_frac=frac.tolist(),
                     observed_life_h=15.0)   # PLACEHOLDER engine hours at failure


def baseline() -> Study:
    return Study(engine=engine("A"), pressure=pressure(), chain=chain(), section=section(),
                 material=material(), duty=duty(),
                 rpm_min=3000.0, rpm_max=21000.0, rpm_step=50.0,
                 rpm_operating=(8000.0, 20000.0),   # EST
                 rpm_detail=18000.0, highlight_order=3.0, out_dir="results")
