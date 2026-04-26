# motodyn — modular motorcycle dynamics framework

A small Python framework for studying motorcycle chassis and suspension
behaviour. Designed for quick **parameter studies** first, full
**maneuver simulation** second — but the same parameter and state
objects serve both, so you don't have to rewrite anything to add an ODE
integrator later.

## Layout

```
motodyn/
├── parameters.py         # frozen dataclasses: MassProperties, Geometry, ...
├── state.py              # OperatingPoint (quasi-static), VehicleState (ODE-ready)
├── presets.py            # generic_sportbike(), generic_adventure()
├── models/
│   ├── load_transfer.py           # static + longitudinal load transfer
│   ├── suspension_kinematics.py   # anti-squat, anti-dive, ICs, Mozzi axis
│   └── chassis_rigid_body.py      # quasi-static equilibrium
└── analyses/
    └── parameter_sweep.py         # sweep / grid_sweep helpers
```

## Quick start

```python
from motodyn.presets import generic_sportbike
from motodyn.models.suspension_kinematics import anti_squat_ratio
from motodyn.analyses.parameter_sweep import sweep

bike = generic_sportbike()
print(f"Anti-squat: {anti_squat_ratio(bike) * 100:.1f} %")

# Sweep swingarm pivot height
results = sweep(
    bike,
    param_path="suspension.swingarm_pivot",
    values=[(0.42, z) for z in [0.26, 0.30, 0.34]],
    metric=anti_squat_ratio,
)
```

## Key design choices

1. **Parameters are frozen dataclasses** → reproducible sweeps.
   `Motorcycle.with_changes("suspension.k_rear_wheel", 110e3)` returns a
   new bike; it never mutates the original.
2. **Each physics module is independent.** You can compute anti-squat
   without ever importing the tyre or chassis module.
3. **Quasi-static by default, dynamic-ready by design.** Every analysis
   takes an `OperatingPoint`; a future ODE driver will just produce
   `OperatingPoint`s at each step and call the same functions.
4. **Sign conventions live in one place** (`parameters.py` docstring) so
   you can't accidentally mix up frames between modules.

## What's implemented

| Area | Status |
|---|---|
| Static + longitudinal load transfer | ✓ |
| Wheelie / stoppie thresholds | ✓ |
| Anti-squat (chain drive, side-view) | ✓ |
| Anti-dive (telescopic & hub-centre) | ✓ |
| Anti-rise at rear under braking | ✓ |
| Load-path split (spring vs. linkage) | ✓ |
| Rear IC (single-pivot + linkage override) | ✓ |
| Mozzi axis (side-view + combined roll/yaw) | ✓ |
| Quasi-static chassis equilibrium | ✓ |
| Pacejka tyre model | planned |
| ODE integrator (chicane etc.) | planned (hooks already in place) |
| Aero | planned |

## Examples

```
python examples/01_antisquat_sweep.py   # 1-D anti-squat study
python examples/02_braking_study.py     # dive & load transfer vs. deceleration
python examples/03_antisquat_grid.py    # 2-D heatmap
```

## Testing

```
pytest tests/
```

## Extending

Adding a new subsystem = one file in `motodyn/models/`. Follow the
pattern: pure functions taking `Motorcycle` and `OperatingPoint`,
returning a frozen dataclass. No global state, no mutation.
