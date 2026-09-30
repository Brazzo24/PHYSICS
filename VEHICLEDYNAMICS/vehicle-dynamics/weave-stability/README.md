# weave — powertrain inertia and gyroscopic effects on motorcycle weave stability

A small, validated Python code base that linearises a single-track vehicle
(rear frame + rider, engine block, steering assembly, two wheels, any number of
spinning powertrain rotors) about straight running and computes capsize / weave /
wobble as functions of speed, with time-domain responses.

Built on the existing Pacejka Ch. 11 code (`pacejka_moto_stage12.py`, `_stage3.py`,
vendored unchanged in `weave/_legacy/`): geometry, masses, CoG heights, wheel spin
inertias, fork inertia, roll inertia / product of inertia, Magic-Formula tyre
linearisation and steering-damper value are all taken from `MotorcycleParams` /
`build_inertia` via `from_legacy()`.

## Quick start

```bash
pip install numpy matplotlib
python tests/test_weave.py            # 9 tests, incl. Meijaard benchmark
python examples/01_validate_benchmark.py
python examples/02_baseline_modes.py
python examples/03_powertrain_study.py
python examples/04_time_domain.py
python examples/05_compare_legacy_stage3.py
```

```python
from weave import *
bike = from_legacy(powertrain=Powertrain.inline4_sportbike())
lb   = linearize(bike)                      # ~50 ms
m    = modes_at(lb, 200/3.6)                # capsize / weave / wobble
print(m["weave"].frequency, m["weave"].damping)
```

## Why a new equation set (important)

The stage-3 `build_A_matrix` is a reduced heuristic (diagonal mass matrix, ad-hoc
gyro terms). Compared with the validated model it gives a real pole that grows to
+30 1/s with speed and a weave frequency that *falls* with speed (`examples/05`).
So the legacy geometry/tyre/parameter code is reused, but the equations of motion
are derived afresh and checked against the Meijaard et al. (2007) benchmark bicycle
(eigenvalues at 5 m/s and critical speeds 4.292 / 6.024 m/s reproduced to 4 digits).

## Model

* Coordinates `[Y, psi, phi, delta]` (lateral position, yaw, roll, steer); pitch and
  height eliminated exactly through the two wheel-ground contact conditions
  (2nd order, so steer-fall / gravity stiffness is complete).
* The Lagrangian is built from exact rigid-body kinematics and expanded to second
  order with forward-mode AD (`weave/jets.py`) — no symbolic algebra, no finite
  differences. Every rotating body (wheels, crank, clutch/input shaft, output
  shaft) enters through the same Lagrangian, so gyroscopic terms are not hand-coded;
  a test confirms a pure rotor adds exactly `dC = H (e_psi e_phi^T - e_phi e_psi^T)`.
* Tyres: linear slip + camber force at the wheel/ground contact, first-order lag
  (relaxation length `sigma`), applied a pneumatic trail behind the contact centre.
* Constant forward speed (longitudinal DOF is cyclic and decouples at linear order).
  Speed enters as a polynomial (`H0 + u H1 + u^2 H2`), so a whole speed sweep costs
  one derivation.
* Conventions: x forward, y left, z up, roll +right lean, steer +left, wheel spin
  +y. Trail > 0 means the steering axis meets the ground ahead of the contact patch.

## Powertrain description (`weave/params.py`)

`Powertrain` = list of `Rotor`s (axial inertia, stage `crank | input | output`,
sense, axis) + primary ratio, gear table, final ratio, gear, `crank_direction`.
The input shaft counter-rotates against the crank automatically (primary gear
pair). Presets: `inline4_sportbike()`, `v_twin_cruiser()`. Non-gyroscopic engine
effects (mass, position) live in the `engine block` body; `shift_body()` moves it
with the total COM held fixed.

## Results with the default (legacy-derived) parameters

* Wheel gyros dominate: switching them off moves the capsize pole from about +0.5 to
  +3.8 1/s at 216 km/h (`figures/02_modes_vs_speed.png`).
* Powertrain angular momentum is 4–8 % of the two wheels' (inline-4, per gear); effect
  on weave damping is a few percent at most (heavy-flywheel twin about -3.6 %,
  inertia x5 about -6 %, worst near 150 km/h). A reversed crank helps slightly
  (+1 % damping, capsize growth -0.7 %). Gear choice at constant speed is a
  second-order effect.
* Moving the engine mass (COM fixed) changes weave damping by roughly -6 % (up) to +8 % (down) per 10 cm —
  larger than the gyro effect of a normal engine.
* Time response (`figures/04_time_response.png`): curves of the variants are almost
  indistinguishable.

## Known limits / assumptions to review

* Added assumptions not present in the legacy code: rear-group pitch/yaw inertia
  (45 / 12 kg m^2), wheel masses (8 / 10 kg), engine block (55 kg at x=0.55, z=0.42 m,
  carved out of the mainframe mass), tyre relaxation length 0.25 m, front-assembly
  inertia shape. All in `from_legacy(...)` arguments.
* Knife-edge wheels (no crown radius / overturning couple), no drive or brake force,
  no load transfer, rigid frame, no aerodynamics, rigid rider (the simple roll-PD
  rider in `simulate.py` exists only to keep capsize from dominating time plots).
* Mode identification is by modal participation. Around 60–75 km/h the weave merges
  with real tyre/lateral roots; the classified weave jumps there (visible as a kink
  in the weave plots). This is the physics of the root locus, not a bug.
* I could not access the earlier chat "Powertrain inertia and gyroscopic effects on
  motorcycle weave stability"; scope follows its title. Anything specific from that
  chat (parameters, equations, plots) can be dropped into `from_legacy()` / new rotors.
