# pycamtrain

A Python counterpart to the `KTM_Camtrain_MBS` Dymola package, built to compare
**workflow, capability and performance** between Modelica/Dymola and Python for
camtrain dynamics.

Stage 1 is complete: the framework skeleton exists and is verified against
closed-form solutions. Camtrain physics is layered on top from Stage 2 onward.

```
python3 tests/verify.py                       # 9 analytic checks
python3 examples/ex01_front_bank_chain.py     # Section 5.2 ratio table
```

Requires `numpy`, `scipy`, `sympy`, `matplotlib`.

---

## 1. The central design decision

Dymola and this framework solve the same mechanics by opposite routes.

| | Modelica / Dymola | pycamtrain |
|---|---|---|
| Modelling paradigm | Acausal equations on connectors | Minimal coordinates, declared directly |
| What a connection means | Generates `phi_a = phi_b`, `tau_a + tau_b = 0` | Shares one symbol; nothing to generate |
| Assembly | Flatten → index reduction → tearing → C code | Kane's method → mass matrix + forcing → `lambdify` |
| Result | DAE, typically index 3 reduced to index 1 | Pure ODE, `M(q)·ẏ = F(q,u,t)` |
| Constraint forces | Solved for at every step | Eliminated analytically, never computed |
| Initialisation | Nonlinear initialisation problem | Just an initial state vector |
| Kinematic loops | Resolved automatically by the translator | Must be handled explicitly |
| Component reuse | High — a flange is a true interface | Lower — components know their DOFs |

The practical consequence for camtrain work: an entire class of Dymola failure
modes disappears. Section 11.3 of the model package guide is a checklist of
structural singularities — unconnected `RealInput`, doubly-supplied
`Torque.torque`, over-determined diagnostics. None of those can occur here,
because there is no connection set to over-determine. What is lost is the
acausal reusability that makes a Modelica bank subsystem droppable into any
cranktrain model.

The core is **spatial, not torsional**. Every shaft is a real `RigidBody` with
a full inertia dyadic, restricted to one rotational DOF by construction rather
than by a constraint equation. Adding radial bearing motion later means giving
the same body two more coordinates — no framework change. The spatial pendulum
in the verification suite runs through exactly the same assembly code as the
gear train, which is the proof that this path is open.

## 2. Package layout

```
pycamtrain/
  core/
    system.py       MBSystem: coordinate/parameter/input registry,
                    Kane assembly, lambdify, ODE right-hand side
    solver.py       solve_ivp wrapper, Result, RunMetadata (wall time)
    component.py    hierarchical Component, dotted instance paths
    connectors.py   RotFlange, SpatialFrame
  components/
    rotational.py   Housing, Shaft, TorsionalSpringDamper,
                    CompliantSpurGearMesh, TorqueSource, SpeedSensor
    spatial.py      PendulumBody (spatial verification case)
tests/verify.py     9 analytic checks
examples/           worked models
results/            generated plots
```

Naming follows Section 15.1 of the guide: physical names for connectors,
lower camel case for instances, upper camel case for classes. Instance paths
are dotted (`frontBank.stepShaft.phi`), so the result namespace reads like a
Dymola result file.

## 3. Component mapping to the Dymola package

| Dymola class | pycamtrain | Notes |
|---|---|---|
| `MultiBody.World` | `Housing` | One instance, inertial reference (Section 3.1) |
| `ShaftRadialBush_Base` | `Shaft` | Currently rigid bearings; radial DOFs are Stage 4 |
| `CompliantSpurGearMesh` | `CompliantSpurGearMesh` | Line-of-action stiffness, damping, transmission error hook |
| `Torque` + source | `TorqueSource` | Prescribed `f(t)`, reacted by housing |
| Speed sensor | `SpeedSensor` | Pure diagnostic, adds no equations |
| `MeasuredCamLoad` | *Stage 6* | Angle integration, table lookup, activation ramp |
| PI controller | *Stage 4* | Needs a controller state — the first non-mechanical DOF |
| `CrankshaftSpeedIrregularity` | *Stage 8* | Order-harmonic speed reference |

### Mesh sign convention

```
delta = r_a·phi_a + r_b·phi_b − e(t)
F     = c·delta + d·deltȧ
T_a   = −r_a·F,   T_b = −r_b·F
```

Both pitch radii are positive for an external mesh, which is what produces the
speed reversal at every stage. The nominal ratio |ω_b/ω_a| is r_a/r_b = z_a/z_b,
matching the Section 5.2 table. This makes Rule 4 of Section 16.3 — sign
conventions explicit and independently testable — a property of the code rather
than a discipline.

## 4. Verification status

All 9 checks pass against closed-form results.

| Check | Relative error |
|---|---|
| Two-inertia torsional eigenfrequency | 9.2e-11 |
| Angular momentum drift (free-free) | 2.7e-16 N·m·s absolute |
| Energy drift over 0.5 s | 4.5e-9 |
| Gear-pair rigid-limit acceleration | 4.9e-5 |
| Gear-pair signed speed ratio | 1.2e-5 |
| Mesh compatibility residual (mean) | 7.7e-6 normalised |
| Mesh force vs `J_b·α_b/r_b` | 2.9e-3 |
| Mesh deflection vs `F/c` | 3.0e-3 |
| Spatial pendulum small-angle frequency | 6.3e-6 |

`examples/ex01_front_bank_chain.py` reproduces the Section 5.2 signed speeds
to machine precision:

```
crankshaft        6000.00 rpm
stepShaft        -3600.00
idlerCamShaft     1764.71
intakeCamshaft   -3000.00
exhaustCamshaft   3000.00
```

and shows zero mesh force from consistent initial speeds — the Section 11.1
acceptance criterion for absence of initialisation shock.

## 5. Performance observations so far

Wall times on the development machine, 5-DOF front-bank chain:

| Phase | Time |
|---|---|
| Symbolic assembly (5 DOF, 4 meshes) | 0.12 s |
| Integration, 20 ms at rtol 1e-9, Radau | 0.003 s |

The split matters for the Dymola comparison. Symbolic assembly here is the
analogue of Dymola's translation step and has the same cost profile: expensive
once, free afterwards. It is currently far cheaper than a Dymola translation,
but it grows superlinearly with DOF count because `lambdify` emits one flat
expression per matrix entry with no common-subexpression elimination — the job
Dymola's tearing does. Expect this to be the first thing that needs attention
somewhere past 20–30 DOF; `sympy.cse` in front of `lambdify` is the fix.

Integration is fast because the system is a small dense ODE with no constraint
projection. Stiff mesh stiffness (1e8 N/m) makes `Radau` the right default;
`DOP853` is faster on non-stiff cases and is used in the verification suite.

## 6. Roadmap

Following the Section 12 commissioning ladder:

| Stage | Content | Status |
|---|---|---|
| 1 | Framework skeleton, analytic verification | **done** |
| 2 | Single bank: 4 shafts, 4 meshes, unloaded, diagnostics | next |
| 3 | Shaft-position matrix, geometry-derived mesh vectors (Section 5.1) | |
| 4 | Radial bearing DOFs, PI mean-speed controller with anti-windup | |
| 5 | External crankshaft, two banks on a shared drive | |
| 6 | `MeasuredCamLoad`: angle integration, torque table, activation ramp | |
| 7 | Four cam loads, phase offsets, independent sign conventions | |
| 8 | Crankshaft speed irregularity, engine-order excitation | |
| 9 | Physical torque excitation with a slow mean-speed controller | |
| 10 | HTML reporting, parameter sweeps, acceptance metrics | |

Two design rules from Section 16.3 are already enforced structurally rather
than by convention: geometry is parameterised once and vectors derived from it
(Rule 3), and sign conventions are explicit and independently tested (Rule 4).
