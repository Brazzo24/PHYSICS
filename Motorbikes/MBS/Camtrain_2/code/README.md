# pycamtrain

A Python counterpart to the `KTM_Camtrain_MBS` Dymola package, built to compare
**workflow, capability and performance** between Modelica/Dymola and Python for
camtrain dynamics.

Stages 1–3 of the Section 12 commissioning ladder are complete: the framework
is verified against closed-form solutions, and one full camtrain bank runs with
geometry-derived meshes and the Section 13 diagnostics.

```
python3 tests/verify.py                       # 9 analytic checks
python3 examples/ex01_front_bank_chain.py     # Section 5.2 ratio table
python3 examples/ex02_single_bank.py          # Stage 2: bank under a speed ramp
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
| Prescribed motion | A constraint the solver must enforce | A coordinate that was never created |
| Kinematic loops | Resolved automatically by the translator | Must be handled explicitly |
| Component reuse | High — a flange is a true interface | Lower — components know their DOFs |

The practical consequence for camtrain work: an entire class of Dymola failure
modes disappears. Section 11.3 is a checklist of structural singularities —
unconnected `RealInput`, doubly-supplied `Torque.torque`, over-determined
diagnostics. None of those can occur here, because there is no connection set
to over-determine. What is lost is the acausal reusability that makes a
Modelica bank subsystem droppable into any cranktrain model.

The core is **spatial, not torsional**. Every shaft is a real `RigidBody` with a
full inertia dyadic, restricted to one rotational DOF by construction rather
than by a constraint equation. Adding radial bearing motion later means giving
the same body two more coordinates — no framework change. The spatial pendulum
in the verification suite runs through exactly the same assembly code as the
gear train, which is the proof that this path is open.

## 2. Package layout

```
pycamtrain/
  geometry.py       BankGeometry: shaft-position matrix, derived centre
                    vectors, pitch/base radii, ratio chain, consistency check
  core/
    system.py       MBSystem: coordinate/parameter/input registry,
                    Kane assembly, lambdify (with CSE), ODE right-hand side
    solver.py       solve_ivp wrapper, Result, RunMetadata (wall time)
    component.py    hierarchical Component, dotted instance paths
    connectors.py   RotFlange, SpatialFrame
  components/
    rotational.py   Housing, Shaft, TorsionalSpringDamper,
                    CompliantSpurGearMesh, TorqueSource, SpeedSensor,
                    KinematicDrive
    bank.py         CamtrainBank -- one bank as a reusable subsystem
    spatial.py      PendulumBody (spatial verification case)
tests/verify.py     9 analytic checks
examples/           ex01 ratio table, ex02 Stage 2 bank
results/            generated plots
```

Naming follows Section 15.1: physical names for connectors, lower camel case
for instances, upper camel case for classes. Instance paths are dotted
(`frontBank.stepShaft.phi`), so the result namespace reads like a Dymola result
file.

## 3. Geometry: one source of truth

`BankGeometry` takes a 5×3 shaft-position matrix (Section 5.1) and tooth counts.
Everything else is derived — centre vectors by row subtraction, centre distances,
pitch and base radii, the signed ratio chain, and the rear-bank mirror. There is
no second place where a centre distance can be typed in and drift, which is
Rule 3 of Section 16.3 made structural rather than procedural.

That also buys a real validation: `BankGeometry.consistency()` compares the
geometric centre distance of every stage against the tooth-derived value. With
the placeholder layout rounded to 0.1 mm, as drawing data would be, the errors
come out at 0.1–35 µm — small enough to accept, large enough to prove the check
is not a tautology.

Mesh compliance acts along the **line of action**, so the compliant mesh uses
base radii (`r·cos α`, α = 20°), not pitch radii. The ratio is unchanged, so the
Section 5.2 table still holds exactly, but the mesh force and deflection are the
physically correct ones.

### Mesh sign convention

```
delta = r_a·phi_a + r_b·phi_b − e(t)
F     = c·delta + d·deltȧ
T_a   = −r_a·F,   T_b = −r_b·F
```

Both radii are positive for an external mesh, which is what produces the speed
reversal at every stage. This makes Rule 4 — sign conventions explicit and
independently testable — a property of the code rather than a discipline.

## 4. Component mapping to the Dymola package

| Dymola class | pycamtrain | Notes |
|---|---|---|
| `MultiBody.World` | `Housing` | One instance, inertial reference (Section 3.1) |
| `ShaftRadialBush_Base` | `Shaft` | Currently rigid bearings; radial DOFs are Stage 4 |
| `CompliantSpurGearMesh` | `CompliantSpurGearMesh` | Base-radius line of action, damping, transmission-error hook |
| Bank subsystem | `CamtrainBank` | 4 shafts, 4 meshes, 6 residuals; no World, no crankshaft |
| `Torque` + source | `TorqueSource` | Prescribed `f(t)`, reacted by housing |
| Speed reference | `KinematicDrive` | Ideal-controller limit of Section 10.1 |
| Speed sensor | `SpeedSensor` | Pure diagnostic, adds no equations |
| `MeasuredCamLoad` | *Stage 6* | Angle integration, table lookup, activation ramp |
| PI controller | *Stage 4* | Needs a controller state — the first non-mechanical DOF |
| `CrankshaftSpeedIrregularity` | *Stage 8* | Order-harmonic speed reference |

`KinematicDrive` deserves a note. In minimal coordinates a prescribed motion is
not a constraint the solver must enforce — it is a coordinate that was never
created, so the angle is a closed-form expression obtained by symbolically
integrating ω(t). This is an infinitely stiff speed controller, which is exactly
what Section 10.3 warns suppresses the torsional irregularity one usually wants
to study. It is the right tool for kinematic and ratio validation and the wrong
one for Stage 9.

## 5. Verification status

**Framework (`tests/verify.py`) — 9/9 pass.**

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

**Stage 2 (`examples/ex02_single_bank.py`) — 20/20 acceptance checks pass.**

A constant-speed run cannot validate Stage 2: with consistent initial speeds and
no load, every mesh force is identically zero. So the crankshaft is ramped
0 → 6000 rpm on a smooth half-cosine, which forces each mesh to transmit exactly
the torque needed to accelerate everything downstream of it.

| Check | Result |
|---|---|
| Signed speeds after the ramp vs Section 5.2 | ≤ 7.9e-8 relative |
| Six compatibility residuals, post-ramp | ≤ 5.5e-6 normalised |
| Six residuals decaying, not drifting | decay ratio 0.025 |
| Drive torque vs `J_reflected·α_crank` | 1.9e-4 relative |
| Last-stage mesh force vs `−J_exh·α_exh/r_base` | 2.6e-4 relative |
| Drive power vs `d(KE)/dt` + damping loss | 1.2e-4 normalised |

The residual check is worth explaining. After the ramp the residuals are not
zero — they are a damped oscillation at the mesh frequency, around 5e-6 of the
speeds they relate. Testing them against an arbitrary threshold is meaningless;
what matters is that they are small *and decaying*. A wrong ratio would produce a
residual that does neither, which is precisely the "no unexplained drift"
criterion of Section 13.1.

## 6. Performance

Wall times on the development machine, `Radau`, rtol 1e-8, 10 ms of simulation:

| Model | DOF | Assembly (CSE off) | Assembly (CSE on) | Solve (CSE off) | Solve (CSE on) |
|---|---|---|---|---|---|
| 1 bank | 4 | 0.110 s | 0.070 s | 0.018 s | 0.009 s |
| 2 banks | 8 | 0.168 s | 0.174 s | 0.034 s | 0.017 s |
| 4 banks | 16 | 0.508 s | 0.511 s | 0.071 s | 0.035 s |

Two findings, one of which corrects an earlier guess.

**CSE does not help assembly; it halves the solve.** The earlier assumption was
that common-subexpression elimination would pay off at assembly time by shrinking
the generated code. It does not — assembly cost is dominated by the Kane
derivation itself, not by `lambdify`. What it does do is cut right-hand-side
evaluation time roughly in half, consistently across model sizes. Since the
right-hand side is called ~10⁴ times per run, that is where the win is. CSE is
on by default.

**Assembly scales superlinearly, integration does not.** Four banks is 4× the DOF
but 4.6× the assembly time and only 3.9× the solve time. The expression count
grows linearly (108 operations per bank in `M` and `F`), so the superlinearity is
in the symbolic derivation, not the generated code. This is the direct analogue
of Dymola's translate-then-simulate split and has the same character: expensive
once, cheap per run. It is not yet a problem at camtrain scale — the full
two-bank model is 8 DOF — but it will dominate if flexible crankshaft segments
push the model past ~50 DOF.

## 7. Roadmap

| Stage | Content | Status |
|---|---|---|
| 1 | Framework skeleton, analytic verification | **done** |
| 2 | Single bank: 4 shafts, 4 meshes, unloaded, diagnostics | **done** |
| 3 | Shaft-position matrix, geometry-derived mesh vectors | **done** |
| 4 | Radial bearing DOFs, PI mean-speed controller with anti-windup | next |
| 5 | External crankshaft, two banks on a shared drive | |
| 6 | `MeasuredCamLoad`: angle integration, torque table, activation ramp | |
| 7 | Four cam loads, phase offsets, independent sign conventions | |
| 8 | Crankshaft speed irregularity, engine-order excitation | |
| 9 | Physical torque excitation with a slow mean-speed controller | |
| 10 | HTML reporting, parameter sweeps, acceptance metrics | |

Stage 4 is the first one that adds a non-mechanical state (the PI integrator)
and the first that makes a shaft translate — both are framework-level firsts
rather than more of the same.
