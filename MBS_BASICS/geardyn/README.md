# Gear dynamics, a notebook series

Gear-mesh dynamics in small steps: from a perfectly rigid tooth contact to a flexible
shaft-bearing-gear system with backlash. Each notebook takes 20–40 minutes. It derives what it
uses, runs it on one shared **reference pair** and ends with take-aways and exercises.

Reference pair: 24/48 teeth, module 2.5 mm, 20°, 15 mm face width (`gd.GearPair()`), with
illustrative round numbers, **not** data from a real gearbox. Override any field, e.g.
`gd.GearPair(module=3e-3, backlash=80e-6)`.

| # | notebook | what it settles | experiments |
|---|---|---|---|
| 1 | `01_rigid_mesh.ipynb` | involute geometry, line of action, contact ratio, rigid EOM (eliminated and Lagrange-multiplier forms), mesh force = load + inertia, bearing reactions, housing reaction torque, helical axial force and tilting couple | run-up, steady state, engine braking (flank change) |
| 2 | `02_mesh_stiffness_damping.ipynb` | ISO 6336 mesh stiffness, 2-DOF model, equivalent mass, mesh mode, damping | **A** gear 2 clamped, preload gear 1, release · **B** twisted free pair (LOA-momentum conservation) · **C** dynamic load factor vs rise time · **D** log-decrement identification |
| 3 | `03_time_varying_stiffness.ipynb` | single/double contact stiffness k(s), static transmission error, 1-DOF DTE equation, speed sweep (K_v), Campbell diagram, Floquet stability | tip relief, high contact ratio, manufacturing error vs load |
| 4 | `04_backlash.ipynb` | two unilateral flanks with a gap, restitution, amplitude-dependent frequency, rattle threshold, softening resonance | **A** clamped preload-and-release with backlash · **B** free pair · **C** idle rattle from order-2 speed irregularity · **D** jump phenomenon (sweep up/down) |
| 5 | `05_bearings_lateral.ipynb` | gear centres on bearing springs, coupling vector g, series stiffness, backlash change under load, coupled lateral-torsional modes | static load, modes vs bearing stiffness, preload-and-release with bearings, dynamic bearing force vs speed |
| 6 | `06_flexible_shafts.ipynb` | Euler–Bernoulli + torsion shaft FE, rigid / torsion / bending / full flexibility, statically determinate reactions, energy storage, modal reduction | **A** drivetrain wind-up released through the backlash (clunk) · **B** running at speed, rigid vs flexible |

## Files

| file | purpose |
|---|---|
| `geardyn.py` | the shared library: `GearPair`, involute drawing, rigid-mesh statics, `Model` (DOFs, masses, springs, bearings, gear meshes with backlash and time-varying stiffness, loads, fixed/prescribed DOFs, modal reduction with static shapes and constraint modes, energy bookkeeping), shaft FE, 1-DOF DTE model, Floquet multipliers |
| `test_geardyn.py` | consistency checks: geometry, energy balance (incl. backlash + time-varying k), LOA momentum, statics, beam FE vs analytic, reduced vs full models — `python -m pytest -q` |
| `build_notebooks.py` | regenerates the notebooks from `_build/nbXX_*.py`; `--execute` runs them; `python build_notebooks.py 04 --execute` for one |
| `_build/` | notebook sources (percent format), edit these rather than the `.ipynb` |

## Running

```bash
pip install numpy scipy sympy matplotlib jupyter nbformat nbclient pytest
python -m pytest -q
jupyter lab 01_rigid_mesh.ipynb
```

Run times on a laptop-class CPU: 01–02 a few seconds, 03 ≈ 4–5 min (speed sweeps), 04 ≈ 3–4 min,
05 ≈ 3–4 min, 06 ≈ 3 min. The sweeps are plain Python + `solve_ivp`, so they are slow but readable.

## Conventions

* Gear 1 (pinion) at the origin, gear 2 at (a, 0), shafts along +z.
* θ1 counter-clockwise positive, **θ2 clockwise positive**, so both grow in forward running.
* Transmission coordinate along the line of action: δ = r_b1 θ1 − r_b2 θ2 (+ lateral terms).
* Drive flank: gear 1 pushes gear 2, force on gear 2 along n_d = (sin α, cos α); coast flank n_r = (sin α, −cos α).
* Mesh force F > 0 on the drive flank, F < 0 on the coast flank.
* Bearing reaction = force the bearing exerts **on the shaft** (the housing sees −R).
* SI units inside the code.

## Building blocks for your own models

```python
import geardyn as gd
gp = gd.GearPair(backlash=60e-6)
m = gd.lateral_pair(gp, kb1=3e8, kb2=3e8, contact="backlash", stiffness="tv")
m.fix("th2")
q0 = m.static(Q)                      # preload
r = m.simulate(5e-3, q0=q0)           # release
r.mesh[0]["F"], r.bearing["B1"], r.energy_residual
```

## Literature

- Kahraman, Singh, *Non-linear dynamics of a spur gear pair*, J. Sound Vib. 142 (1990): backlash, jumps
- Özgüven, Houser, *Mathematical models used in gear dynamics: a review*, J. Sound Vib. 121 (1988)
- ISO 6336-1: mesh stiffness c′, c_γ, dynamic factor K_v, face load factor K_Hβ
- Linke, *Stirnradverzahnung* (Hanser): tooth stiffness, load distribution, practical design
- Niemann, Winter, *Maschinenelemente Bd. II*: gear dynamics and loading
- Parker, Lin and colleagues on mesh stiffness variation and parametric instability (planetary and spur)
- Smith, *Gear Noise and Vibration*: transmission error, practical noise causes
