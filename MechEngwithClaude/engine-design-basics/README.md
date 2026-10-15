# Engine design basics

A notebook series to rebuild the *big picture* around the crank-drive MBS work:
how the main engine components are loaded, and how they are sized by hand to
the right order of magnitude. Each notebook is 20–40 minutes, explains the
physics, derives the formulas it uses, runs them on one shared reference engine
and ends with take-aways and exercises.

The level is deliberately basic: **rough design calculations and hand calcs**
that tell you which load case drives which feature. FE, detailed fatigue
methods and full thermodynamic models come later (or live in the MBS / FE work).

## Reference engine

Every notebook uses `ec.Engine()` — an *illustrative* 1000 cc four-cylinder race
engine (81 × 48.5 mm, 105 mm rod, ε = 13.5, rev limit 16 000 rpm). The numbers
are round and plausible, **not** data of any real engine. Override any field,
e.g. `ec.Engine(rod_length=0.100, m_piston_group=0.25)`, to try your own.

## Notebooks

| # | notebook | status | what it settles |
|---|---|---|---|
| 1 | `01_crank_kinematics.ipynb` | ✅ | exact and series piston motion, λ, mean piston speed, peak acceleration, orders 1-2-4-6, rod angular motion |
| 2 | `02_thermo_basics.ipynb` | ✅ | polytropic compression, Otto cycle, energy chain fuel → imep → power, Wiebe single-zone pressure trace, spark timing vs p_max, fmep/bmep |
| 3 | `03_crank_drive_forces.ipynb` | ✅ | rod mass split, gas vs inertia vs speed, rod/side/tangential/radial force, torque & energy check, torque orders (gas vs inertia, order-3 crossover), crank-pin polar diagram |
| 4 | `04_conrod_handcalc.ipynb` | ✅ | tension/compression load cases, I-section, two-plane buckling (Johnson–Euler), per-speed Goodman fatigue, whipping, small end, cap, bolt preload |
| 5 | `05_piston_and_pin.ipynb` | planned | crown temperature & strength (plate), ring-land/ring loads, skirt side load & pressure, pin bending/ovalisation/shear, boss pressure, piston-pin offset |
| 6 | `06_crankshaft_strength.ipynb` | planned | one throw as a beam (statically determinate), pin/web/journal stresses, fillet stress concentration, combined bending + torsion fatigue, overlap |
| 7 | `07_plain_bearings.ipynb` | planned | big-end and main bearing polar diagrams, specific load, Sommerfeld number, short-bearing (Ocvirk) film thickness, Petroff friction, oil flow & temperature rise |
| 8 | `08_multicylinder_balancing.ipynb` | planned | free forces and moments by order for single, twin, I4, V4 (90°, crank-pin offset); firing order; links to the crank-balancing app |
| 9 | `09_torque_and_flywheel.ipynb` | planned | multi-cylinder torque superposition, cyclic irregularity, flywheel sizing, which torsional orders survive for a given layout |
| 10 | `10_crank_torsional_basics.ipynb` | planned | lumped torsional model of the crank (Holzer), natural frequencies, order excitation, Campbell diagram — bridge to the MBS model |
| 11 | `11_thermo_gas_exchange.ipynb` | planned | volumetric efficiency, valve flow & timing basics, simple λ_a(n) → torque curve shape, heat transfer (Woschni) |
| 12 | `12_materials_and_fatigue.ipynb` | planned | S–N curves, mean-stress (Goodman/Haigh), surface/size/notch factors, typical engine materials |

Order is a suggestion; 5–7 only need notebooks 1–3.

## Files

| file | purpose |
|---|---|
| `enginecalc.py` | the small shared library: geometry, kinematics, Wiebe pressure model, crank forces, strength helpers, materials, plot style |
| `test_enginecalc.py` | consistency checks (derivatives, energy balance, motored polytropic) — `python -m pytest -q` |
| `build_notebooks.py` | regenerates the notebooks from `_build/nbXX.py`; `--execute` also runs them |
| `_build/` | notebook sources — edit these rather than the `.ipynb` files |

## Running

```bash
pip install numpy scipy matplotlib jupyter nbformat nbclient pytest
jupyter lab 01_crank_kinematics.ipynb
```

## Conventions

θ = 0 at firing TDC; a cycle runs −360° … +360° (intake, compression, expansion,
exhaust). Piston travel $x$ from TDC, positive towards the crank. Axial forces
positive when pushing the piston towards the crank. SI units inside the code.

## Literature for going deeper

- Köhler, Flierl — *Verbrennungsmotoren: Motormechanik, Berechnung und Auslegung des Hubkolbenmotors* (the mechanics reference for this series)
- van Basshuysen, Schäfer — *Handbuch Verbrennungsmotor*
- Heywood — *Internal Combustion Engine Fundamentals* (thermodynamics)
- Pischinger, Klell, Sams — *Thermodynamik der Verbrennungskraftmaschine*
- Hoag, Dondlinger — *Vehicular Engine Design*
- Shigley — *Mechanical Engineering Design* (columns, fatigue, bolted joints); VDI 2230 for bolts
