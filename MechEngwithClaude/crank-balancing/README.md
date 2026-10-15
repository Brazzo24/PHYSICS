# crank-balancing

Mass balancing of crankshafts and balance shafts for **any inline or V engine** – built with a
motorcycle **V4** in mind (any bank angle, shared or split crankpins, any throw phasing).

This is a Python version and generalisation of
A. Dagna, C. Delprete, C. Gastaldi, *A General Framework for Crankshaft Balancing and
Counterweight Design*, Appl. Sci. 2021, 11, 8997 (`resources/`).

## Quick start

```bash
cd crank-balancing
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 16 tests, incl. the paper's numbers
streamlit run app.py         # opens in your browser
```

## Layout

```
app.py               Streamlit GUI (tabs: Engine · Balancing elements · Results · Theory)
crankbal/
  model.py           dataclasses Engine / Cylinder / Balancer, JSON save/load
  solver.py          phasor formulation, plane solve, time series, harmonic table
  presets.py         paper validation cases + generic V4 / I4 examples
  plots.py           Plotly figures (3-D geometry, end view, θ-plots, loci, rpm sweep)
configs/             presets as JSON (load them in the app's sidebar, or copy to make your own)
tests/               pytest: paper numbers, phasors vs. direct kinematics, full-balance checks
resources/           the paper
```

## Use it from Python / a notebook

```python
from crankbal import Engine, solve, harmonic_table, timeseries
from crankbal.presets import v4

eng = v4("my V4", vee_deg=75, throw_deg=(0, 180), split_deg=0,
         shafts_enabled=(False, True, False, False))     # +1x, -1x, +2x, -2x shafts
res = solve(eng)
for b in res.balancers:
    for p in b.planes:
        print(b.spec.name, p.y_mm, round(p.mass_g, 1), "g @", round(p.angle_deg, 1), "deg")
```

## Conventions

* **y** = crank axis, **θ** = crank angle measured from **+z** towards **+x** (positive rotation about +y).
* Per cylinder: `bank_deg` β (cylinder axis direction), `pin_deg` ψ (crankpin angle at θ = 0), `y_mm`.
  Cylinder at TDC when θ = β − ψ (shown in the app).
* Masses: `m_osc` (piston group + conrod small-end share), `m_rot` (conrod big-end share),
  `m_throw` (this cylinder's share of the throw's own unbalance – pin + webs without CW – reduced to r).
  The *Mass helper* in the app does the two-mass conrod split and web reduction for you.
* One **plane** = one lumped counterweight. Results are m·r per plane plus mass at the chosen r_c and the
  phase at θ = 0 (and relative to cylinder 1's pin for co-rotating elements).

## Method (short)

Every inertia force in the x-z plane is a complex number $c = F_z + jF_x$. With the 2nd-order slider-crank
expansion, each cylinder contributes rotating phasors of order ±1 and ±2:

$F_{osc,n} = \tfrac12 m_{osc} r\omega^2 k_n [e^{j(n\psi-(n-1)\beta)}e^{+jn\theta} + e^{j((n+1)\beta-n\psi)}e^{-jn\theta}]$, $k_1=1$, $k_2=\Lambda$.

Each balancer (crank CW = +1, shafts = ±1, ±2) solves its own harmonic for force *and* couple:
$\sum n^2U_k = -P_F$, $\sum n^2 y_k U_k = -P_M$ (exact with 2 planes, min-norm with more, force-only with 1).
This is the paper's Eqs. (10)–(23) in one matrix solve, valid for any bank angle and split pins.

## Validation (tests/test_crankbal.py)

| case | paper | here |
|---|---|---|
| I3, crank CW per web | 1.2413 kg | 1.2413 kg (2 webs lumped per plane → 2.4826 kg/plane) |
| I3, primary shaft CW | 0.4884 kg | 0.4884 kg |
| I3, secondary shaft CW | 0.0362 kg | 0.0362 kg |
| V4 90°, crank CW per web | 1.700 kg | 1.700 kg |

The paper's V4 balance-shaft masses are not reproduced on purpose: the paper treats the V reciprocating
forces as vertical only, here the full 2-D phasors are used (e.g. a 90° V with shared pins has *no* backward
primary component, so no counter-rotating primary shaft is needed).

## Limits of v1 / ideas for next steps

* 2nd-order expansion only (no 4th/6th order, no pin/crank offset) → exact-kinematics check.
* Inertia torque about the crank axis not included (M_y shown = rocking couple from shaft offsets only).
* Global rigid-body balance only (no bending / bearing loads).
* Counterweight geometry optimiser (paper Sec. 4).
