# okane-moto

Companion code and notebooks for chapters 1–4 of

> T. G. O'Kane, *High Fidelity Modelling of Motorcycle Dynamics*, PhD thesis,
> National University of Ireland Maynooth, 2018.

11-d.o.f. motorcycle multibody model (main frame, swing-arm, fork lower,
steering assembly, two wheels) derived with Kane's method, with the thesis'
co-ordinates, frames and input vector.  Tyre model (ch. 5) and controllers
(4.12–4.13) are not included yet.

## Layout

| Path | Thesis | Content |
|---|---|---|
| `notebooks/01_ch1-3_multibody_methods.ipynb` | Ch. 3 | cart–pendulum by Newton, Lagrange, Kane → eq. (3.7) |
| `notebooks/02_ch4_frames_and_kinematics.ipynb` | 4.2–4.4, 4.8 | d.o.f., frames, ξ/γ₁/δ′, loop closure, q13…q19, linearisation (Figs. 4.27–4.32) |
| `notebooks/03_ch4_force_blocks.ipynb` | 4.5–4.7 | suspension tables, chain, crank, drag fit, rider offset |
| `notebooks/04_ch4_kane_dynamics.ipynb` | 4.10, 4.11, 4.14 | partial velocities, K, A/B/R form, C matrix, verification |
| `okane/frames.py` | Table 4.4 | direction-cosine matrices, front wheel frames G, H |
| `okane/params.py` | Fig. 4.24 | parameters; **estimated** CBR600RR data set |
| `okane/symbolic.py` | 4.10 | SymPy kinematics + code generation (`_generated.py`) |
| `okane/kinematics.py` | 4.8 | exact loop closure, thesis closed forms, partial velocities |
| `okane/model.py` | 4.10–4.11 | Kane assembly, state-space extraction, output matrix |
| `okane/forces.py` | 4.5–4.7 | input blocks |
| `okane/analysis.py` | – | static trim, linearisation, steady-turn trim |
| `okane/pendulum.py` | Ch. 3 | the three derivations |
| `tests/` | – | 14 verification tests (`pytest -q`) |

## Use

```bash
pip install -e .[dev]        # or just run the notebooks from ./notebooks
pytest -q
./build_notebooks.sh         # regenerate notebooks from nbsrc/*.py (jupytext)
python -m okane.symbolic     # re-derive okane/_generated.py (≈3 s)
```

```python
from okane import cbr600rr_estimate, Motorcycle, make_q, make_inputs
p, design = cbr600rr_estimate()
bike = Motorcycle(p)                      # pitch_model="exact", coef_rates=True
k = bike.config(q, u)                     # positions, partial velocities, ...
udot = bike.accelerations(q, u, u_in)     # K udot = Bg u_in - cb
```

## Conventions

Axis 1 forward, 2 up, 3 right (thesis Figs. 4.4/4.5). Yaw and steer positive
to the left, roll positive to the right, pitch positive nose-up, forward
rolling wheels have negative spin rate. Input sign conventions are documented
at the top of `okane/model.py`.

## Where this deviates from the thesis

1. **Parameters are estimates.** The thesis does not publish geometry,
   masses, inertias, spring/damper or chain data. Values taken from the text
   are marked as such in `params.py` / `forces.py`.
2. **Pitch angle** from the exact vector-loop closure by default; eq. (4.17)
   is available as `pitch_model="thesis"`.
3. **Yaw reference**: q10 is the heading of the frame / rear-wheel plane, so
   the thesis' q13 and q17 are zero by construction (they are still evaluated
   in notebook 2 for comparison).
4. **No terms discarded** in the equations of motion; the thesis' simplified
   `A, B, R` structure is extracted from the full model
   (`Motorcycle.state_space`, `accelerations_thesis`). The split into
   `alpha` / `gamma` is an interpretation, the thesis does not print them.
5. **Tyre profile** is a torus (crown radii from Table 5.6); tyre forces are
   plain inputs, a linear vertical spring is used as placeholder in demos.
6. Axle torques `Tfwy`, `Trwy` are internal (wheel ↔ fork lower / swing-arm).

## Verification status

* Ch. 3: all three derivations identical to eq. (3.7).
* Printed coefficients H, H1…H11 (p. 73) equal the symbolic gradient of (4.17).
* Thesis formulas for γ₁ and δ′ equal the exact frame construction.
* Partial velocities and bias accelerations checked by finite differences.
* Energy conserved to 1e-6 J (of 5e4 J) in free 3-D motion with springs.
* Static wheel loads equal elementary statics; steady-turn force balance.
* Not possible: number-for-number comparison with the thesis' model output.
