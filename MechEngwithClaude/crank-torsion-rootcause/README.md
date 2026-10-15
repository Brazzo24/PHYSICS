# crank-torsion-rootcause

Hand-calc / reduced-model study that builds the evidence chain for the V4 crank failure next to the flywheel:

**excitation (gas + mass, per order) → firing order × mode shape → torsional response → fillet stress → Haigh / Miner → predicted vs observed life**, plus **counterfactuals** that remove one link at a time.

```bash
python run_study.py        # numpy + matplotlib only, ~10 s
```
Output goes to `results/`: 13 figures, `summary.md` (tables + key numbers) and `sweep.csv`.

## Files
| file | content |
|---|---|
| `inputs.py` | **all inputs**: every `# EST` value is an estimate and must be replaced |
| `torsion/excitation.py` | slider-crank kinematics (exact), Wiebe single-zone pressure scaled to IMEP, or CSV traces; gas and mass torque; order spectrum |
| `torsion/chain.py` | lumped torsional chain, eigenmodes, tuning to the FE frequency, modal and direct harmonic response, static mean-torque path |
| `torsion/analysis.py` | rpm sweep, section stress per order and per mode, modal excitation factor, resonance crossings, duty-cycle damage |
| `torsion/fatigue.py` | rainflow on the periodic stress signal, FKM mean-stress line, Basquin/Miner-Haibach |
| `torsion/plots.py` | report figures |
| `data/pressure_template.csv` | format for measured pressure traces (the model output, for illustration only) |

Angle conventions are the same as in `crank-balancing/crankbal`: β is the bank angle, ψ the pin angle, and TDC is where θ + ψ = β.

## Data to replace (in order of influence)
1. **Firing order**: `fire_on_second_tdc` per cylinder in `inputs.engine()`. This sets every phase in the phase star.
2. **Chain inertias/stiffnesses**: reduce them from your FE/MBS model. After tuning, only the ratios matter, so the mode *shape* must be right (compare fig 04 with the FE mode shape). Check that the section torque path is correct too.
3. **Damping ζ**: the stress scales roughly with 1/ζ at resonance. If you have dyno torsional-vibration data, fit ζ to the measured resonance peak.
4. **Section geometry and torsional Kt** from FE (local/nominal), plus **τ_W, M, k** for your material and surface treatment.
5. **Pressure**: set `pressure().csv_path = "data/pressure.csv"` with measured or GT-Power traces (columns `theta_deg` with 0 = firing TDC, then `p_bar_<rpm>`).
6. **m_osc, r, l, bore**.
7. **Duty cycle**: the logger rpm histogram, plus the engine hours at failure (`observed_life_h`).
8. **FE frequencies**: `chain().fe_modes_hz`, including the bending modes, so they show up in the Campbell diagram.

## Figures → report storyline
| # | figure | argument |
|---|---|---|
| 01 | pressure + torque over the cycle | plausible inputs |
| 02 | single-cylinder order spectrum | gas vs mass contribution per order |
| 03 | order 3 gas vs mass, amplitude and phase vs rpm | mass torque ∝ ω² dominates at high rpm; at that speed gas and mass are in **anti-phase**, so overrun and part load excite order 3 more strongly |
| 04 | mode shapes and modal spring torque | mode 1 puts its largest shaft torque in the throw 2 – flywheel section (the failure location) |
| 05 | Campbell with stress bubbles | which crossings are inside the operating range and how strong they are |
| 06/07 | phase star × mode shape, modal excitation factor | whether the V4 layout lets order 3 excite mode 1 (factor 0 means it is cancelled whatever the torque) |
| 08 | order-resolved fillet stress vs rpm | which order drives the peak |
| 09 | modal contribution | the peak comes from mode 1 and not from quasi-static twist |
| 10/11 | counterfactual stress and life | **causality**: remove order 3, detune ±10 %, double the damping, change firing, add the drivetrain |
| 12 | Haigh | margin to the endurance limit |
| 13 | damage per rpm bin | where the damage is accumulated in service. Predicted life ≈ observed life closes the chain |

## Limitations (say so in the report)
- **Torsion only.** Bending, flywheel wobble and gyroscopic whirl are not modelled. Fractography (45° helical fracture vs. flat bending fracture) decides whether this model is the right one.
- Steady-state constant speed and linear. Variable reciprocating inertia, which causes secondary resonances, is not included.
- All cylinders use the same pressure trace. Conrod rotary inertia is neglected. Kt is static.
- Crank-only free-free by default. The case `with primary drive + driveline` shows how much the system mode moves. Validate it with a dyno torsional-vibration measurement (encoders at the nose and the flywheel, order tracking).
