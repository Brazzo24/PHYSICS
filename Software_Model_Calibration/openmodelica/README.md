# GearMeshBenchmark: OpenModelica twin of the Python model

`GearMeshBenchmark.mo` is a self-contained Modelica package. It does not need the Modelica Standard Library. It reproduces `../python_calibration/gearmesh_model.py` component by component.

## Run it
* **OMEdit:** File → Open → `GearMeshBenchmark.mo`, then simulate `Examples.Stage1`, `Examples.Stage2` or `Examples.Stage2Calibrated`. The experiment settings are stored in the models (DASSL, tolerance 1e-8, output interval 1 µs / 2 µs).
* **Command line:** run `omc run_all.mos`. It writes the three `*_res.csv` files, which takes about 10 s in total.
* **Cross-check:** run `python compare_openmodelica.py`. It overlays OpenModelica, Python and the digitised ValDyn curves and writes `om_vs_python_*.png`.

Tested with OpenModelica 1.27.2.

## Structure

| Modelica | Python equivalent | Notes |
|---|---|---|
| `Frame` connector | – | Potentials: **displacements** u, v, φ from the nominal position. Flows: fx, fy, t = cut forces acting on the component |
| `Components.Body` | gear dofs x, y, θ | Initial angular velocity `w0` |
| `Components.PointMass` | shaft nodes 8 / M / 9 | φ = 0 (no rotational dof) |
| `Components.ConstantSpeed` + `TorsionSpringDamper` | `Td = kd(Ωt−θ1)+cd(Ω−θ̇1)` | Excitation and drive spring |
| `Components.SpringDamper2D` | supports `k1x…c2y` | Linear x/y spring-dampers |
| `Components.ConstantTorque` | `T_drag` | Stage 1: 10 N·m |
| `Components.SineAngleTorque` | `T_sin_amp*sin(Ωt)` | SINUS10.txt |
| `Components.GearMesh2D` | `_mesh_forces` | Flanks A/B, backlash, k·δ + c·δ̇, no clipping (option `clipTension`), lash on the pitch circle |
| `Components.RadialBearingClearance` | bearing in `_rhs2` | `mode` 0 = circular, 1 = per plane, 2 = x only; plus friction torque |
| `Components.ShaftCondensed` | `shaft_condensed_K` | 2 Euler–Bernoulli elements, rotations condensed (function `Functions.shaftCondensedK`) |
| `Components.NonlinearSupport2D` | Block/Case tables | ValDyn table columns: disp [mm], force [N], damping [N·s/m] |
| `Examples.Stage1 / Stage2 / Stage2Calibrated` | `simulate_stage1/2` | Same parameter names where possible (`k_mesh`, `c_mesh`, `jn`, `w2_ratio`, …), all values in SI |

The output variables have the same names and units as the ValDyn plots: `angle_deg`, `lashA_mm`, `FA`, `gearX_mm`, `midY_mm`, `FXBlock` (force on the shaft), and so on.

## Verification (OpenModelica DASSL vs Python RK4)
* **Stage 1:** all 15 impacts agree to within 0.05° in angle and 1.4 N in peak force. Distances to the ValDyn data are identical (lash 0.0076 / 0.0081).
* **Stage 2 calibrated:** the two results agree to within micrometres until about 460–700° of driver rotation, then diverge. That divergence is the expected chaotic behaviour, the same as when the Python time step changes. The statistical objective against ValDyn is J = 0.029 for OpenModelica and 0.026 for Python, which is within the objective's noise.

## Notes for working in OMEdit
* Use displacement potentials, not absolute positions: deflections are µm against positions of about 0.13 m, and with tolerance 1e-8 the solver would otherwise lose them.
* Contact, clearance and table kinks generate state events. DASSL locates the contact onsets exactly, so no RK4 step-size question arises.
* The components have no Placement in the component models themselves. The two example diagrams are laid out, but the lines are straight; tidy them in OMEdit if you like.
* To calibrate in OpenModelica, change the top-level parameters of `Examples.Stage2` (`jn`, `k_mesh`, `clr`, `clrMode`, `speed_rpm`, `dampScale`). Alternatively, drive the simulation from Python via OMPython or FMU export and reuse `calib_tools.py`.
