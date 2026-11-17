# okane-moto — companion code to O'Kane (2018), "High Fidelity Modelling of Motorcycle Dynamics" (PhD thesis, Maynooth)

Status after session 3 (2026-11-16): chapters 1–5 covered, including the controls of Sections 4.7 / 4.12 / 4.13. Delivered as `okane-moto.zip` (not yet placed in a repo on disk).

## What exists
- `okane/` Python package (numpy, scipy, sympy, matplotlib): `frames.py` (Table 4.4), `params.py`, `symbolic.py` (SymPy derivation + codegen to `_generated.py`), `kinematics.py` (exact loop closure + thesis closed forms 4.17–4.25), `model.py` (Kane assembly: K, bias, input matrix for the 21 inputs of Table 4.9, A/B/R extraction, C matrix), `forces.py` (fork/shock tables, chain, drag fit, crank, rider offset), `tyre.py` (Magic Formula as printed in App. A, pneumatic trail, optional Mzr, relaxation, parameter sets of Tables 5.1–5.5, `TyreSet` coupling), `analysis.py` (static trim, linearisation, `BikeSystem` = model + force blocks + tyres with 4 relaxation states, cornering trim, lateral modes), `controls.py` (PID, set-point shaper, speed/roll controllers, powertrain with engine map and Table 6.1 gears, traction control, ABS, Kalman filter, gyro steering law, `simulate` 800 Hz loop, `run_observer`), `benchmark.py` (Meijaard 2007 bicycle in the motorcycle model), `pendulum.py` (ch. 3).
- 6 executed notebooks (sources in `nbsrc/*.py`, jupytext): ch. 3 methods; ch. 4 frames/kinematics; force blocks; Kane dynamics + verification; ch. 5 tyres; controls.
- 23 pytest tests.

## Key decisions / deviations from the thesis
- Thesis publishes no geometry/inertia/suspension data → `cbr600rr_estimate()` is an ESTIMATE (wheelbase 1.390 m, rake 24°, trail 95 mm; from thesis only: 270 kg, wheel spin inertias 0.484/0.638, tyre radii, drag fit).
- Axis convention as thesis: 1 fwd, 2 up, 3 right; yaw/steer + = left, roll + = right, forward wheel spin negative; driving axle torque negative, braking positive.
- Pitch mu = q14 from exact loop closure (default) or eq. (4.17); yaw q10 = heading of the frame/rear-wheel plane, so thesis q13 = q17 = 0.
- Full nonlinear Kane equations, nothing dropped; thesis' x_dot = Ax + Bu_in + Rw form is extracted from it (alpha/gamma split is an interpretation).
- Tyre profile: measured polynomials of Tables 5.9/5.10 by default (gamma read as an angle), `profile="torus"` for an exact torus (used in notebooks 2 and 4).
- Axle torques Tfwy/Trwy are internal (wheel vs fork lower / swing-arm); tyre moments about the vertical act through the non-spinning carrier.
- Magic Formula in ISO system as printed (thesis negates pKy1, pKy6): alpha* = −Vsy(right)/|Vx|, Fy(MF) positive left; `TyreSet` flips to model axes. Aligning moment = trail × side-slip force at zero camber; Mx, My not applied; Mzr optional, off.
- Tyre data missing in the thesis taken from Sharp, Evangelou & Limebeer (2004): Fz0 1100/1600 N, trail parameters (Table III), lateral carcass stiffness from their relaxation formula, radial stiffness 130/141 kN/m. Longitudinal carcass stiffness estimated.
- Controls: structures from the thesis; all gains, slip-target tables and noise covariances tuned here. Engine curves digitised from Figs. 4.20/4.21. Discrete 800 Hz loop with zero-order hold, Heun integration.
- Plant for time simulation: `TyreSet.thesis(scale_carcass=True)` (front carcass stiffness scaled with optimised pKy1, relaxation length ≈ 0.2 m) and steering damper 15 N m s/rad.
- Observer steering law uses the recorded roll angle (g0) in addition to the rate-gyro terms of Fig. 4.36.

## Findings
- Printed H coefficients (p. 73) are the exact gradient of (4.17); thesis formulas for front camber and kinematic steer angle are exact.
- (4.17) is exact when roll or steer is zero, about 1° off at 60° roll / 10° steer; H11 ~100x too small there. Printed q13 misses its dominant terms.
- Energy conserved to 1e-6 J (torus profile); momentum and angular momentum conserved in free motion.
- Benchmark bicycle (Meijaard 2007) reproduced to 3 digits (weave −0.7747±4.4656j, capsize −0.3227, castering −14.08 at 5 m/s; weave speed 4.292 m/s).
- Modal analysis must keep yaw q10 as a state: u7, u8 are inertial-frame speeds, so heading is not cyclic.
- Steady state of thesis 7.4.2/7.5 (20 m/s, 50° roll) with Table 5.5 tyres: steer −0.84° (thesis ≈ −0.8°), rear slip 0.82° (0.83°), yaw rate −0.480 (−0.457) rad/s; steering torque −25 N m (thesis ≈ −7) not reproduced. Original Table 5.2/5.4 tyres need ~1.5° more steer into the turn.
- Fig. 5.7 numbers (6.2° / 1.7°) not reproducible with any of the tabulated parameter sets.
- Stability with estimated chassis data: weave 1→5 Hz, stable; capsize slowly unstable; wobble unstable with the Table 5.5 front tyre unless carcass stiffness is scaled and a damper is used.
- Roll PD through steering torque couples into wobble (≈18 Hz): Kp 300 / Kd 60 is stable 15–50 m/s with the 15 N m s/rad damper; Kp 600 / Kd 130 destabilises wobble from ~35 m/s. `RollController` default: Kp 300, Ki 450 (integration band 0.08 rad), Kd 60.
- Rate-gyro-only steering feedback does not stabilise the capsize mode of this model; roll estimate drifts (rms ~17° in the demo) without the roll-angle term.
- Demos: TC prevents a first-gear full-throttle crash at 45° roll (slip ≤ 0.084 vs spin-out at 0.8 s); ABS on low grip: lock without, slip −0.29 with Kabs 2 (thesis example value), −0.07 with Kabs 10; observer with wrong model (no drag, +10 % torque): speed error 9 m/s without Kalman filter, 0.7 m/s with it; position and heading drift in all cases.
- Thesis profile polynomials are not an exact envelope (rear worse); small spurious energy error (~1e-5 relative).

## Next sessions
- Chapter 6 (sensing/data processing: brake pressure → torque, gear reconstruction), chapter 7 (coast-down simulations, root locus 7.8, turning radius 7.5), chapter 8 (application examples).
- Replace estimated parameters with measured ones; revisit carcass stiffness / pneumatic trail (wobble); real recorded data for the observer.
