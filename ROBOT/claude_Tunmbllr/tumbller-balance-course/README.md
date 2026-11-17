# Tumbller balance course

A four-notebook course on the ELEGOO Tumbller self-balancing robot: the physics,
the control theory, a simulation for tuning, and the step onto the hardware.
Each notebook is 30–45 minutes, derives what it uses, runs it on one shared
model and ends with take-aways and exercises.

| # | notebook | what it settles |
|---|---|---|
| 1 | `01_physics_and_model.ipynb` | coordinates, Lagrange equations (SymPy, checked against the code), reflected rotor inertia, PWM → torque motor model, linear state-space model, the unstable pole and how fast the robot falls |
| 2 | `02_control_theory.ipynb` | PD on tilt as spring + damper, why balancing alone runs away, the positive-feedback speed loop, the integral, cascade and time-scale separation, sampling, accelerometer/gyro fusion and the Kalman filter, LQR as a reference |
| 3 | `03_simulation_and_tuning.ipynb` | firmware-in-the-loop simulation, sliders, what each gain does, stability maps, saturation, gearbox friction, battery, parameter uncertainty, driving, a tuning order |
| 4 | `04_from_simulation_to_hardware.ipynb` | `balanceCar()` line by line, units of the gains, five experiments to measure your robot, sensor calibration, telemetry + live tuning sketch, comparing recordings with the model, hardware tuning procedure, troubleshooting |

## The shared model: `tumbller.py`

- `tb.Params()` — physical parameters. Motor data are from the GA37-520 datasheet
  in the Elegoo tutorial folder; masses, inertias, centre-of-mass height and
  encoder resolution are **estimates** until you measure them (notebook 4).
- `tb.Gains()` — the four firmware gains, in firmware units. `tb.STOCK` = 55 / 0.75 / 10 / 0.26.
- `tb.dynamics`, `tb.linearize` — nonlinear and linear plant.
- `tb.FirmwareController`, `tb.KalmanAngle` — Python ports of `balanceCar()` and
  `KalmanFilter.cpp` from the stock firmware (same names, same order of operations).
- `tb.simulate(...)` → `Result`; `tb.plot_run`, `tb.filmstrip`, `tb.animate`.
- `tb.stability_margin(p, gains)` — exact discrete stability check of plant + firmware.

```python
import tumbller as tb
res = tb.simulate(tb.Params(), tb.Gains(kp_balance=70), theta0_deg=0, push=(1.0, 0.1, 3.0))
tb.plot_run(res)
```

## Hardware side (`arduino/`, `tools/`)

- `arduino/Motor_Step_Test/` — stand-alone sketch: encoder edge counter, battery
  voltage, open-loop PWM steps. **Robot on a stand, it does not balance.**
- `arduino/Tumbller_Tuning/` — copy of the stock sketch (`Arduino/TumbllerComplete`)
  plus `Tuning.h`: telemetry and gain changes over serial (`P55` `D0.75` `S10` `I0.26` `Z0` `?`).
  The four changed lines in the `.ino` are marked `// TUNING`.
- `tools/tumbller_logger.py` — records the serial stream to CSV (`pip install pyserial`).

The Arduino code passes a C++ syntax check against stubs but has **not been
compiled for the Nano or run on a robot** — treat the first upload as a test.
The CSV files in `data/` are synthetic examples written by notebook 4.

## Requirements

`numpy scipy matplotlib sympy jupyter`; optional `ipywidgets` (sliders in notebook 3)
and `pyserial` (logger).

## Rebuilding

The notebooks are generated from `_build/nbXX.py`:

```
python build_notebooks.py --execute        # all
python build_notebooks.py --execute nb03   # one
```

Animation and slider cells are stored without output (file size); run them in Jupyter.
