"""Time response to a short handlebar torque pulse at 200 km/h, with a minimal
roll-feedback rider so that capsize does not dominate the picture."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from weave import *
from weave import plots

FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
u = 200 / 3.6
pulse = lambda t: 20.0 if t < 0.05 else 0.0
base = from_legacy()
models = {
    "no powertrain gyro": base,
    "inline-4, x5 inertia, forward": scale_rotor(base.with_powertrain(Powertrain.inline4_sportbike()), 5.0),
    "inline-4, x5 inertia, reversed": scale_rotor(base.with_powertrain(Powertrain.inline4_sportbike(crank_direction=-1)), 5.0),
}
runs = {}
for lab, m in models.items():
    lb = linearize(m)
    K = rider_roll_pd(lb, u)                     # same rider gains for all
    ev = closed_loop_eigs(lb, u, K)
    print(f"{lab:32s} closed-loop max Re = {ev.real.max():+.3f} 1/s")
    runs[lab] = simulate(lb, u, t_end=2.5, dt=1e-3, torque=pulse, K=K)
plots.time_response(runs, os.path.join(FIG, "04_time_response.png"),
                    "Handlebar torque pulse (20 N m, 50 ms) at 200 km/h, rider roll feedback on")
print("figure ->", os.path.join(FIG, "04_time_response.png"))
