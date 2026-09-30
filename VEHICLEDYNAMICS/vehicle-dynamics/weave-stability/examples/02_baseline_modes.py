"""Baseline weave / wobble / capsize map of the legacy-parameter motorcycle,
with the wheel gyros switched on and off to show what they contribute."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from weave import *
from weave import plots

FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
speeds = np.linspace(6.0, 70.0, 65)                     # 22 ... 250 km/h

base = from_legacy()                                       # no powertrain gyro
sweeps = {
    "wheel gyros on (baseline)": speed_sweep(linearize(base), speeds),
    "front wheel gyro off": speed_sweep(linearize(without_gyro(base, which_wheels=("front",))), speeds),
    "both wheel gyros off": speed_sweep(linearize(without_gyro(base)), speeds),
}
plots.modes_vs_speed(sweeps, os.path.join(FIG, "02_modes_vs_speed.png"),
                     "Legacy-parameter motorcycle: capsize, weave and wobble vs speed")
lb = linearize(base)
print(f"Rider-off modes (baseline)\n{'km/h':>6} {'capsize':>9} {'weave f':>8} {'weave z':>8} {'wobble f':>9} {'wobble z':>9}")
for u in (10, 20, 30, 40, 50, 60, 70):
    m = modes_at(lb, u)
    print(f"{u*3.6:6.0f} {m['capsize'].growth:9.3f} {m['weave'].frequency:8.2f} {m['weave'].damping:8.3f}"
          f" {m['wobble'].frequency:9.2f} {m['wobble'].damping:9.3f}")
print("figure ->", os.path.join(FIG, "02_modes_vs_speed.png"))
