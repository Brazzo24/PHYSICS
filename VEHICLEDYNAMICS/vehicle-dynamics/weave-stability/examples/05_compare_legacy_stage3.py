"""Why a new model?  Compare the legacy stage-3 matrix with the validated one."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from weave import *
from weave._legacy.pacejka_moto_stage12 import MotorcycleParams
from weave._legacy.pacejka_moto_stage3 import build_inertia, build_A_matrix

p = MotorcycleParams(); ip = build_inertia(p)
lb = linearize(from_legacy(p))
print(f"{'km/h':>5} | legacy stage-3 eigenvalues (osc. pairs, real)            | this model: capsize, weave, wobble")
for u in (10, 20, 30, 40, 50):
    ev = np.linalg.eigvals(build_A_matrix(u, p, ip))
    ev = ev[np.argsort(-ev.real)]
    m = modes_at(lb, u)
    old = "  ".join(f"{e.real:+.1f}{e.imag:+.0f}j" if e.imag > 0 else f"{e.real:+.1f}" for e in ev if e.imag >= 0)
    print(f"{u*3.6:5.0f} | {old:55s} | cap {m['capsize'].growth:+.2f}  weave {m['weave'].frequency:.2f} Hz  wobble {m['wobble'].frequency:.1f} Hz")
print("\nlegacy: real pole grows to +30 1/s with speed and the weave frequency FALLS with speed;")
print("new   : capsize pole stays ~0.5-3 1/s and the weave frequency RISES with speed (as measured on real motorcycles).")
