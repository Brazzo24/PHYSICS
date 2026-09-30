"""Validate the derivation against the Meijaard et al. (2007) benchmark bicycle."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from weave import linearize, benchmark_bicycle

lb = linearize(benchmark_bicycle(stiff_tyre=1e8))    # stiff tyre ~ no-slip wheels
ev = np.linalg.eigvals(lb.state_space(5.0)[0])
ev = ev[np.argsort(-ev.real)][:4]
print("Eigenvalues at 5 m/s")
print("  this model :", np.round(ev, 4))
print("  benchmark  : [-0.3229, -0.7753+-4.4649j, -14.078]")

us = np.linspace(3, 8, 5001)
mr = np.array([np.linalg.eigvals(lb.state_space(u)[0]).real.max() for u in us])
crit = us[np.where(np.diff(np.sign(mr)))[0]]
print(f"Critical speeds: weave {crit[0]:.3f} m/s (4.292), capsize {crit[1]:.3f} m/s (6.024)")
