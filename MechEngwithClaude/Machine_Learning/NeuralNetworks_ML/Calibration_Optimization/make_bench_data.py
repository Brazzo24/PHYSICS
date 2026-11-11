"""
make_bench_data.py - produces the synthetic "test-bench" data in ./data

SPOILER WARNING: this file contains the true parameter values that the
notebooks try to recover.  Don't read it before finishing notebook 03 :-)

Data sets (order amplitudes of speed fluctuation at crank and gearbox input):
  bench_A      full load, reality == model structure + measurement noise
  bench_A_val  50 % load, same "bench", kept aside for validation
  bench_B      full load, reality has effects the model does NOT contain
               (hysteretic clutch damper, alternator rotor on a flexible shaft)
  bench_B_val  50 % load version of B
"""
import os
import numpy as np
import driveline as dl

TRUE_MULT = dict(k_prim=1.25, k_damper=0.82, k_shaft=1.15, k_cush=1.30,
                 c_damper=1.90, c_cush=0.70, J_gbx=1.12, gas_scale=0.94)
X_TRUE = dl.NOMINAL * np.array([TRUE_MULT[n] for n in dl.NAMES])

# structure the calibration model does not know about (data set B)
_f2 = dl.natural_frequencies(X_TRUE)[0][1]
STRUCT_B = dict(damper_eta=X_TRUE[dl.NAMES.index("c_damper")] * 2 * np.pi * _f2
                / X_TRUE[dl.NAMES.index("k_damper")],        # same damping at mode 2
                J_alt=4.0e-3, k_alt=1.4e4, eta_alt=0.06)

NOISE_REL = 0.03      # 3 % multiplicative (order-tracking scatter)
NOISE_ABS = 0.01      # rad/s encoder noise floor


def measure(load, structure, seed):
    rng = np.random.default_rng(seed)
    amp = dl.simulate(X_TRUE, load=load, structure=structure)
    amp = amp * np.exp(NOISE_REL * rng.standard_normal(amp.shape))
    amp = np.abs(amp + NOISE_ABS * rng.standard_normal(amp.shape))
    return amp


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "data"), exist_ok=True)
    for name, load, struct, seed in [("A", 1.0, None, 1), ("A_val", 0.5, None, 2),
                                     ("B", 1.0, STRUCT_B, 3), ("B_val", 0.5, STRUCT_B, 4)]:
        np.savez(os.path.join(here, "data", f"bench_{name}.npz"),
                 rpm=dl.RPM, orders=dl.ORDERS, amp=measure(load, struct, seed), load=load)
        print("wrote bench_%s" % name)
