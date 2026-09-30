"""
Powertrain inertia & gyroscopic effects on weave stability.

  A  How big is the powertrain angular momentum vs the wheels?  (per gear)
  B  Weave / wobble / capsize change for inline-4, heavy-flywheel twin,
     counter-rotating crank, and x5 rotating inertia.
  C  Sensitivity map: rotating-inertia scale x speed, forward vs reversed crank.
  D  Gear choice at constant speed (engine speed changes, wheel speed does not).
  E  Non-gyroscopic effect: where the engine mass sits (COM held fixed).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from dataclasses import replace
from weave import *
from weave import plots

FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
speeds = np.linspace(8.0, 70.0, 32)
base = from_legacy()
sw = lambda m, s=speeds: speed_sweep(linearize(m), s)

# ---------------------------------------------------------------- A: momentum
u0 = 55.0
print(f"A) angular momenta at {u0*3.6:.0f} km/h  [N m s]")
g = gyro_ratio(base, u0)
print(f"   front wheel {g['front_wheel']:.0f}   rear wheel {g['rear_wheel']:.0f}")
labels, vals, rats = [], [], []
pt6 = Powertrain.inline4_sportbike()
for gear in range(1, 7):
    pt = replace(pt6, gear=gear)
    H = pt.angular_momentum(u0, base.geom.r_rear)
    labels.append(str(gear)); vals.append(100 * H / (g['front_wheel'] + g['rear_wheel']))
    rats.append(pt.overall_crank_ratio())
    print(f"   inline-4 gear {gear}: crank {pt.crank_rpm(u0, base.geom.r_rear):7.0f} rpm  "
          f"net powertrain H = {H:6.1f}  ({vals[-1]:4.1f} % of wheels)")
plots.gear_bars(labels, vals, rats, os.path.join(FIG, "03a_powertrain_momentum_by_gear.png"),
                "net powertrain angular momentum / wheels [%]",
                f"Inline-4 powertrain at {u0*3.6:.0f} km/h")

# ------------------------------------------------------------ B: variants
ref = sw(base)
variants = {
    "inline-4, forward crank": sw(base.with_powertrain(Powertrain.inline4_sportbike())),
    "inline-4, crank reversed": sw(base.with_powertrain(Powertrain.inline4_sportbike(crank_direction=-1))),
    "heavy-flywheel twin": sw(base.with_powertrain(Powertrain.v_twin_cruiser())),
    "inline-4, rotating inertia x5": sw(scale_rotor(base.with_powertrain(Powertrain.inline4_sportbike()), 5.0)),
}
plots.delta_vs_speed(ref, variants, os.path.join(FIG, "03b_powertrain_delta_vs_speed.png"),
                     "Powertrain gyro effect relative to the model without powertrain gyro")
plots.pole_map({"no powertrain gyro": ref, "inline-4 x5, forward": variants["inline-4, rotating inertia x5"],
                "twin": variants["heavy-flywheel twin"]},
               os.path.join(FIG, "03b_pole_map.png"))
print("\nB) change vs no-powertrain-gyro model at 60 / 150 / 250 km/h  (weave zeta, capsize growth)")
for lab, s in variants.items():
    row = []
    for kmh in (60, 150, 250):
        i = int(np.argmin(abs(speeds * 3.6 - kmh)))
        row.append(f"{kmh}: dz={s['weave_zeta'][i]-ref['weave_zeta'][i]:+.4f}  dcap={s['capsize_re'][i]-ref['capsize_re'][i]:+.3f}")
    print(f"   {lab:32s}", "   ".join(row))

# ------------------------------------------------------------ C: maps
scales = np.linspace(0.0, 10.0, 21)
spd = np.linspace(14.0, 70.0, 15)
maps = {}
for lab, direction in (("forward crank", 1), ("reversed crank", -1)):
    M = np.zeros((scales.size, spd.size))
    ref_z = np.array([modes_at(linearize(base), u)["weave"].damping for u in spd])
    for i, sc in enumerate(scales):
        pt = Powertrain.inline4_sportbike(crank_direction=direction)
        lb = linearize(scale_rotor(base.with_powertrain(pt), sc))
        M[i] = [100 * (modes_at(lb, u)["weave"].damping - z) / z for u, z in zip(spd, ref_z)]
    maps[lab] = M
plots.sensitivity_maps(maps, spd, scales, os.path.join(FIG, "03c_sensitivity_map.png"),
                       "Weave damping ratio: powertrain rotating inertia x speed")

# ------------------------------------------------------------ D: gears at const speed
print("\nD) constant 200 km/h, inline-4, gear choice")
u = 200 / 3.6
lb0 = linearize(base)
m0 = modes_at(lb0, u)
for gear in (3, 4, 5, 6):
    pt = replace(Powertrain.inline4_sportbike(), gear=gear)
    m = modes_at(linearize(base.with_powertrain(pt)), u)
    print(f"   gear {gear}: {pt.crank_rpm(u, base.geom.r_rear):6.0f} rpm  weave zeta {m['weave'].damping:.4f} "
          f"(base {m0['weave'].damping:.4f})  capsize {m['capsize'].growth:+.3f} (base {m0['capsize'].growth:+.3f})")

# ------------------------------------------------------------ E: engine mass location
print("\nE) engine block moved with total COM fixed (pure inertia-distribution effect), 200 km/h")
for lab, kw in (("baseline", {}), ("engine 10 cm higher", dict(dz=0.10)), ("engine 10 cm lower", dict(dz=-0.10)),
                ("engine 10 cm forward", dict(dx=0.10)), ("engine 10 cm rearward", dict(dx=-0.10))):
    m = base if not kw else shift_body(base, "engine block", **kw)
    md = modes_at(linearize(m), u)
    print(f"   {lab:22s} weave {md['weave'].frequency:5.2f} Hz z={md['weave'].damping:.4f}   "
          f"wobble {md['wobble'].frequency:5.2f} Hz z={md['wobble'].damping:.4f}   capsize {md['capsize'].growth:+.3f}")
print("\nfigures ->", os.path.abspath(FIG))
