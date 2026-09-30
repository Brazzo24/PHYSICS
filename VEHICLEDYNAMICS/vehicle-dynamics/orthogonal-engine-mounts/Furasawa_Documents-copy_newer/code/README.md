# Orthogonal Engine Mount System — code and walkthrough

An implementation of **Furusawa et al., SAE 830228 (Yamaha, 1983), "Orthogonal
Engine Mount System"**, plus the extra machinery needed to apply it to a real
machine.

```
orthomount.py                    the library
params_rz250.py                  reconstructed parameters for the paper's own engine
orthogonal_engine_mounts.ipynb   the level-by-level walkthrough of the paper
test_orthomount.py               verification against every closed form in the paper
math_primer/                     seven short notebooks building the maths from scratch
```

## Where to start

* **Comfortable with modal analysis?** Go straight to
  `orthogonal_engine_mounts.ipynb`.
* **Want the machinery built up first?** Start at `math_primer/README.md` — seven
  short notebooks on toy systems, ending with the paper's own derivation done
  symbolically. Notebook 4 is the one that matters; the rest are scaffolding for
  it.

## Quick start

```bash
pip install numpy scipy matplotlib sympy jupyter
python3 test_orthomount.py          # should print a wall of [ok]
jupyter lab orthogonal_engine_mounts.ipynb
```

## The idea in four lines

A mode responds to an excitation only through the inner product
`u_r^T F` — the modal participation factor. Make that product zero and the mode
is invisible to the excitation, at any frequency. Furusawa's insight: for an
engine excited by a pure rolling couple, one of the two roll–yaw modes can be
made exactly orthogonal to the excitation, provided the **inertial** cross
coupling and the **elastic** cross coupling are matched:

```
R = I_zx / I_z = -K_phipsi / K_psipsi          (paper Eq. 16)
```

You then place that one surviving mode below the usable rev range and make
everything else as stiff as the chassis wants — instead of trading isolation
against stiffness along a single axis.

## Library layers

| layer | what it does | scope |
|---|---|---|
| `RigidBody`, `Mount`, `MountSystem` | assemble 6×6 M and K from an arbitrary mount layout; eigenanalysis, modal participation, damped FRFs, transmitted force | general |
| `R_from_inertia`, `solve_beta_for_R`, `omega1_paper`, `delta_paper`, … | the paper's closed-form design algebra, Eqs. (16)–(32) | the paper |
| `Engine`, `Cylinder` | slider-crank primary/secondary forces and couples from crank phasing, recip mass, rod ratio, cylinder tilt and spacing | beyond the paper |

## Sign convention — read this before trusting a result

Frame: `x` forward, `y` lateral, `z` up, right-handed. DOF order
`[x, y, z, phi, theta, psi]`.

Tilt angles (`alpha` inertia axis, `beta` elastic axis, `delta` rocking axis) are
right-handed rotations about **+y**, so a positive angle takes `+x` toward `-z`.
That is the convention that reproduces the paper's Eqs. (20) and (21) verbatim.
Only the *relative* geometry of the three angles is physical; a sign slip flips
`R`, `I_zx`, `K_phipsi` and `delta` together.

## Beyond the paper

* real crank excitation instead of an assumed moment about `x`
* arbitrary mount positions and orientations, no symmetry assumed — the elastic
  centre need not sit on the CG
* hysteretic damping, complex FRFs, force transmitted into the chassis
* the feasibility bound `|R| <= (q-1)/(2 sqrt(q))`, which says how anisotropic
  the mounts must be before the scheme is possible at all
* a sensitivity study: how much the second mode wakes up when `alpha`, `beta` or
  `q` are off
* an inverse solver (`synthesise` in the notebook) that fits mount positions and
  rates to frequency targets plus the orthogonality condition

## Caveat on `params_rz250.py`

SAE 830228 publishes `alpha ≈ 30°`, `beta ≈ 10°` the other way, the mount
arrangement in words, and six natural frequencies (Table 2). It does **not**
publish inertias, mount coordinates or rates. Everything in `params_rz250.py`
marked `[est]` or `[fit]` is a reconstruction chosen to land on those
frequencies. It is a good sanity target, not the real machine.
