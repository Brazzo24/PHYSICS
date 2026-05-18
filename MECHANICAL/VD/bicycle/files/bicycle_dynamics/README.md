# Bicycle Dynamics and Control
### Based on Åström, Klein & Lennartsson (2005)

A multi-fidelity Python simulation framework for bicycle dynamics,
progressing from a simple inverted-pendulum roll model up to the full
linearised Whipple fourth-order model with gyroscopic coupling.

---

## Project Structure

```
bicycle_dynamics/
│
├── models/
│   ├── level1_inverted_pendulum.py   # 2nd-order, steer-angle as input
│   ├── level2_front_fork.py          # 2nd-order + static front-fork model
│   └── level3_whipple.py             # 4th-order linear Whipple model (Eq. 24)
│
├── analysis/
│   ├── stability.py                  # Root-locus vs velocity, pole plots
│   ├── step_response.py              # Closed-loop & open-loop step sims
│   └── self_stabilization.py        # Critical-velocity sweep, riderless push
│
├── utils/
│   ├── params.py                     # Bicycle + rider parameter sets
│   └── plotting.py                   # Shared matplotlib helpers
│
└── main.py                           # Run all demo scenarios
```

## Models

| Level | Equations | States | Control input |
|-------|-----------|--------|---------------|
| 1 | Eq. (1) — inverted pendulum | φ, dφ/dt | steer angle δ |
| 2 | Eq. (1)+(11) — with front fork | φ, dφ/dt | handlebar torque T |
| 3 | Eq. (24) — Whipple 4th order  | φ, δ, dφ/dt, dδ/dt | handlebar torque T |

## Quick Start

```bash
python main.py
```

Each scenario saves a figure to `figures/`.

## Reference
K. J. Åström, R. E. Klein, A. Lennartsson, "Bicycle Dynamics and Control,"
*IEEE Control Systems Magazine*, vol. 25, no. 4, pp. 26–47, Aug. 2005.
