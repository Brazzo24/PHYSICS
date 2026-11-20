# Physics-informed neural networks (PINNs) – introduction

`01_pinn_introduction.ipynb` – one executed notebook, PyTorch, CPU only (≈ 5 min total).

| Section | Example | Key result (from the executed run) |
|---|---|---|
| 1 | Derivatives of a network w.r.t. its input | autograd exact to 1e-15; ReLU nets have zero 2nd derivative → useless for PINNs |
| 2a | Damped oscillator (1.5 Hz, ζ = 0.15), data only in first 0.7 s | extrapolation RMS error: plain NN 0.18, PINN 0.002 |
| 2b | Same, no data, hard initial conditions | PINN 9e-4 RMS in ≈ 40 s vs. `solve_ivp` 2e-9 in ≈ 15 ms; badly scaled hard IC → 0.14 |
| 3 | Identify k, c (m = 300 kg) from 30 noisy samples | PINN −0.5 % / −4.5 % in ≈ 40 s; least squares +0.6 % / −1.3 % in 0.5 s |
| 4 | Unknown asymmetric, degressive damper F_d(v) from displacement only | 2.8 % error inside the excited velocity range; needs dissipativity constraint + warm start |
| 5 | 1-D heat equation, α and initial profile unknown, 3 sensors | α = 0.1006 (true 0.1), field RMS error 0.005 |
| 6 | Assessment table: when a PINN is worth it, practical rules, alternatives | |

Setup: `pip install numpy scipy matplotlib torch` (Python 3.10+). Exercises at the end.
