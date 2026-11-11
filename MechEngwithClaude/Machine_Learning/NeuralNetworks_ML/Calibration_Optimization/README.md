# Sweeps, sensitivity, calibration, surrogates and optimisation – a notebook course

Six Jupyter notebooks that walk through the complete model-calibration and
design-optimisation workflow on one example: a torsional model of a two-cylinder
engine with clutch damper, gearbox and final drive on a test bench.

| # | Notebook | Content | Run time |
|---|---|---|---|
| 01 | `01_model_sweeps_doe.ipynb` | the model, excitation from crank geometry, bench data, cost functions, 1-D/2-D sweeps, LHS/Sobol sampling | 10 s |
| 02 | `02_sensitivity_analysis.ipynb` | local sensitivities, Fisher information and identifiability, test planning, Morris, Sobol | 30 s |
| 03 | `03_calibration.ipynb` | least squares vs. scalar optimisers, multi-start, covariance, profile likelihood, priors, validation, structural model error | 20 s |
| 04 | `04_surrogate_models.ipynb` | when to use a surrogate; DoE, response surface, Gaussian process, MLP; validation; Bayesian optimisation with a constraint | 1–2 min |
| 05 | `05_design_optimisation.ipynb` | constraints, Pareto front, Monte-Carlo robustness, robust optimisation | 30 s |
| 06 | `06_bridge_to_neural_networks.ipynb` | NN surrogate for whole run-up curves (PyTorch), calibration by autodiff, hybrid physics + NN discrepancy model | 1–2 min |

Run them in order: notebook 03 writes `data/calibration.npz`, which 05 and 06 read.

## Files

```
driveline.py          the model: engine excitation, linear frequency-domain chain (≈1 ms),
                      nonlinear time-domain clutch damper (≈ seconds, the "expensive" model)
make_bench_data.py    generates the synthetic bench data – contains the true parameters (spoiler!)
plotstyle.py          shared matplotlib style
data/bench_*.npz      synthetic bench run-ups (A: model structure correct, B: structure incomplete)
data/nl_doe_*.npz     cached runs of the expensive model for notebook 04 (delete to recompute, ~5–10 min)
data/calibration.npz  written by notebook 03
```

## Setup

```
pip install -r requirements.txt
jupyter lab
```

Python 3.10 or newer. Everything runs on a laptop CPU.

## Using your own model

Each notebook needs only a function `outputs = f(parameters)`. Replace
`dl.simulate` (vector outputs, for calibration) or `dl.nl_kpi` (scalar KPIs, for
surrogates) by a wrapper around your Dymola FMU or Python model; notebook 01,
section 6, lists the practical points (FMPy, caching, parallel runs, failed runs).
