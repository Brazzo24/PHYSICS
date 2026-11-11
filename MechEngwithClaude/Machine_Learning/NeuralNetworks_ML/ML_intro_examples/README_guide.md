# Neural networks and machine learning for vehicle dynamics and simulation – a first guide

This folder contains four runnable Python scripts and this guide. The scripts are small on purpose: each one answers exactly one question about *when* a machine-learning method is worth using. Everything runs on a laptop CPU without a GPU.

```
common.py                      shared plot style + nonlinear quarter-car simulator
00_nn_basics.py                what training a neural net actually is (1-D curve)
01_tire_curve_fitting.py       Magic Formula vs. polynomial vs. Gaussian process vs. NN
02_system_identification.py    learning vehicle dynamics from time series
03_simulation_tuning.py        tuning uncertain simulation parameters with a limited run budget
plots/                         figures written by the scripts
```

Setup: `pip install numpy scipy scikit-learn torch matplotlib` (Python 3.10+). Run the scripts in order from this folder, e.g. `python 00_nn_basics.py`. Each script prints a results table and writes its figures to `plots/`. The scripts use fixed random seeds; the numbers quoted below come from those runs, and yours may differ slightly (especially the neural-net numbers).

## 1. The core idea, in the language of an engineer

Supervised machine learning is regression: you have inputs *x* and measured outputs *y*, you choose a family of functions *f(x; θ)* with adjustable parameters θ, define a loss (almost always the mean squared error) and search for the θ that minimises it on your data. If you have ever fitted a Magic Formula to tyre data or identified a transfer function, you have done supervised learning.

What differs between methods is only the family of functions. The Magic Formula has four parameters and a shape chosen by physics. A polynomial has as many parameters as its degree plus one and no physics. A Gaussian process has, in effect, one parameter per data point and a smoothness assumption. A neural network is a nested chain of "weighted sum, then a smooth nonlinearity" and has hundreds to millions of parameters; it is fitted by gradient descent (the optimiser Adam is the standard choice), with the gradients supplied automatically by PyTorch. That is the whole story of `00_nn_basics.py`.

The vocabulary you need is small. *Training data* is what the optimiser sees. *Validation data* is held back to decide when to stop and which model to keep. *Test data* is touched once at the end. *Overfitting* means the training error keeps falling while the error on unseen data does not, and it is the central danger, because a flexible model can memorise noise. Standard cures are more data, a smaller model, weight decay, and early stopping. Always standardise inputs and outputs to roughly zero mean and unit variance before feeding a neural network; forgetting this is the most common reason for "it does not train".

## 2. When is a method efficient? Five questions to ask first

**Do I know the physics structure?** If yes, and the number of uncertain parameters is small, fit those parameters. It needs the least data, it extrapolates sensibly, and the parameters have meaning. This is the message of `02` (grey box, 99 % fit) and of `01` world 1 (four Magic Formula parameters matched or beat a network with about 1 000 parameters for N ≥ 20).

**Is my known model incomplete, and do I have a lot of data?** This is where flexible models pay off. In `01` world 2 the Magic Formula error stalls at about 2 % of Fz no matter how many points you add, while the Gaussian process, the network and the polynomial keep improving down to 0.3–0.5 %. The practical version of this is a *hybrid*: keep the physics model and let a small network learn only its residual.

**How much data, and does it cover the operating range?** Every black-box model is an interpolator. In `02`, on a road twice as rough as the training road, the linear model fell to 19 % fit and the network to 28 % (from 55 % and 70 % on the normal road), while the physics model stayed at 99–100 %. In `01` all models failed to extrapolate reliably beyond the measured slip range: the polynomial exploded, the Gaussian process fell back to predicting zero force, and even the correct Magic Formula was 6–19 % off because data that stop before the peak do not determine its shape parameters. No method can tell you what your data never contained.

**How many inputs?** Classical curve fitting and Gaussian processes work well for one to about ten inputs and tens to a few thousand data points. Neural networks become the sensible choice with many inputs (dozens), large data sets (thousands to millions of points), or when you need a fast evaluation, for example a surrogate for a slow multibody or engine model in real-time or hardware-in-the-loop use.

**Do I need extrapolation, guarantees or interpretability?** If so, prefer physics-based or hybrid models, and treat a network as an interpolator inside a validated envelope. If you need an honest "I don't know", use a Gaussian process (see `plots/01_gp_uncertainty.png`) or an ensemble of networks.

A compact summary:

| Situation | Reasonable first choice |
|---|---|
| Right model structure known, few parameters | Nonlinear least squares on the physics model |
| Structure known but incomplete, plenty of data | Physics model + NN or GP for the residual |
| 1–5 inputs, tens to hundreds of points, smooth response | Gaussian process |
| Many inputs, thousands or more of points, need fast evaluation | Neural network (surrogate) |
| Expensive simulator, scalar result per run, tens of runs | Bayesian optimisation with a GP |
| Smooth cost with a residual vector, cheap to differentiate | Classic least squares (hard to beat, see `03`) |

## 3. What the three example problems showed

**Tyre curve (`01`).** Noisy lateral force over slip angle, measured only up to 6°, before the peak at about 8°. With the right structure the four-parameter Magic Formula was best or tied at every N ≥ 20 inside the measured range (0.11 % of Fz at N = 1000 versus 0.19–0.31 % for the others). At N = 10–20 the generic models were erratic; the degree-7 polynomial was off by more than 8 %. When the data contained an effect the formula cannot represent, the formula's error plateaued at about 2 %. The lesson is that the decision between physics and ML is a decision about *structure error versus data amount*, not about which method is fashionable.

**System identification (`02`).** Suspension travel of a nonlinear quarter car, 60 s at 100 Hz. One-step prediction looked good for both black boxes (linear 94 %, NN 97 %), but free-run simulation, which is what you need from a simulation model, gave 55 % for the linear ARX model and 70 % for the network. The network is better because the system is nonlinear, yet still far from usable. A grey-box fit of only four physical parameters reached 99 % and recovered the true values to within 0.1 %. An important caveat: the "measurement" was generated by the very equations the grey box uses, so this is the best case. On real data, structural errors remain, and a network trained on the residual would be the natural next step.

**Simulation tuning (`03`).** Five uncertain parameters, 60 simulator runs per method, five random starts. Classic least squares reached the noise floor (parameters within 0.3 %), because it uses the full residual vector and the cost valley is smooth. Among methods that see only one number per run, Gaussian-process Bayesian optimisation was clearly best (median cost 0.054, against 0.19 for random search, 0.30 for Nelder–Mead and 0.11 for a neural-network surrogate), but 60 runs still left some parameters 4–27 % off. A network surrogate trained on 400 runs modelled the cost accurately overall (R² 0.99, 0.90 near the optimum) yet was still not competitive with 60 runs of least squares. The takeaway: ML-based tuning is attractive when you only get a scalar per run (lap time, peak load, a KPI), when gradients or residuals are unavailable, or when the cost is rough with local minima. It is not attractive merely because a problem "feels complex". Local minima were not tested here, and they are exactly the case where least squares can fail.

## 4. Pitfalls worth knowing early

Judge a time-series model by free-run simulation, not by one-step prediction; the gap between them (94 % versus 55 % above) is the single most instructive number in this folder. Split time-series data by experiment or by contiguous blocks, never by random samples, because neighbouring samples are almost identical and random splitting makes every model look excellent. Report errors against noise-free truth or held-out data, never on the training set. Beware of tuning a network's hyper-parameters on the same data you report. Start with a simple baseline (linear model, physics fit) and require the network to beat it clearly; in `03` the simple baseline won.

## 5. Where this could go in your own work

Natural candidates for your domain, roughly from easiest to hardest: a Gaussian-process or small-MLP surrogate of an engine or torque map from sparse dyno points; a residual model on top of a Pacejka tyre fitted to real measurements; a virtual sensor that predicts a hard-to-measure channel from cheap ones; a fast surrogate of a slow multibody model for parameter studies; and Bayesian optimisation for tuning an expensive model against a few KPIs. Good next exercises in this folder would be to (1) add a network for the residual in `02` and compare its free-run fit against the pure grey box when the grey-box structure is deliberately made slightly wrong, (2) train the NARX network on multi-step rollouts instead of one-step targets, (3) rerun `03` with only scalar KPIs so that least squares loses its advantage, and (4) replace the single network in `01` by an ensemble of five to obtain an uncertainty estimate.
