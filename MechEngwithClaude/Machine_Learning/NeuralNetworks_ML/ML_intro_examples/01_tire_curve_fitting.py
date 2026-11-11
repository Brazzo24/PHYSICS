"""
01_tire_curve_fitting.py - Tyre lateral force: physics-based fit vs. ML models

QUESTION THIS SCRIPT ANSWERS
    "When is a neural network better than the classical parametric fit
     I already know - and when is it worse?"

SETUP
    Measured: lateral force Fy over slip angle alpha (noisy, like a test-rig run).
    Four models are fitted to the SAME data:

      (1) Magic Formula (Pacejka) - 4 parameters, physics-inspired structure
      (2) Polynomial degree 7       - generic "curve fit", 8 parameters
      (3) Gaussian process (GP)     - non-parametric, gives uncertainty
      (4) Neural network (MLP)      - flexible, ~1000 parameters

    Two worlds are tested:
      WORLD 1  the data really follows a Magic Formula ("your model structure is right")
      WORLD 2  the data follows a Magic Formula PLUS an effect it cannot represent
               ("your model structure is incomplete" - the realistic case:
                combined slip, temperature, carcass effects, ...)

    Measurements only cover |alpha| <= 6 deg (BEFORE the force peak at ~8 deg);
    models are also tested out to 15 deg (EXTRAPOLATION) - typical when you need the limit behaviour that you
    could not measure safely.

WHAT TO TAKE AWAY (numbers below are from the reference run; yours will differ slightly)
    * Structure right, inside the measured range: the 4-parameter Magic Formula is best or
      tied at every N >= 20 (N=1000: 0.11 % of Fz vs. 0.19-0.31 % for GP / polynomial / NN).
      With very few points (N=10-20) the generic models are erratic (polynomial: >8 %).
    * Structure incomplete (world 2): the Magic Formula error stalls at ~2 % of Fz however much
      data you add. GP, NN and polynomial keep improving (0.3-0.5 % at N=1000). THIS is the
      situation where ML pays off: you have plenty of data and the physics model is known
      to be incomplete.
    * Extrapolation beyond the data is unreliable for EVERY model here:
        - polynomial explodes (errors of hundreds to thousands of %),
        - GP falls back to its prior mean, i.e. predicts ~0 force (13-50 % error),
        - Magic Formula: right structure, but data that stops before the peak leave
          parameters C and E poorly determined (print the covariance to see it) ->
          6-19 % error even with 1000 points in world 1 (in world 2 it is still the
          best extrapolator, 2-11 %, because its saturating shape is broadly right),
        - NN stays bounded because tanh saturates - an architecture bias that happens
          to resemble a tyre curve, NOT knowledge of the physics. Do not rely on it.
      Lesson: no model can tell you what the data never contained. Cover the operating
      range you care about, or put physics constraints into the model.

Run:  python 01_tire_curve_fitting.py     (~3 min on CPU)
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt

from common import COL, setup_plot_style, save_fig

warnings.filterwarnings("ignore")
setup_plot_style()
torch.set_num_threads(2)

FZ = 1500.0                     # vertical load [N] -> peak force scale
NOISE = 0.03 * FZ               # measurement noise std: 3 % of Fz
A_TRAIN = 6.0                   # measured slip-angle range +/- [deg]  (stops BEFORE the peak at ~8 deg)
A_TEST = 15.0                   # evaluation range +/- [deg]


# ---------------------------------------------------------------------------
# "Reality": the true tyre. Magic Formula + optional structural extra effect
# ---------------------------------------------------------------------------
def magic_formula(a, B, C_, D, E):
    x = B * a
    return D * np.sin(C_ * np.arctan(x - E * (x - np.arctan(x))))


TRUE = dict(B=0.18, C_=1.45, D=1.0 * FZ, E=-0.6)


def reality(a, world):
    fy = magic_formula(a, **TRUE)
    if world == 2:
        # unmodelled effect: a localised stiffening bump near 3 deg plus a mild asymmetry
        fy = fy + 0.07 * FZ * np.exp(-((a - 3.0) / 1.5) ** 2) + 0.01 * FZ * np.tanh(a / 6.0)
    return fy


def make_data(n, world, rng, a_max):
    a = rng.uniform(-a_max, a_max, n)
    return a, reality(a, world) + rng.normal(0, NOISE, n)


# ---------------------------------------------------------------------------
# The four models: each has fit(a, y) -> predict(a) function
# ---------------------------------------------------------------------------
def fit_magic(a, y):
    """Classic approach: nonlinear least squares on the 4 MF parameters."""
    p0 = [0.2, 1.3, y.max(), -0.3]
    try:
        p, _ = curve_fit(magic_formula, a, y, p0=p0, maxfev=20000,
                         bounds=([0.02, 0.8, 0.3 * FZ, -3.0], [1.5, 2.5, 2.5 * FZ, 1.0]))
    except Exception:
        p = p0
    return lambda q: magic_formula(q, *p)


def fit_poly(a, y, deg=7):
    s = A_TRAIN
    c = np.polynomial.polynomial.polyfit(a / s, y, deg)
    return lambda q: np.polynomial.polynomial.polyval(q / s, c)


def fit_gp(a, y):
    k = C(FZ ** 2 * 0.3, (FZ ** 2 * 0.05, FZ ** 2 * 5)) * RBF(length_scale=4.0, length_scale_bounds=(1.5, 40.0)) \
        + WhiteKernel(NOISE ** 2, (1e-2 * NOISE ** 2, 1e2 * NOISE ** 2))
    gp = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2, random_state=0)
    gp.fit(a.reshape(-1, 1), y)
    return lambda q: gp.predict(np.asarray(q).reshape(-1, 1)), gp


def fit_nn(a, y, hidden=32, epochs=2500, seed=0):
    """MLP 1 -> 32 -> 32 -> 1 (tanh). Inputs/outputs standardised: ALWAYS do this."""
    torch.manual_seed(seed)
    a_s, y_s = A_TRAIN, FZ
    X = torch.tensor(a / a_s, dtype=torch.float32).reshape(-1, 1)
    Y = torch.tensor(y / y_s, dtype=torch.float32).reshape(-1, 1)
    net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                        nn.Tanh(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # hold out 15 % for early stopping (keeps the NN honest with small N)
    n = len(a)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(3, int(0.15 * n))
    va, tr = idx[:n_val], idx[n_val:]
    best, best_state = np.inf, None
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((net(X[tr]) - Y[tr]) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if ep % 10 == 0:
            with torch.no_grad():
                lv = ((net(X[va]) - Y[va]) ** 2).mean().item()
            if lv < best:
                best, best_state = lv, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    def predict(q):
        with torch.no_grad():
            t = torch.tensor(np.asarray(q) / a_s, dtype=torch.float32).reshape(-1, 1)
            return net(t).numpy().ravel() * y_s
    return predict


def fit_all(a, y):
    gp_pred, gp = fit_gp(a, y)
    return {
        "Magic Formula (4 par.)": fit_magic(a, y),
        "Polynomial deg 7": fit_poly(a, y),
        "Gaussian process": gp_pred,
        "Neural net (MLP)": fit_nn(a, y),
    }, gp


MODELS = ["Magic Formula (4 par.)", "Polynomial deg 7", "Gaussian process", "Neural net (MLP)"]
MCOL = dict(zip(MODELS, [COL["blue"], COL["aqua"], COL["violet"], COL["orange"]]))

# Noise-free evaluation grids (error vs. the TRUE curve, not vs. noisy data)
a_in = np.linspace(-A_TRAIN, A_TRAIN, 300)
a_out = np.concatenate([np.linspace(-A_TEST, -A_TRAIN, 100), np.linspace(A_TRAIN, A_TEST, 100)])


def rmse_pct(pred, world, grid):
    return 100 * np.sqrt(np.mean((pred(grid) - reality(grid, world)) ** 2)) / FZ


# ---------------------------------------------------------------------------
# Experiment: learning curves. RMSE (% of Fz) vs. number of measurement points
# ---------------------------------------------------------------------------
N_LIST = [10, 20, 40, 100, 300, 1000]
N_REPEAT = 5   # repeated with different random data -> median, shows robustness
rows = {}      # (world, model, region) -> list of median errors per N
print(f"Measurement noise = {100 * NOISE / FZ:.1f} % of Fz  (that is the error floor "
      f"of a single noisy point, not of a good fit)\n")
for world in (1, 2):
    for n in N_LIST:
        errs = {m: {"in": [], "out": []} for m in MODELS}
        for rep in range(N_REPEAT):
            rng = np.random.default_rng(1000 * world + 10 * rep + n)
            a, y = make_data(n, world, rng, A_TRAIN)
            models, _ = fit_all(a, y)
            for m in MODELS:
                errs[m]["in"].append(rmse_pct(models[m], world, a_in))
                errs[m]["out"].append(rmse_pct(models[m], world, a_out))
        for m in MODELS:
            for reg in ("in", "out"):
                rows.setdefault((world, m, reg), []).append(float(np.median(errs[m][reg])))
        print(f"  world {world}, N={n:5d} done")

def fmt(v):
    return f"{v:8.2f}" if v < 1000 else "    >999"


titles = {1: "WORLD 1 - data follows the Magic Formula exactly",
          2: "WORLD 2 - data has an effect the Magic Formula cannot represent"}
for world in (1, 2):
    print("\n" + titles[world])
    print("RMSE vs. true curve [% of Fz], INSIDE measured range (|alpha| <= 6 deg):")
    print(f"{'N points':>28s}" + "".join(f"{n:>8d}" for n in N_LIST))
    for m in MODELS:
        print(f"{m:>28s}" + "".join(fmt(v) for v in rows[(world, m, 'in')]))
    print("RMSE vs. true curve [% of Fz], EXTRAPOLATION (6 < |alpha| <= 15 deg):")
    for m in MODELS:
        print(f"{m:>28s}" + "".join(fmt(v) for v in rows[(world, m, 'out')]))

# ---------------------------------------------------------------------------
# Plot 1: learning curves (2 worlds x inside/outside)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for i, world in enumerate((1, 2)):
    for j, reg in enumerate(("in", "out")):
        ax = axes[i, j]
        for m in MODELS:
            ax.loglog(N_LIST, rows[(world, m, reg)], "o-", color=MCOL[m], ms=4, label=m)
        ax.set_title(("inside" if reg == "in" else "EXTRAPOLATION") +
                     f" - world {world}" + (" (structure right)" if world == 1 else " (structure incomplete)"),
                     fontsize=10)
        if j == 0:
            ax.set_ylabel("RMSE vs. true curve [% of Fz]")
        if i == 1:
            ax.set_xlabel("number of measurement points N")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, "01_learning_curves.png")

# ---------------------------------------------------------------------------
# Plot 2: what the fits look like - N=40 points in each world
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
a_plot = np.linspace(-A_TEST, A_TEST, 400)
for ax, world, n in zip(axes, (1, 2), (40, 300)):
    rng = np.random.default_rng(7 + world)
    a, y = make_data(n, world, rng, A_TRAIN)
    models, gp = fit_all(a, y)
    ax.axvspan(-A_TEST, -A_TRAIN, color="#eeede8", zorder=0)
    ax.axvspan(A_TRAIN, A_TEST, color="#eeede8", zorder=0)
    ax.scatter(a, y, s=8, color=COL["grey"], alpha=0.7, label="measurements", zorder=2)
    ax.plot(a_plot, reality(a_plot, world), color=COL["ink"], lw=1.2, ls="--", label="truth")
    for m in MODELS:
        ax.plot(a_plot, np.clip(models[m](a_plot), -2.5 * FZ, 2.5 * FZ), color=MCOL[m], label=m, lw=1.6)
    ax.set_ylim(-1.6 * FZ, 1.6 * FZ)
    ax.set_title(f"World {world}, N={n}  (grey band = never measured)", fontsize=10)
    ax.set_xlabel("slip angle [deg]")
axes[0].set_ylabel("lateral force Fy [N]")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
save_fig(fig, "01_fits.png")

# ---------------------------------------------------------------------------
# Plot 3: the GP's superpower - it knows what it does not know
# ---------------------------------------------------------------------------
rng = np.random.default_rng(3)
a, y = make_data(25, 1, rng, A_TRAIN)
_, gp = fit_gp(a, y)
mu, sd = gp.predict(a_plot.reshape(-1, 1), return_std=True)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.fill_between(a_plot, mu - 2 * sd, mu + 2 * sd, color=COL["violet"], alpha=0.18, label="GP +/- 2 sigma")
ax.plot(a_plot, mu, color=COL["violet"], label="GP mean")
ax.plot(a_plot, reality(a_plot, 1), color=COL["ink"], ls="--", lw=1.1, label="truth")
ax.scatter(a, y, s=16, color=COL["grey"], zorder=3, label="25 measurements")
ax.set_xlabel("slip angle [deg]")
ax.set_ylabel("Fy [N]")
ax.set_title("Gaussian process: uncertainty grows where there is no data")
ax.legend(fontsize=8, loc="lower right")
save_fig(fig, "01_gp_uncertainty.png")
