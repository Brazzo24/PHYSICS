"""
03_simulation_tuning.py - Tuning simulation-model parameters when each run is expensive

QUESTION THIS SCRIPT ANSWERS
    "My simulation model has uncertain parameters. Every simulation run is expensive
     (in real life: an MBS run of minutes to hours). How do ML techniques help me find
     the parameters that make the simulation match the measurement - and when do they
     NOT help?"

SETUP
    Measured: suspension travel of the quarter car on a known road (20 s, noise 0.2 mm).
    Unknown: five parameters (ks, k3, c_reb, c_bump, ms), each only known within a wide
    range (factor ~2-4, log scale). Cost = 1 - fit, i.e. relative RMS error between simulated and
    measured travel (0 = perfect; ~0.01 is the noise floor).

    The simulator is fast here, but we pretend each run is expensive and therefore give
    every method the SAME BUDGET of simulator runs and compare progress per run.

METHODS
    Random search           - the baseline any smarter method must beat
    Nelder-Mead             - classic derivative-free local optimiser (no ML)
    Least squares (LSQ)     - classic: uses the residual VECTOR and finite differences
    Bayesian optimisation   - ML: Gaussian-process surrogate of the cost + Expected Improvement
                              chooses the next run (sequential, learns from every run)
    NN surrogate            - ML: sample N runs at once (Latin hypercube), train a neural
                              net cost(parameters), minimise the NN for free, verify with 1 run

WHAT TO TAKE AWAY (reference run, median of 5 random starts, 60 runs each)
    * Classic least squares reached the noise floor (cost 0.011, parameters within 0.3 %) -
      far ahead of everything else. It is allowed to use the whole residual VECTOR and
      the cost valley here is smooth and single. ML is NOT automatically better.
    * The methods that only see ONE scalar per run (random, Nelder-Mead, BO, NN surrogate)
      are the fair comparison for ML: the GP-based Bayesian optimisation is clearly best of
      them (0.054 vs 0.19 random / 0.30 Nelder-Mead / 0.11 NN surrogate), because it learns
      from every run. Its parameters are still 4-27 % off: 60 runs is little for 5 unknowns.
    * The NN surrogate is not a magic shortcut either: with 400 runs it fits the cost well
      globally (printed R^2), but a rough valley bottom is exactly where it is least accurate,
      and it still does not beat 60 runs of least squares.
    * Use ML for tuning when runs are expensive AND classic methods are handicapped:
      only scalar KPIs (lap time, peak load), noisy or non-smooth cost, no gradients, or
      many local minima. NOT tested here: local minima - LSQ would suffer there.

Run:  python 03_simulation_tuning.py         (~2-3 min on CPU)
"""
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize, least_squares
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt

from common import (COL, setup_plot_style, save_fig, make_road, simulate_quarter_car,
                    DEFAULT_PARAMS, DT)

warnings.filterwarnings("ignore")
setup_plot_style()
torch.set_num_threads(2)

# ---------------------------------------------------------------------------
# The "measurement" and the parameter space
# ---------------------------------------------------------------------------
PN = ["ks", "k3", "c_reb", "c_bump", "ms"]
LO = np.array([12000.0, 5e6, 1200.0, 500.0, 250.0])
HI = np.array([40000.0, 6e7, 5000.0, 2500.0, 350.0])
LLO, LHI = np.log(LO), np.log(HI)
TRUTH = np.array([DEFAULT_PARAMS[k] for k in PN])
NDIM = len(PN)
DEC = 2
SENSOR_NOISE = 0.0002
BUDGET = 60
N_SEEDS = 5

N_STEPS = int(20 / DT)
ROAD = make_road(N_STEPS, std=0.015, seed=11)
y_meas = simulate_quarter_car(ROAD)["d"][::DEC] + \
    np.random.default_rng(5).normal(0, SENSOR_NOISE, N_STEPS // DEC)
Y_NORM = np.linalg.norm(y_meas - y_meas.mean())


def to_phys(x01):
    """Normalised parameters in [0,1]^5 -> physical values (log scale)."""
    return np.exp(LLO + np.clip(x01, 0, 1) * (LHI - LLO))


def residual(x01):
    p = dict(zip(PN, to_phys(x01)))
    sim = simulate_quarter_car(ROAD, p)["d"][::DEC]
    return (sim - y_meas) / Y_NORM


class Budget(Exception):
    pass


class Problem:
    """Counts simulator runs and records best-so-far cost after each run."""

    def __init__(self, budget):
        self.budget, self.n, self.best, self.trace, self.best_x = budget, 0, np.inf, [], None

    def __call__(self, x01, vector=False):
        if self.n >= self.budget:
            raise Budget
        x01 = np.clip(np.asarray(x01, float), 0, 1)
        r = residual(x01)
        cost = float(np.linalg.norm(r))
        self.n += 1
        if cost < self.best:
            self.best, self.best_x = cost, x01.copy()
        self.trace.append(self.best)
        return r if vector else cost

    def finish(self):
        while len(self.trace) < self.budget:
            self.trace.append(self.best)
        return np.array(self.trace), self.best_x


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def run_random(seed):
    pr, rng = Problem(BUDGET), np.random.default_rng(seed)
    try:
        while True:
            pr(rng.random(NDIM))
    except Budget:
        pass
    return pr.finish()


def run_nelder_mead(seed):
    pr, rng = Problem(BUDGET), np.random.default_rng(seed)
    x0 = rng.random(NDIM)                      # a rough first guess
    try:
        minimize(pr, x0, method="Nelder-Mead", options=dict(maxfev=10 ** 6, xatol=1e-4, fatol=1e-8))
    except Budget:
        pass
    return pr.finish()


def run_lsq(seed):
    pr, rng = Problem(BUDGET), np.random.default_rng(seed)
    x0 = rng.random(NDIM)
    try:
        least_squares(lambda x: pr(x, vector=True), x0, bounds=(0, 1), diff_step=1e-3)
    except Budget:
        pass
    return pr.finish()


def run_bo(seed, n_init=15):
    """Bayesian optimisation: GP on log(cost), Expected Improvement, random-candidate search."""
    pr, rng = Problem(BUDGET), np.random.default_rng(seed)
    X = qmc.LatinHypercube(d=NDIM, seed=seed).random(n_init)
    Y = np.array([pr(x) for x in X])
    kernel = C(1.0) * Matern(length_scale=np.ones(NDIM) * 0.5, length_scale_bounds=(0.05, 10.0), nu=2.5) \
        + WhiteKernel(1e-3, (1e-6, 1e-1))
    try:
        while True:
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
            gp.fit(X, np.log(Y))
            best_x = X[np.argmin(Y)]
            cand = np.vstack([rng.random((3000, NDIM)),
                              np.clip(best_x + rng.normal(0, 0.05, (1000, NDIM)), 0, 1),
                              np.clip(best_x + rng.normal(0, 0.15, (1000, NDIM)), 0, 1)])
            mu, sd = gp.predict(cand, return_std=True)
            f_best = np.log(Y.min())
            z = (f_best - mu) / np.maximum(sd, 1e-9)
            ei = (f_best - mu) * norm.cdf(z) + sd * norm.pdf(z)
            x_new = cand[np.argmax(ei)]
            X = np.vstack([X, x_new])
            Y = np.append(Y, pr(x_new))
    except Budget:
        pass
    return pr.finish()


def train_surrogate(X, Y, seed, epochs=1500):
    """MLP: 5 normalised parameters -> log(cost). Small ensemble average is more robust."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yl = np.log(Y)
    m, s = yl.mean(), yl.std()
    Yt = torch.tensor(((yl - m) / s).reshape(-1, 1), dtype=torch.float32)
    nets = []
    for k in range(3):
        net = nn.Sequential(nn.Linear(NDIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            ((net(Xt) - Yt) ** 2).mean().backward()
            opt.step()
        nets.append(net)

    def predict(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return np.mean([n(t).numpy().ravel() for n in nets], axis=0) * s + m
    return predict


def run_surrogate(seed, n_runs=BUDGET - 1):
    """Spend n_runs on a space-filling design, train NN, optimise NN for free, verify with 1 run."""
    pr, rng = Problem(n_runs + 1), np.random.default_rng(seed)
    X = qmc.LatinHypercube(d=NDIM, seed=seed).random(n_runs)
    Y = np.array([pr(x) for x in X])
    sur = train_surrogate(X, Y, seed)
    cand = rng.random((200000, NDIM))                # 200k free "simulations" on the NN
    pred = sur(cand)
    top = cand[np.argsort(pred)[:20]]
    # local refinement of the best candidates on the surrogate
    refined = []
    for x0 in top:
        r = minimize(lambda x: float(sur(np.clip(x, 0, 1)[None, :])[0]), x0, method="Nelder-Mead",
                     options=dict(maxfev=400, xatol=1e-4, fatol=1e-6))
        refined.append((r.fun, np.clip(r.x, 0, 1)))
    x_star = min(refined, key=lambda t: t[0])[1]
    try:
        pr(x_star)                                   # ONE verification run on the real simulator
    except Budget:
        pass
    return pr.finish()          # trace has n_runs + 1 entries


METHODS = {
    "Random search": run_random,
    "Nelder-Mead": run_nelder_mead,
    "Least squares (LSQ)*": run_lsq,
    "Bayesian opt. (GP)": run_bo,
    f"NN surrogate ({BUDGET - 1}+1 runs)": run_surrogate,
}
MC = dict(zip(METHODS, [COL["grey"], COL["aqua"], COL["blue"], COL["violet"], COL["orange"]]))

# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------
print(f"Cost floor from sensor noise ~ {SENSOR_NOISE * np.sqrt(len(y_meas)) / Y_NORM:.3f}\n"
      f"Budget = {BUDGET} simulator runs per method, {N_SEEDS} repetitions with different random starts.\n")
traces, finals = {}, {}
for name, fn in METHODS.items():
    tr, xs = [], []
    for seed in range(N_SEEDS):
        t, x = fn(seed)
        tr.append(t)
        xs.append(to_phys(x))
    traces[name], finals[name] = np.array(tr), np.array(xs)
    print(f"  {name} done")

print("\nBest cost found (median over repetitions; [min ... max]) - lower is better")
print(f"{'method':32s}{'after 20 runs':>22s}{'after 60 runs':>22s}")
for name in METHODS:
    t = traces[name]
    def cell(i):
        return f"{np.median(t[:, i]):.3f} [{t[:, i].min():.3f}..{t[:, i].max():.3f}]"
    print(f"{name:32s}{cell(19):>22s}{cell(BUDGET - 1):>22s}")

print("\nParameter error of the final result, median |fitted/true - 1| in % (identifiability!)")
print(f"{'method':32s}" + "".join(f"{k:>9s}" for k in PN))
for name in METHODS:
    err = 100 * np.median(np.abs(finals[name] / TRUTH - 1), axis=0)
    print(f"{name:32s}" + "".join(f"{e:9.1f}" for e in err))

print("  * LSQ additionally uses the full residual vector, not just one scalar per run.")

# Extra data point: surrogate with a much bigger design (NOT the same budget!)
tr_big, xs_big = [], []
for seed in range(N_SEEDS):
    pr_big = run_surrogate(seed, n_runs=400)
    tr_big.append(pr_big[0][-1])
    xs_big.append(to_phys(pr_big[1]))
print(f"\nFor comparison, NN surrogate with 400+1 runs (7x the budget!): median cost "
      f"{np.median(tr_big):.3f} [{np.min(tr_big):.3f}..{np.max(tr_big):.3f}]")

# How good is that surrogate as a model of the cost? Check on 200 fresh random parameter sets.
X_d = qmc.LatinHypercube(d=NDIM, seed=0).random(400)
Y_d = np.array([np.linalg.norm(residual(x)) for x in X_d])
X_t = np.random.default_rng(123).random((200, NDIM))
Y_t = np.array([np.linalg.norm(residual(x)) for x in X_t])
pred = train_surrogate(X_d, Y_d, 0)(X_t.astype(np.float32))
lt = np.log(Y_t)
good = Y_t < np.quantile(Y_t, 0.2)
r2 = lambda m: 1 - np.sum((pred[m] - lt[m]) ** 2) / np.sum((lt[m] - lt[m].mean()) ** 2)
print(f"Surrogate (400 runs) on unseen parameter sets: R^2 of log-cost = {r2(np.ones(200, bool)):.2f} "
      f"overall, {r2(good):.2f} within the best 20 % region")

# ---------------------------------------------------------------------------
# Plot: progress per simulator run
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.6))
runs = np.arange(1, BUDGET + 1)
for name in METHODS:
    t = traces[name]
    ax.semilogy(runs, np.median(t, axis=0), color=MC[name], lw=2, label=name)
    ax.fill_between(runs, t.min(0), t.max(0), color=MC[name], alpha=0.12, lw=0)
ax.axhline(SENSOR_NOISE * np.sqrt(len(y_meas)) / Y_NORM, color=COL["ink"], ls=":", lw=1)
ax.text(BUDGET, SENSOR_NOISE * np.sqrt(len(y_meas)) / Y_NORM * 1.1, "noise floor", ha="right", fontsize=8)
ax.set_xlabel("simulator runs used")
ax.set_ylabel("best cost so far (1 - fit)")
ax.set_title(f"Progress per expensive run ({N_SEEDS} random starts; line = median, band = min/max)")
ax.legend(fontsize=8, loc="upper right")
save_fig(fig, "03_tuning_progress.png")
