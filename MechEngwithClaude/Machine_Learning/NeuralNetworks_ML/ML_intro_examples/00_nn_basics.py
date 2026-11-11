"""
00_nn_basics.py - What is a neural network actually doing?

Goal: understand the four ingredients of ANY supervised learning problem
using the smallest possible example, a 1-D curve.

   1. DATA         pairs (x_i, y_i)  - here: noisy samples of an unknown curve
   2. MODEL        y_hat = f(x; theta) - here: a small multilayer perceptron (MLP)
   3. LOSS         a number that says how wrong the model is - here: mean squared error
   4. OPTIMISER    changes theta to reduce the loss - here: Adam (a gradient-descent variant)

If you know least-squares curve fitting you already know 95 % of this:
a neural net is just a very flexible nonlinear function with many parameters,
fitted by gradient descent on a least-squares loss.

What the script shows
  * A: the training loop (what "training" means)
  * B: overfitting - a model that fits the training points perfectly but is wrong
       between them. Detected via a VALIDATION set the model never trains on.
  * C: two standard cures: weight decay (regularisation) and early stopping.

Run:  python 00_nn_basics.py       (takes ~20 s on a CPU, no GPU needed)
"""
import numpy as np
import torch
import torch.nn as nn

from common import COL, setup_plot_style, save_fig
import matplotlib.pyplot as plt

torch.manual_seed(0)
rng = np.random.default_rng(0)
setup_plot_style()


# ---------------------------------------------------------------------------
# 1. DATA - an "unknown" physical relation plus measurement noise
# ---------------------------------------------------------------------------
def truth(x):
    return np.sin(2.0 * x) + 0.3 * x


x_train = rng.uniform(-3, 3, 25)
y_train = truth(x_train) + rng.normal(0, 0.15, x_train.size)
x_val = rng.uniform(-3, 3, 200)                      # never used for fitting
y_val = truth(x_val) + rng.normal(0, 0.15, x_val.size)
x_grid = np.linspace(-3, 3, 400)


def to_t(a):
    """NumPy vector -> torch column tensor (N,1), float32."""
    return torch.tensor(a, dtype=torch.float32).reshape(-1, 1)


Xtr, Ytr, Xva, Yva = map(to_t, (x_train, y_train, x_val, y_val))


# ---------------------------------------------------------------------------
# 2. MODEL - MLP: y = W3 * tanh(W2 * tanh(W1 x + b1) + b2) + b3
#    "hidden" = number of neurons per layer. Parameters theta = all W and b.
# ---------------------------------------------------------------------------
def make_model(hidden):
    return nn.Sequential(
        nn.Linear(1, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, 1),
    )


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 3.+4. LOSS and OPTIMISER - the training loop
# ---------------------------------------------------------------------------
def train(hidden, epochs, weight_decay=0.0, lr=1e-2, keep_best=False):
    model = make_model(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    hist_tr, hist_va = [], []
    best = (np.inf, None, 0)
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), Ytr)   # forward pass + loss
        loss.backward()                   # gradients dLoss/dtheta (backpropagation)
        opt.step()                        # theta <- theta - lr * (adapted gradient)
        with torch.no_grad():
            lv = loss_fn(model(Xva), Yva).item()
        hist_tr.append(loss.item())
        hist_va.append(lv)
        if lv < best[0]:
            best = (lv, {k: v.clone() for k, v in model.state_dict().items()}, ep)
    if keep_best:                          # early stopping: restore best-on-validation
        model.load_state_dict(best[1])
    return model, np.array(hist_tr), np.array(hist_va), best[2]


def predict(model, x):
    with torch.no_grad():
        return model(to_t(x)).numpy().ravel()


def rmse(model, x, y):
    return float(np.sqrt(np.mean((predict(model, x) - y) ** 2)))


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
print("Noise level (irreducible RMSE floor) = 0.15\n")

configs = [
    ("A) small net (8 neurons), 1500 epochs", dict(hidden=8, epochs=1500)),
    ("B) big net (128 neurons), 3000 epochs, no regularisation", dict(hidden=128, epochs=3000, lr=5e-3)),
    ("C) big net + weight decay", dict(hidden=128, epochs=3000, lr=5e-3, weight_decay=1e-3)),
    ("D) big net + early stopping", dict(hidden=128, epochs=3000, lr=5e-3, keep_best=True)),
]
results = []
for name, cfg in configs:
    model, htr, hva, best_ep = train(**cfg)
    r_tr, r_va = rmse(model, x_train, y_train), rmse(model, x_val, y_val)
    results.append((name, model, htr, hva, best_ep))
    print(f"{name:62s} params={n_params(model):6d}  train RMSE={r_tr:.3f}  "
          f"validation RMSE={r_va:.3f}")

print("\nRead this as: the training error alone tells you NOTHING about quality.\n"
      "Only the error on data the model has not seen (validation) does.")

# ---------------------------------------------------------------------------
# Plot 1: the fits;  Plot 2: training vs validation loss curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(15, 3.3), sharey=True)
for ax, (name, model, *_rest) in zip(axes, results):
    ax.plot(x_grid, truth(x_grid), color=COL["grey"], lw=1.4, label="true curve")
    ax.plot(x_grid, predict(model, x_grid), color=COL["orange"], label="network")
    ax.scatter(x_train, y_train, s=16, color=COL["blue"], zorder=3, label="training data")
    ax.set_title(name.split(",")[0].replace(") ", ")\n", 1), fontsize=9)
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
save_fig(fig, "00_fits.png")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
name, model, htr, hva, best_ep = results[1]
ax.semilogy(htr, color=COL["blue"], label="training loss")
ax.semilogy(hva, color=COL["orange"], label="validation loss")
ax.axvline(results[3][4], color=COL["grey"], ls="--", lw=1)
ax.text(results[3][4] + 40, ax.get_ylim()[1] * 0.4, "early-stopping\noptimum (case D)",
        fontsize=8, color=COL["ink"])
ax.set_xlabel("epoch")
ax.set_ylabel("MSE")
ax.set_title("Big net (case B): training keeps improving, validation does not")
ax.legend()
save_fig(fig, "00_loss_curves.png")
