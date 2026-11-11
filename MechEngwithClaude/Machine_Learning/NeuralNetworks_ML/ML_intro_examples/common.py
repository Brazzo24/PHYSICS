"""
common.py - shared helpers for the example scripts.

Contains
  * a small plot style (consistent colours across all scripts)
  * a quarter-car simulator (the "vehicle dynamics" test bench used in
    02_system_identification.py and 03_simulation_tuning.py)

Everything is plain NumPy / pure Python so you can read and modify it.
"""
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
COL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#4a3aa7",
    "grey": "#8a8985",
    "ink": "#2b2b2a",
}
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def setup_plot_style():
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.7,
        "axes.edgecolor": "#8a8985",
        "axes.labelcolor": COL["ink"],
        "xtick.color": COL["ink"],
        "ytick.color": COL["ink"],
        "text.color": COL["ink"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "font.size": 10,
        "lines.linewidth": 1.8,
    })


def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [plot saved] plots/{name}")


# ---------------------------------------------------------------------------
# Quarter-car model (nonlinear) - the "simulation" we identify / tune
# ---------------------------------------------------------------------------
#   ms * zs'' = -Fs(d) - Fd(vd)                    (sprung mass, quarter body)
#   mu * zu'' =  Fs(d) + Fd(vd) - kt * (zu - zr)   (unsprung mass, wheel)
#   d = zs - zu ,  vd = zs' - zu'
#   Fs(d) = ks*d + k3*d^3                          (hardening spring)
#   Fd(vd) = c_reb*vd  if vd > 0 (rebound)  else  c_bump*vd   (asymmetric damper)
#
# Nonlinearities (cubic spring + asymmetric damper) are what make a purely
# linear model insufficient - and are what a neural net can pick up from data.

DT = 0.005  # integration step [s]  (200 Hz)

DEFAULT_PARAMS = dict(
    ms=300.0,     # sprung mass (quarter) [kg]
    mu=40.0,      # unsprung mass [kg]
    kt=200000.0,  # tyre stiffness [N/m]
    ks=22000.0,   # spring stiffness [N/m]
    k3=3.0e7,     # cubic spring term [N/m^3]
    c_reb=2800.0,  # rebound damping [N s/m]
    c_bump=1200.0,  # bump damping [N s/m]
)


def make_road(n_steps, dt=DT, std=0.015, f_cut=8.0, seed=0):
    """Random road profile zr(t) [m]: low-pass filtered white noise."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n_steps)
    a = math.exp(-2 * math.pi * f_cut * dt)  # 1st order low-pass, applied twice
    y = np.zeros(n_steps)
    for _ in range(2):
        acc = 0.0
        for i in range(n_steps):
            acc = a * acc + (1 - a) * w[i]
            y[i] = acc
        w = y.copy()
    y -= y.mean()
    return y * (std / y.std())


def simulate_quarter_car(zr, params=None, dt=DT):
    """Integrate the quarter car with RK4 for road input zr (array, step dt).

    Returns dict with body acceleration 'acc_s' [m/s^2], suspension travel
    'd' [m] and wheel-hop related 'zu' [m] - all arrays of len(zr).
    Pure-Python floats are used on purpose: fast enough and easy to read.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    ms, mu, kt = p["ms"], p["mu"], p["kt"]
    ks, k3, cr, cb = p["ks"], p["k3"], p["c_reb"], p["c_bump"]

    def deriv(zs, vs, zu, vu, r):
        d = zs - zu
        vd = vs - vu
        fs = ks * d + k3 * d ** 3
        fd = cr * vd if vd > 0 else cb * vd
        a_s = -(fs + fd) / ms
        a_u = (fs + fd - kt * (zu - r)) / mu
        return vs, a_s, vu, a_u

    n = len(zr)
    zr_l = zr.tolist()
    acc_s = [0.0] * n
    dd = [0.0] * n
    zu_out = [0.0] * n
    zs = vs = zu = vu = 0.0
    for i in range(n):
        r0 = zr_l[i]
        r1 = zr_l[i + 1] if i + 1 < n else r0
        rm = 0.5 * (r0 + r1)
        k1 = deriv(zs, vs, zu, vu, r0)
        k2 = deriv(zs + 0.5 * dt * k1[0], vs + 0.5 * dt * k1[1],
                   zu + 0.5 * dt * k1[2], vu + 0.5 * dt * k1[3], rm)
        k3_ = deriv(zs + 0.5 * dt * k2[0], vs + 0.5 * dt * k2[1],
                    zu + 0.5 * dt * k2[2], vu + 0.5 * dt * k2[3], rm)
        k4 = deriv(zs + dt * k3_[0], vs + dt * k3_[1],
                   zu + dt * k3_[2], vu + dt * k3_[3], r1)
        acc_s[i] = k1[1]
        dd[i] = zs - zu
        zu_out[i] = zu
        zs += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3_[0] + k4[0])
        vs += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3_[1] + k4[1])
        zu += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3_[2] + k4[2])
        vu += dt / 6 * (k1[3] + 2 * k2[3] + 2 * k3_[3] + k4[3])
    return dict(acc_s=np.array(acc_s), d=np.array(dd), zu=np.array(zu_out))
