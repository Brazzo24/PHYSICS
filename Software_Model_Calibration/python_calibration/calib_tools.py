"""
calib_tools.py - generic helpers for calibrating a simulation model against
reference curves (here: digitized ValDyn plots).

Contents
--------
RefCurve / load_reference   : digitized reference point clouds + plot ranges
curve_distance              : symmetric (Chamfer) distance curve <-> point cloud,
                              in normalised plot coordinates (robust for
                              steep / impact-like signals and digitized data)
impact_events               : feature extraction (peak angle + peak value)
Objective                   : weighted sum of curve distances (+ prior term)
local_sensitivity           : one-at-a-time normalised finite differences
morris_screening            : global elementary-effects screening (Morris)
sweep_1d / sweep_2d         : parameter sweeps
calibrate                   : bounded optimisation in log/linear parameter space
jacobian_identifiability    : Jacobian based parameter correlation / conditioning
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from scipy.signal import find_peaks


# ---------------------------------------------------------------- reference --
@dataclass
class RefCurve:
    name: str
    pts: np.ndarray          # (n,2) digitized points (x=angle deg, y=value)
    xr: tuple
    yr: tuple


def load_reference(path):
    R = np.load(path)
    out = {}
    for k in R.files:
        if k.endswith('_xr') or k.endswith('_yr'):
            continue
        out[k] = RefCurve(k, R[k], tuple(R[k+'_xr']), tuple(R[k+'_yr']))
    return out


# ---------------------------------------------------------------- metrics ----
def curve_distance(ref: RefCurve, x, y, xwin=None, mode='sym', hide_y0=0.0):
    """RMS Chamfer distance in normalised plot units (1.0 = full plot size).

    ref->sim : every digitized point should lie on the simulated curve
    sim->ref : every simulated point (inside the plot) should be near a
               digitized point (penalises spurious peaks)
    """
    xr, yr = ref.xr, ref.yr
    sx, sy = (xr[1]-xr[0]), (yr[1]-yr[0])
    m = (x >= xr[0]) & (x <= xr[1])
    if xwin is not None:
        m &= (x >= xwin[0]) & (x <= xwin[1])
    S = np.c_[(x[m]-xr[0])/sx, np.clip((y[m]-yr[0])/sy, -0.05, 1.05)]
    P = ref.pts
    if xwin is not None:
        P = P[(P[:, 0] >= xwin[0]) & (P[:, 0] <= xwin[1])]
    Pn = np.c_[(P[:, 0]-xr[0])/sx, (P[:, 1]-yr[0])/sy]
    d1, _ = cKDTree(S).query(Pn)
    if mode == 'ref':
        return float(np.sqrt(np.mean(d1**2)))
    if hide_y0 > 0:   # simulated points hidden behind the black y=0 axis line
        y0n = (0-yr[0])/sy
        S = S[np.abs(S[:, 1]-y0n) > hide_y0]
        if len(S) == 0:
            return float(np.sqrt(np.mean(d1**2)))
    d2, _ = cKDTree(Pn).query(S)
    return float(np.sqrt(0.5*np.mean(d1**2) + 0.5*np.mean(d2**2)))


def impact_events(x, y, threshold, min_sep=3.0):
    """Peak angle / height of each impact (contiguous excursion above threshold).
    Works for simulated curves and for digitized point clouds (x sorted)."""
    o = np.argsort(x); x = np.asarray(x)[o]; y = np.asarray(y)[o]
    above = y > threshold
    xs, ys = x[above], y[above]
    if len(xs) == 0:
        return np.zeros((0, 2))
    split = np.nonzero(np.diff(xs) > min_sep)[0]+1
    ev = []
    for seg_x, seg_y in zip(np.split(xs, split), np.split(ys, split)):
        k = np.argmax(seg_y)
        ev.append((seg_x[k], seg_y[k]))
    return np.array(ev)


def event_residuals(ev_sim, ev_ref, n, x_scale=10.0, y_scale=1000.0, missing=3.0):
    """Residual vector comparing the first n events (angle and height)."""
    r = []
    for i in range(n):
        if i < len(ev_sim) and i < len(ev_ref):
            r += [(ev_sim[i, 0]-ev_ref[i, 0])/x_scale, (ev_sim[i, 1]-ev_ref[i, 1])/y_scale]
        elif i < len(ev_ref) or i < len(ev_sim):
            r += [missing, missing]           # event missing / additional
    return np.array(r)


# ---------------------------------------------------------------- objective --
@dataclass
class Objective:
    """J(theta) = sum_i w_i * d_i(theta)^2  + prior term.

    simulate : callable(dict params) -> result dict
    base     : nominal parameter dict (the ValDyn values)
    names    : list of calibrated parameter names
    channels : list of (ref_key, result_key, weight, sign[, opts]) ;
               opts: dict(hide_y0=0.01) to ignore simulated points that would
               sit on the black zero-axis of the ValDyn plot
    log      : names that are optimised in log space (positive scale params)
    prior_sigma : {name: sigma in log10 units (or abs units for lin params)}
    """
    simulate: callable
    base: dict
    names: list
    channels: list
    ref: dict
    log: set = field(default_factory=set)
    prior_sigma: dict = field(default_factory=dict)
    xwin: tuple = None
    history: list = field(default_factory=list)

    def to_params(self, theta):
        p = dict(self.base)
        for n, v in zip(self.names, theta):
            p[n] = 10**v if n in self.log else v
        return p

    def theta0(self):
        return np.array([np.log10(self.base[n]) if n in self.log else self.base[n] for n in self.names])

    def channel_errors(self, p):
        res = self.simulate(p)
        d = {}
        for c in self.channels:
            rk, sk, w, sgn = c[:4]; opts = c[4] if len(c) > 4 else {}
            d[rk] = curve_distance(self.ref[rk], res['angle'], sgn*res[sk], self.xwin, **opts)
        return d, res

    def __call__(self, theta, return_parts=False):
        p = self.to_params(theta)
        d, _ = self.channel_errors(p)
        J = sum(c[2]*d[c[0]]**2 for c in self.channels)
        Jp = 0.0
        th0 = self.theta0()
        for i, n in enumerate(self.names):
            if n in self.prior_sigma:
                Jp += ((theta[i]-th0[i])/self.prior_sigma[n])**2
        Jtot = J + 1e-4*Jp            # weak pull towards the reference values
        self.history.append((np.array(theta, float), Jtot, J))
        return (Jtot, d) if return_parts else Jtot


# ---------------------------------------------------------------- sensitivity
def local_sensitivity(fun, theta0, names, steps):
    """Central finite differences of a scalar or vector function.
    Returns (f0, S) with S[:, j] = df/dtheta_j * step_j  (i.e. change of f
    for a 'typical' parameter change) - comparable across parameters."""
    f0 = np.atleast_1d(fun(theta0))
    S = np.zeros((len(f0), len(names)))
    for j in range(len(names)):
        tp = np.array(theta0, float); tm = tp.copy()
        tp[j] += steps[j]; tm[j] -= steps[j]
        S[:, j] = (np.atleast_1d(fun(tp)) - np.atleast_1d(fun(tm)))/2
    return f0, S


def morris_screening(fun, lower, upper, r=12, levels=6, seed=0):
    """Morris elementary effects. Returns mu*, sigma per parameter.
    mu*   (mean |EE|): overall influence
    sigma (std  EE ) : non-linearity / interaction
    """
    rng = np.random.default_rng(seed)
    k = len(lower); lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    delta = levels/(2*(levels-1))
    grid = np.linspace(0, 1, levels)
    EE = [[] for _ in range(k)]
    for _ in range(r):
        x = rng.choice(grid[:levels//2], size=k)          # start in lower half
        f_prev = fun(lower + x*(upper-lower))
        for j in rng.permutation(k):
            x2 = x.copy(); x2[j] += delta
            f_new = fun(lower + x2*(upper-lower))
            EE[j].append((f_new - f_prev)/delta)
            x, f_prev = x2, f_new
    EE = [np.array(e) for e in EE]
    return np.array([np.mean(np.abs(e)) for e in EE]), np.array([np.std(e) for e in EE]), EE


def sweep_1d(fun, theta0, j, values):
    out = []
    for v in values:
        t = np.array(theta0, float); t[j] = v
        out.append(fun(t))
    return np.array(out)


def sweep_2d(fun, theta0, j1, v1, j2, v2):
    Z = np.zeros((len(v2), len(v1)))
    for a, x in enumerate(v1):
        for b, y in enumerate(v2):
            t = np.array(theta0, float); t[j1] = x; t[j2] = y
            Z[b, a] = fun(t)
    return Z


def jacobian_identifiability(resfun, theta, steps):
    """J = d r / d theta (residual vector).  Returns correlation matrix of the
    linearised parameter estimate and the singular values of J (scaled)."""
    r0 = resfun(theta)
    Jm = np.zeros((len(r0), len(theta)))
    for j in range(len(theta)):
        t = np.array(theta, float); t[j] += steps[j]
        Jm[:, j] = (resfun(t)-r0)/steps[j]*steps[j]      # per typical step
    JTJ = Jm.T @ Jm
    cov = np.linalg.pinv(JTJ)
    sd = np.sqrt(np.diag(cov))
    corr = cov/np.outer(sd, sd)
    sv = np.linalg.svd(Jm, compute_uv=False)
    return corr, sv, Jm


# ------------------------------------------------- statistical (chaotic) ----
def envelope_stats(x, y, xr, bin_w=2.0, win_w=45.0, ref_cloud=False):
    """Phase-windowed envelope statistics of a signal / point cloud.

    1. split x into bins of width bin_w -> upper/lower envelope per bin
       (max/min for simulated curves, 97/3 % percentile for digitized clouds)
    2. average upper and lower envelope over windows of width win_w
    Returns vector [upper_1..upper_n, lower_1..lower_n] (NaN where no data).

    For chaotic responses the exact trajectory is not reproducible (not even by
    the reference code with another time step), but these envelopes are."""
    x = np.asarray(x); y = np.asarray(y)
    eb = np.arange(xr[0], xr[1]+1e-9, bin_w)
    ib = np.digitize(x, eb)-1
    nb = len(eb)-1
    up = np.full(nb, np.nan); lo = np.full(nb, np.nan)
    order = np.argsort(ib, kind='stable'); ibs = ib[order]; ys = y[order]
    starts = np.searchsorted(ibs, np.arange(nb)); ends = np.searchsorted(ibs, np.arange(nb), side='right')
    for i in range(nb):
        v = ys[starts[i]:ends[i]]
        if len(v) >= (3 if ref_cloud else 1):
            if ref_cloud:
                up[i] = np.percentile(v, 97); lo[i] = np.percentile(v, 3)
            else:
                up[i] = v.max(); lo[i] = v.min()
    nw = int(round((xr[1]-xr[0])/win_w)); per = int(round(win_w/bin_w))
    U = np.array([np.nanmean(up[k*per:(k+1)*per]) if np.any(np.isfinite(up[k*per:(k+1)*per])) else np.nan for k in range(nw)])
    L = np.array([np.nanmean(lo[k*per:(k+1)*per]) if np.any(np.isfinite(lo[k*per:(k+1)*per])) else np.nan for k in range(nw)])
    return np.r_[U, L]


def stat_distance(ref: RefCurve, sims, bin_w=2.0, win_w=45.0, cache={}):
    """RMS difference of envelope statistics, normalised by plot height.
    sims: list of (x, y) tuples (ensemble) -> ensemble-averaged statistics."""
    key = (ref.name, bin_w, win_w)
    if key not in cache:
        cache[key] = envelope_stats(ref.pts[:, 0], ref.pts[:, 1], ref.xr, bin_w, win_w, ref_cloud=True)
    r = cache[key]
    S = np.nanmean([envelope_stats(x, y, ref.xr, bin_w, win_w) for x, y in sims], axis=0)
    m = np.isfinite(r) & np.isfinite(S)
    yr = ref.yr
    Sc = np.clip(S, yr[0]-0.05*(yr[1]-yr[0]), yr[1]+0.05*(yr[1]-yr[0]))   # plot clipping
    return float(np.sqrt(np.mean(((Sc[m]-r[m])/(yr[1]-yr[0]))**2)))
