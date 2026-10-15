"""Phasor (rotating-vector) formulation of crankshaft mass balancing.

Every inertia force in the x-z plane is written as a complex number  c = F_z + j F_x.
A vector of fixed length turning with the crank (angle theta from +z towards +x) is then
c = P exp(+j theta); a counter-rotating one is P exp(-j theta).

With the usual 2nd-order expansion of the slider-crank acceleration, the inertia force of
cylinder i (axis direction beta, crankpin angle psi, theta_i = theta + psi - beta) is

    rotating      : m_rot r w^2 e^{j(theta+psi)}
    recip, order n: m_osc r w^2 k_n cos(n theta_i) e^{j beta},    k_1 = 1, k_2 = Lambda = r/l
                  = m_osc r w^2 k_n / 2 [ e^{j(n psi - (n-1) beta)} e^{+jn theta}
                                       + e^{j((n+1) beta - n psi)} e^{-jn theta} ]

i.e. every term is a forward (+n) or backward (-n) rotating phasor - exactly the
"fictitious rotating / counter-rotating masses" of the paper (Sec. 2, Fig. 3/4), but valid
for any bank angle, split crankpins and arbitrary axial positions.

A moment phasor is obtained by weighting each force phasor with its axial position y:
    M_x = Re(sum y c),   M_z = -Im(sum y c)       (r = (0, y, 0), F = (F_x, 0, F_z))

Each balancer of order n and sense s with plane positions y_k produces  n^2 U_k e^{j s n theta}
(U_k = m_k r_c,k e^{j phi_k}).  For each harmonic (n, s) two complex equations

    sum_k n^2 U_k       = -P_F(n, s)
    sum_k n^2 y_k U_k   = -P_M(n, s)

are solved: exactly with two planes, minimum-norm with more, force-only with one plane.
This is the matrix form of Eqs. (10)-(23) of the paper, solved in one go instead of picking
the equation with non-zero numerator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Balancer, Engine

HARMONICS = [(1, +1), (1, -1), (2, +1), (2, -1)]
HARMONIC_LABEL = {(1, +1): "1st order, forward (+1x)", (1, -1): "1st order, backward (-1x)",
                  (2, +1): "2nd order, forward (+2x)", (2, -1): "2nd order, backward (-2x)"}

D2R = np.pi / 180.0


@dataclass
class Phasors:
    """Force phasors [kg m] and moment phasors [kg m^2]; multiply by omega^2 for N, N m."""
    F: dict = field(default_factory=lambda: {h: 0j for h in HARMONICS})
    M: dict = field(default_factory=lambda: {h: 0j for h in HARMONICS})

    def __add__(self, o: "Phasors") -> "Phasors":
        return Phasors({h: self.F[h] + o.F[h] for h in HARMONICS},
                       {h: self.M[h] + o.M[h] for h in HARMONICS})

    def scaled(self, k: float) -> "Phasors":
        return Phasors({h: k * v for h, v in self.F.items()}, {h: k * v for h, v in self.M.items()})


@dataclass
class PlaneResult:
    y_mm: float
    U: complex              # mass * radius [kg m], phase at theta = 0
    r_c_mm: float

    @property
    def mr_gmm(self) -> float:
        return abs(self.U) * 1e6

    @property
    def mass_g(self) -> float:
        return abs(self.U) / (self.r_c_mm * 1e-3) * 1e3 if self.r_c_mm > 0 else float("nan")

    @property
    def angle_deg(self) -> float:
        """Counterweight CG direction at theta = 0, from +z towards +x [deg, 0..360)."""
        return float(np.degrees(np.arctan2(self.U.imag, self.U.real)) % 360.0) if abs(self.U) > 0 else 0.0


@dataclass
class BalancerResult:
    spec: Balancer
    target: Phasors          # what this balancer was asked to cancel (only its own harmonic is non-zero)
    planes: list[PlaneResult]
    contribution: Phasors    # what it actually produces

    @property
    def total_mass_g(self) -> float:
        return sum(p.mass_g for p in self.planes)


@dataclass
class Results:
    engine: Engine
    rotating: Phasors
    recip1: Phasors
    recip2: Phasors
    balancers: list[BalancerResult]

    @property
    def unbalanced(self) -> Phasors:
        return self.rotating + self.recip1 + self.recip2

    @property
    def balancing(self) -> Phasors:
        out = Phasors()
        for b in self.balancers:
            out = out + b.contribution
        return out

    @property
    def residual(self) -> Phasors:
        return self.unbalanced + self.balancing


# --------------------------------------------------------------------------- engine phasors
def engine_phasors(eng: Engine) -> tuple[Phasors, Phasors, Phasors]:
    r = eng.r_mm * 1e-3
    lam = eng.Lambda
    rot, rec1, rec2 = Phasors(), Phasors(), Phasors()
    for c in eng.cylinders:
        y = c.y_mm * 1e-3
        psi, beta = c.pin_deg * D2R, c.bank_deg * D2R
        m_rot = (c.m_rot_g + c.m_throw_g) * 1e-3
        m_osc = c.m_osc_g * 1e-3

        p = m_rot * r * np.exp(1j * psi)
        rot.F[(1, 1)] += p
        rot.M[(1, 1)] += y * p

        for n, k, tgt in ((1, 1.0, rec1), (2, lam, rec2)):
            a = 0.5 * m_osc * r * k
            pf = a * np.exp(1j * (n * psi - (n - 1) * beta))
            pb = a * np.exp(1j * ((n + 1) * beta - n * psi))
            tgt.F[(n, 1)] += pf
            tgt.M[(n, 1)] += y * pf
            tgt.F[(n, -1)] += pb
            tgt.M[(n, -1)] += y * pb
    return rot, rec1, rec2


# --------------------------------------------------------------------------- balancer solve
def solve_planes(n: int, ys_m: np.ndarray, PF: complex, PM: complex) -> np.ndarray:
    """Unbalances U_k [kg m] so that sum n^2 U = -PF and sum n^2 y U = -PM."""
    ys_m = np.asarray(ys_m, float)
    K = len(ys_m)
    if K == 0:
        return np.zeros(0, complex)
    if K == 1:  # only the force can be cancelled
        return np.array([-PF / n**2])
    L = max(np.max(np.abs(ys_m)), 1e-3)  # scale moment row -> same units as force row
    A = n**2 * np.vstack([np.ones(K), ys_m / L]).astype(complex)
    b = -np.array([PF, PM / L], complex)
    U, *_ = np.linalg.lstsq(A, b, rcond=None)
    return U


def _run_balancer(spec: Balancer, target: Phasors) -> BalancerResult:
    h = (spec.order, spec.sense)
    ys = np.array(spec.planes_y_mm, float) * 1e-3
    U = solve_planes(spec.order, ys, target.F[h], target.M[h])
    planes = [PlaneResult(float(y * 1e3), complex(u), spec.r_c_mm) for y, u in zip(ys, U)]
    contrib = Phasors()
    contrib.F[h] = complex(np.sum(spec.order**2 * U))
    contrib.M[h] = complex(np.sum(spec.order**2 * ys * U))
    return BalancerResult(spec, target, planes, contrib)


def _only(ph: Phasors, h, k: float = 1.0) -> Phasors:
    out = Phasors()
    out.F[h], out.M[h] = k * ph.F[h], k * ph.M[h]
    return out


def solve(eng: Engine) -> Results:
    rot, rec1, rec2 = engine_phasors(eng)
    f = float(np.clip(eng.fwd_primary_on_crank, 0.0, 1.0))
    corot1 = eng.shaft(1, +1)
    if corot1 is None or not corot1.enabled:
        pass  # whatever is not on the crank stays as residual

    balancers: list[BalancerResult] = []
    if eng.crank_cw.enabled:
        tgt = _only(rot, (1, 1)) + _only(rec1, (1, 1), f)
        balancers.append(_run_balancer(eng.crank_cw, tgt))

    targets = {
        (1, +1): _only(rec1, (1, 1), 1.0 - f),
        (1, -1): _only(rec1, (1, -1)),
        (2, +1): _only(rec2, (2, 1)),
        (2, -1): _only(rec2, (2, -1)),
    }
    for s in eng.shafts:
        if s.enabled:
            balancers.append(_run_balancer(s, targets[(s.order, s.sense)]))
    return Results(eng, rot, rec1, rec2, balancers)


# --------------------------------------------------------------------------- time domain
def phasor_timeseries(ph: Phasors, theta_rad: np.ndarray, omega: float) -> dict[str, np.ndarray]:
    cF = np.zeros_like(theta_rad, complex)
    cM = np.zeros_like(theta_rad, complex)
    for (n, s) in HARMONICS:
        e = np.exp(1j * s * n * theta_rad)
        cF += ph.F[(n, s)] * e
        cM += ph.M[(n, s)] * e
    w2 = omega**2
    return {"Fx": w2 * cF.imag, "Fz": w2 * cF.real, "Mx": w2 * cM.real, "Mz": -w2 * cM.imag}


def timeseries(res: Results, theta_deg: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Forces [N] and moments [N m] about the origin for every contributor over crank angle."""
    th = np.asarray(theta_deg, float) * D2R
    w = res.engine.omega
    out = {
        "Rotating masses": phasor_timeseries(res.rotating, th, w),
        "Reciprocating 1st": phasor_timeseries(res.recip1, th, w),
        "Reciprocating 2nd": phasor_timeseries(res.recip2, th, w),
        "Engine (unbalanced)": phasor_timeseries(res.unbalanced, th, w),
    }
    My_bal = np.zeros_like(th)
    for b in res.balancers:
        ts = phasor_timeseries(b.contribution, th, w)
        ts["My"] = b.spec.z_mm * 1e-3 * ts["Fx"] - b.spec.x_mm * 1e-3 * ts["Fz"]
        My_bal += ts["My"]
        out[b.spec.name] = ts
    out["Balancers (sum)"] = phasor_timeseries(res.balancing, th, w) | {"My": My_bal}
    out["Residual"] = phasor_timeseries(res.residual, th, w) | {"My": My_bal}
    return out


def harmonic_table(res: Results) -> list[dict]:
    """Amplitudes per rotating harmonic before and after balancing."""
    w2 = res.engine.omega**2
    u, rsd = res.unbalanced, res.residual
    rows = []
    for h in HARMONICS:
        rows.append({
            "Harmonic": HARMONIC_LABEL[h],
            "Engine |F| [N]": abs(u.F[h]) * w2,
            "Engine |M| [N m]": abs(u.M[h]) * w2,
            "Residual |F| [N]": abs(rsd.F[h]) * w2,
            "Residual |M| [N m]": abs(rsd.M[h]) * w2,
        })
    return rows


def tdc_angles(eng: Engine) -> list[float]:
    """Crank angle theta [deg, 0..360) at which each cylinder is at TDC."""
    return [float((c.bank_deg - c.pin_deg) % 360.0) for c in eng.cylinders]
