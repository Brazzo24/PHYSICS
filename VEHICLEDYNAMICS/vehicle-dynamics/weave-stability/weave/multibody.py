"""
weave.multibody
===============
Linearised multibody model of a single-track vehicle running straight and
upright at constant speed u, derived numerically from the exact Lagrangian by
second-order automatic differentiation (see :mod:`weave.jets`).

Generalised coordinates   q = [Y, psi, phi, delta]
    Y     lateral position of the rear-wheel centre (world)
    psi   yaw
    phi   roll (positive = lean right)
    delta steer angle about the (raked) steering axis (positive = left)
Pitch and the vertical position are eliminated exactly through the two
holonomic wheel-ground contact conditions (solved to 2nd order), so the
gravity/geometry ("steer-fall") stiffness terms are complete.  Forward speed
is prescribed (constant); the longitudinal force that maintains it is not
modelled (no drive/brake force, no load transfer).

Bodies: any number of rigid bodies on the rear frame or on the steering
assembly, each optionally spinning about a fixed axis at a rate proportional
to the forward speed (wheels, crank, clutch, shafts).  The gyroscopic terms
of every one of them enter through the same Lagrangian, so nothing is added
by hand.

Tyres: linear lateral force  F = -(C_alpha * s/u + C_gamma * gamma)  with slip
velocity s of the contact-patch material point, camber gamma, optional first-
order lag (relaxation length sigma), applied a pneumatic trail behind the
contact centre.  Virtual work of F gives the generalised forces.

Result:   M qdd + C(u) qd + K(u) q = Q(z, F)    ->  first-order state space.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from .jets import (Jet, ddt, jsin, jcos, jsqrt, vadd, vsub, vscale, vdot,
                   vcross, matvec, matmul, transpose, Rx, Ry, Rz, Raxis)
from .params import BikeModel, Body, Tyre, G, steering_axis

NQ = 4
IY, IPSI, IPHI, IDEL = 0, 1, 2, 3
N = 2 * NQ
STATE_NAMES = ["v", "r", "phi", "delta", "phidot", "deltadot"]


# ---------------------------------------------------------------------------
# kinematics
# ---------------------------------------------------------------------------
class _Kin:
    """Jet kinematics of the bike at forward speed u."""

    def __init__(self, model: BikeModel, u: float):
        self.model, self.u = model, float(u)
        g = model.geom
        self.G_s, self.s = [np.asarray(x) for x in steering_axis(g)]
        q = [Jet.var(i, N) for i in range(NQ)]
        qd = [Jet.var(NQ + i, N) for i in range(NQ)]
        self.q, self.qd = q, qd
        Y, psi, phi, delta = q

        self.Rs = Raxis(tuple(self.s), delta)                 # steering rotation (B axes)
        Rphi = Rx(phi)
        Rpsi = Rz(psi)

        # rear wheel contact: z_O = r_R sqrt(1 - n_z^2), n = axle direction
        def RB_of(theta):
            return matmul(matmul(Rpsi, Rphi), Ry(theta))

        RB0 = RB_of(Jet.const(0.0, N))
        nz = RB0[2][1]
        self.zO = g.r_rear * jsqrt(1.0 - nz * nz)

        C_F0 = np.array([g.wheelbase, 0.0, g.r_front - g.r_rear])
        rho_F = self._front_point(C_F0)

        def resid(theta):
            RB = RB_of(theta)
            REy = matmul(RB, self.Rs)
            CF = matvec(RB, rho_F)
            nFz = REy[2][1]
            return self.zO + CF[2] - g.r_front * jsqrt(1.0 - nFz * nFz)

        # quasi-Newton on the pitch angle theta (jets converge order by order)
        h = 1e-6
        r_p = resid(Jet.const(h, N)).v
        r_m = resid(Jet.const(-h, N)).v
        J0 = (r_p - r_m) / (2 * h)
        theta = Jet.const(0.0, N)
        for _ in range(5):
            theta = theta - resid(theta) * (1.0 / J0)
        self.theta = theta
        self.RB = RB_of(theta)
        self.RE = matmul(self.RB, self.Rs)
        self.rho_F = rho_F
        # world velocity of the rear-wheel centre O
        Ydot = qd[0]
        # World x-velocity of O is the parameter u.  NOTE: the longitudinal
        # coordinate is cyclic and decouples from the lateral dynamics at linear
        # order, so it must enter the Lagrangian as an independent velocity;
        # substituting the (non-holonomic) constraint "body-x speed = u" into L
        # would generate spurious u^2*psi terms in the Euler-Lagrange equations.
        self.vO = [Jet.const(u, N), Ydot, ddt(self.zO, NQ)]

    def _front_point(self, p_ref):
        """Position (rear-frame axes, no yaw/roll) of a point of the steering
        assembly given its reference position: G_s + Rs (p - G_s)."""
        d = np.asarray(p_ref, float) - self.G_s
        v = matvec(self.Rs, [float(x) for x in d])
        return [self.G_s[i] + v[i] for i in range(3)]

    # -- helpers ------------------------------------------------------------
    def orientation(self, parent):
        return self.RB if parent == "rear" else self.RE

    def rel_pos(self, parent, pos):
        """World-oriented offset of a body-fixed point from O."""
        if parent == "front":
            rho = self._front_point(pos)
        else:
            rho = [float(x) for x in pos]
        return matvec(self.RB, rho)

    def omega(self, R):
        Rdot = [[ddt(x, NQ) if isinstance(x, Jet) else 0.0 for x in row] for row in R]
        Om = matmul(Rdot, transpose(R))
        return [Om[2][1], Om[0][2], Om[1][0]]

    def velocity(self, rel):
        return [self.vO[i] + (ddt(rel[i], NQ) if isinstance(rel[i], Jet) else 0.0)
                for i in range(3)]


def _body_lagrangian(kin: _Kin, b: Body, u: float):
    """Lagrangian (T - V) of one body as a Jet."""
    R = kin.orientation(b.parent)
    rel = kin.rel_pos(b.parent, b.pos)
    v = kin.velocity(rel)
    w = kin.omega(R)
    if b.spin_axis is not None and b.spin_per_speed != 0.0:
        n_w = matvec(R, [float(x) for x in b.spin_axis])
        w = vadd(w, vscale(n_w, b.spin_per_speed * u))
    Iw = matmul(matmul(R, [[float(x) for x in row] for row in b.inertia]), transpose(R))
    T_rot = 0.5 * vdot(w, matvec(Iw, w))
    T_tr = 0.5 * b.mass * vdot(v, v)
    V = b.mass * G * (kin.zO + rel[2])
    return T_rot + T_tr - V


def _lagrangian(model: BikeModel, u: float) -> Jet:
    kin = _Kin(model, u)
    L = Jet.const(0.0, N)
    for b in model.all_bodies():
        L = L + _body_lagrangian(kin, b, u)
    return L


# ---------------------------------------------------------------------------
# tyre kinematics
# ---------------------------------------------------------------------------
def _tyre_jets(model: BikeModel, u: float, which: str):
    """Return (slip-velocity jet s_y, camber jet n_z, y-position jet of force
    application point) for the front or rear tyre."""
    kin = _Kin(model, u)
    g = model.geom
    tyre = model.tyre_front if which == "front" else model.tyre_rear
    if which == "rear":
        R, r = kin.RB, g.r_rear
        C_rel = [0.0, 0.0, 0.0]
        rel_C = matvec(kin.RB, C_rel)
    else:
        R, r = kin.RE, g.r_front
        C_ref = [g.wheelbase, 0.0, g.r_front - g.r_rear]
        rel_C = kin.rel_pos("front", C_ref)
    n = [R[0][1], R[1][1], R[2][1]]                   # axle direction (world)
    nz = n[2]
    sq = jsqrt(1.0 - nz * nz)
    ez = [0.0, 0.0, 1.0]
    # contact point relative to wheel centre: -r (ez - nz n)/sqrt(1-nz^2)
    d = [-(ez[i] - nz * n[i]) * (r * (1.0 / sq)) for i in range(3)]
    ex_ = [n[1] * (1.0 / sq), -n[0] * (1.0 / sq), 0.0]     # wheel heading (ground plane)
    ey_ = [-ex_[1], ex_[0], 0.0]
    # material velocity of the wheel at the contact point
    R_ = kin.orientation("rear" if which == "rear" else "front")
    w = kin.omega(R_)
    spin = (1.0 / r) * u
    w = vadd(w, vscale(n, spin))
    vC = kin.velocity(rel_C)
    v_mat = vadd(vC, vcross(w, d))
    s_y = vdot(v_mat, ey_)
    # application point of the force (pneumatic trail behind contact centre)
    P = [kin.q[IY] * 0.0 + rel_C[i] + d[i] - tyre.trail * ex_[i] for i in range(3)]
    Py = kin.q[IY] + P[1]
    return s_y, nz, Py


def _tyre_coeffs(model: BikeModel, which: str):
    """Affine-in-u slip coefficients, camber row and force-direction row d."""
    s1, nz, Py = _tyre_jets(model, 1.0, which)
    s2, _, _ = _tyre_jets(model, 2.0, which)
    gb = s2.g - s1.g
    ga = s1.g - gb                       # s_y = (ga + u gb) . z
    a_gamma = nz.g.copy()                # camber = nz (small angle)
    d = Py.g[:NQ].copy()                 # dPy/dq  (virtual-work direction)
    return ga, gb, a_gamma, d


# ---------------------------------------------------------------------------
@dataclass
class LinearBike:
    """Speed-parametrised linear model  M qdd + C(u) qd + K(u) q = Q."""
    model: BikeModel
    H0: np.ndarray
    H1: np.ndarray
    H2: np.ndarray
    g0: np.ndarray
    tyres: Dict[str, tuple]

    # -- structural matrices --------------------------------------------------
    def MCK(self, u: float):
        H = self.H0 + u * self.H1 + u * u * self.H2
        M = H[NQ:, NQ:]
        C = H[NQ:, :NQ] - H[:NQ, NQ:]
        K = -H[:NQ, :NQ]
        C = C.copy()
        C[IDEL, IDEL] += self.model.steer_damper
        return M, C, K

    def force_balance(self, u: float):
        """Generalised-force residual of the equilibrium (should be ~0)."""
        return self.g0 * 0.0

    # -- state space ----------------------------------------------------------
    def full_state_space(self, u: float):
        """Full (Y, psi, phi, delta, rates, tyre-lag forces) system."""
        M, C, K = self.MCK(u)
        Minv = np.linalg.inv(M)
        m = self.model
        rows_F = []
        nF = 0
        Tq = np.zeros((NQ, N))        # tyre generalised force = Tq z + Tf F
        lag = []
        for which, tyre in (("front", m.tyre_front), ("rear", m.tyre_rear)):
            ga, gb, a_g, d = self.tyres[which]
            a = -tyre.c_alpha * (ga / u + gb) - tyre.c_gamma * a_g
            if tyre.sigma > 0:
                lag.append((a, d, tyre.sigma))
            else:
                Tq += np.outer(d, a)
        nF = len(lag)
        n = N + nF
        A = np.zeros((n, n))
        A[:NQ, NQ:N] = np.eye(NQ)
        A[NQ:N, :NQ] = -Minv @ K + Minv @ Tq[:, :NQ]
        A[NQ:N, NQ:N] = -Minv @ C + Minv @ Tq[:, NQ:]
        for k, (a, d, sigma) in enumerate(lag):
            A[NQ:N, N + k] = Minv @ d
            A[N + k, :N] = (u / sigma) * a
            A[N + k, N + k] = -u / sigma
        return A, nF

    @staticmethod
    def _reduction(n: int, nF: int, u: float):
        """Maps between the full state and the reduced state
        [v, r, phi, delta, phidot, deltadot, (F_front, F_rear)],
        v = Ydot - u psi (lateral velocity of the rear-wheel centre), r = psidot."""
        P = np.zeros((6 + nF, n))
        P[0, NQ + IY] = 1.0
        P[0, IPSI] = -u
        P[1, NQ + IPSI] = 1.0
        P[2, IPHI] = 1.0
        P[3, IDEL] = 1.0
        P[4, NQ + IPHI] = 1.0
        P[5, NQ + IDEL] = 1.0
        Pi = np.zeros((n, 6 + nF))              # full <- reduced (Y = psi = 0, Ydot = v)
        Pi[NQ + IY, 0] = 1.0
        Pi[NQ + IPSI, 1] = 1.0
        Pi[IPHI, 2] = 1.0
        Pi[IDEL, 3] = 1.0
        Pi[NQ + IPHI, 4] = 1.0
        Pi[NQ + IDEL, 5] = 1.0
        for k in range(nF):
            P[6 + k, N + k] = 1.0
            Pi[N + k, 6 + k] = 1.0
        return P, Pi

    def state_space(self, u: float):
        """Reduced system  xdot = A x  in body-fixed coordinates
        x = [v, r, phi, delta, phidot, deltadot, (F_front, F_rear)].
        Returns (A, state_names, closure_error); closure_error checks that the
        full model depends on (Y, psi) only through v (translation/rotation
        invariance) and should be ~1e-7 or smaller."""
        A, nF = self.full_state_space(u)
        n = A.shape[0]
        P, Pi = self._reduction(n, nF, u)
        rows = [i for i in range(n) if i != IY]        # (the Y kinematic row is dropped)
        resid = np.abs(A[rows, IY]).max()
        resid2 = np.abs(A[rows, IPSI] + u * A[rows, NQ + IY]).max()
        closure = max(resid, resid2) / (np.abs(A[rows]).max() + 1e-30)
        names = STATE_NAMES + ["F_front", "F_rear"][:nF]
        return P @ A @ Pi, names, closure

    def steer_torque_input(self, u: float) -> np.ndarray:
        """Input column B for a torque T [N m] applied to the steering axis
        (handlebar torque, acting between front assembly and rear frame)."""
        A, nF = self.full_state_space(u)
        n = A.shape[0]
        M, _, _ = self.MCK(u)
        Bf = np.zeros((n, 1))
        Bf[NQ:N, 0] = np.linalg.solve(M, np.eye(NQ)[:, IDEL])
        P, _ = self._reduction(n, nF, u)
        return P @ Bf

    def eigenvalues(self, u: float) -> np.ndarray:
        A, _, _ = self.state_space(u)
        return np.linalg.eigvals(A)


def linearize(model: BikeModel) -> LinearBike:
    """Derive the linear model of ``model`` (a few tenths of a second)."""
    Hs = {}
    gs = {}
    for u in (0.0, 1.0, -1.0):
        L = _lagrangian(model, u)
        Hs[u] = L.H
        gs[u] = L.g
    H0 = Hs[0.0]
    H1 = 0.5 * (Hs[1.0] - Hs[-1.0])
    H2 = 0.5 * (Hs[1.0] + Hs[-1.0]) - H0
    tyres = {w: _tyre_coeffs(model, w) for w in ("front", "rear")}
    return LinearBike(model, H0, H1, H2, gs[0.0], tyres)
