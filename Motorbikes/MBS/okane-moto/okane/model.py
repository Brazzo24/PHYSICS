"""Equations of motion of the 11-d.o.f. motorcycle by Kane's method
(thesis Section 4.10).

    K(q) u_dot + f(q, u, u_in) = 0                              (cf. 4.35)

with  K = sum_bodies  m Jv^T Jv + Jw^T I Jw          (generalised inertia)
      f = cb(q, u) - Bg(q) u_in
      cb = sum_bodies  m Jv^T a_bias + Jw^T (I alpha_bias + w x I w)

Jv / Jw are the tables of partial velocities / partial angular velocities
(3 x 11, one column per generalised speed), i.e. Kane's
    F_r  =  sum F . v_r + T . w_r          (generalised active force)
    F_r* = -sum m a . v_r + (I alpha + w x I w) . w_r   (generalised inertia force)

The input vector u_in is the one of Table 4.9 (21 entries); all forces that
the thesis treats as inputs (tyres, suspension, aerodynamics, chain, gravity)
enter through it.

Sign conventions of the inputs (axis 1 fwd, 2 up, 3 right):

 Ffwx,Ffwy,Ffwz  force of the ground on the FRONT wheel at contact point S,
                 along h1 (wheel heading on the ground), h3 (lateral, right)
                 and h2 = up.
 Frwx,Frwy,Frwz  the same for the REAR wheel at T, along m1, m3, m2.
 Flift, Fdrag    at the centre of pressure R on the main frame: +m2 / -m1.
 Ffork           telescopic fork force, positive EXTENDS the fork
                 (+f2 on the steering assembly F, -f2 on the fork lower C).
 Tmfx,Tmfy,Tmfz  torque on the main frame about a1, a3 (transverse), a2.
 Tsay            suspension/chain torque about the swing-arm pivot, positive
                 INCREASES q2 (+a3 on swing-arm B, -a3 on main frame A).
 Tfwx,Tfwz       torque on the front wheel about h1 and h2 (tyre Mx, Mz).
 Tfwy            torque on the front wheel about its axle f3, reacted by the
                 fork lower C.  Forward rolling is a NEGATIVE spin rate, so
                 a braking torque is positive.
 Trwx,Trwz,Trwy  the same for the rear wheel (m1, m2; axle torque reacted by
                 the swing-arm B; a driving torque is negative).
 Tsteer          rider steering torque about e2 (+ on F, - on A).
 g               gravitational acceleration (so that weight is an 'input').
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .kinematics import Kinematics, NQ, _skew
from .params import BikeParams

INPUTS = ["Ffwx", "Ffwy", "Ffwz", "Frwx", "Frwy", "Frwz", "Flift", "Fdrag",
          "Ffork", "Tmfx", "Tmfy", "Tmfz", "Tsay", "Tfwx", "Tfwy", "Tfwz",
          "Trwx", "Trwy", "Trwz", "Tsteer", "g"]
NIN = len(INPUTS)
IIN = {n: i for i, n in enumerate(INPUTS)}

# body -> (mass attr, inertia attr, point of mass centre, carrier frame,
#          body whose angular velocity applies)
BODY_TABLE = {
    "A": ("MA", "IA", "A", "A", "A"),     # main frame + engine + rider
    "B": ("MB", "IB", "B", "B", "B"),     # swing-arm
    "C": ("MC", "IC", "C", "F", "F"),     # fork lower (same orientation as F)
    "D": ("MD", "ID", "D", "B", "D"),     # rear wheel (axisymmetric about b3)
    "F": ("MF", "IF", "F", "F", "F"),     # steering assembly
    "G": ("MG", "IG", "G", "F", "G"),     # front wheel (axisymmetric about f3)
}


def make_inputs(**kw) -> np.ndarray:
    """Input vector, e.g. ``make_inputs(g=9.81, Ffork=1000.)``."""
    v = np.zeros(NIN)
    for k, val in kw.items():
        v[IIN[k]] = val
    return v


class Motorcycle:
    """Numeric multibody model.

    Parameters
    ----------
    pitch_model : 'exact' | 'thesis'
        how mu = q14 and the coefficients H_r are obtained (see kinematics).
    coef_rates : bool
        include d/dt of the kinematic coefficients in the accelerations
        (True = consistent multibody model, energy conserving;
         False = thesis' assumption of piecewise-constant coefficients).
    """

    def __init__(self, params: BikeParams, pitch_model="exact", coef_rates=True):
        self.p = params
        self.kin = Kinematics(params, pitch_model)
        self.coef_rates = coef_rates

    # ------------------------------------------------------------------
    def config(self, q, u=None):
        return self.kin.config(q, u, coef_rates=self.coef_rates)

    # ------------------------------------------------------------------
    # Kane: generalised inertia forces
    # ------------------------------------------------------------------
    def mass_matrix(self, k) -> np.ndarray:
        """K of eq. (4.35): symmetric, positive definite, 11 x 11."""
        K = np.zeros((NQ, NQ))
        for mass, inertia, pt, frame, wb in BODY_TABLE.values():
            m, I = getattr(self.p, mass), getattr(self.p, inertia)
            R = k["R_" + frame]
            Jv, Jw = k["Jv_" + pt], k["Jw_" + wb]
            K += m * Jv.T @ Jv + Jw.T @ (R @ I @ R.T) @ Jw
        return K

    def bias(self, k) -> np.ndarray:
        """Velocity-dependent part of -F_r* (centripetal, Coriolis, gyroscopic)."""
        cb = np.zeros(NQ)
        for mass, inertia, pt, frame, wb in BODY_TABLE.values():
            m, I = getattr(self.p, mass), getattr(self.p, inertia)
            R = k["R_" + frame]
            IN = R @ I @ R.T
            w = k["w_" + wb]
            cb += m * k["Jv_" + pt].T @ k["ab_" + pt]
            cb += k["Jw_" + wb].T @ (IN @ k["alb_" + wb] + np.cross(w, IN @ w))
        return cb

    # ------------------------------------------------------------------
    # Kane: generalised active forces, F_r = Bg[r, :] @ u_in
    # ------------------------------------------------------------------
    def input_matrix(self, k) -> np.ndarray:
        """Bg (11 x 21): generalised active force per unit input (the thesis'
        -beta of eq. 4.36)."""
        p = self.p
        B = np.zeros((NQ, NIN))
        RM, RA, RE, RF, RH = k["R_M"], k["R_A"], k["R_E"], k["R_F"], k["R_H"]
        m1, m2, m3 = RM.T
        a1, a2, a3 = RA.T
        e2, f2, f3 = RE[:, 1], RF[:, 1], RF[:, 2]
        h1, h2, h3 = RH.T
        JS, JT, JR = k["Jv_S"], k["Jv_T"], k["Jv_R"]
        JwA, JwB, JwD, JwF, JwG = (k["Jw_" + b] for b in "ABDFG")
        c = IIN
        # tyre forces at the contact points (acting on the wheels)
        B[:, c["Ffwx"]], B[:, c["Ffwy"]], B[:, c["Ffwz"]] = JS.T @ h1, JS.T @ h3, JS.T @ h2
        B[:, c["Frwx"]], B[:, c["Frwy"]], B[:, c["Frwz"]] = JT.T @ m1, JT.T @ m3, JT.T @ m2
        # aerodynamics at R
        B[:, c["Flift"]], B[:, c["Fdrag"]] = JR.T @ m2, -JR.T @ m1
        # fork: equal and opposite on F and C along the sliding axis
        # (prismatic joint: both forces share one line of action along f2, the
        #  relative velocity of the two bodies along it is -u1, so only the
        #  generalised force of q1 is non-zero and equals Ffork)
        B[0, c["Ffork"]] = 1.0
        # torques on the main frame
        B[:, c["Tmfx"]], B[:, c["Tmfy"]], B[:, c["Tmfz"]] = JwA.T @ a1, JwA.T @ a3, JwA.T @ a2
        # swing-arm torque (suspension + chain), equal and opposite on B / A
        B[:, c["Tsay"]] = (JwB - JwA).T @ a3
        # front wheel
        B[:, c["Tfwx"]], B[:, c["Tfwz"]] = JwG.T @ h1, JwG.T @ h2
        B[:, c["Tfwy"]] = (JwG - JwF).T @ f3
        # rear wheel
        B[:, c["Trwx"]], B[:, c["Trwz"]] = JwD.T @ m1, JwD.T @ m2
        B[:, c["Trwy"]] = (JwD - JwB).T @ a3
        # steering torque
        B[:, c["Tsteer"]] = (JwF - JwA).T @ e2
        # gravity: -M g n2 at every mass centre
        n2 = np.array([0.0, 1.0, 0.0])
        for mass, _, pt, _, _ in BODY_TABLE.values():
            B[:, c["g"]] -= getattr(p, mass) * k["Jv_" + pt].T @ n2
        return B

    # ------------------------------------------------------------------
    def accelerations(self, q, u, u_in, k=None):
        """u_dot from K u_dot = Bg u_in - cb, solved by Cholesky (Sec. 4.14)."""
        k = k or self.config(q, u)
        K = self.mass_matrix(k)
        rhs = self.input_matrix(k) @ u_in - self.bias(k)
        return cho_solve(cho_factor(K), rhs)

    def rhs(self, t, x, input_fn):
        """State derivative for x = [q (11), u (11)].
        ``input_fn(t, k)`` returns u_in given the configuration dict k."""
        q, u = x[:NQ], x[NQ:]
        k = self.config(q, u)
        return np.concatenate([u, self.accelerations(q, u, input_fn(t, k), k)])

    # ------------------------------------------------------------------
    # energies (for verification)
    # ------------------------------------------------------------------
    def kinetic_energy(self, k):
        return 0.5 * k["u"] @ self.mass_matrix(k) @ k["u"]

    def potential_energy(self, k, g=None):
        g = self.p.g if g is None else g
        return g * sum(getattr(self.p, m) * k["r_" + pt][1]
                       for m, _, pt, _, _ in BODY_TABLE.values())

    # ------------------------------------------------------------------
    # thesis-style structure:  f = alpha x + beta u_in + gamma w   (4.36)
    # ------------------------------------------------------------------
    def quadratic_bias(self, q):
        """cb is a quadratic form in the generalised speeds:
        cb_r = sum_{j<=k} Gam[r, j, k] u_j u_k.  Returns Gam (11, 11, 11),
        upper triangular in (j, k) -- the analogue of eq. (4.33)."""
        def cb(u):
            return self.bias(self.config(q, u))
        e = np.eye(NQ)
        single = [cb(e[j]) for j in range(NQ)]
        Gam = np.zeros((NQ, NQ, NQ))
        for j in range(NQ):
            Gam[:, j, j] = single[j]
            for kk in range(j + 1, NQ):
                Gam[:, j, kk] = cb(e[j] + e[kk]) - single[j] - single[kk]
        return Gam

    W_TERMS = [(8, 8), (8, 9), (8, 10), (9, 10)]   # u9^2, u9u10, u9u11, u10u11

    def state_space(self, q, u):
        """Thesis state-space form about the scheduling point (q, u):

            x_dot = A x + B u_in + R w ,   x = [u1, q1, u2, q2, ..., u11, q11]
            w     = [u9^2, u9 u10, u9 u11, u10 u11]                 (4.38)

        Interpretation used here (the thesis does not print alpha, gamma):
        * terms of cb that contain a wheel spin rate u5 or u6 are linear in
          the remaining speeds once u5, u6 are scheduled -> matrix A
          ("wheel gyroscopic moments are handled within the A matrix");
        * the four products in w are kept explicitly -> matrix R;
        * all other velocity products are discarded.
        Also returns K, alpha_u (11x11), Bg, gamma, and Gam.
        """
        k = self.config(q, u)
        K = self.mass_matrix(k)
        Bg = self.input_matrix(k)
        Gam = self.quadratic_bias(q)
        alpha = np.zeros((NQ, NQ))
        for s in (4, 5):                       # scheduled wheel spin rates
            for j in range(NQ):
                if j in (4, 5):
                    continue
                a, b = min(s, j), max(s, j)
                alpha[:, j] += Gam[:, a, b] * u[s]
        gamma = np.stack([Gam[:, a, b] for a, b in self.W_TERMS], axis=1)
        Ki = np.linalg.inv(K)
        A = np.zeros((2 * NQ, 2 * NQ))
        B = np.zeros((2 * NQ, NIN))
        R = np.zeros((2 * NQ, 4))
        A[0::2, 0::2] = -Ki @ alpha            # u_dot rows, u columns
        A[1::2, 0::2] = np.eye(NQ)             # q_dot = u
        B[0::2] = Ki @ Bg
        R[0::2] = -Ki @ gamma
        return dict(A=A, B=B, R=R, K=K, alpha=alpha, Bg=Bg, gamma=gamma,
                    Gam=Gam, k=k)

    def accelerations_thesis(self, q, u, u_in):
        """u_dot of the simplified model (4.37): only the velocity products
        retained by the thesis (see :meth:`state_space`), obtained from a few
        evaluations of the exact quadratic form."""
        def cb(v):
            return self.bias(self.config(q, v))
        spin = np.zeros(NQ); spin[[4, 5]] = u[[4, 5]]
        rest = u - spin
        gyro = cb(u) - cb(spin) - cb(rest)            # 2 Q(spin, rest)
        r3 = np.zeros(NQ); r3[8:] = u[8:]
        e10 = np.zeros(NQ); e10[9] = u[9]
        e11 = np.zeros(NQ); e11[10] = u[10]
        quad = cb(r3) - cb(e10) - cb(e11)             # u9^2,u9u10,u9u11,u10u11
        k = self.config(q, u)
        K = self.mass_matrix(k)
        return cho_solve(cho_factor(K), self.input_matrix(k) @ u_in - gyro - quad)

    # ------------------------------------------------------------------
    # output matrix C (Section 4.11)
    # ------------------------------------------------------------------
    def output_matrix(self, k):
        """C (8 x 22) for x = [u1, q1, ..., u11, q11]: q1, q2, the three body
        rates of the main frame (rate gyro: about a1, a2, a3), u5, u6, q9."""
        C = np.zeros((8, 2 * NQ))
        C[0, 1] = 1.0                       # q1
        C[1, 3] = 1.0                       # q2
        C[2:5, 0::2] = k["R_A"].T @ k["Jw_A"]
        C[5, 8] = 1.0                       # u5
        C[6, 10] = 1.0                      # u6
        C[7, 17] = 1.0                      # q9
        return C


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def to_thesis_state(q, u):
    x = np.empty(2 * NQ)
    x[0::2], x[1::2] = u, q
    return x


def rk4(f, t0, x0, dt, n, record=None):
    """Fixed-step classical Runge-Kutta; returns (t, X)."""
    X = np.empty((n + 1, len(x0)))
    X[0] = x0
    t = t0 + dt * np.arange(n + 1)
    x = np.array(x0, float)
    for i in range(n):
        k1 = f(t[i], x)
        k2 = f(t[i] + dt / 2, x + dt / 2 * k1)
        k3 = f(t[i] + dt / 2, x + dt / 2 * k2)
        k4 = f(t[i] + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[i + 1] = x
    return t, X
