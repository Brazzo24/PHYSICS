"""Chapter 3: the inverted pendulum on a cart (Fig. 3.2) derived three ways.

q1, u1 : cart position / speed along n1
q2, u2 : pendulum angle from the vertical (about n3) / angular velocity
Pendulum mass centre:  x_p = q1 - L sin q2 ,  y_p = L cos q2.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

t = sp.symbols("t")
MP, MC, IP, L, k, d, g = sp.symbols("M_P M_C I_P L k d g", positive=True)
FC, FP = sp.symbols("F_C F_P")
q1, q2 = sp.Function("q1")(t), sp.Function("q2")(t)
u1, u2, u1d, u2d = sp.symbols("u1 u2 udot1 udot2")
_to_u = {q1.diff(t, 2): u1d, q2.diff(t, 2): u2d, q1.diff(t): u1, q2.diff(t): u2}


def thesis_eom():
    """Eqns. (3.7) written as  lhs - rhs = 0  (list of two expressions)."""
    c2, s2 = sp.cos(q2), sp.sin(q2)
    e1 = (MP + MC) * u1d - MP * L * c2 * u2d - (-MP * L * s2 * u2**2 - d * u1 + FC + FP)
    e2 = MP * L * c2 * u1d - (MP * L**2 + IP) * u2d - (k * q2 + FP * L * c2 - MP * g * L * s2)
    return [e1, e2]


def newton():
    """Section 3.6: free-body equations, rod force F_T eliminated."""
    FT = sp.symbols("F_T")
    c2, s2 = sp.cos(q2), sp.sin(q2)
    xp, yp = q1 - L * s2, L * c2
    xpp, ypp = xp.diff(t, 2), yp.diff(t, 2)
    # pendulum, x and y  (3.2), (3.3); cart, x  (3.4)
    px = FP - MP * xpp + IP * q2.diff(t, 2) * c2 / L + k * q2 * c2 / L + FT * s2
    py = -MP * ypp - MP * g + IP * q2.diff(t, 2) * s2 / L + k * q2 * s2 / L - FT * c2
    cx = FC - MC * q1.diff(t, 2) - d * q1.diff(t) - IP * q2.diff(t, 2) * c2 / L \
        - k * q2 * c2 / L - FT * s2
    e1 = px + cx                                  # (3.5)
    e2 = sp.expand(c2 * px + s2 * py) * L         # (3.6)
    return [_elim(-e1, FT), _elim(e2, FT)]


def _elim(e, FT):
    e = sp.simplify(e.subs(_to_u))
    assert not e.has(FT)
    return e


def lagrange():
    """Section 3.7: d/dt(dL/dqdot) - dL/dq = Q."""
    c2 = sp.cos(q2)
    q1d, q2d = q1.diff(t), q2.diff(t)
    vP2 = (q1d - L * q2d * c2) ** 2 + (L * q2d * sp.sin(q2)) ** 2
    K = MC * q1d**2 / 2 + MP * vP2 / 2 + IP * q2d**2 / 2
    U = k * q2**2 / 2 + MP * g * L * c2
    Lag = K - U
    Q = [-d * q1d + FC + FP, -FP * L * c2]
    eqs = []
    for qi, Qi in zip((q1, q2), Q):
        e = sp.diff(Lag.diff(qi.diff(t)), t) - Lag.diff(qi) - Qi
        eqs.append(sp.simplify(e.subs(_to_u)))
    return eqs


def kane():
    """Section 3.8: F_r + F_r* = 0 with partial velocities (Table 3.2).

    Returns (equations, partial velocity table)."""
    c2, s2 = sp.cos(q2), sp.sin(q2)
    n1, n2, n3 = sp.Matrix([1, 0, 0]), sp.Matrix([0, 1, 0]), sp.Matrix([0, 0, 1])
    us, uds = [u1, u2], [u1d, u2d]
    vC, wC = u1 * n1, sp.zeros(3, 1)
    wP = u2 * n3
    vP = vC + wP.cross(L * (-s2 * n1 + c2 * n2))        # v^C + w x r^CP
    table = {r + 1: dict(wC=wC.diff(x), vC=vC.diff(x), wP=wP.diff(x), vP=vP.diff(x))
             for r, x in enumerate(us)}

    def accel(v):       # d/dt with q_dot = u
        a = sp.zeros(3, 1)
        for x, xd in zip(us, uds):
            a += v.diff(x) * xd
        return a + v.diff(q2) * u2

    aC, aP, alP = accel(vC), accel(vP), accel(wP)
    F_C, F_P = (FC - d * u1) * n1, -MP * g * n2 + FP * n1
    T_C, T_P = k * q2 * n3, -k * q2 * n3
    eqs = []
    for r in (1, 2):
        pv = table[r]
        Fr = F_C.dot(pv["vC"]) + T_C.dot(pv["wC"]) + F_P.dot(pv["vP"]) + T_P.dot(pv["wP"])
        Frs = -MC * aC.dot(pv["vC"]) - MP * aP.dot(pv["vP"]) - IP * alP.dot(pv["wP"])
        eqs.append(sp.simplify(Fr + Frs))
    return eqs, table


def equivalent(eqs_a, eqs_b):
    """True if the two sets of two equations describe the same dynamics
    (identical solution for u1dot, u2dot)."""
    sa = sp.solve(eqs_a, [u1d, u2d], dict=True)[0]
    sb = sp.solve(eqs_b, [u1d, u2d], dict=True)[0]
    return all(sp.simplify(sa[x] - sb[x]) == 0 for x in (u1d, u2d))


def numeric_rhs(params):
    """Lambdified state derivative for x = [q1, q2, u1, u2]; inputs F_C, F_P."""
    sol = sp.solve(thesis_eom(), [u1d, u2d], dict=True)[0]
    q2s = sp.symbols("q2s")
    f = sp.lambdify([q2s, u1, u2, FC, FP],
                    [sol[u1d].subs(params).subs(q2, q2s),
                     sol[u2d].subs(params).subs(q2, q2s)])

    def rhs(tt, x, F=lambda tt, x: (0.0, 0.0)):
        fc, fp = F(tt, x)
        a1, a2 = f(x[1], x[2], x[3], fc, fp)
        return np.array([x[2], x[3], a1, a2])
    return rhs
