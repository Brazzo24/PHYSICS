"""Trim and linearisation utilities built on the model and force blocks."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .kinematics import make_q, NQ
from .model import Motorcycle, make_inputs, IIN


def static_trim(model: Motorcycle, forces, q9=0.0, q11=0.0):
    """Static equilibrium on its wheels (bike held at roll q9, steer q11).

    Solves the four in-plane equations (r = 1..4) for q1..q4.  Returns
    (q, u_in, residual of all 11 equations): the residuals of the roll and
    steer equations are the holding torques a rider/stand must supply.
    """
    def res(z):
        q = make_q(q1=z[0], q2=z[1], q3=z[2], q4=z[3], q9=q9, q11=q11)
        k = model.config(q)
        uin = forces.inputs(0.0, k, damping=False)
        return (model.input_matrix(k) @ uin)[:4] / 1e3
    p = model.p
    z0 = [0.08, 0.5 * (p.q2_top + p.q2_bottom) + 0.05, 0.006, 0.007]
    sol = least_squares(res, z0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    q = make_q(q1=sol.x[0], q2=sol.x[1], q3=sol.x[2], q4=sol.x[3], q9=q9, q11=q11)
    k = model.config(q)
    uin = forces.inputs(0.0, k, damping=False)
    return q, uin, model.input_matrix(k) @ uin


def linearise(model: Motorcycle, input_fn, q0, u0, h=1e-6):
    """Numerical Jacobian of the closed-loop state derivative
    x_dot = f(x), x = [q, u], including the state-dependent force blocks
    (the 'missing' suspension/tyre dynamics of the Hammerstein A matrix)."""
    x0 = np.concatenate([q0, u0])
    f = lambda x: model.rhs(0.0, x, input_fn)
    n = len(x0)
    J = np.zeros((n, n))
    for i in range(n):
        d = np.zeros(n); d[i] = h
        J[:, i] = (f(x0 + d) - f(x0 - d)) / (2 * h)
    return J


def rolling_spin_rates(k, u):
    """Wheel spin rates u5, u6 for which the wheel-fixed points at the
    contacts have no longitudinal velocity (free rolling, zero slip)."""
    u = u.copy()
    for col, pt, heading in ((5, "T", k["R_M"][:, 0]), (4, "S", k["R_H"][:, 0])):
        a = k["Jv_" + pt].T @ heading            # d(v . heading)/du
        u[col] = 0.0
        u[col] = -(a @ u) / a[col]
    return u


def steady_turn(model: Motorcycle, forces, V, radius, turn="right"):
    """Steady-state cornering trim without a tyre model.

    The rear reference point O moves on a circle of the given radius at
    speed V, all relative speeds are zero and both wheels roll freely.
    Unknowns (11): q1..q4, roll q9, steer q11, the lateral tyre forces
    Ffwy, Frwy, the rear longitudinal force Frwx with its axle torque Trwy,
    and the rider's steering torque.  Equations (12): the eleven equations
    of motion with the centripetal u_dot, plus zero lateral velocity of the
    front contact point (kinematic steering).  Solved by least squares; the
    front-wheel spin equation is satisfied identically.
    """
    p = model.p
    sgn = -1.0 if turn == "right" else 1.0           # yaw rate: + = left
    yaw = sgn * V / radius
    c = IIN

    def evaluate(z):
        q = make_q(q1=z[0], q2=z[1], q3=z[2] * 0.1, q4=z[3] * 0.1,
                   q9=z[4], q11=z[5] * 0.1)
        u = np.zeros(NQ)
        u[6], u[9] = V, yaw                           # O moves along n1 (psi = 0)
        u = rolling_spin_rates(model.config(q, u), u)
        k = model.config(q, u)
        uin = forces.inputs(0.0, k, damping=False)
        uin[c["Ffwy"]], uin[c["Frwy"]], uin[c["Frwx"]] = z[6:9] * 1e3
        uin[c["Trwy"]], uin[c["Tsteer"]] = z[9] * 1e3, z[10] * 1e2
        ud = np.zeros(NQ)
        ud[7] = -V * yaw              # d/dt (V m1) = -V psi_dot n3 at psi = 0
        r = model.mass_matrix(k) @ ud + model.bias(k) - model.input_matrix(k) @ uin
        slip = (k["Jv_S"] @ u) @ k["R_H"][:, 2] / V
        return np.concatenate([r / 1e3, [10 * slip]]), (q, u, uin, k)

    z0 = np.array([0.08, 0.15, 0.06, 0.07, -sgn * np.arctan(V**2 / (p.g * radius)),
                   sgn * 1.39 / radius * 10, 0, 0, 0, 0, 0], float)
    sol = least_squares(lambda z: evaluate(z)[0], z0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    r, (q, u, uin, k) = evaluate(sol.x)
    return dict(q=q, u=u, u_in=uin, k=k, residual=r, yaw_rate=yaw)
