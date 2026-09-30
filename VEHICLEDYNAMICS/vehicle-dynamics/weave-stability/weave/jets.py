"""
weave.jets
==========
Second-order forward-mode automatic differentiation ("jets").

A :class:`Jet` carries a value, a gradient and a Hessian with respect to a
fixed list of ``n`` independent variables.  Building the Lagrangian of a
multibody system out of Jets gives its exact second-order Taylor expansion,
i.e. exactly the M / C / K matrices of the linearised equations of motion,
without symbolic algebra and without finite differences.

Only what the motorcycle model needs is implemented: + - * /, sin, cos,
sqrt, plus small 3-vector / 3x3-matrix helpers on plain Python lists.
"""
from __future__ import annotations
import numpy as np


class Jet:
    __slots__ = ("v", "g", "H")

    def __init__(self, v, g, H):
        self.v, self.g, self.H = v, g, H

    # ---- constructors ------------------------------------------------------
    @staticmethod
    def const(c, n):
        return Jet(float(c), np.zeros(n), np.zeros((n, n)))

    @staticmethod
    def var(i, n, value=0.0):
        g = np.zeros(n)
        g[i] = 1.0
        return Jet(float(value), g, np.zeros((n, n)))

    @property
    def n(self):
        return self.g.size

    # ---- arithmetic --------------------------------------------------------
    def __add__(self, o):
        if isinstance(o, Jet):
            return Jet(self.v + o.v, self.g + o.g, self.H + o.H)
        return Jet(self.v + o, self.g, self.H)
    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.g, -self.H)

    def __sub__(self, o):
        if isinstance(o, Jet):
            return Jet(self.v - o.v, self.g - o.g, self.H - o.H)
        return Jet(self.v - o, self.g, self.H)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Jet):
            gg = np.outer(self.g, o.g)
            return Jet(self.v * o.v,
                       self.v * o.g + o.v * self.g,
                       self.v * o.H + o.v * self.H + gg + gg.T)
        return Jet(self.v * o, self.g * o, self.H * o)
    __rmul__ = __mul__

    def _apply(self, f0, f1, f2):
        """Chain rule for a scalar function with derivatives f0, f1, f2."""
        return Jet(f0, f1 * self.g, f2 * np.outer(self.g, self.g) + f1 * self.H)

    def recip(self):
        x = self.v
        return self._apply(1.0 / x, -1.0 / x**2, 2.0 / x**3)

    def __truediv__(self, o):
        if isinstance(o, Jet):
            return self * o.recip()
        return self * (1.0 / o)

    def __rtruediv__(self, o):
        return self.recip() * o

    # ---- elementary functions ---------------------------------------------
    def sin(self):
        s, c = np.sin(self.v), np.cos(self.v)
        return self._apply(s, c, -s)

    def cos(self):
        s, c = np.sin(self.v), np.cos(self.v)
        return self._apply(c, -s, -c)

    def sqrt(self):
        r = np.sqrt(self.v)
        return self._apply(r, 0.5 / r, -0.25 / r**3)

    def __repr__(self):
        return f"Jet(v={self.v:.6g}, |g|={np.linalg.norm(self.g):.3g})"


def jsin(x):  return x.sin() if isinstance(x, Jet) else np.sin(x)
def jcos(x):  return x.cos() if isinstance(x, Jet) else np.cos(x)
def jsqrt(x): return x.sqrt() if isinstance(x, Jet) else np.sqrt(x)


def ddt(f: Jet, nq: int) -> Jet:
    """
    Total time derivative of a Jet that depends on the coordinates only
    (variables 0..nq-1), returned as a Jet in the full (q, qdot) variable space
    (variables nq..2nq-1 are the generalised velocities):

        d f / dt = sum_i (df/dq_i) qdot_i

    Its second-order expansion about qdot = 0 follows from f's Hessian.
    """
    n = f.n
    assert n == 2 * nq
    g = np.zeros(n)
    H = np.zeros((n, n))
    g[nq:] = f.g[:nq]
    H[:nq, nq:] = f.H[:nq, :nq]
    H[nq:, :nq] = f.H[:nq, :nq].T
    return Jet(0.0, g, H)


# ---------------------------------------------------------------------------
# tiny linear algebra on lists of Jets / floats
# ---------------------------------------------------------------------------
def vadd(a, b): return [x + y for x, y in zip(a, b)]
def vsub(a, b): return [x - y for x, y in zip(a, b)]
def vscale(a, s): return [x * s for x in a]


def vdot(a, b):
    s = a[0] * b[0]
    for x, y in zip(a[1:], b[1:]):
        s = s + x * y
    return s


def vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def matvec(R, v):
    return [vdot(R[i], v) for i in range(3)]


def matmul(A, B):
    return [[vdot(A[i], [B[0][j], B[1][j], B[2][j]]) for j in range(3)]
            for i in range(3)]


def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def Rx(a):
    c, s = jcos(a), jsin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def Ry(a):
    c, s = jcos(a), jsin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def Rz(a):
    c, s = jcos(a), jsin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def Raxis(k, a):
    """Rotation by angle a (Jet) about constant unit vector k (Rodrigues)."""
    c, s = jcos(a), jsin(a)
    omc = 1.0 - c
    kx, ky, kz = k
    return [[c + kx * kx * omc, kx * ky * omc - kz * s, kx * kz * omc + ky * s],
            [ky * kx * omc + kz * s, c + ky * ky * omc, ky * kz * omc - kx * s],
            [kz * kx * omc - ky * s, kz * ky * omc + kx * s, c + kz * kz * omc]]
