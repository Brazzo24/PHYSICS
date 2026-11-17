"""Reference frames of the motorcycle model (thesis Section 4.4, Table 4.4).

Axis convention used throughout the thesis (Fig. 4.4 / 4.5):

    axis 1 : forward        ("x")
    axis 2 : up             ("z" in the thesis' wording of forces/positions)
    axis 3 : to the right   ("y")

so the triads are right-handed (1 x 2 = 3).  Positive yaw (about axis 2)
turns the bike to the LEFT, positive roll (about axis 1) leans it to the
RIGHT, positive pitch (about axis 3) lifts the front, positive steer
(about the steering axis e2) steers to the LEFT.  Wheels that roll forward
have NEGATIVE spin rate about axis 3.

All matrices here are direction-cosine matrices ``R_PC`` whose COLUMNS are
the child unit vectors expressed in the parent basis, i.e.

    v_parent = R_PC @ v_child

The rows printed in Table 4.4 are the child vectors in the parent basis, so
``R_PC`` is the transpose of the table entry.

Every function takes the maths namespace ``m`` (``numpy`` or ``sympy``) so
the same code serves the numeric and the symbolic model.
"""
from __future__ import annotations

import numpy as np


def _mat(m, rows):
    if m is np:
        return np.array(rows, dtype=float)
    return m.Matrix(rows)


def Rx(a, m=np):
    """Rotation about axis 1 (roll)."""
    c, s = m.cos(a), m.sin(a)
    return _mat(m, [[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a, m=np):
    """Rotation about axis 2 (yaw / steer)."""
    c, s = m.cos(a), m.sin(a)
    return _mat(m, [[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a, m=np):
    """Rotation about axis 3 (pitch / swing-arm)."""
    c, s = m.cos(a), m.sin(a)
    return _mat(m, [[c, -s, 0], [s, c, 0], [0, 0, 1]])


def chain(q9, q10, q11, q2, mu, eps, m=np):
    """Rotation matrices of all frames w.r.t. the inertial frame N.

    Returns a dict with keys M, D, A, B, E, F (Table 4.4, exact chain):

        N --yaw q10--> M --roll q9--> D --pitch mu--> A
        A --q2--> B          (swing-arm)
        A --eps--> E --steer q11--> F   (e2 = f2 = steering axis)
    """
    R = {}
    R["M"] = Ry(q10, m)
    R["D"] = R["M"] @ Rx(q9, m)
    R["A"] = R["D"] @ Rz(mu, m)
    R["B"] = R["A"] @ Rz(q2, m)
    R["E"] = R["A"] @ Rz(eps, m)
    R["F"] = R["E"] @ Ry(q11, m)
    return R


# --------------------------------------------------------------------------
# Front wheel frames G and H (Sections 4.4.2 and 4.4.3) -- numeric, exact
# --------------------------------------------------------------------------
def front_wheel_frames(R_F, R_M):
    """Exact G and H frames of the front wheel.

    G : F rotated about the axle f3 until g1 lies in the ground plane
        (g3 = f3, g1 horizontal, g2 = 'up' direction inside the wheel plane).
    H : G rotated about g1 until h3 lies in the ground plane
        (h1 = g1, h2 = vertical, h3 = horizontal axle projection).

    Returns R_G, R_H (columns = unit vectors in N) and the angles
    xi, gamma1 (front camber) and delta' (kinematic steering angle).
    """
    f1, f2, f3 = R_F[:, 0], R_F[:, 1], R_F[:, 2]
    m1, m2, m3 = R_M[:, 0], R_M[:, 1], R_M[:, 2]
    g1 = np.cross(m2, f3)
    g1 /= np.linalg.norm(g1)
    g3 = f3
    g2 = np.cross(g3, g1)
    R_G = np.column_stack([g1, g2, g3])
    h1, h2 = g1, m2
    h3 = np.cross(h1, h2)
    R_H = np.column_stack([h1, h2, h3])
    # xi: rotation about f3 taking f1 to g1  (g1 = c_xi f1 - s_xi f2)
    xi = np.arctan2(-g1 @ f2, g1 @ f1)
    # camber: positive when the wheel leans to the right (axle tip f3 dips)
    gamma1 = np.arcsin(np.clip(-f3 @ m2, -1.0, 1.0))
    # kinematic steering angle: heading of g1 w.r.t. m1, positive to the left
    delta_k = np.arctan2(-g1 @ m3, g1 @ m1)
    return R_G, R_H, xi, gamma1, delta_k


def camber_thesis(q9, q11, eps, mu):
    """Front camber, thesis eq. below (4.1)."""
    return np.arcsin(np.cos(q11) * np.sin(q9)
                     - np.cos(q9) * np.sin(q11) * np.sin(eps + mu))


def kin_steer_thesis(q9, q11, eps, mu):
    """Kinematic steering angle delta', thesis Section 4.4.3."""
    return np.arctan2(np.sin(q11) * np.cos(eps + mu),
                      np.cos(q9) * np.cos(q11)
                      + np.sin(q9) * np.sin(q11) * np.sin(eps + mu))
