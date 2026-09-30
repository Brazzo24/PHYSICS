"""
orthomount.py
=============

Rigid-body-on-mounts vibration analysis for motorcycle engine mounting, with a
full implementation of the "Orthogonal Engine Mount System" theory of

    T. Furusawa et al. (Yamaha), SAE 830228 (1983),
    "Orthogonal Engine Mount System"

The library has three layers, deliberately kept separate:

1. ``rigidbody`` layer  -- completely general.  Build the 6x6 mass and stiffness
   matrices of a rigid body (the engine) supported on an arbitrary set of
   linear mounts at arbitrary positions/orientations.  No symmetry assumed.

2. ``paper`` layer      -- the closed-form 2-DOF (roll-yaw) algebra of the
   paper, Eqs. (16)-(32).  Useful for *design*: it tells you directly what
   stiffness-axis inclination and stiffness ratio you need.

3. ``engine`` layer     -- slider-crank inertia forces and couples, so that the
   excitation vector fed into layers 1/2 comes from real crank geometry rather
   than being assumed.

--------------------------------------------------------------------------
COORDINATE AND SIGN CONVENTIONS  (read this once, it saves a lot of pain)
--------------------------------------------------------------------------

Global frame, following the paper's Fig. 1:

    x : longitudinal, positive FORWARD
    y : lateral
    z : vertical, positive UP

and the frame is right-handed, so y points to the rider's LEFT.

Generalised coordinates, in the paper's order::

    q = [x, y, z, phi, theta, psi]

with ``phi`` = rotation about x (roll), ``theta`` = about y (pitch),
``psi`` = about z (yaw).

Tilt angles (``alpha`` for the principal inertia axis, ``beta`` for the
principal elasticity axis, ``delta`` for the resulting axis of vibration) are
measured in the x-z plane, positive in the right-handed sense about +y.  This
is the convention that reproduces the paper's Eqs. (20) and (21) verbatim; see
:func:`rot_y_paper` for the one caveat about which way "positive" points on the
machine.  Only the *relative* geometry of alpha, beta and delta matters
physically, but a sign slip flips ``R``, ``I_zx``, ``K_phipsi`` and ``delta``
together and is the most common way to mis-apply the paper.

--------------------------------------------------------------------------
Units: SI throughout (m, kg, N/m, N*m/rad, rad, rad/s).  Frequencies returned
in Hz where the function name says ``_hz``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "rot_y_paper",
    "rot_from_euler_zyx",
    "skew",
    "inertia_tensor_from_principal",
    "stiffness_block_from_principal",
    "Mount",
    "RigidBody",
    "MountSystem",
    "ModalResult",
    "R_from_inertia",
    "R_from_stiffness",
    "solve_beta_for_R",
    "omega1_paper",
    "delta_paper",
    "modal_mass_paper",
    "Cylinder",
    "Engine",
    "ExcitationSpec",
]


# ===========================================================================
#  Small geometry helpers
# ===========================================================================

def skew(v: Sequence[float]) -> np.ndarray:
    """Return the skew-symmetric matrix S(v) such that S(v) @ w == cross(v, w)."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def rot_y_paper(angle: float) -> np.ndarray:
    """Rotation about +y, right-handed -- the paper's tilt convention.

    ``angle`` > 0 rotates an axis that started along +x toward -z.  With the
    frame as drawn (x forward, z up) that is "nose DOWN toward the front"; if
    you prefer to read positive as "nose up toward the front", flip the sign of
    every tilt angle, or equivalently take z positive downward.  Nothing in the
    theory depends on which you pick -- only the *relative* geometry of alpha,
    beta and delta matters -- but you must be consistent, because the sign
    convention decides the sign of ``R``, ``I_zx``, ``K_phipsi`` and ``delta``.

    This choice is the one that reproduces the paper's Eqs. (20) and (21)
    verbatim, which is why it is used here.

    >>> np.allclose(rot_y_paper(np.deg2rad(30)) @ [1, 0, 0],
    ...             [np.cos(np.deg2rad(30)), 0, -np.sin(np.deg2rad(30))])
    True
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def rot_from_euler_zyx(rz: float = 0.0, ry: float = 0.0, rx: float = 0.0) -> np.ndarray:
    """Standard right-handed rotation matrix, R = Rz(rz) Ry(ry) Rx(rx).

    Used for mount orientation, where no special convention is needed.
    """
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    Ry = np.array([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
    Rx = np.array([[1.0, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def inertia_tensor_from_principal(I_xi: float, I_eta: float, I_zeta: float,
                                  alpha: float) -> np.ndarray:
    """Inertia tensor about the CG for a body whose principal axes (xi, eta, zeta)
    are tilted by ``alpha`` (paper sense, nose-up-forward positive) about y.

    Reproduces the paper's Eq. (20)::

        I_x  = I_xi cos^2 a + I_zeta sin^2 a
        I_z  = I_xi sin^2 a + I_zeta cos^2 a
        I_zx = (I_xi - I_zeta) sin a cos a

    where the returned *tensor* has ``J[0, 2] = -I_zx`` (note the sign -- the
    paper's mass matrix carries ``-I_zx`` in the phi-psi off-diagonal).
    """
    R = rot_y_paper(alpha)
    J_principal = np.diag([I_xi, I_eta, I_zeta])
    return R @ J_principal @ R.T


def stiffness_block_from_principal(K_xi: float, K_eta: float, K_zeta: float,
                                   beta: float) -> np.ndarray:
    """Rotational stiffness 3x3 block for principal elastic axes tilted by
    ``beta`` (paper sense) about y.  Reproduces Eq. (21)::

        K_phiphi = K_xi cos^2 b + K_zeta sin^2 b
        K_psipsi = K_xi sin^2 b + K_zeta cos^2 b
        K_phipsi = -(K_xi - K_zeta) sin b cos b
    """
    R = rot_y_paper(beta)
    return R @ np.diag([K_xi, K_eta, K_zeta]) @ R.T


# ===========================================================================
#  Layer 1: general rigid body on mounts
# ===========================================================================

@dataclass
class Mount:
    """A linear elastic mount idealised as three orthogonal springs.

    Parameters
    ----------
    position : (3,) array
        Mount location relative to the body CG, in the global frame [m].
    stiffness : (3,) array
        Principal rates of the mount in its *own* axes [N/m], typically
        (radial, radial, axial) for a cylindrical rubber bush.
    orientation : (3, 3) array
        Rotation from mount-local axes to global axes.  Identity means the
        mount's principal directions are x, y, z.
    loss_factor : float or (3,) array
        Structural (hysteretic) damping loss factor eta.  The complex stiffness
        used in FRF calculations is k*(1 + i*eta).  Rubber engine mounts are
        typically eta = 0.1 ... 0.3.
    name : str
    """

    position: np.ndarray
    stiffness: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    loss_factor: float | np.ndarray = 0.15
    name: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.stiffness = np.asarray(self.stiffness, dtype=float).reshape(3)
        self.orientation = np.asarray(self.orientation, dtype=float).reshape(3, 3)

    # -- derived -----------------------------------------------------------
    @property
    def k_global(self) -> np.ndarray:
        """3x3 translational stiffness of this mount in global axes."""
        R = self.orientation
        return R @ np.diag(self.stiffness) @ R.T

    @property
    def k_global_complex(self) -> np.ndarray:
        """3x3 complex stiffness k*(1 + i eta) in global axes."""
        R = self.orientation
        eta = np.broadcast_to(np.asarray(self.loss_factor, dtype=float), (3,))
        kc = np.diag(self.stiffness * (1.0 + 1j * eta))
        return R @ kc @ R.T.astype(complex)

    def K6(self, complex_stiffness: bool = False) -> np.ndarray:
        """This mount's contribution to the 6x6 stiffness matrix.

        For a body displacement ``q = [u, theta]`` the mount point moves by
        ``d = u + theta x r = [I, -S(r)] q``; the generalised force it applies
        back on the body is ``-[I; S(r)] k d``.  Hence::

            K = [[      k    ,   -k S(r)   ],
                 [   S(r) k  , -S(r) k S(r)]]
        """
        k = self.k_global_complex if complex_stiffness else self.k_global
        S = skew(self.position)
        K = np.zeros((6, 6), dtype=k.dtype)
        K[:3, :3] = k
        K[:3, 3:] = -k @ S
        K[3:, :3] = S @ k
        K[3:, 3:] = -S @ k @ S
        return K

    def recovery(self) -> np.ndarray:
        """3x6 matrix B such that mount deflection = B @ q."""
        return np.hstack([np.eye(3), -skew(self.position)])


@dataclass
class RigidBody:
    """Engine (or any rigid body) described about its own CG."""

    mass: float
    inertia: np.ndarray  # 3x3 tensor about CG, global axes

    def __post_init__(self) -> None:
        self.inertia = np.asarray(self.inertia, dtype=float).reshape(3, 3)

    @classmethod
    def from_principal(cls, mass: float, I_xi: float, I_eta: float,
                       I_zeta: float, alpha: float) -> "RigidBody":
        """Build from principal inertias and the paper's tilt angle alpha."""
        return cls(mass, inertia_tensor_from_principal(I_xi, I_eta, I_zeta, alpha))

    @property
    def M6(self) -> np.ndarray:
        M = np.zeros((6, 6))
        M[:3, :3] = self.mass * np.eye(3)
        M[3:, 3:] = self.inertia
        return M

    @property
    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """(principal inertias sorted ascending, corresponding axes as columns)."""
        w, V = np.linalg.eigh(self.inertia)
        return w, V

    @property
    def alpha_paper(self) -> float:
        """Inclination of the in-plane principal axis, in the paper's sense [rad].

        Only meaningful when y is a principal axis (the usual assumption).
        Returns the tilt of the principal axis that is closest to x -- that is
        the one the paper calls xi.  Note the inherent ambiguity: (I_xi, alpha)
        and (I_zeta, alpha + 90 deg) describe the same tensor, so "closest to x"
        is what pins it down.
        """
        J = self.inertia
        sub = np.array([[J[0, 0], J[0, 2]], [J[2, 0], J[2, 2]]])
        _, V = np.linalg.eigh(sub)
        # pick the eigenvector most aligned with x
        v = V[:, int(np.argmax(np.abs(V[0, :])))]
        if v[0] < 0:
            v = -v
        # R_y(alpha) @ [1,0,0] = [cos a, 0, -sin a]
        return float(np.arctan2(-v[1], v[0]))

    @property
    def principal_inplane(self) -> tuple[float, float, float]:
        """(I_xi, I_zeta, alpha) for the x-z principal pair, paper convention."""
        a = self.alpha_paper
        J = self.inertia
        c, s = np.cos(a), np.sin(a)
        I_xi = c * c * J[0, 0] - 2 * c * s * J[0, 2] + s * s * J[2, 2]
        I_zeta = s * s * J[0, 0] + 2 * c * s * J[0, 2] + c * c * J[2, 2]
        return float(I_xi), float(I_zeta), float(a)


@dataclass
class ModalResult:
    """Undamped normal-mode solution of a MountSystem."""

    omega: np.ndarray          # (n,) natural frequencies [rad/s]
    modes: np.ndarray          # (6, n) eigenvectors, mass-normalised is NOT assumed
    modal_mass: np.ndarray     # (n,) m_r = u_r^T M u_r
    labels: list[str]

    @property
    def f_hz(self) -> np.ndarray:
        return self.omega / (2.0 * np.pi)

    def participation(self, f: np.ndarray) -> np.ndarray:
        """Modal participation factors ``u_r^T f`` for excitation vector ``f``.

        This is the heart of the paper.  A mode with ``u_r^T f == 0`` is
        *orthogonal to the excitation* and simply does not respond, no matter
        how close the forcing frequency comes to its natural frequency.
        """
        f = np.asarray(f, dtype=float).reshape(6)
        return self.modes.T @ f

    def normalised_participation(self, f: np.ndarray) -> np.ndarray:
        """Participation normalised so the values are comparable between modes.

        ``|u_r^T f| / (||u_r|| * ||f||)`` -- the cosine of the angle between the
        excitation vector and the mode shape, in the plain Euclidean sense.
        1.0 means perfectly aligned, 0.0 means orthogonal.
        """
        f = np.asarray(f, dtype=float).reshape(6)
        p = self.modes.T @ f
        n = np.linalg.norm(self.modes, axis=0) * np.linalg.norm(f)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.abs(p) / n


DOF_LABELS = ["x (fore/aft)", "y (lateral)", "z (bounce)",
              "phi (roll)", "theta (pitch)", "psi (yaw)"]


class MountSystem:
    """A rigid body on a set of mounts, grounded to an infinitely stiff chassis.

    This is the paper's "six-degree of freedom system model": the chassis is
    assumed rigid and of infinite mass, so the mounts connect the engine
    straight to ground.  (Section "MULTI-DEGREE OF FREEDOM SYSTEM MODEL" of the
    paper relaxes that; here we keep the assumption, and the paper's own
    conclusion was that the 35 Hz roll mode barely moved when they relaxed it,
    because the chassis inertia is much larger than the engine's.)
    """

    def __init__(self, body: RigidBody, mounts: Sequence[Mount]):
        self.body = body
        self.mounts = list(mounts)

    # -- matrices ----------------------------------------------------------
    @property
    def M(self) -> np.ndarray:
        return self.body.M6

    @property
    def K(self) -> np.ndarray:
        K = np.zeros((6, 6))
        for m in self.mounts:
            K += m.K6()
        return 0.5 * (K + K.T)  # symmetrise against round-off

    def K_complex(self) -> np.ndarray:
        K = np.zeros((6, 6), dtype=complex)
        for m in self.mounts:
            K += m.K6(complex_stiffness=True)
        return 0.5 * (K + K.T)

    @property
    def elastic_centre(self) -> np.ndarray:
        """Point at which a force produces no rotation (the 'centre of elasticity').

        The paper assumes this coincides with the CG.  For a real layout it
        usually does not, and the offset is exactly what creates the
        translation-rotation coupling blocks in K.  Returns the position
        relative to the CG [m].  ``nan`` if the translational block is singular.
        """
        K = self.K
        Kt, Kc = K[:3, :3], K[:3, 3:]
        # Shifting the origin by r_e changes the coupling block to
        #   Kc' = Kc + Kt S(r_e).   Setting Kc' = 0 column by column gives
        #   Kt S(e_i) r_e = Kc[:, i]  for i = 0, 1, 2.
        try:
            A = np.zeros((9, 3))
            b = np.zeros(9)
            for i in range(3):
                e = np.zeros(3)
                e[i] = 1.0
                A[3 * i:3 * i + 3, :] = Kt @ skew(e)
                b[3 * i:3 * i + 3] = Kc[:, i]
            r, *_ = np.linalg.lstsq(A, b, rcond=None)
            return r
        except np.linalg.LinAlgError:
            return np.full(3, np.nan)

    # -- eigen -------------------------------------------------------------
    def modal(self) -> ModalResult:
        """Undamped natural frequencies and normal modes (paper Eq. 3)."""
        from scipy.linalg import eigh

        w2, V = eigh(self.K, self.M)
        w2 = np.clip(w2, 0.0, None)
        omega = np.sqrt(w2)
        order = np.argsort(omega)
        omega, V = omega[order], V[:, order]
        # eigh with a B matrix returns V^T M V = I, so modal mass is 1.
        # Rescale so the largest component of each mode is +1: easier to read.
        for r in range(V.shape[1]):
            j = np.argmax(np.abs(V[:, r]))
            V[:, r] = V[:, r] / V[j, r]
        mr = np.array([V[:, r] @ self.M @ V[:, r] for r in range(V.shape[1])])
        labels = [self.describe_mode(V[:, r]) for r in range(V.shape[1])]
        return ModalResult(omega, V, mr, labels)

    @staticmethod
    def describe_mode(u: np.ndarray, char_length: float = 0.25) -> str:
        """Crude but useful mode label: which DOF dominates.

        Rotations are scaled by ``char_length`` [m] so that translations and
        rotations can be compared on a like-for-like basis.
        """
        w = np.abs(u.copy())
        w[3:] *= char_length
        return DOF_LABELS[int(np.argmax(w))]

    # -- forced response ---------------------------------------------------
    def receptance(self, omega: np.ndarray, damped: bool = True) -> np.ndarray:
        """Direct frequency response ``H(w) = (-w^2 M + K)^-1``.

        Returns array of shape (len(omega), 6, 6).  With ``damped=True`` the
        complex (hysteretic) stiffness of each mount is used.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        M = self.M
        K = self.K_complex() if damped else self.K.astype(complex)
        H = np.empty((omega.size, 6, 6), dtype=complex)
        for i, w in enumerate(omega):
            H[i] = np.linalg.inv(-w ** 2 * M + K)
        return H

    def response(self, omega: np.ndarray, f: np.ndarray,
                 damped: bool = True) -> np.ndarray:
        """Steady-state response amplitude vector Q(w) for a fixed force vector.

        ``f`` may be (6,) for a frequency-independent force, or (len(omega), 6)
        for one that grows with speed -- which is what inertia forces do.
        """
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        H = self.receptance(omega, damped=damped)
        f = np.asarray(f)
        if f.ndim == 1:
            f = np.broadcast_to(f, (omega.size, 6))
        return np.einsum("wij,wj->wi", H, f.astype(complex))

    def modal_response(self, omega: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Undamped modal-superposition response, the paper's Eq. (6)::

            Q = sum_r  u_r (u_r^T F) / (m_r (w_r^2 - w^2))

        Returned as (len(omega), 6, n_modes) so you can see each mode's share.
        """
        res = self.modal()
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        f = np.asarray(f, dtype=float).reshape(6)
        p = res.participation(f)                       # (n,)
        denom = res.modal_mass[None, :] * (res.omega[None, :] ** 2 - omega[:, None] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            qr = p[None, :] / denom                    # (w, n)
        return res.modes[None, :, :] * qr[:, None, :]

    def transmitted_force(self, omega: np.ndarray, f: np.ndarray,
                          damped: bool = True) -> np.ndarray:
        """Total force transmitted into the chassis, (len(omega), 3) complex.

        This is what the rider actually feels -- not the engine's own motion.
        """
        Q = self.response(omega, f, damped=damped)
        tot = np.zeros((Q.shape[0], 3), dtype=complex)
        for m in self.mounts:
            k = m.k_global_complex if damped else m.k_global.astype(complex)
            B = m.recovery().astype(complex)
            tot += (k @ (B @ Q.T)).T
        return tot


# ===========================================================================
#  Layer 2: the paper's closed-form design algebra
# ===========================================================================
#
#  Assumptions behind everything in this section (paper, p.2):
#    * engine rigid, small motions
#    * mounts massless linear springs
#    * chassis rigid, infinite mass
#    * y is a principal axis of BOTH inertia and elasticity
#    * the centre of elasticity coincides with the CG
#    * the only excitation is a moment about x
#
#  Under those assumptions the 6-DOF problem splits into
#    (x, z, theta) : untouched by the excitation
#    (y)           : untouched
#    (phi, psi)    : the 2-DOF system that carries all the action.
#
#  With  p = I_xi / I_zeta  and  q = K_xi / K_zeta  the paper's key result is
#  that the "orthogonality condition" -- excitation orthogonal to mode 2 --
#  holds if and only if
#
#      R = I_zx / I_z = -K_phipsi / K_psipsi                        (Eq. 16)
#
#  which in terms of the tilt angles is Eq. (22):
#
#      (p - 1) tan(alpha)            (q - 1) tan(beta)
#      -------------------  =  R  =  ------------------
#      p tan^2(alpha) + 1            q tan^2(beta) + 1
# ===========================================================================


def R_from_inertia(p: float, alpha: float) -> float:
    """R = I_zx / I_z from the inertia ratio p = I_xi/I_zeta and tilt alpha.

    Paper Eq. (22), left-hand side.
    """
    t = np.tan(alpha)
    return (p - 1.0) * t / (p * t ** 2 + 1.0)


def R_from_stiffness(q: float, beta: float) -> float:
    """R = -K_phipsi / K_psipsi from q = K_xi/K_zeta and tilt beta.

    Paper Eq. (22), right-hand side -- identical in form to the inertia side.
    """
    t = np.tan(beta)
    return (q - 1.0) * t / (q * t ** 2 + 1.0)


def solve_beta_for_R(R: float, q: float) -> tuple[float, float]:
    """Invert Eq. (22): find the elastic-axis tilt beta that gives a required R.

    ``R q tan^2(b) - (q - 1) tan(b) + R = 0`` is a quadratic in tan(beta), so
    there are generally two solutions.  Both are returned (radians), smallest
    magnitude first.  Returns ``(nan, nan)`` if no real solution exists, which
    happens when ``q`` is too close to 1 to generate the required coupling::

        |R| <= (q - 1) / (2 sqrt(q))            for q > 1

    That inequality is the practical design limit: it says how much stiffness
    anisotropy you need before an inclined elastic axis can do the job at all.
    """
    if np.isclose(R, 0.0):
        return 0.0, np.nan
    a, b, c = R * q, -(q - 1.0), R
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return np.nan, np.nan
    roots = np.array([(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)])
    betas = np.arctan(roots)
    order = np.argsort(np.abs(betas))
    return float(betas[order[0]]), float(betas[order[1]])


def q_min_for_R(R: float) -> float:
    """Smallest stiffness ratio q > 1 that can realise a coupling R.

    From ``|R| = (q-1)/(2 sqrt(q))`` at the tangency point.
    """
    a = abs(R)
    return float((a * 2 + np.sqrt(4 * a ** 2 + 4)) ** 2 / 4.0) if a else 1.0


def modal_mass_paper(I_xi: float, I_zeta: float, alpha: float) -> float:
    """Modal mass (modal inertia) of the surviving mode, Eq. (24)-(25)::

        m1 = (I_x I_z - I_zx^2) / I_z = I_xi I_zeta / I_z
           = I_xi / (p sin^2 a + cos^2 a)
    """
    p = I_xi / I_zeta
    return I_xi / (p * np.sin(alpha) ** 2 + np.cos(alpha) ** 2)


def omega1_paper(K_xi: float, I_xi: float, p: float, q: float,
                 alpha: float, beta: float) -> float:
    """Resonant frequency of the surviving mode [rad/s], Eq. (27)::

        w1^2 = K_xi (p sin^2 a + cos^2 a) / (I_xi (q sin^2 b + cos^2 b))

    Note what this says: once alpha and beta are fixed by the orthogonality
    condition, the resonance is set by a single number, K_xi.  That is the
    design knob you use to place the mode below the usable rev range.
    """
    num = K_xi * (p * np.sin(alpha) ** 2 + np.cos(alpha) ** 2)
    den = I_xi * (q * np.sin(beta) ** 2 + np.cos(beta) ** 2)
    return float(np.sqrt(num / den))


def K_xi_for_target_f1(f1_hz: float, I_xi: float, p: float, q: float,
                       alpha: float, beta: float) -> float:
    """Invert Eq. (27): the principal rotational rate needed for a target f1."""
    w1 = 2 * np.pi * f1_hz
    return float(w1 ** 2 * I_xi * (q * np.sin(beta) ** 2 + np.cos(beta) ** 2)
                 / (p * np.sin(alpha) ** 2 + np.cos(alpha) ** 2))


def delta_paper(p: float, alpha: float) -> float:
    """Inclination of the resulting axis of rotational vibration [rad], Eq. (29)::

        tan(delta) = (1 - p) tan(a) / (p tan^2 a + 1) = -R

    The engine rocks about *this* axis, not about x.  Useful sanity checks,
    Eqs. (31) and (32)::

        tan(alpha - delta) = p tan(alpha)
        tan(beta  - delta) = q tan(beta)
    """
    return float(np.arctan(-R_from_inertia(p, alpha)))


# ===========================================================================
#  Layer 3: engine excitation
# ===========================================================================

@dataclass
class Cylinder:
    """One cylinder of a reciprocating engine.

    Parameters
    ----------
    crank_phase : float
        Crank throw angle relative to cylinder 1 [rad].  180 deg twin -> [0, pi].
    y : float
        Position of the cylinder along the crankshaft axis, relative to the
        engine CG [m].  This is the lever arm that turns reciprocating forces
        into a rocking couple.
    axis_tilt : float
        Cylinder axis inclination [rad], paper sense: 0 = vertical (along +z),
        positive = leaning FORWARD (toward +x).  The RZ250's cylinders lean
        24 deg forward.
    m_recip : float
        Reciprocating mass: piston + rings + pin + small-end share of the rod [kg].
    m_rot : float
        Rotating mass at crank radius: big end + crankpin share [kg].
    """

    crank_phase: float
    y: float
    axis_tilt: float = 0.0
    m_recip: float = 0.0
    m_rot: float = 0.0

    @property
    def axis(self) -> np.ndarray:
        """Unit vector along the cylinder axis, pointing away from the crank."""
        return np.array([np.sin(self.axis_tilt), 0.0, np.cos(self.axis_tilt)])


@dataclass
class Engine:
    """Reciprocating engine inertia excitation.

    The classical slider-crank result for one cylinder, with crank radius r,
    rod length l, ratio lam = r/l, crank angle th::

        a_piston  ~=  r w^2 ( cos(th) + lam cos(2 th) + ... )

    so the force the engine structure must react is

        F(th) = m_recip r w^2 ( cos th + lam cos 2th )   along the cylinder axis
              + m_rot   r w^2 ( rotating vector )        radial

    ``balance_factor`` is the usual crankshaft balance factor: the fraction of
    the reciprocating mass that the counterweights also carry.  The
    counterweight generates a *rotating* force, so balancing the primary force
    along the cylinder axis necessarily creates an equal one perpendicular to
    it.  That trade-off is exactly what reference (1) of the paper is about,
    and it is why a two-cylinder engine cannot simply be balanced into silence.
    """

    cylinders: list[Cylinder]
    crank_radius: float          # r = stroke / 2 [m]
    rod_length: float            # l [m]
    balance_factor: float = 0.5  # fraction of m_recip carried by counterweights

    @property
    def lam(self) -> float:
        return self.crank_radius / self.rod_length

    # -- time-domain -------------------------------------------------------
    def forces_at(self, theta: float, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """Resultant inertia force [N] and couple about the CG [N*m] at crank
        angle ``theta`` and speed ``omega`` [rad/s].

        Both are returned as 3-vectors in the global frame.
        """
        r, lam = self.crank_radius, self.lam
        F = np.zeros(3)
        Mo = np.zeros(3)
        for cyl in self.cylinders:
            th = theta + cyl.crank_phase
            e = cyl.axis
            # crank pin direction: along the cylinder axis at th = 0
            e_perp = np.array([np.cos(cyl.axis_tilt), 0.0, -np.sin(cyl.axis_tilt)])

            # reciprocating: primary + secondary, along the cylinder axis
            f_recip = cyl.m_recip * r * omega ** 2 * (np.cos(th) + lam * np.cos(2 * th))
            # counterweight: rotating vector opposing the crank pin
            m_cw = self.balance_factor * cyl.m_recip + cyl.m_rot
            f_cw = -m_cw * r * omega ** 2 * (np.cos(th) * e + np.sin(th) * e_perp)
            # rotating unbalance of the crank throw itself
            f_rot = cyl.m_rot * r * omega ** 2 * (np.cos(th) * e + np.sin(th) * e_perp)

            f_total = f_recip * e + f_cw + f_rot
            F += f_total
            Mo += np.cross(np.array([0.0, cyl.y, 0.0]), f_total)
        return F, Mo

    def sweep_cycle(self, omega: float, n: int = 720) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(theta, F(3,n), M(3,n)) over one crank revolution."""
        th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        F = np.zeros((3, n))
        Mo = np.zeros((3, n))
        for i, t in enumerate(th):
            F[:, i], Mo[:, i] = self.forces_at(t, omega)
        return th, F, Mo

    # -- harmonic decomposition -------------------------------------------
    def harmonics(self, omega: float, n_harm: int = 4,
                  n: int = 720) -> dict[int, np.ndarray]:
        """Complex amplitude of each order of the 6-component excitation vector.

        Returns ``{order: complex (6,) vector}`` where order 1 is once per crank
        revolution ("primary"), order 2 is "secondary", and the 6 components are
        ``[Fx, Fy, Fz, Mx, My, Mz]``.  Amplitudes are peak values.
        """
        th, F, Mo = self.sweep_cycle(omega, n=n)
        g = np.vstack([F, Mo])                       # (6, n)
        out = {}
        for k in range(1, n_harm + 1):
            c = 2.0 / n * np.sum(g * np.exp(-1j * k * th)[None, :], axis=1)
            out[k] = c
        return out

    def excitation_vector(self, omega: float, order: int = 1) -> np.ndarray:
        """Real-valued peak excitation vector of the given order.

        The vector's *direction* is the physically meaningful part: it is the
        thing the mode shapes have to be orthogonal to.
        """
        h = self.harmonics(omega, n_harm=max(order, 1))[order]
        # take the amplitude, preserving relative sign via the dominant phase
        j = np.argmax(np.abs(h))
        phase = np.exp(-1j * np.angle(h[j]))
        return np.real(h * phase)


@dataclass
class ExcitationSpec:
    """Convenience wrapper: an excitation direction plus how it scales with rpm.

    ``direction`` is a fixed unit 6-vector; ``scale(omega)`` multiplies it.
    Inertia excitation scales with omega^2, which is why a resonance at low rpm
    can be tolerable even if the isolation there is poor.
    """

    direction: np.ndarray
    exponent: float = 2.0
    reference_omega: float = 1.0
    reference_magnitude: float = 1.0

    def at(self, omega: np.ndarray) -> np.ndarray:
        omega = np.atleast_1d(np.asarray(omega, dtype=float))
        mag = self.reference_magnitude * (omega / self.reference_omega) ** self.exponent
        d = np.asarray(self.direction, dtype=float).reshape(6)
        return mag[:, None] * d[None, :]


# ===========================================================================
#  Convenience: build the paper's idealised mount system directly
# ===========================================================================

def paper_system(mass: float, I_xi: float, I_eta: float, I_zeta: float, alpha: float,
                 K_xi: float, K_eta: float, K_zeta: float, beta: float,
                 K_trans: Sequence[float] = (1e7, 1e7, 1e7)) -> tuple[np.ndarray, np.ndarray]:
    """M and K exactly as the paper writes them (elastic centre at the CG).

    Returns ``(M, K)`` 6x6.  Translational rates are decoupled from rotation by
    construction -- that is the paper's "centre of elasticity coincides with the
    centre of gravity" assumption.
    """
    M = np.zeros((6, 6))
    M[:3, :3] = mass * np.eye(3)
    M[3:, 3:] = inertia_tensor_from_principal(I_xi, I_eta, I_zeta, alpha)

    K = np.zeros((6, 6))
    K[:3, :3] = np.diag(K_trans)
    K[3:, 3:] = stiffness_block_from_principal(K_xi, K_eta, K_zeta, beta)
    return M, K


def modal_from_matrices(M: np.ndarray, K: np.ndarray) -> ModalResult:
    """Eigen-solve arbitrary (M, K) with the same post-processing as MountSystem."""
    from scipy.linalg import eigh

    w2, V = eigh(K, M)
    w2 = np.clip(w2, 0.0, None)
    omega = np.sqrt(w2)
    order = np.argsort(omega)
    omega, V = omega[order], V[:, order]
    for r in range(V.shape[1]):
        j = np.argmax(np.abs(V[:, r]))
        V[:, r] = V[:, r] / V[j, r]
    mr = np.array([V[:, r] @ M @ V[:, r] for r in range(V.shape[1])])
    labels = [MountSystem.describe_mode(V[:, r]) for r in range(V.shape[1])]
    return ModalResult(omega, V, mr, labels)
