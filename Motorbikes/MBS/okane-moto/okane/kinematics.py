"""Numeric kinematics of the motorcycle (thesis Sections 4.4 and 4.8).

Two descriptions of the main-frame pitch angle mu = q14 are available:

``exact``   mu solves the vector loop (4.12) exactly: the lowest point of the
            front tyre is q3 below the ground plane.  Coefficients
            H_r = d mu / d q_r follow from the implicit function theorem.
``thesis``  the closed form (4.17) of the thesis (gamma1 ~ q9, sin mu ~ mu)
            with H_r from its exact gradient.

The class also evaluates the remaining closed forms of Section 4.8 (q13..q19)
so that the thesis' approximations can be compared with the exact geometry.
"""
from __future__ import annotations

import numpy as np

from . import frames
from .params import BikeParams

try:
    from . import _generated as gen
except ImportError:                                  # first use: derive
    from . import symbolic
    symbolic.generate(verbose=False)
    from . import _generated as gen

NQ = 11
IDX = {f"q{i}": i - 1 for i in range(1, 12)}


def make_q(**kw) -> np.ndarray:
    """Build a co-ordinate vector, e.g. ``make_q(q1=0.08, q9=0.3)``."""
    q = np.zeros(NQ)
    for k, v in kw.items():
        q[IDX[k]] = v
    return q


class Kinematics:
    def __init__(self, params: BikeParams, pitch_model: str = "exact"):
        assert pitch_model in ("exact", "thesis")
        self.p = params
        self.g = params.geom()
        self.pitch_model = pitch_model

    # ------------------------------------------------------------------
    # pitch angle mu = q14
    # ------------------------------------------------------------------
    def mu_thesis(self, q):
        """Closed-form pitch angle, eq. (4.17)."""
        return gen.thesis_pose(q, self.g)["pose"][1]

    def mu_exact(self, q, tol=1e-13):
        """Exact pitch angle: Newton iteration on the loop-closure residual."""
        mu = float(self.mu_thesis(q))
        for _ in range(30):
            c = gen.closure(mu, q, self.g)
            step = c["g"] / c["g_mu"]
            mu -= step
            if abs(step) < tol:
                break
        return mu

    def pitch(self, q, u=None, model=None):
        """Return ``(mu, H, hb)``.

        H  : coefficients of eq. (4.18), mu_dot = H @ u   (H[r] = d mu/d q_r)
        hb : sum_r (dH_r/dt) u_r, the part of mu_ddot that does not contain
             u_dot.  The thesis treats the H_r as (scheduled) constants,
             i.e. hb = 0.  ``None`` if ``u`` is not given.
        """
        model = model or self.pitch_model
        if model == "exact":
            mu = self.mu_exact(q)
            c = gen.closure(mu, q, self.g)
            H = -c["g_q"] / c["g_mu"]
            hb = None
            if u is not None:
                md = H @ u
                hb = -(u @ c["g_qq"] @ u + 2 * md * (c["g_qmu"] @ u)
                       + c["g_mumu"] * md**2) / c["g_mu"]
        else:
            t = gen.thesis_pose(q, self.g)
            mu, H = t["pose"][1], t["jac"][1]
            hb = None
            if u is not None:          # directional derivative of H along u
                h = 1e-6
                Hp = gen.thesis_pose(q + h * u, self.g)["jac"][1]
                Hm = gen.thesis_pose(q - h * u, self.g)["jac"][1]
                hb = (Hp - Hm) @ u / (2 * h)
        return mu, H, hb

    # ------------------------------------------------------------------
    # thesis closed forms
    # ------------------------------------------------------------------
    def thesis_pose(self, q):
        """q13..q19 of Section 4.8 and their gradients (Taylor coefficients
        G, H, J, K, L of Table 4.6, plus wheelbase and trail)."""
        t = gen.thesis_pose(q, self.g)
        names = ["q13", "q14", "q15", "q16", "q17", "q18", "q19"]
        return dict(zip(names, t["pose"])), dict(zip(names, t["jac"]))

    def H_printed(self, q, frozen_profile=True):
        """Coefficients H, H1, H2, H3, H4, H9, H11 and W0 exactly as printed
        on thesis p. 73 (tyre contact co-ordinates T treated as constants).
        Returns a dict."""
        p = self.p
        q1, q2, q9, q11 = q[0], q[1], q[8], q[10]
        ce, se = np.cos(p.eps), np.sin(p.eps)
        c2, s2, c11, s11 = np.cos(q2), np.sin(q2), np.cos(q11), np.sin(q11)
        t9, sec9 = np.tan(q9), 1 / np.cos(q9)
        TFz = p.Rf - p.rhof + p.rhof * np.cos(q9)
        TFy = p.rhof * np.sin(q9)
        TRz = p.Rr - p.rhor + p.rhor * np.cos(q9)
        TRy = p.rhor * np.sin(q9)
        Ra, Px, Pz, Hx, Hz, Qx, Qz = p.Ra, p.Px, p.Pz, p.Hx, p.Hz, p.Qx, p.Qz
        W0 = c2 * Ra + ce * c11 * Qx + Hx + Px + se * (q1 - Qz)
        H = -(Ra * s2 + ce * (Qz - q1) + t9 * (TRy - TFy) - TFz + Hz + Pz
              + Qx * c11 * se + TRz) / W0
        H1 = (ce * (Ra * c2 + Hx + Px)
              + se * (Ra * s2 + t9 * (TRy - TFy) - TFz + Hz + Pz
                      + Qx * c11 * se + TRz) + ce**2 * Qx * c11) / W0**2
        H2 = Ra * (-Ra - s2 * (ce * (Qz - q1) - TFz + Hz + Pz + Qx * c11 * se + TRz)
                   - c2 * (ce * Qx * c11 + Hx + se * (q1 - Qz) + Px)
                   + t9 * (TFy - TRy) * s2) / W0**2
        H3, H4 = -sec9 / W0, sec9 / W0
        H9 = sec9**2 * (TFy - TRy) / W0
        H11 = Qx * s11 * (-ce * (Ra * s2 + t9 * (TRy - TFy) - TFz + Hz + Pz + TRz)
                          + se * (Ra * c2 + Hx + se * (q1 - Qz) + Px)
                          + ce**2 * (q1 - Qz)) / W0**2
        return dict(H=H, H1=H1, H2=H2, H3=H3, H4=H4, H9=H9, H11=H11, W0=W0)

    # ------------------------------------------------------------------
    # full configuration
    # ------------------------------------------------------------------
    def config(self, q, u=None, mu=None, H=None, hb=None, coef_rates=True):
        """Evaluate the complete configuration.

        Returns a dict with rotation matrices ``R_*``, positions ``r_*``,
        partial velocities ``Jv_*`` (3x11), partial angular velocities
        ``Jw_*`` (3x11), bias accelerations ``ab_*`` / ``alb_*`` and the
        contact geometry.  ``coef_rates=False`` reproduces the thesis'
        constant-coefficient assumption (hb = 0).
        """
        q = np.asarray(q, float)
        uu = np.zeros(NQ) if u is None else np.asarray(u, float)
        if mu is None:
            mu, H, hb = self.pitch(q, uu)
        if hb is None or not coef_rates:
            hb = 0.0
        mud = float(H @ uu)
        k = gen.kin(q, uu, mu, mud, hb, self.g)
        k.update(mu=mu, mud=mud, H=H, hb=hb, q=q, u=uu)
        from .symbolic import POINTS, BODIES
        for n in POINTS:
            k["Jv_" + n] = k["Jq_" + n] + np.outer(k["rmu_" + n], H)
            k["v_" + n] = k["Jv_" + n] @ uu
        for n in BODIES:
            k["Jw_" + n] = k["Jw_" + n] + np.outer(k["wmud_" + n], H)
        self._contacts(k)
        return k

    def _contacts(self, k):
        """Tyre contact points S (front) and T (rear) and the wheel frames."""
        p = self.p
        q = k["q"]
        m2 = k["R_M"][:, 1]
        R_G, R_H, xi, gam1, dk = frames.front_wheel_frames(k["R_F"], k["R_M"])
        k.update(R_G=R_G, R_H=R_H, xi=xi, gamma1=gam1, delta_k=dk)
        # lowest point of the toroidal tyre, shifted up by the deflection
        k["r_S"] = k["r_G"] - (p.Rf - p.rhof) * R_G[:, 1] - (p.rhof - q[2]) * m2
        k["r_T"] = k["r_D"] - (p.Rr - p.rhor) * k["R_D"][:, 1] - (p.rhor - q[3]) * m2
        # velocity of the wheel-fixed (material) point at the contact:
        #   v = v_axle + w_wheel x r      ->  J = Jv - skew(r) Jw
        for pt, ax, body in (("S", "G", "G"), ("T", "D", "D")):
            rr = k["r_" + pt] - k["r_" + ax]
            k["Jv_" + pt] = k["Jv_" + ax] - _skew(rr) @ k["Jw_" + body]
        # exact geometric outputs ------------------------------------------
        m1 = k["R_M"][:, 0]
        d = k["r_S"] - k["r_T"]
        k["wheelbase"] = float(np.hypot(d @ m1, d @ k["R_M"][:, 2]))
        # trail: steering axis / ground intersection ahead of S along h1
        e2 = k["R_F"][:, 1]
        s = -(k["r_H"] @ m2) / (e2 @ m2)
        axis_pt = k["r_H"] + s * e2
        k["trail"] = float((axis_pt - k["r_S"]) @ R_H[:, 0])
        # normal trail: distance of S behind the steering axis, in the wheel plane
        k["normal_trail"] = float(-(k["r_S"] - k["r_H"]) @ k["R_F"][:, 0])
        k["wheelbase_x"] = float(d @ m1)
        # heading of the frame plane relative to the line through the
        # contact points (the quantity the thesis calls q13, exact)
        k["q13_exact"] = float(np.arctan2(d @ k["R_M"][:, 2], d @ m1))
        return k

    # ------------------------------------------------------------------
    def mass_centre(self, k):
        p = self.p
        m = dict(A=p.MA, B=p.MB, C=p.MC, D=p.MD, F=p.MF, G=p.MG)
        return sum(m[b] * k["r_" + b] for b in m) / p.total_mass


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
