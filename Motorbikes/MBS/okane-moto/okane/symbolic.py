"""Symbolic kinematics of the motorcycle model (SymPy) and code generation.

This module plays the role Wolfram Mathematica plays in the thesis
(Section 4.10): it writes down the position of every key point as a function
of the generalised co-ordinates, differentiates symbolically and emits plain
Python functions (``okane/_generated.py``) for the numeric model.

Dependent co-ordinate
---------------------
The eleven generalised co-ordinates q1..q11 (Table 4.3) do not contain the
pitch angle of the main frame, mu = q14.  It follows from the condition that
both tyres touch the ground (Section 4.8.1).  Exactly as in the thesis the
pitch *rate* is written as a linear combination of the generalised speeds,

    mu_dot = H_1 u_1 + ... + H_11 u_11                         (cf. 4.18)

Symbolically ``mu`` and ``mud`` (= mu_dot) are kept as separate symbols; the
numeric model supplies mu, the coefficients H_r and -- optionally -- the
coefficient-rate term  hb = sum_r (dH_r/dt) u_r  that the thesis neglects by
treating the scheduled coefficients as constants.
"""
from __future__ import annotations

import os
import sympy as sp

from . import frames
from .params import GEOM_ORDER

NQ = 11
q = sp.symbols("q1:12", real=True)
u = sp.symbols("u1:12", real=True)
mu, mud, hb = sp.symbols("mu mud hb", real=True)
P = sp.symbols(" ".join(GEOM_ORDER), real=True)
par = dict(zip(GEOM_ORDER, P))

# order in which points / bodies are exported by the generated code
POINTS = ["A", "B", "C", "D", "F", "G", "P", "H", "Q", "R"]
BODIES = ["A", "B", "D", "F", "G"]          # angular velocity carriers
FRAMES = ["M", "D", "A", "B", "E", "F"]


def build():
    """Positions (in N) of the key points and angular velocities of the bodies.

    Returns ``(r, R, w)``: dict of 3x1 position matrices, dict of rotation
    matrices, dict of 3x1 angular velocity matrices (all expressed in N).
    """
    (q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11) = q
    g = par
    R = frames.chain(q9, q10, q11, q2, mu, g["eps"], sp)
    col = lambda X, i: R[X][:, i - 1]
    O = sp.Matrix([q7, 0, q8])                    # ground point under rear axle

    r = {}
    # rear axle: along d2 from O (Fig. 4.25, toroidal tyre:
    #   T'_Rz = T_Rz + T_Ry tan(q9) - q4/cos(q9) = Rr - rhor + (rhor - q4)/cos q9)
    hR = (g["Rr"] - g["rhor"]) + (g["rhor"] - q4) / sp.cos(q9)
    r["D"] = O + hR * col("D", 2)
    r["P"] = r["D"] + g["Ra"] * col("B", 1)
    r["A"] = r["P"] + g["Px"] * col("A", 1) + g["Pz"] * col("A", 2) \
        + g["Py"] * col("A", 3)
    r["H"] = r["P"] + (g["Px"] + g["Hx"]) * col("A", 1) \
        + (g["Pz"] + g["Hz"]) * col("A", 2)
    r["R"] = r["P"] + (g["Px"] + g["Rx"]) * col("A", 1) \
        + (g["Pz"] + g["Rz"]) * col("A", 2)
    r["B"] = r["P"] + g["Bx"] * col("B", 1) + g["Bz"] * col("B", 2)
    r["F"] = r["H"] + g["Fx"] * col("F", 1) + g["Fz"] * col("F", 2)
    r["Q"] = r["H"] + g["Qx"] * col("F", 1) + g["Qz"] * col("F", 2)
    r["G"] = r["Q"] - q1 * col("F", 2)
    r["C"] = r["G"] + g["Cx"] * col("F", 1) + g["Cz"] * col("F", 2)

    # angular velocities in N: sums of simple rotations along the chain
    w = {}
    wD = u[9] * col("M", 2) + u[8] * col("M", 1)          # yaw + roll
    w["A"] = wD + mud * col("D", 3)                        # + pitch
    w["B"] = w["A"] + u[1] * col("A", 3)                   # + swing-arm
    w["D"] = w["B"] + u[5] * col("A", 3)                   # + rear wheel spin
    w["F"] = w["A"] + u[10] * col("E", 2)                  # + steer
    w["G"] = w["F"] + u[4] * col("F", 3)                   # + front wheel spin
    return r, R, w


def closure_residual(r, R):
    """Front tyre touches the ground with vertical deflection q3.

    Height of the front axle above ground = height of the lowest point of
    the (toroidal) tyre:  (Rf - rhof) cos(gamma1) + rhof - q3 ,
    with sin(gamma1) = -f3 . m2.
    """
    f3_up = R["F"][1, 2]                     # f3 . n2
    cg1 = sp.sqrt(1 - f3_up**2)
    return r["G"][1] - (par["Rf"] - par["rhof"]) * cg1 - par["rhof"] + q[2]


def thesis_pose():
    """Closed-form main frame position of Section 4.8 (as printed).

    Returns a list of expressions [q13, q14, q15, q16, q17, q18, q19] in
    q1..q11 using the thesis' approximations (gamma1 ~ q9, small mu) and a
    toroidal tyre profile:  T_z = R - rho + rho cos(q9),  T_y = rho sin(q9).
    """
    (q1, q2, q3, q4, _, _, _, _, q9, _, q11) = q
    g = par
    Ra, Px, Pz, Hx, Hz, Qx, Qz, eps = (g[k] for k in
                                       ["Ra", "Px", "Pz", "Hx", "Hz", "Qx", "Qz", "eps"])
    c, s = sp.cos, sp.sin
    TFz = g["Rf"] - g["rhof"] + g["rhof"] * c(q9)
    TFy = g["rhof"] * s(q9)
    TRz = g["Rr"] - g["rhor"] + g["rhor"] * c(q9)
    TRy = g["rhor"] * s(q9)
    # (4.17)
    num = (-Ra * s(q2) - Hz - Pz - c(eps) * (Qz - q1) - Qx * c(q11) * s(eps)
           + TFz - TRz + sp.tan(q9) * (TFy - TRy) + (q4 - q3) / c(q9))
    den = Ra * c(q2) + Hx + Px + c(eps) * c(q11) * Qx - s(eps) * (Qz - q1)
    q14 = num / den
    # (4.20), (4.21)
    q15 = Ra * c(q2 + q14) + Px * c(q14) - Pz * s(q14)
    q16 = TRz + TRy * sp.tan(q9) - q4 / c(q9) + Ra * s(q2 + q14) \
        + Px * s(q14) + Pz * c(q14)
    # wheelbase, Section 4.8.4
    q18 = (Ra * c(q2 + q14) + (Px + Hx) * c(q14) - (Pz + Hz) * s(q14)
           + Qx * c(q11) * c(q14 + eps) + (q1 - Qz) * s(q14 + eps))
    # trail, (4.25) with the small-angle form
    q19 = (g["Rf"] * (q14 * c(eps) + s(eps)) - Qx) / (c(eps) - q14 * s(eps))
    # yaw, Section 4.8.6
    dk = sp.atan(c(eps + q14) * sp.tan(q11) / c(q9))
    y = (Qx * s(q11) - (TFz + TFy * sp.tan(q9) - q3 / c(q9)) * s(q9) * c(q9)
         * (1 - c(dk)) + s(q9) * (q3 - q4))
    q13 = y / q18
    q17 = -q13 * q15 + q4 * s(q9)
    return [q13, q14, q15, q16, q17, q18, q19]


# --------------------------------------------------------------------------
# code generation
# --------------------------------------------------------------------------
def _emit(name, args_src, outputs):
    """Emit a python function computing ``outputs`` (dict name -> sympy
    Matrix / list / scalar) with common sub-expression elimination."""
    flat, shapes = [], {}
    for k, v in outputs.items():
        if isinstance(v, sp.MatrixBase):
            shapes[k] = v.shape
            flat += list(v)
        elif isinstance(v, (list, tuple)):
            shapes[k] = (len(v),)
            flat += list(v)
        else:
            shapes[k] = ()
            flat.append(v)
    rep, red = sp.cse(flat, symbols=sp.numbered_symbols("x"), order="none")
    L = [f"def {name}({args_src}):"]
    L.append("    q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = q")
    if "u" in [a.strip() for a in args_src.split(",")]:
        L.append("    u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11 = u")
    L.append("    " + ", ".join(GEOM_ORDER) + " = p")
    for sym, ex in rep:
        L.append(f"    {sym} = {sp.pycode(ex, fully_qualified_modules=False)}")
    L.append("    out = {}")
    i = 0
    for k, shp in shapes.items():
        n = 1
        for d in shp:
            n *= d
        items = ", ".join(sp.pycode(e, fully_qualified_modules=False)
                          for e in red[i:i + n])
        i += n
        if shp == ():
            L.append(f"    out['{k}'] = {items}")
        elif len(shp) == 1 or shp[1] == 1:
            L.append(f"    out['{k}'] = _np.array([{items}])")
        else:
            L.append(f"    out['{k}'] = _np.array([{items}]).reshape({shp})")
    L.append("    return out")
    return "\n".join(L) + "\n\n\n"


def generate(path=None, verbose=True):
    """Derive everything and write ``okane/_generated.py``."""
    r, R, w = build()
    qv, uv = sp.Matrix(q), sp.Matrix(u)
    src = ('"""AUTO-GENERATED by okane.symbolic.generate() -- do not edit."""\n'
           "from math import sin, cos, tan, sqrt, atan\n"
           "import numpy as _np\n\n\n")

    # ---- 1. exact loop closure for mu ----------------------------------
    gres = closure_residual(r, R)
    g_q = sp.Matrix([gres]).jacobian(qv)
    out = {"g": gres, "g_mu": sp.diff(gres, mu), "g_q": g_q.T,
           "g_mumu": sp.diff(gres, mu, 2), "g_qmu": g_q.diff(mu).T,
           "g_qq": g_q.T.jacobian(qv)}
    src += _emit("closure", "mu, q, p", out)
    if verbose:
        print("closure done")

    # ---- 2. positions, partial velocities, acceleration bias ------------
    out = {}
    for k in FRAMES:
        out["R_" + k] = R[k]
    for k in POINTS:
        rk = r[k]
        Jq = rk.jacobian(qv)                 # d r / d q_r
        rmu = rk.diff(mu)                    # d r / d mu
        v = Jq * uv + rmu * mud
        ab = v.jacobian(qv) * uv + v.diff(mu) * mud + rmu * hb
        out["r_" + k], out["Jq_" + k], out["rmu_" + k] = rk, Jq, rmu
        out["ab_" + k] = ab
    for k in BODIES:
        wk = w[k]
        out["w_" + k] = wk
        out["Jw_" + k] = wk.jacobian(uv)     # partial ang. vel. (explicit u)
        out["wmud_" + k] = wk.diff(mud)      # ... and through mu_dot
        out["alb_" + k] = wk.jacobian(qv) * uv + wk.diff(mu) * mud \
            + wk.diff(mud) * hb
    src += _emit("kin", "q, u, mu, mud, hb, p", out)
    if verbose:
        print("kinematics done")

    # ---- 3. thesis closed-form pose and its Taylor coefficients ---------
    pose = sp.Matrix(thesis_pose())
    src += _emit("thesis_pose", "q, p", {"pose": pose, "jac": pose.jacobian(qv)})
    if verbose:
        print("thesis pose done")

    path = path or os.path.join(os.path.dirname(__file__), "_generated.py")
    with open(path, "w") as fh:
        fh.write(src)
    return path


if __name__ == "__main__":
    print(generate())
