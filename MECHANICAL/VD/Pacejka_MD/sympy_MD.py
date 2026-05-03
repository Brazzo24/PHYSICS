"""
pacejka_moto_sympy.py
=====================
Symbolic companion to pacejka_moto_stage3.py

Provides the linearised 6×6 A-matrix in fully symbolic form using SymPy,
with the same equation structure and sign conventions as Stage 3.  The
primary purposes are:

  1. Readability  – inspect any A[i,j] as a closed-form expression in
                    named physical parameters (Pacejka Ch.11 symbols).
  2. Verification – evaluate the symbolic A numerically via lambdify and
                    compare element-wise with build_A_matrix(); max error
                    should be < 1e-12 (floating-point round-off only).
  3. Sensitivity  – diff(A_sym[i,j], symbol) gives exact partial derivatives,
                    showing analytically which parameters control which coupling.
  4. LaTeX export – sp.latex(A_sym[i,j]) or print_latex_summary() for reports.

Design decisions
----------------
* The characteristic polynomial of a 6×6 symbolic matrix with 25 free
  parameters is computationally intractable.  We therefore compute it
  numerically (via numpy) and offer a 2×2 reduced block for each mode pair
  to give analytic insight into the dominant physics.
* lambdify is used throughout for fast numerical evaluation; it produces
  numpy-compatible functions from the symbolic expressions.
* All symbols carry the same physical names as used in the Stage 3 comments
  and in Pacejka Ch.11, making it straightforward to map between the code
  and the textbook.

State vector:  x = [φ, p, δ, δ̇, v, r]
  φ   – mainframe roll angle          [rad]
  p   – roll rate dφ/dt               [rad/s]
  δ   – steer angle                   [rad]
  δ̇   – steer rate                    [rad/s]
  v   – lateral velocity              [m/s]
  r   – yaw rate                      [rad/s]
"""

from __future__ import annotations
import sympy as sp
import numpy as np
from typing import Dict, List, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pacejka_moto_stage12 import MotorcycleParams, normal_loads, _Ky
from pacejka_moto_stage3  import build_inertia, InertiaParams, build_A_matrix


# ============================================================================
# 1.  SYMBOL DEFINITIONS
# ============================================================================

def make_symbols() -> Dict[str, sp.Symbol]:
    """
    Declare all symbols used in the linearised A-matrix.

    Returns a dict mapping short Python names to SymPy symbols.
    The LaTeX rendering of each symbol is set via the 'name' argument
    so that sp.latex() produces readable output.

    Physical parameter groups
    -------------------------
    Kinematics  : u, g
    Geometry    : m, h, a_c, b_c, lam (λ), t_c
    Front tyre  : Cy1, Cg1, t1, rc1Fz1
    Rear tyre   : Cy2, Cg2, t2, rc2Fz2
    Inertia     : Ixx, Izz, Ids
    Gyroscopic  : Gamma1 (Γ₁ = Iwy1·u/r1), Gamma2 (Γ₂ = Iwy2·u/r2)
    Front assy  : mf_tot, ef, es
    Damper      : cs
    """
    syms = {}

    # Scalars that appear as positive reals in the physics
    pos = dict(positive=True)

    # Kinematics / environment
    syms['u']        = sp.Symbol(r'u',        **pos)   # forward speed
    syms['g']        = sp.Symbol(r'g',        **pos)   # gravity

    # Vehicle geometry (global)
    syms['m']        = sp.Symbol(r'm',        **pos)   # total mass
    syms['h']        = sp.Symbol(r'h',        **pos)   # CoG height
    syms['a_c']      = sp.Symbol(r'a_c',      **pos)   # CoG–front axle (longitudinal)
    syms['b_c']      = sp.Symbol(r'b_c',      **pos)   # CoG–rear  axle (longitudinal)
    syms['lam']      = sp.Symbol(r'\lambda',  **pos)   # rake (steer head) angle
    syms['t_c']      = sp.Symbol(r't_c',      **pos)   # caster length

    # Linearised tyre stiffnesses (evaluated at operating point φ=δ=0)
    syms['Cy1']      = sp.Symbol(r'C_{y1}',   **pos)   # front cornering stiffness
    syms['Cy2']      = sp.Symbol(r'C_{y2}',   **pos)   # rear  cornering stiffness
    syms['Cg1']      = sp.Symbol(r'C_{\gamma1}', **pos) # front camber stiffness
    syms['Cg2']      = sp.Symbol(r'C_{\gamma2}', **pos) # rear  camber stiffness
    syms['t1']       = sp.Symbol(r't_1',      **pos)   # front pneumatic trail
    syms['t2']       = sp.Symbol(r't_2',      **pos)   # rear  pneumatic trail
    syms['rc1Fz1']   = sp.Symbol(r'r_{c1}F_{z1}', **pos) # front overturning parameter
    syms['rc2Fz2']   = sp.Symbol(r'r_{c2}F_{z2}', **pos) # rear  overturning parameter

    # Inertia
    syms['Ixx']      = sp.Symbol(r'I_{xx}',   **pos)   # roll moment of inertia
    syms['Izz']      = sp.Symbol(r'I_{zz}',   **pos)   # yaw  moment of inertia
    syms['Ids']      = sp.Symbol(r'I_{\delta}', **pos) # steer-axis inertia

    # Gyroscopic coefficients  Γ_i = Iwy_i · u / r_i  (speed-dependent)
    syms['Gamma1']   = sp.Symbol(r'\Gamma_1', **pos)   # front wheel gyroscopic
    syms['Gamma2']   = sp.Symbol(r'\Gamma_2', **pos)   # rear  wheel gyroscopic

    # Front assembly (for castoring gravity term)
    syms['mf_tot']   = sp.Symbol(r'm_{f,tot}', **pos)  # front assembly mass
    syms['ef']       = sp.Symbol(r'e_f',       **pos)  # front upper frame CoG offset
    syms['es']       = sp.Symbol(r'e_s',       **pos)  # front subframe CoG offset

    # Steering damper
    syms['cs']       = sp.Symbol(r'c_s',       **pos)  # damper coefficient

    return syms


# ============================================================================
# 2.  SYMBOLIC A-MATRIX
# ============================================================================

def build_A_symbolic(syms: Dict[str, sp.Symbol]) -> sp.Matrix:
    """
    Construct the symbolic 6×6 state matrix A.

    Entry A[i,j] is the coefficient of state x[j] in the equation for ẋ[i].
    All expressions are unsimplified (rational functions of the symbols)
    so that the physical origin of each term remains transparent.

    Equation derivation summary (Pacejka Ch.11, §11.3, linearised)
    ---------------------------------------------------------------
    Slip angles (Eq. 11.16 at φ=δ=0, ω'=0):
        α₁ = δ·cos λ − φ·sin λ − (v − a_c·r)/u
        α₂ =                    − (v + b_c·r)/u

    Camber angles (Eq. 11.12–11.14):
        γ₁ = φ + δ·sin λ
        γ₂ = φ

    Lateral tyre forces (linearised MF):
        Fy1 = Cy1·α₁ + Cg1·γ₁
        Fy2 = Cy2·α₂ + Cg2·γ₂

    Aligning torques:
        Mz1 = −t1·Fy1,   Mz2 = −t2·Fy2

    Overturning couples (linearised, Eq. 11.5 approx.):
        Mx  = −(rc1·Fz1 + rc2·Fz2)·φ

    Gyroscopic moments:
        Γ₁ = Iwy1·u/r1,   Γ₂ = Iwy2·u/r2
        (pre-substituted as free symbols, speed enters via u and Γ)

    Four dynamic equations + two kinematic identities:
        [0] dφ/dt  = p                          (identity)
        [1] Ixx·ṗ  = roll  Newton–Euler
        [2] dδ/dt  = δ̇                          (identity)
        [3] Ids·δ̈  = steer Newton–Euler
        [4] m·v̇   = lateral Newton (−m·u·r moves to RHS)
        [5] Izz·ṙ  = yaw Newton–Euler
    """
    s = syms   # shorthand
    cl  = sp.cos(s['lam'])
    sl  = sp.sin(s['lam'])

    # -- Tyre force partials (∂Fy_i/∂x_j) ----------------------------------
    # Front wheel (wheel 1)
    dFy1_phi   = -s['Cy1']*sl + s['Cg1']
    dFy1_delta =  s['Cy1']*cl + s['Cg1']*sl
    dFy1_v     = -s['Cy1'] / s['u']
    dFy1_r     =  s['Cy1'] * s['a_c'] / s['u']

    # Rear wheel (wheel 2)
    dFy2_phi   =  s['Cg2']
    dFy2_delta =  sp.Integer(0)
    dFy2_v     = -s['Cy2'] / s['u']
    dFy2_r     = -s['Cy2'] * s['b_c'] / s['u']

    # Totals (Fy = Fy1 + Fy2)
    dFy_phi   = dFy1_phi   + dFy2_phi
    dFy_delta = dFy1_delta                  # dFy2_delta = 0
    dFy_v     = dFy1_v     + dFy2_v
    dFy_r     = dFy1_r     + dFy2_r

    # Overturning couple partial
    dMx_phi = -(s['rc1Fz1'] + s['rc2Fz2'])

    # -- Aligning torque partials (∂Mzi/∂x_j = −ti·∂Fyi/∂x_j) -------------
    dMz1_phi   = -s['t1'] * dFy1_phi
    dMz1_delta = -s['t1'] * dFy1_delta
    dMz1_v     = -s['t1'] * dFy1_v
    dMz1_r     = -s['t1'] * dFy1_r
    dMz2_phi   = -s['t2'] * dFy2_phi
    dMz2_delta =  sp.Integer(0)
    dMz2_v     = -s['t2'] * dFy2_v
    dMz2_r     = -s['t2'] * dFy2_r

    # -- Effective caster arm and steer gravity ------------------------------
    caster_arm  = s['t1']*cl + s['t_c']
    steer_grav  = s['mf_tot']*s['g']*(s['ef']+s['es'])/2 * sl

    # -- Assemble A ----------------------------------------------------------
    A = sp.zeros(6, 6)

    # Kinematic identities
    A[0, 1] = sp.Integer(1)    # dφ/dt = p
    A[2, 3] = sp.Integer(1)    # dδ/dt = δ̇

    # Row 1 – Roll equation: Ixx·ṗ = ...
    #   m·g·h·φ      (gravitational overturning)
    # + h·Fy         (lateral forces × height)
    # + Mx           (tyre overturning couples)
    # + Γ1·cosλ·r    (front gyro projected onto roll axis)
    # + Γ2·r         (rear  gyro projected onto roll axis)
    # − m·u·h·r      (centripetal in roll)
    A[1, 0] = (s['m']*s['g']*s['h'] + s['h']*dFy_phi  + dMx_phi) / s['Ixx']
    A[1, 2] = (                        s['h']*dFy_delta          ) / s['Ixx']
    A[1, 4] = (                        s['h']*dFy_v              ) / s['Ixx']
    A[1, 5] = (-s['m']*s['u']*s['h'] + s['h']*dFy_r
               + s['Gamma1']*cl + s['Gamma2'])                     / s['Ixx']

    # Row 4 – Lateral Newton: m·v̇ = Fy − m·u·r
    A[4, 0] = dFy_phi   / s['m']
    A[4, 2] = dFy_delta / s['m']
    A[4, 4] = dFy_v     / s['m']
    A[4, 5] = (dFy_r - s['m']*s['u']) / s['m']

    # Row 5 – Yaw equation: Izz·ṙ = ...
    #   Fy1·a_c − Fy2·b_c    (yaw moment from lateral forces)
    # + Mz1 + Mz2             (self-aligning torques)
    # − (Γ1+Γ2)·p             (gyroscopic reaction to roll rate)
    A[5, 0] = (s['a_c']*dFy1_phi  - s['b_c']*dFy2_phi
               + dMz1_phi + dMz2_phi)                             / s['Izz']
    A[5, 1] = -(s['Gamma1'] + s['Gamma2'])                        / s['Izz']
    A[5, 2] = (s['a_c']*dFy1_delta + dMz1_delta)                  / s['Izz']
    A[5, 4] = (s['a_c']*dFy1_v  - s['b_c']*dFy2_v
               + dMz1_v + dMz2_v)                                 / s['Izz']
    A[5, 5] = (s['a_c']*dFy1_r  - s['b_c']*dFy2_r
               + dMz1_r + dMz2_r)                                 / s['Izz']

    # Row 3 – Steer equation: Ids·δ̈ = ...
    #   steer_grav·φ                     (castoring gravity, restores δ when leaned)
    # − caster_arm·Fy1                   (aligning + caster force projected to steer axis)
    # − Γ1·cosλ·p                        (front gyro reacts to roll rate → steer torque)
    # − cs·δ̇                             (steering damper)
    A[3, 0] = (steer_grav - caster_arm*dFy1_phi  ) / s['Ids']
    A[3, 1] = -s['Gamma1']*cl                       / s['Ids']
    A[3, 2] = (-caster_arm*dFy1_delta)              / s['Ids']
    A[3, 3] = -s['cs']                              / s['Ids']
    A[3, 4] = (-caster_arm*dFy1_v    )              / s['Ids']
    A[3, 5] = (-caster_arm*dFy1_r    )              / s['Ids']

    return A


# ============================================================================
# 3.  NUMERICAL SUBSTITUTION HELPERS
# ============================================================================

def make_subs(p: MotorcycleParams, ip: InertiaParams,
              u_val: float, cs_val: float = 15.0,
              Fz0: float = 1500.0) -> Dict[sp.Symbol, float]:
    """
    Build the substitution dict {symbol: float} for a given operating point.

    Can be used with A_sym.subs(subs).evalf() for spot-checks, or passed
    to build_lambdified() which is much faster for sweeps.
    """
    syms = make_symbols()
    Fz1, Fz2 = normal_loads(p)
    return {
        syms['u']:      u_val,
        syms['g']:      9.81,
        syms['m']:      p.m,
        syms['h']:      p.h,
        syms['a_c']:    p.ac,
        syms['b_c']:    p.bc,
        syms['lam']:    p.lam,
        syms['t_c']:    p.tc,
        syms['Cy1']:    _Ky(p.pKy1, p.pKy2, Fz1, Fz0),
        syms['Cy2']:    _Ky(p.pKy1, p.pKy2, Fz2, Fz0),
        syms['Cg1']:    p.pDy_gamma * Fz1 * p.pCy_gamma,
        syms['Cg2']:    p.pDy_gamma * Fz2 * p.pCy_gamma,
        syms['t1']:     p.t0,
        syms['t2']:     p.t0 * 0.85,
        syms['rc1Fz1']: p.rc1 * Fz1,
        syms['rc2Fz2']: p.rc2 * Fz2,
        syms['Ixx']:    ip.Imx0,
        syms['Izz']:    p.m * p.a * p.b,
        syms['Ids']:    ip.If_steer,
        syms['Gamma1']: ip.Iwy1 * u_val / p.r1,
        syms['Gamma2']: ip.Iwy2 * u_val / p.r2,
        syms['cs']:     cs_val,
        syms['mf_tot']: p.mf + p.ms,
        syms['ef']:     p.ef,
        syms['es']:     p.es,
    }


def build_lambdified(A_sym: sp.Matrix,
                     syms:  Dict[str, sp.Symbol]):
    """
    Create a fast numpy-backed function from the symbolic A-matrix.

    Returns
    -------
    A_fn : callable(*vals) -> np.ndarray shape (6,6)
    sym_order : list of symbol names in the order expected by A_fn

    Usage
    -----
        A_fn, sym_order = build_lambdified(A_sym, syms)
        subs = make_subs(p, ip, u_val)
        vals = [float(subs[syms[k]]) for k in sym_order]
        A_num = A_fn(*vals)
    """
    sym_order = list(syms.keys())
    sym_list  = [syms[k] for k in sym_order]
    A_fn = sp.lambdify(sym_list, A_sym, 'numpy')
    return A_fn, sym_order


def eval_A(A_fn, sym_order: List[str],
           syms: Dict[str, sp.Symbol],
           subs: Dict[sp.Symbol, float]) -> np.ndarray:
    """Evaluate the lambdified A at a substitution dict."""
    vals = [float(subs[syms[k]]) for k in sym_order]
    return np.asarray(A_fn(*vals), dtype=float)


# ============================================================================
# 4.  SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity(A_sym:  sp.Matrix,
                syms:   Dict[str, sp.Symbol],
                row:    int,
                col:    int,
                wrt:    str,
                simplify: bool = True) -> sp.Expr:
    """
    Return ∂A[row,col]/∂(syms[wrt]) as a SymPy expression.

    Parameters
    ----------
    row, col : entry indices (0-based)
    wrt      : key into syms dict, e.g. 'Cy1', 'Gamma1', 'lam'
    simplify : whether to call sp.simplify() — fast for most entries
    """
    expr = sp.diff(A_sym[row, col], syms[wrt])
    return sp.simplify(expr) if simplify else expr


def sensitivity_table(A_sym: sp.Matrix,
                      syms:  Dict[str, sp.Symbol],
                      params: List[str] = None) -> None:
    """
    Print a table of ∂A[i,j]/∂param for all non-zero entries,
    for each parameter in `params` (defaults to key physical params).

    This is the main 'understanding' tool: it shows at a glance which
    A-matrix entries respond to which physical changes.
    """
    if params is None:
        params = ['Cy1', 'Cy2', 'Cg1', 'Cg2', 'Gamma1', 'Gamma2',
                  'lam', 't_c', 'h', 'cs']

    state_names = ['φ', 'p', 'δ', 'δ̇', 'v', 'r']
    eq_names    = ['dφ/dt', 'dp/dt', 'dδ/dt', 'dδ̇/dt', 'dv/dt', 'dr/dt']

    for param in params:
        if param not in syms:
            print(f"  [skip: {param} not in syms]")
            continue
        print(f"\n── ∂A/∂({param}) = ∂A/∂{syms[param]} ──")
        found_any = False
        for i in range(6):
            for j in range(6):
                if A_sym[i, j] == 0:
                    continue
                d = sp.diff(A_sym[i, j], syms[param])
                if d == 0:
                    continue
                found_any = True
                d_s = sp.simplify(d)
                print(f"  A[{eq_names[i]}, {state_names[j]}]  =  {d_s}")
        if not found_any:
            print("  (no non-zero partials)")


# ============================================================================
# 5.  REDUCED 2×2 BLOCK ANALYSIS  (analytic mode approximations)
# ============================================================================

def capsize_block(syms: Dict[str, sp.Symbol]) -> Tuple[sp.Matrix, sp.Expr]:
    """
    The capsize mode is dominated by the roll-only subsystem
    (ignoring steer, lateral, and yaw at very low speed / large Γ limit).
    Isolate the scalar roll equation at δ=v=r=0:

        Ixx·ṗ = (m·g·h + h·Cg_total − rc_total)·φ

    This gives a single real eigenvalue:
        σ_capsize ≈ sqrt(A[1,0])

    Returns the symbolic expression for A[1,0] and the approximate
    eigenvalue symbol.
    """
    s = syms
    cl, sl = sp.cos(s['lam']), sp.sin(s['lam'])
    dFy_phi = (-s['Cy1']*sl + s['Cg1']) + s['Cg2']
    dMx_phi = -(s['rc1Fz1'] + s['rc2Fz2'])
    a10 = (s['m']*s['g']*s['h'] + s['h']*dFy_phi + dMx_phi) / s['Ixx']
    return a10


def wobble_block(syms: Dict[str, sp.Symbol]) -> sp.Matrix:
    """
    The wobble mode is dominated by the steer subsystem.
    At high speed, the 2×2 block [δ, δ̇] with φ, v, r frozen gives:

        | 0         1    | [δ  ]
        | A[3,2]   A[3,3]| [δ̇  ]

    A[3,2] is the steer restoring stiffness (negative → oscillatory)
    A[3,3] is the steer damping (negative → stable)

    Returns the 2×2 matrix and its characteristic polynomial.
    """
    s = syms
    cl, sl = sp.cos(s['lam']), sp.sin(s['lam'])
    caster = s['t1']*cl + s['t_c']
    dFy1_delta = s['Cy1']*cl + s['Cg1']*sl
    a32 = -caster * dFy1_delta / s['Ids']
    a33 = -s['cs'] / s['Ids']

    M = sp.Matrix([[sp.Integer(0), sp.Integer(1)],
                   [a32,           a33           ]])
    lam_eig = sp.Symbol('lambda')
    cp = M.charpoly(lam_eig)
    return M, sp.Poly(cp, lam_eig)


# ============================================================================
# 6.  LATEX / PRETTY PRINT UTILITIES
# ============================================================================

def print_A_entry(A_sym: sp.Matrix, syms: Dict[str, sp.Symbol],
                  row: int, col: int,
                  mode: str = 'pretty') -> None:
    """
    Print one A-matrix entry in a chosen format.

    mode : 'pretty'  – sp.pretty (unicode, terminal-friendly)
           'latex'   – sp.latex  (paste into a document)
           'str'     – plain str (compact)
    """
    state_names = ['φ', 'p', 'δ', 'δ̇', 'v', 'r']
    eq_names    = ['dφ/dt', 'dp/dt', 'dδ/dt', 'dδ̇/dt', 'dv/dt', 'dr/dt']
    expr = sp.simplify(A_sym[row, col])
    label = f"A[{eq_names[row]}, {state_names[col]}]"
    if mode == 'latex':
        print(f"% {label}")
        print(f"  {sp.latex(expr)}")
    elif mode == 'pretty':
        print(f"\n{label} =")
        sp.pprint(expr, use_unicode=True)
    else:
        print(f"{label} = {expr}")


def print_latex_summary(A_sym: sp.Matrix, syms: Dict[str, sp.Symbol],
                         entries: List[Tuple[int,int]] = None) -> None:
    """
    Print LaTeX for selected (or all non-trivial) A-matrix entries.
    """
    if entries is None:
        entries = [(i, j) for i in range(6) for j in range(6)
                   if A_sym[i,j] not in (sp.Integer(0), sp.Integer(1))]
    state_names = ['\\phi','p','\\delta','\\dot\\delta','v','r']
    eq_names    = ['\\dot\\phi','\\dot p','\\dot\\delta',
                   '\\ddot\\delta','\\dot v','\\dot r']
    print("% Auto-generated LaTeX — Pacejka Ch.11 linearised A-matrix")
    print("% State: x = [phi, p, delta, ddelta, v, r]")
    for i, j in entries:
        expr = sp.simplify(A_sym[i, j])
        print(f"% A[{i},{j}] = d({eq_names[i]})/d({state_names[j]})")
        print(f"A_{{\\{eq_names[i]},{state_names[j]}}} &= {sp.latex(expr)} \\\\")


# ============================================================================
# 7.  NUMERICAL CROSS-CHECK
# ============================================================================

def cross_check(A_sym:  sp.Matrix,
                syms:   Dict[str, sp.Symbol],
                p:      MotorcycleParams,
                ip:     InertiaParams,
                u_val:  float = 20.0,
                cs_val: float = 15.0,
                tol:    float = 1e-10) -> bool:
    """
    Evaluate A_sym numerically and compare to build_A_matrix().
    Prints element-wise comparison and returns True if max |diff| < tol.
    """
    subs  = make_subs(p, ip, u_val, cs_val)
    A_fn, sym_order = build_lambdified(A_sym, syms)
    A_s   = eval_A(A_fn, sym_order, syms, subs)
    A_n   = build_A_matrix(u_val, p, ip, cs=cs_val)
    diff  = np.abs(A_s - A_n)
    maxd  = diff.max()

    print(f"Cross-check at u = {u_val} m/s:")
    print(f"  max |A_symbolic − A_numeric| = {maxd:.2e}  "
          f"({'PASS' if maxd < tol else 'FAIL – check equations'})")

    state = ['φ','p','δ','δ̇','v','r']
    eq    = ['dφ','dp','dδ','dδ̇','dv','dr']
    print(f"\n  {'':6s} " + "  ".join(f"{s:>10}" for s in state))
    for i in range(6):
        row_str = "  ".join(
            f"{diff[i,j]:10.2e}" if diff[i,j] > 1e-14 else f"{'0':>10}"
            for j in range(6)
        )
        print(f"  {eq[i]:5s}  {row_str}")
    return maxd < tol


# ============================================================================
# 8.  SELF-TEST / DEMO
# ============================================================================

if __name__ == '__main__':
    import textwrap

    print("=" * 65)
    print("Pacejka Ch.11 – Symbolic A-matrix companion")
    print("=" * 65)

    # -- Build symbols and symbolic A ----------------------------------------
    syms  = make_symbols()
    A_sym = build_A_symbolic(syms)

    print(f"\nSymbolic A built:  {A_sym.shape[0]}×{A_sym.shape[1]}  matrix")
    print(f"Non-zero entries:  "
          f"{sum(1 for i in range(6) for j in range(6) if A_sym[i,j] != 0)}")
    print(f"Free symbols:      {len(syms)}")

    # -- Print every non-trivial entry ---------------------------------------
    state_names = ['φ', 'p', 'δ', 'δ̇', 'v', 'r']
    eq_names    = ['dφ/dt', 'dp/dt', 'dδ/dt', 'dδ̇/dt', 'dv/dt', 'dr/dt']
    trivial     = {sp.Integer(0), sp.Integer(1)}

    print("\n" + "─" * 65)
    print("A-matrix entries (simplified)")
    print("─" * 65)
    for i in range(6):
        for j in range(6):
            e = A_sym[i, j]
            if e in trivial:
                continue
            e_s = sp.simplify(e)
            lbl = f"A[{eq_names[i]}, {state_names[j]}]"
            print(f"\n{lbl}")
            sp.pprint(e_s, use_unicode=True)

    # -- Numerical cross-check -----------------------------------------------
    print("\n" + "─" * 65)
    p  = MotorcycleParams()
    ip = build_inertia(p)
    cross_check(A_sym, syms, p, ip, u_val=20.0)

    # -- Selected sensitivities ----------------------------------------------
    print("\n" + "─" * 65)
    print("Key sensitivities  ∂A[i,j]/∂param")
    print("─" * 65)

    cases = [
        # (row, col, param, physical meaning)
        (1, 0, 'h',      "Roll instability sensitivity to CoG height"),
        (1, 5, 'Gamma1', "Roll gyroscopic coupling: how front wheel spin inertia helps"),
        (3, 2, 'Cy1',    "Steer restoring: effect of front cornering stiffness"),
        (3, 2, 'lam',    "Steer restoring: effect of rake angle"),
        (3, 3, 'cs',     "Steer damping: steering damper sensitivity"),
        (5, 1, 'Gamma1', "Yaw gyroscopic: front wheel gyro coupling to roll rate"),
        (1, 0, 'Cg1',    "Roll stability: effect of front camber stiffness"),
    ]
    for row, col, param, desc in cases:
        d = sensitivity(A_sym, syms, row, col, param, simplify=True)
        lbl = f"  ∂A[{eq_names[row]},{state_names[col]}]/∂{param}"
        print(f"\n{desc}")
        print(f"{lbl} =")
        sp.pprint(d, use_unicode=True)

    # -- Wobble 2×2 block analysis ------------------------------------------
    print("\n" + "─" * 65)
    print("Wobble 2×2 block  [δ, δ̇]  (steer subsystem)")
    print("─" * 65)
    M_wob, cp_wob = wobble_block(syms)
    print("\nMatrix:")
    sp.pprint(M_wob, use_unicode=True)
    print("\nCharacteristic polynomial (λ):")
    sp.pprint(cp_wob.as_expr(), use_unicode=True)
    print("\nNatural frequency squared  ωn² = A[3,2]  (negated for display):")
    sp.pprint(sp.simplify(-M_wob[1, 0]), use_unicode=True)
    print("\nDamping term  2ζωn = −A[3,3]:")
    sp.pprint(sp.simplify(-M_wob[1, 1]), use_unicode=True)

    # -- Capsize scalar approximation ----------------------------------------
    print("\n" + "─" * 65)
    print("Capsize scalar approximation  σ² ≈ A[roll/φ]")
    print("─" * 65)
    a10 = capsize_block(syms)
    print("\nA[dp/dt, φ] (roll restoring / overturning):")
    sp.pprint(sp.simplify(a10), use_unicode=True)
    print(textwrap.dedent("""
    Interpretation:
      Positive → overturning dominates → unstable (capsize)
      The three additive groups are:
        + m·g·h / Ixx          : gravitational overturning moment
        + h·(Cg1+Cg2−Cy1·sinλ) / Ixx : tyre lateral force effect (camber lifts,
                                         slip-induced Fy1 stabilises via geometry)
        − (rc1·Fz1+rc2·Fz2) / Ixx   : tyre overturning couples (stabilising)
    """).strip())

    # -- LaTeX for the non-trivial entries ------------------------------------
    print("\n" + "─" * 65)
    print("LaTeX for dynamic rows (paste into document)")
    print("─" * 65)
    print_latex_summary(A_sym, syms)