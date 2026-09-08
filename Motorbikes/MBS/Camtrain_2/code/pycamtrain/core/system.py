"""MBSystem -- the assembly and code-generation layer.

How this differs from Dymola/Modelica
-------------------------------------
Modelica is *acausal*: components declare equations, connections are flattened
into one large DAE, and the translator performs index reduction, tearing and
symbolic simplification before emitting C code.

This framework is *minimal-coordinate symbolic*. Components register
generalized coordinates directly and Kane's method eliminates the constraint
forces analytically, so what comes out is a plain ODE:

    M(q) * d/dt [q; u] = F(q, u, t)

No DAE index reduction, no constraint stabilisation, no initialisation solver.
The price is that closed kinematic loops must be handled explicitly rather than
being resolved for free by the translator.

The symbolic assembly here plays the same role as Dymola's translation step,
and it has the same cost profile: expensive once, cheap per simulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import sympy as sp
from sympy.physics.mechanics import KanesMethod, Point, ReferenceFrame, dynamicsymbols

T_SYM = dynamicsymbols._t


@dataclass
class Coordinate:
    """One generalized coordinate and its generalized speed."""

    name: str
    q: sp.Expr
    u: sp.Expr
    q0: float = 0.0
    u0: float = 0.0


@dataclass
class InputSignal:
    """An externally prescribed scalar signal, evaluated as f(t)."""

    name: str
    sym: sp.Expr
    func: Callable[[float], float]


class MBSystem:
    """Registry for coordinates, bodies, loads, parameters and diagnostics."""

    def __init__(self, name: str = "system", gravity=None):
        self.name = name
        self.frame = ReferenceFrame("N")
        self.origin = Point("O")
        self.origin.set_vel(self.frame, 0)

        self.coords: List[Coordinate] = []
        self.bodies: List[object] = []
        self.loads: List[tuple] = []
        self.params: Dict[sp.Symbol, float] = {}
        self.inputs: List[InputSignal] = []
        self.outputs: Dict[str, sp.Expr] = {}
        self.components: List[object] = []

        # gravity vector expressed in the inertial frame, or None
        self.gravity = gravity

        self._assembled = False
        self.assembly_time: Optional[float] = None

    # ------------------------------------------------------------------
    # declaration API used by components
    # ------------------------------------------------------------------
    def parameter(self, name: str, value: float) -> sp.Symbol:
        """Declare a named numeric parameter. Returns the symbol to build with."""
        sym = sp.Symbol(f"p_{len(self.params)}", real=True)
        self.params[sym] = float(value)
        self._param_names = getattr(self, "_param_names", {})
        self._param_names[sym] = name
        return sym

    def coordinate(self, name: str, q0: float = 0.0, u0: float = 0.0) -> Coordinate:
        """Declare one degree of freedom. Returns a Coordinate."""
        idx = len(self.coords)
        c = Coordinate(name, dynamicsymbols(f"q{idx}"), dynamicsymbols(f"u{idx}"), q0, u0)
        self.coords.append(c)
        return c

    def input(self, name: str, func: Callable[[float], float]) -> sp.Expr:
        """Declare a prescribed time signal. Returns the symbol to build with."""
        sym = dynamicsymbols(f"in{len(self.inputs)}")
        self.inputs.append(InputSignal(name, sym, func))
        return sym

    def add_body(self, body) -> None:
        self.bodies.append(body)

    def add_load(self, load) -> None:
        self.loads.append(load)

    def output(self, name: str, expr) -> None:
        """Register a diagnostic expression, evaluated after the solve."""
        self.outputs[name] = expr

    # ------------------------------------------------------------------
    # assembly ("translation")
    # ------------------------------------------------------------------
    def assemble(self, verbose: bool = False, cse: bool = True):
        """Symbolic assembly -- the analogue of Dymola's translation step.

        ``cse`` enables sympy common-subexpression elimination inside
        ``lambdify``. This is the cheap counterpart of the tearing and
        simplification Dymola does before emitting C: without it every matrix
        entry is a flat expression and the generated code grows superlinearly
        with the number of meshes.
        """
        if not self.coords:
            raise ValueError("system has no degrees of freedom")

        t0 = time.perf_counter()

        qs = [c.q for c in self.coords]
        us = [c.u for c in self.coords]
        kdes = [q.diff(T_SYM) - u for q, u in zip(qs, us)]

        km = KanesMethod(self.frame, q_ind=qs, u_ind=us, kd_eqs=kdes)
        km.kanes_equations(self.bodies, self.loads)

        M = km.mass_matrix_full
        F = km.forcing_full

        self._param_syms = list(self.params.keys())
        self._pvals = np.array([self.params[s] for s in self._param_syms], dtype=float)

        in_syms = [i.sym for i in self.inputs]
        if not in_syms:                      # lambdify dislikes empty arg groups
            in_syms = [sp.Symbol("_no_input")]
        self._n_inputs = len(self.inputs)

        args = (T_SYM, tuple(qs + us), tuple(self._param_syms) or (sp.Symbol("_no_param"),),
                tuple(in_syms))

        self._M_func = sp.lambdify(args, M, "numpy", cse=cse)
        self._F_func = sp.lambdify(args, F, "numpy", cse=cse)

        if self.outputs:
            names = list(self.outputs)
            exprs = [sp.sympify(self.outputs[n]) for n in names]
            self._out_names = names
            self._out_func = sp.lambdify(args, exprs, "numpy", cse=cse)
        else:
            self._out_names, self._out_func = [], None

        self.km = km
        self.n_states = 2 * len(self.coords)
        self.assembly_time = time.perf_counter() - t0
        self.cse = cse
        self._assembled = True

        if verbose:
            print(f"[{self.name}] assembled {len(self.coords)} DOF "
                  f"({self.n_states} states) in {self.assembly_time:.3f} s")
        return self

    # ------------------------------------------------------------------
    # numeric evaluation
    # ------------------------------------------------------------------
    def _inputs_at(self, t: float):
        if self._n_inputs == 0:
            return (0.0,)
        return tuple(s.func(t) for s in self.inputs)

    def _params_arg(self):
        return tuple(self._pvals) if len(self._pvals) else (0.0,)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """State derivative. Signature matches scipy.integrate.solve_ivp."""
        ys = tuple(y)
        p = self._params_arg()
        inp = self._inputs_at(t)
        M = np.asarray(self._M_func(t, ys, p, inp), dtype=float)
        F = np.asarray(self._F_func(t, ys, p, inp), dtype=float).reshape(-1)
        return np.linalg.solve(M, F)

    @property
    def y0(self) -> np.ndarray:
        return np.array([c.q0 for c in self.coords] + [c.u0 for c in self.coords], float)

    def eval_outputs(self, t: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate all registered diagnostics over a trajectory."""
        if self._out_func is None:
            return {}
        p = self._params_arg()
        cols = []
        for k, tk in enumerate(t):
            cols.append(self._out_func(tk, tuple(y[:, k]), p, self._inputs_at(tk)))
        arr = np.asarray(cols, dtype=float).T
        return {n: arr[i] for i, n in enumerate(self._out_names)}

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = [f"MBSystem '{self.name}'",
                 f"  degrees of freedom : {len(self.coords)}",
                 f"  states             : {2 * len(self.coords)}",
                 f"  rigid bodies       : {len(self.bodies)}",
                 f"  loads              : {len(self.loads)}",
                 f"  parameters         : {len(self.params)}",
                 f"  prescribed inputs  : {len(self.inputs)}",
                 f"  diagnostics        : {len(self.outputs)}"]
        if self.assembly_time is not None:
            lines.append(f"  assembly time      : {self.assembly_time:.3f} s")
        lines.append("  coordinates:")
        for c in self.coords:
            lines.append(f"    - {c.name}")
        return "\n".join(lines)
