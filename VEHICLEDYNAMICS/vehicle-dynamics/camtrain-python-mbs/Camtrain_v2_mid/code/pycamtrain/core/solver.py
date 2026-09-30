"""Simulation driver and result container."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class RunMetadata:
    """Everything needed to reproduce and to compare against a Dymola run."""

    model: str
    method: str
    rtol: float
    atol: float
    t_end: float
    n_dof: int
    n_states: int
    assembly_time_s: Optional[float] = None
    solve_time_s: Optional[float] = None
    n_rhs_evals: Optional[int] = None
    n_jac_evals: Optional[int] = None
    n_steps: Optional[int] = None
    success: bool = False
    message: str = ""

    def __str__(self) -> str:
        rows = [
            ("model", self.model),
            ("solver", f"{self.method}  rtol={self.rtol:g}  atol={self.atol:g}"),
            ("size", f"{self.n_dof} DOF / {self.n_states} states"),
            ("t_end", f"{self.t_end:g} s"),
            ("assembly", f"{self.assembly_time_s:.3f} s" if self.assembly_time_s else "-"),
            ("solve", f"{self.solve_time_s:.3f} s" if self.solve_time_s else "-"),
            ("rhs evals", str(self.n_rhs_evals)),
            ("jac evals", str(self.n_jac_evals)),
            ("status", "ok" if self.success else f"FAILED: {self.message}"),
        ]
        w = max(len(r[0]) for r in rows)
        return "\n".join(f"  {k:<{w}} : {v}" for k, v in rows)


class Result:
    """Trajectory plus named access to coordinates, speeds and diagnostics."""

    def __init__(self, system, t, y, meta: RunMetadata):
        self.t = t
        self.y = y
        self.meta = meta
        self._system = system
        n = len(system.coords)
        self._q = {c.name: y[i] for i, c in enumerate(system.coords)}
        self._u = {c.name: y[n + i] for i, c in enumerate(system.coords)}
        self.signals: Dict[str, np.ndarray] = system.eval_outputs(t, y)

    def q(self, name: str) -> np.ndarray:
        return self._q[name]

    def u(self, name: str) -> np.ndarray:
        return self._u[name]

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self.signals:
            return self.signals[name]
        if name in self._q:
            return self._q[name]
        if name in self._u:
            return self._u[name]
        raise KeyError(f"unknown signal '{name}'. available: "
                       f"{sorted(set(self.signals) | set(self._q))}")

    @property
    def names(self):
        return sorted(set(self.signals) | set(self._q))


def simulate(system, t_end: float, n_points: int = 2001, method: str = "Radau",
             rtol: float = 1e-8, atol: float = 1e-10, y0=None,
             verbose: bool = False) -> Result:
    """Integrate an assembled MBSystem and return a Result.

    'Radau' (implicit, stiff-capable) is the closest scipy analogue to Dymola's
    default Dassl/Cvode configuration. Use 'DOP853' for cheap non-stiff runs.
    """
    if not system._assembled:
        system.assemble()

    y0 = system.y0 if y0 is None else np.asarray(y0, float)
    t_eval = np.linspace(0.0, t_end, n_points)

    t0 = time.perf_counter()
    sol = solve_ivp(system.rhs, (0.0, t_end), y0, method=method,
                    t_eval=t_eval, rtol=rtol, atol=atol)
    solve_time = time.perf_counter() - t0

    meta = RunMetadata(
        model=system.name, method=method, rtol=rtol, atol=atol, t_end=t_end,
        n_dof=len(system.coords), n_states=system.n_states,
        assembly_time_s=system.assembly_time, solve_time_s=solve_time,
        n_rhs_evals=int(sol.nfev), n_jac_evals=int(getattr(sol, "njev", 0) or 0),
        n_steps=len(sol.t), success=bool(sol.success), message=str(sol.message),
    )

    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    if verbose:
        print(meta)

    return Result(system, sol.t, sol.y, meta)
