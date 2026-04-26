"""
Parameter sweeps.

Usage
-----
    from motodyn.analyses.parameter_sweep import sweep

    results = sweep(
        bike,
        param_path="suspension.swingarm_pivot",
        values=[(0.40, 0.28), (0.42, 0.30), (0.44, 0.32)],
        metric=lambda b: anti_squat_ratio(b),
    )

Returns a list of (value, metric) tuples, or a pandas DataFrame if
pandas is installed and `as_dataframe=True`.

A `grid_sweep` helper does a cartesian product over multiple parameters.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, Optional, Sequence

from ..parameters import Motorcycle


def sweep(
    bike: Motorcycle,
    param_path: str,
    values: Iterable[Any],
    metric: Callable[[Motorcycle], Any],
    as_dataframe: bool = False,
):
    """
    Evaluate `metric(bike)` for each value of a single parameter.

    Parameters
    ----------
    bike : Motorcycle
        Baseline vehicle; never mutated.
    param_path : str
        Dotted path, e.g. "suspension.k_rear_wheel" or a top-level field
        like "mass" (for swapping entire sub-objects).
    values : iterable
        Values to assign.
    metric : callable
        Function of a Motorcycle returning the quantity of interest.
        Returning a dict or dataclass works fine; it will be kept intact.
    as_dataframe : bool
        If True, return a pandas DataFrame with one row per value and the
        metric's fields expanded into columns (dataclass or dict metrics).
    """
    rows = []
    for v in values:
        modified = bike.with_changes(**{param_path: v})
        rows.append((v, metric(modified)))

    if not as_dataframe:
        return rows

    import pandas as pd
    from dataclasses import is_dataclass, asdict

    records = []
    for v, m in rows:
        rec = {"value": v}
        if is_dataclass(m):
            rec.update(asdict(m))
        elif isinstance(m, dict):
            rec.update(m)
        else:
            rec["metric"] = m
        records.append(rec)
    return pd.DataFrame.from_records(records)


def grid_sweep(
    bike: Motorcycle,
    grid: dict[str, Sequence[Any]],
    metric: Callable[[Motorcycle], Any],
    as_dataframe: bool = False,
):
    """
    Cartesian sweep over multiple parameters.

    Example:
        grid_sweep(bike,
                   {"suspension.k_rear_wheel": [80e3, 90e3, 100e3],
                    "brakes.front_brake_fraction": [0.6, 0.7, 0.8]},
                   metric=lambda b: anti_squat_ratio(b))
    """
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    rows = []
    for combo in product(*value_lists):
        changes = dict(zip(keys, combo))
        modified = bike.with_changes(**changes)
        rows.append((combo, metric(modified)))

    if not as_dataframe:
        return rows

    import pandas as pd
    from dataclasses import is_dataclass, asdict

    records = []
    for combo, m in rows:
        rec = dict(zip(keys, combo))
        if is_dataclass(m):
            rec.update(asdict(m))
        elif isinstance(m, dict):
            rec.update(m)
        else:
            rec["metric"] = m
        records.append(rec)
    return pd.DataFrame.from_records(records)
