"""Hierarchical component base class.

Mirrors Modelica's ownership model: a component owns sub-components and its
instance path is dotted, e.g. ``frontBank.stepShaft.bearingA``. Every
parameter, coordinate and diagnostic is registered under that path, so the
result namespace looks like a Dymola result file.
"""

from __future__ import annotations

from typing import List, Optional


class Component:
    def __init__(self, system, name: str, parent: Optional["Component"] = None):
        self.system = system
        self.name = name
        self.parent = parent
        self.children: List["Component"] = []
        if parent is not None:
            parent.children.append(self)
        else:
            system.components.append(self)

    # -- naming --------------------------------------------------------
    @property
    def path(self) -> str:
        if self.parent is None:
            return self.name
        return f"{self.parent.path}.{self.name}"

    def sub(self, name: str) -> str:
        """Fully qualified name for something this component owns."""
        return f"{self.path}.{name}"

    # -- registration helpers -----------------------------------------
    def parameter(self, name: str, value: float):
        return self.system.parameter(self.sub(name), value)

    def coordinate(self, name: str, q0: float = 0.0, u0: float = 0.0):
        return self.system.coordinate(self.sub(name), q0, u0)

    def output(self, name: str, expr) -> None:
        self.system.output(self.sub(name), expr)

    # -- tree ----------------------------------------------------------
    def tree(self, indent: int = 0) -> str:
        lines = ["  " * indent + f"{self.name}  <{type(self).__name__}>"]
        for c in self.children:
            lines.append(c.tree(indent + 1))
        return "\n".join(lines)
