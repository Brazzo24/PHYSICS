"""Spatial components -- proof that the core is not restricted to 1D rotation.

The camtrain skeleton itself only needs rotational DOFs so far, but the whole
point of an MBS core (rather than a torsional lumped model) is that the same
machinery handles bodies that translate and rotate in space. PendulumBody is
here as the verification case for that path: it is a spatial rigid body under
gravity, solved by exactly the same assembly code as the gear train.
"""

from __future__ import annotations

import sympy as sp
from sympy.physics.mechanics import RigidBody, inertia

from ..core.component import Component
from ..core.connectors import RotFlange


class PendulumBody(Component):
    """Rigid body on a revolute joint about the housing z axis, under gravity.

    The centre of mass sits at distance ``l`` from the joint along the body's
    x axis, so the mass centre genuinely translates in the inertial frame and
    the velocity kinematics are not trivial.
    """

    def __init__(self, system, name, mass, l, J_com, parent=None,
                 phi0: float = 0.0, w0: float = 0.0, g: float = 9.81):
        super().__init__(system, name, parent)

        self.coord = self.coordinate("phi", phi0, w0)
        N, O = system.frame, system.origin

        self.frame = N.orientnew(f"pf_{len(system.coords)}", "Axis", (self.coord.q, N.z))
        self.frame.set_ang_vel(N, self.coord.u * N.z)

        l_s = self.parameter("l", l)
        m_s = self.parameter("m", mass)
        g_s = self.parameter("g", g)
        J_s = self.parameter("J_com", J_com)

        self.com = O.locatenew(f"pc_{len(system.coords)}", l_s * self.frame.x)
        self.com.v2pt_theory(O, N, self.frame)

        I = inertia(self.frame, J_s / 2, J_s / 2, J_s)
        self.body = RigidBody(self.path.replace(".", "_"), self.com, self.frame,
                              m_s, (I, self.com))
        system.add_body(self.body)

        # Gravity acts along +x of the inertial frame and the mass centre sits
        # at +l*x when phi = 0, so phi = 0 is the hanging (stable) equilibrium
        # and the restoring torque is -m*g*l*sin(phi).
        system.add_load((self.com, m_s * g_s * N.x))

        self.flange = RotFlange(self.sub("flange"), self.coord.q, self.coord.u,
                                self.frame, N.z, body_frame=self.frame)

        self.output("phi", self.coord.q)
        self.output("w", self.coord.u)
