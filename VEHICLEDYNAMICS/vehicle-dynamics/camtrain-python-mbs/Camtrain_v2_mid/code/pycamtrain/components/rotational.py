"""Rotational drivetrain components.

Every shaft here is a genuine spatial rigid body with a full inertia dyadic,
constrained to one rotational degree of freedom by construction rather than by
a constraint equation. Adding radial bearing motion later means giving the same
body two more coordinates -- no change to the framework.
"""

from __future__ import annotations

import sympy as sp
from sympy.physics.mechanics import RigidBody, inertia

from ..core.component import Component
from ..core.connectors import RotFlange, SpatialFrame

_AXES = {"x": 0, "y": 1, "z": 2}


def _axis_vec(frame, axis: str):
    return {"x": frame.x, "y": frame.y, "z": frame.z}[axis]


class Housing(Component):
    """Stationary engine housing. Provides the inertial reference.

    Equivalent to the single inner ``MultiBody.World`` of Section 3.1: it is
    instantiated once at the top level and every stationary attachment refers
    to it.
    """

    def __init__(self, system, name: str = "housing", parent=None):
        super().__init__(system, name, parent)
        self.frame = system.frame
        self.point = system.origin
        self.flange = RotFlange(self.sub("flange"), sp.Integer(0), sp.Integer(0),
                                system.frame, system.frame.z, body_frame=None)
        self.mount = SpatialFrame(self.sub("mount"), system.origin, system.frame)


class Shaft(Component):
    """A rigid shaft on ideal (rigid) bearings: one rotational DOF.

    Parameters
    ----------
    J        : polar moment of inertia about the rotation axis [kg m^2]
    J_t      : transverse moment of inertia [kg m^2], defaults to J/2
    mass     : shaft mass [kg]; only matters once the centre may translate
    position : offset of the shaft centre from the housing origin, as a
               sympy vector in the housing frame (shaft-centre geometry,
               Section 5.1)
    """

    def __init__(self, system, name, J, parent=None, housing=None, position=None,
                 axis: str = "z", mass: float = 0.0, J_t=None, phi0: float = 0.0,
                 w0: float = 0.0):
        super().__init__(system, name, parent)

        host_frame = housing.frame if housing is not None else system.frame
        host_point = housing.point if housing is not None else system.origin

        self.coord = self.coordinate("phi", phi0, w0)
        ax = _axis_vec(host_frame, axis)

        self.frame = host_frame.orientnew(f"f_{len(system.coords)}", "Axis",
                                          (self.coord.q, ax))
        self.frame.set_ang_vel(host_frame, self.coord.u * ax)

        self.centre = host_point if position is None else \
            host_point.locatenew(f"p_{len(system.coords)}", position)
        self.centre.set_vel(system.frame, 0)

        J_sym = self.parameter("J", J)
        Jt = J / 2.0 if J_t is None else J_t
        Jt_sym = self.parameter("J_t", Jt)

        comps = [Jt_sym, Jt_sym, Jt_sym]
        comps[_AXES[axis]] = J_sym
        I = inertia(self.frame, *comps)

        self.body = RigidBody(self.path.replace(".", "_"), self.centre, self.frame,
                              float(mass), (I, self.centre))
        system.add_body(self.body)

        self.flange = RotFlange(self.sub("flange"), self.coord.q, self.coord.u,
                                self.frame, ax, body_frame=self.frame)

        self.output("phi", self.coord.q)
        self.output("w", self.coord.u)
        self.output("rpm", self.coord.u * 60.0 / (2.0 * sp.pi))


class TorsionalSpringDamper(Component):
    """Compliant coaxial coupling: T = c*(phi_a - phi_b) + d*(w_a - w_b)."""

    def __init__(self, system, name, flange_a: RotFlange, flange_b: RotFlange,
                 c: float, d: float = 0.0, parent=None):
        super().__init__(system, name, parent)
        c_s = self.parameter("c", c)
        d_s = self.parameter("d", d)

        dphi = flange_a.angle - flange_b.angle
        dw = flange_a.speed - flange_b.speed
        T = c_s * dphi + d_s * dw

        flange_a.apply_torque(system, -T)
        flange_b.apply_torque(system, T)

        self.output("phi_rel", dphi)
        self.output("torque", T)


class CompliantSpurGearMesh(Component):
    """Linear compliant spur-gear mesh (the CompliantSpurGearMesh of Section 6.2).

    The mesh deflection is measured along the line of action:

        delta = r_a*phi_a + r_b*phi_b - e(t)
        F     = c*delta + d*d(delta)/dt
        T_a   = -r_a*F,   T_b = -r_b*F

    Both pitch radii are positive for an external mesh, which is what produces
    the speed reversal at every stage. The nominal ratio |w_b/w_a| is r_a/r_b,
    i.e. z_a/z_b -- matching the ratio table of Section 5.2.

    ``e(t)`` is static transmission error; zero for baseline validation.
    """

    def __init__(self, system, name, flange_a: RotFlange, flange_b: RotFlange,
                 r_a: float, r_b: float, c: float, d: float = 0.0,
                 transmission_error=None, parent=None):
        super().__init__(system, name, parent)

        ra = self.parameter("r_a", r_a)
        rb = self.parameter("r_b", r_b)
        c_s = self.parameter("c", c)
        d_s = self.parameter("d", d)

        e = sp.Integer(0) if transmission_error is None else transmission_error

        delta = ra * flange_a.angle + rb * flange_b.angle - e
        ddelta = ra * flange_a.speed + rb * flange_b.speed
        F = c_s * delta + d_s * ddelta

        flange_a.apply_torque(system, -ra * F)
        flange_b.apply_torque(system, -rb * F)

        self.nominal_ratio = r_a / r_b

        self.output("deflection", delta)
        self.output("force", F)
        self.output("damping_power", d_s * ddelta ** 2)
        # compatibility residual of Section 5.2 / Appendix A
        self.output("residual", flange_b.speed + (ra / rb) * flange_a.speed)


class TorqueSource(Component):
    """Prescribed torque applied to a flange, reacted by the housing."""

    def __init__(self, system, name, flange: RotFlange, func, parent=None):
        super().__init__(system, name, parent)
        self.sym = system.input(self.sub("torque"), func)
        flange.apply_torque(system, self.sym)
        self.output("torque", self.sym)


class SpeedSensor(Component):
    """Ideal speed sensor. Pure diagnostic: adds no equations."""

    def __init__(self, system, name, flange: RotFlange, parent=None):
        super().__init__(system, name, parent)
        self.output("w", flange.speed)
        self.output("rpm", flange.speed * 60.0 / (2.0 * sp.pi))
