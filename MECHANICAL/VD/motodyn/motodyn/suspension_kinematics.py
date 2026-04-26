"""
Suspension kinematics (quasi-static, side-view plane).

What lives here
---------------
* Instant-centre locations for the rear swingarm and front end.
* Anti-squat ratio (chain-force + swingarm-axis contributions).
* Anti-lift at the front (equivalent to anti-squat under drive).
* Anti-dive at the front (telescopic & non-telescopic).
* Anti-rise at the rear under braking.
* Mozzi axis (instantaneous screw axis of the chassis in roll + yaw)
  — a 3D concept; here we provide the *side-view* instantaneous axis of
  the sprung mass and the full spatial form for combined roll/yaw.

Core definitions
----------------
Anti-squat (%):
    AS = 100 * tan(β) / tan(γ)
where γ is the angle from the rear contact patch to the CoG (side view)
and β is the angle from the rear contact patch to the effective "force
line" of the rear suspension under drive. For chain drive the relevant
line passes through the intersection of the chain line and the swingarm
line, referenced back to the rear contact patch.

Anti-dive (%):
    AD = 100 * tan(δ) / tan(γ_f)
where γ_f is the angle from the front contact patch to the CoG and δ is
the angle from the front contact patch to the line defined by the
caliper-mounted brake reaction geometry (for telescopic forks the
reaction line is purely along the fork axis → anti-dive is a function
of caster only if the caliper is fork-mounted; hub-centre geometries use
the front IC).

See Cossalter §3.5 and Foale ch. 5 for full derivations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

from ..parameters import Motorcycle


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SideViewIC:
    """An instant centre in the side-view plane."""
    x: float     # m, forward of rear contact patch
    z: float     # m, above ground


def rear_instant_centre(bike: Motorcycle) -> SideViewIC:
    """
    Rear-wheel IC in the side-view plane.

    For a single-pivot swingarm the IC coincides with the swingarm pivot.
    For a linkage bike pass `suspension.rear_ic` explicitly.
    """
    susp = bike.suspension
    if susp.rear_ic is not None:
        return SideViewIC(*susp.rear_ic)
    return SideViewIC(*susp.swingarm_pivot)


def front_instant_centre(bike: Motorcycle) -> SideViewIC:
    """
    Front-end IC in the side-view plane.

    Telescopic fork: the IC is at infinity along the fork axis, which for
    small-angle kinematics we represent by a point far along the steering
    axis direction. Hub-centre / Hossack / girder designs must supply an
    explicit `front_ic`.
    """
    susp = bike.suspension
    if susp.front_ic is not None:
        return SideViewIC(*susp.front_ic)

    # Telescopic: place a "far" point along the steering axis for ratio math.
    p = bike.geometry.wheelbase
    eps = bike.geometry.caster_angle
    # Front contact patch at x = p; steering axis inclined at eps from vertical.
    # Use a large offset along the axis so ratios stay numerically stable.
    L = 50.0  # metres — effectively infinity for our ratios
    x = p - L * math.sin(eps)
    z = L * math.cos(eps)
    return SideViewIC(x=x, z=z)


# --------------------------------------------------------------------------- #
# Anti-squat
# --------------------------------------------------------------------------- #

def chain_force_line_intersection(bike: Motorcycle) -> SideViewIC:
    """
    Intersection of the upper chain run with the line through the swingarm
    pivot that the rear wheel traces.

    For a chain-driven bike the net longitudinal force on the sprung mass
    passes through this intersection point under steady drive. This is
    the point whose side-view angle to the rear contact patch defines β.
    """
    susp = bike.suspension
    geom = bike.geometry

    p = geom.wheelbase
    R_r = geom.rear_wheel_radius
    rear_wheel_centre = (p, R_r)

    cs = susp.sprocket_center                            # countershaft sprocket (x, z)
    sa = susp.swingarm_pivot                             # (x, z)
    r_cs = susp.countershaft_sprocket_radius
    r_rs = susp.rear_sprocket_radius

    # Chain line: upper external tangent between countershaft sprocket and rear
    # sprocket (concentric with rear wheel). For two circles with centres C1, C2
    # and radii r1, r2, the upper tangent has slope found from the angle θ where
    # sin(θ) = (r2 - r1) / d and the tangent is perpendicular to the offset by θ.
    if susp.chain_angle_override is not None:
        # User-specified chain angle (rad, above horizontal, positive = up toward front)
        ang = susp.chain_angle_override
        m_chain = math.tan(ang)
        # Anchor chain-line at the upper tangent point on the countershaft sprocket
        chain_anchor_x = cs[0] - r_cs * math.sin(ang)
        chain_anchor_z = cs[1] + r_cs * math.cos(ang)
    else:
        dx_c = rear_wheel_centre[0] - cs[0]
        dz_c = rear_wheel_centre[1] - cs[1]
        d = math.hypot(dx_c, dz_c)
        # Angle of the centres line above horizontal
        phi = math.atan2(dz_c, dx_c)
        # Rotation to upper external tangent direction
        if d <= abs(r_rs - r_cs):
            raise ValueError("Sprocket geometry invalid: circles overlap.")
        theta = math.asin((r_rs - r_cs) / d)
        # Upper tangent direction is the centres direction rotated by +theta
        ang = phi + theta
        m_chain = math.tan(ang)
        # Tangent point on countershaft sprocket: perpendicular to tangent,
        # offset from cs by r_cs in the direction normal-to-tangent (upward-ish).
        chain_anchor_x = cs[0] - r_cs * math.sin(ang)
        chain_anchor_z = cs[1] + r_cs * math.cos(ang)

    # Swingarm line: from pivot through rear wheel centre
    dx = rear_wheel_centre[0] - sa[0]
    dz = rear_wheel_centre[1] - sa[1]
    m_sa = dz / dx if abs(dx) > 1e-9 else float("inf")

    if math.isinf(m_chain) or math.isinf(m_sa):
        raise ValueError("Degenerate chain / swingarm geometry; check sprocket and pivot.")
    if abs(m_chain - m_sa) < 1e-9:
        raise ValueError("Chain line is parallel to swingarm line; anti-squat undefined.")

    # Solve intersection of:
    #   z = chain_anchor_z + m_chain * (x - chain_anchor_x)
    #   z = sa[1] + m_sa * (x - sa[0])
    x_int = (sa[1] - chain_anchor_z + m_chain * chain_anchor_x - m_sa * sa[0]) / (m_chain - m_sa)
    z_int = chain_anchor_z + m_chain * (x_int - chain_anchor_x)
    return SideViewIC(x=x_int, z=z_int)


def anti_squat_ratio(bike: Motorcycle) -> float:
    """
    Anti-squat as a fraction (1.0 = 100 %).

    AS = tan(β) / tan(γ)
        γ = atan2(h_cog, p - b_rear_to_cog)
            where the argument is horizontal distance from rear contact
            patch to CoG, and vertical distance is CoG height.
        β = atan2(z_int, x_int) with (x_int, z_int) relative to the
            rear contact patch.
    """
    geom = bike.geometry
    m = bike.mass
    p = geom.wheelbase

    # Horizontal distance from rear contact patch to CoG projection = b
    dx_cog = m.b
    gamma = math.atan2(m.h_cog, dx_cog)

    ic = chain_force_line_intersection(bike)
    beta = math.atan2(ic.z, ic.x)

    if math.tan(gamma) == 0:
        return float("inf")
    return math.tan(beta) / math.tan(gamma)


# --------------------------------------------------------------------------- #
# Anti-dive / anti-lift
# --------------------------------------------------------------------------- #

def anti_dive_ratio(bike: Motorcycle) -> float:
    """
    Anti-dive as a fraction, for a telescopic fork with fork-mounted
    caliper.

    For a telescopic fork the brake reaction passes along the fork leg,
    i.e. along the steering axis. The anti-dive is then purely a function
    of caster angle and the fraction of total braking at the front.

        AD_front = (front_brake_fraction) * tan(ε) * p / h_cog

    For a hub-centre or leading-link front the brake reaction line passes
    through `front_ic`; the user should supply that IC and we compute the
    ratio geometrically.
    """
    geom = bike.geometry
    brakes = bike.brakes
    m = bike.mass

    if bike.suspension.front_ic is None:
        # Telescopic form
        return brakes.front_brake_fraction * math.tan(geom.caster_angle) * geom.wheelbase / m.h_cog

    # Non-telescopic: angle from front contact patch to front IC
    p = geom.wheelbase
    ic = bike.suspension.front_ic
    dx = ic[0] - p                  # front contact patch at x = p
    dz = ic[1]
    gamma_f = math.atan2(m.h_cog, p - m.b)     # angle from front contact to CoG
    delta = math.atan2(dz, -dx) if dx != 0 else math.pi / 2
    return brakes.front_brake_fraction * math.tan(delta) / math.tan(gamma_f)


def anti_rise_ratio(bike: Motorcycle) -> float:
    """
    Anti-rise at the rear under braking.

    The rear wheel unloads under braking; the swingarm geometry resists
    this unloading by an amount governed by the angle from the rear
    contact patch to the rear IC, scaled by the rear brake fraction.
    """
    brakes = bike.brakes
    m = bike.mass
    ic = rear_instant_centre(bike)

    gamma = math.atan2(m.h_cog, m.b)
    beta = math.atan2(ic.z, ic.x)
    rear_brake_fraction = 1.0 - brakes.front_brake_fraction
    if math.tan(gamma) == 0:
        return float("inf")
    return rear_brake_fraction * math.tan(beta) / math.tan(gamma)


# --------------------------------------------------------------------------- #
# Load-path split (suspension vs. axle-path)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LoadPathSplit:
    """
    Decomposition of longitudinal load transfer into the component that
    goes through the suspension springs (causing pitch / squat) and the
    component that passes through the rigid linkage (causing no
    compression).

    The sum always equals the total load transfer; anti-squat /
    anti-dive only redistribute between the two columns.
    """
    delta_N_total: float        # N (rear gains this, front loses this under accel)
    through_spring_rear: float  # N
    through_linkage_rear: float # N
    through_spring_front: float # N
    through_linkage_front: float# N


def effective_load_paths(bike: Motorcycle, a_x: float) -> LoadPathSplit:
    """
    Split the total load transfer according to the anti-squat and
    anti-dive ratios (acceleration case uses AS; braking uses AD/AR).
    """
    m = bike.mass.mass
    h = bike.mass.h_cog
    p = bike.geometry.wheelbase
    delta = m * a_x * h / p                       # + under accel, - under braking

    if a_x >= 0:
        AS = anti_squat_ratio(bike)
        # On the rear: fraction (1 - AS) goes through the spring, AS through linkage
        thru_link_r = AS * delta
        thru_spr_r = (1.0 - AS) * delta
        # Front: nothing special from chain, so the front unloading goes entirely through the spring
        thru_spr_f = -delta
        thru_link_f = 0.0
    else:
        AD = anti_dive_ratio(bike)
        AR = anti_rise_ratio(bike)
        # Front loads up by -delta (delta is negative here, so -delta > 0)
        thru_link_f = AD * (-delta)
        thru_spr_f = (1.0 - AD) * (-delta)
        # Rear unloads by -delta
        thru_link_r = -AR * (-delta)
        thru_spr_r = -(1.0 - AR) * (-delta)

    return LoadPathSplit(
        delta_N_total=delta,
        through_spring_rear=thru_spr_r,
        through_linkage_rear=thru_link_r,
        through_spring_front=thru_spr_f,
        through_linkage_front=thru_link_f,
    )


# --------------------------------------------------------------------------- #
# Mozzi axis (instantaneous screw axis of the sprung mass)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MozziAxis:
    """
    Instantaneous screw axis of the chassis (side-view projection when
    yaw = 0; full 3D when yaw rate and roll rate are both nonzero).

    Parametrised as point-on-axis + unit direction vector.
    """
    point: tuple[float, float, float]     # metres
    direction: tuple[float, float, float] # unit vector
    pitch: float                          # translation per radian (m/rad); 0 for a pure rotation axis


def mozzi_axis(
    bike: Motorcycle,
    roll_rate: float = 0.0,
    yaw_rate: float = 0.0,
    v: float = 0.0,
) -> MozziAxis:
    """
    Compute the instantaneous screw axis of the sprung mass.

    Inputs
    ------
    roll_rate : rad/s, about the longitudinal axis
    yaw_rate  : rad/s, about the vertical axis
    v         : forward speed (m/s). At v = 0 the axis degenerates to a
                pure roll axis along x through the roll-axis height.

    Approach
    --------
    For a bike the roll axis is commonly defined as the line joining the
    front and rear roll centres (projections of the suspension ICs onto
    the lateral plane). When yaw is also present, the true instantaneous
    screw axis (Mozzi axis) is the combined axis of the roll-rate and
    yaw-rate vectors — found by the classical formula:

        ω = ω_roll + ω_yaw,  axis direction = ω / |ω|
        point = (ω × v_ref) / |ω|^2, using any known point's velocity v_ref.

    We take the reference point as the CoG with velocity (v, 0, 0) in the
    body frame; the axis is expressed in the body frame.
    """
    # Roll axis intersection height: use front + rear IC heights averaged along wheelbase.
    rear_ic = rear_instant_centre(bike)
    p = bike.geometry.wheelbase
    # Rear roll centre height ≈ rear_ic.z; front roll centre heights for a
    # telescopic fork are at ground level (IC at infinity along fork axis).
    if bike.suspension.front_ic is None:
        h_front_rc = 0.0
    else:
        h_front_rc = bike.suspension.front_ic[1]

    # Roll axis is a line from (0, 0, rear_ic.z) to (p, 0, h_front_rc).
    axis_dir_roll = (p, 0.0, h_front_rc - rear_ic.z)
    n = math.sqrt(sum(c * c for c in axis_dir_roll))
    axis_dir_roll = tuple(c / n for c in axis_dir_roll)  # type: ignore

    omega_roll = tuple(roll_rate * c for c in axis_dir_roll)
    omega_yaw = (0.0, 0.0, yaw_rate)
    omega = tuple(a + b for a, b in zip(omega_roll, omega_yaw))
    omega_mag = math.sqrt(sum(c * c for c in omega))

    if omega_mag < 1e-12:
        # Degenerate — just return the static roll axis
        return MozziAxis(
            point=(0.0, 0.0, rear_ic.z),
            direction=axis_dir_roll,  # type: ignore
            pitch=0.0,
        )

    direction = tuple(c / omega_mag for c in omega)

    # Reference point: CoG at (p - b_to_front, 0, h_cog). Use b as rear->cog.
    cog = (bike.mass.b, 0.0, bike.mass.h_cog)
    v_ref = (v, 0.0, 0.0)
    # point_on_axis = (ω × v_ref)/|ω|^2 + cog (shift so axis passes near the CoG region)
    cross = (
        omega[1] * v_ref[2] - omega[2] * v_ref[1],
        omega[2] * v_ref[0] - omega[0] * v_ref[2],
        omega[0] * v_ref[1] - omega[1] * v_ref[0],
    )
    point = tuple(cog[i] + cross[i] / (omega_mag ** 2) for i in range(3))

    # Screw pitch: (ω · v_ref) / |ω|^2
    pitch = sum(omega[i] * v_ref[i] for i in range(3)) / (omega_mag ** 2)

    return MozziAxis(point=point, direction=direction, pitch=pitch)  # type: ignore
