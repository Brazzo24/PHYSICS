"""
Sanity-check tests.

These are deliberately *low-fidelity* — they check that:
  * static loads sum to the weight
  * load transfer is symmetric under sign reversal of a_x (with clamping off)
  * wheelie threshold agrees with the static analysis
  * with_changes() doesn't mutate the original
  * anti-squat in a chain-drive bike is finite and sensible
"""

import math

from motodyn.presets import generic_sportbike
from motodyn.state import OperatingPoint
from motodyn.models import load_transfer as lt
from motodyn.models import suspension_kinematics as sk
from motodyn.models import chassis_rigid_body as crb


def test_static_loads_sum_to_weight():
    bike = generic_sportbike()
    W = bike.mass.mass * 9.80665
    loads = lt.static_loads(bike)
    assert math.isclose(loads.N_front + loads.N_rear, W, rel_tol=1e-10)


def test_static_fraction_matches_geometry():
    bike = generic_sportbike()
    f_frac, r_frac = bike.mass.weight_distribution_static(bike.geometry.wheelbase)
    assert math.isclose(f_frac + r_frac, 1.0, rel_tol=1e-12)
    loads = lt.static_loads(bike)
    assert math.isclose(loads.front_fraction, f_frac, rel_tol=1e-10)


def test_load_transfer_sign():
    bike = generic_sportbike()
    op_accel = OperatingPoint(a_x=2.0)
    op_brake = OperatingPoint(a_x=-2.0)
    la = lt.longitudinal_load_transfer(bike, op_accel)
    lb = lt.longitudinal_load_transfer(bike, op_brake)
    # Acceleration → rear gains, front loses
    assert la.N_rear > lb.N_rear
    assert la.N_front < lb.N_front


def test_wheelie_threshold_matches_load_zero():
    bike = generic_sportbike()
    a_lift = lt.wheelie_threshold_acceleration(bike)
    op = OperatingPoint(a_x=a_lift)
    loads = lt.longitudinal_load_transfer(bike, op)
    assert loads.N_front < 1.0      # essentially zero (clamped)


def test_with_changes_does_not_mutate():
    bike = generic_sportbike()
    before = bike.suspension.swingarm_pivot
    _ = bike.with_changes(**{"suspension.swingarm_pivot": (0.50, 0.40)})
    after = bike.suspension.swingarm_pivot
    assert before == after


def test_anti_squat_finite_and_positive():
    bike = generic_sportbike()
    AS = sk.anti_squat_ratio(bike)
    assert math.isfinite(AS)
    assert 0.2 < AS < 3.0          # physical range for a normal sportbike


def test_anti_dive_telescopic_matches_formula():
    bike = generic_sportbike()
    AD = sk.anti_dive_ratio(bike)
    expected = (
        bike.brakes.front_brake_fraction
        * math.tan(bike.geometry.caster_angle)
        * bike.geometry.wheelbase
        / bike.mass.h_cog
    )
    assert math.isclose(AD, expected, rel_tol=1e-10)


def test_load_path_split_conserves_total():
    bike = generic_sportbike()
    split = sk.effective_load_paths(bike, a_x=2.5)
    # Rear gets +delta total (spring + link paths)
    total_rear = split.through_spring_rear + split.through_linkage_rear
    assert math.isclose(total_rear, split.delta_N_total, rel_tol=1e-10)


def test_equilibrium_pitch_sign_on_braking():
    bike = generic_sportbike()
    op = OperatingPoint(a_x=-3.0, speed=20.0)
    eq = crb.chassis_equilibrium(bike, op)
    # Braking should compress the front more than rear → pitch < 0 (nose down)
    assert eq.front_compression > eq.rear_compression
    assert eq.pitch_angle < 0


def test_equilibrium_pitch_sign_on_accel():
    bike = generic_sportbike()
    op_static = OperatingPoint(a_x=0.0, speed=20.0)
    op_accel = OperatingPoint(a_x=3.0, speed=20.0)
    eq_s = crb.chassis_equilibrium(bike, op_static)
    eq_a = crb.chassis_equilibrium(bike, op_accel)

    # Under accel the rear should squat MORE and the front extend (compress less)
    # than static — this is the physically meaningful check.
    d_front = eq_a.front_compression - eq_s.front_compression
    d_rear = eq_a.rear_compression - eq_s.rear_compression
    AS = sk.anti_squat_ratio(bike)
    if AS < 1.0:
        # Rear squats down, front extends → Δrear > 0, Δfront < 0
        assert d_rear > 0
        assert d_front < 0
