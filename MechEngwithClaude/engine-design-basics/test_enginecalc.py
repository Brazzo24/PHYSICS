"""Consistency checks for enginecalc.  Run:  python -m pytest -q"""
import numpy as np
import enginecalc as ec

eng = ec.Engine()
th = np.linspace(-2 * np.pi, 2 * np.pi, 14401)


def test_kinematic_derivatives():
    x, v, a = ec.piston_kinematics(eng, th, 1.0)
    i = slice(1, -1)                      # central differences only
    assert np.max(abs(np.gradient(x, th) - v)[i]) < 1e-4 * eng.r
    assert np.max(abs(np.gradient(v, th) - a)[i]) < 1e-4 * eng.r


def test_tdc_bdc_acceleration():
    _, _, a = ec.piston_kinematics(eng, np.array([0.0, np.pi]), 1.0)
    assert np.isclose(a[0], eng.r * (1 + eng.lam))
    assert np.isclose(a[1], -eng.r * (1 - eng.lam))


def test_rod_angular_kinematics():
    beta = ec.rod_angle(eng, th)
    bd, bdd = ec.rod_angular_kinematics(eng, th, 1.0)
    i = slice(1, -1)
    assert np.max(abs(np.gradient(beta, th) - bd)[i]) < 1e-5
    assert np.max(abs(np.gradient(bd, th) - bdd)[i]) < 1e-5


def test_volume_limits():
    V = ec.cylinder_volume(eng, np.array([0.0, np.pi]))
    assert np.isclose(V[1] / V[0], eng.compression_ratio)


def test_energy_balance_torque():
    """Mean torque x 4 pi must equal the indicated work (inertia averages out)."""
    t, p, ex = ec.pressure_trace(eng)
    for rpm in (3000, 15000):
        f = ec.crank_forces(eng, t, p, ec.rpm_to_omega(rpm))
        T_mean = ec.trapz(f["T"], t) / (4 * np.pi)
        assert np.isclose(T_mean * 4 * np.pi, ex["W_ind"], rtol=1e-3)


def test_motored_trace_is_polytropic():
    c = ec.Combustion(eta_q=0.0)
    t, p, _ = ec.pressure_trace(eng, c)
    i0 = np.argmin(abs(t))
    assert np.isclose(p[i0], c.p_ivc * eng.compression_ratio ** c.gamma, rtol=1e-3)


def test_wiebe_complete():
    xb, _ = ec.wiebe(np.radians(30.0), np.radians(-20.0), np.radians(50.0), a=6.9)
    assert xb > 0.998
