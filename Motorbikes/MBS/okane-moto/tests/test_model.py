"""Verification tests.  Run with  pytest -q  from the repository root."""
import numpy as np
import pytest

from okane import pendulum as P
from okane import frames
from okane.params import cbr600rr_estimate
from okane.kinematics import Kinematics, make_q
from okane.model import Motorcycle, make_inputs, rk4, IIN
from okane.forces import ForceSet, Drag
from okane.analysis import static_trim, steady_turn

p, design = cbr600rr_estimate()
Q = make_q(q1=0.07, q2=0.12, q3=0.005, q4=0.006, q9=0.5, q10=0.3, q11=0.05,
           q7=3.0, q8=-2.0)
U = np.array([0.1, -0.2, 0.05, 0.03, -100, -95, 28, -9, 0.4, 0.3, -0.5])


# ---- chapter 3 -----------------------------------------------------------
@pytest.mark.parametrize("method", ["newton", "lagrange", "kane"])
def test_pendulum_methods_agree_with_eq_3_7(method):
    eqs = getattr(P, method)()
    eqs = eqs[0] if method == "kane" else eqs
    assert P.equivalent(eqs, P.thesis_eom())


# ---- kinematics ------------------------------------------------------------
def test_design_configuration():
    k = Kinematics(p).config(make_q(q1=design["q1"], q2=design["q2"]))
    assert abs(k["mu"]) < 1e-12
    assert k["wheelbase"] == pytest.approx(design["wheelbase"], abs=1e-9)
    assert k["trail"] == pytest.approx(design["trail"], abs=1e-9)


def test_contact_points_on_ground():
    k = Kinematics(p).config(Q)
    assert abs(k["r_S"][1]) < 1e-12 and abs(k["r_T"][1]) < 1e-12


def test_camber_and_kinematic_steer_formulas_are_exact():
    k = Kinematics(p).config(Q)
    assert k["gamma1"] == pytest.approx(frames.camber_thesis(Q[8], Q[10], p.eps, k["mu"]))
    assert k["delta_k"] == pytest.approx(frames.kin_steer_thesis(Q[8], Q[10], p.eps, k["mu"]))


def test_printed_H_coefficients_are_gradient_of_4_17():
    # the printed coefficients are expanded about q3 = q4 = 0
    K = Kinematics(p)
    Q0 = Q.copy(); Q0[2] = Q0[3] = 0.0
    H = K.pitch(Q0, model="thesis")[1]
    pr = K.H_printed(Q0)
    for name, i in [("H1", 0), ("H2", 1), ("H3", 2), ("H4", 3), ("H9", 8), ("H11", 10)]:
        assert pr[name] == pytest.approx(H[i], rel=1e-10, abs=1e-12)
    assert pr["H"] == pytest.approx(K.mu_thesis(Q0))


def test_partial_velocities_by_finite_differences():
    m = Motorcycle(p)
    k = m.config(Q, U)
    h = 1e-6
    kp, km = m.config(Q + h * U), m.config(Q - h * U)
    for pt in "ABCDFGPHQR":
        assert np.allclose((kp["r_" + pt] - km["r_" + pt]) / (2 * h), k["v_" + pt], atol=1e-6)
    kp, km = m.config(Q + h * U, U), m.config(Q - h * U, U)
    for pt in "ABCDFG":
        assert np.allclose((kp["v_" + pt] - km["v_" + pt]) / (2 * h), k["ab_" + pt], atol=1e-4)


# ---- dynamics --------------------------------------------------------------
def test_mass_matrix_spd():
    m = Motorcycle(p)
    K = m.mass_matrix(m.config(Q, U))
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.linalg.eigvalsh(K).min() > 0


def test_energy_conservation():
    m = Motorcycle(p)
    kf, ks, kt = 20e3, 30e3, 180e3

    def inp(t, k):
        q = k["q"]
        return make_inputs(g=9.81, Ffork=kf * (0.12 - q[0]), Tsay=ks * (0.2 - q[1]),
                           Ffwz=kt * q[2], Frwz=kt * q[3])

    def E(x):
        k = m.config(x[:11], x[11:]); q = x[:11]
        return (m.kinetic_energy(k) + m.potential_energy(k) + 0.5 * kf * (0.12 - q[0])**2
                + 0.5 * ks * (0.2 - q[1])**2 + 0.5 * kt * (q[2]**2 + q[3]**2))
    u0 = np.zeros(11); u0[[4, 5]] = -60; u0[6] = 19
    u0[8], u0[9], u0[10], u0[0] = 0.8, 0.5, -0.6, 0.1
    x0 = np.concatenate([make_q(q1=0.08, q2=0.15, q3=0.004, q4=0.006, q9=0.2, q11=0.03), u0])
    _, X = rk4(lambda t, x: m.rhs(t, x, inp), 0, x0, 2.5e-4, 400)
    assert abs(E(X[-1]) - E(X[0])) < 1e-5 * 1.0       # of ~5e4 J


def test_static_wheel_loads_match_statics():
    m = Motorcycle(p); F = ForceSet.default(p)
    q, uin, res = static_trim(m, F)
    k = m.config(q)
    cg = m.kin.mass_centre(k)
    Fzf = p.total_mass * p.g * (cg[0] - k["r_T"][0]) / (k["r_S"][0] - k["r_T"][0])
    assert uin[IIN["Ffwz"]] == pytest.approx(Fzf, rel=1e-8)
    assert uin[IIN["Ffwz"]] + uin[IIN["Frwz"]] == pytest.approx(p.total_mass * p.g, rel=1e-8)
    assert np.abs(res).max() < 1e-6


def test_steady_turn_force_balance():
    m = Motorcycle(p); F = ForceSet.default(p)
    V, R = 30.0, 80.0
    s = steady_turn(m, F, V, R, "right")
    assert np.abs(s["residual"]).max() < 1e-9
    k, uin = s["k"], s["u_in"]
    w = s["yaw_rate"]
    cg = m.kin.mass_centre(k)
    centre = np.array([0.0, 0.0, -V / w])              # turn centre (psi = 0)
    a = -w**2 * (cg - centre) * np.array([1, 0, 1])    # centripetal acc. of c.g.
    Fsum = (uin[IIN["Ffwy"]] * k["R_H"][:, 2] + uin[IIN["Frwy"]] * k["R_M"][:, 2]
            + uin[IIN["Frwx"]] * k["R_M"][:, 0])
    assert np.allclose(Fsum[[0, 2]], p.total_mass * a[[0, 2]], rtol=1e-6, atol=1e-6)
    assert uin[IIN["Ffwz"]] + uin[IIN["Frwz"]] == pytest.approx(p.total_mass * p.g)
    assert s["q"][8] > np.arctan(V**2 / (p.g * R))     # tyre width: more lean


def test_state_space_reproduces_simplified_accelerations():
    m = Motorcycle(p)
    uin = make_inputs(g=9.81, Ffork=1000, Tsay=600, Ffwz=1300, Frwz=1300, Ffwy=500)
    ss = m.state_space(Q, U)
    x = np.empty(22); x[0::2], x[1::2] = U, Q
    w = np.array([U[8]**2, U[8] * U[9], U[8] * U[10], U[9] * U[10]])
    ud = (ss["A"] @ x + ss["B"] @ uin + ss["R"] @ w)[0::2]
    assert np.allclose(ud, m.accelerations_thesis(Q, U, uin), atol=1e-8)


def test_drag_numbers_of_section_4_5_5():
    d = Drag()
    assert d.CdA == pytest.approx(0.4078, abs=2e-4)
    assert d.mechanical(1.0) - d.mechanical(0.0) == pytest.approx(-0.7857, abs=1e-4)
    assert d.mechanical(0.0) == pytest.approx(-51.2, abs=0.02)
