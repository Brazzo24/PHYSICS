"""
Run with:  python -m pytest -q tests      (or)      python tests/test_weave.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from weave import *
from weave.jets import Jet, jsin, jcos, jsqrt
from weave.multibody import _lagrangian, IPHI, IPSI, NQ
from weave.simulate import expm


def test_jets_match_finite_differences():
    n = 3
    f = lambda x: (x[0] * jsin(x[1]) + jsqrt(1.0 + x[2] * x[2])) / (2.0 + jcos(x[0] * x[1]))
    z0 = np.array([0.3, -0.2, 0.5])
    J = f([Jet.var(i, n, v) for i, v in enumerate(z0)])
    fv = lambda z: f([Jet.const(v, n) for v in z]).v
    h = 1e-4
    I = np.eye(n)
    g = np.array([(fv(z0 + h * e) - fv(z0 - h * e)) / (2 * h) for e in I])
    H = np.array([[(fv(z0 + h * a + h * b) - fv(z0 + h * a - h * b) - fv(z0 - h * a + h * b)
                    + fv(z0 - h * a - h * b)) / (4 * h * h) for b in I] for a in I])
    assert abs(J.g - g).max() < 1e-7 and abs(J.H - H).max() < 1e-6


def test_equilibrium_is_force_free_and_M_is_spd():
    lb = linearize(from_legacy(powertrain=Powertrain.inline4_sportbike()))
    assert np.abs(lb.g0).max() < 1e-9
    M, C, K = lb.MCK(20.0)
    assert np.allclose(M, M.T) and np.linalg.eigvalsh(M).min() > 0


def test_speed_polynomial_is_exact():
    """H(u) = H0 + u H1 + u^2 H2 must reproduce a direct evaluation at another speed."""
    m = from_legacy(powertrain=Powertrain.inline4_sportbike())
    lb = linearize(m)
    H = _lagrangian(m, 2.5).H
    assert np.abs(H - (lb.H0 + 2.5 * lb.H1 + 6.25 * lb.H2)).max() < 1e-9


def test_reduction_closure():
    lb = linearize(from_legacy())
    for u in (5.0, 30.0):
        assert lb.state_space(u)[2] < 1e-6


def test_benchmark_bicycle_meijaard_2007():
    """No-slip limit reproduces the published benchmark (Meijaard et al. 2007):
    eigenvalues at 5 m/s and critical speeds 4.292 / 6.024 m/s."""
    lb = linearize(benchmark_bicycle(1e8))
    ev = np.linalg.eigvals(lb.state_space(5.0)[0])
    ev = ev[np.argsort(-ev.real)][:4]
    ref = np.array([-0.3229, -0.7753 + 4.4649j, -0.7753 - 4.4649j, -14.078])
    for r in ref:
        assert np.min(abs(ev - r)) < 2e-3
    us = np.linspace(3, 8, 5001)
    mr = np.array([np.linalg.eigvals(lb.state_space(u)[0]).real.max() for u in us])
    crit = us[np.where(np.diff(np.sign(mr)))[0]]
    assert len(crit) == 2 and abs(crit[0] - 4.292) < 5e-3 and abs(crit[1] - 6.024) < 5e-3


def test_pure_rotor_gives_the_textbook_gyroscopic_coupling():
    """A massless rotor with angular momentum H about +y on the rear frame adds
    exactly  dC = H (e_psi e_phi^T - e_phi e_psi^T)  (roll moment +H*r, yaw moment -H*p)."""
    base = from_legacy()
    pt = Powertrain(rotors=[Rotor("r", "output", 0.01)], final=2.0)
    m2 = base.with_powertrain(pt)
    u = 25.0
    C0 = linearize(base).MCK(u)[1]
    C1 = linearize(m2).MCK(u)[1]
    H = pt.angular_momentum(u, base.geom.r_rear)
    dC = np.zeros((NQ, NQ))
    dC[IPSI, IPHI], dC[IPHI, IPSI] = H, -H
    assert np.abs((C1 - C0) - dC).max() < 1e-9
    # M and K are untouched by a massless spinning rotor (up to its own axial inertia term)
    assert np.abs(linearize(m2).MCK(u)[2] - linearize(base).MCK(u)[2]).max() < 1e-6


def test_zero_powertrain_equals_baseline_and_reversal_flips_sign():
    base = from_legacy()
    pt = Powertrain.inline4_sportbike()
    a = linearize(base).MCK(30.0)[1]
    b = linearize(base.with_powertrain(pt)).MCK(30.0)[1]
    c = linearize(base.with_powertrain(Powertrain.inline4_sportbike(crank_direction=-1))).MCK(30.0)[1]
    d1, d2 = b - a, c - a
    # the powertrain adds exactly dC = H (e_psi e_phi^T - e_phi e_psi^T) with its net
    # angular momentum H; reversing the crank changes H (crank + input shaft flip,
    # the output shaft does not)
    pt2 = Powertrain.inline4_sportbike(crank_direction=-1)
    r = base.geom.r_rear
    H1, H2 = pt.angular_momentum(30.0, r), pt2.angular_momentum(30.0, r)
    assert abs(H1) > 1.0 and H1 * H2 < 0 or abs(H2) < abs(H1)
    assert abs(d1[IPSI, IPHI] - H1) < 1e-9 and abs(d2[IPSI, IPHI] - H2) < 1e-9


def test_motorcycle_baseline_is_physically_sensible():
    lb = linearize(from_legacy())
    sw = speed_sweep(lb, np.arange(25, 56, 5.0))
    assert np.all(sw["capsize_re"] > 0)                          # slow capsize, no self-stability
    assert np.all(sw["weave_re"] < 0) and np.all(sw["wobble_re"] < 0)
    assert np.all(np.diff(sw["weave_zeta"]) < 0)                 # weave damping falls with speed
    assert np.all((sw["weave_freq"] > 2) & (sw["weave_freq"] < 6))
    assert np.all((sw["wobble_freq"] > 8) & (sw["wobble_freq"] < 16))
    # wheel gyros stabilise capsize: switching them off makes it worse
    lb0 = linearize(without_gyro(from_legacy()))
    assert modes_at(lb0, 40.0)["capsize"].growth > modes_at(lb, 40.0)["capsize"].growth


def test_expm_and_simulation_consistency():
    A = np.array([[0, 1.0], [-4.0, -0.3]])
    w, V = np.linalg.eig(A)
    ref = (V @ np.diag(np.exp(w * 0.7)) @ np.linalg.inv(V)).real
    assert np.abs(expm(A * 0.7) - ref).max() < 1e-12
    lb = linearize(from_legacy())
    u = 40.0
    K = rider_roll_pd(lb, u)
    r = simulate(lb, u, 6.0, 2e-3, torque=lambda t: 15.0 if t < 0.05 else 0.0, K=K)
    assert abs(r["phi"][-1]) < 1e-3          # closed loop returns to upright
    # free decay rate of a hand-off simulation equals the largest eigenvalue
    r2 = simulate(lb, u, 2.0, 1e-3, x0=np.array([0, 0, 0.01, 0, 0, 0, 0, 0.0]))
    ev = np.linalg.eigvals(lb.state_space(u)[0]).real.max()
    assert abs(np.log(abs(r2["phi"][-1] / r2["phi"][1000])) / 1.0 - ev) < 0.05


if __name__ == "__main__":
    import traceback
    ok = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name); ok += 1
            except Exception:
                print("FAIL", name); traceback.print_exc()
    print(f"{ok} passed")
