import numpy as np
import pytest

from crankbal import PRESETS, solve, timeseries
from crankbal.presets import inline4_flatplane, paper_inline3, paper_v4, v4


def _bal(res, order, sense):
    for b in res.balancers:
        if b.spec.order == order and b.spec.sense == sense:
            return b
    raise KeyError


# ----------------------------------------------------------------- paper Table 2 / Table 5
def test_paper_inline3_masses():
    res = solve(paper_inline3())
    crank = res.balancers[0]
    # paper: 1.2413 kg per web, here two webs lumped per plane
    for p in crank.planes:
        assert p.mass_g / 2 / 1000 == pytest.approx(1.2413, abs=5e-4)
    for p in _bal(res, 1, -1).planes:
        assert p.mass_g / 1000 == pytest.approx(0.4884, abs=5e-4)
    co1 = [b for b in res.balancers if b.spec.order == 1 and b.spec.sense == 1][1]
    for p in co1.planes:
        assert p.mass_g / 1000 == pytest.approx(0.4884, abs=5e-4)
    for s in (+1, -1):
        for p in _bal(res, 2, s).planes:
            assert p.mass_g / 1000 == pytest.approx(0.0362, abs=5e-4)


def test_paper_v4_crank_counterweight():
    res = solve(paper_v4())
    for p in res.balancers[0].planes:
        assert p.mass_g / 2 / 1000 == pytest.approx(1.700, abs=5e-4)


# ----------------------------------------------------------------- physics checks
def _direct_forces(eng, theta_deg):
    """Brute-force 2nd-order inertia forces/moments straight from cylinder kinematics."""
    th = np.radians(theta_deg)
    r, lam, w2 = eng.r_mm * 1e-3, eng.Lambda, eng.omega**2
    Fx = np.zeros_like(th); Fz = np.zeros_like(th); Mx = np.zeros_like(th); Mz = np.zeros_like(th)
    for c in eng.cylinders:
        psi, beta, y = np.radians(c.pin_deg), np.radians(c.bank_deg), c.y_mm * 1e-3
        phi = th + psi
        ti = phi - beta
        fr = (c.m_rot_g + c.m_throw_g) * 1e-3 * r * w2
        fa = c.m_osc_g * 1e-3 * r * w2 * (np.cos(ti) + lam * np.cos(2 * ti))
        fx = fr * np.sin(phi) + fa * np.sin(beta)
        fz = fr * np.cos(phi) + fa * np.cos(beta)
        Fx += fx; Fz += fz; Mx += y * fz; Mz += -y * fx
    return Fx, Fz, Mx, Mz


@pytest.mark.parametrize("key", list(PRESETS))
def test_phasors_match_direct_kinematics(key):
    eng = PRESETS[key]()
    th = np.linspace(0, 720, 361)
    ts = timeseries(solve(eng), th)["Engine (unbalanced)"]
    for a, b in zip(_direct_forces(eng, th), (ts["Fx"], ts["Fz"], ts["Mx"], ts["Mz"])):
        assert np.allclose(a, b, atol=1e-6 * max(1.0, np.abs(a).max()))


@pytest.mark.parametrize("vee,throws,split", [(90, (0, 180), 0), (65, (0, 180), 0), (90, (0, 180), 70),
                                              (75, (0, 90), 30)])
def test_full_balance_with_all_shafts(vee, throws, split):
    eng = v4("t", vee, throws, split, shafts_enabled=(True, True, True, True), fwd_primary_on_crank=0.5)
    ts = timeseries(solve(eng), np.linspace(0, 720, 181))["Residual"]
    for k in ("Fx", "Fz", "Mx", "Mz"):
        assert np.abs(ts[k]).max() < 1e-6


def test_v4_90_shared_pin_primary_backward_vanishes():
    res = solve(v4("t", 90.0, (0.0, 180.0)))
    assert abs(res.recip1.F[(1, -1)]) < 1e-12 and abs(res.recip1.M[(1, -1)]) < 1e-12


def test_inline4_classic_secondary():
    eng = inline4_flatplane()
    res = solve(eng)
    rs = res.residual
    assert abs(rs.F[(1, 1)]) < 1e-12 and abs(rs.F[(1, -1)]) < 1e-12
    # free 2nd-order force amplitude along z: 4 m Lambda r w^2 (split into two half phasors)
    m = eng.cylinders[0].m_osc_g * 1e-3
    expected = 4 * m * eng.Lambda * eng.r_mm * 1e-3 / 2
    assert abs(rs.F[(2, 1)]) == pytest.approx(expected)


def test_json_roundtrip(tmp_path):
    from crankbal import Engine
    e = paper_v4()
    e.save(tmp_path / "e.json")
    e2 = Engine.load(tmp_path / "e.json")
    assert e2 == e
