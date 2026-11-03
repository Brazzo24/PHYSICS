"""consistency checks for geardyn -- run with  python -m pytest -q"""
import numpy as np
import geardyn as gd

gp = gd.GearPair()


def torsional_model(gp, contact="linear", stiffness="const"):
    m = gd.Model()
    m.add_dof("th1"); m.add_dof("th2")
    m.add_mass("th1", gp.J1); m.add_mass("th2", gp.J2)
    m.add_mesh(gp, "th1", "th2", contact=contact, stiffness=stiffness)
    return m


def test_geometry():
    assert abs(gp.eps_alpha - 1.675) < 2e-3
    s = np.linspace(0, 1, 20001)[:-1]
    assert abs(gp.k_of_s(s).mean() / gp.k_mesh - 1) < 1e-6
    h = 1e-6
    for s0 in [0.1, 0.33, 0.66, 0.9]:
        fd = (gp.k_of_s(s0 + h) - gp.k_of_s(s0 - h)) / (2 * h)
        assert abs(fd - gp.dk_ds(s0)) < 1e-5 * gp.k_mesh / gp.tv_smooth


def test_rigid_pair_consistency():
    a1, a2, F = gd.rigid_pair(gp, 30.0, 20.0)
    assert np.isclose((30.0 - gp.J1 * a1) / gp.rb1, F)
    assert np.isclose(a1 / a2, gp.i)


def test_bearing_reactions_equilibrium():
    Fv = np.array([300.0, -1200.0, 450.0]); P = (0.03, 0.0, 0.04)
    RA, RB = gd.shaft_bearing_reactions(Fv, P, 0.0, 0.1)
    assert np.allclose(RA + RB + Fv, 0)
    rP = np.array([P[0], P[1], P[2]]); rB = np.array([0, 0, 0.1])
    Mo = np.cross(rP, Fv) + np.cross(rB, RB)
    assert np.allclose(Mo[:2], 0)


def test_free_pair_frequency_and_energy():
    m = torsional_model(gp)
    f, _ = m.linear_modes()
    assert f[0] < 1e-3 and abs(f[1] / gp.f_n - 1) < 1e-9
    q0 = np.array([10e-6 / gp.rb1, 0.0])
    r = m.simulate(2e-3, q0=q0, n_out=2000)
    E0 = r.T[0] + r.V[0]
    assert np.max(np.abs(r.energy_residual)) < 1e-6 * E0
    p = gp.J1 * r.vel("th1") / gp.rb1 + gp.J2 * r.vel("th2") / gp.rb2
    assert np.max(np.abs(p)) < 1e-9


def test_tv_energy_with_backlash():
    g = gp.with_(backlash=40e-6)
    m = torsional_model(g, contact="backlash", stiffness="tv")
    q0 = np.array([0.0, 0.0]); v0 = np.array([2000 / 60 * 2 * np.pi * 0 + 50.0, 0.0])
    r = m.simulate(5e-3, q0=q0, v0=v0, n_out=3000)
    E0 = r.T[0]
    assert np.max(np.abs(r.energy_residual)) < 1e-5 * E0


def test_lateral_static_bearing_force():
    m = gd.Model()
    for d in ["x1", "y1", "th1", "x2", "y2", "th2"]:
        m.add_dof(d)
    for d, v in zip(["x1", "y1", "th1", "x2", "y2", "th2"], [gp.m1, gp.m1, gp.J1, gp.m2, gp.m2, gp.J2]):
        m.add_mass(d, v)
    m.add_bearing("B1", "x1", "y1", 2e8); m.add_bearing("B2", "x2", "y2", 2e8)
    m.add_mesh(gp, "th1", "th2", x1="x1", y1="y1", x2="x2", y2="y2", contact="backlash")
    m.fix("th2")
    T = 50.0
    Q = np.zeros(m.n); Q[m.index("th1")] = T
    q = m.static(Q)
    F = T / gp.rb1
    R1 = -2e8 * np.array([q[0], q[1]])
    assert np.allclose(R1, F * np.array([np.sin(gp.alpha), np.cos(gp.alpha)]), rtol=1e-9)


def test_shaft_fe_simply_supported():
    m = gd.Model()
    L, d = 0.2, 0.02
    sh = gd.add_shaft(m, "s", L, d, n_el=10)
    k0, kN = 0, len(sh["z"]) - 1
    for c in ["x", "y"]:
        m.add_spring(sh["dof"](k0, c), None, 1e14); m.add_spring(sh["dof"](kN, c), None, 1e14)
    m.add_spring(sh["dof"](k0, "th"), None, 1e14)
    Q = np.zeros(m.n); Q[m.index(sh["dof"](5, "y"))] = 1000.0
    M, K, _ = m.matrices()
    q = np.linalg.solve(K, Q)
    EI = 2.1e11 * np.pi * d**4 / 64
    assert abs(q[m.index(sh["dof"](5, "y"))] / (1000 * L**3 / (48 * EI)) - 1) < 1e-6
    f, _ = m.linear_modes(include_mesh=False)
    f_bend = (np.pi / L)**2 * np.sqrt(EI / (7850 * np.pi * d**2 / 4)) / (2 * np.pi)
    fb = np.sort(f[f > 1])[0]
    assert abs(fb / f_bend - 1) < 1e-3


def test_modal_reduction_matches_full():
    m = torsional_model(gp)
    q0 = np.array([10e-6 / gp.rb1, 0.0])
    r1 = m.simulate(1e-3, q0=q0, n_out=500)
    r2 = m.simulate(1e-3, q0=q0, n_out=500, f_cut=1e6)
    assert np.allclose(r1.mesh[0]["F"], r2.mesh[0]["F"], atol=1e-6 * gp.k_mesh * 10e-6)


def test_dte_static():
    r = gd.simulate_dte(gp.with_(zeta=0.3), 3000, 1000.0, stiffness="const", cycles=60)
    assert abs(r["x"][-1] / (1000.0 / gp.k_mesh) - 1) < 1e-6


def test_reduction_with_prescribed_motion():
    """constraint modes make the reduced model follow a large prescribed rotation"""
    g = gp.with_(backlash=40e-6)
    m = gd.Model()
    s1 = gd.add_shaft(m, "s1", 0.1, 0.02, n_el=4, stations=(0.02,), torsion_only=True)
    s2 = gd.add_shaft(m, "s2", 0.1, 0.025, n_el=4, stations=(0.02,), torsion_only=True)
    gd.add_disc(m, s1, 0.02, 0, g.J1, torsion_only=True); gd.add_disc(m, s2, 0.02, 0, g.J2, torsion_only=True)
    gd.add_disc(m, s2, 0.1, 0, 0.01, torsion_only=True)
    m.add_mesh(g, s1["dof"](s1["node"](0.02), "th"), s2["dof"](s2["node"](0.02), "th"), contact="backlash")
    W = 500.0
    m.prescribe(s1["dof"](len(s1["z"]) - 1, "th"), lambda t: W * t, lambda t: W, lambda t: 0.0)
    v0 = np.zeros(m.n)
    for s, w in [(s1, W), (s2, W / g.i)]:
        for k in range(len(s["z"])):
            v0[m.index(s["dof"](k, "th"))] = w
    out = s2["dof"](len(s2["z"]) - 1, "th")
    m.add_load(out, lambda t, q, v: -20.0)
    Q = np.zeros(m.n); Q[m.index(out)] = -20.0
    q0 = m.static(Q)
    ra = m.simulate(3e-3, q0=q0, v0=v0, n_out=300, rtol=1e-9)
    rb = m.simulate(3e-3, q0=q0, v0=v0, n_out=300, f_cut=1e9, rtol=1e-9)
    assert np.mean(ra.mesh[0]["F"]) > 100
    Fa, Fb = ra.mesh[0]["F"], rb.mesh[0]["F"]
    assert np.max(np.abs(Fa - Fb)) < 1e-4 * np.max(np.abs(Fa))
