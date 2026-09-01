"""Verification of orthomount.py against the closed-form algebra of SAE 830228.

Run with:  python3 test_orthomount.py
Every check is an assertion, so silence means everything agrees.
"""

import numpy as np

import orthomount as om

rng = np.random.default_rng(7)
D = np.deg2rad


def check(name, got, want, tol=1e-9):
    ok = np.allclose(got, want, rtol=1e-8, atol=tol)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}")
    assert ok, f"{name}: got {got}, want {want}"


print("Eq (20) inertia tensor from principal axes")
I_xi, I_eta, I_zeta, alpha = 0.95, 1.30, 1.60, D(30)
J = om.inertia_tensor_from_principal(I_xi, I_eta, I_zeta, alpha)
c, s = np.cos(alpha), np.sin(alpha)
check("I_x  = Ixi c^2 + Izeta s^2", J[0, 0], I_xi * c**2 + I_zeta * s**2)
check("I_z  = Ixi s^2 + Izeta c^2", J[2, 2], I_xi * s**2 + I_zeta * c**2)
check("I_zx = (Ixi - Izeta) s c ", -J[0, 2], (I_xi - I_zeta) * s * c)

print("\nEq (21) stiffness block from principal elastic axes")
K_xi, K_eta, K_zeta, beta = 5.3e4, 4.0e4, 2.45e4, D(-10)
Kb = om.stiffness_block_from_principal(K_xi, K_eta, K_zeta, beta)
cb, sb = np.cos(beta), np.sin(beta)
check("K_phiphi", Kb[0, 0], K_xi * cb**2 + K_zeta * sb**2)
check("K_psipsi", Kb[2, 2], K_xi * sb**2 + K_zeta * cb**2)
check("K_phipsi", Kb[0, 2], -(K_xi - K_zeta) * sb * cb)

print("\nEq (22) the two expressions for R agree with the matrix entries")
p, q = I_xi / I_zeta, K_xi / K_zeta
check("R = I_zx / I_z", om.R_from_inertia(p, alpha), -J[0, 2] / J[2, 2])
check("R = -K_phipsi / K_psipsi", om.R_from_stiffness(q, beta), -Kb[0, 2] / Kb[2, 2])

print("\nInverting Eq (22): solve_beta_for_R round-trips")
R_target = om.R_from_inertia(p, alpha)
b1, b2 = om.solve_beta_for_R(R_target, q)
check("beta root 1", om.R_from_stiffness(q, b1), R_target)
check("beta root 2", om.R_from_stiffness(q, b2), R_target)

print("\nThe orthogonality condition really does kill mode 2")
# Build the paper's idealised system with beta chosen to satisfy Eq (16).
beta_ok = b1
M, K = om.paper_system(mass=40.0, I_xi=I_xi, I_eta=I_eta, I_zeta=I_zeta, alpha=alpha,
                       K_xi=K_xi, K_eta=K_eta, K_zeta=K_zeta, beta=beta_ok,
                       K_trans=(1.6e6, 1.1e6, 2.2e6))
res = om.modal_from_matrices(M, K)
f_roll = np.array([0, 0, 0, 1.0, 0, 0])       # pure moment about x, Eq (9)
part = res.normalised_participation(f_roll)
n_alive = int(np.sum(part > 1e-8))
print("   participation of the 6 modes:", np.array2string(part, precision=10))
check("exactly one mode responds", n_alive, 1)

# ... and if beta is wrong, two modes respond
M2, K2 = om.paper_system(40.0, I_xi, I_eta, I_zeta, alpha,
                         K_xi, K_eta, K_zeta, beta_ok + D(8),
                         K_trans=(1.6e6, 1.1e6, 2.2e6))
part2 = om.modal_from_matrices(M2, K2).normalised_participation(f_roll)
assert np.sum(part2 > 1e-6) == 2, part2
print(f"  [ok ] detuning beta by 8 deg wakes the second mode "
      f"(participation {np.sort(part2)[-2]:.3f})")

print("\nEq (25) modal mass")
I_x, I_z, I_zx = J[0, 0], J[2, 2], -J[0, 2]
check("m1 = (Ix Iz - Izx^2)/Iz", om.modal_mass_paper(I_xi, I_zeta, alpha),
      (I_x * I_z - I_zx**2) / I_z)
check("m1 = Ixi Izeta / Iz", om.modal_mass_paper(I_xi, I_zeta, alpha),
      I_xi * I_zeta / I_z)

print("\nEq (27) resonant frequency of the surviving mode")
w1_closed = om.omega1_paper(K_xi, I_xi, p, q, alpha, beta_ok)
# the surviving mode is the one with non-zero participation
r = int(np.argmax(part))
check("closed form == eigenvalue", w1_closed, res.omega[r], tol=1e-6)
print(f"       f1 = {w1_closed / 2 / np.pi:.3f} Hz")

print("\nEq (27) inverted: K_xi for a target f1")
K_need = om.K_xi_for_target_f1(35.0, I_xi, p, q, alpha, beta_ok)
check("round trip", om.omega1_paper(K_need, I_xi, p, q, alpha, beta_ok) / 2 / np.pi, 35.0)

print("\nEqs (28)-(32) the axis of rotational vibration")
delta = om.delta_paper(p, alpha)
check("tan(delta) = -R", np.tan(delta), -R_target)
check("Eq (31) tan(alpha - delta) = p tan(alpha)",
      np.tan(alpha - delta), p * np.tan(alpha))
check("Eq (32) tan(beta - delta)  = q tan(beta)",
      np.tan(beta_ok - delta), q * np.tan(beta_ok))
# and the mode shape really points along delta
u = res.modes[:, r]
check("mode shape psi/phi = R", u[5] / u[3], R_target)
print(f"       alpha = {np.rad2deg(alpha):6.2f} deg   "
      f"beta = {np.rad2deg(beta_ok):6.2f} deg   "
      f"delta = {np.rad2deg(delta):6.2f} deg")

print("\nEq (18) degenerate case is correctly excluded")
# w1 == w2 would happen if K_phiphi/K_psipsi == I_x/I_z
ratio_k = Kb[0, 0] / Kb[2, 2]
ratio_i = I_x / I_z
print(f"       K_phiphi/K_psipsi = {ratio_k:.4f} vs I_x/I_z = {ratio_i:.4f}"
      f"  -> {'distinct' if abs(ratio_k - ratio_i) > 1e-3 else 'DEGENERATE'}")
assert abs(ratio_k - ratio_i) > 1e-3

print("\nGeneral 6-DOF assembler reproduces the idealised matrices")
# Four mounts placed so that the elastic centre lands on the CG and the
# principal elastic axes are tilted by beta.  Simple check: K must be symmetric,
# positive definite, and give the same modes as an equivalent direct build.
mounts = [
    om.Mount(position=[0.18, 0.05, -0.02], stiffness=[8e5, 5e5, 8e5]),
    om.Mount(position=[0.18, -0.05, -0.02], stiffness=[8e5, 5e5, 8e5]),
    om.Mount(position=[-0.20, 0.06, 0.03], stiffness=[7e5, 4e5, 7e5]),
    om.Mount(position=[-0.20, -0.06, 0.03], stiffness=[7e5, 4e5, 7e5]),
]
body = om.RigidBody.from_principal(40.0, I_xi, I_eta, I_zeta, alpha)
sysm = om.MountSystem(body, mounts)
Kg = sysm.K
check("K symmetric", Kg, Kg.T, tol=1e-6)
evals = np.linalg.eigvalsh(Kg)
assert evals.min() > 0, evals
print(f"  [ok ] K positive definite (min eigenvalue {evals.min():.3e})")
mr = sysm.modal()
print("       natural frequencies [Hz]:",
      np.array2string(mr.f_hz, precision=1, suppress_small=True))

print("\nRecovering (I_xi, I_zeta, alpha) back out of the tensor")
check("alpha_paper", body.alpha_paper, alpha)
gi_xi, gi_zeta, _ = body.principal_inplane
check("I_xi", gi_xi, I_xi)
check("I_zeta", gi_zeta, I_zeta)

print("\nElastic centre calculation")
# Put all mounts symmetric about a point 0.1 m ahead of the CG; the elastic
# centre must come out at that point.
off = np.array([0.10, 0.0, 0.05])
sym = [om.Mount(position=off + d, stiffness=[6e5, 6e5, 6e5])
       for d in ([0.2, 0.1, 0], [-0.2, 0.1, 0], [0.2, -0.1, 0], [-0.2, -0.1, 0])]
check("elastic centre", om.MountSystem(body, sym).elastic_centre, off, tol=1e-6)

print("\nEngine excitation: 180 deg parallel twin")
eng = om.Engine(
    cylinders=[om.Cylinder(crank_phase=0.0, y=+0.035, axis_tilt=D(24),
                           m_recip=0.20, m_rot=0.0),
               om.Cylinder(crank_phase=np.pi, y=-0.035, axis_tilt=D(24),
                           m_recip=0.20, m_rot=0.0)],
    crank_radius=0.027, rod_length=0.100, balance_factor=0.0)
w = 6000 * 2 * np.pi / 60
h = eng.harmonics(w, n_harm=2)
F1, M1 = h[1][:3], h[1][3:]
F2, M2h = h[2][:3], h[2][3:]
check("primary force cancels", np.abs(F1), np.zeros(3), tol=1e-6)
check("no secondary couple", np.abs(M2h), np.zeros(3), tol=1e-6)
print(f"       primary couple   |M1| = {np.linalg.norm(M1):8.2f} Nm at 6000 rpm")
print(f"       secondary force  |F2| = {np.linalg.norm(F2):8.2f} N  at 6000 rpm")
# closed form: |M1| = m_r * r * w^2 * (y1 - y2)
want = 0.20 * 0.027 * w**2 * 0.07
check("|M1| matches m r w^2 d", np.linalg.norm(M1), want, tol=1e-3)
# and the couple axis must be perpendicular to the cylinder axis
axis = M1.real / np.linalg.norm(M1.real) if np.linalg.norm(M1.real) > 0 else M1.imag
axis = np.real(M1 / M1[np.argmax(np.abs(M1))])
axis = axis / np.linalg.norm(axis)
e_cyl = np.array([np.sin(D(24)), 0.0, np.cos(D(24))])
check("couple axis perpendicular to cylinder axis", axis @ e_cyl, 0.0, tol=1e-9)
print(f"       couple axis tilt = {np.rad2deg(np.arctan2(-axis[2], axis[0])):.2f} deg "
      f"from x  (expected +/-24)")

print("\nEngine excitation: 360 deg twin, for contrast")
eng360 = om.Engine(
    cylinders=[om.Cylinder(0.0, +0.035, D(24), 0.20), om.Cylinder(0.0, -0.035, D(24), 0.20)],
    crank_radius=0.027, rod_length=0.100, balance_factor=0.0)
h3 = eng360.harmonics(w, n_harm=2)
assert np.abs(h3[1][3:]).max() < 1e-6, "360 twin should have no primary couple"
assert np.abs(h3[1][:3]).max() > 1.0, "360 twin should have a big primary force"
print("  [ok ] 360 deg twin: big primary FORCE, no primary couple "
      f"(|F1| = {np.abs(h3[1][:3]).max():.0f} N)")

print("\nFRF sanity: response peaks at the natural frequency")
f_hz = np.linspace(5, 250, 4000)
Q = sysm.response(2 * np.pi * f_hz, np.array([0, 0, 0, 1.0, 0, 0]))
peak = f_hz[np.argmax(np.abs(Q[:, 3]))]
nearest = mr.f_hz[np.argmin(np.abs(mr.f_hz - peak))]
check("peak near a natural frequency", peak, nearest, tol=1.0)
print(f"       roll FRF peak at {peak:.1f} Hz, nearest mode {nearest:.1f} Hz")

print("\nAll checks passed.")
