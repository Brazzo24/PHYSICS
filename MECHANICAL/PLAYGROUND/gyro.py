"""
Flywheel on a gimbal — transverse bearing reaction via auxiliary speed,
formulated so the auxiliary tilt is a rotation of the bearing housing.
"""
import sympy as sm
import sympy.physics.mechanics as me
me.mechanics_printing()

qb, qr = me.dynamicsymbols('q_b q_r')
ub, ur = me.dynamicsymbols('u_b u_r')
ua     = me.dynamicsymbols('u_a')            # auxiliary transverse tilt rate
t = me.dynamicsymbols._t
Ib_xx = sm.symbols('I_b_xx')
Ir_ax, Ir_tr = sm.symbols('I_r_ax I_r_tr')
Tb, Tr = me.dynamicsymbols('T_b T_r')
Tra    = me.dynamicsymbols('T_ra')

N = me.ReferenceFrame('N')
B = me.ReferenceFrame('B')                    # carrier / bearing housing
H = me.ReferenceFrame('H')                    # housing with aux transverse tilt
R = me.ReferenceFrame('R')                    # rotor

B.orient_axis(N, qb, N.x)
B.set_ang_vel(N, ub * N.x)

# Housing H = carrier B plus a fictitious transverse tilt about B.y
H.orient_axis(B, 0, B.y)                       # nominal alignment
H.set_ang_vel(B, ua * B.y)                     # auxiliary rate about transverse B.y

# Rotor spins about the housing's z relative to H
R.orient_axis(H, qr, H.z)
R.set_ang_vel(H, ur * H.z)

Bc = me.Point('Bc'); Bc.set_vel(N, 0)
Ib = me.inertia(B, Ib_xx, 0, 0)
carrier = me.RigidBody('carrier', Bc, B, 0, (Ib, Bc))

Rc = me.Point('Rc'); Rc.set_vel(N, 0)
Ir = me.inertia(R, Ir_tr, Ir_tr, Ir_ax)
rotor = me.RigidBody('rotor', Rc, R, 0, (Ir, Rc))

loads = [
    (B, Tb * N.x),
    (R, Tr * H.z),
    (H, Tra * B.y),        # reaction torque conjugate to the aux transverse tilt
]

kd = [qb.diff(t) - ub, qr.diff(t) - ur]
KM = me.KanesMethod(N, q_ind=[qb, qr], u_ind=[ub, ur], kd_eqs=kd, u_auxiliary=[ua])
fr, frstar = KM.kanes_equations([carrier, rotor], loads)

reaction = sm.simplify(sm.solve(KM.auxiliary_eqs[0], Tra)[0])
print("\n=== Transverse bearing reaction torque T_ra (about B.y) ===")
sm.pprint(reaction)