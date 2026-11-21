"""
gearmesh_model.py
=================
Python re-implementation of the two ValDyn benchmark models
("GEAR PAIR MODEL with GEARMESH2D element" and "... with Shaft and Bearing").

Units inside the solver: SI (m, kg, s, N, rad).  Parameters are given in the
units of the ValDyn input masks (mm, N/mm, ...) and converted in `to_si()`.

Stage 1  (6 DOF):  gear 1 (x1, y1, th1)  +  gear 2 (x2, y2, th2)
         driver: prescribed constant speed -> torsional spring/damper -> gear 1
         gear 1/2: linear x/y spring-dampers to ground
         gear 2: drag torque (preload of a zero-stiffness torsional spring)
         GEARMESH2D: two flanks A/B with backlash, linear stiffness+damping

Stage 2  (12 DOF): gear 1 as in stage 1, gear 2 (x2, y2, th2) now rides on a
         radial bearing with clearance on a NON-rotating hollow shaft
         (3 nodes: block end, gear/mid node, case end; Euler-Bernoulli,
         rotations statically condensed), shaft ends carried by the
         non-linear BlockX/BlockY/CaseX/CaseY tables.
         Load on gear 2: angle dependent torque  T = A*sin(theta_driver)

The integrator is a fixed-step RK4 (dt ~ 2e-7 s).  If numba is available
the kernels are JIT-compiled (~0.1-0.5 s per run); otherwise pure Python is
used (works, but ~100x slower).
"""
from __future__ import annotations
import copy
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

RPM = 2*np.pi/60.0

# ----------------------------------------------------------------------------
# 1. Reference parameters, transcribed 1:1 from the ValDyn masks (photos)
# ----------------------------------------------------------------------------
STAGE1_REF = dict(
    # excitation / drive (slide "Antrieb")
    speed_rpm=2500.0,
    k_drive=4.3e7,        # N.m/rad
    c_drive=172.0,        # N.m.s/rad
    # gear 1 (BUSH_1_*): masses, inertia, support
    m1=2.67, J1=0.01709,  # kg, kg.m^2
    k1x=3.6e9, k1y=3.6e9, # N/m
    c1x=10218., c1y=10218.,  # N.s/m
    # gear 2 (BUSH_2_*)
    m2=1.5, J2=0.0054,
    k2x=3.6e9, k2y=3.6e9,
    c2x=7397., c2y=7397.,
    x2=131.56, y2=47.88,  # mm, centre of gear 2
    w2_ratio=-1.37,       # initial velocity of gear 2 [deg/deg of driver]
    # drag torque (zero stiffness spring with preload)
    T_drag=10.0,          # N.m
    # GEARMESH2D_1
    k_mesh=308000.0,      # N/mm
    c_mesh=4300.0,        # N.s/m
    z1=40, z2=30,
    a=140.0,              # mm centre distance (nominal)
    alpha=20.0,           # deg pressure angle
    backlash_n=0.25,      # mm, initial tooth NORMAL backlash
    # modelling switches (not in the mask -> structural assumptions)
    clip_tension=False,   # True: contact force never < 0 (damping clipped)
    gap0_split=0.5,       # initial gap share on side A (0.5 = centred)
)

STAGE2_REF = dict(
    STAGE1_REF,
    speed_rpm=3000.0,     # result files are labelled "03000" (slide text: 2500)
    w2_ratio=-40/30,      # assumption: no initial over-speed in stage 2
    T_drag=0.0,
    T_sin_amp=10.0,       # N.m, SINUS10.txt, period 360 deg of driver angle
    # shaft (SHAFT_4 / SHAFT_5)
    dAo=26.0, dAi=20.0, lA1=50.0, lA2=50.0, E=210000.0, rhoA=7.85e-6,
    c_bend=100.0,         # N.s/m bending damping
    # node masses as typed in the masks (MASS_8 has no "/2*pi/4" !)
    m_node8=None, m_node9=None, m_nodeM=None,  # None -> formula from masks
    # R_BEARING_1 (simplified)
    k_brg=500000.0,       # N/mm radial stiffness
    c_brg=1.26,           # N.s/m radial damping
    clr_brg=30.0,         # um radial clearance
    clr_is_diametral=False,  # True -> free radius = clr/2
    clr_mode='radial',    # 'radial' (circular) | 'per_plane' (XZ/YZ separately) | 'x_only'
    mu_brg=0.001, d_brg=28.0,
    # non-linear supports: (disp mm, force N, damping N.s/m) tables
    block_x=[(-2,-2e6,200),(-1,-2e5,20),(-0.002,-10,20),(0,0,20),(0.002,10,20),(1,2e5,20),(2,2e6,200)],
    block_y=[(-2,-2e6,200),(-1,-2e5,20),(-0.002,-10,20),(0,0,20),(0.002,10,20),(1,2e5,20),(2,2e6,200)],
    case_x=[(-2,-1e6,200),(-1,-1e5,20),(-0.002,-10,20),(0,0,20),(0.002,10,20),(1,1e5,20),(2,1e6,200)],
    case_y=[(-2,-1e6,200),(-1,-1e5,20),(-0.002,-10,20),(0,0,20),(0.002,10,20),(1,1e5,20),(2,1e6,200)],
    damp_scale=1.0,       # multiplies the 3rd table column (unit question N.s/m vs N.s/mm)
)


def read_valdyn_table(path):
    """Read a ValDyn non-linear stiffness file (plain text or the .rtf export).
    Format per line: displacement[mm]  stiffness-force[N]  damping"""
    import re
    txt = open(path, encoding='utf-8', errors='ignore').read()
    rows = []
    for line in txt.splitlines():
        line = re.sub(r'\\[a-z]+-?\d* ?', ' ', line).replace('}', ' ').replace('{', ' ')
        if line.strip().startswith('#'):
            continue
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        if len(nums) == 3:
            rows.append(tuple(float(n) for n in nums))
    return rows


# ----------------------------------------------------------------------------
# 2. Geometry helpers
# ----------------------------------------------------------------------------
def mesh_geometry(p):
    """Base radii and flank normals for the planar mesh.

    e  : unit centre line gear1 -> gear2
    t  : e rotated +90 deg (direction of pitch-point velocity for w1>0)
    nA : direction of the side-A (drive flank) force ON gear 2
    nB : direction of the side-B (coast flank) force ON gear 2
    """
    al = np.deg2rad(p['alpha'])
    c2 = np.array([p['x2'], p['y2']])*1e-3
    a_geo = np.linalg.norm(c2)
    e = c2/a_geo
    t = np.array([-e[1], e[0]])
    z1, z2 = p['z1'], p['z2']
    a = p['a']*1e-3
    r1, r2 = a*z1/(z1+z2), a*z2/(z1+z2)
    rb1, rb2 = r1*np.cos(al), r2*np.cos(al)
    nA = np.cos(al)*t + np.sin(al)*e
    nB = -np.cos(al)*t + np.sin(al)*e
    return dict(rb1=rb1, rb2=rb2, r1=r1, r2=r2, nA=nA, nB=nB, e=e, t=t, alpha=al, a=a)


def shaft_condensed_K(p):
    """3x3 lateral stiffness (block end, mid, case end) of the 2-element
    Euler-Bernoulli shaft with free (massless) nodal rotations, per plane."""
    E = p['E']*1e6                       # N/m^2
    I = np.pi/64*((p['dAo']*1e-3)**4-(p['dAi']*1e-3)**4)
    def ke(L):
        return E*I/L**3*np.array([[12, 6*L, -12, 6*L], [6*L, 4*L*L, -6*L, 2*L*L],
                                  [-12, -6*L, 12, -6*L], [6*L, 2*L*L, -6*L, 4*L*L]])
    L1, L2 = p['lA1']*1e-3, p['lA2']*1e-3
    K = np.zeros((6, 6))                 # dofs v8, p8, vm, pm, v9, p9
    K[0:4, 0:4] += ke(L1); K[2:6, 2:6] += ke(L2)
    t = [0, 2, 4]; r = [1, 3, 5]
    Ktt = K[np.ix_(t, t)]; Ktr = K[np.ix_(t, r)]; Krr = K[np.ix_(r, r)]
    return Ktt - Ktr @ np.linalg.solve(Krr, Ktr.T)


def shaft_node_masses(p):
    A2 = (p['dAo']**2 - p['dAi']**2)
    rho = p['rhoA']
    m8 = p['m_node8'] if p['m_node8'] is not None else rho*A2*p['lA1']              # as typed
    m9 = p['m_node9'] if p['m_node9'] is not None else rho*A2*p['lA2']/2*np.pi/4
    mM = p['m_nodeM'] if p['m_nodeM'] is not None else rho*A2*(p['lA1']+p['lA2'])/2*np.pi/4
    return m8, mM, m9


def _table(tab, damp_scale=1.0):
    a = np.array(tab, float)
    return (np.ascontiguousarray(a[:, 0]*1e-3), np.ascontiguousarray(a[:, 1]),
            np.ascontiguousarray(a[:, 2]*damp_scale))


# ----------------------------------------------------------------------------
# 3. Numba kernels
# ----------------------------------------------------------------------------
@njit(cache=True)
def _interp_lin(x, xs, ys):
    n = xs.shape[0]
    if x <= xs[0]:
        return ys[0] + (ys[1]-ys[0])/(xs[1]-xs[0])*(x-xs[0])
    if x >= xs[n-1]:
        return ys[n-1] + (ys[n-1]-ys[n-2])/(xs[n-1]-xs[n-2])*(x-xs[n-1])
    i = 0
    while xs[i+1] < x:
        i += 1
    return ys[i] + (ys[i+1]-ys[i])/(xs[i+1]-xs[i])*(x-xs[i])


@njit(cache=True)
def _mesh_forces(q, v, P, i1, i2):
    """Return dA, dB (penetrations), FA, FB for gear dofs starting at i1 / i2.
    P: [rb1, rb2, nAx, nAy, nBx, nBy, k, c, gA0, gB0, clip]"""
    rb1, rb2 = P[0], P[1]
    nAx, nAy, nBx, nBy = P[2], P[3], P[4], P[5]
    k, c, gA0, gB0, clip = P[6], P[7], P[8], P[9], P[10]
    rot = rb1*q[i1+2] + rb2*q[i2+2]
    rotd = rb1*v[i1+2] + rb2*v[i2+2]
    dux = q[i1] - q[i2]; duy = q[i1+1] - q[i2+1]
    dvx = v[i1] - v[i2]; dvy = v[i1+1] - v[i2+1]
    dA = rot + dux*nAx + duy*nAy - gA0
    dB = -rot + dux*nBx + duy*nBy - gB0
    dAd = rotd + dvx*nAx + dvy*nAy
    dBd = -rotd + dvx*nBx + dvy*nBy
    FA = 0.0; FB = 0.0
    if dA > 0.0:
        FA = k*dA + c*dAd
        if clip > 0.5 and FA < 0.0:
            FA = 0.0
    if dB > 0.0:
        FB = k*dB + c*dBd
        if clip > 0.5 and FB < 0.0:
            FB = 0.0
    return dA, dB, FA, FB


@njit(cache=True)
def _rhs1(t, y, P, G, out):
    """Stage 1 rhs.  y = [q(6), v(6)];  q = x1,y1,th1,x2,y2,th2"""
    q = y[:6]; v = y[6:]
    Om, kd, cd = G[0], G[1], G[2]
    m1, J1, k1x, k1y, c1x, c1y = G[3], G[4], G[5], G[6], G[7], G[8]
    m2, J2, k2x, k2y, c2x, c2y = G[9], G[10], G[11], G[12], G[13], G[14]
    Tdrag = G[15]
    dA, dB, FA, FB = _mesh_forces(q, v, P, 0, 3)
    nAx, nAy, nBx, nBy, rb1, rb2 = P[2], P[3], P[4], P[5], P[0], P[1]
    Fx = FA*nAx + FB*nBx; Fy = FA*nAy + FB*nBy          # force ON gear 2
    Td = kd*(Om*t - q[2]) + cd*(Om - v[2])
    out[0:6] = v
    out[6] = (-k1x*q[0] - c1x*v[0] - Fx)/m1
    out[7] = (-k1y*q[1] - c1y*v[1] - Fy)/m1
    out[8] = (Td - FA*rb1 + FB*rb1)/J1
    out[9] = (-k2x*q[3] - c2x*v[3] + Fx)/m2
    out[10] = (-k2y*q[4] - c2y*v[4] + Fy)/m2
    out[11] = (Tdrag - FA*rb2 + FB*rb2)/J2


@njit(cache=True)
def _support(u, ud, xs, fs, cs):
    return _interp_lin(u, xs, fs) + _interp_lin(u, xs, cs)*ud


@njit(cache=True)
def _rhs2(t, y, P, G, Ks, Cs, TB, out, aux):
    """Stage 2 rhs.  q = x1,y1,th1, x2,y2,th2, x8,y8, xm,ym, x9,y9  (12)"""
    n = 12
    q = y[:n]; v = y[n:]
    Om, kd, cd = G[0], G[1], G[2]
    m1, J1, k1x, k1y, c1x, c1y = G[3], G[4], G[5], G[6], G[7], G[8]
    m2, J2 = G[9], G[10]
    Tamp = G[15]
    m8, mM, m9 = G[16], G[17], G[18]
    kb, cb, clr, mu, rbrg = G[19], G[20], G[21], G[22], G[23]
    dA, dB, FA, FB = _mesh_forces(q, v, P, 0, 3)
    nAx, nAy, nBx, nBy, rb1, rb2 = P[2], P[3], P[4], P[5], P[0], P[1]
    Fx = FA*nAx + FB*nBx; Fy = FA*nAy + FB*nBy
    th_d = Om*t
    Td = kd*(th_d - q[2]) + cd*(Om - v[2])
    Tload = Tamp*np.sin(th_d)
    # --- bearing gear2 <-> shaft mid node, radial clearance (circular)
    rx = q[3] - q[8]; ry = q[4] - q[9]
    rvx = v[3] - v[8]; rvy = v[4] - v[9]
    Fbx = 0.0; Fby = 0.0; Tf = 0.0; Fn = 0.0
    mode = G[24]
    if mode < 0.5:                      # circular radial clearance (isotropic)
        r = np.sqrt(rx*rx + ry*ry)
        if r > clr:
            ex = rx/r; ey = ry/r
            rd = rvx*ex + rvy*ey
            Fn = kb*(r - clr) + cb*rd
            if Fn < 0.0:
                Fn = 0.0
            Fbx = -Fn*ex; Fby = -Fn*ey        # force on gear 2
    else:                               # per-plane clearance (XZ / YZ plane)
        cx = clr
        cy = clr if mode < 1.5 else 0.0   # mode 2: clearance only in X
        if rx > cx:
            Fbx = -(kb*(rx-cx) + cb*rvx)
        elif rx < -cx:
            Fbx = -(kb*(rx+cx) + cb*rvx)
        if ry > cy:
            Fby = -(kb*(ry-cy) + cb*rvy)
        elif ry < -cy:
            Fby = -(kb*(ry+cy) + cb*rvy)
        if cy == 0.0:
            Fby = -(kb*ry + cb*rvy)
        Fn = np.sqrt(Fbx*Fbx + Fby*Fby)
    Tf = -mu*Fn*rbrg*np.tanh(v[5]/1.0)   # rotational friction (shaft fixed)
    # --- shaft (condensed), dofs order (8, M, 9) per plane
    ux0, ux1, ux2 = q[6], q[8], q[10]
    uy0, uy1, uy2 = q[7], q[9], q[11]
    vx0, vx1, vx2 = v[6], v[8], v[10]
    vy0, vy1, vy2 = v[7], v[9], v[11]
    fsx0 = -(Ks[0, 0]*ux0 + Ks[0, 1]*ux1 + Ks[0, 2]*ux2) - (Cs[0, 0]*vx0 + Cs[0, 1]*vx1 + Cs[0, 2]*vx2)
    fsx1 = -(Ks[1, 0]*ux0 + Ks[1, 1]*ux1 + Ks[1, 2]*ux2) - (Cs[1, 0]*vx0 + Cs[1, 1]*vx1 + Cs[1, 2]*vx2)
    fsx2 = -(Ks[2, 0]*ux0 + Ks[2, 1]*ux1 + Ks[2, 2]*ux2) - (Cs[2, 0]*vx0 + Cs[2, 1]*vx1 + Cs[2, 2]*vx2)
    fsy0 = -(Ks[0, 0]*uy0 + Ks[0, 1]*uy1 + Ks[0, 2]*uy2) - (Cs[0, 0]*vy0 + Cs[0, 1]*vy1 + Cs[0, 2]*vy2)
    fsy1 = -(Ks[1, 0]*uy0 + Ks[1, 1]*uy1 + Ks[1, 2]*uy2) - (Cs[1, 0]*vy0 + Cs[1, 1]*vy1 + Cs[1, 2]*vy2)
    fsy2 = -(Ks[2, 0]*uy0 + Ks[2, 1]*uy1 + Ks[2, 2]*uy2) - (Cs[2, 0]*vy0 + Cs[2, 1]*vy1 + Cs[2, 2]*vy2)
    # --- non-linear supports (force ON shaft = -table force)
    FbxB = _support(ux0, vx0, TB[0], TB[1], TB[2])
    FbyB = _support(uy0, vy0, TB[3], TB[4], TB[5])
    FcxC = _support(ux2, vx2, TB[6], TB[7], TB[8])
    FcyC = _support(uy2, vy2, TB[9], TB[10], TB[11])
    out[:n] = v
    out[n+0] = (-k1x*q[0] - c1x*v[0] - Fx)/m1
    out[n+1] = (-k1y*q[1] - c1y*v[1] - Fy)/m1
    out[n+2] = (Td - FA*rb1 + FB*rb1)/J1
    out[n+3] = (Fx + Fbx)/m2
    out[n+4] = (Fy + Fby)/m2
    out[n+5] = (Tload - FA*rb2 + FB*rb2 + Tf)/J2
    out[n+6] = (fsx0 - FbxB)/m8
    out[n+7] = (fsy0 - FbyB)/m8
    out[n+8] = (fsx1 - Fbx)/mM
    out[n+9] = (fsy1 - Fby)/mM
    out[n+10] = (fsx2 - FcxC)/m9
    out[n+11] = (fsy2 - FcyC)/m9
    aux[0] = FbxB; aux[1] = FbyB; aux[2] = FcxC; aux[3] = FcyC
    aux[4] = Tload; aux[5] = -Fbx; aux[6] = -Fby


@njit(cache=True)
def _integrate1(y0, P, G, dt, nsteps, every):
    nout = nsteps//every + 1
    Y = np.zeros((nout, y0.shape[0])); T = np.zeros(nout)
    y = y0.copy(); k1 = np.zeros_like(y); k2 = np.zeros_like(y); k3 = np.zeros_like(y); k4 = np.zeros_like(y)
    Y[0] = y; j = 1; t = 0.0
    for i in range(nsteps):
        _rhs1(t, y, P, G, k1)
        _rhs1(t+0.5*dt, y+0.5*dt*k1, P, G, k2)
        _rhs1(t+0.5*dt, y+0.5*dt*k2, P, G, k3)
        _rhs1(t+dt, y+dt*k3, P, G, k4)
        y = y + dt/6.0*(k1+2*k2+2*k3+k4)
        t += dt
        if (i+1) % every == 0:
            Y[j] = y; T[j] = t; j += 1
    return T, Y


@njit(cache=True)
def _integrate2(y0, P, G, Ks, Cs, TB, dt, nsteps, every):
    nout = nsteps//every + 1
    Y = np.zeros((nout, y0.shape[0])); T = np.zeros(nout); A = np.zeros((nout, 7))
    y = y0.copy(); k1 = np.zeros_like(y); k2 = np.zeros_like(y); k3 = np.zeros_like(y); k4 = np.zeros_like(y)
    aux = np.zeros(7)
    _rhs2(0.0, y, P, G, Ks, Cs, TB, k1, aux)
    Y[0] = y; A[0] = aux; j = 1; t = 0.0
    for i in range(nsteps):
        _rhs2(t, y, P, G, Ks, Cs, TB, k1, aux)
        _rhs2(t+0.5*dt, y+0.5*dt*k1, P, G, Ks, Cs, TB, k2, aux)
        _rhs2(t+0.5*dt, y+0.5*dt*k2, P, G, Ks, Cs, TB, k3, aux)
        _rhs2(t+dt, y+dt*k3, P, G, Ks, Cs, TB, k4, aux)
        y = y + dt/6.0*(k1+2*k2+2*k3+k4)
        t += dt
        if (i+1) % every == 0:
            _rhs2(t, y, P, G, Ks, Cs, TB, k1, aux)
            Y[j] = y; T[j] = t; A[j] = aux; j += 1
    return T, Y, A


# ----------------------------------------------------------------------------
# 4. Public API
# ----------------------------------------------------------------------------
def params(stage=1, **overrides):
    p = copy.deepcopy(STAGE1_REF if stage == 1 else STAGE2_REF)
    for k, v in overrides.items():
        if k not in p:
            raise KeyError(f'unknown parameter {k}')
        p[k] = v
    return p


def _mesh_P(p, g):
    jn = p['backlash_n']*1e-3
    return np.array([g['rb1'], g['rb2'], g['nA'][0], g['nA'][1], g['nB'][0], g['nB'][1],
                     p['k_mesh']*1e3, p['c_mesh'], p['gap0_split']*jn, (1-p['gap0_split'])*jn,
                     1.0 if p['clip_tension'] else 0.0])


def _postprocess_mesh(res, p, g, Y, i1, i2):
    q = Y[:, :Y.shape[1]//2]; v = Y[:, Y.shape[1]//2:]
    P = _mesh_P(p, g)
    out = np.array([_mesh_forces(q[i], v[i], P, i1, i2) for i in range(len(q))]) if not HAVE_NUMBA else \
        _mesh_batch(q, v, P, i1, i2)
    cosa = np.cos(g['alpha'])
    # ValDyn "LASH" = remaining free play per flank, measured on the pitch circle
    res['lashA'] = -out[:, 0]/cosa*1e3          # mm
    res['lashB'] = -out[:, 1]/cosa*1e3
    res['FA'] = out[:, 2]; res['FB'] = out[:, 3]
    res['Ft'] = (out[:, 2]-out[:, 3])*cosa      # tangential force (A positive)
    # alternative lash definition: rotational part only (centre motions ignored)
    rot = g['rb1']*q[:, i1+2] + g['rb2']*q[:, i2+2]
    res['lashA_rot'] = -(rot - P[8])/cosa*1e3
    res['lashB_rot'] = -(-rot - P[9])/cosa*1e3


@njit(cache=True)
def _mesh_batch(q, v, P, i1, i2):
    o = np.zeros((q.shape[0], 4))
    for i in range(q.shape[0]):
        a, b, c, d = _mesh_forces(q[i], v[i], P, i1, i2)
        o[i, 0] = a; o[i, 1] = b; o[i, 2] = c; o[i, 3] = d
    return o


def simulate_stage1(p=None, angle_end=360.0, dt=2e-7, out_dt=1e-6):
    p = p or params(1)
    g = mesh_geometry(p)
    Om = p['speed_rpm']*RPM
    G = np.array([Om, p['k_drive'], p['c_drive'],
                  p['m1'], p['J1'], p['k1x'], p['k1y'], p['c1x'], p['c1y'],
                  p['m2'], p['J2'], p['k2x'], p['k2y'], p['c2x'], p['c2y'], p['T_drag']])
    P = _mesh_P(p, g)
    y0 = np.zeros(12); y0[6+2] = Om; y0[6+5] = p['w2_ratio']*Om
    t_end = np.deg2rad(angle_end)/Om
    nsteps = int(round(t_end/dt)); every = max(1, int(round(out_dt/dt)))
    T, Y = _integrate1(y0, P, G, dt, nsteps, every)
    res = dict(t=T, angle=np.rad2deg(Om*T), Y=Y, p=p, g=g)
    _postprocess_mesh(res, p, g, Y, 0, 3)
    res['x1'] = Y[:, 0]*1e3; res['y1'] = Y[:, 1]*1e3
    res['x2'] = p['x2'] + Y[:, 3]*1e3; res['y2'] = p['y2'] + Y[:, 4]*1e3
    res['Fs1x'] = p['k1x']*Y[:, 0] + p['c1x']*Y[:, 6]          # support force gear 1 (x)
    res['Fs1y'] = p['k1y']*Y[:, 1] + p['c1y']*Y[:, 7]
    return res


def simulate_stage2(p=None, angle_end=1440.0, dt=2e-7, out_dt=2e-6):
    p = p or params(2)
    g = mesh_geometry(p)
    Om = p['speed_rpm']*RPM
    m8, mM, m9 = shaft_node_masses(p)
    clr = p['clr_brg']*1e-6*(0.5 if p['clr_is_diametral'] else 1.0)
    G = np.array([Om, p['k_drive'], p['c_drive'],
                  p['m1'], p['J1'], p['k1x'], p['k1y'], p['c1x'], p['c1y'],
                  p['m2'], p['J2'], 0, 0, 0, 0, p['T_sin_amp'],
                  m8, mM, m9, p['k_brg']*1e3, p['c_brg'], clr, p['mu_brg'], p['d_brg']*0.5e-3,
                  {'radial': 0.0, 'per_plane': 1.0, 'x_only': 2.0}[p['clr_mode']]])
    Ks = shaft_condensed_K(p)
    Cs = p['c_bend']*Ks/Ks[1, 1]
    TB = []
    for key in ('block_x', 'block_y', 'case_x', 'case_y'):
        TB += list(_table(p[key], p['damp_scale']))
    TB = tuple(TB)
    P = _mesh_P(p, g)
    y0 = np.zeros(24); y0[12+2] = Om; y0[12+5] = p['w2_ratio']*Om
    t_end = np.deg2rad(angle_end)/Om
    nsteps = int(round(t_end/dt)); every = max(1, int(round(out_dt/dt)))
    T, Y, A = _integrate2(y0, P, G, Ks, np.ascontiguousarray(Cs), TB, dt, nsteps, every)
    res = dict(t=T, angle=np.rad2deg(Om*T), Y=Y, p=p, g=g)
    _postprocess_mesh(res, p, g, Y, 0, 3)
    X0, Y0 = p['x2'], p['y2']
    res['gearX'] = X0 + Y[:, 3]*1e3;  res['gearY'] = Y0 + Y[:, 4]*1e3
    res['blockX'] = X0 + Y[:, 6]*1e3; res['blockY'] = Y0 + Y[:, 7]*1e3
    res['midX'] = X0 + Y[:, 8]*1e3;   res['midY'] = Y0 + Y[:, 9]*1e3
    res['caseX'] = X0 + Y[:, 10]*1e3; res['caseY'] = Y0 + Y[:, 11]*1e3
    res['FXBlock'], res['FYBlock'], res['FXCase'], res['FYCase'] = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    res['Tload'] = A[:, 4]; res['FbrgX'] = A[:, 5]; res['FbrgY'] = A[:, 6]
    return res


def natural_frequencies(p, stage=1):
    """Quick linear estimates (Hz) used for sanity checks / time-step choice."""
    g = mesh_geometry(p)
    f = {}
    f['drive torsion (gear1 on drive spring)'] = np.sqrt(p['k_drive']/p['J1'])/2/np.pi
    f['gear1 support'] = np.sqrt(p['k1x']/p['m1'])/2/np.pi
    if stage == 1:
        f['gear2 support'] = np.sqrt(p['k2x']/p['m2'])/2/np.pi
    meq = 1/(g['rb1']**2/p['J1'] + g['rb2']**2/p['J2'])
    f['mesh (in contact)'] = np.sqrt(p['k_mesh']*1e3/meq)/2/np.pi
    f['_meq_kg'] = meq
    f['_zeta_mesh'] = p['c_mesh']/(2*np.sqrt(p['k_mesh']*1e3*meq))
    return f
