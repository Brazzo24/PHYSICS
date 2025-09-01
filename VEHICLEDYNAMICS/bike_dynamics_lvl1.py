import sympy as sp

# === 1. Generalized coordinates ===
t = sp.symbols('t', real=True)
phi = sp.Function('phi')(t)    # roll angle
delta = sp.Function('delta')(t)  # steering angle

# === 2. Parameters ===
m, g, h = sp.symbols('m g h', positive=True)         # mass, gravity, CG height
I_phi, I_delta = sp.symbols('I_phi I_delta', positive=True)  # roll & steering inertia
v = sp.symbols('v', real=True)                      # forward speed
c_coupling = sp.symbols('c_coupling', real=True)    # symbolic coupling coefficient

# === 3. Kinetic energy ===
T = (1/sp.Integer(2))*I_phi*sp.diff(phi, t)**2 + (1/sp.Integer(2))*I_delta*sp.diff(delta, t)**2

# === 4. Potential energy ===
V = (1/sp.Integer(2))*m*g*h*phi**2

# === 5. Lagrangian ===
L = T - V

# === 6. Generalized coordinates vector ===
q = [phi, delta]

# === 7. Lagrange's equations ===
eqns = []
for qi in q:
    dLdqi_dot = sp.diff(L, sp.diff(qi, t))
    ddt_dLdqi_dot = sp.diff(dLdqi_dot, t)
    dLdqi = sp.diff(L, qi)
    # Add symbolic velocity coupling term: c_coupling*v*delta_dot appears in roll eqn, etc.
    # For now, treat it as generalized non-conservative force Q_i
    Q_i = 0  # We will later add C*v*dot(q)
    eq = ddt_dLdqi_dot - dLdqi - Q_i
    eqns.append(sp.simplify(eq))

eqns