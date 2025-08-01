from sympy import symbols, lambdify, Symbol
from sympy.physics.mechanics import *
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# === Step 1: Define generalized coordinates and speeds ===
q1, q2 = dynamicsymbols('x theta')        # q1: cart position, q2: pendulum angle
u1, u2 = dynamicsymbols('x_dot theta_dot')  # u1: dx/dt, u2: dtheta/dt

# === Step 2: Define system parameters ===
m1, m2, l, g, F = symbols('m1 m2 l g F')

# === Step 3: Reference frames ===
N = ReferenceFrame('N')
P = N.orientnew('P', 'Axis', [q2, N.z])  # Pendulum rotates by theta about z-axis

# === Step 4: Points and velocities ===
O = Point('O')
O.set_vel(N, 0)

G1 = O.locatenew('G1', q1 * N.x)
G1.set_vel(N, u1 * N.x)

G2 = G1.locatenew('G2', l * P.x)
G2.v2pt_theory(G1, N, P)

# === Step 5: Define particles ===
cart = Particle('Cart', G1, m1)
pendulum = Particle('Pendulum', G2, m2)

# === Step 6: Kinematic differential equations ===
kin_diff = [u1 - q1.diff(), u2 - q2.diff()]

# === Step 7: Forces ===
forces = [
    (G1, F * N.x - m1 * g * N.y),
    (G2, -m2 * g * N.y)
]

# === Step 8: Kane's Method ===
coordinates = [q1, q2]
speeds = [u1, u2]

kane = KanesMethod(N, q_ind=coordinates, u_ind=speeds, kd_eqs=kin_diff)
fr, frstar = kane.kanes_equations([cart, pendulum], forces)

MM = kane.mass_matrix_full
forcing = kane.forcing_full

# === Step 9: Lambdify symbols ===
# Convert q1(t), u1(t) etc. to 'x', 'x_dot' etc. safely
state_syms = [Symbol(str(s).replace('(t)', '')) for s in coordinates + speeds]
param_syms = [m1, m2, l, g, F]

# === Lambdify EOMs ===
M_func = lambdify(state_syms + param_syms, MM)
F_func = lambdify(state_syms + param_syms, forcing)

# === Step 10: Define RHS for ODE solver ===
def rhs(t, y, param_vals):
    q_vals = y[:2]   # [x, theta]
    u_vals = y[2:]   # [x_dot, theta_dot]
    state_vals = q_vals + u_vals

    M_eval = np.array(M_func(*state_vals, *param_vals), dtype=float).squeeze()
    F_eval = np.array(F_func(*state_vals, *param_vals), dtype=float).squeeze()

    udot = np.linalg.solve(M_eval, F_eval)
    return np.concatenate([u_vals, udot])

# === Step 11: Simulate ===
y0 = [0.0, 0.1, 0.0, 0.0]  # Initial state: [x, theta, dx, dtheta]
param_vals = [1.0, 0.1, 1.0, 9.81, 0.0]  # [m1, m2, l, g, F]

t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

sol = solve_ivp(lambda t, y: rhs(t, y, param_vals), t_span, y0, t_eval=t_eval)

# === Step 12: Plot ===
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='x (cart position)')
plt.plot(sol.t, sol.y[1], label='θ (pendulum angle)')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.title('Pendulum on a Cart Simulation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()