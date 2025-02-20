import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def two_dof_ode(state, t, M_inv, K):
    """
    ODE for undamped 2-DOF system:
       M x''(t) + K x(t) = 0
    state = [x1, x2, v1, v2]
    """
    x1, x2, v1, v2 = state
    x = np.array([x1, x2])
    v = np.array([v1, v2])
    
    # acceleration = -M^{-1} * K * x
    a = -M_inv.dot(K).dot(x)
    return [v1, v2, a[0], a[1]]

def compute_energy(x, v, M, K):
    """
    Compute kinetic + potential energy for
    2-DOF system at each time step.
    Inputs:
      x: shape (N,2) - displacement
      v: shape (N,2) - velocity
    Returns:
      T: shape (N,) - kinetic energy
      U: shape (N,) - potential energy
      E: shape (N,) - total energy
    """
    # Kinetic: 1/2 v^T M v
    # Potential: 1/2 x^T K x
    T = np.empty(len(x))
    U = np.empty(len(x))
    
    for i in range(len(x)):
        xi = x[i]   # shape (2,)
        vi = v[i]   # shape (2,)
        
        T[i] = 0.5 * vi @ (M @ vi)
        U[i] = 0.5 * xi @ (K @ xi)
    
    E = T + U
    return T, U, E

# ---------------------------------------------------------
# 1. Define M and K for a 2-DOF system
# ---------------------------------------------------------
m1, m2 = 1.0, 2.0
k1, k2, k3 = 50.0, 80.0, 50.0

M = np.array([[m1, 0.0],
              [0.0, m2]])
K = np.array([[k1 + k2,     -k2 ],
              [   -k2  , k2 + k3]])

# Precompute M inverse (for convenience in ODE)
M_inv = np.linalg.inv(M)

# ---------------------------------------------------------
# 2. Initial Conditions (x(0), v(0))
# ---------------------------------------------------------
x0 = np.array([0.1, -0.05])  # initial displacement
v0 = np.array([0.0,  0.1 ])  # initial velocity

state0 = [x0[0], x0[1], v0[0], v0[1]]

# ---------------------------------------------------------
# 3. Time Integration
# ---------------------------------------------------------
t_end = 2.0
t_eval = np.linspace(0, t_end, 1001)

sol = odeint(two_dof_ode, state0, t_eval, args=(M_inv, K))

# Extract x(t) and v(t)
x_sol = sol[:, 0:2]   # shape (N,2)
v_sol = sol[:, 2:4]   # shape (N,2)

# ---------------------------------------------------------
# 4. Compute energies and check constancy
# ---------------------------------------------------------
T, U, E = compute_energy(x_sol, v_sol, M, K)

E0 = E[0]  # initial total energy
E_diff = E - E0  # difference from initial energy

# A measure of how well energy is conserved:
max_abs_diff = np.max(np.abs(E_diff))
print(f"Maximum deviation in total energy: {max_abs_diff:.5e} J (approx)")

# ---------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------
plt.figure(figsize=(8,6))

plt.subplot(2,1,1)
plt.plot(t_eval, T, label='Kinetic')
plt.plot(t_eval, U, label='Potential')
plt.plot(t_eval, E, 'k--', label='Total')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()
plt.title('Energy in Undamped 2-DOF System')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t_eval, E_diff, label='E(t) - E(0)')
plt.xlabel('Time (s)')
plt.ylabel('Deviation from Initial Energy (J)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
