import numpy as np
from scipy.integrate import solve_ivp

print("solvers.py loaded")

def simulate_time_response(system, t_span, x0=None, damping=None, forcing_func=None):
    I_mat, K_mat = system.build_matrices()
    n = system._num_inertias
    C_mat = damping if damping is not None else np.zeros((n, n))

    def ode_system(t, x):
        theta, theta_dot = x[:n], x[n:2*n]
        T_ext = forcing_func(t) if forcing_func else np.zeros(n)
        rhs = T_ext - C_mat @ theta_dot - K_mat @ theta
        theta_ddot = np.linalg.solve(I_mat, rhs)
        return np.hstack((theta_dot, theta_ddot))

    t_eval = np.linspace(*t_span)
    x0 = np.zeros(2*n) if x0 is None else x0
    sol = solve_ivp(ode_system, (t_eval[0], t_eval[-1]), x0, t_eval=t_eval)

    # Compute energy over time
    theta, theta_dot = sol.y[:n, :].T, sol.y[n:2*n, :].T
    ke, pe = system.compute_energy_equilibrium(theta, theta_dot)

    return sol.t, sol.y.T, ke, pe
