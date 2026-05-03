"""
Advanced Usage Patterns for Motorcycle Roll Dynamics Solver
===========================================================

This guide covers:
- Numerical optimization techniques
- Solving coupled systems
- Performance optimization
- Integration with numerical simulators
"""

import sympy as sp
from sympy.utilities.lambdify import lambdify
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import numpy as np
from scipy.optimize import fsolve, minimize, root


# ============================================================================
# PATTERN 1: Use lambdify for Fast Numerical Evaluation
# ============================================================================

def pattern_1_lambdify_optimization():
    """
    For repeated numerical evaluation, convert SymPy expressions to NumPy
    functions using lambdify. This is MUCH faster than repeated substitution.
    """
    print("=" * 70)
    print("PATTERN 1: Lambdify for Fast Evaluation")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    # Get equations
    N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()
    
    # Convert to NumPy function
    N_r_func = lambdify(
        (model.M, model.p, model.a_var, model.h_bar, 
         model.b_bar, model.g, model.a_y, model.phi),
        N_r_expr,
        'numpy'
    )
    
    N_f_func = lambdify(
        (model.M, model.p, model.a_var, model.h_bar,
         model.b_bar, model.g, model.a_y, model.phi),
        N_f_expr,
        'numpy'
    )
    
    # Parameters
    M, p, a, h, g = 200, 1.4, 0.6, 0.5, 9.81
    h_bar = h / p
    b_bar = (p - a) / p
    
    # Now evaluate for an array of lateral accelerations (very fast!)
    a_y_array = np.linspace(0, 10, 100)
    phi_array = np.arctan(a_y_array / g)
    
    # Vectorized evaluation
    N_r_array = N_r_func(M, p, a, h_bar, b_bar, g, a_y_array, phi_array)
    N_f_array = N_f_func(M, p, a, h_bar, b_bar, g, a_y_array, phi_array)
    
    print(f"\nEvaluated {len(a_y_array)} points efficiently using lambdify")
    print(f"Sample values:")
    print(f"  a_y[0] = {a_y_array[0]:.2f} m/s² → N_r = {N_r_array[0]:.0f} N, N_f = {N_f_array[0]:.0f} N")
    print(f"  a_y[50] = {a_y_array[50]:.2f} m/s² → N_r = {N_r_array[50]:.0f} N, N_f = {N_f_array[50]:.0f} N")
    print(f"  a_y[-1] = {a_y_array[-1]:.2f} m/s² → N_r = {N_r_array[-1]:.0f} N, N_f = {N_f_array[-1]:.0f} N")


# ============================================================================
# PATTERN 2: Solve a System of Nonlinear Equations
# ============================================================================

def pattern_2_nonlinear_system_solver():
    """
    Solve for unknowns given constraints. Example: find the acceleration
    that keeps the motorcycle in a stable turn with a target lateral load transfer.
    """
    print("\n" + "=" * 70)
    print("PATTERN 2: Solve Nonlinear System with Constraints")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    # System parameters
    M = 200
    p = 1.4
    a = 0.6
    h = 0.5
    g = 9.81
    h_bar = h / p
    a_prime = a / p
    b_bar = (p - a) / p
    
    # Constraint: We want the front tire to carry exactly 60% of the weight
    # during cornering. Find what lateral acceleration achieves this.
    
    N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()
    
    def system_equations(a_y_val):
        """
        System of equations. Returns residual (should be zero).
        """
        phi_val = np.arctan(a_y_val / g)
        
        N_r_val = float(N_r_expr.subs([
            (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
            (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
            (model.a_y, a_y_val), (model.phi, phi_val)
        ]).evalf())
        
        N_f_val = float(N_f_expr.subs([
            (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
            (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
            (model.a_y, a_y_val), (model.phi, phi_val)
        ]).evalf())
        
        total_load = N_r_val + N_f_val
        
        # Constraint: N_f / total = 0.60
        residual = (N_f_val / total_load) - 0.60
        
        return residual
    
    # Solve using scipy
    solution = fsolve(system_equations, x0=5.0)
    a_y_solution = solution[0]
    
    # Verify solution
    phi_solution = np.arctan(a_y_solution / g)
    
    N_r_val = float(N_r_expr.subs([
        (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
        (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
        (model.a_y, a_y_solution), (model.phi, phi_solution)
    ]).evalf())
    
    N_f_val = float(N_f_expr.subs([
        (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
        (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
        (model.a_y, a_y_solution), (model.phi, phi_solution)
    ]).evalf())
    
    total = N_r_val + N_f_val
    
    print(f"\nTarget constraint: Front tire carries exactly 60% of weight")
    print(f"\nSolution found:")
    print(f"  Required lateral acceleration: {a_y_solution:.2f} m/s²")
    print(f"  Corresponding roll angle: {np.degrees(phi_solution):.1f}°")
    print(f"  Rear load: {N_r_val:.0f} N ({N_r_val/total*100:.1f}%)")
    print(f"  Front load: {N_f_val:.0f} N ({N_f_val/total*100:.1f}%)")


# ============================================================================
# PATTERN 3: Optimization with Constraints (Friction Ellipse)
# ============================================================================

def pattern_3_friction_ellipse_optimization():
    """
    Find maximum achievable lateral acceleration given friction constraints.
    """
    print("\n" + "=" * 70)
    print("PATTERN 3: Maximize Lateral Acceleration Subject to Friction")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    # Parameters
    M = 200
    p = 1.4
    a = 0.6
    h = 0.5
    g = 9.81
    h_bar = h / p
    b_bar = (p - a) / p
    mu = 1.2  # coefficient of friction
    
    # Get equations
    N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()
    F_r_expr, F_f_expr = model.eq_5_non_dimensional_lateral_forces()
    
    def friction_constraint_violation(a_y_val):
        """
        Returns max(0, utilization_violation) where violation > 0
        means constraints are exceeded.
        """
        phi_val = np.arctan(a_y_val / g)
        
        # Calculate loads
        N_r_val = float(N_r_expr.subs([
            (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
            (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
            (model.a_y, a_y_val), (model.phi, phi_val)
        ]).evalf())
        
        N_f_val = float(N_f_expr.subs([
            (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
            (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
            (model.a_y, a_y_val), (model.phi, phi_val)
        ]).evalf())
        
        if N_r_val < 0 or N_f_val < 0:
            return 1e10  # Impossible configuration
        
        # Calculate forces
        F_r_val = float(F_r_expr.subs([
            (model.a_y, a_y_val), (model.g, g), (model.N_r, N_r_val)
        ]).evalf())
        
        F_f_val = float(F_f_expr.subs([
            (model.a_y, a_y_val), (model.g, g), (model.N_f, N_f_val)
        ]).evalf())
        
        # Friction circle check
        friction_r = np.sqrt(F_r_val**2) / (mu * N_r_val) if N_r_val > 0 else 0
        friction_f = np.sqrt(F_f_val**2) / (mu * N_f_val) if N_f_val > 0 else 0
        
        # Return constraint violation (positive = violated)
        return max(friction_r - 1.0, friction_f - 1.0, 0)
    
    def objective(a_y_val):
        """Minimize negative of a_y (i.e., maximize a_y)"""
        return -a_y_val[0]
    
    def constraint(a_y_val):
        """Constraint function: must be >= 0"""
        return mu - friction_constraint_violation(a_y_val)
    
    # Find maximum a_y using binary search (simpler approach)
    a_y_low, a_y_high = 0, 15
    tolerance = 0.01
    
    while a_y_high - a_y_low > tolerance:
        a_y_mid = (a_y_low + a_y_high) / 2
        
        violation = friction_constraint_violation(a_y_mid)
        
        if violation > 0:
            a_y_high = a_y_mid
        else:
            a_y_low = a_y_mid
    
    a_y_max = a_y_low
    phi_max = np.arctan(a_y_max / g)
    
    print(f"\nMaximum achievable lateral acceleration with μ = {mu}:")
    print(f"  a_y,max = {a_y_max:.2f} m/s² ({a_y_max/g:.2f} g)")
    print(f"  φ,max = {np.degrees(phi_max):.1f}°")


# ============================================================================
# PATTERN 4: Time Integration (Simulation Loop)
# ============================================================================

def pattern_4_time_integration():
    """
    Simple time integration example: simulate a vehicle doing a lane change
    and track how roll angle, loads, and forces evolve.
    """
    print("\n" + "=" * 70)
    print("PATTERN 4: Time Integration / Simulation")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    # Parameters
    M = 200
    p = 1.4
    a = 0.6
    h = 0.5
    g = 9.81
    h_bar = h / p
    b_bar = (p - a) / p
    
    # Get lambdified functions for speed
    N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()
    F_r_expr, F_f_expr = model.eq_5_non_dimensional_lateral_forces()
    
    N_r_func = lambdify(
        (model.M, model.p, model.a_var, model.h_bar, model.b_bar, 
         model.g, model.a_y, model.phi),
        N_r_expr, 'numpy'
    )
    
    N_f_func = lambdify(
        (model.M, model.p, model.a_var, model.h_bar, model.b_bar,
         model.g, model.a_y, model.phi),
        N_f_expr, 'numpy'
    )
    
    # Simulation: lateral acceleration vs time
    # (Simplified: just track kinematics, not full dynamics)
    
    t = np.linspace(0, 3, 300)  # 3 seconds
    
    # Lateral acceleration profile: ramp up to 8 m/s², hold, ramp down
    a_y = np.piecewise(t, 
        [t < 1, (t >= 1) & (t < 2), t >= 2],
        [lambda t: 8 * t, 
         8,
         lambda t: 8 * (3 - t)]
    )
    
    # Roll angle from kinematics
    phi = np.arctan(a_y / g)
    
    # Calculate tire loads
    N_r = N_r_func(M, p, a, h_bar, b_bar, g, a_y, phi)
    N_f = N_f_func(M, p, a, h_bar, b_bar, g, a_y, phi)
    
    # Calculate forces
    F_r = (a_y / g) * N_r
    F_f = (a_y / g) * N_f
    
    print(f"\nSimulation: Lane change maneuver")
    print(f"Duration: {t[-1]:.1f} s")
    print(f"Max lateral acceleration: {np.max(a_y):.2f} m/s²")
    print(f"Max roll angle: {np.degrees(np.max(phi)):.1f}°")
    
    print(f"\nTime-series sample (every 30 points):")
    print(f"{'t (s)':<8} {'a_y (m/s²)':<13} {'φ (°)':<8} {'N_r (N)':<10} {'N_f (N)':<10}")
    print("-" * 50)
    
    for i in range(0, len(t), 30):
        print(f"{t[i]:<8.2f} {a_y[i]:<13.2f} {np.degrees(phi[i]):<8.1f} "
              f"{N_r[i]:<10.0f} {N_f[i]:<10.0f}")


# ============================================================================
# PATTERN 5: Sensitivity Analysis (How do parameters affect results?)
# ============================================================================

def pattern_5_sensitivity_analysis():
    """
    Parametric sensitivity analysis: how does CoM height affect load transfer?
    """
    print("\n" + "=" * 70)
    print("PATTERN 5: Sensitivity Analysis")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    # Fixed parameters
    M = 200
    p = 1.4
    a = 0.6
    g = 9.81
    a_y = 5.0
    
    # Variable: CoM height h
    h_values = np.linspace(0.3, 0.8, 6)
    
    N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()
    
    print(f"\nSensitivity of load transfer to CoM height:")
    print(f"Fixed: M={M} kg, p={p} m, a={a} m, a_y={a_y} m/s²")
    print(f"\n{'h (m)':<8} {'h/p':<8} {'N_r (N)':<12} {'N_f (N)':<12} {'ΔN (N)':<12}")
    print("-" * 55)
    
    for h_val in h_values:
        h_bar = h_val / p
        b_bar = (p - a) / p
        phi_val = np.arctan(a_y / g)
        
        subs_dict = [
            (model.M, M), (model.p, p), (model.a_var, a), (model.h, h_val),
            (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
            (model.a_y, a_y), (model.phi, phi_val)
        ]
        
        N_r_val = float(N_r_expr.subs(subs_dict).evalf())
        N_f_val = float(N_f_expr.subs(subs_dict).evalf())
        delta_N = N_f_val - N_r_val
        
        print(f"{h_val:<8.2f} {h_bar:<8.3f} {N_r_val:<12.0f} {N_f_val:<12.0f} {delta_N:<12.0f}")
    
    print("\nObservation: As CoM height increases, load transfer increases")
    print("(more load goes to the outer tire in a turn)")


# ============================================================================
# PATTERN 6: Export to Other Formats
# ============================================================================

def pattern_6_export_to_formats():
    """
    Export SymPy expressions to different formats for use in other tools.
    """
    print("\n" + "=" * 70)
    print("PATTERN 6: Export Expressions to Different Formats")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    phi_eq = model.eq_3_roll_angle_kinematics()
    N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()
    
    print("\n1. LaTeX format (for papers):")
    print("-" * 55)
    latex_phi = sp.latex(phi_eq)
    print(latex_phi)
    
    print("\n2. C/C++ code generation:")
    print("-" * 55)
    from sympy.printing.ccode import ccode
    
    phi_c = ccode(phi_eq.rhs)
    print(f"phi = {phi_c};")
    
    print("\n3. MATLAB/Simulink code:")
    print("-" * 55)
    from sympy.printing.octave import octave_code
    
    # For simple equation
    phi_matlab = octave_code(sp.atan(model.a_y / model.g))
    print(f"phi = {phi_matlab};")
    
    print("\n4. NumPy/Python code:")
    print("-" * 55)
    from sympy.printing.pycode import pycode
    
    phi_py = pycode(sp.atan(model.a_y / model.g))
    print(f"phi = {phi_py}")


# ============================================================================
# Run all patterns
# ============================================================================

if __name__ == "__main__":
    pattern_1_lambdify_optimization()
    pattern_2_nonlinear_system_solver()
    pattern_3_friction_ellipse_optimization()
    pattern_4_time_integration()
    pattern_5_sensitivity_analysis()
    pattern_6_export_to_formats()
    
    print("\n" + "=" * 70)
    print("Advanced Patterns Complete")
    print("=" * 70)
