"""
Interactive Examples for Motorcycle Roll Dynamics Solver
=========================================================

This file contains copy-paste-ready examples you can run in a Python REPL,
Jupyter notebook, or IPython shell.

Each example is self-contained and demonstrates a specific aspect of the solver.
"""

# ============================================================================
# EXAMPLE 1: Print All Equations
# ============================================================================

from motorcycle_dynamics_solver import MotorcycleDynamicsModel

# Initialize the model
model = MotorcycleDynamicsModel()

# Print all equations in LaTeX-friendly format
model.print_equations()

# Output: All six Newton-Euler equations plus Eqs. 2-6 from the paper


# ============================================================================
# EXAMPLE 2: Calculate Roll Angle for Different Lateral Accelerations
# ============================================================================

from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import numpy as np
import matplotlib.pyplot as plt

model = MotorcycleDynamicsModel()
g = 9.81

# Create array of lateral accelerations (in units of g)
a_y_array = np.linspace(0, 1.5, 20) * g

# Calculate corresponding roll angles
phi_array = []
for a_y_val in a_y_array:
    phi_eq = model.eq_3_roll_angle_kinematics()
    phi_solution = phi_eq.subs([(model.a_y, a_y_val), (model.g, g)])
    phi_numeric = float(phi_solution.rhs.evalf())
    phi_array.append(np.degrees(phi_numeric))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(a_y_array / g, phi_array, 'b-', linewidth=2, label='φ = arctan(a_y/g)')
plt.xlabel('Lateral Acceleration (g)')
plt.ylabel('Roll Angle (degrees)')
plt.title('Motorcycle Roll Angle vs. Lateral Acceleration')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('roll_angle_vs_lateral_accel.png', dpi=150)
plt.show()

print("Plot saved as 'roll_angle_vs_lateral_accel.png'")


# ============================================================================
# EXAMPLE 3: Tire Load Transfer Analysis (No Roll)
# ============================================================================

from motorcycle_dynamics_solver import MotorcycleDynamicsModel

model = MotorcycleDynamicsModel()

# Motorcycle parameters (typical sport bike)
M = 200      # kg
p = 1.4      # m
a = 0.6      # m (CoM ~57% of wheelbase from rear)
h = 0.5      # m
g = 9.81

# Non-dimensional parameters
h_bar = h / p
a_prime = a / p
b_bar = (p - a) / p

print("=" * 60)
print("TIRE LOAD TRANSFER ANALYSIS")
print("=" * 60)
print(f"\nMotorcycle parameters:")
print(f"  Mass: {M} kg")
print(f"  Wheelbase: {p} m")
print(f"  CoM distance from rear: {a} m ({a_prime:.1%})")
print(f"  CoM height: {h} m")
print(f"\nNon-dimensional parameters:")
print(f"  h' = {h_bar:.3f}")
print(f"  a' = {a_prime:.3f}")
print(f"  b' = {b_bar:.3f}")

# Static load distribution (no dynamics)
N_r_static = M * g * b_bar
N_f_static = M * g * a_prime

print(f"\nStatic load distribution (φ=0, a_y=0):")
print(f"  N_r = {N_r_static:.0f} N ({N_r_static / (M*g) * 100:.1f}%)")
print(f"  N_f = {N_f_static:.0f} N ({N_f_static / (M*g) * 100:.1f}%)")
print(f"  Total: {N_r_static + N_f_static:.0f} N")

# Now analyze load transfer with varying lateral acceleration
print(f"\nLoad transfer with varying lateral acceleration:")
print(f"{'a_y (m/s²)':<15} {'a_y/g':<10} {'N_r (N)':<12} {'N_f (N)':<12} {'ΔN (N)':<12}")
print("-" * 60)

N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()

a_y_values = [0, 2.5, 5.0, 7.5, 10.0]
phi = 0  # No roll for this example

for a_y_val in a_y_values:
    subs_dict = [
        (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
        (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
        (model.a_y, a_y_val), (model.phi, phi)
    ]
    
    N_r_val = float(N_r_expr.subs(subs_dict).evalf())
    N_f_val = float(N_f_expr.subs(subs_dict).evalf())
    delta_N = N_f_val - N_r_val
    
    print(f"{a_y_val:<15.1f} {a_y_val/g:<10.3f} {N_r_val:<12.0f} {N_f_val:<12.0f} {delta_N:<12.0f}")


# ============================================================================
# EXAMPLE 4: Longitudinal Force Distribution During Braking
# ============================================================================

from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import numpy as np

model = MotorcycleDynamicsModel()

# Parameters
M = 200
p = 1.4
a = 0.6
h = 0.5
g = 9.81
h_bar = h / p
b_bar = (p - a) / p

print("\n" + "=" * 60)
print("LONGITUDINAL FORCE DISTRIBUTION (BRAKING)")
print("=" * 60)

# Get equations
S_r_expr, S_f_expr = model.eq_6_non_dimensional_longitudinal_forces()

# Simulate braking from 0 to 10 m/s² deceleration
a_x_values = np.linspace(0, -10, 6)

print(f"\nBraking force distribution (α = 0.3, a_y = 0):")
print(f"{'a_x (m/s²)':<15} {'S_r (N)':<12} {'S_f (N)':<12} {'S_r/(S_r+S_f)':<15}")
print("-" * 55)

a_y = 0
alpha = 0.3

# Calculate vertical loads
N_r = M * g * b_bar
N_f = M * g * a_prime

for a_x_val in a_x_values:
    subs_dict = [
        (model.a_x, a_x_val), (model.a_y, a_y), (model.alpha, alpha),
        (model.p, p), (model.a_var, a), (model.h_bar, h_bar), 
        (model.b_bar, b_bar), (model.g, g),
        (model.N_r, N_r), (model.N_f, N_f)
    ]
    
    try:
        S_r_val = float(S_r_expr.subs(subs_dict).evalf())
        S_f_val = float(S_f_expr.subs(subs_dict).evalf())
        ratio = S_r_val / (S_r_val + S_f_val) if (S_r_val + S_f_val) != 0 else 0
        
        print(f"{a_x_val:<15.1f} {S_r_val:<12.0f} {S_f_val:<12.0f} {ratio:<15.3f}")
    except:
        print(f"{a_x_val:<15.1f} {'(error)':<12} {'(error)':<12} {'(error)':<15}")


# ============================================================================
# EXAMPLE 5: Combined Acceleration + Lateral Acceleration (Cornering + Acceleration)
# ============================================================================

from motorcycle_dynamics_solver import MotorcycleDynamicsModel

model = MotorcycleDynamicsModel()

# Parameters
M = 200
p = 1.4
a = 0.6
h = 0.5
g = 9.81
h_bar = h / p
a_prime = a / p
b_bar = (p - a) / p

print("\n" + "=" * 60)
print("COMBINED MANEUVER: ACCELERATION + LEAN")
print("=" * 60)

# Scenario: Exiting a corner with hard throttle
a_x = 5.0   # m/s² (acceleration)
a_y = 8.0   # m/s² (lateral acceleration in corner)

# Get equations
N_r_expr, N_f_expr = model.eq_4_tire_vertical_loads()
phi_eq = model.eq_3_roll_angle_kinematics()
F_r_expr, F_f_expr = model.eq_5_non_dimensional_lateral_forces()

# Calculate roll angle
phi_solution = phi_eq.subs([(model.a_y, a_y), (model.g, g)])
phi_val = float(phi_solution.rhs.evalf())

print(f"\nManeuver: a_x = {a_x} m/s², a_y = {a_y} m/s²")
print(f"Roll angle: φ = {np.degrees(phi_val):.1f}°")

# Calculate loads
subs_dict = [
    (model.M, M), (model.p, p), (model.a_var, a), (model.h, h),
    (model.h_bar, h_bar), (model.b_bar, b_bar), (model.g, g),
    (model.a_y, a_y), (model.phi, phi_val)
]

N_r_val = float(N_r_expr.subs(subs_dict).evalf())
N_f_val = float(N_f_expr.subs(subs_dict).evalf())

print(f"\nVertical tire loads:")
print(f"  N_r = {N_r_val:.0f} N")
print(f"  N_f = {N_f_val:.0f} N")
print(f"  Total = {N_r_val + N_f_val:.0f} N (expected: {M*g:.0f} N)")

# Calculate lateral forces
subs_dict_forces = subs_dict + [(model.N_r, N_r_val), (model.N_f, N_f_val)]

F_r_val = float(F_r_expr.subs(subs_dict_forces).evalf())
F_f_val = float(F_f_expr.subs(subs_dict_forces).evalf())

print(f"\nLateral tire forces:")
print(f"  F_r = {F_r_val:.0f} N")
print(f"  F_f = {F_f_val:.0f} N")
print(f"  Total = {F_r_val + F_f_val:.0f} N (expected: {M * a_y:.0f} N)")

# Check friction circle constraints (assuming μ = 1.2)
mu = 1.2
F_max_r = mu * N_r_val
F_max_f = mu * N_f_val

print(f"\nFriction limit check (μ = {mu}):")
print(f"  Max lateral force at rear: F_r,max = {F_max_r:.0f} N (actual {F_r_val:.0f} N)")
print(f"  Max lateral force at front: F_f,max = {F_max_f:.0f} N (actual {F_f_val:.0f} N)")

# If we apply longitudinal force too:
alpha = 0.5
S_r_expr, S_f_expr = model.eq_6_non_dimensional_longitudinal_forces()

subs_dict_long = subs_dict_forces + [(model.a_x, a_x), (model.alpha, alpha)]

try:
    S_r_val = float(S_r_expr.subs(subs_dict_long).evalf())
    S_f_val = float(S_f_expr.subs(subs_dict_long).evalf())
    
    print(f"\nLongitudinal tire forces (α = {alpha}):")
    print(f"  S_r = {S_r_val:.0f} N")
    print(f"  S_f = {S_f_val:.0f} N")
    print(f"  Total = {S_r_val + S_f_val:.0f} N (expected: {M * a_x:.0f} N)")
    
    # Friction ellipse check
    friction_r = np.sqrt((F_r_val**2 + S_r_val**2)) / (mu * N_r_val)
    friction_f = np.sqrt((F_f_val**2 + S_f_val**2)) / (mu * N_f_val)
    
    print(f"\nFriction ellipse utilization:")
    print(f"  Rear:  {friction_r:.2%} of available friction")
    print(f"  Front: {friction_f:.2%} of available friction")
    
    if friction_r > 1.0 or friction_f > 1.0:
        print("\n⚠️  WARNING: Tire grip limit exceeded! Maneuver not physically achievable.")
    else:
        print("\n✓ Maneuver is within tire adhesion limits.")
except Exception as e:
    print(f"Could not calculate longitudinal forces: {e}")


# ============================================================================
# EXAMPLE 6: Create Custom Friction Constraint
# ============================================================================

from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import sympy as sp

print("\n" + "=" * 60)
print("CUSTOM CONSTRAINT: FRICTION ELLIPSE")
print("=" * 60)

model = MotorcycleDynamicsModel()

# Define friction constraint: sqrt(F² + S²) ≤ μ * N
mu = sp.Symbol('mu', positive=True)

# Friction ellipse constraint at rear
friction_constraint_r = sp.sqrt(
    model.F_r**2 + model.S_r**2
) - mu * model.N_r

print(f"\nFriction ellipse constraint (rear tire):")
print(f"√(F_r² + S_r²) ≤ μ·N_r")
print(f"\nSymbolic form:")
print(f"{friction_constraint_r} ≤ 0")


# ============================================================================
# EXAMPLE 7: Simplify and Expand Expressions
# ============================================================================

from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import sympy as sp

print("\n" + "=" * 60)
print("SYMBOLIC MANIPULATION EXAMPLES")
print("=" * 60)

model = MotorcycleDynamicsModel()

# Get the roll angle equation
phi_eq = model.eq_3_roll_angle_kinematics()
print(f"\nOriginal roll angle equation:")
print(f"  {phi_eq}")

# Solve for a_y
a_y_solved = sp.solve(phi_eq, model.a_y)
print(f"\nSolve for a_y:")
print(f"  a_y = {a_y_solved[0]}")

# Take derivative of φ with respect to a_y
phi_solution = phi_eq.rhs
dphi_day = sp.diff(phi_solution, model.a_y)
print(f"\nDerivative dφ/da_y:")
print(f"  {dphi_day}")
print(f"  Simplified: {sp.simplify(dphi_day)}")

# Get lateral force equation
F_r, F_f = model.eq_5_non_dimensional_lateral_forces()

# Solve for a_y in terms of lateral force
a_y_from_F = sp.solve(F_r - 0, model.a_y)  # Setting F_r = 0 → a_y = 0
print(f"\nLateral force equation:")
print(f"  F_r = {F_r}")

# Substitute φ from Eq. 3 into Eq. 5
phi_solution = sp.atan(model.a_y / model.g)
# (Can't directly substitute because Eq. 5 doesn't have φ, but demonstrates the approach)

print("\n" + "=" * 60)
print("End of Examples")
print("=" * 60)
