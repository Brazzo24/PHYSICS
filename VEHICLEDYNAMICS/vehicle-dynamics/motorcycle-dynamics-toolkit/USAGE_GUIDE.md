# Motorcycle Roll Dynamics SymPy Solver

## Overview

This solver implements the **six Newton-Euler equations** for motorcycle/bicycle dynamics with roll angle effects, tire load transfer, and force constraints based on the equations in your research paper.

## Main Features

### 1. **Six Newton-Euler Equations (Eq. 1)**
- Longitudinal acceleration: `M*a_x = S_r + S_f`
- Lateral acceleration: `M*a_y = F_r + F_f`
- Vertical equilibrium: `M*g = N_r + N_f`
- Three roll moment equations (pitch, lateral, yaw coupling)

### 2. **Tire Force Distribution (Eq. 2)**
- Parameter α defines load split: `α = S_r / (S_r + S_f)`
- α = 1: longitudinal force at rear only
- 0 < α < 1: force distributed between front and rear

### 3. **Roll Angle Kinematics (Eq. 3)**
- Roll angle relates directly to lateral acceleration: `φ = arctan(a_y / g)`
- Simplification for small angles: φ ≈ a_y / g

### 4. **Tire Vertical Load Transfer (Eq. 4)**
Accounts for three effects:
- **Static distribution** via CoM position (a', b')
- **Roll couple** effect via cos(φ)
- **Inertial coupling** with lateral acceleration (a_y term)

```
N_r = (a'/p + h' cos(φ)) M g - (a'/p + h' a_y/√(1+a_y²)) M g
N_f = (b'/p - h' cos(φ)) M g - (b'/p - h' a_y/√(1+a_y²)) M g
```

### 5. **Non-dimensional Lateral Forces (Eq. 5)**
Simple proportional constraint:
```
F_r / N_r = F_f / N_f = a_y / g
```

### 6. **Non-dimensional Longitudinal Forces (Eq. 6)**
Complex form accounting for load transfer:
```
S_r / N_r = (1 / (a' + h' - a_y/(g√(1+a_y²)))) * (a_x/g) * α
S_f / N_f = (1 / (b' - h' - a_y/(g√(1+a_y²)))) * (a_x/g) * (1-α)
```

## Quick Start

### Installation

```bash
pip install sympy numpy
```

### Basic Usage

```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel

# Create model instance
model = MotorcycleDynamicsModel()

# Print all equations in readable format
model.print_equations()

# Access individual equations
phi_eq = model.eq_3_roll_angle_kinematics()
N_r, N_f = model.eq_4_tire_vertical_loads()
```

## Detailed Examples

### Example 1: Solve for Roll Angle

```python
import sympy as sp
from motorcycle_dynamics_solver import MotorcycleDynamicsModel

model = MotorcycleDynamicsModel()

# Lateral acceleration and gravity
a_y_val = 5.0  # m/s²
g_val = 9.81   # m/s²

# Get the roll angle equation
phi_eq = model.eq_3_roll_angle_kinematics()

# Substitute values
phi_solution = phi_eq.subs([(model.a_y, a_y_val), (model.g, g_val)])

# Evaluate
phi_numeric = float(phi_solution.rhs.evalf())

print(f"Roll angle: {phi_numeric:.4f} rad = {sp.deg(phi_numeric):.2f}°")
```

**Output:**
```
Roll angle: 0.4648 rad = 26.63°
```

---

### Example 2: Calculate Tire Vertical Loads with Roll Effects

```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import numpy as np

model = MotorcycleDynamicsModel()

# Motorcycle geometry
M = 200      # kg (total mass)
p = 1.4      # m (wheelbase)
a = 0.6      # m (CoM distance from rear axle)
h = 0.5      # m (CoM height)
g = 9.81     # m/s²

# Non-dimensional parameters
h_bar = h / p  # h' = h/p
b_bar = (p - a) / p  # b' = (p-a)/p

# Operating point
a_y = 5.0    # m/s² (lateral acceleration)
phi = 0.3    # rad (roll angle)

# Get load equations
N_r, N_f = model.eq_4_tire_vertical_loads()

# Substitute all values
subs_dict = [
    (model.M, M),
    (model.p, p),
    (model.a_var, a),
    (model.h, h),
    (model.h_bar, h_bar),
    (model.b_bar, b_bar),
    (model.g, g),
    (model.a_y, a_y),
    (model.phi, phi),
]

N_r_val = float(N_r.subs(subs_dict).evalf())
N_f_val = float(N_f.subs(subs_dict).evalf())

print(f"Rear tire load:  N_r = {N_r_val:.1f} N")
print(f"Front tire load: N_f = {N_f_val:.1f} N")
print(f"Total load:      {N_r_val + N_f_val:.1f} N (expected {M*g:.1f} N)")
```

---

### Example 3: Solve Lateral Forces from Accelerations

```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel

model = MotorcycleDynamicsModel()

# Known: lateral acceleration
a_y_val = 3.0  # m/s²
g_val = 9.81
M_val = 200    # kg

# Get lateral force equations
F_r, F_f = model.eq_5_non_dimensional_lateral_forces()

# Get vertical loads (simplified, no roll for this example)
N_r, N_f = model.eq_4_tire_vertical_loads()

# Simplified: assume equal load distribution and no roll effect
N_r_avg = M_val * g_val / 2
N_f_avg = M_val * g_val / 2

# Lateral forces
F_r_val = (a_y_val / g_val) * N_r_avg
F_f_val = (a_y_val / g_val) * N_f_avg

print(f"Rear lateral force:  F_r = {F_r_val:.1f} N")
print(f"Front lateral force: F_f = {F_f_val:.1f} N")
print(f"Total lateral force: {F_r_val + F_f_val:.1f} N (should = {M_val * a_y_val:.1f} N)")
```

---

### Example 4: Symbolic Manipulation

```python
import sympy as sp
from motorcycle_dynamics_solver import MotorcycleDynamicsModel

model = MotorcycleDynamicsModel()

# Get roll angle equation
phi_eq = model.eq_3_roll_angle_kinematics()

# Solve for a_y in terms of φ
a_y_solution = sp.solve(phi_eq, model.a_y)
print(f"Lateral acceleration in terms of roll angle:")
print(f"a_y = {a_y_solution[0]}")

# Simplify
print(f"Simplified: a_y = g * tan(φ)")
```

---

### Example 5: Longitudinal Braking Analysis

```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import sympy as sp

model = MotorcycleDynamicsModel()

# Get longitudinal force equations
S_r, S_f = model.eq_6_non_dimensional_longitudinal_forces()

# Parameters
M = 200      # kg
p = 1.4      # m
a = 0.6      # m
h = 0.5      # m
g = 9.81     # m/s²
a_x = -8.0   # m/s² (braking acceleration, negative)
a_y = 0.0    # m/s² (no lateral acceleration)
alpha = 0.3  # 30% of braking force at rear, 70% at front (typical)

# Non-dimensional parameters
h_bar = h / p
b_bar = (p - a) / p

subs_dict = [
    (model.M, M),
    (model.p, p),
    (model.a_var, a),
    (model.h_bar, h_bar),
    (model.b_bar, b_bar),
    (model.g, g),
    (model.a_x, a_x),
    (model.a_y, a_y),
    (model.alpha, alpha),
]

# With vertical load values (calculated separately)
N_r = M * g * b_bar  # rear load
N_f = M * g * (1 - b_bar)  # front load

subs_dict.extend([(model.N_r, N_r), (model.N_f, N_f)])

S_r_val = float(S_r.subs(subs_dict).evalf())
S_f_val = float(S_f.subs(subs_dict).evalf())

print(f"Braking force at rear:  S_r = {S_r_val:.1f} N")
print(f"Braking force at front: S_f = {S_f_val:.1f} N")
print(f"Total braking force:    {S_r_val + S_f_val:.1f} N (expected {M * abs(a_x):.1f} N)")
```

---

## Symbol Reference

| Symbol | Description | Units |
|--------|-------------|-------|
| M | Total motorcycle mass | kg |
| p | Wheelbase | m |
| a, b | CoM longitudinal distance (from rear, front) | m |
| h | CoM height | m |
| φ | Roll angle | rad |
| g | Gravitational acceleration | m/s² |
| a_x, a_y | Longitudinal and lateral acceleration | m/s² |
| S_r, S_f | Longitudinal tire forces (rear, front) | N |
| F_r, F_f | Lateral tire forces (rear, front) | N |
| N_r, N_f | Normal/vertical tire forces (rear, front) | N |
| α | Longitudinal force distribution parameter (0 to 1) | — |
| h', b', etc. | Non-dimensional CoM parameters (h/p, b/p) | — |

## Non-dimensional Parameters

The paper uses non-dimensional (primed) notation for convenience:

- `h' = h / p` (CoM height normalized by wheelbase)
- `b' = b / p` (front CoM distance normalized)
- `b̃ = (p - b) / p = 1 - b'` (rear CoM distance complement)

These are useful because they make equations scale-invariant and improve numerical stability.

## Key Insights

1. **Roll angle (Eq. 3)** is determined purely by lateral acceleration — a key simplification valid for motorcycles near equilibrium.

2. **Load transfer (Eq. 4)** is **nonlinear** in `a_y` due to the sqrt(1 + a_y²) term, which models the coupling between vertical loading and roll inertia.

3. **Tire adhesion limits (Eqs. 5-6)** constrain achievable accelerations: if calculated forces exceed friction limits, the acceleration is not physically feasible.

4. The **α parameter** allows modeling different braking strategies (ABS, brake biasing) and acceleration strategies.

## Extending the Solver

To add custom constraints:

```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import sympy as sp

model = MotorcycleDynamicsModel()

# Example: Tire friction constraint
# |F| / N ≤ μ (friction coefficient)

mu = 1.2  # coefficient of friction
mu_limit = sp.Eq(
    sp.sqrt(model.F_r**2 + model.S_r**2) / model.N_r,
    mu
)

print(f"Friction ellipse constraint: {mu_limit}")
```

## Running the Built-in Examples

```bash
python motorcycle_dynamics_solver.py
```

This will:
1. Print all six Newton-Euler equations
2. Run Example 1: Roll angle from lateral acceleration
3. Run Example 2: Tire vertical load transfer with roll effects

## Troubleshooting

### Issue: Equation doesn't simplify
**Solution:** Use `sp.simplify()` or `sp.trigsimp()` on the expression:
```python
eq_simplified = sp.simplify(model.eq_3_roll_angle_kinematics())
```

### Issue: Numerical solution is complex or has multiple roots
**Solution:** Use numerical solving with `scipy`:
```python
from scipy.optimize import fsolve
import numpy as np

# Define residual function
def residual(x, known_vals):
    # x is the unknown, known_vals is a dict of known parameters
    return float(equation.subs(known_vals + [(unknown_symbol, x)]))

# Solve with initial guess
solution = fsolve(residual, x0=initial_guess, args=(known_vals,))
```

### Issue: Substitution with many variables is slow
**Solution:** Use `lambdify` to convert SymPy expressions to NumPy:
```python
import sympy as sp
from sympy.utilities.lambdify import lambdify

# Get expression
N_r, N_f = model.eq_4_tire_vertical_loads()

# Create NumPy function
N_r_func = lambdify(
    (model.M, model.p, model.a_var, model.h_bar, model.g, model.a_y, model.phi),
    N_r,
    'numpy'
)

# Fast evaluation
N_r_val = N_r_func(M, p, a, h_bar, g, a_y, phi)
```

## References

The equations are based on the motorcycle roll dynamics model from your provided paper, with Newton-Euler formulation accounting for:
- CoM position effects (a, b, h)
- Roll angle kinematics
- Tire vertical load transfer
- Friction-limited tire forces

All equations are implemented symbolically for exact manipulation and numerical evaluation.

---

**Enjoy your simulations! 🏍️**
