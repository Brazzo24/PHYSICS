# Motorcycle Roll Dynamics Solver – Complete Package

**A SymPy-based tool for symbolic and numerical analysis of motorcycle/bicycle roll dynamics equations from your research paper.**

---

## 📦 Package Contents

### Main Module
- **`motorcycle_dynamics_solver.py`** – Complete SymPy implementation of all six Newton-Euler equations plus tire constraints

### Documentation
- **`USAGE_GUIDE.md`** – Comprehensive guide with theory and basic examples
- **`INTERACTIVE_EXAMPLES.py`** – Copy-paste ready examples for quick exploration
- **`ADVANCED_PATTERNS.py`** – Advanced techniques (optimization, sensitivity analysis, integration)
- **`README.md`** – This file

---

## 🚀 Quick Start (30 seconds)

```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel

# Create model
model = MotorcycleDynamicsModel()

# Print all equations
model.print_equations()

# Access individual equations
phi_eq = model.eq_3_roll_angle_kinematics()  # Roll angle kinematics
N_r, N_f = model.eq_4_tire_vertical_loads()  # Vertical load transfer
```

---

## 📚 Equations Implemented

| Equation | Description | Function |
|----------|-------------|----------|
| **Eq. 1** | Six Newton-Euler equations | `eq_group_1_newton_euler()` |
| **Eq. 2** | Tire force distribution parameter | `eq_2_tire_force_distribution()` |
| **Eq. 3** | Roll angle kinematics | `eq_3_roll_angle_kinematics()` |
| **Eq. 4** | Tire vertical load transfer | `eq_4_tire_vertical_loads()` |
| **Eq. 5** | Non-dimensional lateral forces | `eq_5_non_dimensional_lateral_forces()` |
| **Eq. 6** | Non-dimensional longitudinal forces | `eq_6_non_dimensional_longitudinal_forces()` |

All equations from pages 1–3 of your research paper.

---

## 🎯 Common Use Cases

### 1. **Understand Roll Dynamics**
```python
model = MotorcycleDynamicsModel()
phi = model.eq_3_roll_angle_kinematics()
# Roll angle = arctan(lateral_acceleration / g)
```

### 2. **Calculate Tire Loads**
```python
N_r, N_f = model.eq_4_tire_vertical_loads()
# Includes static distribution + roll couple + inertial effects
```

### 3. **Find Maximum Cornering Speed**
```python
# See ADVANCED_PATTERNS.py for friction ellipse optimization
```

### 4. **Simulate a Maneuver**
```python
# See INTERACTIVE_EXAMPLES.py Example 5 (combined acceleration + lean)
```

### 5. **Parametric Analysis**
```python
# See ADVANCED_PATTERNS.py Pattern 5 (sensitivity to CoM height)
```

---

## 🔧 Installation

### Prerequisites
```bash
pip install sympy numpy scipy matplotlib
```

### Import
```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
```

---

## 📖 Getting Started by Topic

### **If you want to...**

**...understand the physics**
→ Start with `USAGE_GUIDE.md` sections "Key Insights"

**...try examples quickly**
→ Run `INTERACTIVE_EXAMPLES.py` (pick Example 1, 2, or 5)

**...solve specific problems**
→ Search `ADVANCED_PATTERNS.py` by pattern name

**...integrate into your code**
→ See the `MotorcycleDynamicsModel` class in `motorcycle_dynamics_solver.py`

**...manipulate equations symbolically**
→ Use SymPy methods directly on model equations

---

## 🧮 Core Class: `MotorcycleDynamicsModel`

### Constructor
```python
model = MotorcycleDynamicsModel()
```
Initializes all symbolic variables (no arguments needed).

### Methods

#### Get Equations
```python
# Individual equations
phi_eq = model.eq_3_roll_angle_kinematics()
N_r, N_f = model.eq_4_tire_vertical_loads()

# All Newton-Euler equations
eqs = model.eq_group_1_newton_euler()  # Returns list of 6 equations
```

#### Symbolic Variables
Access via attributes:
```python
model.M        # Mass
model.p        # Wheelbase
model.a_var    # CoM distance from rear (a)
model.h        # CoM height
model.phi      # Roll angle
model.a_x, model.a_y  # Accelerations
model.g        # Gravity
# ... and many more
```

#### Print All Equations
```python
model.print_equations()  # Pretty-printed output
```

---

## 💡 Working with Equations

### Substitute Known Values
```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import sympy as sp

model = MotorcycleDynamicsModel()

# Get an equation
phi_eq = model.eq_3_roll_angle_kinematics()

# Substitute values
solution = phi_eq.subs([
    (model.a_y, 5.0),      # 5 m/s²
    (model.g, 9.81)        # 9.81 m/s²
])

# Evaluate numerically
phi_numeric = float(solution.rhs.evalf())
print(f"Roll angle: {phi_numeric:.4f} rad")
```

### Solve Algebraically
```python
# Solve for a_y in terms of φ
phi_eq = model.eq_3_roll_angle_kinematics()
a_y_solution = sp.solve(phi_eq, model.a_y)
print(a_y_solution[0])  # a_y = g * tan(φ)
```

### Take Derivatives
```python
phi_eq = model.eq_3_roll_angle_kinematics()
dphi_day = sp.diff(phi_eq.rhs, model.a_y)
# Shows sensitivity: dφ/da_y
```

### Simplify
```python
expr = some_complicated_expression
simplified = sp.simplify(expr)
# Or: sp.trigsimp(expr), sp.expand(expr), etc.
```

---

## ⚡ Performance Tips

### For Repeated Numerical Evaluation
Use `lambdify` to convert to NumPy functions (100–1000× faster):

```python
from sympy.utilities.lambdify import lambdify

N_r_func = lambdify(
    (model.M, model.p, model.a_var, model.h_bar, 
     model.b_bar, model.g, model.a_y, model.phi),
    N_r_expr,
    'numpy'
)

# Fast vectorized evaluation
N_r_values = N_r_func(M, p, a, h_bar, b_bar, g, a_y_array, phi_array)
```

See `ADVANCED_PATTERNS.py` Pattern 1 for a complete example.

---

## 📊 Typical Workflow

1. **Define motorcycle geometry** (M, p, a, h)
2. **Get equation** from model
3. **Substitute known values** (a_y, g, etc.)
4. **Evaluate numerically** or solve symbolically
5. **Check constraints** (friction, wheel contact, etc.)
6. **Visualize results** (use matplotlib)

Example:
```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import matplotlib.pyplot as plt
import numpy as np

model = MotorcycleDynamicsModel()

# Parameters
M, p, a, h, g = 200, 1.4, 0.6, 0.5, 9.81
h_bar, b_bar = h/p, (p-a)/p

# Lateral acceleration range
a_y_vals = np.linspace(0, 10, 50)

# Calculate roll angles
phi_vals = np.arctan(a_y_vals / g)

# Plot
plt.plot(a_y_vals, np.degrees(phi_vals), 'b-', linewidth=2)
plt.xlabel('Lateral Acceleration (m/s²)')
plt.ylabel('Roll Angle (°)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🔗 Related Work

Your model incorporates:
- **Newton-Euler formulation** for rigid body dynamics
- **Pacejka-style tire models** (force constraints)
- **Roll couple geometry** (height effect on load transfer)
- **Friction ellipse** (combined longitudinal/lateral adhesion limits)

This is a foundation for:
- Path-tracking control design (like your Stanley Controller work)
- Stability analysis and robustness testing
- Hardware-in-the-loop validation on Arduino platforms
- Integration with your chicane/Mozzi-axis analysis

---

## 📝 Examples by Complexity

| Complexity | Example | File |
|-----------|---------|------|
| **Beginner** | Print equations | `motorcycle_dynamics_solver.py` main |
| **Beginner** | Roll angle from lateral accel | `INTERACTIVE_EXAMPLES.py` Ex. 1 |
| **Intermediate** | Tire load transfer | `INTERACTIVE_EXAMPLES.py` Ex. 2 |
| **Intermediate** | Braking force distribution | `INTERACTIVE_EXAMPLES.py` Ex. 4 |
| **Advanced** | Friction-constrained optimization | `ADVANCED_PATTERNS.py` Pattern 3 |
| **Advanced** | Time integration (simulation) | `ADVANCED_PATTERNS.py` Pattern 4 |
| **Advanced** | Sensitivity analysis | `ADVANCED_PATTERNS.py` Pattern 5 |

---

## 🛠️ Extending the Solver

Add custom equations or constraints:

```python
from motorcycle_dynamics_solver import MotorcycleDynamicsModel
import sympy as sp

model = MotorcycleDynamicsModel()

# Define a new constraint (e.g., suspension travel limit)
suspension_limit = sp.Eq(model.phi, sp.pi / 4)  # Max 45° roll

# Or add a custom force balance
my_constraint = sp.Eq(
    model.S_r + model.S_f,
    model.M * model.a_x
)

# Include in system solving
# (See ADVANCED_PATTERNS.py Pattern 2 for full example)
```

---

## 🐛 Troubleshooting

### Q: Expression is too complicated to work with
**A:** Use `sp.simplify()`, `sp.trigsimp()`, or break into sub-expressions

### Q: Substitution is very slow
**A:** Use `lambdify()` instead (100–1000× faster for repeated evaluation)

### Q: Getting complex numbers in numerical solution
**A:** Check that inputs are physically meaningful (positive mass, gravity, etc.)

### Q: Want to integrate into Simulink/Matlab
**A:** Export using `sympy.printing.octave.octave_code()` (see ADVANCED_PATTERNS.py Pattern 6)

---

## 📚 Theory References

For detailed background on:
- **Motorcycle dynamics**: Your paper (page 1-3)
- **SymPy basics**: https://docs.sympy.org/
- **Numerical optimization**: scipy.optimize documentation
- **Tire models**: Pacejka, "Tire and Vehicle Dynamics"

---

## 📄 License & Attribution

This solver implements the equations from your research paper. 
- Developed for: **Dominik's vehicle dynamics research**
- Focus: Motorbike roll dynamics, chicane analysis, control design
- Status: Active development for integration with Arduino-based hardware

---

## 🎓 Learning Objectives

After working through this package, you should understand:

✅ How to represent engineering equations symbolically in Python  
✅ How motorcycle roll dynamics couples longitudinal, lateral, and vertical dynamics  
✅ Why tire load transfer is nonlinear and speed-dependent  
✅ How to solve constrained optimization problems (e.g., maximum cornering speed)  
✅ How to balance symbolic precision with numerical efficiency  

---

## 🚀 Next Steps

1. **Run the solver:** `python motorcycle_dynamics_solver.py`
2. **Explore examples:** `python INTERACTIVE_EXAMPLES.py`
3. **Try advanced techniques:** `python ADVANCED_PATTERNS.py`
4. **Integrate into your project:** Import the `MotorcycleDynamicsModel` class
5. **Extend with custom constraints:** Add your own equations and solver methods

---

## 📧 Notes for Your Research

This solver is designed to support your work on:
- **Roll dynamics simulation** in chicane maneuvers
- **Mozzi axis analysis** and instantaneous screw axis calculations
- **Path-tracking control** (integration with your Stanley Controller implementation)
- **Rear tire load loss** diagnostics during rapid cornering
- **Hardware validation** on Arduino-based motorcycle platforms

The symbolic foundation makes it easy to:
- Export equations to embedded C code
- Perform sensitivity studies
- Validate control law design
- Benchmark against experimental data

---

**Happy simulating! 🏍️**

For questions or extensions, revisit the equations in your paper and adapt the SymPy methods shown in this package.
