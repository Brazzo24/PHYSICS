"""
Motorcycle Roll Dynamics Equation Solver
=========================================

This module provides SymPy-based solvers for the Newton-Euler equations
governing motorcycle roll dynamics, including:
- Longitudinal, lateral, and vertical acceleration equations
- Roll angle dynamics
- Tire load transfer
- Tire force constraints

Based on the six Newton-Euler equations (Eq. 1) and related constraint equations
for motorcycle/bicycle model dynamics.
"""

import sympy as sp
from sympy import symbols, sin, cos, sqrt, atan, simplify, solve, Eq
from sympy import Matrix, diff
from typing import Dict, List, Tuple, Optional
import numpy as np


class MotorcycleDynamicsModel:
    """
    Comprehensive motorcycle dynamics model with roll angle effects.
    
    Implements the Newton-Euler equations with:
    - Center of mass (CoM) position effects (a, b, h coordinates)
    - Roll angle (φ) and its dynamics
    - Tire vertical load transfer
    - Tire force constraints (friction ellipse)
    """
    
    def __init__(self):
        """Initialize symbolic variables."""
        # State variables
        self.M, self.p, self.b_var, self.h = symbols('M p b h', real=True, positive=True)
        self.a_var = symbols('a', real=True, positive=True)
        self.phi = symbols('phi', real=True)  # roll angle
        self.g = symbols('g', real=True, positive=True)
        
        # Accelerations
        self.a_x = symbols('a_x', real=True)  # longitudinal acceleration
        self.a_y = symbols('a_y', real=True)  # lateral acceleration
        
        # Tire forces
        self.S_r, self.S_f = symbols('S_r S_f', real=True)  # longitudinal forces (rear, front)
        self.F_r, self.F_f = symbols('F_r F_f', real=True)  # lateral forces (rear, front)
        self.N_r, self.N_f = symbols('N_r N_f', real=True, positive=True)  # normal forces
        
        # Tire force distribution parameter
        self.alpha = symbols('alpha', real=True)
        
        # Non-dimensional parameters
        self.h_bar = symbols('h_bar', real=True)  # h'/p (non-dimensional height)
        self.b_bar = symbols('b_bar', real=True)  # b'/p (non-dimensional rear position)
        self.b_prime_bar = symbols('b_prime_bar', real=True)  # b̃ = (p-b)/p = 1 - b'
        
        # Auxiliary acceleration parameter
        self.a_l = symbols('a_l', real=True)  # lateral acceleration component
        
    def eq_group_1_newton_euler(self) -> List[sp.Expr]:
        """
        Return the six Newton-Euler equations (Equation 1).
        
        Returns:
            List of six equations in symbolic form (not set to zero)
        """
        equations = [
            self.M * self.a_x - (self.S_r + self.S_f),  # longitudinal
            self.M * self.a_y - (self.F_r + self.F_f),  # lateral
            self.M * self.g - (self.N_r + self.N_f),    # vertical
            # Roll moment equations
            self.M * self.a_x * self.h * cos(self.phi) - self.M * self.g * self.h * sin(self.phi),
            self.M * self.a_y * self.h * cos(self.phi) - self.h * self.N_r + (self.p - self.b_var) * self.N_f,
            self.M * self.a_x * self.h * sin(self.phi) - self.b_var * self.F_r - (self.p - self.b_var) * self.F_f,
        ]
        return equations
    
    def eq_2_tire_force_distribution(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        Equation 2: Tire force distribution parameter.
        
        α = S_r / (S_r + S_f)
        
        Returns:
            Tuple of (alpha_definition, constraint_equation)
        """
        alpha_def = self.S_r / (self.S_r + self.S_f)
        constraint = Eq(self.alpha, alpha_def)
        return alpha_def, constraint
    
    def eq_3_roll_angle_kinematics(self) -> sp.Expr:
        """
        Equation 3: Roll angle kinematics (motorcycle roll depends on lateral acceleration).
        
        φ = arctan(a_y / g)
        
        Returns:
            Equation relating roll angle to lateral acceleration
        """
        phi_expr = atan(self.a_y / self.g)
        return Eq(self.phi, phi_expr)
    
    def eq_4_tire_vertical_loads(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        Equation 4: Tire vertical load transfer.
        
        From the paper (Eq. 4):
        N_r = (a' + h' cos(φ)) M g - (a' + h' * a_y/sqrt(1 + a_y²)) M g
        N_f = (b' - h' cos(φ)) M g - (b' - h' * a_y/sqrt(1 + a_y²)) M g
        
        Where a' = a/p, b' = b'/p, h' = h/p
        
        This accounts for:
        - Static load distribution (a', b')
        - Roll-axis couple effect (cos(φ) term)
        - Inertial coupling with roll (a_y term via sqrt(1+a_y²))
        
        Returns:
            Tuple of (N_r expression, N_f expression)
        """
        # The form from Equation 4 in the paper
        # Note: This is written as sum of two effects
        # Static/roll geometry term: (a' ± h'cos(φ)) M g
        # Inertial roll-acceleration coupling: (a' ± h' a_y/sqrt(1+a_y²)) M g
        
        N_r_expr = (self.a_var/self.p + self.h_bar * cos(self.phi)) * self.M * self.g - \
                   (self.a_var/self.p + self.h_bar * self.a_y / sqrt(1 + self.a_y**2)) * self.M * self.g
        
        N_f_expr = (self.b_bar - self.h_bar * cos(self.phi)) * self.M * self.g - \
                   (self.b_bar - self.h_bar * self.a_y / sqrt(1 + self.a_y**2)) * self.M * self.g
        
        # Simplify: can be rewritten as
        # N_r = M*g * h_bar * (cos(φ) - a_y/sqrt(1+a_y²))
        # N_f = M*g * (-h_bar) * (cos(φ) - a_y/sqrt(1+a_y²))
        
        return N_r_expr, N_f_expr
    
    def eq_5_non_dimensional_lateral_forces(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        Equation 5: Non-dimensional lateral tire forces (simple form).
        
        F_r / N_r = F_f / N_f = a_y / g
        
        Returns:
            Tuple of (F_r expression, F_f expression)
        """
        F_r_expr = (self.a_y / self.g) * self.N_r
        F_f_expr = (self.a_y / self.g) * self.N_f
        
        return F_r_expr, F_f_expr
    
    def eq_6_non_dimensional_longitudinal_forces(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        Equation 6: Non-dimensional longitudinal tire forces (complex form with load transfer).
        
        For rear tire (acceleration, α=1):
        S_r / N_r = (1 / (a' + h' - a_y/(g*sqrt(1+a_y²)))) * (a_x/g) * α
        
        For front tire (acceleration, α=1):
        S_f / N_f = (1 / (b' - h' - a_y/(g*sqrt(1+a_y²)))) * (a_x/g) * (1-α)
        
        For braking, 0 ≤ α < 1 and a_x < 0.
        
        Returns:
            Tuple of (S_r expression, S_f expression)
        """
        a_prime = self.a_var / self.p
        
        # Denominator terms with non-dimensional CoM height effect
        denom_r = a_prime + self.h_bar - self.a_y / (self.g * sqrt(1 + self.a_y**2))
        denom_f = self.b_bar - self.h_bar - self.a_y / (self.g * sqrt(1 + self.a_y**2))
        
        S_r_expr = (1 / denom_r) * (self.a_x / self.g) * self.alpha * self.N_r
        S_f_expr = (1 / denom_f) * (self.a_x / self.g) * (1 - self.alpha) * self.N_f
        
        return S_r_expr, S_f_expr
    
    def solve_system(self, known_values: Dict, solve_for: List[str]) -> Dict:
        """
        Solve the system of equations for specified unknowns.
        
        Parameters:
            known_values: Dictionary of {symbol: numerical_value}
            solve_for: List of variable names to solve for
        
        Returns:
            Dictionary of solutions
        """
        # Build equations
        eqs = self.eq_group_1_newton_euler()
        
        # Convert known values dict
        subs_dict = {}
        for key, val in known_values.items():
            if isinstance(key, str):
                # Map string to symbol
                attr = getattr(self, key.split('_')[0], None)
                if attr is None:
                    # Try direct lookup
                    for sym in self.free_symbols:
                        if sym.name == key:
                            subs_dict[sym] = val
                            break
            else:
                subs_dict[key] = val
        
        # Solve
        try:
            solutions = solve(eqs, solve_for, dict=True)
            return solutions
        except Exception as e:
            print(f"Error solving system: {e}")
            return {}
    
    def print_equations(self):
        """Print all equations in readable format."""
        print("=" * 70)
        print("MOTORCYCLE ROLL DYNAMICS EQUATIONS")
        print("=" * 70)
        
        print("\n[EQ. 1] Newton-Euler Equations (6 equations):")
        print("-" * 70)
        eqs = self.eq_group_1_newton_euler()
        labels = ["Longitudinal", "Lateral", "Vertical", 
                  "Roll moment (pitch)", "Roll moment (lateral)", "Roll moment (yaw)"]
        for i, (eq, label) in enumerate(zip(eqs, labels)):
            print(f"{i+1}. {label}:")
            print(f"   {eq} = 0\n")
        
        print("\n[EQ. 2] Tire Force Distribution:")
        print("-" * 70)
        _, alpha_eq = self.eq_2_tire_force_distribution()
        print(f"{alpha_eq}\n")
        
        print("\n[EQ. 3] Roll Angle Kinematics:")
        print("-" * 70)
        phi_eq = self.eq_3_roll_angle_kinematics()
        print(f"{phi_eq}\n")
        
        print("\n[EQ. 4] Tire Vertical Load Transfer:")
        print("-" * 70)
        N_r, N_f = self.eq_4_tire_vertical_loads()
        print(f"N_r = {N_r}")
        print(f"N_f = {N_f}\n")
        
        print("\n[EQ. 5] Non-dimensional Lateral Forces:")
        print("-" * 70)
        F_r, F_f = self.eq_5_non_dimensional_lateral_forces()
        print(f"F_r = {F_r}")
        print(f"F_f = {F_f}\n")
        
        print("\n[EQ. 6] Non-dimensional Longitudinal Forces:")
        print("-" * 70)
        S_r, S_f = self.eq_6_non_dimensional_longitudinal_forces()
        print(f"S_r = {S_r}")
        print(f"S_f = {S_f}\n")


def example_solve_for_roll_angle():
    """Example: Solve for roll angle given lateral acceleration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Roll Angle from Lateral Acceleration")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    # Given: a_y = 0.5 m/s², g = 9.81 m/s²
    a_y_val = 0.5
    g_val = 9.81
    
    phi_eq = model.eq_3_roll_angle_kinematics()
    
    # Substitute known values
    phi_solution = phi_eq.subs([(model.a_y, a_y_val), (model.g, g_val)])
    
    print(f"\nGiven: a_y = {a_y_val} m/s², g = {g_val} m/s²")
    print(f"\nEquation: {phi_eq}")
    print(f"\nSolution: {phi_solution}")
    
    # Evaluate numerically
    phi_numeric = float(phi_solution.rhs.evalf())
    print(f"φ = {phi_numeric:.4f} rad = {np.degrees(phi_numeric):.2f}°")


def example_tire_load_transfer():
    """Example: Calculate tire vertical loads with roll effects."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Tire Vertical Load Transfer")
    print("=" * 70)
    
    model = MotorcycleDynamicsModel()
    
    # Motorcycle parameters
    M_val = 200  # kg
    p_val = 1.4  # m (wheelbase)
    a_val = 0.6  # m (CoM longitudinal distance from rear)
    h_val = 0.5  # m (CoM height)
    g_val = 9.81  # m/s²
    
    # Operating conditions
    a_y_val = 5.0  # m/s² (lateral acceleration)
    phi_val = 0.3  # rad (roll angle)
    
    N_r, N_f = model.eq_4_tire_vertical_loads()
    
    # Define non-dimensional parameters
    h_bar_val = h_val / p_val
    b_bar_val = (p_val - a_val) / p_val
    
    subs_values = [
        (model.M, M_val),
        (model.p, p_val),
        (model.a_var, a_val),
        (model.h, h_val),
        (model.h_bar, h_bar_val),
        (model.b_bar, b_bar_val),
        (model.g, g_val),
        (model.a_y, a_y_val),
        (model.phi, phi_val),
    ]
    
    N_r_numeric = N_r.subs(subs_values).evalf()
    N_f_numeric = N_f.subs(subs_values).evalf()
    
    print(f"\nMotorcycle parameters:")
    print(f"  Mass M = {M_val} kg")
    print(f"  Wheelbase p = {p_val} m")
    print(f"  CoM distance from rear a = {a_val} m")
    print(f"  CoM height h = {h_val} m")
    
    print(f"\nOperating conditions:")
    print(f"  Lateral acceleration a_y = {a_y_val} m/s²")
    print(f"  Roll angle φ = {phi_val:.3f} rad = {np.degrees(phi_val):.2f}°")
    
    print(f"\nNon-dimensional parameters:")
    print(f"  h' = {h_bar_val:.3f}")
    print(f"  b' = {b_bar_val:.3f}")
    
    print(f"\nVertical tire loads:")
    print(f"  N_r = {float(N_r_numeric):.2f} N")
    print(f"  N_f = {float(N_f_numeric):.2f} N")
    print(f"  Total = {float(N_r_numeric + N_f_numeric):.2f} N (should ≈ {M_val * g_val:.2f} N)")


if __name__ == "__main__":
    # Initialize model
    model = MotorcycleDynamicsModel()
    
    # Print all equations
    model.print_equations()
    
    # Run examples
    example_solve_for_roll_angle()
    example_tire_load_transfer()
