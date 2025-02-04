import pytest
import numpy as np
from system import TorsionalSystem

"""
python -m pytest test_torsional_system.py
"""

# Define sample test data
@pytest.fixture
def sample_system():
    """Creates a simple torsional system with test inertias and stiffnesses."""
    system = TorsionalSystem(ignore_rigid_body_modes=True, rbm_tolerance=1e-12)
    system.add_inertia(0.15)
    system.add_inertia(0.25)
    system.add_inertia(0.35)

    system.add_spring(1, 2, 5000)
    system.add_spring(2, 3, 7000)

    return system

def test_add_inertia():
    """Test if inertias are correctly added."""
    system = TorsionalSystem()
    node1 = system.add_inertia(0.2)
    node2 = system.add_inertia(0.4)

    assert node1 == 1
    assert node2 == 2
    assert system._inertia_dict[node1] == 0.2
    assert system._inertia_dict[node2] == 0.4

def test_add_spring():
    """Test if springs are correctly added."""
    system = TorsionalSystem()
    system.add_spring(1, 2, 5000)
    system.add_spring(2, 3, 7000)

    assert len(system._springs) == 2
    assert system._springs[0] == (1, 2, 5000)
    assert system._springs[1] == (2, 3, 7000)

def test_build_matrices(sample_system):
    """Test if mass and stiffness matrices are built correctly."""
    I_mat, K_mat = sample_system.build_matrices()

    assert I_mat.shape == (3, 3)
    assert K_mat.shape == (3, 3)
    assert I_mat[0, 0] == 0.15
    assert I_mat[1, 1] == 0.25
    assert I_mat[2, 2] == 0.35

def test_solve_eigenmodes(sample_system):
    """Test if eigenmodes solve without errors."""
    eigsq, modes = sample_system.solve_eigenmodes()

    assert eigsq is not None
    assert modes is not None
    assert len(eigsq) > 0

def test_compute_energy_distributions(sample_system):
    """Test energy distribution calculation."""
    sample_system.solve_eigenmodes()  # Ensure modes are solved before calling energy computation
    ke_frac, pe_frac = sample_system.compute_energy_distributions()

    assert len(ke_frac) > 0
    assert len(pe_frac) > 0
    assert isinstance(ke_frac, list)
    assert isinstance(pe_frac, list)