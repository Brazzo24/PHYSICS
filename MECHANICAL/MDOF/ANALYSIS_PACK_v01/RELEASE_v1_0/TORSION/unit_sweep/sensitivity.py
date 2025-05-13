import numpy as np
import pandas as pd
from FDcalculations import free_vibration_analysis_free_chain, modal_energy_analysis


def run_modal_energy_sensitivity(
    m_base,
    k_base,
    target_mode,
    dof_indices,
    param_name,
    param_index,
    values,
    energy_type="kinetic"
):
    """
    Perform sensitivity analysis on modal energy at specific DOFs or springs.

    Parameters:
    - m_base: np.ndarray, base inertia vector
    - k_base: np.ndarray, base stiffness vector
    - target_mode: int, index of mode to analyze (0-based)
    - dof_indices: list of int, indices of DOFs or spring elements to extract energy from
    - param_name: str, 'm' or 'k'
    - param_index: int, index of the parameter to modify
    - values: list or array of parameter values to test
    - energy_type: str, either 'kinetic' (default) or 'potential'

    Returns:
    - pd.DataFrame with parameter values and energy per DOF or spring
    """
    results = []

    for val in values:
        m = m_base.copy()
        k = k_base.copy()

        if param_name == "m":
            m[param_index] = val
        elif param_name == "k":
            k[param_index] = val
        else:
            raise ValueError(f"Unsupported parameter: {param_name}")

        f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k)
        modal_energies = modal_energy_analysis(m, k, f_n, eigvecs, M_free)

        try:
            mode_data = modal_energies[target_mode]
            if energy_type == "kinetic":
                energy_vector = mode_data["T_dof"]
            elif energy_type == "potential":
                energy_vector = mode_data["V_springs"]
            else:
                raise ValueError(f"Invalid energy_type '{energy_type}'. Must be 'kinetic' or 'potential'.")
        except (IndexError, KeyError):
            raise RuntimeError(f"Could not extract energy data from mode {target_mode}.")

        energies = [energy_vector[i] for i in dof_indices]
        results.append([val] + energies)

    param_col = f"{param_name}_{param_index}"
    columns = [param_col] + [f"Energy_DOF_{i}" for i in dof_indices]
    return pd.DataFrame(results, columns=columns)
