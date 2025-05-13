import numpy as np
import pandas as pd
from FDcalculations import free_vibration_analysis_free_chain, modal_energy_analysis

def run_2D_modal_energy_sensitivity(
    m_base,
    k_base,
    target_mode,
    dof_indices,
    param1_name,
    param1_index,
    param1_values,
    param2_name,
    param2_index,
    param2_values
):
    """
    Perform 2D sensitivity analysis on modal energy at specific DOFs.

    Parameters:
    - m_base, k_base: system parameters
    - target_mode: index of the mode to evaluate
    - dof_indices: which DOFs to track energy at
    - param1_name, param2_name: 'm' or 'k'
    - param1_index, param2_index: index in the array to modify
    - param1_values, param2_values: arrays of values to test

    Returns:
    - DataFrame with param1, param2, and energy at each DOF
    """
    results = []

    for val1 in param1_values:
        for val2 in param2_values:
            m = m_base.copy()
            k = k_base.copy()

            if param1_name == 'm':
                m[param1_index] = val1
            elif param1_name == 'k':
                k[param1_index] = val1
            else:
                raise ValueError(f"Unsupported param1_name: {param1_name}")

            if param2_name == 'm':
                m[param2_index] = val2
            elif param2_name == 'k':
                k[param2_index] = val2
            else:
                raise ValueError(f"Unsupported param2_name: {param2_name}")

            f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k)
            modal_energies = modal_energy_analysis(m, k, f_n, eigvecs, M_free)

            try:
                T_kin = modal_energies[target_mode]['T_dof']
            except (KeyError, IndexError):
                raise RuntimeError(f"Could not extract 'T_dof' from mode {target_mode}.")

            energies = [T_kin[i] for i in dof_indices]
            results.append([val1, val2] + energies)

    columns = [f"{param1_name}_{param1_index}", f"{param2_name}_{param2_index}"] + [f"Energy_DOF_{i}" for i in dof_indices]
    return pd.DataFrame(results, columns=columns)
