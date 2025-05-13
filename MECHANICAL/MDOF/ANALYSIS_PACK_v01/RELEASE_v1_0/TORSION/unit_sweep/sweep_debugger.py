import numpy as np
from sensitivity import run_modal_energy_sensitivity
from sensitivity_ranking import rank_parameter_influence
import pandas as pd


def debug_parameter_sweep(
    m_base,
    k_base,
    param_type,
    param_index,
    target_mode,
    dof_indices,
    sweep_range=0.5,
    num_points=5
):
    """
    Debug a single parameter sweep and print whether any energy values change.
    Also prints sweep values and energy min/max for each DOF.
    """
    param_name = f"{param_type}_{param_index}"
    base_val = m_base[param_index] if param_type == 'm' else k_base[param_index]
    values = np.linspace((1 - sweep_range) * base_val, (1 + sweep_range) * base_val, num_points)

    print(f"\n🔍 Debugging sweep for {param_name} over values: {values}")

    df = run_modal_energy_sensitivity(
        m_base=m_base,
        k_base=k_base,
        target_mode=target_mode,
        dof_indices=dof_indices,
        param_name=param_type,
        param_index=param_index,
        values=values
    )

    print(f"\n🧾 Energy values for each DOF (min/max across sweep):")
    for dof in dof_indices:
        col = f"Energy_DOF_{dof}"
        e_min = df[col].min()
        e_max = df[col].max()
        delta = e_max - e_min
        print(f"  {col}: min={e_min:.4e}, max={e_max:.4e}, delta={delta:.4e}")

    ranking = rank_parameter_influence(df, param_name=param_name)
    print(f"\n📊 Ranking table for {param_name}:")
    print(ranking.to_string(index=False))

    return df, ranking