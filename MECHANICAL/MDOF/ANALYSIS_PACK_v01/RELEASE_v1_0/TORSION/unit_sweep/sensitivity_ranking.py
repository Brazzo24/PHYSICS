import pandas as pd
import numpy as np

def rank_parameter_influence(sweep_df, param_name):
    """
    Analyze a 1D sensitivity DataFrame and rank the influence of a parameter
    on the modal energy at each DOF.

    Parameters:
    - sweep_df: DataFrame returned from run_modal_energy_sensitivity()
    - param_name: str, name of the parameter column (e.g. 'm_0', 'k_7')

    Returns:
    - DataFrame summarizing energy change and ranking per DOF
    """
    dof_columns = [col for col in sweep_df.columns if col.startswith("Energy_DOF_")]
    rankings = []

    for col in dof_columns:
        energy_vals = sweep_df[col].values
        param_vals = sweep_df[param_name].values

        energy_range = np.max(energy_vals) - np.min(energy_vals)
        slope = np.polyfit(param_vals, energy_vals, 1)[0]  # linear fit slope
        rel_change = energy_range / np.mean(energy_vals) if np.mean(energy_vals) != 0 else 0

        rankings.append({
            "DOF": col.replace("Energy_DOF_", ""),
            "MaxEnergy": np.max(energy_vals),
            "MinEnergy": np.min(energy_vals),
            "Range": energy_range,
            "RelChange": rel_change,
            "Slope": slope,
            "ImpactScore": abs(slope) * rel_change  # simple composite metric
        })

    return pd.DataFrame(rankings).sort_values(by="ImpactScore", ascending=False)
