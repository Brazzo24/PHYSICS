import numpy as np
import matplotlib.pyplot as plt
from sensitivity import run_modal_energy_sensitivity

def plot_energy_vs_parameter(
    m_base,
    k_base,
    param_type,
    param_index,
    target_mode,
    dof_indices,
    sweep_range,
    num_points=7
):
    """
    Visualize how modal energy at selected DOFs changes as one parameter is varied.
    """
    param_name = f"{param_type}_{param_index}"
    base_val = m_base[param_index] if param_type == 'm' else k_base[param_index]
    values = np.linspace((1 - sweep_range) * base_val, (1 + sweep_range) * base_val, num_points)

    df = run_modal_energy_sensitivity(
        m_base=m_base,
        k_base=k_base,
        target_mode=target_mode,
        dof_indices=dof_indices,
        param_name=param_type,
        param_index=param_index,
        values=values
    )

    plt.figure(figsize=(10, 6))
    for dof in dof_indices:
        col = f"Energy_DOF_{dof}"
        plt.plot(df[param_name], df[col], marker='o', label=f"DOF {dof}")

    plt.title(f"Energy vs {param_name} (Mode {target_mode + 1})")
    plt.xlabel(f"{param_name} value")
    plt.ylabel("Kinetic Energy at DOF [J]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df
