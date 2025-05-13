import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sensitivity import run_modal_energy_sensitivity
from sensitivity_ranking import rank_parameter_influence
import os

def batch_sweep_and_rank(
    m_base,
    k_base,
    param_set,
    target_mode,
    dof_indices,
    sweep_range_m=0.5,
    sweep_range_k=2.0,
    num_points=5,
    save_csv_path=None,
    plot=True,
    save_plot_dir=None,
    energy_type="kinetic"  # or "potential"
):
    all_rankings = []

    for param_type, idx in param_set:
        param_name = f"{param_type}_{idx}"
        base_val = m_base[idx] if param_type == 'm' else k_base[idx]
        sweep_range = sweep_range_m if param_type == 'm' else sweep_range_k

        values = np.linspace((1 - sweep_range) * base_val, (1 + sweep_range) * base_val, num_points)

        df = run_modal_energy_sensitivity(
            m_base=m_base,
            k_base=k_base,
            target_mode=target_mode,
            dof_indices=dof_indices,
            param_name=param_type,
            param_index=idx,
            values=values,
            energy_type=energy_type
        )

        ranking = rank_parameter_influence(df, param_name=param_name)
        ranking.insert(0, 'Parameter', param_name)
        ranking.insert(1, 'Type', param_type)
        all_rankings.append(ranking)

    combined = pd.concat(all_rankings, ignore_index=True)

    if save_csv_path:
        combined.to_csv(save_csv_path, index=False)
        print(f"\n📁 Saved ranking summary to: {save_csv_path}")

    if plot:
        _plot_per_dof_grouped(combined, save_plot_dir, label=energy_type.capitalize())

    return combined


def _plot_per_dof_grouped(df, save_dir=None, label="Kinetic"):
    dof_list = sorted(df['DOF'].unique(), key=lambda x: int(x))
    for dof in dof_list:
        subset = df[df['DOF'] == dof]
        for ptype in ['m', 'k']:
            sub = subset[subset['Type'] == ptype]
            if sub.empty:
                continue

            # Sort by parameter index (natural order)
            sub = sub.copy()
            sub['index'] = sub['Parameter'].str.extract(r'_(\d+)').astype(int)
            sub = sub.sort_values(by='index')

            plt.figure(figsize=(10, 5))
            plt.bar(sub['Parameter'], sub['ImpactScore'], color='skyblue' if ptype == 'm' else 'lightcoral')
            plt.xticks(rotation=45, ha='right')
            plt.title(f"{label} Energy Impact Score for DOF {dof} ({ptype.upper()} parameters)")
            plt.ylabel("Impact Score (slope × rel. change)")
            plt.grid(True, axis='y')
            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fname = f"impact_{label.lower()}_dof_{dof}_{ptype}.png"
                plt.savefig(os.path.join(save_dir, fname))
                print(f"📷 Saved: {fname}")

            plt.show()
