import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from FDcalculations import *
from plotting import *
from sensitivity import run_modal_energy_sensitivity 
from sensitivity_2d import run_2D_modal_energy_sensitivity
from sensitivity_plotting import plot_sensitivity_line, plot_sensitivity_heatmap
from sensitivity_ranking import rank_parameter_influence
from sweep_batch_runner import batch_sweep_and_rank
from sweep_debugger import debug_parameter_sweep
from sweep_plot_curves import plot_energy_vs_parameter
import os
import pickle
import json

try:
    from recommender import recommend_damper_location
    USE_RECOMMENDER = True
except ImportError:
    USE_RECOMMENDER = False

USE_RECOMMENDER = False

# Flags
RUN_SENSITIVITY_ANALYSIS = False
RUN_SENSITIVITY_ANALYSIS_2D = False
FULL_RANKING = False


def define_system():
    m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
                  1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
                  7.73e-2, 2.69e+1])
    c_inter = np.array([0.05] * 9)
    k_inter = np.array([2.34e4, 1.62e5, 1.81e3, 1.10e5, 1.10e5,
                        2.72e4, 4.97e3, 2.73e2, 8.57e2])

    m[0] *= 1.0
    m[8] *= 1.2
    k_inter[2] *= 1.0
    k_inter[7] *= 1.0
    k_inter[8] *= 1.8

    dof_labels = [
        "Crankshaft", "DMF", "CRCS", "PG", "Clutch1",
        "Clutch2", "Input", "Output", "Hub", "Wheel", "Road"
    ]
    return m, c_inter, k_inter, dof_labels


def run_analysis():
    m, c_inter, k_inter, dof_labels = define_system()
    N = len(m)
    F_ext = np.zeros(N, dtype=complex)
    F_ext[0] = 1.0
    f_vals = np.linspace(0.1, 400.0, 10000)

    if COMPUTE_FORCED_RESPONSE:
        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)
        if PLOT_OVERVIEW_FORCED_RESPONSE:
            plot_forced_response_overview(f_vals, response['X_vals'], response['V_vals'], response['A_vals'],
                                          response['P_damp'], response['P_spring'], response['Q_mass'], response['F_bound'], m)
        if PLOT_PHASE_ANGLES:
            plot_phase_relationships(f_vals, response['phase_vals'])

    if COMPUTE_FREE_VIBRATION_ANALYSIS or COMPUTE_MODAL_ENERGY_ANALYSIS:
        f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k_inter)
        print("\nUndamped Natural Frequencies (Hz):")
        for i, fn in enumerate(f_n):
            print(f"  Mode {i+1}: {fn:.3f} Hz")

        if COMPUTE_MODAL_ENERGY_ANALYSIS:
            modal_energies = modal_energy_analysis(m, k_inter, f_n, eigvecs, M_free)
            if PLOT_MODAL_ENERGY:
                plot_modal_energy_overview(modal_energies, mode_plot_limit, N)

    if PLOT_POLES:
        poles = compute_poles_free_chain(m, c_inter, k_inter)
        plot_poles_overview(poles)

    if USE_RECOMMENDER and COMPUTE_MODAL_ENERGY_ANALYSIS and COMPUTE_FORCED_RESPONSE:
        user_mode_number = int(input("\nEnter the mode number you are concerned about: "))
        recommend_damper_location(f_vals, response['A_vals'], response['P_damp'], modal_energies, m, f_n, user_mode_number)
        print("\nAnalysis complete.")

    return m, k_inter


def main():
    m, k_inter = run_analysis()

    # 1D Sensitivity Analysis Example
    if RUN_SENSITIVITY_ANALYSIS:
        param_type = "k"
        param_index = 4
        param_column_name = f"{param_type}_{param_index}"
        sweep_values = np.linspace(0.5 * k_inter[param_index], 1.5 * k_inter[param_index], 5)

        df_sweep = run_modal_energy_sensitivity(
            m_base=m,
            k_base=k_inter,
            target_mode=2,
            dof_indices=list(range(9)),
            param_name=param_type,
            param_index=param_index,
            values=sweep_values,
            energy_type='potential'  
        )

        plot_sensitivity_line(df_sweep, param_name=param_column_name)
        ranking = rank_parameter_influence(df_sweep, param_name=param_column_name)
        print("\n📊 Ranked influence of", param_column_name)
        print(ranking.to_string(index=False))

    # 2D Sensitivity
    if RUN_SENSITIVITY_ANALYSIS_2D:
        df_2d = run_2D_modal_energy_sensitivity(
            m_base=m,
            k_base=k_inter,
            target_mode=1,
            dof_indices=[0, 8],
            param1_name='m', param1_index=0, param1_values=np.linspace(0.5*m[0], 1.5*m[0], 5),
            param2_name='k', param2_index=7, param2_values=np.linspace(0.5*k_inter[7], 1.5*k_inter[7], 5)
        )
        plot_sensitivity_heatmap(df_2d, param1_prefix="m", param2_prefix="k")

    energy_type = "kinetic"

    if FULL_RANKING:
    # Full Ranking for Potential Energy
        df_potential = batch_sweep_and_rank(
            m_base=m,
            k_base=k_inter,
            param_set=[('m', i) for i in range(9)] + [('k', i) for i in range(9)],
            target_mode=1,
            dof_indices=list(range(9)),
            energy_type=energy_type,
            sweep_range_m=0.5,
            sweep_range_k=2.0,
            num_points=7,
            save_csv_path=f"results/{energy_type}_energy_ranking.csv",
            save_plot_dir=f"results/{energy_type}_energy_plots",
            plot=True
        )

if __name__ == "__main__":
    main()
