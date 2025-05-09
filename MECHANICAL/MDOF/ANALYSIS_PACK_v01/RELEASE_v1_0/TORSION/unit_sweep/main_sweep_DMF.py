###############################################################################
# MAIN SCRIPT
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from FDcalculations import*
from plotting import *

# Optionally import the recommender module
try:
    from recommender import recommend_damper_location
    USE_RECOMMENDER = True
except ImportError:
    USE_RECOMMENDER = False

# overwritee Recommender
USE_RECOMMENDER = False

def main():
    # ------------------------------
    # User-defined System Parameters
    # ------------------------------
    # Example: Reduced Crankshaft Model
    # ([Crankshaft, CRCS, PG, Clutch 1, Clutch 2, Input, Output, Hub, Wheel, Road])

    # m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
    #               1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
    #               2.73e-1, 2.69e+1])  # kgm^2
    
    # m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
    #             1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
    #             2.73e-1, 2.69e+1])  # kgm^2


    
    # # ([Gear, Gear, Primary Damper, Clutch, Spline, GBX, Chain, RWD, Tyre])
    # # c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nm.s/rad

    # c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nm.s/rad

    # # k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
    # #                     2.72e4, 4.97e3, 7.73e2, 8.57e2]) # Nm/rad
    
    # k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
    #                 2.72e4, 4.97e3, 7.73e2, 8.57e2]) # Nm/rad
    


    # Original definition of m, c_inter, k_inter

    # Add DMF between DOF0 and the rest
    m_dmf = 5e-3
    k_dmf = 1.0e4
    c_dmf = 0.2

    m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
                1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
                2.73e-1, 2.69e+1])
    c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
                        2.72e4, 4.97e3, 7.73e2, 8.57e2])

    # Insert DMF at index 1
    m = np.insert(m, 1, m_dmf)
    c_inter = np.insert(c_inter, 0, c_dmf)
    k_inter = np.insert(k_inter, 0, k_dmf)


    N = len(m)
    
    # External force vector: force applied at DOF 0.
    F_ext = np.zeros(N, dtype=complex)
    F_ext[0] = 1.0
    
    # Frequency range for forced response analysis.
    f_min, f_max = 0.1, 400.0
    num_points = 10000
    f_vals = np.linspace(f_min, f_max, num_points)
    
    # ------------------------------
    # Forced Response and Post-Processing
    # ------------------------------
    if COMPUTE_FORCED_RESPONSE:
        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)
        X_vals = response['X_vals']
        A_vals = response['A_vals']
        P_damp = response['P_damp']
        P_spring = response['P_spring']
        Q_mass = response['Q_mass']
        F_bound = response['F_bound']
        phase_vals = response['phase_vals']
        
        if PLOT_OVERVIEW_FORCED_RESPONSE:
            plot_forced_response_overview(f_vals, X_vals, A_vals, P_damp, P_spring, Q_mass, F_bound, m)
        
        # if PLOT_EXCITATION_POWER:
        #     plot_excitation_power(f_vals, X_vals)

        if PLOT_PHASE_ANGLES:
            plot_phase_relationships(f_vals, phase_vals)
    
    # ------------------------------
    # Free-Vibration Analysis and Modal Energy Distribution
    # ------------------------------
    if COMPUTE_FREE_VIBRATION_ANALYSIS or COMPUTE_MODAL_ENERGY_ANALYSIS:
        f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k_inter)
        print("\nUndamped Natural Frequencies (Hz):")
        for i, fn in enumerate(f_n):
            print(f"  Mode {i+1}: {fn:.3f} Hz")
            
        if COMPUTE_MODAL_ENERGY_ANALYSIS:
            modal_energies = modal_energy_analysis(m, k_inter, f_n, eigvecs, M_free)
            if PLOT_MODAL_ENERGY:
                plot_modal_energy_overview(modal_energies, mode_plot_limit, N)
    
    # ------------------------------
    # Poles (State-Space Eigenvalues)
    # ------------------------------

    if PLOT_POLES:
        poles = compute_poles_free_chain(m, c_inter, k_inter)
        plot_poles_overview(poles)

    
    # ------------------------------
    # Optional: Damper Placement Recommendation
    # ------------------------------
    if USE_RECOMMENDER and COMPUTE_MODAL_ENERGY_ANALYSIS and COMPUTE_FORCED_RESPONSE:
        user_mode_number = int(input("\nEnter the mode number you are concerned about: "))
        recommend_damper_location(f_vals, A_vals, P_damp, modal_energies, m, f_n, user_mode_number)
        print("\nAnalysis complete.")

    import os
    import pickle

    save_name = "DMF_run"  # change per run
    save_dir = os.path.join("results", save_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save all numerical arrays as .npz
    np.savez(os.path.join(save_dir, "results.npz"),
            f_vals=f_vals if COMPUTE_FORCED_RESPONSE else None,
            A_vals=A_vals if COMPUTE_FORCED_RESPONSE else None,
            P_damp=P_damp if COMPUTE_FORCED_RESPONSE else None,
            P_spring=P_spring if COMPUTE_FORCED_RESPONSE else None,
            Q_mass=Q_mass if COMPUTE_FORCED_RESPONSE else None,
            F_bound=F_bound if COMPUTE_FORCED_RESPONSE else None,
            phase_vals=phase_vals if COMPUTE_FORCED_RESPONSE else None,
            f_n=f_n if COMPUTE_FREE_VIBRATION_ANALYSIS else None,
            eigvecs=eigvecs if COMPUTE_FREE_VIBRATION_ANALYSIS else None,
            m=m, c=c_inter, k=k_inter)

    # Save modal_energies separately using pickle
    if COMPUTE_MODAL_ENERGY_ANALYSIS:
        with open(os.path.join(save_dir, "modal_energies.pkl"), "wb") as f:
            pickle.dump(modal_energies, f)

    print(f"\n✅ Results saved to: {save_dir}")



if __name__ == "__main__":
    main()

