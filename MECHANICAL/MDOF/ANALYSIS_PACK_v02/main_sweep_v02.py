###############################################################################
# MAIN SCRIPT
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from FDcalculations_ import*
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
    
    m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
                1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
                7.73e-2, 2.69e+1])  # kgm^2
    
    # ([Gear, Gear, Primary Damper, Clutch, Spline, GBX, Chain, RWD, Tyre])
    # c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nm.s/rad

    c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nm.s/rad

    # k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
    #                     2.72e4, 4.97e3, 7.73e2, 8.57e2]) # Nm/rad
    
    k_inter = np.array([2.34e4, 1.62e5, 1.1e3, 1.10e5, 1.10e5,
                    2.72e4, 4.97e3, 2.73e2, 8.57e2]) # Nm/rad
    
    m[0] *= 1.0
    m[8] *= 1.2
    k_inter[2] *= 1.0
    k_inter[7] *= 1.0
    k_inter[8] *= 1.8
    

    N = len(m)


    f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k_inter)
    print("\nUndamped Natural Frequencies (Hz):")
    for i, fn in enumerate(f_n):
        print(f"  Mode {i+1}: {fn:.3f} Hz")

    f_vals = np.linspace(0.1, 40.0, 1000)

    # Case A: Cancel Mode 3
    F_cancel = compute_mode_interaction_force(mode_idx=2, dof_primary=0, dof_secondary=8,
                                            eigvecs=eigvecs, N=N, mode='cancel')
    response_cancel = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_cancel)

    # Case B: Reinforce Mode 3
    F_reinforce = compute_mode_interaction_force(mode_idx=2, dof_primary=0, dof_secondary=8,
                                                eigvecs=eigvecs, N=N, mode='reinforce')
    
    response_reinforce = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_reinforce)

        # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(f_vals, np.abs(response_cancel['X_vals'][0]), label='Cancel Mode 3 (DOF 0)')
    plt.plot(f_vals, np.abs(response_reinforce['X_vals'][0]), label='Reinforce Mode 3 (DOF 0)', linestyle='--')
    plt.title("Response at DOF 0: Cancel vs Reinforce Mode 3")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    F_ext = F_cancel
    # F_ext = F_reinforce 

    # ------------------------------
    # Forced Response and Post-Processing
    # ------------------------------
    if COMPUTE_FORCED_RESPONSE:
        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)
        X_vals = response['X_vals']
        V_vals = response['V_vals']
        A_vals = response['A_vals']
        P_damp = response['P_damp']
        P_spring = response['P_spring']
        Q_mass = response['Q_mass']
        F_bound = response['F_bound']
        phase_vals = response['phase_vals']
        
        if PLOT_OVERVIEW_FORCED_RESPONSE:
            plot_forced_response_overview(f_vals, X_vals, V_vals, A_vals, P_damp, P_spring, Q_mass, F_bound, m)
        
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

if __name__ == "__main__":
    main()
