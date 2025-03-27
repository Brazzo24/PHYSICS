import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from FDcalculations import*
from plotting import *
from excitation_generator import create_engine_excitation

# Optionally import the recommender module
try:
    from recommender import recommend_damper_location
    USE_RECOMMENDER = True
except ImportError:
    USE_RECOMMENDER = False

# overwritee Recommender
USE_RECOMMENDER = True

###############################################################################
# MAIN SCRIPT
###############################################################################
def main():
    # ------------------------------
    # User-defined System Parameters
    # ------------------------------
    # Example: Reduced Crankshaft Model
    m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
                  1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
                  2.73e-1, 2.69e+1])  # kgm^2


    # ([Crankshaft, CRCS, PG, Clutch 1, Clutch 2, Input, Output, Hub, Wheel, Road])
    c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nm.s/rad

    k_inter = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5,
                        2.72e4, 4.97e3, 7.73e2, 8.57e2]) # Nm/rad
    

    N = len(m)
    
    f_min, f_max = 0.1, 1000.0
    num_points = 10000
    f_vals = np.linspace(f_min, f_max, num_points)

    engine_speed_rpm = 6000

    # (order, real part, imaginary part)
    harmonics = [
    (0.5, 70, 0),
    (1, 305, 0),
    (1.5, 105, 0),
    (2, 10, 0),
    (2.5, 60, 0),
    (3, 20, 0),
    (3.5, 110, 0),
    (4, 80, 0)
    ]



    excitation_array = create_engine_excitation(f_vals, harmonics, engine_speed_rpm)
    
    F_ext_matrix = np.zeros((len(m), num_points), dtype=complex)
    F_ext_matrix[0, :] = excitation_array

    plt.figure(figsize=(9, 5))
    plt.plot(f_vals, np.abs(F_ext_matrix[0, :]), label="Excitation Amplitude at DOF 0")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Nm)")
    plt.title("Frequency-Dependent Input Excitation")
    plt.grid()
    plt.legend()
    plt.show()
    # ------------------------------
    # Forced Response and Post-Processing
    # ------------------------------
    if COMPUTE_FORCED_RESPONSE:
        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext_matrix)
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
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
