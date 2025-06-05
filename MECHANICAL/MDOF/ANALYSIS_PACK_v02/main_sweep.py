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

PowerFlow = True

def main():
    # ------------------------------
    # User-defined System Parameters
    # ------------------------------
    # Example: Reduced Crankshaft Model
    # ([Crankshaft, CRCS, PG, Clutch 1, Clutch 2, Input, Output, Hub, Wheel, Road])

    # m = np.array([1.21e-2, 3.95e-4, 7.92e-4,
    #               1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3,
    #               2.73e-1, 2.69e+1])  # kgm^2

    # Add DMF between DOF0 and the rest
    f_dmf = 25.65 # Hz
    m_dmf = 8e-3
    k_dmf = ((2*np.pi * f_dmf) ** 2) * m_dmf
    c_dmf = 2 * np.sqrt(m_dmf * k_dmf) * 0.15 # last factor is zeta, being the amount of critical damping (0 - 1.0)
    
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

    # Insert DMF at index 1
    m = np.insert(m, 8, m_dmf)
    c_inter = np.insert(c_inter, 7, c_dmf)
    k_inter = np.insert(k_inter, 7, k_dmf)
    
    print(k_inter)

    N = len(m)


    f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k_inter)
    print("\nUndamped Natural Frequencies (Hz):")
    for i, fn in enumerate(f_n):
        print(f"  Mode {i+1}: {fn:.3f} Hz")

    # Cancel Mode 3 (index 2)
    mode_idx = 2
    phi_mode3 = eigvecs[:, mode_idx]
    phi_mode3 /= np.max(np.abs(phi_mode3))

    dof_primary = 5
    dof_secondary = 0
    F_primary = 1.0 + 0j
    F_secondary = 0.0 + 0j
    # F_secondary = - (phi_mode3[dof_primary] / phi_mode3[dof_secondary]) * F_primary
    print(phi_mode3[dof_primary])
    print(phi_mode3[dof_secondary])
    print(F_secondary)

    # External force vector: force applied at DOF 0.
    F_ext = np.zeros(N, dtype=complex)
    F_ext[dof_primary] = F_primary
    F_ext[dof_secondary] = F_secondary
    print(F_ext)
    
    # Frequency range for forced response analysis.
    f_min, f_max = 0.1, 200.0
    num_points = 1000
    f_vals = np.linspace(f_min, f_max, num_points)
    
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


    if PowerFlow:
        P_spring_complex, power_map = compute_power_flow(f_vals, X_vals, k_inter, m)
      

    plt.figure()
    plt.plot(power_map['abs'], label='|Power Flow|')
    plt.plot(power_map['real'], label='Real (stored)')
    plt.plot(power_map['imag'], label='Imag (reactive)')
    plt.xlabel("Spring Index (i = between DOF i and i+1)")
    plt.ylabel("Power (Nm/s)")
    plt.title("Spring Power Flow at ~24 Hz")
    plt.legend()
    plt.grid(True)
    plt.show()

    top_indices = np.argsort(power_map['abs'])[::-1]
    for idx in top_indices[:3]:
        print(f"Top flow between DOF {idx} and {idx+1}: {power_map['abs'][idx]:.2e} Nm/s")



if __name__ == "__main__":
    main()