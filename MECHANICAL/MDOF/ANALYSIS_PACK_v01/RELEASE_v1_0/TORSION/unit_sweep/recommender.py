# recommender.py
import numpy as np
import matplotlib.pyplot as plt
import os

def recommend_damper_location(f_vals, A_vals, P_damp, modal_energies, m, mode_frequencies, mode_number):
    print("\n======= Damper Placement Recommendation =======")
    target_mode_freq = mode_frequencies[mode_number - 1]
    print(f"Analyzing for Mode {mode_number} at {target_mode_freq:.2f} Hz")

    # --- 1) Modal Kinetic Energy distribution ---
    modal_kinetic_energy = modal_energies[mode_number - 1]['T_dof']
    print("\nModal Kinetic Energy Distribution (DOF-wise):")
    for i, val in enumerate(modal_kinetic_energy):
        print(f"DOF {i}: {val:.3e} J")

    # --- 2) Approximate relative velocity differences from mode shape ---
    phi_norm = modal_energies[mode_number - 1]['phi_norm']
    delta_velocity_estimates = np.abs(np.diff(phi_norm))
    print("\nRelative velocity differences (proxy for damper effectiveness):")
    for i, val in enumerate(delta_velocity_estimates):
        print(f"Between DOF {i} and {i+1}: {val:.3e}")

    # --- 3) Check forced response damping power around the mode frequency ---
    freq_idx = np.argmin(np.abs(f_vals - target_mode_freq))
    print("\nDamping power contributions at mode frequency:")
    for i in range(P_damp.shape[0]):
        power_at_mode = np.real(P_damp[i, freq_idx])
        print(f"Damper between DOF {i} and {i+1}: {power_at_mode:.3e} W")

    # --- 4) Combined ranking suggestion ---
    combined_score = delta_velocity_estimates * np.real(P_damp[:, freq_idx])
    best_location = np.argmax(combined_score)

    print("\n===== Recommended Damper Location =====")
    print(f"Place damper between DOF {best_location} and {best_location + 1}.")
    print("Reason: Highest combined score of modal velocity difference and damping power.")

    # --- Optional: Visualization ---
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(combined_score)), combined_score, alpha=0.7)
    plt.xlabel('Connection Index (between DOFs)')
    plt.ylabel('Combined Score')
    plt.title(f'Combined Damper Placement Score for Mode {mode_number} ({target_mode_freq:.2f} Hz)')
    plt.grid(True)
    plt.tight_layout()
    
    # Automatically save the recommendation plot
    plot_dir = "PLOTS"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f"damper_recommendation_mode_{mode_number}.png"))
    plt.show()
