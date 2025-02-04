import matplotlib.pyplot as plt
import numpy as np
import math

def plot_energy_distributions(system, modes_per_figure=4, save_figures=True, show_last=True):
    """
    Plots and optionally saves kinetic and potential energy distributions 
    for all modes, grouping multiple modes per figure.

    Parameters
    ----------
    system : TorsionalSystem
        The system whose energy distributions should be plotted.
    modes_per_figure : int, optional
        Number of modes per figure (default = 4).
    save_figures : bool, optional
        If True, saves the figures instead of displaying all of them (default = True).
    show_last : bool, optional
        If True, displays only the last figure (useful for debugging).
    """
    # Compute energy distributions
    kinetic_fractions, potential_fractions = system.compute_energy_distributions()
    omegas = system._omega
    n_modes_kept = len(omegas)

    if n_modes_kept == 0:
        print("No valid modes to plot.")
        return

    # Determine how many figures are needed
    num_figures = math.ceil(n_modes_kept / modes_per_figure)

    for fig_idx in range(num_figures):
        start_idx = fig_idx * modes_per_figure
        end_idx = min(start_idx + modes_per_figure, n_modes_kept)
        num_modes_in_fig = end_idx - start_idx

        fig, axs = plt.subplots(num_modes_in_fig, 2, figsize=(10, 4 * num_modes_in_fig), dpi=100)
        fig.suptitle(f"Energy Distributions (Modes {start_idx+1} - {end_idx})", fontsize=14)

        # Handle the case where there's only one mode in this figure (ensures axs is iterable)
        if num_modes_in_fig == 1:
            axs = np.array([axs])

        for i, r in enumerate(range(start_idx, end_idx)):
            ax_left = axs[i, 0]
            ax_right = axs[i, 1]

            # --- Left: Kinetic Energy Fractions ---
            ke_data = kinetic_fractions[r]
            x_inertias = np.arange(len(ke_data))
            ax_left.bar(x_inertias, ke_data, color='dodgerblue')
            ax_left.set_xlabel("Inertia Index")
            ax_left.set_ylabel("Fraction of Kinetic Energy")
            ax_left.set_ylim([0, 1])
            ax_left.set_title(f"Mode {r+1} (ω = {omegas[r]:.2f} rad/s)")

            # Annotate bars
            for j, val in enumerate(ke_data):
                ax_left.text(j, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

            # --- Right: Potential Energy Fractions ---
            pe_data = potential_fractions[r]
            x_springs = np.arange(len(pe_data))
            ax_right.bar(x_springs, pe_data, color='orange')
            ax_right.set_xlabel("Spring Index")
            ax_right.set_ylabel("Fraction of Potential Energy")
            ax_right.set_ylim([0, 1])
            ax_right.set_title(f"Mode {r+1} Potential Energy")

            # Annotate bars
            for j, val in enumerate(pe_data):
                ax_right.text(j, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        if save_figures:
            filename = f"energy_modes_{start_idx+1}_{end_idx}.png"
            plt.savefig(filename)
            print(f"Saved: {filename}")

        # Show only the last figure if show_last is True
        if fig_idx == num_figures - 1 and show_last:
            plt.show()
        else:
            plt.close()  # Close intermediate figures to avoid excessive windows