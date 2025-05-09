import numpy as np
import matplotlib.pyplot as plt
import os

def compare_quantity(run_labels, quantity_key, index, label_fmt="{run}", 
                     ylabel=None, title=None, is_complex=True, transpose=False):
    """
    Compare a specific quantity across saved runs.

    Parameters:
    - run_labels: list of folder names (e.g. ["Baseline_run", "DMF_run"])
    - quantity_key: key in the .npz file (e.g. "A_vals", "P_damp", "phase_vals")
    - index: DOF index or spring index to extract for comparison
    - label_fmt: how to label the runs in the plot
    - ylabel, title: override axis and title labels
    - is_complex: if True, take abs(); if False, plot directly
    - transpose: if True, assumes shape is [freq, DOF] instead of [DOF, freq]
    """
    plt.figure(figsize=(10, 5))

    for run in run_labels:
        file_path = f"results/{run}/results.npz"
        if not os.path.exists(file_path):
            print(f"⚠️  Missing {file_path}")
            continue
        
        data = np.load(file_path, allow_pickle=True)
        f_vals = data["f_vals"]
        quantity = data[quantity_key]

        if transpose:
            slice_vals = quantity[:, index]
        else:
            slice_vals = quantity[index, :]

        if is_complex:
            y = np.abs(slice_vals)
        else:
            y = slice_vals

        plt.plot(f_vals, y, label=label_fmt.format(run=run))

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(ylabel if ylabel else quantity_key)
    plt.title(title if title else f"{quantity_key} at index {index}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

run_labels = ["Baseline_run", "DMF_run"]
mode_index = 3  # mode to compare (0-indexed = Mode 4)

plt.figure(figsize=(10, 5))

for label in run_labels:
    print(f"🔍 Loading {label}...")

    npz_path = f"results/{label}/results.npz"
    pkl_path = f"results/{label}/modal_energies.pkl"

    if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
        print(f"⚠️  Missing results for {label}")
        continue

    data = np.load(npz_path, allow_pickle=True)
    with open(pkl_path, "rb") as f:
        modal_energies = pickle.load(f)

    # modal_energies is a list of dicts
    KE_list = [mode["T_dof"] for mode in modal_energies]
    f_n = data["f_n"]

    plt.plot(np.arange(len(KE_list[mode_index])), KE_list[mode_index],
             label=f"{label} ({f_n[mode_index]:.2f} Hz)")

plt.title(f"Kinetic Energy Distribution – Mode {mode_index + 1}")
plt.xlabel("DOF Index")
plt.ylabel("Kinetic Energy [J]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


runs = ["Baseline_run", "DMF_run"]
compare_quantity(runs, quantity_key="A_vals", index=0, ylabel="Acceleration [rad/s²]", title="Acceleration at DOF 0")
compare_quantity(runs, quantity_key="P_damp", index=2, ylabel="Power [W]", title="Damping Power at Spring 2")
compare_quantity(runs, quantity_key="phase_vals", index=0, ylabel="Phase [rad]", title="Phase of DOF 0", is_complex=False)
