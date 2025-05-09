# compare_runs_by_label.py
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pickle

def load_meta(run_dir):
    with open(os.path.join(run_dir, "meta.json")) as f:
        return json.load(f)

def compare_by_label(run_dirs, quantity_key, label_to_plot, is_complex=True):
    """
    Compare a saved quantity (e.g., acceleration) for the same physical component
    across multiple simulation runs using DOF labels.
    """
    plt.figure(figsize=(10, 5))

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        data_path = os.path.join(run_dir, "results.npz")
        meta_path = os.path.join(run_dir, "meta.json")

        if not os.path.exists(data_path) or not os.path.exists(meta_path):
            print(f"❌ Missing data or meta for {run_name}")
            continue

        data = np.load(data_path, allow_pickle=True)
        meta = load_meta(run_dir)
        dof_labels = meta.get("dof_labels", [])

        if label_to_plot not in dof_labels:
            print(f"⚠️ '{label_to_plot}' not found in {run_name}")
            continue

        idx = dof_labels.index(label_to_plot)
        f_vals = data["f_vals"]
        quantity = data[quantity_key]

        y = np.abs(quantity[idx, :]) if is_complex else quantity[idx, :]
        plt.plot(f_vals, y, label=f"{run_name} – {label_to_plot}")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(quantity_key)
    plt.title(f"{quantity_key} at '{label_to_plot}'")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compare_modal_kinetic_energy(run_dirs, mode_index):
    """
    Compare kinetic energy distribution for a specific mode number across runs.
    """
    plt.figure(figsize=(10, 5))
    for run_dir in run_dirs:
        label = os.path.basename(run_dir)
        npz_path = os.path.join(run_dir, "results.npz")
        pkl_path = os.path.join(run_dir, "modal_energies.pkl")

        if not os.path.exists(npz_path) or not os.path.exists(pkl_path):
            print(f"⚠️  Missing results for {label}")
            continue

        data = np.load(npz_path, allow_pickle=True)
        with open(pkl_path, "rb") as f:
            modal_energies = pickle.load(f)

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

# Example usage
if __name__ == "__main__":
    run_dirs = [
        "results/Baseline_run",
        "results/DMF_run"
    ]

    # Compare kinetic energy distribution
    compare_modal_kinetic_energy(run_dirs, mode_index=3)

    # Compare acceleration by physical label
    compare_by_label(run_dirs, quantity_key="A_vals", label_to_plot="PG", is_complex=True)
    compare_by_label(run_dirs, quantity_key="P_damp", label_to_plot="Clutch1", is_complex=True)
    compare_by_label(run_dirs, quantity_key="phase_vals", label_to_plot="PG", is_complex=False)
