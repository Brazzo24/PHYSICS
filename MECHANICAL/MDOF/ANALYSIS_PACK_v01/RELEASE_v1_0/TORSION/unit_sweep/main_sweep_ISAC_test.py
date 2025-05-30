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
from excitation_generator import create_engine_excitation
import pandas as pd

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
RUN_ENGINE_SPEED_SWEEP = True
RUN_ORDER_BREAKDOWN_PLOT = True
FAST_TEST_MODE = True  # Set to True to reduce computation during testing

def define_system():
    m = np.array([1.21e-2, 3.95e-4, 4.438e-4,
                  9.044e-4, 7.173e-4, 2.639e-4, 7.257e-4, 3.88e-4,
                  1.244e-2]) # kgm^2
    c_inter = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]) # Nms/rad
    k_inter = np.array([2.34e4, 1.62e5, 1.112e3, 1.10e5, 1.10e5,
                        2.72e4, 1.94e3, 9.127e1])   # Nm/rad
    dof_labels = [
        "Crankshaft", "CRCS", "PG", "Clutch1",
        "Clutch2", "Input", "Output", "Hub", "Wheel"
    ]
    return m, c_inter, k_inter, dof_labels

def run_engine_speed_sweep(m, c_inter, k_inter, f_vals, harmonics, rpm_range):
    N = len(m)
    results = {
        'rpm': [],
        'max_velocity': [],
        'max_acceleration': [],
        'dof_max_vel': [],
        'dof_max_acc': []
    }

    for rpm in rpm_range:
        excitation_array = create_engine_excitation(f_vals, harmonics, rpm)
        F_ext = np.zeros((N, len(f_vals)), dtype=complex)
        F_ext[0, :] = excitation_array

        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)

        V_vals = response['V_vals']
        A_vals = response['A_vals']

        max_vel = np.max(np.abs(V_vals))
        max_acc = np.max(np.abs(A_vals))

        dof_max_vel = np.argmax(np.max(np.abs(V_vals), axis=1))
        dof_max_acc = np.argmax(np.max(np.abs(A_vals), axis=1))

        results['rpm'].append(rpm)
        results['max_velocity'].append(max_vel)
        results['max_acceleration'].append(max_acc)
        results['dof_max_vel'].append(dof_max_vel)
        results['dof_max_acc'].append(dof_max_acc)

    return pd.DataFrame(results)

def run_order_breakdown(m, c_inter, k_inter, f_vals, dof_labels, rpm_range, orders, dof_target_label, quantity="displacement"):
    dof_index = dof_labels.index(dof_target_label)
    N = len(m)
    order_labels = [f"{order:.2f}" for order in orders]
    order_data = np.zeros((len(orders), len(rpm_range)))
    synthesis = np.zeros(len(rpm_range))

    for i, rpm in enumerate(rpm_range):
        print(f"\n[INFO] RPM: {rpm}...")
        for j, order in enumerate(orders):
            excitation_array = create_engine_excitation(f_vals, [(order, 1.0, 0.0)], rpm)
            F_ext = np.zeros((N, len(f_vals)), dtype=complex)
            F_ext[0, :] = excitation_array
            response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)

            if quantity == "displacement":
                data_vals = np.abs(response['X_vals'][dof_index, :]) * 180 / np.pi
            elif quantity == "velocity":
                data_vals = np.abs(response['V_vals'][dof_index, :]) * 180 / np.pi
            else:
                raise ValueError("Quantity must be 'displacement' or 'velocity'")

            order_data[j, i] = np.max(data_vals)
            print(f"  Order {order:.2f} done")

        # Synthesis
        excitation_array = create_engine_excitation(f_vals, [(o, 1.0, 0.0) for o in orders], rpm)
        F_ext = np.zeros((N, len(f_vals)), dtype=complex)
        F_ext[0, :] = excitation_array
        response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)

        if quantity == "displacement":
            synthesis_vals = np.abs(response['X_vals'][dof_index, :]) * 180 / np.pi
        elif quantity == "velocity":
            synthesis_vals = np.abs(response['V_vals'][dof_index, :]) * 180 / np.pi

        synthesis[i] = np.max(synthesis_vals)

    return order_data, synthesis, order_labels

def run_analysis():
    m, c_inter, k_inter, dof_labels = define_system()
    N = len(m)
    if FAST_TEST_MODE:
        f_vals = np.linspace(1.0, 300.0, 2000)
    else:
        f_vals = np.linspace(0.1, 400.0, 10000)
    harmonics = [
        (0.5, 10, 0),
        (1, 100, 50),
        (1.5, 20, 0),
        (2, 50, 0),
        (2.5, 5, 0),
        (3, 10, 0)
    ]
    return m, c_inter, k_inter, f_vals, harmonics, dof_labels

def main():
    m, c_inter, k_inter, f_vals, harmonics, dof_labels = run_analysis()

    if RUN_ENGINE_SPEED_SWEEP:
        rpm_range = np.arange(6000, 20001, 500) if not FAST_TEST_MODE else np.arange(6000, 10001, 1000)
        df_rpm = run_engine_speed_sweep(m, c_inter, k_inter, f_vals, harmonics, rpm_range)

        plt.figure()
        plt.plot(df_rpm['rpm'], df_rpm['max_velocity'], label='Max Velocity')
        plt.plot(df_rpm['rpm'], df_rpm['max_acceleration'], label='Max Acceleration')
        plt.xlabel("Engine Speed [RPM]")
        plt.ylabel("Peak Response Amplitude")
        plt.title("Response Peaks vs. Engine Speed")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\nRPM with highest velocity:")
        print(df_rpm.iloc[df_rpm['max_velocity'].idxmax()])

        print("\nRPM with highest acceleration:")
        print(df_rpm.iloc[df_rpm['max_acceleration'].idxmax()])

    if RUN_ORDER_BREAKDOWN_PLOT:
        rpm_range = np.arange(4000, 20001, 250) if not FAST_TEST_MODE else np.arange(4000, 20001, 500)
        dof_target_label = "Clutch1"
        orders = np.arange(0.5, 5.0, 0.5) if not FAST_TEST_MODE else np.arange(0.5, 5.0, 0.5)

        for quantity in ["displacement", "velocity"]:
            order_data, synthesis, order_labels = run_order_breakdown(m, c_inter, k_inter, f_vals,
                                                                       dof_labels, rpm_range,
                                                                       orders, dof_target_label, quantity)

            plt.figure(figsize=(10, 6))
            for i, label in enumerate(order_labels):
                plt.plot(rpm_range, order_data[i], label=f"Order {label} (deg/s)" if quantity=="velocity" else f"Order {label} (deg)", linewidth=0.75)
            plt.plot(rpm_range, synthesis, label="Synthesis", linewidth=2.0, color='red')
            plt.xlabel("Engine Speed (rpm)")
            plt.ylabel("Angular Velocity (deg/s)" if quantity=="velocity" else "Angular Displacement (deg)")
            plt.title(f"{quantity.capitalize()} at DOF '{dof_target_label}' vs Engine Speed")
            plt.legend(loc="upper right", fontsize=8, ncol=2)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
