import os
import math
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ------------------------------
# Helper: HUD Display Function
# ------------------------------
def hud(v, a, rpm, gear, t, x, t_start, x_start):
    # Print formatted simulation info:
    # Speed in km/h, Acceleration in G, RPM (rounded), Gear, Time since braking start, Absolute distance, Relative time, Relative distance
    print(f"{v*3.6:7.2f}\t{a/9.81:7.2f}\t{round(rpm):7d}\t{gear:7d}\t{(t - t_start):7.2f}\t{x:7.2f}\t{t - t_start:7.2f}\t{x - x_start:7.2f}")

# ------------------------------
# Main OpenDRAG Simulation
# ------------------------------
def main():
    # --- Simulation Settings ---
    use_date_time_in_name = False
    dt = 1e-3            # Time step [s]
    t_max = 60           # Maximum simulation time [s]
    ax_sens = 0.05       # Acceleration sensitivity for drag limitation [m/s²]
    # Speed traps (given in km/h, then converted to m/s)
    speed_trap = np.array([50, 100, 150, 200, 250, 300, 350]) / 3.6
    # Track data (assumed zero if not provided)
    bank = 0             # [deg]
    incl = 0             # [deg]

    # --- Loading Vehicle Data ---
    # Here we assume that the vehicle file (created by OpenVEHICLE) is stored as a pickle file.
    # Adjust the file name and path as needed.
    vehiclefile = 'OpenVEHICLE Vehicles/OpenVEHICLE_Formula 1_Open Wheel.pkl'
    if not os.path.exists(vehiclefile):
        raise FileNotFoundError(f"Vehicle file not found: {vehiclefile}")
    with open(vehiclefile, 'rb') as f:
        veh = pickle.load(f)

    # --- Vehicle Data Preprocessing ---
    M = veh["M"]                   # mass [kg]
    g = 9.81
    dmx = veh["factor_grip"] * veh["sens_x"]
    mux = veh["factor_grip"] * veh["mu_x"]
    Nx = veh["mu_x_M"] * g
    # Normal load on all wheels (using bank/incl in degrees)
    Wz = M * g * math.cos(math.radians(bank)) * math.cos(math.radians(incl))
    # Induced weights from banking/inclination
    Wy = M * g * math.sin(math.radians(bank))
    Wx = M * g * math.sin(math.radians(incl))
    # Drivetrain ratios (assuming gear indices are 1-indexed as in MATLAB)
    rf = veh["ratio_final"]
    rg = veh["ratio_gearbox"]      # array
    rp = veh["ratio_primary"]
    Rt = veh["tyre_radius"]        # tire radius [m]
    np_eff = veh["n_primary"]
    ng_eff = veh["n_gearbox"]
    nf = veh["n_final"]
    # Engine curves: prepend initial value as in MATLAB
    rpm_curve = np.concatenate(([0], veh["en_speed_curve"]))
    torque_curve = veh["factor_power"] * np.concatenate(([veh["en_torque_curve"][0]], veh["en_torque_curve"]))
    # Shift points: assume veh["shifting"] is a DataFrame-like structure.
    # Here we assume it is stored as a dictionary with key "shift_points" (or a NumPy array)
    # For example, veh["shifting"] could be a dict with key 'shift_points'
    # If it is a 2D array, take the first column and append the last engine speed.
    shift_points = np.array(veh["shifting"][:, 0]) if isinstance(veh["shifting"], np.ndarray) else np.array(veh["shifting"]["shift_points"])
    shift_points = np.concatenate((shift_points, [veh["en_speed_curve"][-1]]))
    
    # --- Preallocation for Simulation Data ---
    N = int(t_max / dt) + 1
    T = -np.ones(N)
    X = -np.ones(N)
    V = -np.ones(N)
    A = -np.ones(N)
    RPM_arr = -np.ones(N)
    TPS = -np.ones(N)
    BPS = -np.ones(N)
    GEAR = -np.ones(N, dtype=int)
    MODE = -np.ones(N, dtype=int)
    
    # --- Initial Conditions ---
    t = 0.0
    x = 0.0
    v = 0.0
    a_val = 0.0
    gear = 1         # starting in first gear
    gear_prev = 1
    shifting = False
    rpm = 0.0
    tps_val = 0.0
    bps_val = 0.0
    trap_number = 0  # using 0-index (first trap is speed_trap[0])
    check_speed_traps = True
    i = 0

    # --- Setup Simulation Output File and HUD ---
    os.makedirs('OpenDRAG Sims', exist_ok=True)
    if use_date_time_in_name:
        date_time = "_" + time.strftime("%Y_%m_%d") + "_" + time.strftime("%H_%M_%S")
    else:
        date_time = ""
    simname = f"OpenDRAG Sims/OpenDRAG_{veh['name']}{date_time}"
    if os.path.exists(simname + ".log"):
        os.remove(simname + ".log")
    # (For simplicity we print to console instead of using diary)
    print("=======================================================================================")
    print(f"Vehicle: {veh['name']}")
    print("Date:    " + time.strftime("%d/%m/%Y"))
    print("Time:    " + time.strftime("%H:%M:%S"))
    print("=======================================================================================")
    print("Acceleration simulation started:")
    print(f"Initial Speed: {v*3.6:.2f} km/h")
    print("|_______Comment________|_Speed_|_Accel_|_EnRPM_|_Gear__|_Tabs__|_Xabs__|_Trel__|_Xrel_|")

    # --- Acceleration Phase ---
    acc_start_time = time.time()
    while True:
        # Save current values
        MODE[i] = 1
        T[i] = t
        X[i] = x
        V[i] = v
        A[i] = a_val
        RPM_arr[i] = rpm
        TPS[i] = tps_val
        BPS[i] = 0  # No brake in acceleration phase
        GEAR[i] = gear

        # Check if vehicle has reached maximum speed (assume veh has key "v_max")
        if v >= veh["v_max"]:
            print("Engine speed limited")
            hud(v, a_val, rpm, gear, t, x, 0, 0)
            break
        if i == N - 1:
            print(f"Did not reach maximum speed at time {t:.3f} s")
            break

        # Compute aerodynamic forces
        Aero_Df = 0.5 * veh["rho"] * veh["factor_Cl"] * veh["Cl"] * veh["A"] * v**2
        Aero_Dr = 0.5 * veh["rho"] * veh["factor_Cd"] * veh["Cd"] * veh["A"] * v**2
        Roll_Dr = veh["Cr"] * (-Aero_Df + Wz)
        Wd = (veh["factor_drive"] * Wz + (-veh["factor_aero"] * Aero_Df)) / veh["driven_wheels"]
        ax_drag = (Aero_Dr + Roll_Dr + Wx) / M

        # Speed trap check
        if check_speed_traps and trap_number < len(speed_trap):
            if v >= speed_trap[trap_number]:
                print(f"Speed Trap #{trap_number+1} {round(speed_trap[trap_number]*3.6)} km/h")
                hud(v, a_val, rpm, gear, t, x, 0, 0)
                trap_number += 1
                if trap_number >= len(speed_trap):
                    check_speed_traps = False

        # RPM calculation: if shifting (gear==0) use previous gear
        if gear == 0:
            rpm = rf * rg[gear_prev - 1] * rp * v / Rt * 60 / (2 * math.pi)
            rpm_shift = shift_points[gear_prev - 1]
        else:
            rpm = rf * rg[gear - 1] * rp * v / Rt * 60 / (2 * math.pi)
            rpm_shift = shift_points[gear - 1]

        # Check for gear shift conditions
        if (rpm >= rpm_shift) and (not shifting):
            if gear == veh["nog"]:
                print("Engine speed limited")
                hud(v, a_val, rpm, gear, t, x, 0, 0)
                break
            else:
                shifting = True
                t_shift = t
                ax = 0
                gear_prev = gear
                gear = 0  # Neutral during shift
        elif shifting:
            ax = 0
            if t - t_shift > veh["shift_time"]:
                print(f"Shifting to gear #{gear_prev+1}")
                hud(v, a_val, rpm, gear_prev+1, t, x, 0, 0)
                shifting = False
                gear = gear_prev + 1
        else:
            # Maximum longitudinal acceleration available from tyres
            ax_tyre_max_acc = (1 / M) * (mux + dmx * (Nx - Wd)) * Wd * veh["driven_wheels"]
            # Engine power limit via interpolation
            engine_torque = np.interp(rpm, rpm_curve, torque_curve)
            wheel_torque = engine_torque * rf * rg[gear - 1] * rp * nf * ng_eff * np_eff
            ax_power_limit = (1 / M) * (wheel_torque / Rt)
            ax = min(ax_power_limit, ax_tyre_max_acc)

        # Throttle position (normalized)
        tps_val = ax / ax_power_limit if ax_power_limit != 0 else 0
        # Total longitudinal acceleration is sum of engine and drag contributions
        a_val = ax + ax_drag

        # Check drag limitation: if throttle is at 100% and (ax + ax_drag) is too small, then simulation stops
        if (tps_val == 1) and (ax + ax_drag <= ax_sens):
            print("Drag limited")
            hud(v, a_val, rpm, gear, t, x, 0, 0)
            break

        # Update states using constant acceleration over dt
        x = x + v * dt + 0.5 * a_val * dt**2
        v = v + a_val * dt
        t = t + dt
        i += 1

    i_acc = i  # Save index at end of acceleration phase
    a_acc_ave = v / t if t > 0 else 0
    print(f"Average acceleration: {a_acc_ave/9.81:6.3f} [G]")
    print(f"Peak acceleration   : {np.max(A[:i])/9.81:6.3f} [G]")
    print(f"Acceleration phase time: {t:.3f} s")
    print("--------------------------------------------------------------")

    # --- Deceleration Phase ---
    decel_start_time = time.time()
    t_start = t
    x_start = x
    check_speed_traps = True
    # Active braking speed traps: consider only those traps below current speed
    speed_trap_decel = speed_trap[speed_trap <= v]
    trap_number = len(speed_trap_decel) - 1  # use last index
    print("Deceleration simulation started:")
    print(f"Initial Speed: {v*3.6:.2f} km/h")
    print("|_______Comment________|_Speed_|_Accel_|_EnRPM_|_Gear__|_Tabs__|_Xabs__|_Trel__|_Xrel_|")

    while True:
        MODE[i] = 2
        T[i] = t
        X[i] = x
        V[i] = v
        A[i] = a_val
        RPM_arr[i] = rpm
        TPS[i] = 0
        BPS[i] = bps_val
        GEAR[i] = gear

        if v <= 0:
            v = 0
            print("Stopped")
            hud(v, a_val, rpm, gear, t, x, t_start, x_start)
            break
        if i == N - 1:
            print(f"Did not stop at time {t:.3f} s")
            break

        if check_speed_traps and trap_number >= 0:
            if v <= speed_trap_decel[trap_number]:
                print(f"Speed Trap #{trap_number+1} {round(speed_trap_decel[trap_number]*3.6)} km/h")
                hud(v, a_val, rpm, gear, t, x, t_start, x_start)
                trap_number -= 1
                if trap_number < 0:
                    check_speed_traps = False

        # Compute aerodynamic forces again
        Aero_Df = 0.5 * veh["rho"] * veh["factor_Cl"] * veh["Cl"] * veh["A"] * v**2
        Aero_Dr = 0.5 * veh["rho"] * veh["factor_Cd"] * veh["Cd"] * veh["A"] * v**2
        Roll_Dr = veh["Cr"] * (-Aero_Df + Wz)
        ax_drag = (Aero_Dr + Roll_Dr + Wx) / M

        # For deceleration, update gear and rpm from vehicle speed maps (assumed stored in veh)
        # Here we use linear interpolation of v over veh['vehicle_speed'] and veh['engine_speed']
        gear = int(np.interp(v, veh["vehicle_speed"], veh["gear"]))
        rpm = np.interp(v, veh["vehicle_speed"], veh["engine_speed"])
        # Maximum long deceleration available from tyres
        ax_tyre_max_dec = - (1 / M) * (mux + dmx * (Nx - (Wz - Aero_Df) / 4)) * (Wz - Aero_Df)
        ax = ax_tyre_max_dec
        bps_val = -veh["beta"] * veh["M"] * ax  # brake pressure
        a_val = ax + ax_drag

        # Update states
        x = x + v * dt + 0.5 * a_val * dt**2
        v = v + a_val * dt
        t = t + dt
        i += 1

    a_dec_ave = V[i_acc] / (t - t_start) if (t - t_start) != 0 else 0
    print(f"Average deceleration: {a_dec_ave/9.81:6.3f} [G]")
    print(f"Peak deceleration   : {-np.min(A[:i])/9.81:6.3f} [G]")
    print("Deceleration phase completed.")

    total_sim_time = t
    print(f"Total simulation time: {total_sim_time:.3f} s")

    # --- Compress Results ---
    valid = T != -1
    T = T[valid]
    X = X[valid]
    V = V[valid]
    A = A[valid]
    RPM_arr = RPM_arr[valid]
    TPS = TPS[valid]
    BPS = BPS[valid]
    GEAR = GEAR[valid]
    MODE = MODE[valid]

    # --- Save Results ---
    sim_data = {
        'T': T,
        'X': X,
        'V': V,
        'A': A,
        'RPM': RPM_arr,
        'TPS': TPS,
        'BPS': BPS,
        'GEAR': GEAR,
        'MODE': MODE
    }
    with open(simname + ".pkl", "wb") as f:
        pickle.dump(sim_data, f)
    print("Simulation results saved.")

    # --- Plots ---
    # Create a figure with several subplots
    plt.figure(figsize=(12, 14))
    plt.suptitle(f"OpenDRAG Simulation Results\nVehicle: {veh['name']}   Date & Time: {time.strftime('%Y/%m/%d %H:%M:%S')}", fontsize=14)
    plot_idx = 1
    total_plots = 14  # 7 rows x 2 columns

    # Plot: Distance vs Time
    plt.subplot(7, 2, plot_idx)
    plt.plot(T, X, 'b-')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance [m]')
    plt.title('Traveled Distance')
    plot_idx += 1

    # Plot: Speed vs Time
    plt.subplot(7, 2, plot_idx)
    plt.plot(T, V*3.6, 'r-')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [km/h]')
    plt.title('Speed vs Time')
    plot_idx += 1

    # Plot: Speed vs Distance
    plt.subplot(7, 2, plot_idx)
    plt.plot(X, V*3.6, 'r-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Speed [km/h]')
    plt.title('Speed vs Distance')
    plot_idx += 1

    # Plot: Acceleration vs Time
    plt.subplot(7, 2, plot_idx)
    plt.plot(T, A, 'g-')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Acceleration vs Time')
    plot_idx += 1

    # Plot: Acceleration vs Distance
    plt.subplot(7, 2, plot_idx)
    plt.plot(X, A, 'g-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Acceleration [m/s²]')
    plt.title('Acceleration vs Distance')
    plot_idx += 1

    # Plot: Engine Speed vs Time
    plt.subplot(7, 2, plot_idx)
    plt.plot(T, RPM_arr, 'm-')
    plt.xlabel('Time [s]')
    plt.ylabel('Engine Speed [rpm]')
    plt.title('Engine Speed vs Time')
    plot_idx += 1

    # Plot: Engine Speed vs Distance
    plt.subplot(7, 2, plot_idx)
    plt.plot(X, RPM_arr, 'm-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Engine Speed [rpm]')
    plt.title('Engine Speed vs Distance')
    plot_idx += 1

    # Plot: Gear vs Time
    plt.subplot(7, 2, plot_idx)
    plt.plot(T, GEAR, 'c-')
    plt.xlabel('Time [s]')
    plt.ylabel('Gear [-]')
    plt.title('Gear vs Time')
    plot_idx += 1

    # Plot: Gear vs Distance
    plt.subplot(7, 2, plot_idx)
    plt.plot(X, GEAR, 'c-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Gear [-]')
    plt.title('Gear vs Distance')
    plot_idx += 1

    # Plot: Throttle Position vs Time
    plt.subplot(7, 2, plot_idx)
    plt.plot(T, TPS*100, 'y-')
    plt.xlabel('Time [s]')
    plt.ylabel('Throttle [%]')
    plt.title('Throttle Position vs Time')
    plot_idx += 1

    # Plot: Throttle Position vs Distance
    plt.subplot(7, 2, plot_idx)
    plt.plot(X, TPS*100, 'y-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Throttle [%]')
    plt.title('Throttle Position vs Distance')
    plot_idx += 1

    # Plot: Brake Pressure vs Time
    plt.subplot(7, 2, plot_idx)
    plt.plot(T, BPS/1e5, 'k-')
    plt.xlabel('Time [s]')
    plt.ylabel('Brake Pressure [bar]')
    plt.title('Brake Pressure vs Time')
    plot_idx += 1

    # Plot: Brake Pressure vs Distance
    plt.subplot(7, 2, plot_idx)
    plt.plot(X, BPS/1e5, 'k-')
    plt.xlabel('Distance [m]')
    plt.ylabel('Brake Pressure [bar]')
    plt.title('Brake Pressure vs Distance')
    plot_idx += 1

    plt.tight_layout()
    plt.savefig(simname + ".png")
    plt.show()
    print("Plots created and saved.")
    
if __name__ == '__main__':
    main()