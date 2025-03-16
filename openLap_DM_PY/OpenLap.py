import os
import math
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

# ------------------------------
# Helper Functions
# ------------------------------

def disp_logo(logid):
    logo = (
        "_______                    _____________________ \n"
        "__  __ \\______________________  /___    |__  __ \n"
        "_  / / /__  __ \\  _ \\_  __ \\_  / __  /| |_  /_/ /\n"
        "/ /_/ /__  /_/ /  __/  / / /  /___  ___ |  ____/ \n"
        "\\____/ _  .___/\\___//_/ /_//_____/_/  |_/_/      \n"
        "       /_/                                       \n"
    )
    print(logo)
    logid.write(logo + "\n")

def smooth(data, window):
    """Smooth data using a Savitzky–Golay filter."""
    if window < 3:
        return data
    if window % 2 == 0:
        window += 1
    return savgol_filter(data, window, polyorder=2)

def next_point(j, j_max, mode, tr_config):
    """Determine the next point index (MATLAB is 1-indexed)."""
    if mode == 1:  # acceleration
        if tr_config.lower() == 'closed':
            if j == j_max - 1:
                return 1, j_max
            elif j == j_max:
                return j + 1, 1
            else:
                return j + 1, j + 1
        elif tr_config.lower() == 'open':
            return j + 1, j + 1
    elif mode == -1:  # deceleration
        if tr_config.lower() == 'closed':
            if j == 2:
                return j_max, 1
            elif j == 1:
                return j_max, j - 1
            else:
                return j - 1, j - 1
        elif tr_config.lower() == 'open':
            return j - 1, j - 1

def other_points(i, i_max):
    """Return a list of indices except i (1-indexed)."""
    return [j for j in range(1, i_max + 1) if j != i]

def flag_update(flag, j, k, prg_size, logid, prg_pos):
    """Update flag state and (optionally) the progress bar."""
    flag[j - 1, k - 1] = True
    return flag

def progress_bar(flag, prg_size, logid, prg_pos):
    p = np.sum(flag) / flag.size
    n = int(p * prg_size)
    bar = "|" * n + " " * (prg_size - n)
    print(f"Running: [{bar}] {p*100:4.0f} [%]", end="\r")
    logid.seek(prg_pos)
    logid.write(f"Running: [{bar}] {p*100:4.0f} [%]\n")
    
# ------------------------------
# Vehicle and Track Model Functions
# ------------------------------

def vehicle_model_lat(veh, tr, p):
    """
    Calculate maximum speed at track mesh point p based on lateral limits.
    (For a straight, v = veh['v_max']; for a corner, a simplified model is used.)
    """
    g = 9.81
    r = tr['r'][p - 1]  # converting from 1-indexed
    incl = tr['incl'][p - 1]
    bank = tr['bank'][p - 1]
    # For straight segments:
    if r == 0:
        return veh['v_max'], 1.0, 0.0  # full throttle, no braking
    else:
        # Simplified corner model: limit speed by lateral grip
        v = min(veh['v_max'], math.sqrt(g * abs(r)))
        return v, 1.0, 0.0

def vehicle_model_comb(veh, tr, v, v_max_next, j, mode):
    """
    Compute combined longitudinal acceleration command from vehicle model.
    Returns:
      v_next: predicted speed at next mesh point,
      ax: net longitudinal acceleration [m/s^2],
      ay: (set to 0 here for simplicity),
      tps: throttle position (0 to 1),
      bps: brake pressure (dummy value),
      overshoot: boolean flag if predicted speed overshoots next point limit.
    """
    dx = tr['dx'][j - 1]
    r = tr['r'][j - 1]
    incl = tr['incl'][j - 1]
    bank = tr['bank'][j - 1]
    g = 9.81
    M = veh['M']
    # Aero forces (simplified)
    Aero_Df = 0.5 * veh['rho'] * veh['factor_Cl'] * veh['Cl'] * veh['A'] * v**2
    Aero_Dr = 0.5 * veh['rho'] * veh['factor_Cd'] * veh['Cd'] * veh['A'] * v**2
    Roll_Dr = veh['Cr'] * (-Aero_Df + M * g * math.cos(math.radians(bank)) * math.cos(math.radians(incl)))
    ax_drag = (Aero_Dr + Roll_Dr) / M
    ax_max = mode * (v_max_next**2 - v**2) / (2 * dx)
    ax_needed = ax_max - ax_drag
    # Simplified driver inputs:
    if ax_needed >= 0:
        tps = 1.0
        bps = 0.0
        ax_com = ax_needed
    else:
        tps = 0.0
        bps = 1.0  # dummy brake input
        ax_com = ax_needed
    ax_total = ax_com + ax_drag
    v_next = math.sqrt(max(0, v**2 + 2 * mode * ax_total * dx))
    overshoot = v_next > v_max_next
    return v_next, ax_total, 0.0, tps, bps, overshoot

# ------------------------------
# Simulation Function
# ------------------------------

def simulate(veh, tr, simname, logid):
    start_solver = time.time()
    print("Simulation started.")
    logid.write("Simulation started.\n")
    
    n_points = len(tr['x'])
    v_max = np.zeros(n_points)
    tps_v_max = np.zeros(n_points)
    bps_v_max = np.zeros(n_points)
    for i in range(1, n_points + 1):
        v, tps, bps = vehicle_model_lat(veh, tr, i)
        v_max[i - 1] = v
        tps_v_max[i - 1] = tps
        bps_v_max[i - 1] = bps
    print("Maximum speed calculated at all points.")
    logid.write("Maximum speed calculated at all points.\n")
    
    # Find apexes (where v_max is minimal in corners)
    peaks, _ = find_peaks(-v_max)
    if peaks.size > 0:
        v_apex = v_max[peaks]
    else:
        idx = np.argmin(v_max)
        peaks = np.array([idx])
        v_apex = np.array([v_max[idx]])
    # For open configuration, ensure standing start at index 0
    if tr['info']['config'].lower() == 'open':
        if peaks[0] != 0:
            peaks = np.insert(peaks, 0, 0)
            v_apex = np.insert(v_apex, 0, 0)
        else:
            v_apex[0] = 0
    print("Found all apexes on track.")
    logid.write("Found all apexes on track.\n")
    
    # Reorder apexes (for solver efficiency) – here we simply sort by speed
    sort_idx = np.argsort(v_apex)
    v_apex = v_apex[sort_idx]
    apex = peaks[sort_idx]
    tps_apex = tps_v_max[apex]
    bps_apex = bps_v_max[apex]
    
    N_apex = len(apex)
    n = n_points
    # Preallocate simulation arrays: dimensions (n, N_apex, 2) for acceleration and deceleration modes
    v_sim = np.full((n, N_apex, 2), np.inf, dtype=np.float32)
    ax_sim = np.zeros((n, N_apex, 2), dtype=np.float32)
    tps_sim = np.zeros((n, N_apex, 2), dtype=np.float32)
    bps_sim = np.zeros((n, N_apex, 2), dtype=np.float32)
    flag = np.zeros((n, 2), dtype=bool)
    
    prg_size = 30
    prg_pos = logid.tell()
    print("Running: [" + " " * prg_size + "]  0 [%]")
    logid.write("Running: [" + " " * prg_size + "]  0 [%]\n")
    
    # Run simulation for each apex and for both acceleration (mode 1) and deceleration (mode -1)
    for i in range(N_apex):
        for mode_idx, mode in enumerate([1, -1]):
            # For open configuration, skip deceleration at standing start
            if tr['info']['config'].lower() == 'open' and mode == -1 and i == 0:
                continue
            # Starting from apex point (convert to 1-index)
            j = int(apex[i]) + 1
            v_sim[j - 1, i, mode_idx] = v_apex[i]
            flag[j - 1, mode_idx] = True
            # Get next point index
            j_next, j = next_point(j, n, mode, tr['info']['config'])
            while True:
                logid.write(f"{i+1}\t{j}\t{mode}\t{tr['x'][j-1]:7.1f}\t{v_sim[j-1,i,mode_idx]:7.2f}\t{v_max[j-1]:7.2f}\n")
                v_next, ax_val, _, tps_val, bps_val, overshoot = vehicle_model_comb(
                    veh, tr, v_sim[j - 1, i, mode_idx], v_max[j_next - 1], j, mode)
                v_sim[j_next - 1, i, mode_idx] = v_next
                ax_sim[j - 1, i, mode_idx] = ax_val
                tps_sim[j - 1, i, mode_idx] = tps_val
                bps_sim[j - 1, i, mode_idx] = bps_val
                if overshoot:
                    break
                if flag[j - 1, mode_idx]:
                    break
                j_next, j = next_point(j, n, mode, tr['info']['config'])
                if tr['info']['config'].lower() == 'closed' and j == apex[i]:
                    break
                if tr['info']['config'].lower() == 'open' and (j == n or j == 1):
                    break
            # (Optional) update progress bar here by calling flag_update and progress_bar
            flag = flag_update(flag, j, mode_idx + 1, prg_size, logid, prg_pos)
    
    progress_bar(flag, prg_size, logid, prg_pos)
    print()
    print("Velocity profile calculated.")
    logid.write("Velocity profile calculated.\n")
    solver_time = time.time() - start_solver
    print(f"Solver time is: {solver_time:.3f} [s]")
    logid.write(f"Solver time is: {solver_time:.3f} [s]\n")
    print("Post-processing initialised.")
    logid.write("Post-processing initialised.\n")
    
    # Select best solution for each track point (choose minimum speed over all apex iterations and modes)
    V_final = np.zeros(n, dtype=np.float32)
    AX_final = np.zeros(n, dtype=np.float32)
    for j in range(n):
        candidates = np.concatenate((v_sim[j, :, 0], v_sim[j, :, 1]))
        idx = np.argmin(candidates)
        V_final[j] = candidates[idx]
        if idx < N_apex:
            AX_final[j] = ax_sim[j, idx, 0]
        else:
            AX_final[j] = ax_sim[j, idx - N_apex, 1]
    
    # Calculate cumulative time using dx and V_final (avoid division by zero)
    time_array = np.cumsum(tr['dx'] / np.maximum(V_final, 1e-3))
    sector_time = []
    for sec in range(1, int(max(tr['sector'])) + 1):
        mask = (tr['sector'] == sec)
        if np.any(mask):
            sector_time.append(np.max(time_array[mask]) - np.min(time_array[mask]))
        else:
            sector_time.append(0)
    laptime = time_array[-1]
    print("Laptime calculated.")
    logid.write("Laptime calculated.\n")
    
    # Calculate forces (simplified)
    Fz_mass = -veh['M'] * 9.81 * np.cos(np.radians(tr['bank'])) * np.cos(np.radians(tr['incl']))
    Fz_aero = 0.5 * veh['rho'] * veh['factor_Cl'] * veh['Cl'] * veh['A'] * V_final**2
    Fz_total = Fz_mass + Fz_aero
    Fx_aero = 0.5 * veh['rho'] * veh['factor_Cd'] * veh['Cd'] * veh['A'] * V_final**2
    Fx_roll = veh['Cr'] * np.abs(Fz_total)
    
    # Yaw rate and steering (simplified)
    yaw_rate = V_final * tr['r']
    delta = np.degrees(np.arctan(veh['L'] * tr['r']))
    beta = np.zeros(n)  # Not computed in detail
    steer = delta * veh['rack']
    
    # Engine metrics (using interpolation, simplified)
    wheel_torque = tps_sim[:, 0, 0] * np.interp(V_final, veh['vehicle_speed'], veh['wheel_torque'])
    Fx_eng = wheel_torque / veh['tyre_radius']
    engine_torque = tps_sim[:, 0, 0] * np.interp(V_final, veh['vehicle_speed'], veh['engine_torque'])
    engine_power = tps_sim[:, 0, 0] * np.interp(V_final, veh['vehicle_speed'], veh['engine_power'])
    engine_speed = np.interp(V_final, veh['vehicle_speed'], veh['engine_speed'])
    gear_sol = np.interp(V_final, veh['vehicle_speed'], veh['gear'], left=1, right=veh['nog'])
    fuel_cons = np.cumsum(wheel_torque / veh['tyre_radius'] * tr['dx'] /
                            (veh['n_primary'] * veh['n_gearbox'] * veh['n_final'] *
                             veh['n_thermal'] * veh['fuel_LHV']))
    fuel_cons_total = fuel_cons[-1]
    
    # KPIs (simplified)
    percent_in_corners = np.sum(tr['r'] != 0) / n * 100
    percent_in_accel = np.sum(tps_sim[:, 0, 0] > 0) / n * 100
    percent_in_decel = np.sum(bps_sim[:, 0, 0] > 0) / n * 100
    percent_in_coast = np.sum((bps_sim[:, 0, 0] == 0) & (tps_sim[:, 0, 0] == 0)) / n * 100
    percent_in_full_tps = np.sum(tps_sim[:, 0, 0] == 1) / n * 100
    percent_in_gear = np.zeros(veh['nog'])
    for i in range(veh['nog']):
        percent_in_gear[i] = np.sum(np.round(gear_sol) == (i + 1)) / n * 100
    energy_spent_fuel = fuel_cons * veh['fuel_LHV']
    energy_spent_mech = energy_spent_fuel * veh['n_thermal']
    gear_shifts = np.sum(np.abs(np.diff(np.round(gear_sol))))
    ay_max = np.max(np.abs(AX_final))
    ax_max_val = np.max(AX_final)
    ax_min_val = np.min(AX_final)
    sector_v_max = np.array([np.max(V_final[tr['sector'] == sec]) for sec in range(1, int(max(tr['sector'])) + 1)])
    sector_v_min = np.array([np.min(V_final[tr['sector'] == sec]) for sec in range(1, int(max(tr['sector'])) + 1)])
    
    print("KPIs calculated.")
    logid.write("KPIs calculated.\n")
    print("Post-processing finished.")
    logid.write("Post-processing finished.\n")
    
    # Assemble simulation results into a dictionary
    sim_res = {
        'sim_name': {'data': simname},
        'distance': {'data': tr['x'], 'unit': 'm'},
        'time': {'data': time_array, 'unit': 's'},
        'N': {'data': N_apex, 'unit': None},
        'apex': {'data': apex, 'unit': None},
        'speed_max': {'data': v_max, 'unit': 'm/s'},
        'flag': {'data': flag, 'unit': None},
        'v': {'data': v_sim, 'unit': 'm/s'},
        'Ax': {'data': ax_sim, 'unit': 'm/s^2'},
        'Ay': {'data': np.zeros_like(ax_sim), 'unit': 'm/s^2'},
        'tps': {'data': tps_sim, 'unit': None},
        'bps': {'data': bps_sim, 'unit': None},
        'elevation': {'data': tr['Z'], 'unit': 'm'},
        'speed': {'data': V_final, 'unit': 'm/s'},
        'yaw_rate': {'data': yaw_rate, 'unit': 'rad/s'},
        'long_acc': {'data': AX_final, 'unit': 'm/s^2'},
        'lat_acc': {'data': np.zeros_like(AX_final), 'unit': 'm/s^2'},
        'sum_acc': {'data': AX_final, 'unit': 'm/s^2'},
        'throttle': {'data': tps_sim, 'unit': None},
        'brake_pres': {'data': bps_sim, 'unit': 'Pa'},
        'brake_force': {'data': bps_sim * veh['phi'], 'unit': 'N'},
        'steering': {'data': steer, 'unit': 'deg'},
        'delta': {'data': delta, 'unit': 'deg'},
        'beta': {'data': beta, 'unit': 'deg'},
        'Fz_aero': {'data': Fz_aero, 'unit': 'N'},
        'Fx_aero': {'data': Fx_aero, 'unit': 'N'},
        'Fx_eng': {'data': Fx_eng, 'unit': 'N'},
        'Fx_roll': {'data': Fx_roll, 'unit': 'N'},
        'Fz_mass': {'data': Fz_mass, 'unit': 'N'},
        'Fz_total': {'data': Fz_total, 'unit': 'N'},
        'wheel_torque': {'data': wheel_torque, 'unit': 'N.m'},
        'engine_torque': {'data': engine_torque, 'unit': 'N.m'},
        'engine_power': {'data': engine_power, 'unit': 'W'},
        'engine_speed': {'data': engine_speed, 'unit': 'rpm'},
        'gear': {'data': gear_sol, 'unit': None},
        'fuel_cons': {'data': fuel_cons, 'unit': 'kg'},
        'fuel_cons_total': {'data': fuel_cons_total, 'unit': 'kg'},
        'laptime': {'data': laptime, 'unit': 's'},
        'sector_time': {'data': np.array(sector_time), 'unit': 's'},
        'percent_in_corners': {'data': percent_in_corners, 'unit': '%'},
        'percent_in_accel': {'data': percent_in_accel, 'unit': '%'},
        'percent_in_decel': {'data': percent_in_decel, 'unit': '%'},
        'percent_in_coast': {'data': percent_in_coast, 'unit': '%'},
        'percent_in_full_tps': {'data': percent_in_full_tps, 'unit': '%'},
        'percent_in_gear': {'data': percent_in_gear, 'unit': '%'},
        'v_min': {'data': np.min(V_final), 'unit': 'm/s'},
        'v_max': {'data': np.max(V_final), 'unit': 'm/s'},
        'v_ave': {'data': np.mean(V_final), 'unit': 'm/s'},
        'energy_spent_fuel': {'data': energy_spent_fuel, 'unit': 'J'},
        'energy_spent_mech': {'data': energy_spent_mech, 'unit': 'J'},
        'gear_shifts': {'data': gear_shifts, 'unit': None},
        'lat_acc_max': {'data': ay_max, 'unit': 'm/s^2'},
        'long_acc_max': {'data': ax_max_val, 'unit': 'm/s^2'},
        'long_acc_min': {'data': ax_min_val, 'unit': 'm/s^2'},
        'sector_v_max': {'data': sector_v_max, 'unit': 'm/s'},
        'sector_v_min': {'data': sector_v_min, 'unit': 'm/s'}
    }
    
    return sim_res

def export_report(veh, tr, sim, freq, logid):
    freq = round(freq)
    # Export only vector channels (assumed to have length equal to len(tr['x']))
    channel_names = []
    data_list = []
    units = []
    for key, val in sim.items():
        data = np.array(val['data']).flatten()
        if data.shape[0] == len(tr['x']):
            channel_names.append(key)
            data_list.append(data)
            units.append(val['unit'])
    # New time vector
    t_new = np.arange(0, sim['laptime']['data'] + 1/freq, 1/freq)
    time_orig = sim['time']['data']
    time_data = []
    for data in data_list:
        time_data.append(np.interp(t_new, time_orig, data))
    time_data = np.column_stack(time_data)
    
    filename = sim['sim_name']['data'] + ".csv"
    with open(filename, 'w') as fid:
        fid.write("Format,OpenLAP Export\n")
        fid.write(f"Venue,{tr['info']['name']}\n")
        fid.write(f"Vehicle,{veh['name']}\n")
        fid.write("Driver,OpenLap\n")
        fid.write("Device\n")
        fid.write("Comment\n")
        fid.write(f"Date,{datetime.now().strftime('%d/%m/%Y')}\n")
        fid.write(f"Time,{datetime.now().strftime('%H:%M:%S')}\n")
        fid.write(f"Frequency,{freq}\n")
        fid.write("\n\n\n\n\n")
        header_line = ",".join(channel_names) + "\n"
        fid.write(header_line)
        fid.write(header_line)
        units_line = ",".join([str(u) for u in units]) + "\n"
        fid.write(units_line)
        fid.write("\n\n")
        for row in time_data:
            row_str = ",".join([f"{val:.6f}" for val in row]) + "\n"
            fid.write(row_str)
    print("Exported .csv file successfully.")
    logid.write("Exported .csv file successfully.\n")

# ------------------------------
# Main Script
# ------------------------------

if __name__ == '__main__':
    start_time = time.time()
    # Filenames (assumed to be pickle files)
    trackfile = 'OpenTRACK Tracks/OpenTRACK_Spa-Francorchamps_Closed_Forward.pkl'
    vehiclefile = 'OpenVEHICLE Vehicles/OpenVEHICLE_Formula 1_Open Wheel.pkl'
    
    with open(trackfile, 'rb') as f:
        tr = pickle.load(f)
    with open(vehiclefile, 'rb') as f:
        veh = pickle.load(f)
        
    freq = 50  # Hz
    use_date_time_in_name = False
    date_time = "" if not use_date_time_in_name else "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    simname = f"OpenLAP Sims/OpenLAP_{veh['name']}_{tr['info']['name']}{date_time}"
    logfile = simname + ".log"
    os.makedirs("OpenLAP Sims", exist_ok=True)
    if os.path.exists(logfile):
        os.remove(logfile)
    logid = open(logfile, 'w')
    disp_logo(logid)
    header = (
        "=================================================\n"
        f"Vehicle: {veh['name']}\n"
        f"Track:   {tr['info']['name']}\n"
        f"Date:    {datetime.now().strftime('%d/%m/%Y')}\n"
        f"Time:    {datetime.now().strftime('%H:%M:%S')}\n"
        "=================================================\n"
    )
    print(header)
    logid.write(header)
    
    sim = simulate(veh, tr, simname, logid)
    
    print(f"Laptime:  {sim['laptime']['data']:.3f} [s]")
    logid.write(f"Laptime   : {sim['laptime']['data']:.3f} [s]\n")
    for i in range(1, int(max(tr['sector'])) + 1):
        sec_time = sim['sector_time']['data'][i - 1]
        print(f"Sector {i}: {sec_time:.3f} [s]")
        logid.write(f"Sector {i}: {sec_time:.3f} [s]\n")
    
    # --- Plotting ---
    plt.figure(figsize=(9, 9))
    # Speed plot
    plt.subplot(7, 2, 1)
    plt.plot(tr['x'], sim['speed']['data'] * 3.6)
    plt.legend(['Speed'], loc='upper right')
    plt.xlabel('Distance [m]')
    plt.ylabel('Speed [km/h]')
    plt.xlim([tr['x'][0], tr['x'][-1]])
    plt.grid(True)
    
    # Elevation and curvature
    plt.subplot(7, 2, 3)
    plt.plot(tr['x'], tr['Z'], label='Elevation')
    plt.plot(tr['x'], tr['r'], label='Curvature')
    plt.xlabel('Distance [m]')
    plt.ylabel('Elevation [m] / Curvature [1/m]')
    plt.xlim([tr['x'][0], tr['x'][-1]])
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Accelerations
    plt.subplot(7, 2, 5)
    plt.plot(tr['x'], sim['long_acc']['data'], label='LonAcc')
    plt.plot(tr['x'], sim['lat_acc']['data'], label='LatAcc')
    plt.plot(tr['x'], sim['sum_acc']['data'], 'k:', label='GSum')
    plt.xlabel('Distance [m]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlim([tr['x'][0], tr['x'][-1]])
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Drive inputs
    plt.subplot(7, 2, 7)
    plt.plot(tr['x'], sim['throttle']['data'] * 100, label='tps')
    plt.plot(tr['x'], sim['brake_pres']['data'] / 1e5, label='bps')
    plt.xlabel('Distance [m]')
    plt.ylabel('Input [%]')
    plt.xlim([tr['x'][0], tr['x'][-1]])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.ylim([-10, 110])
    
    # Steering inputs
    plt.subplot(7, 2, 9)
    plt.plot(tr['x'], sim['steering']['data'], label='Steering wheel')
    plt.plot(tr['x'], sim['delta']['data'], label='Steering δ')
    plt.plot(tr['x'], sim['beta']['data'], label='Vehicle slip angle β')
    plt.xlabel('Distance [m]')
    plt.ylabel('Angle [deg]')
    plt.xlim([tr['x'][0], tr['x'][-1]])
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # GGV circle (3D plot)
    ax = plt.subplot(7, 2, 11, projection='3d')
    ax.scatter(sim['lat_acc']['data'], sim['long_acc']['data'], sim['speed']['data'] * 3.6, c='r', marker='o')
    X_ggv = veh['GGV'][:, :, 1]
    Y_ggv = veh['GGV'][:, :, 0]
    Z_ggv = veh['GGV'][:, :, 2] * 3.6
    ax.plot_surface(X_ggv, Y_ggv, Z_ggv, alpha=0.8, edgecolor='none')
    ax.set_xlabel('LatAcc [m/s^2]')
    ax.set_ylabel('LonAcc [m/s^2]')
    ax.set_zlabel('Speed [km/h]')
    ax.view_init(105, 5)
    ax.grid(True)
    
    # Track map
    plt.subplot(7, 2, 12)
    plt.scatter(tr['X'], tr['Y'], s=5, c=sim['speed']['data'] * 3.6)
    plt.plot(tr['arrow'][:, 0], tr['arrow'][:, 1], 'k', linewidth=2)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.colorbar()
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(simname + ".fig")
    plt.show()
    print("Plots created and saved.")
    logid.write("Plots created and saved.\n")
    
    # --- Report generation ---
    export_report(veh, tr, sim, freq, logid)
    
    with open(simname + ".pkl", "wb") as f:
        pickle.dump({'veh': veh, 'tr': tr, 'sim': sim}, f)
    print("Simulation completed successfully.")
    logid.write("Simulation completed successfully.\n")
    elapsed = time.time() - start_time
    print(f"Elapsed time is: {elapsed:.3f} [s]")
    logid.write(f"Elapsed time is: {elapsed:.3f} [s]\n")
    logid.close()