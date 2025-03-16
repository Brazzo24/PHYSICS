import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.signal import savgol_filter, find_peaks
import pickle

# ------------------------------
# Helper Functions
# ------------------------------
trackname = "myTrack"

def rotz(theta_deg):
    """Return a 3x3 rotation matrix about the z-axis (theta in degrees)."""
    theta = np.radians(theta_deg)
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

def smooth(data, window):
    """Smooth data using a Savitzky–Golay filter.
    If the window length is even or too small, adjust accordingly."""
    if window < 3:
        return data
    if window % 2 == 0:
        window += 1
    return savgol_filter(data, window, polyorder=2)

def read_info(filename, sheet_name='Info', start_row=0, end_row=7):
    """Reads the track info from an Excel file into a dictionary."""
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    df = df.iloc[start_row:end_row, :2]
    info = {
        'name': str(df.iloc[0, 1]),
        'country': str(df.iloc[1, 1]),
        'city': str(df.iloc[2, 1]),
        'type': str(df.iloc[3, 1]),
        'config': str(df.iloc[4, 1]),
        'direction': str(df.iloc[5, 1]),
        'mirror': str(df.iloc[6, 1])
    }
    return info

def read_shape_data(filename, sheet_name='Shape'):
    """Reads shape data from the given Excel sheet."""
    return pd.read_excel(filename, sheet_name=sheet_name)

def read_data(filename, sheet_name):
    """Reads a two‐column table from an Excel file."""
    return pd.read_excel(filename, sheet_name=sheet_name)

def read_logged_data(filename, header_start_row=0, header_end_row=12, data_start_row=13):
    """Reads logged data from a CSV file.
       Returns a DataFrame for the header and another for the numerical data."""
    header = pd.read_csv(filename, nrows=header_end_row - header_start_row, header=None)
    data = pd.read_csv(filename, skiprows=data_start_row, header=None)
    return header, data

# ------------------------------
# Main OpenTRACK Script
# ------------------------------

def main():
    # === Track file selection ===
    # (Uncomment the file you wish to use)
    # filename = 'Paul Ricard data.csv'
    # filename = 'Spa-Francorchamps.xlsx'
    # filename = 'Monza Data.csv'
    # filename = 'OpenTRACK Laguna Seca Data.csv'
    # filename = 'OpenTRACK Paul Ricard Data.csv'
    filename = 'OpenTRACK_FSAE_UK_Endurance_2015.xlsx'
    # filename = 'OpenTRACK KZ2 Kart Data - Rhodes.csv'
    # filename = 'OpenTRACK KZ2 Kart Data - Athens.csv'
    
    # === Mode selection ===
    mode = 'shape data'         # Options: 'shape data' or 'logged data'
    # mode = 'logged data'
    log_mode = 'speed & latacc'  # Options for logged data: 'speed & latacc' or 'speed & yaw'
    # log_mode = 'speed & yaw'
    
    # === Settings ===
    mesh_size = 1           # [m]
    filter_dt = 0.1         # [s] (recommended 0.5 s in MATLAB example; adjust as needed)
    rotation_angle = 0      # [deg] track map rotation
    lambda_val = 1          # shape adjuster
    kappa = 1000            # [deg] long corner adjuster
    
    # === Reading file ===
    print(f"Reading track file: {filename}")
    if mode == 'logged data':
        # --- Logged data mode ---
        head, data = read_logged_data(filename)
        # Extract header info (adjust row indices if necessary)
        info = {
            'name': head.iloc[1, 1],
            'country': head.iloc[2, 1],
            'city': head.iloc[3, 1],
            'type': head.iloc[4, 1],
            'config': head.iloc[5, 1],
            'direction': head.iloc[6, 1],
            'mirror': head.iloc[7, 1]
        }
        freq = float(head.iloc[8, 1])
        # Data columns (0-indexed)
        col_dist, col_vel, col_yaw, col_ay = 0, 1, 2, 3
        col_el, col_bk, col_gf, col_sc = 4, 5, 6, 7
        x = data.iloc[:, col_dist].to_numpy()
        v = data.iloc[:, col_vel].to_numpy()
        w = data.iloc[:, col_yaw].to_numpy()
        ay = data.iloc[:, col_ay].to_numpy()
        el = data.iloc[:, col_el].to_numpy()
        bk = data.iloc[:, col_bk].to_numpy()
        gf = data.iloc[:, col_gf].to_numpy()
        sc = data.iloc[:, col_sc].to_numpy()
        # --- Convert units (if needed) ---
        # For example, if distance is not in meters, convert here.
        # (The MATLAB code uses unit strings from header; here you may hard-code conversions.)
        # Shift x to start at 0:
        x = x - x[0]
        # --- Curvature calculation ---
        if log_mode == 'speed & yaw':
            r = lambda_val * w / v
        elif log_mode == 'speed & latacc':
            r = lambda_val * ay / (v**2)
        r = smooth(r, round(freq * filter_dt))
        if str(info.get('mirror','Off')).strip().lower() == "on":
            r = -r
        L_total = x[-1]
        # Save coarse position vectors:
        xx, xe, xb, xg, xs = x.copy(), x.copy(), x.copy(), x.copy(), x.copy()
    else:
        # --- Shape data mode ---
        info = read_info(filename, sheet_name='Info', start_row=0, end_row=7)
        table_shape = read_shape_data(filename, sheet_name='Shape')
        table_el = read_data(filename, sheet_name='Elevation')
        table_bk = read_data(filename, sheet_name='Banking')
        table_gf = read_data(filename, sheet_name='Grip Factors')
        table_sc = read_data(filename, sheet_name='Sectors')
        # Extract variables from shape table
        R = table_shape.iloc[:, 2].to_numpy()
        l = table_shape.iloc[:, 1].to_numpy()
        type_tmp = table_shape.iloc[:, 0].astype(str).to_numpy()
        R[R == 0] = np.inf  # Correct straight segment radius
        L_total = np.sum(l)
        type_arr = np.zeros(len(l))
        type_arr[np.array(type_tmp) == "Left"] = 1
        type_arr[np.array(type_tmp) == "Right"] = -1
        if str(info.get('mirror','Off')).strip().lower() == "on":
            type_arr = -type_arr
        # Remove zero-length segments
        valid = l != 0
        R, type_arr, l = R[valid], type_arr[valid], l[valid]
        # Inject points at long corners
        angle_seg = np.degrees(l / R)
        RR, ll, tt = list(R), list(l), list(type_arr)
        j = 0
        while j < len(ll):
            if angle_seg[j] > kappa:
                l_inj = min(ll[j] / 3, math.radians(kappa) * R[j])
                ll.insert(j, l_inj)
                ll[j+1] = ll[j+1] - 2 * l_inj
                ll.insert(j+2, l_inj)
                RR.insert(j, RR[j])
                RR.insert(j+1, RR[j])
                RR.insert(j+2, RR[j])
                tt.insert(j, tt[j])
                tt.insert(j+1, tt[j])
                tt.insert(j+2, tt[j])
                j += 3
            else:
                j += 1
            angle_seg = np.degrees(np.array(ll) / np.array(RR))
        # Replace consecutive straights
        i = 0
        while i < len(ll) - 1:
            j = 1
            while (i + j < len(ll)) and (tt[i+j] == 0 and tt[i] == 0 and ll[i] != -1):
                ll[i] += ll[i+j]
                ll[i+j] = -1
                j += 1
            i += 1
        mask = np.array(ll) != -1
        R, type_arr, l = np.array(RR)[mask], np.array(tt)[mask], np.array(ll)[mask]
        X_seg = np.cumsum(l)
        XC = np.cumsum(l) - l/2
        x_list, r_list = [], []
        for i in range(len(X_seg)):
            if np.isinf(R[i]):
                x_list.extend([X_seg[i] - l[i], X_seg[i]])
                r_list.extend([0, 0])
            else:
                x_list.append(XC[i])
                r_list.append(type_arr[i] / R[i])
        x = np.array(x_list)
        r = np.array(r_list)
        # Elevation, banking, grip factors, sector from tables
        xe = table_el.iloc[:, 0].to_numpy()
        el = table_el.iloc[:, 1].to_numpy()
        mask = xe <= L_total
        xe, el = xe[mask], el[mask]
        xb = table_bk.iloc[:, 0].to_numpy()
        bk = table_bk.iloc[:, 1].to_numpy()
        mask = xb <= L_total
        xb, bk = xb[mask], bk[mask]
        xg = table_gf.iloc[:, 0].to_numpy()
        gf = table_gf.iloc[:, 1].to_numpy()
        mask = xg <= L_total
        xg, gf = xg[mask], gf[mask]
        xs = table_sc.iloc[:, 0].to_numpy()
        sc = table_sc.iloc[:, 1].to_numpy()
        mask = xs < L_total
        xs, sc = xs[mask], sc[mask]
    
    print("Pre-processing completed.")
    
    # === Meshing ===
    if math.floor(L_total) < L_total:
        x_fine = np.append(np.arange(0, math.floor(L_total) + mesh_size, mesh_size), L_total)
    else:
        x_fine = np.arange(0, math.floor(L_total) + mesh_size, mesh_size)
    dx = np.diff(x_fine)
    dx = np.append(dx, dx[-1])
    n = len(x_fine)
    # Fine curvature interpolation (using PCHIP)
    pchip_interp = PchipInterpolator(xx, r)
    r_fine = pchip_interp(x_fine)
    # Elevation interpolation (linear)
    f_el = interp1d(xe, el, kind='linear', fill_value="extrapolate")
    Z = f_el(x_fine)
    # Banking interpolation
    f_bk = interp1d(xb, bk, kind='linear', fill_value="extrapolate")
    bank_fine = f_bk(x_fine)
    # Inclination calculation [deg]
    incl = -np.degrees(np.arctan(np.diff(Z) / np.diff(x_fine)))
    incl = np.append(incl, incl[-1])
    # Grip factor interpolation
    f_gf = interp1d(xg, gf, kind='linear', fill_value="extrapolate")
    factor_grip_fine = f_gf(x_fine)
    # Sector (using step/interpolation with previous value)
    f_sc = interp1d(xs, sc, kind='previous', fill_value="extrapolate")
    sector = f_sc(x_fine)
    
    print(f"Fine meshing completed with mesh size: {mesh_size} [m]")
    
    # === Map Generation ===
    X_coords = np.zeros(n)
    Y_coords = np.zeros(n)
    angle_seg = np.degrees(dx * r_fine)
    angle_head = np.cumsum(angle_seg)
    if str(info.get('config','')).strip().lower() == 'closed':
        dh_options = [angle_head[-1] % (np.sign(angle_head[-1]) * 360),
                      angle_head[-1] - np.sign(angle_head[-1]) * 360]
        dh = dh_options[np.argmin(np.abs(dh_options))]
        angle_head = angle_head - x_fine / L_total * dh
        angle_seg = np.concatenate(([angle_head[0]], np.diff(angle_head)))
    angle_head = angle_head - angle_head[0]
    for i in range(1, n):
        p_prev = np.array([X_coords[i-1], Y_coords[i-1], 0])
        delta = np.array([dx[i-1], 0, 0])
        new_point = p_prev + rotz(angle_head[i-1]).dot(delta)
        X_coords[i] = new_point[0]
        Y_coords[i] = new_point[1]
    
    # === Apexes ===
    peaks, _ = find_peaks(np.abs(r_fine))
    apex = peaks
    r_apex = r_fine[apex]
    print("Apex calculation completed.")
    
    # === Map Edit ===
    if str(info.get('direction','')).strip().lower() == 'backward':
        x_fine = x_fine[-1] - x_fine[::-1]
        r_fine = -r_fine[::-1]
        apex = len(x_fine) - apex[::-1]
        r_apex = -r_apex[::-1]
        incl = -incl[::-1]
        bank_fine = -bank_fine[::-1]
        factor_grip_fine = factor_grip_fine[::-1]
        sector = sector[::-1]
        X_coords = X_coords[::-1]
        Y_coords = Y_coords[::-1]
        Z = Z[::-1]
    
    # Rotate track map
    xyz = np.vstack((X_coords, Y_coords, Z))
    xyz_rot = rotz(rotation_angle).dot(xyz)
    X_coords, Y_coords, Z = xyz_rot[0, :], xyz_rot[1, :], xyz_rot[2, :]
    print("Track rotated.")
    
    if str(info.get('config','')).strip().lower() == 'closed':
        DX = x_fine / L_total * (X_coords[0] - X_coords[-1])
        DY = x_fine / L_total * (Y_coords[0] - Y_coords[-1])
        DZ = x_fine / L_total * (Z[0] - Z[-1])
        db = x_fine / L_total * (bank_fine[0] - bank_fine[-1])
        X_coords = X_coords + DX
        Y_coords = Y_coords + DY
        Z = Z + DZ
        bank_fine = bank_fine + db
        incl = -np.degrees(np.arctan(np.diff(Z) / np.diff(x_fine)))
        incl = np.append(incl, (incl[-1] + incl[0]) / 2)
        print("Fine mesh map closed.")
    
    incl = smooth(incl, 5)
    print("Fine mesh map created.")
    
    # === Plotting Results ===
    # Draw finish-line arrow
    factor_scale = 25
    half_angle = 40
    scale_val = max(np.ptp(X_coords), np.ptp(Y_coords)) / factor_scale
    arrow_n = np.array([X_coords[0]-X_coords[1], Y_coords[0]-Y_coords[1], Z[0]-Z[1]])
    arrow_n = arrow_n / np.linalg.norm(arrow_n)
    arrow_1 = scale_val * rotz(half_angle).dot(arrow_n) + np.array([X_coords[0], Y_coords[0], Z[0]])
    arrow_2 = scale_val * rotz(-half_angle).dot(arrow_n) + np.array([X_coords[0], Y_coords[0], Z[0]])
    arrow_x = [arrow_1[0], X_coords[0], arrow_2[0]]
    arrow_y = [arrow_1[1], Y_coords[0], arrow_2[1]]
    arrow_z = [arrow_1[2], Z[0], arrow_2[2]]
    arrow = np.vstack((arrow_x, arrow_y, arrow_z))
    
    plt.figure(figsize=(9,9))
    plt.suptitle(f"OpenTRACK\nTrack Name: {info['name']}  Configuration: {info['config']}  Mirror: {info['mirror']}\nDate & Time: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
    
    # 3D Map subplot (occupying several grid cells)
    ax1 = plt.subplot(5,2,(1,3,5,7,9), projection='3d')
    ax1.set_title('3D Map')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.scatter(X_coords, Y_coords, Z, c=sector, cmap='viridis', s=20)
    ax1.plot(arrow_x, arrow_y, arrow_z, 'k-', linewidth=2)
    ax1.grid(True)
    ax1.axis('equal')
    
    # Curvature subplot
    plt.subplot(5,2,2)
    plt.title('Curvature')
    plt.xlabel('position [m]')
    plt.ylabel('curvature [1/m]')
    plt.plot(x_fine, r_fine)
    plt.scatter(x_fine[apex], r_apex)
    plt.xlim([x_fine[0], x_fine[-1]])
    plt.legend(['curvature','apex'])
    
    # Elevation subplot
    plt.subplot(5,2,4)
    plt.title('Elevation')
    plt.xlabel('position [m]')
    plt.ylabel('elevation [m]')
    plt.plot(x_fine, Z)
    plt.xlim([x_fine[0], x_fine[-1]])
    
    # Inclination subplot
    plt.subplot(5,2,6)
    plt.title('Inclination')
    plt.xlabel('position [m]')
    plt.ylabel('inclination [deg]')
    plt.plot(x_fine, incl)
    plt.xlim([x_fine[0], x_fine[-1]])
    
    # Banking subplot
    plt.subplot(5,2,8)
    plt.title('Banking')
    plt.xlabel('position [m]')
    plt.ylabel('banking [deg]')
    plt.plot(x_fine, bank_fine)
    plt.xlim([x_fine[0], x_fine[-1]])
    
    # Grip factor subplot
    plt.subplot(5,2,10)
    plt.title('Grip Factor')
    plt.xlabel('position [m]')
    plt.ylabel('grip factor [-]')
    plt.plot(x_fine, factor_grip_fine)
    plt.xlim([x_fine[0], x_fine[-1]])
    
    plt.tight_layout()
    plt.savefig(trackname + ".png")
    plt.show()
    print("Plots created and saved.")
    
    # === Saving Circuit Data ===
    circuit_data = {
        'info': info,
        'x': x_fine,
        'dx': dx,
        'n': n,
        'r': r_fine,
        'bank': bank_fine,
        'incl': incl,
        'factor_grip': factor_grip_fine,
        'sector': sector,
        'r_apex': r_apex,
        'apex': apex,
        'X': X_coords,
        'Y': Y_coords,
        'Z': Z,
        'arrow': arrow
    }
    with open(trackname + ".pkl", "wb") as f:
        pickle.dump(circuit_data, f)
    print("Track generated successfully.")
    
    # === ASCII Map Generation ===
    charh = 15  # font height [pixels]
    charw = 8   # font width [pixels]
    linew = 66  # log file character width
    mapw_val = max(X_coords) - min(X_coords)
    YY = np.round(Y_coords / (charh/charw) / mapw_val * linew).astype(int)
    XX = np.round((X_coords - min(X_coords)) / mapw_val * linew).astype(int)
    YY = -YY - min(-YY)
    XX = XX - min(XX)
    unique_points = np.unique(np.vstack((XX, YY)).T, axis=0)
    XX = unique_points[:, 0] + 1
    YY = unique_points[:, 1] + 1
    maph = YY.max()
    mapw_new = XX.max()
    ascii_map = np.full((maph, mapw_new), ' ', dtype=str)
    for i in range(maph):
        for j in range(mapw_new):
            if any((XX == j) & (YY == i)):
                ascii_map[i, j] = 'o'
    print("Map:")
    for row in ascii_map:
        print("".join(row))
    
if __name__ == '__main__':
    main()