import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# ------------------------------
# Helper functions to read Excel data
# ------------------------------

def read_torque_curve(filename, sheet_name='Torque Curve', start_row=1):
    """
    Reads the torque curve data from an Excel file.
    Assumes the first row of the sheet is a header and skips one row.
    """
    data = pd.read_excel(filename, sheet_name=sheet_name, skiprows=start_row)
    return data

def read_info(filename, sheet_name='Info', start_row=1):
    """
    Reads the vehicle info from an Excel file.
    Assumes the first row of the sheet is a header and skips one row.
    """
    data = pd.read_excel(filename, sheet_name=sheet_name, skiprows=start_row)
    return data

# ------------------------------
# Main code
# ------------------------------

# Vehicle file selection
filename = 'Formula 1.xlsx'

# Read Excel files (adjust sheet names if needed)
info = read_info(filename, sheet_name='Info')
data = read_torque_curve(filename, sheet_name='Torque Curve')

# ------------------------------
# Extracting Variables from the "info" DataFrame
# (Assuming that the Excel file’s “Info” sheet has two columns 
# and that the values are in the second column, starting at row 0)
# ------------------------------

name      = info.iloc[0, 1]
veh_type  = info.iloc[1, 1]
M         = float(info.iloc[2, 1])
df        = float(info.iloc[3, 1]) / 100.0
L         = float(info.iloc[4, 1]) / 1000.0
rack      = float(info.iloc[5, 1])
Cl        = float(info.iloc[6, 1])
Cd        = float(info.iloc[7, 1])
factor_Cl = float(info.iloc[8, 1])
factor_Cd = float(info.iloc[9, 1])
da        = float(info.iloc[10, 1]) / 100.0
A         = float(info.iloc[11, 1])
rho       = float(info.iloc[12, 1])
br_disc_d = float(info.iloc[13, 1]) / 1000.0
br_pad_h  = float(info.iloc[14, 1]) / 1000.0
br_pad_mu = float(info.iloc[15, 1])
br_nop    = float(info.iloc[16, 1])
br_pist_d = float(info.iloc[17, 1]) / 1000.0
br_mast_d = float(info.iloc[18, 1]) / 1000.0
br_ped_r  = float(info.iloc[19, 1])
factor_grip = float(info.iloc[20, 1])
tyre_radius = float(info.iloc[21, 1]) / 1000.0
Cr        = float(info.iloc[22, 1])
mu_x      = float(info.iloc[23, 1])
mu_x_M    = float(info.iloc[24, 1])
sens_x    = float(info.iloc[25, 1])
mu_y      = float(info.iloc[26, 1])
mu_y_M    = float(info.iloc[27, 1])
sens_y    = float(info.iloc[28, 1])
CF        = float(info.iloc[29, 1])
CR        = float(info.iloc[30, 1])
factor_power = float(info.iloc[31, 1])
n_thermal = float(info.iloc[32, 1])
fuel_LHV  = float(info.iloc[33, 1])
drive     = info.iloc[34, 1]
shift_time = float(info.iloc[35, 1])
n_primary = float(info.iloc[36, 1])
n_final   = float(info.iloc[37, 1])
n_gearbox = float(info.iloc[38, 1])
ratio_primary = float(info.iloc[39, 1])
ratio_final   = float(info.iloc[40, 1])
# Ratio gearbox values are stored from row index 41 until the end:
ratio_gearbox = info.iloc[41:, 1].astype(float).to_numpy()
nog = len(ratio_gearbox)

# ------------------------------
# Create output folder and set up logging (console printing in this case)
# ------------------------------

os.makedirs("OpenVEHICLE Vehicles", exist_ok=True)
vehname = f"OpenVEHICLE Vehicles/OpenVEHICLE_{name}_{veh_type}"
if os.path.exists(vehname + ".log"):
    os.remove(vehname + ".log")

# Print HUD information
print("_______                    ___    ________________  ________________________________")
print("__  __ \\_____________________ |  / /__  ____/__  / / /___  _/_  ____/__  /___  ____/")
print("_  / / /__  __ \\  _ \\_  __ \\_ | / /__  __/  __  /_/ / __  / _  /    __  / __  __/   ")
print("/ /_/ /__  /_/ /  __/  / / /_ |/ / _  /___  _  __  / __/ /  / /___  _  /___  /___   ")
print("\\____/ _  .___/\\___//_/ /_/_____/  /_____/  /_/ /_/  /___/  \\____/  /_____/_____/   ")
print("       /_/                                                                          ")
print("="*84)
print(filename)
print("File read successfully")
print("="*84)
print("Name:", name)
print("Type:", veh_type)
print("Date:", datetime.now().strftime('%d/%m/%Y'))
print("Time:", datetime.now().strftime('%H:%M:%S'))
print("="*84)
print("Vehicle generation started.")

# ------------------------------
# Brake Model
# ------------------------------

br_pist_a = br_nop * math.pi * (br_pist_d)**2 / 4
br_mast_a = math.pi * (br_mast_d)**2 / 4
# The MATLAB expression: beta = tyre_radius/(br_disc_d/2 - br_pad_h/2)/br_pist_a/br_pad_mu/4
beta = tyre_radius / ( (br_disc_d/2 - br_pad_h/2) ) / br_pist_a / br_pad_mu / 4
phi = br_mast_a / br_ped_r * 2
print("Braking model generated successfully.")

# ------------------------------
# Steering Model
# ------------------------------

a = (1 - df) * L      # Distance of front axle from center of mass [m]
b = -df * L           # Distance of rear axle from center of mass [m]
C = 2 * np.array([[CF, CF + CR],
                  [CF * a, CF * a + CR * b]])
print("Steering model generated successfully.")

# ------------------------------
# Driveline Model
# ------------------------------

# Fetch engine curves from the torque curve data
en_speed_curve  = data.iloc[:, 0].to_numpy()  # Engine speed [rpm]
en_torque_curve = data.iloc[:, 1].to_numpy()  # Engine torque [N*m]
en_power_curve  = en_torque_curve * en_speed_curve * 2 * math.pi / 60  # Engine power [W]

# Preallocate arrays for each gear
wheel_speed_gear   = np.zeros((len(en_speed_curve), nog))
vehicle_speed_gear = np.zeros((len(en_speed_curve), nog))
wheel_torque_gear  = np.zeros((len(en_torque_curve), nog))

for j in range(nog):
    wheel_speed_gear[:, j] = en_speed_curve / (ratio_primary * ratio_gearbox[j] * ratio_final)
    vehicle_speed_gear[:, j] = wheel_speed_gear[:, j] * 2 * math.pi / 60 * tyre_radius
    wheel_torque_gear[:, j] = en_torque_curve * ratio_primary * ratio_gearbox[j] * ratio_final * n_primary * n_gearbox * n_final

v_min = vehicle_speed_gear.min()
v_max = vehicle_speed_gear.max()
dv = 0.5 / 3.6  # 0.5 km/h converted to m/s

# Create a finely meshed speed vector
vehicle_speed = np.arange(v_min, v_max + dv, dv)

# Preallocate arrays for gear selection and engine tractive force
gear = np.zeros(len(vehicle_speed), dtype=int)
fx_engine = np.zeros(len(vehicle_speed))
fx = np.zeros((len(vehicle_speed), nog))

for i, vs in enumerate(vehicle_speed):
    for j in range(nog):
        # Linear interpolation; out-of-bound values default to 0
        fx[i, j] = np.interp(vs, vehicle_speed_gear[:, j], wheel_torque_gear[:, j] / tyre_radius, left=0, right=0)
    fx_engine[i] = np.max(fx[i, :])
    gear[i] = np.argmax(fx[i, :])

# Add a point for 0 speed for interpolation at low speeds
vehicle_speed = np.insert(vehicle_speed, 0, 0)
gear = np.insert(gear, 0, gear[0])
fx_engine = np.insert(fx_engine, 0, fx_engine[0])

# Compute engine speed, wheel torque, engine torque, and engine power
engine_speed = ratio_final * np.array([ratio_gearbox[g] for g in gear]) * ratio_primary * vehicle_speed / tyre_radius * 60 / (2 * math.pi)
wheel_torque = fx_engine * tyre_radius
engine_torque = wheel_torque / (ratio_final * np.array([ratio_gearbox[g] for g in gear]) * ratio_primary * n_primary * n_gearbox * n_final)
engine_power = engine_torque * engine_speed * 2 * math.pi / 60
print("Driveline model generated successfully.")

# ------------------------------
# Shifting Points and Rev Drops
# ------------------------------

# Find indices where the gear changes (using diff)
gear_change_indices = np.where(np.diff(gear, prepend=gear[0]) != 0)[0]
engine_speed_gear_change = engine_speed[gear_change_indices]

# Separate shift points and arrival points (assuming alternating order)
shift_points  = engine_speed_gear_change[::2]
arrive_points = engine_speed_gear_change[1::2]
rev_drops = shift_points - arrive_points

# Create a shifting table as a Pandas DataFrame
shift_rows = [f"{i}-{i+1}" for i in range(1, len(shift_points)+1)]
shifting = pd.DataFrame({
    'shift_points': shift_points,
    'arrive_points': arrive_points,
    'rev_drops': rev_drops
}, index=shift_rows)
print("Shift points calculated successfully.")

# ------------------------------
# Force Model
# ------------------------------

g_val = 9.81

if drive.strip().upper() == 'RWD':
    factor_drive = (1 - df)
    factor_aero  = (1 - da)
    driven_wheels = 2
elif drive.strip().upper() == 'FWD':
    factor_drive = df
    factor_aero  = da
    driven_wheels = 2
else:  # AWD
    factor_drive = 1
    factor_aero  = 1
    driven_wheels = 4

fz_mass  = -M * g_val
fz_aero  = 0.5 * rho * factor_Cl * Cl * A * vehicle_speed**2
fz_total = fz_mass + fz_aero
fz_tyre  = (factor_drive * fz_mass + factor_aero * fz_aero) / driven_wheels

fx_aero = 0.5 * rho * factor_Cd * Cd * A * vehicle_speed**2
fx_roll = Cr * np.abs(fz_total)
fx_tyre = driven_wheels * (mu_x + sens_x * (mu_x_M * g_val - np.abs(fz_tyre))) * np.abs(fz_tyre)
print("Forces calculated successfully.")

# ------------------------------
# GGV Map
# ------------------------------

bank = 0
incl = 0
dmy = factor_grip * sens_y
muy_val = factor_grip * mu_y
Ny = mu_y_M * g_val
dmx = factor_grip * sens_x
mux_val = factor_grip * mu_x
Nx = mu_x_M * g_val
Wz = M * g_val * np.cos(np.radians(bank)) * np.cos(np.radians(incl))
Wy = -M * g_val * np.sin(np.radians(bank))
Wx = M * g_val * np.sin(np.radians(incl))
dv_map = 2
v_vec = np.arange(0, v_max + dv_map, dv_map)
if v_vec[-1] != v_max:
    v_vec = np.append(v_vec, v_max)
N_pts = 45
GGV = np.zeros((len(v_vec), 2 * N_pts - 1, 3))

for i, vi in enumerate(v_vec):
    Aero_Df = 0.5 * rho * factor_Cl * Cl * A * vi**2
    Aero_Dr = 0.5 * rho * factor_Cd * Cd * A * vi**2
    Roll_Dr = Cr * np.abs(-Aero_Df + Wz)
    Wd = (factor_drive * Wz - factor_aero * Aero_Df) / driven_wheels
    ax_drag = (Aero_Dr + Roll_Dr + Wx) / M
    ay_max = (1 / M) * (muy_val + dmy * (Ny - (Wz - Aero_Df) / 4)) * (Wz - Aero_Df)
    ax_tyre_max_acc = (1 / M) * (mux_val + dmx * (Nx - Wd)) * Wd * driven_wheels
    ax_tyre_max_dec = - (1 / M) * (mux_val + dmx * (Nx - (Wz - Aero_Df) / 4)) * (Wz - Aero_Df)
    ax_power_limit = (1 / M) * np.interp(vi, vehicle_speed, factor_power * fx_engine)
    ax_power_limit_arr = np.ones(N_pts) * ax_power_limit
    angles = np.linspace(0, 180, N_pts)
    ay = ay_max * np.cos(np.radians(angles))
    ax_tyre_acc = ax_tyre_max_acc * np.sqrt(1 - (ay / ay_max)**2)
    ax_acc = np.minimum(ax_tyre_acc, ax_power_limit_arr) + ax_drag
    ax_dec = ax_tyre_max_dec * np.sqrt(1 - (ay / ay_max)**2) + ax_drag
    # Concatenate arrays: ax_acc (length N_pts) and ax_dec (excluding the first element, length N_pts-1)
    arr_ax = np.concatenate((ax_acc, ax_dec[1:]))
    # Similarly for lateral acceleration: ay and reversed ay (excluding the first element)
    arr_ay = np.concatenate((ay, ay[1:][::-1]))
    GGV[i, :, 0] = arr_ax
    GGV[i, :, 1] = arr_ay
    GGV[i, :, 2] = vi
print("GGV map generated successfully.")

# ------------------------------
# Save the Vehicle Data
# ------------------------------

vehicle_data = {
    'name': name,
    'type': veh_type,
    'M': M,
    'df': df,
    'L': L,
    'rack': rack,
    'Cl': Cl,
    'Cd': Cd,
    'factor_Cl': factor_Cl,
    'factor_Cd': factor_Cd,
    'da': da,
    'A': A,
    'rho': rho,
    'br_disc_d': br_disc_d,
    'br_pad_h': br_pad_h,
    'br_pad_mu': br_pad_mu,
    'br_nop': br_nop,
    'br_pist_d': br_pist_d,
    'br_mast_d': br_mast_d,
    'br_ped_r': br_ped_r,
    'factor_grip': factor_grip,
    'tyre_radius': tyre_radius,
    'Cr': Cr,
    'mu_x': mu_x,
    'mu_x_M': mu_x_M,
    'sens_x': sens_x,
    'mu_y': mu_y,
    'mu_y_M': mu_y_M,
    'sens_y': sens_y,
    'CF': CF,
    'CR': CR,
    'factor_power': factor_power,
    'n_thermal': n_thermal,
    'fuel_LHV': fuel_LHV,
    'drive': drive,
    'shift_time': shift_time,
    'n_primary': n_primary,
    'n_final': n_final,
    'n_gearbox': n_gearbox,
    'ratio_primary': ratio_primary,
    'ratio_final': ratio_final,
    'ratio_gearbox': ratio_gearbox,
    'wheel_speed_gear': wheel_speed_gear,
    'vehicle_speed_gear': vehicle_speed_gear,
    'wheel_torque_gear': wheel_torque_gear,
    'v_min': v_min,
    'v_max': v_max,
    'vehicle_speed': vehicle_speed,
    'gear': gear,
    'fx_engine': fx_engine,
    'engine_speed': engine_speed,
    'wheel_torque': wheel_torque,
    'engine_torque': engine_torque,
    'engine_power': engine_power,
    'shifting': shifting,
    'fz_total': fz_total,
    'fz_tyre': fz_tyre,
    'fx_aero': fx_aero,
    'fx_roll': fx_roll,
    'fx_tyre': fx_tyre,
    'GGV': GGV
}

with open(vehname + ".pkl", "wb") as f:
    pickle.dump(vehicle_data, f)
print("Vehicle generated successfully and saved.")

# ------------------------------
# Plotting
# ------------------------------

fig = plt.figure(constrained_layout=True, figsize=(14, 12))
gs = GridSpec(4, 2, figure=fig)
fig.suptitle(name, fontsize=16)

# Engine Curve Plot (subplot 1,1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Engine Curve")
ax1.set_xlabel("Engine Speed [rpm]")
ax1.set_ylabel("Engine Torque [Nm]")
ax1.plot(en_speed_curve, factor_power * en_torque_curve, color='blue', label='Torque')
ax1.grid(True)
# Secondary y-axis for engine power in horsepower
ax1b = ax1.twinx()
ax1b.set_ylabel("Engine Power [Hp]")
ax1b.plot(en_speed_curve, factor_power * en_power_curve / 745.7, color='orange', label='Power')
ax1.legend(loc='upper left')
ax1b.legend(loc='upper right')

# Gearing Plot (subplot 2,1)
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title("Gearing")
ax2.set_xlabel("Speed [m/s]")
ax2.set_ylabel("Engine Speed [rpm]")
ax2.plot(vehicle_speed, engine_speed, color='green', label='Engine Speed')
ax2.grid(True)
ax2b = ax2.twinx()
ax2b.set_ylabel("Gear")
ax2b.plot(vehicle_speed, gear, color='red', label='Gear')
ax2.legend(loc='upper left')
ax2b.legend(loc='upper right')

# Traction Model Plot (subplot spanning rows 3-4, col 1)
ax3 = fig.add_subplot(gs[2:4, 0])
ax3.set_title("Traction Model")
ax3.plot(vehicle_speed, factor_power * fx_engine, 'k-', linewidth=4, label='Engine tractive force')
ax3.plot(vehicle_speed, np.minimum(factor_power * fx_engine, fx_tyre), 'r-', linewidth=2, label='Final tractive force')
ax3.plot(vehicle_speed, -fx_aero, label='Aero drag')
ax3.plot(vehicle_speed, -fx_roll, label='Rolling resistance')
ax3.plot(vehicle_speed, fx_tyre, label='Max tyre tractive force')
# Plot engine tractive force per gear (skipping the 0-speed point for clarity)
for j in range(nog):
    ax3.plot(vehicle_speed[1:], fx[1:, j], 'k--', label=f'Gear {j+1}' if j == 0 else None)
ax3.set_xlabel("Speed [m/s]")
ax3.set_ylabel("Force [N]")
ax3.grid(True)
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)

# GGV Map Plot (subplot spanning all rows, col 2)
ax4 = fig.add_subplot(gs[:, 1], projection='3d')
ax4.set_title("GGV Map")
# In MATLAB: surf(GGV(:,:,2), GGV(:,:,1), GGV(:,:,3))
# Here, X: lateral acceleration, Y: long acceleration, Z: speed.
X_ggv = GGV[:, :, 1]
Y_ggv = GGV[:, :, 0]
Z_ggv = GGV[:, :, 2]
surf = ax4.plot_surface(X_ggv, Y_ggv, Z_ggv, cmap='viridis')
ax4.set_xlabel("Lat acc [m/s²]")
ax4.set_ylabel("Long acc [m/s²]")
ax4.set_zlabel("Speed [m/s]")
ax4.view_init(105, 5)

plt.tight_layout()
plt.savefig(vehname + ".png")
plt.show()
print("Plots created and saved.")