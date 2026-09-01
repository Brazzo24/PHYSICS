import numpy as np
import matplotlib.pyplot as plt

#define functions
# define functions to calculate the main parameters

def calc_dF_z_roll(m,h,phi_d):
    dF_z_roll = -m * h * phi_d**2
    
    return dF_z_roll

def calc_lambda_rear(phi_d, theta_d, v, L):
    lambda_rear = (phi_d**2) /((phi_d**2) + (theta_d*v/L)**2)
    return lambda_rear

def calc_dF_z_r(lambda_rear, dF_z_roll):
    dF_z_r = lambda_rear * dF_z_roll
    return dF_z_r

# use functions


#default params
m = 200
theta_d = 0.05 # (rad/s)
h = 0.658 # height of Center of mass (m)
L = 1.4 # wheelbase (m)
m = 229 # mass of bike and rider (kg)
v = 27.78 # bike forward velcoity (m/s)

#sweep params

vals = np.linspace(0, 1, 101)
v_values = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
h_values = [0.2, 0.4, 0.6, 0.8]



fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)

ax1 = fig.add_subplot(1, 3, 1)
ax1.set_xlabel('$\\phi_d$')
ax1.set_ylabel('F_z_roll [N]')
ax1.grid()

ax2 = fig.add_subplot(1, 3, 2)
ax2.set_xlabel('$\\phi_d$')
ax2.set_ylabel('F_z_roll [N]')
ax2.grid()

ax3 = fig.add_subplot(1, 3, 3)
ax3.set_xlabel('$\\phi_d$')
ax3.set_ylabel('F_z_roll [N]')
ax3.grid()

for v in v_values:
    res_F_z_roll = calc_dF_z_roll(m, h, vals)  # ✅ vectorized call
    ax1.plot(vals, res_F_z_roll, label=f'h={h:.2f}')

    res_lambda = calc_lambda_rear(vals, theta_d, v, L)  # ✅ vectorized call # phi_d, theta_d, v, L
    ax2.plot(vals, res_lambda, label=f'v={v:.2f}')
    
    res_F_z_r = calc_dF_z_r(res_F_z_roll, res_lambda)  # ✅ vectorized call # lambda_rear, dF_z_roll
    ax3.plot(vals, res_F_z_r, label=f'v={v:.2f}')

ax1.legend()
ax2.legend()
ax3.legend()

plt.show()