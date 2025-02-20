import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import fft, fftfreq

# System parameters
J1 = 0.1  # kg*m^2 (engine side inertia)
J2 = 0.2  # kg*m^2 (wheel side inertia)
k = 5000  # Nm/rad (torsional stiffness of DMF)
c = 50    # Nm.s/rad (torsional damping)
T_avg = 100  # Nm (mean engine torque)
T_amp = 50   # Nm (oscillating component)
rolling_resistance = 20  # Nm (constant resistance at wheel side)

# Simulation time
t_max = 5  # seconds
sampling_rate = 1000  # Hz
t_eval = np.linspace(0, t_max, t_max * sampling_rate)

# Engine torque function
def engine_torque(t):
    return T_avg + T_amp * np.sin(2 * np.pi * 10 * t)  # 10 Hz excitation

# Equations of motion
def driveline_dynamics(t, y):
    theta1, omega1, theta2, omega2 = y
    
    T_engine = engine_torque(t)
    T_wheel = rolling_resistance
    
    # Torsional interaction force
    T_dmf = k * (theta1 - theta2) + c * (omega1 - omega2)
    
    # Equations of motion
    d_omega1 = (T_engine - T_dmf) / J1
    d_omega2 = (T_dmf - T_wheel) / J2
    
    return [omega1, d_omega1, omega2, d_omega2]

# Initial conditions (starting from rest)
y0 = [0, 0, 0, 0]

# Solve ODE
sol = solve_ivp(driveline_dynamics, [0, t_max], y0, t_eval=t_eval, method='RK45')

theta1, omega1, theta2, omega2 = sol.y

# Power computations
P_engine = engine_torque(sol.t) * omega1
P_wheel = rolling_resistance * omega2
P_dmf = (k * (theta1 - theta2) + c * (omega1 - omega2)) * (omega1 - omega2)

# FFT Analysis
N = len(t_eval)
dt = 1 / sampling_rate
freqs = fftfreq(N, dt)

P_engine_fft = np.abs(fft(P_engine))[:N//2]
P_wheel_fft = np.abs(fft(P_wheel))[:N//2]
P_dmf_fft = np.abs(fft(P_dmf))[:N//2]
freqs = freqs[:N//2]

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(sol.t, P_engine, label='Engine Power')
plt.plot(sol.t, P_wheel, label='Wheel Power')
plt.plot(sol.t, P_dmf, label='DMF Power')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Power [W]')
plt.title('Instantaneous Power Flow')

plt.subplot(3, 1, 2)
plt.plot(freqs, P_engine_fft, label='Engine Power Spectrum')
plt.plot(freqs, P_wheel_fft, label='Wheel Power Spectrum')
plt.plot(freqs, P_dmf_fft, label='DMF Power Spectrum')
plt.legend()
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.title('FFT of Power Flow')
plt.xlim(0, 100)  # Focus on low frequencies

plt.tight_layout()
plt.show()
