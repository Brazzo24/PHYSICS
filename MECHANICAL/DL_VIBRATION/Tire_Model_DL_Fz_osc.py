import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go

# Simulation parameters
dt = 0.001
t_end = 5.0
t = np.arange(0, t_end, dt)

# Vertical load oscillation: suspension (2 Hz) + unbalance (20 Hz)
Fz_mean = 2000
Fz_amp_susp = 500
Fz_amp_unbal = 100
Fz = Fz_mean + Fz_amp_susp * np.sin(2 * np.pi * 2 * t) + Fz_amp_unbal * np.sin(2 * np.pi * 20 * t)

# Pacejka simplified parameters
mu = 1.1
C = 1.65
E = -1.6
B_base = 10.0

# Wheel and vehicle parameters
wheel_inertia = 1.2  # kg*m^2
wheel_radius = 0.3   # m
vehicle_speed = 20.0 # m/s (constant for simplicity)
engine_torque = 120.0 # Nm

# Initialize states
wheel_speed = np.zeros_like(t)
slip_ratio = np.zeros_like(t)
Fx = np.zeros_like(t)

# Simulation loop
for i in range(1, len(t)):
    # Compute slip ratio
    v_wheel = wheel_speed[i-1] * wheel_radius
    slip_ratio[i] = (v_wheel - vehicle_speed) / max(vehicle_speed, 0.1)

    # Pacejka longitudinal force
    B = B_base
    kappa = slip_ratio[i]
    Fx[i] = Fz[i] * mu * np.sin(C * np.arctan(B * kappa - E * (B * kappa - np.arctan(B * kappa))))

    # Wheel dynamics
    wheel_acc = (engine_torque / wheel_radius - Fx[i] * wheel_radius) / wheel_inertia
    wheel_speed[i] = wheel_speed[i-1] + wheel_acc * dt

# Create subplots
fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                       subplot_titles=("Vertical Load Fz", "Longitudinal Force Fx", "Slip Ratio", "Wheel Speed"))

fig.add_trace(go.Scatter(x=t, y=Fz, name="Fz", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=Fx, name="Fx", line=dict(color="orange")), row=2, col=1)
fig.add_trace(go.Scatter(x=t, y=slip_ratio, name="Slip Ratio", line=dict(color="green")), row=3, col=1)
fig.add_trace(go.Scatter(x=t, y=wheel_speed, name="Wheel Speed", line=dict(color="purple")), row=4, col=1)

# Update layout
fig.update_layout(height=900, width=900, title_text="Dynamic Fz Oscillation with Tire Unbalance and Wheel Dynamics",
                  showlegend=False)
fig.update_xaxes(title_text="Time [s]", row=4, col=1)
fig.update_yaxes(title_text="Fz [N]", row=1, col=1)
fig.update_yaxes(title_text="Fx [N]", row=2, col=1)
fig.update_yaxes(title_text="Slip Ratio", row=3, col=1)
fig.update_yaxes(title_text="Wheel Speed [rad/s]", row=4, col=1)

fig.show()