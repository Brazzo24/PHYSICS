import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from scipy.linalg import eigh


###############################################################################
# USER INPUT SECTION
###############################################################################
m = np.array([1.0, 1.0, 2.0])  # Masses [kg]
c = np.array([25.0, 2.0, 3.0])  # Damping [Ns/m]
k = np.array([2000.0, 12000.0, 6000.0])  # Stiffness [N/m]

f_min = 0.1  # Hz
f_max = 35.0  # Hz
num_points = 1000
f_vals = np.linspace(f_min, f_max, num_points)
w_vals = 2 * np.pi * f_vals


###############################################################################
# COMPUTATION FUNCTIONS
###############################################################################

def free_vibration_analysis(m, k):
    """Computes natural frequencies and mode shapes."""
    N = len(m)
    M_free = np.diag(m)
    K_free = np.zeros((N, N))

    if N == 1:
        K_free[0, 0] = k[0]
    else:
        K_free[0, 0] = k[0] + k[1]
        K_free[0, 1] = -k[1]
        K_free[1, 0] = -k[1]
        for i in range(1, N - 1):
            K_free[i, i] = k[i] + k[i+1]
            K_free[i, i+1] = -k[i+1]
            K_free[i+1, i] = -k[i+1]
        K_free[N-1, N-1] = k[N-1]

    eigvals, eigvecs = eigh(K_free, M_free)
    omega_n = np.sqrt(eigvals)   # rad/s
    f_n = omega_n / (2*np.pi)    # Hz
    return f_n, eigvecs


def modal_energy_analysis(m, k, f_n, eigvecs):
    """Computes kinetic & potential energy distributions per mode."""
    N = len(m)
    modal_energies = []

    for i in range(eigvecs.shape[1]):
        phi = eigvecs[:, i]
        norm_factor = np.sqrt(np.real(np.conjugate(phi).T @ np.diag(m) @ phi))
        phi_norm = phi / norm_factor

        omega_i = 2 * np.pi * f_n[i]  # rad/s
        T_dof = 0.5 * m * (omega_i * np.abs(phi_norm))**2
        V_springs = np.zeros(N)
        V_springs[0] = 0.5 * k[0] * (np.abs(phi_norm[0]))**2
        for s in range(1, N):
            V_springs[s] = 0.5 * k[s] * (np.abs(phi_norm[s] - phi_norm[s-1]))**2

        modal_energies.append({
            'mode': i + 1,
            'freq_hz': f_n[i],
            'T_dof': T_dof,
            'V_springs': V_springs
        })
    return modal_energies


# Compute modal analysis results
f_n, eigvecs = free_vibration_analysis(m, k)
modal_energies = modal_energy_analysis(m, k, f_n, eigvecs)


###############################################################################
# DASH APP SETUP
###############################################################################

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("N-DOF System Dashboard", style={'text-align': 'center'}),

    dcc.Tabs([
        dcc.Tab(label='Kinetic & Potential Energy', children=[
            html.Label("Select Mode:"),
            dcc.Dropdown(
                id='mode-selector',
                options=[{'label': f'Mode {i+1}', 'value': i} for i in range(len(modal_energies))],
                value=0
            ),
            dcc.Graph(id='energy-plot')
        ]),
    ])
])


###############################################################################
# CALLBACKS
###############################################################################

@app.callback(
    Output('energy-plot', 'figure'),
    [Input('mode-selector', 'value')]
)
def update_energy_plot(mode_index):
    """Updates energy plots dynamically based on user selection."""
    mode_data = modal_energies[mode_index]
    dof_indices = np.arange(len(m)) + 1

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dof_indices, y=mode_data['T_dof'], name='Kinetic Energy', marker_color='blue'))
    fig.add_trace(go.Bar(x=dof_indices, y=mode_data['V_springs'], name='Potential Energy', marker_color='red'))

    fig.update_layout(
        title=f"Kinetic vs. Potential Energy (Mode {mode_data['mode']} - {mode_data['freq_hz']:.2f} Hz)",
        xaxis_title="DOF Index",
        yaxis_title="Energy [J]",
        barmode='group'
    )
    return fig


###############################################################################
# RUN APP
###############################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
