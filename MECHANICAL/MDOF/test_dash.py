import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# --- Compute System Response ---
def compute_NDOF_response_base_excitation(m, c, k, w):
    """
    Compute displacement response X for an NDOF chain system
    under base excitation with unit base velocity for a SINGLE frequency w.
    """
    N = len(m)
    X_base = 1 / (1j * w)  # Base displacement from unit velocity
    
    # Dynamic stiffness matrix A
    A = np.zeros((N, N), dtype=complex)
    if N == 1:
        A[0, 0] = k[0] - w**2 * m[0] + 1j * w * c[0]
    else:
        A[0, 0] = k[0] - w**2 * m[0] + 1j * w * c[0] + k[1] + 1j * w * c[1]
        A[0, 1] = -(k[1] + 1j * w * c[1])
        for i in range(1, N - 1):
            A[i, i-1] = -(k[i] + 1j * w * c[i])
            A[i, i] = k[i] - w**2 * m[i] + 1j * w * c[i] + k[i+1] + 1j * w * c[i+1]
            A[i, i+1] = -(k[i+1] + 1j * w * c[i+1])
        A[N-1, N-2] = -(k[N-1] + 1j * w * c[N-1])
        A[N-1, N-1] = k[N-1] - w**2 * m[N-1] + 1j * w * c[N-1]

    # Forcing vector from base motion
    F = np.zeros(N, dtype=complex)
    F[0] = -(k[0] + 1j * w * c[0]) * X_base  # Force applied at mass 1

    # Solve for displacements
    X = np.linalg.solve(A, F)
    return X


# --- System Setup ---
m = np.array([1.0, 1.0, 2.0])
c = np.array([25.0, 2.0, 3.0])
k = np.array([2000.0, 12000.0, 6000.0])

f_vals = np.linspace(0.1, 35.0, 1000)
w_vals = 2 * np.pi * f_vals

# --- Compute System Response Over Frequency Range ---
X_vals = np.zeros((len(m), len(w_vals)), dtype=complex)  # Store displacement results
A_vals = np.zeros((len(m), len(w_vals)), dtype=complex)  # Store acceleration results

for i, w in enumerate(w_vals):
    X_vals[:, i] = compute_NDOF_response_base_excitation(m, c, k, w)  # Compute for each w

A_vals = -(w_vals**2) * X_vals  # Compute acceleration response



# --- Initialize Dash App ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("N-DOF System Dashboard", style={'text-align': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Acceleration Response', children=[
            dcc.Graph(id='acceleration-plot')
        ]),
        dcc.Tab(label='Transmissibility', children=[
            dcc.Graph(id='transmissibility-plot')
        ]),
        dcc.Tab(label='Power Dissipation', children=[
            dcc.Graph(id='power-dissipation-plot')
        ])
    ]),
    
    dcc.Slider(
        id='mass-selector',
        min=1,
        max=len(m),
        step=1,
        value=1,
        marks={i: f'Mass {i}' for i in range(1, len(m)+1)}
    )
])

@app.callback(
    Output('acceleration-plot', 'figure'),
    [Input('mass-selector', 'value')]
)
def update_acceleration_plot(mass_index):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f_vals, y=np.abs(A_vals[mass_index-1, :]), mode='lines', name=f'Mass {mass_index}'))
    fig.update_layout(title="Acceleration Response", xaxis_title="Frequency (Hz)", yaxis_title="Acceleration |A(ω)|")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
