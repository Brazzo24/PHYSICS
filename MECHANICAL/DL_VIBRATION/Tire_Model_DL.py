import numpy as np
import plotly.express as px
import plotly.io as pio

# Set renderer for VS Code Interactive or Jupyter
pio.renderers.default = "vscode"  # Use "notebook" if running in Jupyter

# Simplified Pacejka Magic Formula for longitudinal force
def pacejka_longitudinal_force(kappa, Fz, B=10.0, C=1.9, D_factor=1.0, E=0.97, mu=1.2):
    D = mu * Fz * D_factor
    return D * np.sin(C * np.arctan(B * kappa - E * (B * kappa - np.arctan(B * kappa))))

# Slip ratio range
kappa_values = np.linspace(-0.3, 0.3, 200)  # -30% to +30%

# Different vertical loads
Fz_values = [1500, 2000, 2500]  # N

# Prepare data
plot_data = {
    "Slip Ratio": [],
    "Longitudinal Force (N)": [],
    "Vertical Load (N)": []
}

for Fz in Fz_values:
    Fx = pacejka_longitudinal_force(kappa_values, Fz)
    plot_data["Slip Ratio"].extend(kappa_values)
    plot_data["Longitudinal Force (N)"].extend(Fx)
    plot_data["Vertical Load (N)"].extend([Fz] * len(kappa_values))

# Create interactive plot
fig = px.line(
    plot_data,
    x="Slip Ratio",
    y="Longitudinal Force (N)",
    color="Vertical Load (N)",
    title="Pacejka Longitudinal Force vs Slip Ratio for Different Fz",
    labels={"Slip Ratio": "Slip Ratio (k)", "Longitudinal Force (N)": "Fx (N)"},
    hover_data={"Vertical Load (N)": True}
)

# Show interactive plot inline in VS Code Interactive
fig.show()

# Save interactive HTML for offline viewing
fig.write_html("pacejka_longitudinal_force_vs_slip.html")

# Save JSON and PNG
fig.write_json("pacejka_longitudinal_force_vs_slip.json")  # ✅ No encoding argument
fig.write_image("pacejka_longitudinal_force_vs_slip.png")

print("Interactive plot displayed inline (VS Code Interactive) and files saved.")