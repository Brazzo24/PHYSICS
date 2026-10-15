"""Plotly figures for the Streamlit app (also usable from notebooks)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .model import Engine
from .solver import Results

# categorical palette in fixed order (validated reference palette)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_COLOR = {
    "Engine (unbalanced)": C[0], "Balancers (sum)": C[1], "Residual": C[2],
    "Rotating masses": C[3], "Reciprocating 1st": C[4], "Reciprocating 2nd": C[5],
}
ELEM_COLOR = {"crank": "#7a7974", "pin": "#d63a3a", "cyl": "#9b9a95",
              "cw": C[0], (1, 1): C[2], (1, -1): C[2], (2, 1): C[4], (2, -1): C[4]}
QTY_LABEL = {"Fx": "F<sub>x</sub> [N]", "Fz": "F<sub>z</sub> [N]", "Mx": "M<sub>x</sub> [N m]",
             "Mz": "M<sub>z</sub> [N m]", "My": "M<sub>y</sub> (roll, from shaft offsets) [N m]"}


def _rot(angle_rad, radius):
    return radius * np.sin(angle_rad), radius * np.cos(angle_rad)  # (x, z)


# --------------------------------------------------------------------------- 3-D geometry
def _throws(eng: Engine, gap_mm: float = 30.0):
    """Group cylinders into crank throws: same pin angle and axially adjacent."""
    cyl = sorted(eng.cylinders, key=lambda c: c.y_mm)
    groups: list[list] = []
    for c in cyl:
        if groups and abs(((groups[-1][-1].pin_deg - c.pin_deg) + 180) % 360 - 180) < 1e-6 \
                and c.y_mm - groups[-1][-1].y_mm < gap_mm:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def geometry_3d(eng: Engine, res: Results, theta_deg: float = 0.0) -> go.Figure:
    th = np.radians(theta_deg)
    r, l = eng.r_mm, eng.l_mm
    fig = go.Figure()

    def line(xs, ys, zs, color, width=6, name=None, show=False, dash=None, group=None):
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width, dash=dash),
                                   name=name, showlegend=show, legendgroup=group or name, hoverinfo="skip"))

    ys_all = [c.y_mm for c in eng.cylinders] + list(eng.crank_cw.planes_y_mm)
    y0, y1 = min(ys_all) - 25, max(ys_all) + 25
    line([0, 0], [y0, y1], [0, 0], ELEM_COLOR["crank"], 4, "Crankshaft", True, "dash", "crank")

    # crank throws
    w = 7.0
    for gi, g in enumerate(_throws(eng)):
        phi = th + np.radians(g[0].pin_deg)
        px, pz = _rot(phi, r)
        ya, yb = g[0].y_mm - w, g[-1].y_mm + w
        line([0, px, px, 0], [ya, ya, yb, yb], [0, pz, pz, 0], ELEM_COLOR["pin"], 8, "Crank throws",
             gi == 0, group="throws")

    # cylinders: conrod + piston + bore axis
    for i, c in enumerate(eng.cylinders):
        beta = np.radians(c.bank_deg)
        phi = th + np.radians(c.pin_deg)
        ti = phi - beta
        s = r * np.cos(ti) + np.sqrt(l**2 - (r * np.sin(ti))**2)
        ex, ez = np.sin(beta), np.cos(beta)
        px, pz = _rot(phi, r)
        line([px, s * ex], [c.y_mm] * 2, [pz, s * ez], ELEM_COLOR["cyl"], 5, "Conrods", i == 0, group="rods")
        smax = r + l + 25
        line([(l - r - 10) * ex, smax * ex], [c.y_mm] * 2, [(l - r - 10) * ez, smax * ez], ELEM_COLOR["cyl"], 2,
             "Bore axes", i == 0, "dot", "bores")
        fig.add_trace(go.Scatter3d(x=[s * ex], y=[c.y_mm], z=[s * ez], mode="markers+text", text=[c.name],
                                   textposition="top center", marker=dict(size=9, color=ELEM_COLOR["cyl"],
                                                                          symbol="square"),
                                   showlegend=False, hovertext=f"{c.name}: bank {c.bank_deg:g}°, pin {c.pin_deg:g}°",
                                   hoverinfo="text"))

    # balancers
    mmax = max([p.mass_g for b in res.balancers for p in b.planes] + [1e-9])
    for b in res.balancers:
        sp = b.spec
        is_crank = sp is eng.crank_cw
        col = ELEM_COLOR["cw"] if is_crank else ELEM_COLOR[(sp.order, sp.sense)]
        cx, cz = sp.x_mm, sp.z_mm
        if not is_crank:
            yy = [min(sp.planes_y_mm) - 15, max(sp.planes_y_mm) + 15]
            line([cx, cx], yy, [cz, cz], col, 4, sp.name, True, group=sp.name)
        for k, p in enumerate(b.planes):
            ang = np.radians(p.angle_deg) + sp.sense * sp.order * th
            dx, dz = _rot(ang, sp.r_c_mm)
            line([cx, cx + dx], [p.y_mm] * 2, [cz, cz + dz], col, 6, sp.name if is_crank else None,
                 is_crank and k == 0, group=sp.name)
            size = 6 + 14 * np.sqrt(p.mass_g / mmax) if mmax > 0 else 6
            fig.add_trace(go.Scatter3d(x=[cx + dx], y=[p.y_mm], z=[cz + dz], mode="markers",
                                       marker=dict(size=size, color=col), legendgroup=sp.name, showlegend=False,
                                       hovertext=f"{sp.name}<br>y = {p.y_mm:.1f} mm<br>m = {p.mass_g:.1f} g "
                                                 f"@ r<sub>c</sub> = {p.r_c_mm:g} mm<br>angle(θ=0) = {p.angle_deg:.1f}°",
                                       hoverinfo="text"))

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y – crank axis [mm]", zaxis_title="z [mm]",
                   camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9), up=dict(x=0, y=0, z=1))),
        margin=dict(l=0, r=0, t=30, b=0), height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        title=dict(text=f"θ = {theta_deg:.0f}°", x=0.99, xanchor="right", font=dict(size=13)),
    )
    return fig


# --------------------------------------------------------------------------- end view
def end_view(eng: Engine, res: Results, theta_deg: float = 0.0) -> go.Figure:
    """View along the crank axis (x horizontal, z up), like the paper's polar plots."""
    th = np.radians(theta_deg)
    r, l = eng.r_mm, eng.l_mm
    fig = go.Figure()
    t = np.linspace(0, 2 * np.pi, 181)
    fig.add_trace(go.Scatter(x=r * np.sin(t), y=r * np.cos(t), mode="lines", line=dict(color="#b5b4ae", dash="dot",
                             width=1), hoverinfo="skip", showlegend=False))
    seen = set()
    for c in eng.cylinders:
        beta = np.radians(c.bank_deg)
        ex, ez = np.sin(beta), np.cos(beta)
        if c.bank_deg not in seen:
            seen.add(c.bank_deg)
            fig.add_trace(go.Scatter(x=[0, (r + l + 20) * ex], y=[0, (r + l + 20) * ez], mode="lines",
                                     line=dict(color=ELEM_COLOR["cyl"], dash="dot", width=1), showlegend=False,
                                     hoverinfo="skip"))
        phi = th + np.radians(c.pin_deg)
        px, pz = _rot(phi, r)
        ti = phi - beta
        s = r * np.cos(ti) + np.sqrt(l**2 - (r * np.sin(ti))**2)
        fig.add_trace(go.Scatter(x=[0, px, s * ex], y=[0, pz, s * ez], mode="lines+markers",
                                 line=dict(color=ELEM_COLOR["pin"], width=2), marker=dict(size=8),
                                 showlegend=False, hovertext=c.name, hoverinfo="text"))
        fig.add_annotation(x=px, y=pz, text=c.name, showarrow=False, yshift=12, font=dict(size=11))
    crank = next((b for b in res.balancers if b.spec is eng.crank_cw), None)
    if crank:
        for p in crank.planes:
            ang = np.radians(p.angle_deg) + th
            dx, dz = _rot(ang, p.r_c_mm)
            fig.add_trace(go.Scatter(x=[0, dx], y=[0, dz], mode="lines+markers", line=dict(color=ELEM_COLOR["cw"],
                                     width=3), marker=dict(size=10), showlegend=False,
                                     hovertext=f"CW y={p.y_mm:.0f} mm: {p.mass_g:.0f} g, {p.angle_deg:.1f}°",
                                     hoverinfo="text"))
    lim = r + l + 30
    fig.update_layout(xaxis=dict(range=[-lim, lim], scaleanchor="y", title="x [mm]", zeroline=False),
                      yaxis=dict(range=[-r - 50, lim], title="z [mm]", zeroline=False),
                      height=560, margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text="End view (looking along −y)", font=dict(size=13)))
    return fig


# --------------------------------------------------------------------------- time series
def time_plot(ts: dict, theta: np.ndarray, qty: str, series: list[str], rpm: float) -> go.Figure:
    fig = go.Figure()
    extra = [C[6], C[7], C[5], C[4], C[3]]
    for i, name in enumerate(series):
        if name not in ts or qty not in ts[name]:
            continue
        col = SERIES_COLOR.get(name, extra[i % len(extra)])
        fig.add_trace(go.Scatter(x=theta, y=ts[name][qty], mode="lines", name=name, line=dict(color=col, width=2, dash="dash" if name == "Residual" else None),
                                 hovertemplate="%{y:.1f}"))
    fig.update_layout(xaxis=dict(title="crank angle θ [deg]", dtick=90, showspikes=True, spikemode="across",
                                 spikethickness=1, spikedash="dot"),
                      yaxis=dict(title=QTY_LABEL[qty]), hovermode="x unified", height=380,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      title=dict(text=f"@ {rpm:.0f} rpm", x=0.99, xanchor="right", font=dict(size=12)))
    return fig


def locus_plot(ts: dict, kind: str = "F") -> go.Figure:
    """Path of the resultant force (Fx, Fz) or moment (Mz, Mx) vector over one revolution set."""
    fig = go.Figure()
    for name in ("Engine (unbalanced)", "Residual"):
        a, b = (ts[name]["Fx"], ts[name]["Fz"]) if kind == "F" else (ts[name]["Mz"], ts[name]["Mx"])
        fig.add_trace(go.Scatter(x=a, y=b, mode="lines", name=name, line=dict(
            color=SERIES_COLOR[name], width=2, dash="dash" if name == "Residual" else None)))
    u = "N" if kind == "F" else "N m"
    xl, yl = ("F<sub>x</sub>", "F<sub>z</sub>") if kind == "F" else ("M<sub>z</sub>", "M<sub>x</sub>")
    fig.update_layout(xaxis=dict(title=f"{xl} [{u}]", scaleanchor="y"), yaxis=dict(title=f"{yl} [{u}]"),
                      height=380, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      title=dict(text="Force vector locus" if kind == "F" else "Moment vector locus", x=0.99,
                                 xanchor="right", font=dict(size=12)))
    return fig


def rpm_sweep_plot(eng: Engine, res: Results, rpm_max: float) -> go.Figure:
    """Peak residual force / moment vs rpm (they scale with ω²)."""
    from .solver import timeseries
    th = np.linspace(0, 360, 361)
    ts = timeseries(res, th)
    w0 = eng.omega**2
    rpm = np.linspace(0, rpm_max, 101)
    w = (rpm * 2 * np.pi / 60) ** 2
    fig = go.Figure()
    for name in ("Engine (unbalanced)", "Residual"):
        F = np.hypot(ts[name]["Fx"], ts[name]["Fz"]).max() / w0
        fig.add_trace(go.Scatter(x=rpm, y=F * w, name=f"{name} – peak |F|", line=dict(color=SERIES_COLOR[name],
                                 width=2)))
    fig.update_layout(xaxis_title="engine speed [rpm]", yaxis_title="peak |F| [N]", height=320,
                      margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig
