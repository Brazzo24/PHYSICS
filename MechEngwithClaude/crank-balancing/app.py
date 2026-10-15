"""Crankshaft mass-balancing app (Streamlit).

    streamlit run app.py
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

from crankbal import PRESETS, Engine, harmonic_table, solve, tdc_angles, timeseries
from crankbal.model import Balancer, Cylinder, default_shafts
from crankbal.plots import end_view, geometry_3d, locus_plot, rpm_sweep_plot, time_plot

st.set_page_config(page_title="Crank Balancer", page_icon="⚙️", layout="wide")

# --------------------------------------------------------------------------- state
DEFAULT_PRESET = next(iter(PRESETS))
if "engine" not in st.session_state:
    st.session_state.engine = PRESETS[DEFAULT_PRESET]().to_dict()
    st.session_state.v = 0  # widget generation: bumped whenever a new engine is loaded


def load_engine(e: Engine):
    st.session_state.engine = e.to_dict()
    st.session_state.v += 1


def k(name: str) -> str:
    return f"{name}__{st.session_state.v}"


E = st.session_state.engine

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Crank Balancer")
    preset = st.selectbox("Preset", list(PRESETS), index=0)
    if st.button("Load preset", width="stretch"):
        load_engine(PRESETS[preset]())
        st.rerun()

    up = st.file_uploader("Load engine (.json)", type="json")
    if up is not None and st.session_state.get("_last_upload") != up.file_id:
        st.session_state._last_upload = up.file_id
        load_engine(Engine.from_dict(json.load(up)))
        st.rerun()

    st.divider()
    rpm = st.number_input("Engine speed [rpm]", 0.0, 30000.0, float(E["rpm"]), 250.0, key=k("rpm"))
    rpm_max = st.number_input("Max rpm for sweep plot", 1000.0, 30000.0, max(float(E["rpm"]) * 1.2, 1000.0),
                              500.0, key=k("rpm_max"))

tab_eng, tab_bal, tab_res, tab_th = st.tabs(["1 · Engine", "2 · Balancing elements", "3 · Results", "Theory"])

# --------------------------------------------------------------------------- tab 1: engine
with tab_eng:
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    name = c1.text_input("Engine name", E["name"], key=k("name"))
    r_mm = c2.number_input("Crank radius r [mm]", 1.0, 200.0, float(E["r_mm"]), 0.5, key=k("r"))
    l_mm = c3.number_input("Conrod length l [mm]", 1.0, 500.0, float(E["l_mm"]), 0.5, key=k("l"))
    c4.metric("Λ = r / l", f"{r_mm / l_mm:.4f}")

    st.subheader("Cylinders")
    st.caption("Angles from +z towards +x, looking along the crank axis y. **bank** = cylinder axis direction, "
               "**pin** = crankpin angle at θ = 0. Split pins: give each cylinder its own pin angle. "
               "m_throw = this cylinder's share of the crank throw's own unbalance (pin + webs) reduced to r.")
    df_cyl = pd.DataFrame(E["cylinders"])
    df_cyl = st.data_editor(
        df_cyl, key=k("cyl"), num_rows="dynamic", width="stretch",
        column_config={
            "name": st.column_config.TextColumn("name"),
            "bank_deg": st.column_config.NumberColumn("bank β [°]", format="%.2f"),
            "pin_deg": st.column_config.NumberColumn("pin ψ [°]", format="%.2f"),
            "y_mm": st.column_config.NumberColumn("y [mm]", format="%.2f"),
            "m_osc_g": st.column_config.NumberColumn("m_osc [g]", format="%.1f"),
            "m_rot_g": st.column_config.NumberColumn("m_rot [g]", format="%.1f"),
            "m_throw_g": st.column_config.NumberColumn("m_throw [g]", format="%.1f"),
        })
    df_cyl = df_cyl.dropna(subset=["bank_deg", "pin_deg", "y_mm"]).fillna(
        {"m_osc_g": 0.0, "m_rot_g": 0.0, "m_throw_g": 0.0, "name": "?"})
    cylinders = [Cylinder(**{kk: (str(v) if kk == "name" else float(v)) for kk, v in row.items()})
                 for row in df_cyl.to_dict("records")]

    with st.expander("Mass helper – fill all cylinders from component masses"):
        h1, h2, h3, h4 = st.columns(4)
        m_pis = h1.number_input("Piston + rings + clips [g]", 0.0, 5000.0, 220.0, 1.0)
        m_wp = h1.number_input("Wrist pin [g]", 0.0, 2000.0, 60.0, 1.0)
        m_rod = h2.number_input("Conrod total [g]", 0.0, 5000.0, 380.0, 1.0)
        rod_cg = h2.number_input("Conrod CG from big-end centre [mm]", 0.0, 500.0, 0.3 * l_mm, 0.5,
                                 help="2-mass split: m_small = m·a/l, m_big = m·(1−a/l)")
        m_cp = h3.number_input("Crank pin (per throw) [g]", 0.0, 10000.0, 250.0, 1.0)
        m_web = h3.number_input("Web unbalance mass, per web [g]", 0.0, 10000.0, 400.0, 1.0,
                                help="Mass of one web WITHOUT counterweight, CG at r_web")
        r_web = h4.number_input("Web CG radius r_web [mm]", 0.0, 200.0, 12.0, 0.5)
        rods_per_throw = h4.number_input("Cylinders per throw", 1, 4, 2)
        a = rod_cg / l_mm
        m_osc = m_pis + m_wp + m_rod * a
        m_rot = m_rod * (1 - a)
        m_thr = (m_cp + 2 * m_web * r_web / r_mm) / rods_per_throw
        st.caption(f"→ m_osc = **{m_osc:.1f} g**, m_rot = **{m_rot:.1f} g**, "
                   f"m_throw per cylinder = **{m_thr:.1f} g** (reduced to r = {r_mm:g} mm)")
        if st.button("Apply to all cylinders"):
            st.session_state._apply_masses = (round(m_osc, 2), round(m_rot, 2), round(m_thr, 2))



# --------------------------------------------------------------------------- tab 2: balancers
def parse_planes(txt: str) -> list[float]:
    out = []
    for t in txt.replace(";", ",").split(","):
        t = t.strip()
        if t:
            try:
                out.append(float(t))
            except ValueError:
                pass
    return out


def balancer_inputs(b: dict, key: str, crank: bool = False) -> Balancer:
    cols = st.columns([1, 3, 1, 1, 1])
    en = cols[0].checkbox("enabled", b["enabled"], key=k(key + "en"))
    planes = cols[1].text_input("plane y positions [mm], comma-separated",
                                ", ".join(f"{y:g}" for y in b["planes_y_mm"]), key=k(key + "pl"))
    rc = cols[2].number_input("CG radius r_c [mm]", 0.1, 500.0, float(b["r_c_mm"]), 0.5, key=k(key + "rc"))
    if crank:
        x = z = 0.0
    else:
        x = cols[3].number_input("shaft x [mm]", -1000.0, 1000.0, float(b["x_mm"]), 1.0, key=k(key + "x"))
        z = cols[4].number_input("shaft z [mm]", -1000.0, 1000.0, float(b["z_mm"]), 1.0, key=k(key + "z"))
    return Balancer(b["name"], b["order"], b["sense"], en, parse_planes(planes), rc, x, z)


with tab_bal:
    st.subheader("Crankshaft counterweights (1st order, co-rotating)")
    st.caption("One plane = one lumped counterweight (mass × radius solved per plane). "
               "Two planes cancel force *and* couple exactly; more planes → minimum-norm split; one plane → force only.")
    crank_cw = balancer_inputs(E["crank_cw"], "crank", crank=True)
    fwd = st.slider("Share of forward 1st-order reciprocating component put on the crank counterweights [%]",
                    0, 100, int(round(100 * E["fwd_primary_on_crank"])), key=k("fwd"),
                    help="100 % = classic 'balance the forward rotating half on the crank'. "
                         "The remainder goes to the primary co-rotating shaft (if enabled) or stays as residual.")

    st.subheader("Balance shafts")
    st.caption("x/z position only affects the 3-D view and the roll moment M_y; the solved masses do not depend on it.")
    shafts_in = E["shafts"] or [s.__dict__ for s in default_shafts()]
    shafts = []
    for i, s in enumerate(shafts_in):
        with st.expander(("✅ " if s["enabled"] else "▫️ ") + s["name"], expanded=s["enabled"]):
            shafts.append(balancer_inputs(s, f"sh{i}"))

# --------------------------------------------------------------------------- build & solve
eng = Engine(name=name, r_mm=r_mm, l_mm=l_mm, rpm=rpm, cylinders=cylinders, crank_cw=crank_cw,
             fwd_primary_on_crank=fwd / 100.0, shafts=shafts, notes=E.get("notes", ""))
if "_apply_masses" in st.session_state:  # mass helper: write into all cylinders, keep all other edits
    mo, mr, mt = st.session_state.pop("_apply_masses")
    for c in eng.cylinders:
        c.m_osc_g, c.m_rot_g, c.m_throw_g = mo, mr, mt
    load_engine(eng)
    st.rerun()

with st.sidebar:
    st.download_button("💾 Save engine (.json)", json.dumps(eng.to_dict(), indent=2),
                       file_name=f"{eng.name.replace(' ', '_')}.json", mime="application/json",
                       width="stretch")
    if eng.notes:
        st.info(eng.notes)

if not cylinders:
    st.warning("Add at least one cylinder.")
    st.stop()

res = solve(eng)

with tab_eng:
    st.dataframe(pd.DataFrame({"cylinder": [c.name for c in eng.cylinders],
                               "TDC at θ [°]": [f"{a:.1f}" for a in tdc_angles(eng)]}).set_index("cylinder").T,
                 width="stretch")
    st.subheader("Geometry")
    theta_view = st.slider("crank angle θ [deg]", 0, 720, 0, 5, key="theta_view")
    g1, g2 = st.columns([3, 2])
    g1.plotly_chart(geometry_3d(eng, res, theta_view), width="stretch")
    g2.plotly_chart(end_view(eng, res, theta_view), width="stretch")


# --------------------------------------------------------------------------- tab 3: results
def plane_table(b, pin0: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "plane y [mm]": p.y_mm, "m·r [g·mm]": p.mr_gmm, "r_c [mm]": p.r_c_mm, "mass [g]": p.mass_g,
        "angle @θ=0 [°]": p.angle_deg,
        "angle rel. to pin of cyl 1 [°]": (p.angle_deg - pin0) % 360 if b.spec.sense > 0 else np.nan,
    } for p in b.planes])


with tab_res:
    pin0 = eng.cylinders[0].pin_deg
    st.subheader("Balancing masses")
    if not res.balancers:
        st.info("No balancing elements enabled.")
    for b in res.balancers:
        st.markdown(f"**{b.spec.name}** — total {b.total_mass_g:.1f} g")
        st.dataframe(plane_table(b, pin0).style.format(precision=1), width="stretch", hide_index=True)

    st.subheader("Free forces & moments per harmonic")
    st.caption("Each entry is the amplitude of a rotating vector (forward = with the crank, backward = against it). "
               "Moments are about the origin (y = 0). A backward + forward pair of equal size = a force that "
               "oscillates along a fixed line.")
    st.dataframe(pd.DataFrame(harmonic_table(res)).style.format(precision=2), width="stretch",
                 hide_index=True)

    st.subheader("Over crank angle")
    theta = np.linspace(0, 720, 721)
    ts = timeseries(res, theta)
    q1, q2 = st.columns([1, 3])
    qty = q1.radio("quantity", ["Fx", "Fz", "Mx", "Mz", "My"], horizontal=False)
    default_series = ["Engine (unbalanced)", "Balancers (sum)", "Residual"]
    series = q2.multiselect("series", [n for n in ts if qty in ts[n]],
                            default=[s for s in default_series if qty in ts[s]])
    st.plotly_chart(time_plot(ts, theta, qty, series, eng.rpm), width="stretch")
    l1, l2 = st.columns(2)
    l1.plotly_chart(locus_plot(ts, "F"), width="stretch")
    l2.plotly_chart(locus_plot(ts, "M"), width="stretch")
    st.plotly_chart(rpm_sweep_plot(eng, res, rpm_max), width="stretch")

    rows = {"theta_deg": theta}
    for n, d in ts.items():
        for qq, arr in d.items():
            rows[f"{n} | {qq}"] = arr
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    st.download_button("⬇️ Time series (.csv)", buf.getvalue(), file_name="balance_timeseries.csv", mime="text/csv")

# --------------------------------------------------------------------------- theory
with tab_th:
    st.markdown(r"""
### Model
Coordinates as in Dagna et al. (2021): **y** = crank axis, crank angle **θ** measured from **+z** towards **+x**.
Every force in the x-z plane is written as a complex number $c = F_z + jF_x$.

For cylinder *i* with bank angle $\beta_i$, crankpin angle $\psi_i$, $\theta_i=\theta+\psi_i-\beta_i$, $\Lambda=r/l$:

$$F_{rot} = m_{rot}\,r\,\omega^2\,e^{j(\theta+\psi_i)}$$

$$F_{osc,n} = m_{osc}\,r\,\omega^2\,k_n\cos(n\theta_i)\,e^{j\beta_i}
= \tfrac12 m_{osc} r\omega^2 k_n\left[e^{j(n\psi_i-(n-1)\beta_i)}e^{+jn\theta} + e^{j((n+1)\beta_i-n\psi_i)}e^{-jn\theta}\right]$$

with $k_1=1,\;k_2=\Lambda$. So every contribution is a **forward** ($+n\omega$) or **backward** ($-n\omega$) rotating
phasor – the paper's "fictitious rotating / counter-rotating masses", but valid for any bank angle and split pins.
Moments: $M_x = \mathrm{Re}\sum y_i c_i$, $M_z = -\mathrm{Im}\sum y_i c_i$.

### Balancing
A counterweight set turning at $s\,n\,\omega$ with planes $y_k$ produces $n^2 U_k e^{jsn\theta}$, $U_k=m_k r_{c,k} e^{j\varphi_k}$.
For its harmonic $(n,s)$:

$$\sum_k n^2 U_k = -P_F(n,s),\qquad \sum_k n^2 y_k U_k = -P_M(n,s)$$

| element | harmonic | cancels |
|---|---|---|
| crank counterweights | +1 | rotating masses + *share* of forward primary |
| primary co-rotating shaft | +1 | rest of forward primary |
| primary counter-rotating shaft | −1 | backward primary |
| secondary shafts | ±2 | forward / backward secondary |

This is the matrix form of the paper's Eqs. (10)–(23).

### Limits (v1)
* 2nd-order expansion of piston acceleration (no 4th, 6th … order; no crank/pin offset).
* Inertia torque about the crank axis (M_y of the reciprocating masses) is not included; M_y shown is only the
  rocking couple created by offset balance shafts.
* Global (rigid-body) balance only – no bending / bearing loads.
""")
