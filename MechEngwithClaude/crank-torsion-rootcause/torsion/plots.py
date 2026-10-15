"""Report figures (matplotlib, static PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import Result
from .excitation import THETA_DEG, PressureSource, cylinder_torques

# categorical slots in fixed order (validated reference palette, light mode)
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BAND = "#eef3fb"
REF = "#e34948"   # reference/limit lines (labelled, so not colour-only)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 110, "savefig.dpi": 150,
})


def _ok(ax, st):
    lo, hi = st.rpm_operating
    ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    return name


def _oi(res, k):
    return int(np.argmin(np.abs(res.orders - k)))


def _ri(res, rpm):
    return int(np.argmin(np.abs(res.rpm - rpm)))


# ----------------------------------------------------------------------------- 01 cycle
def fig_cycle(res: Result, out: Path):
    st = res.study
    rpm = st.rpm_detail
    p = PressureSource(st.pressure, st.engine).p_bar(rpm)
    Tg, Ti = cylinder_torques(st.engine, p, rpm)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    a1.plot(THETA_DEG, p, color=S[0])
    a1.set_ylabel("cylinder pressure [bar]")
    a1.set_title(f"Single cylinder at {rpm:.0f} rpm  (p_max {p.max():.0f} bar)")
    for lab, y, c in [("gas", Tg, S[0]), ("oscillating mass", Ti, S[1]), ("total", Tg + Ti, INK)]:
        a2.plot(THETA_DEG, y, color=c, label=lab, lw=1.2 if lab != "total" else 1.8)
    a2.set_ylabel("torque on crank [Nm]")
    a2.set_xlabel("crank angle from firing TDC [deg]")
    a2.set_xticks(np.arange(0, 721, 90))
    a2.legend(ncol=3, loc="upper right")
    return _save(fig, out, "01_cycle_pressure_torque.png")


# ----------------------------------------------------------------------------- 02 single-cyl spectrum
def fig_spectrum(res: Result, exc, out: Path):
    st = res.study
    a = _ri(res, st.rpm_detail)
    o = res.orders
    w = 0.13
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(o - w, np.abs(exc.Cg[a]), width=w * 0.92, color=S[0], label="gas")
    ax.bar(o, np.abs(exc.Ci[a]), width=w * 0.92, color=S[1], label="oscillating mass")
    ax.bar(o + w, np.abs(exc.Cg[a] + exc.Ci[a]), width=w * 0.92, color=MUTED, label="combined (vector sum)")
    ax.set_xticks(o[o <= 8])
    ax.set_xlim(0.2, 8.4)
    ax.set_xlabel("order")
    ax.set_ylabel("harmonic amplitude [Nm]")
    ax.set_title(f"Single-cylinder torque harmonics at {st.rpm_detail:.0f} rpm")
    ax.legend()
    return _save(fig, out, "02_single_cylinder_order_spectrum.png")


# ----------------------------------------------------------------------------- 03 highlight order vs rpm
def fig_order_vs_rpm(res: Result, exc, out: Path):
    st = res.study
    k = st.highlight_order
    b = _oi(res, k)
    g, i = exc.Cg[:, b], exc.Ci[:, b]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    for ax in (a1, a2):
        _ok(ax, st)
    a1.plot(res.rpm, np.abs(g), color=S[0], label="gas")
    a1.plot(res.rpm, np.abs(i), color=S[1], label="oscillating mass (~ w^2)")
    a1.plot(res.rpm, np.abs(g + i), color=INK, lw=2, label="combined")
    a1.set_ylabel(f"order {k:g} amplitude [Nm]")
    a1.set_title(f"Order {k:g} excitation of one cylinder - gas vs mass forces")
    a1.legend()
    dphi = np.rad2deg(np.angle(i * np.conj(g)))
    a2.plot(res.rpm, dphi, color=S[2])
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.axhline(180, color=MUTED, lw=0.8, ls=":")
    a2.axhline(-180, color=MUTED, lw=0.8, ls=":")
    a2.set_ylim(-190, 190)
    a2.set_yticks([-180, -90, 0, 90, 180])
    a2.set_ylabel("phase inertia - gas [deg]")
    a2.set_xlabel("engine speed [rpm]")
    a2.text(0.01, 0.92, "0 deg = adding, +/-180 deg = cancelling", transform=a2.transAxes, color=INK2)
    return _save(fig, out, f"03_order{k:g}_gas_vs_inertia.png")


# ----------------------------------------------------------------------------- 04 mode shapes
def fig_modes(res: Result, out: Path):
    m = res.model
    st = res.study
    el = [r for r in range(len(m.omega)) if m.omega[r] > 1e-6 * m.omega.max()][:4]
    x = np.arange(len(m.names))
    s = m.spring_index(*st.section.spring_ab)
    i, j, _ = m.springs[s]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.axvspan(min(i, j), max(i, j), color=BAND, lw=0)
    a1.text((i + j) / 2, 1.04, st.section.label, ha="center", va="bottom", color=INK2, fontsize=8)
    for c, r in enumerate(el):
        phi = m.Phi[:, r] / np.max(np.abs(m.Phi[:, r]))
        a1.plot(x, phi, "-o", color=S[c], ms=5, label=f"mode {r}: {m.f_hz[r]:.0f} Hz")
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xticks(x, m.names, rotation=20)
    a1.set_ylabel("normalised torsional amplitude")
    a1.set_title("Torsional mode shapes (lumped chain)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    # modal torque in each spring per unit modal amplitude (mass normalised) -> where the mode 'hurts'
    lbl = [f"{m.names[a]}-{m.names[b]}" for a, b, _ in m.springs]
    w = 0.8 / len(el)
    for c, r in enumerate(el):
        tq = np.array([k * (m.Phi[a, r] - m.Phi[b, r]) for a, b, k in m.springs])
        a2.bar(np.arange(len(lbl)) + (c - (len(el) - 1) / 2) * w, np.abs(tq) / np.abs(tq).max(),
               width=w * 0.9, color=S[c], label=f"mode {r}")
    a2.set_xticks(np.arange(len(lbl)), lbl, rotation=20)
    a2.set_ylabel("|modal spring torque| (norm. per mode)")
    a2.set_title("Where each mode loads the shaft")
    a2.legend()
    return _save(fig, out, "04_mode_shapes.png")


# ----------------------------------------------------------------------------- 05 Campbell
def fig_campbell(res: Result, out: Path, crossings):
    st = res.study
    m = res.model
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _ok(ax, st)
    fmax = max(m.f_hz[1:3].max() * 1.15 if len(m.f_hz) > 2 else 1500, 1200)
    for k in res.orders:
        f = k * res.rpm / 60
        major = np.isclose(k % 1, 0)
        ax.plot(res.rpm, f, color=MUTED if not np.isclose(k, st.highlight_order) else S[1],
                lw=0.6 if not major else 0.9, ls="-" if major else ":", zorder=1)
        n_lab = min(res.rpm[-1], fmax * 60 / k)
        if n_lab > res.rpm[0] * 1.05:
            ax.text(n_lab, k * n_lab / 60, f" {k:g}", color=INK2, fontsize=7, va="bottom", ha="right")
    for r, f in enumerate(m.f_hz):
        if 1 < f < fmax:
            ax.axhline(f, color=S[0], lw=1.4)
            ax.text(res.rpm[0], f, f" chain mode {r}  {f:.0f} Hz", color=INK, va="bottom", fontsize=8)
    for f in st.chain.fe_modes_hz:
        if f < fmax:
            ax.axhline(f, color=S[0], lw=1.0, ls="--")
    tmax = max(c["tau_order"] for c in crossings) if crossings else 1
    for c in crossings:
        if c["f_hz"] < fmax:
            ax.scatter(c["rpm"], c["f_hz"], s=20 + 400 * c["tau_order"] / tmax, color=S[1],
                       edgecolor=SURF, linewidth=1.5, zorder=5)
    ax.set_ylim(0, fmax)
    ax.set_xlim(res.rpm[0], res.rpm[-1])
    ax.set_xlabel("engine speed [rpm]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("Campbell diagram - bubble size = section stress of that order at the crossing; "
                 "dashed = FE modes; shaded = operating range", fontsize=9)
    return _save(fig, out, "05_campbell.png")


# ----------------------------------------------------------------------------- 06 phase star
def fig_phase_star(res: Result, out: Path, mode=1, orders=(1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)):
    st, m = res.study, res.model
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.8), subplot_kw=dict(projection="polar"))
    phi = m.Phi[:, mode] / np.max(np.abs(m.Phi[:, mode]))
    vmax = sum(abs(phi[m.idx(c.node)]) for c in st.engine.cylinders)   # all cylinders in phase
    for ax, k in zip(axs.flat, orders):
        tot = 0j
        for c_i, c in enumerate(st.engine.cylinders):
            v = phi[m.idx(c.node)] * np.exp(-1j * k * np.deg2rad(c.fire_deg))
            ax.annotate("", xy=(np.angle(v), abs(v)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=S[c_i], lw=1.6))
            ax.text(np.angle(v), abs(v) * 1.08, c.name, color=INK2, fontsize=7, ha="center")
            tot += v
        ax.annotate("", xy=(np.angle(tot), abs(tot)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.4))
        ax.set_ylim(0, vmax * 1.05)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        hl = np.isclose(k, st.highlight_order)
        ax.set_title(f"order {k:g}:  {abs(tot) / vmax:.2f}", fontsize=9, color=S[1] if hl else INK, loc="center")
    fig.suptitle(f"Phase star weighted with mode {mode} ({m.f_hz[mode]:.0f} Hz) - black = vector sum, "
                 "value = relative modal excitation (1 = all cylinders in phase)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    return _save(fig, out, f"06_phase_star_mode{mode}.png")


# ----------------------------------------------------------------------------- 07 excitation factor bars
def fig_excitation_factor(res: Result, out: Path):
    m, st = res.model, res.study
    el = [r for r in range(len(m.omega)) if m.omega[r] > 1e-6 * m.omega.max()][:3]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.4 / len(el)
    for c, r in enumerate(el):
        norm = sum(abs(m.Phi[m.idx(cy.node), r]) for cy in st.engine.cylinders)
        ax.bar(res.orders + (c - (len(el) - 1) / 2) * w, np.abs(res.G[r]) / norm, width=w * 0.9,
               color=S[c], label=f"mode {r} ({m.f_hz[r]:.0f} Hz)")
    ax.set_xticks(res.orders[res.orders <= 8])
    ax.set_xlim(0.2, 8.4)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("order")
    ax.set_ylabel("relative modal excitation  |G| / sum|phi|")
    ax.set_title("Which orders can excite which mode (firing order x mode shape; 1 = all cylinders in phase)")
    ax.legend()
    return _save(fig, out, "07_modal_excitation_factor.png")


# ----------------------------------------------------------------------------- 08 section stress by order
def fig_stress_orders(res: Result, out: Path, n_top=6):
    st = res.study
    op = (res.rpm >= st.rpm_operating[0]) & (res.rpm <= st.rpm_operating[1])
    peak = np.abs(res.tau_k[op]).max(0)
    top = list(np.argsort(-peak)[:n_top])
    hb = _oi(res, st.highlight_order)
    if hb not in top:
        top[-1] = hb
    top = sorted(top, key=lambda b: res.orders[b])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _ok(ax, st)
    for c, b in enumerate(top):
        ax.plot(res.rpm, np.abs(res.tau_k[:, b]), color=S[c], lw=2.2 if b == hb else 1.3,
                label=f"order {res.orders[b]:g}")
    ax.plot(res.rpm, res.tau_amp, color=INK, lw=2.0, label="total alternating (time domain)")
    ax.axhline(st.material.tau_w_MPa, color=REF, ls="--", lw=1.2)
    ax.text(res.rpm[0], st.material.tau_w_MPa, " local fatigue strength tau_W", color=REF, va="bottom", fontsize=8)
    ax.set_xlabel("engine speed [rpm]")
    ax.set_ylabel("shear stress amplitude at fillet [MPa]")
    ax.set_title(f"Order-resolved stress at {st.section.label} (Kt {st.section.kt:g}, zeta {st.chain.zeta:g})")
    ax.legend(ncol=2, loc="upper left")
    return _save(fig, out, "08_section_stress_by_order.png")


# ----------------------------------------------------------------------------- 09 modal contribution
def fig_modal_contribution(res: Result, out: Path):
    st, m = res.study, res.model
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _ok(ax, st)
    ax.plot(res.rpm, res.tau_amp, color=INK, lw=2, label="total")
    for c, r in enumerate(range(len(m.omega))):
        lab = "rigid body (quasi-static)" if m.omega[r] < 1e-6 * m.omega.max() else f"mode {r} ({m.f_hz[r]:.0f} Hz)"
        ax.plot(res.rpm, res.tau_amp_mode[:, r], color=S[c % 8], lw=1.3, label=lab)
    ax.set_xlabel("engine speed [rpm]")
    ax.set_ylabel("alternating shear stress [MPa]")
    ax.set_title("Contribution of each mode to the fillet stress (all orders)")
    ax.legend(loc="upper left")
    return _save(fig, out, "09_modal_contribution.png")


# ----------------------------------------------------------------------------- 10/11 counterfactuals
def fig_counterfactual_curves(base: Result, groups: dict[str, list[Result]], out: Path):
    st = base.study
    fig, axs = plt.subplots(len(groups), 1, figsize=(9, 3.9 * len(groups)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, (title, cases) in zip(axs, groups.items()):
        _ok(ax, st)
        ax.plot(base.rpm, base.tau_amp, color=INK, lw=2.2, label="baseline")
        for c, r in enumerate(cases):
            ax.plot(r.rpm, r.tau_amp, color=S[c], lw=1.4, label=r.name)
        ax.axhline(st.material.tau_w_MPa, color=REF, ls="--", lw=1.0)
        ax.set_ylabel("alternating shear [MPa]")
        ax.set_title(title)
        ax.legend(loc="upper left", ncol=2)
    axs[-1].set_xlabel("engine speed [rpm]")
    return _save(fig, out, "10_counterfactual_stress.png")


def fig_counterfactual_life(cases: list[Result], out: Path, observed=None):
    names = [c.name for c in cases][::-1]
    life = np.array([min(c.life_h, 1e7) for c in cases])[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(cases) + 1.4))
    ax.barh(names, life, color=[INK if n == "baseline" else S[0] for n in names], height=0.6)
    ax.set_xscale("log")
    for y, v in enumerate(life):
        ax.text(v, y, f"  {v:,.3g} h" if v < 1e7 else "  > 1e7 h", va="center", color=INK2, fontsize=8)
    if observed:
        ax.axvline(observed, color=REF, ls="--", lw=1.2)
        ax.text(observed, len(names) - 0.4, " observed failure", color=REF, fontsize=8)
    ax.set_xlabel("predicted life with duty cycle [engine hours, log]")
    ax.set_title("What happens to the life if one link of the chain is removed?")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "11_counterfactual_life.png")


# ----------------------------------------------------------------------------- 12 Haigh
def fig_haigh(res: Result, out: Path):
    st, mat = res.study, res.study.material
    op = (res.rpm >= st.rpm_operating[0]) & (res.rpm <= st.rpm_operating[1])
    tm = (res.tau_t.max(1) + res.tau_t.min(1)) / 2
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(tm[op], res.tau_amp[op], c=res.rpm[op], cmap="Blues", s=14, vmin=st.rpm_operating[0] - 4000)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("rpm")
    xm = max(tm[op].max() * 2.0, 150.0)
    x = np.linspace(min(0, tm.min()), xm, 50)
    ax.plot(x, mat.tau_w_MPa - mat.mean_sens_M * np.maximum(x, 0), color=REF, ls="--", lw=1.2)
    ax.text(x[2], mat.tau_w_MPa * 1.02, "endurance limit (FKM mean-stress line)", color=REF, fontsize=8)
    ax.set_xlabel("mid stress of the cycle (max+min)/2 [MPa]")
    ax.set_ylabel("alternating shear stress [MPa]")
    ax.set_title(f"Haigh diagram at {st.section.label} - operating range")
    return _save(fig, out, "12_haigh.png")


# ----------------------------------------------------------------------------- 13 damage per bin
def fig_damage(res: Result, out: Path):
    du = res.study.duty
    bins = np.asarray(du.rpm_bins)
    frac = np.asarray(du.time_frac) / np.sum(du.time_frac)
    w = np.diff(bins).mean() * 0.85
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
    a1.bar(bins, 100 * frac, width=w, color=S[0])
    a1.set_ylabel("time share [%]")
    a1.set_title("Duty cycle (rpm histogram)")
    share = 100 * res.bin_damage / res.bin_damage.sum() if res.bin_damage.sum() > 0 else res.bin_damage
    a2.bar(bins, share, width=w, color=S[1])
    a2.set_ylabel("share of fatigue damage [%]")
    a2.set_xlabel("engine speed [rpm]")
    a2.set_title(f"Damage contribution - predicted life {res.life_h:,.3g} h")
    return _save(fig, out, "13_damage_by_rpm.png")
