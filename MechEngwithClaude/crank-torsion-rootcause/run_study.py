"""Root-cause study: is the ~1000 Hz torsional mode (excited by order 3) what breaks the crank?

    python run_study.py            -> results/*.png, results/summary.md, results/sweep.csv

Edit inputs.py (all  # EST  values) - nothing else needs to change.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np

import inputs
from torsion import analysis as A
from torsion import plots as P
from torsion.chain import build_model, scaled_frequency


def prescaled(chain, scale):
    """Freeze the FE-tuned stiffness so the chain can be extended (drivetrain) without re-tuning."""
    ch = copy.deepcopy(chain)
    for s in ch.springs:
        if s.tunable:
            s.k_Nm_rad *= scale
    ch.tune_to_fe_hz = None
    return ch


def main():
    st = inputs.baseline()
    out = Path(st.out_dir)
    out.mkdir(exist_ok=True)
    k3 = st.highlight_order

    print("excitation sweep ...")
    exc = A.excitation_for(st)
    base = A.run(st, exc, "baseline")
    m = base.model
    print("chain modes [Hz]:", np.round(m.f_hz, 1), " stiffness scale:", round(m.stiffness_scale, 3))

    # ---------------------------------------------------------------- counterfactuals
    exc_cases = [
        A.run(st, exc, f"no order-{k3:g} mass torque", A.only(k3, gas=True, inertia=False)),
        A.run(st, exc, f"no order-{k3:g} gas torque", A.only(k3, gas=False, inertia=True)),
        A.run(st, exc, f"no order {k3:g} at all", A.only(k3, gas=False, inertia=False)),
        A.run(st, exc, f"order {k3:g} only", A.single_order(k3)),
        A.run(st, exc, "overrun approx. (gas torque off)", lambda k: (0.0, 1.0)),
    ]
    st_up = copy.deepcopy(st); st_up.chain = scaled_frequency(st.chain, 1.10)
    st_dn = copy.deepcopy(st); st_dn.chain = scaled_frequency(st.chain, 0.90)
    st_z = copy.deepcopy(st); st_z.chain.zeta *= 2
    st_fB = copy.deepcopy(st); st_fB.engine = inputs.engine("B")
    st_dt = copy.deepcopy(st); st_dt.chain = inputs.add_drivetrain(prescaled(st.chain, m.stiffness_scale))
    exc_fB = exc   # single-cylinder excitation does not depend on the firing order
    struct_cases = [
        A.run(st_up, exc, "mode +10 %"),
        A.run(st_dn, exc, "mode -10 %"),
        A.run(st_z, exc, "damping x2"),
        A.run(st_fB, exc_fB, "firing order B"),
        A.run(st_dt, exc, "with primary drive + driveline"),
    ]
    all_cases = [base] + exc_cases + struct_cases

    # ---------------------------------------------------------------- figures
    cr = A.crossings(base, st.rpm_operating)
    figs = [
        P.fig_cycle(base, out),
        P.fig_spectrum(base, exc, out),
        P.fig_order_vs_rpm(base, exc, out),
        P.fig_modes(base, out),
        P.fig_campbell(base, out, A.crossings(base)),
        P.fig_phase_star(base, out, mode=1),
        P.fig_excitation_factor(base, out),
        P.fig_stress_orders(base, out),
        P.fig_modal_contribution(base, out),
        P.fig_counterfactual_curves(base, {
            "Excitation counterfactuals (same structure)": exc_cases,
            "Structural / layout counterfactuals (all orders)": struct_cases}, out),
        P.fig_counterfactual_life(all_cases, out, st.duty.observed_life_h),
        P.fig_haigh(base, out),
        P.fig_damage(base, out),
    ]

    # ---------------------------------------------------------------- csv
    op = (base.rpm >= st.rpm_operating[0]) & (base.rpm <= st.rpm_operating[1])
    top_orders = list(np.argsort(-np.abs(base.tau_k[op]).max(0))[:8])
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rpm", "tau_mean_MPa", "tau_amp_MPa", "D_per_cycle"]
                   + [f"tau_order_{base.orders[b]:g}_MPa" for b in top_orders]
                   + [f"tau_mode_{r}_MPa" for r in range(len(m.omega))])
        for a in range(len(base.rpm)):
            w.writerow([base.rpm[a], round(base.tau_mean[a], 3), round(base.tau_amp[a], 3), f"{base.D_cycle[a]:.3e}"]
                       + [round(abs(base.tau_k[a, b]), 3) for b in top_orders]
                       + [round(v, 3) for v in base.tau_amp_mode[a]])

    # ---------------------------------------------------------------- summary
    a_pk = int(np.flatnonzero(op)[np.argmax(base.tau_amp[op])])
    b_dom = int(np.argmax(np.abs(base.tau_k[a_pk])))
    r_dom = int(np.argmax(base.tau_amp_mode[a_pk]))
    shares = base.bin_damage / base.bin_damage.sum()
    top_bins = np.argsort(-shares)[:3]
    b3 = int(np.argmin(np.abs(base.orders - k3)))
    norm = sum(abs(m.Phi[m.idx(c.node), 1]) for c in st.engine.cylinders)
    G_rel = np.abs(base.G[1]) / norm
    low = base.orders <= 6
    rank3 = int(np.sum(G_rel[low] > G_rel[b3])) + 1

    L = []
    L.append("# Crankshaft torsional root-cause study - summary\n")
    L.append("> **All inputs tagged `# EST` in `inputs.py` are estimates.** Numbers below show the method "
             "and the mechanism, not your engine, until the real data is in.\n")
    L.append("## Model")
    L.append(f"- Chain nodes: {', '.join(m.names)}; modes [Hz]: {', '.join(f'{f:.0f}' for f in m.f_hz)}")
    L.append(f"- Tunable stiffness scaled x{m.stiffness_scale:.3f} to match FE {st.chain.tune_to_fe_hz} Hz; "
             f"modal damping zeta = {st.chain.zeta}")
    L.append(f"- Failure section {st.section.label}: D {st.section.d_outer_mm} mm, d {st.section.d_inner_mm} mm, "
             f"Kt {st.section.kt}; tau_W {st.material.tau_w_MPa} MPa\n")
    L.append("## Resonance crossings inside the operating range (sorted by section stress)")
    L.append("| mode | f [Hz] | order | rpm | stress of that order [MPa] | rel. modal excitation |")
    L.append("|---|---|---|---|---|---|")
    for c in cr[:10]:
        L.append(f"| {c['mode']} | {c['f_hz']:.0f} | {c['order']:g} | {c['rpm']:.0f} | {c['tau_order']:.1f} | "
                 f"{G_rel[int(np.argmin(np.abs(base.orders - c['order'])))]:.2f} |" if c['mode'] == 1 else
                 f"| {c['mode']} | {c['f_hz']:.0f} | {c['order']:g} | {c['rpm']:.0f} | {c['tau_order']:.1f} | - |")
    L.append("")
    L.append("## Key findings (baseline)")
    L.append(f"- Peak alternating shear in operating range: **{base.tau_amp[a_pk]:.0f} MPa at {base.rpm[a_pk]:.0f} rpm** "
             f"(mean {base.tau_mean[a_pk]:.0f} MPa); dominant order there: **{base.orders[b_dom]:g}**; "
             f"dominant mode: **{r_dom}** ({base.tau_amp_mode[a_pk, r_dom]:.0f} MPa of it)")
    L.append(f"- Order {k3:g} relative modal excitation of mode 1: {G_rel[b3]:.2f} "
             f"(rank {rank3} of {int(low.sum())} orders up to 6)")
    a3 = int(np.argmin(np.abs(base.rpm - 60 * m.f_hz[1] / k3)))
    dph = np.rad2deg(np.angle(exc.Ci[a3, b3] * np.conj(exc.Cg[a3, b3])))
    L.append(f"- At the order-{k3:g} crossing ({base.rpm[a3]:.0f} rpm) one cylinder's order-{k3:g} torque: gas "
             f"{abs(exc.Cg[a3, b3]):.0f} Nm, mass {abs(exc.Ci[a3, b3]):.0f} Nm, phase difference {dph:.0f} deg "
             f"-> {'they partly CANCEL (overrun / part load is worse!)' if abs(dph) > 90 else 'they ADD'}")
    L.append(f"- Predicted life with duty cycle: **{base.life_h:,.3g} h**"
             + (f" vs observed {st.duty.observed_life_h} h (ratio {base.life_h / st.duty.observed_life_h:.2f})"
                if st.duty.observed_life_h else ""))
    L.append("- Damage concentrated at: " + ", ".join(
        f"{st.duty.rpm_bins[i]:.0f} rpm ({100 * shares[i]:.0f} %)" for i in top_bins) + "\n")
    L.append("## Counterfactuals")
    L.append("| case | peak tau_a in op. range [MPa] | at rpm | life [h] | life / baseline |")
    L.append("|---|---|---|---|---|")
    for c in all_cases:
        a = int(np.flatnonzero(op)[np.argmax(c.tau_amp[op])])
        ratio = c.life_h / base.life_h
        L.append(f"| {c.name} | {c.tau_amp[a]:.0f} | {c.rpm[a]:.0f} | {c.life_h:,.3g} | {ratio:,.3g} |")
    L.append("\n**Reading the table:** if removing order "
             f"{k3:g} (or detuning the mode) raises the life by orders of magnitude while the other cases do not, "
             "the order-3 / mode-1 resonance is the cause. If 'no order 3 at all' barely helps, another crossing "
             "(see the crossing table / fig 08) is doing the damage.\n")
    L.append("## Figures")
    for f in figs:
        L.append(f"- ![{f}]({f})")
    (out / "summary.md").write_text("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nwritten to {out.resolve()}")


if __name__ == "__main__":
    main()
