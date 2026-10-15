"""Example configurations.

* The two "Paper" presets reproduce the validation cases of Dagna et al. (2021), Tables 1-5.
* The generic V4 presets use round, made-up numbers for a ~1 l motorcycle V4. Replace them
  with your own data (or save your engine as JSON and load it in the app).
"""
from __future__ import annotations

from .model import Balancer, Cylinder, Engine, default_shafts


# --------------------------------------------------------------------------- paper validation
def paper_inline3() -> Engine:
    """Paper Sec. 6.1 / Table 1: 3-cyl inline, firing order 1-3-2, 120 deg, a = 90 mm.

    m_osc = piston 500 + wrist pin 240 + conrod reciprocating 200 = 940 g
    m_rot = conrod rotating 400 g, throw = crank pin 400 + 2 webs * 1800 * 15/40 = 1750 g
    Paper places 4 counterweights (Y_cw = 110011); here the two webs of crank 1 and 3 are
    lumped into one plane each, so the plane mass is 2 x the paper's per-web m_c.
    """
    m = dict(m_osc_g=940.0, m_rot_g=400.0, m_throw_g=1750.0)
    cyl = [Cylinder("1", 0.0, 0.0, -90.0, **m),
           Cylinder("2", 0.0, 240.0, 0.0, **m),
           Cylinder("3", 0.0, 120.0, 90.0, **m)]
    return Engine(
        name="Paper validation: inline-3 (Table 1)", r_mm=40.0, l_mm=135.0, rpm=2000.0, cylinders=cyl,
        crank_cw=Balancer("Crankshaft counterweights", 1, +1, True, [-90.0, 90.0], 30.0),
        fwd_primary_on_crank=0.0,
        shafts=default_shafts(r_c_mm=30.0, l_sh1=200.0, l_sh2=200.0, x1=100.0, x2=200.0,
                              enabled=(True, True, True, True)),
        notes="Expected (paper Table 2): m_c = 1.2413 kg per web (=2.4826 kg per lumped plane), "
              "m_c,1 = 0.4884 kg, m_c,2 = 0.0362 kg.",
    )


def paper_v4() -> Engine:
    """Paper Sec. 6.2 / Table 4: 4-cyl 90 deg V, two shared crankpins at 180 deg, a = 90 mm."""
    m = dict(m_osc_g=940.0, m_rot_g=400.0, m_throw_g=875.0)  # throw 1750 g shared by 2 rods
    cyl = [Cylinder("A1", -45.0, 0.0, -45.0, **m), Cylinder("B1", 45.0, 0.0, -45.0, **m),
           Cylinder("A2", -45.0, 180.0, 45.0, **m), Cylinder("B2", 45.0, 180.0, 45.0, **m)]
    return Engine(
        name="Paper validation: V4 90 deg (Table 4)", r_mm=40.0, l_mm=135.0, rpm=2000.0, cylinders=cyl,
        crank_cw=Balancer("Crankshaft counterweights", 1, +1, True, [-45.0, 45.0], 30.0),
        fwd_primary_on_crank=0.0,
        shafts=default_shafts(r_c_mm=30.0, l_sh1=200.0, l_sh2=200.0, x1=100.0, x2=200.0,
                              enabled=(True, True, True, True)),
        notes="Expected (paper Table 5): m_c = 1.700 kg per web (=3.400 kg per lumped plane). "
              "The paper's balance-shaft masses for this case are not comparable: it treats the V "
              "reciprocating forces as purely vertical; here the full 2-D phasors are used.",
    )


# --------------------------------------------------------------------------- generic V4
def v4(name: str, vee_deg: float = 90.0, throw_deg=(0.0, 180.0), split_deg: float = 0.0,
       throw_spacing_mm: float = 80.0, rod_offset_mm: float = 18.0,
       r_mm: float = 26.0, l_mm: float = 105.0, rpm: float = 13000.0,
       m_osc_g: float = 330.0, m_rot_g: float = 250.0, m_throw_g: float = 300.0,
       r_c_mm: float = 35.0, fwd_primary_on_crank: float = 1.0,
       shafts_enabled=(False, False, False, False)) -> Engine:
    """Motorcycle V4: bank A at -V/2, bank B at +V/2, one throw per A/B cylinder pair.

    split_deg : extra crankpin angle of the B-cylinder relative to its A-partner (0 = shared pin).
    m_throw_g : per-cylinder share of the throw's own unbalance (pin + webs) reduced to r.
    """
    m = dict(m_osc_g=m_osc_g, m_rot_g=m_rot_g, m_throw_g=m_throw_g)
    cyl = []
    for t, psi in enumerate(throw_deg):
        yc = (t - (len(throw_deg) - 1) / 2) * throw_spacing_mm
        cyl.append(Cylinder(f"A{t + 1}", -vee_deg / 2, psi % 360, yc - rod_offset_mm / 2, **m))
        cyl.append(Cylinder(f"B{t + 1}", +vee_deg / 2, (psi + split_deg) % 360, yc + rod_offset_mm / 2, **m))
    y_out = throw_spacing_mm / 2 + rod_offset_mm / 2 + 15.0
    return Engine(
        name=name, r_mm=r_mm, l_mm=l_mm, rpm=rpm, cylinders=cyl,
        crank_cw=Balancer("Crankshaft counterweights", 1, +1, True, [-y_out, y_out], r_c_mm),
        fwd_primary_on_crank=fwd_primary_on_crank,
        shafts=default_shafts(r_c_mm=20.0, l_sh1=120.0, l_sh2=120.0, x1=90.0, x2=90.0, z=-60.0,
                              enabled=shafts_enabled),
        notes="Generic example numbers - replace with your own engine data.",
    )


def inline4_flatplane() -> Engine:
    m = dict(m_osc_g=330.0, m_rot_g=250.0, m_throw_g=600.0)
    ys = [-120.0, -40.0, 40.0, 120.0]
    pins = [0.0, 180.0, 180.0, 0.0]
    cyl = [Cylinder(str(i + 1), 0.0, p, y, **m) for i, (p, y) in enumerate(zip(pins, ys))]
    return Engine("Inline-4, 180 deg flat-plane crank", 26.0, 105.0, 13000.0, cyl,
                  crank_cw=Balancer("Crankshaft counterweights", 1, +1, True, [-135.0, -25.0, 25.0, 135.0], 35.0),
                  shafts=default_shafts(r_c_mm=20.0, l_sh1=120.0, l_sh2=120.0, x1=90.0, x2=90.0, z=-60.0),
                  notes="Classic result: primary forces and couples cancel; 2nd-order free force 4 m Lambda r w^2.")


PRESETS = {
    "V4 90 deg - 180 deg crank, shared pins": lambda: v4("V4 90 deg - 180 deg crank, shared pins", 90.0, (0.0, 180.0)),
    "V4 90 deg - 360 deg crank, shared pins": lambda: v4("V4 90 deg - 360 deg crank, shared pins", 90.0, (0.0, 0.0)),
    "V4 90 deg - 180 deg crank, 70 deg split pins": lambda: v4(
        "V4 90 deg - 180 deg crank, 70 deg split pins", 90.0, (0.0, 180.0), split_deg=70.0),
    "V4 65 deg - 180 deg crank, shared pins + counter shaft": lambda: v4(
        "V4 65 deg - 180 deg crank, shared pins + counter shaft", 65.0, (0.0, 180.0),
        shafts_enabled=(False, True, False, False)),
    "Inline-4 flat-plane": inline4_flatplane,
    "Paper validation: inline-3": paper_inline3,
    "Paper validation: V4 90 deg": paper_v4,
}
