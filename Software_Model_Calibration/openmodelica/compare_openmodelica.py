"""
compare_openmodelica.py
Cross-check of the OpenModelica package against the Python model and the ValDyn reference.

1. run   omc run_all.mos   (in this folder)  -> GearMeshBenchmark.Examples.*_res.csv
2. run   python compare_openmodelica.py
   -> prints code-to-code differences and objective values, writes om_vs_python_*.png
Needs ../python_calibration (gearmesh_model.py, calib_tools.py, stage2_objective.py,
reference_valdyn_digitized.npz).
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PYDIR = os.path.join(HERE, '..', 'python_calibration')
sys.path.insert(0, PYDIR)
import gearmesh_model as gm          # noqa: E402
import calib_tools as ct             # noqa: E402
import stage2_objective as so        # noqa: E402

REF = ct.load_reference(os.path.join(PYDIR, 'reference_valdyn_digitized.npz'))


def read_om(name):
    with open(os.path.join(HERE, f'GearMeshBenchmark.Examples.{name}_res.csv')) as fh:
        r = csv.reader(fh)
        head = next(r)
        d = np.array([[float(x) for x in row] for row in r])
    D = {k: d[:, i] for i, k in enumerate(head)}
    _, idx = np.unique(D['time'], return_index=True)      # drop duplicate event rows
    return {k: v[idx] for k, v in D.items()}


def stage1():
    o = read_om('Stage1')
    p = gm.simulate_stage1()
    print('=== Stage 1: OpenModelica (DASSL) vs Python (RK4) ===')
    for k in ('FA', 'FB'):
        e1 = ct.impact_events(p['angle'], p[k], 400, 4)
        e2 = ct.impact_events(o['angle_deg'], o[k], 400, 4)
        n = min(len(e1), len(e2))
        print(f'{k}: impacts {len(e1)}/{len(e2)}, max angle diff {np.abs(e1[:n, 0]-e2[:n, 0]).max():.3f} deg, '
              f'max peak diff {np.abs(e1[:n, 1]-e2[:n, 1]).max():.1f} N')
    pairs = [('lashA_mm', 'lashA', 's1_lashA_red'), ('lashB_mm', 'lashB', 's1_lashB_red'), ('FA', 'FA', 's1_FA_green'),
             ('FB', 'FB', 's1_FB_green'), ('y1_mm', 'y1', 's1_YM1_green'), ('y2_mm', 'y2', 's1_YM2_green')]
    fig, axs = plt.subplots(3, 2, figsize=(14, 10)); axs = axs.ravel()
    for ax, (om, py, rk) in zip(axs, pairs):
        R = REF[rk]
        ax.plot(R.pts[:, 0], R.pts[:, 1], '.', ms=1, color='0.65', label='ValDyn (digitised)')
        ax.plot(p['angle'], p[py], 'r', lw=.8, label='Python')
        ax.plot(o['angle_deg'], o[om], 'b--', lw=.8, label='OpenModelica')
        ax.set_xlim(R.xr); ax.set_ylim(R.yr); ax.set_title(om)
        h = dict(hide_y0=0.012) if om in ('FA', 'FB') else {}
        print(f'{om:9s} distance to ValDyn: Python {ct.curve_distance(R, p["angle"], p[py], **h):.4f}  '
              f'OpenModelica {ct.curve_distance(R, o["angle_deg"], o[om], **h):.4f}')
    axs[0].legend(); plt.tight_layout(); plt.savefig(os.path.join(HERE, 'om_vs_python_stage1.png'), dpi=80)


def stage2(name, kw):
    o = read_om(name)
    p = gm.simulate_stage2(gm.params(2, **kw))
    # map OpenModelica outputs to the Python result keys (Python support forces = table force = -ValDyn sign)
    om = {'angle': o['angle_deg'], 'lashA': o['lashA_mm'], 'lashB': o['lashB_mm'], 'Ft': o['Ft'], 'Tload': o['Tload'],
          'gearX': o['gearX_mm'], 'midY': o['midY_mm'], 'blockX': o['blockX_mm'], 'blockY': o['blockY_mm'],
          'FXBlock': -o['FXBlock'], 'FYBlock': -o['FYBlock'], 'FXCase': -o['FXCase'], 'FYCase': -o['FYCase']}

    def J(r):
        s = 0.0
        for rk, sk, w, sg in so.DET_CHANNELS:
            s += w*ct.curve_distance(REF[rk], r['angle'], sg*r[sk], so.DET_WINDOW)**2
        for rk, sk, w, sg in so.STAT_CHANNELS:
            s += w*ct.stat_distance(REF[rk], [(r['angle'], sg*r[sk])])**2
        return s
    print(f'=== {name}: objective vs ValDyn (single run)  Python J={J(p):.4f}   OpenModelica J={J(om):.4f}')
    yi = np.interp(p['angle'], om['angle'], om['gearX'])
    dev = np.abs(yi - p['gearX'])
    i = np.argmax(dev > 1e-3) if np.any(dev > 1e-3) else len(dev)-1
    print(f'    gear X: Python and OpenModelica identical (<1 um) up to {p["angle"][i]:.0f} deg (chaotic divergence afterwards)')
    pairs = [('s2_lash_red', 'lashA'), ('s2_lash_blue', 'lashB'), ('s2_Ft_red', 'Ft'), ('s2_X_red', 'gearX'),
             ('s2_Y_red', 'midY'), ('s2_FY_red', 'FYBlock')]
    fig, axs = plt.subplots(3, 2, figsize=(14, 10)); axs = axs.ravel()
    for ax, (rk, k) in zip(axs, pairs):
        R = REF[rk]; sg = -1 if k.startswith('FY') else 1
        ax.plot(R.pts[:, 0], R.pts[:, 1], '.', ms=1, color='0.65', label='ValDyn (digitised)')
        ax.plot(p['angle'], sg*p[k], 'r', lw=.6, label='Python')
        ax.plot(om['angle'], sg*om[k], 'b--', lw=.6, label='OpenModelica')
        ax.set_xlim(R.xr); ax.set_ylim(R.yr); ax.set_title(rk + '  <-  ' + k)
    axs[0].legend(); plt.tight_layout(); plt.savefig(os.path.join(HERE, f'om_vs_python_{name}.png'), dpi=80)


if __name__ == '__main__':
    stage1()
    stage2('Stage2', {})
    stage2('Stage2Calibrated', dict(backlash_n=0.012, k_mesh=25000))
