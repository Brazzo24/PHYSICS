"""
run_postprocessing.py - one-command post-processing of a GearMeshBenchmark-type model.

Examples
  # existing result file (e.g. simulated in OMEdit):
  python run_postprocessing.py --mat ../openmodelica/GearMeshBenchmark.Examples.Stage1_res.mat

  # compile + simulate + report, with parameter overrides and a modal analysis:
  python run_postprocessing.py --mo ../openmodelica/GearMeshBenchmark.mo \
      --model GearMeshBenchmark.Examples.GearTrain3 --set jn=0.08e-3 --revs 3 \
      --modes 0.05 --lin-set jn=1e-7 clr_idler=0 --orders 1 2 4 20 --rpm 0 8000

Writes <out>/report.html and <out>/metrics_<Type>.csv
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import mbspost as mp


def kv(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mat', help='existing OpenModelica result (.mat or .csv)')
    ap.add_argument('--mo', help='Modelica file (to compile and simulate)')
    ap.add_argument('--model', help='full model name, e.g. GearMeshBenchmark.Examples.Stage1')
    ap.add_argument('--omc', help='path to omc (default: OMC_PATH env, PATH, common install folders)')
    ap.add_argument('--set', nargs='*', help='parameter overrides name=value (top-level parameters)')
    ap.add_argument('--window', nargs=2, type=float, help='analysis window t0 t1 [s]')
    ap.add_argument('--revs', nargs='+', type=float, help='analysis window in reference revolutions: r0 [r1]')
    ap.add_argument('--reference', help='angle variable for x-axis / order analysis (default: first ConstantSpeed)')
    ap.add_argument('--modes', type=float, help='linearisation time [s] for modal analysis (needs --mo/--model)')
    ap.add_argument('--lin-set', nargs='*', help='overrides for the linearisation only, e.g. jn=1e-7 clr=0')
    ap.add_argument('--orders', nargs='*', type=float, default=[1, 2], help='excitation orders for the Campbell diagram')
    ap.add_argument('--rpm', nargs=2, type=float, default=[0, 10000])
    ap.add_argument('--out', default='pp_report', help='output folder')
    a = ap.parse_args()

    runner = None
    if a.mat:
        res = mp.OMResult(a.mat)
    else:
        if not (a.mo and a.model):
            ap.error('give --mat, or --mo and --model')
        runner = mp.OMRunner(a.mo, a.model, omc=a.omc)
        runner.build()
        res = runner.simulate(overrides=kv(a.set))
    sysm = mp.System(res, reference=a.reference)
    print(sysm.summary())
    revs = None
    if a.revs:
        revs = (a.revs[0], a.revs[1] if len(a.revs) > 1 else None)
    rep = sysm.report(window=tuple(a.window) if a.window else None, revs=revs)

    os.makedirs(a.out, exist_ok=True)
    modes, extra = None, []
    if a.modes is not None and runner is not None:
        modes = runner.modes(a.modes, overrides={**kv(a.set), **kv(a.lin_set)})
        print(modes.table().to_string())
        fig, tab = mp.campbell(modes, a.orders, rpm=tuple(a.rpm), only='torsional')
        extra.append(fig)
        print('resonances (torsional modes x orders):'); print(tab)
    rep.to_csv(a.out)
    path = rep.to_html(os.path.join(a.out, 'report.html'), title=os.path.basename(res.path), modes=modes, extra_figs=extra)
    print('written', path)


if __name__ == '__main__':
    main()
