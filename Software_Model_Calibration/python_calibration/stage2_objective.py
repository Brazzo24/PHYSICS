"""Stage-2 objective: deterministic start window + ensemble envelope statistics."""
import numpy as np
import gearmesh_model as gm
import calib_tools as ct

H = dict(hide_y0=0.012)
# (ref_key, result_key, weight, sign)
DET_CHANNELS = [('s2_lash_red', 'lashA', 1, 1), ('s2_lash_blue', 'lashB', 1, 1), ('s2_Ft_red', 'Ft', 1, 1)]
STAT_CHANNELS = [('s2_lash_red', 'lashA', 1, 1), ('s2_lash_blue', 'lashB', 1, 1), ('s2_Ft_red', 'Ft', 1, 1),
                 ('s2_X_red', 'gearX', .5, 1),   # ValDyn 'BearingX' behaves like the GEAR centre
                 ('s2_Y_red', 'midY', .5, 1),    # ValDyn 'BearingY' behaves like the SHAFT node under the bearing
                 ('s2_X_cyan', 'blockX', .5, 1), ('s2_Y_cyan', 'blockY', .5, 1),
                 ('s2_FX_red', 'FXBlock', .5, -1), ('s2_FX_blue', 'FXCase', .5, -1),
                 ('s2_FY_red', 'FYBlock', .5, -1), ('s2_FY_blue', 'FYCase', .5, -1)]
DET_WINDOW = (0.0, 200.0)


def run_ensemble(p, n=3, dt=4e-7, out_dt=4e-6, eps=1e-6):
    """n runs with tiny (1e-6) mass perturbations: samples of the chaotic attractor."""
    out = []
    for i in range(n):
        q = dict(p); q['m2'] = p['m2']*(1+eps*i)
        out.append(gm.simulate_stage2(q, dt=dt, out_dt=out_dt))
    return out


def evaluate(p, ref, n=3, w_det=1.0, w_stat=1.0, return_parts=False):
    runs = run_ensemble(p, n)
    parts = {}
    r0 = runs[0]
    for rk, sk, w, sg in DET_CHANNELS:
        parts['det:'+rk] = ct.curve_distance(ref[rk], r0['angle'], sg*r0[sk], DET_WINDOW)
    for rk, sk, w, sg in STAT_CHANNELS:
        parts['stat:'+rk] = ct.stat_distance(ref[rk], [(r['angle'], sg*r[sk]) for r in runs])
    J = w_det*sum(w*parts['det:'+rk]**2 for rk, sk, w, sg in DET_CHANNELS) + \
        w_stat*sum(w*parts['stat:'+rk]**2 for rk, sk, w, sg in STAT_CHANNELS)
    return (J, parts, runs) if return_parts else J
