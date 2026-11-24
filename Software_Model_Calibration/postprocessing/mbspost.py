"""
mbspost.py - post-processing for drivetrain / camtrain models built from the
GearMeshBenchmark Modelica components (OpenModelica results).

Workflow
--------
    runner = OMRunner('GearMeshBenchmark.mo', 'GearMeshBenchmark.Examples.Stage1')
    runner.build()                                   # compile once
    res = runner.simulate(overrides={'jn': 0.1e-3})  # fast re-runs with other parameters
    # or: res = OMResult('model_res.mat')            # any existing result file

    sysm = System(res)                               # finds all components automatically
    print(sysm.summary())                            # what was found
    rep = sysm.report(window=(0.01, None))           # metrics for every component
    rep.to_html('report.html')                       # figures + tables in one file

    modes = runner.modes(t_lin=0.08)                 # linearise -> eigenfrequencies, mode shapes
    campbell(modes, orders=[1, 2, 20], rpm=(0, 8000))

Design
------
* Component discovery is signature based: every instance whose variables
  contain the signature of a registered component type is recognised, at any
  nesting depth and with any instance name.  New component types are added
  with `register()` (see the bottom of the file).
* All metrics are computed in an analysis window (default: whole run) so the
  start-up transient can be excluded.
* Units: SI inside, reported with unit suffixes in the metric names
  (_N, _Nm, _um, _deg, _Hz, _pct ...).
"""
from __future__ import annotations

import base64
import io
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.io as sio

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

import matplotlib
import matplotlib.pyplot as plt

_trapz = getattr(np, 'trapezoid', None) or np.trapz

__all__ = ['OMResult', 'OMRunner', 'System', 'Report', 'Modes', 'ComponentType', 'register', 'REGISTRY',
           'order_spectrum', 'spectrum', 'campbell', 'contact_state', 'compare_reports', 'sweep']


# =============================================================================
# 1. Reading OpenModelica results
# =============================================================================
class OMResult:
    """Reader for OpenModelica result files (.mat 'Atrajectory' v1.1, or .csv).

    res.time            time vector [s]
    res['gear1.w']      time series (parameters are returned as constant arrays)
    res.value('mesh.k') scalar value of a parameter (or final value of a variable)
    res.names           all variable names
    res.has(name)
    """

    def __init__(self, path):
        self.path = path
        if path.endswith('.csv'):
            self._read_csv(path)
        else:
            self._read_mat(path)
        self._index = {n: i for i, n in enumerate(self.names)}

    # -- .mat ---------------------------------------------------------------
    def _read_mat(self, path):
        m = sio.loadmat(path, chars_as_strings=False)
        trans = ''.join(m['Aclass'][3]).strip('\x00 ').lower() == 'bintrans'

        def strings(a):
            a = a if not trans else a.T
            return [''.join(r).rstrip('\x00 ') for r in a]
        self.names = strings(m['name'])
        self.descriptions = strings(m['description'])
        info = m['dataInfo'] if not trans else m['dataInfo'].T      # (nvar, 4)
        d1 = m['data_1'] if not trans else m['data_1'].T            # (ntime1, n1)
        d2 = m['data_2'] if not trans else m['data_2'].T            # (ntime, n2)
        self.time = d2[:, 0].astype(float)
        self._info = info
        self._d1, self._d2 = d1, d2

    def _col(self, i):
        ds, idx = int(self._info[i, 0]), int(self._info[i, 1])
        sgn = 1.0 if idx >= 0 else -1.0
        k = abs(idx) - 1
        if ds == 0 or (ds == 2 and k == 0 and self.names[i] == 'time'):
            return self.time.copy()
        if ds == 1:
            return np.full(len(self.time), sgn*self._d1[0, k])
        return sgn*self._d2[:, k].astype(float)

    # -- .csv ---------------------------------------------------------------
    def _read_csv(self, path):
        import csv
        with open(path) as fh:
            r = csv.reader(fh)
            head = [h.strip('"') for h in next(r)]
            data = np.array([[float(x) for x in row] for row in r])
        self.names = head
        self.descriptions = [''] * len(head)
        self.time = data[:, head.index('time')]
        self._csv = data
        self._info = None

    # -- access ---------------------------------------------------------------
    def has(self, name):
        return name in self._index

    def __getitem__(self, name):
        i = self._index[name]
        if self._info is None:
            return self._csv[:, i].copy()
        return self._col(i)

    def get(self, name, default=None):
        return self[name] if self.has(name) else default

    def value(self, name):
        """Parameter value (or last value of a variable)."""
        return float(self[name][-1])

    def is_parameter(self, name):
        return self._info is not None and int(self._info[self._index[name], 0]) == 1

    def instances(self):
        """Set of instance prefixes (everything before the last '.')."""
        out = set()
        for n in self.names:
            if n.startswith('der(') or '.' not in n:
                continue
            parts = n.split('.')
            for k in range(1, len(parts)):
                out.add('.'.join(parts[:k]))
        return out

    def __repr__(self):
        return f'OMResult({os.path.basename(self.path)}: {len(self.names)} variables, {len(self.time)} samples, t = {self.time[0]:g}..{self.time[-1]:g} s)'


# =============================================================================
# 2. Running OpenModelica from Python (compile once, run many times)
# =============================================================================
def _find_omc():
    for c in ('/opt/openmodelica/bin/omc', '/Applications/OpenModelica/build_cmake/install_cmake/bin/omc',
              '/usr/local/bin/omc', '/opt/homebrew/bin/omc', '/usr/bin/omc',
              r'C:\Program Files\OpenModelica1.24.0-64bit\bin\omc.exe'):
        if os.path.exists(c):
            return c
    return 'omc'


class OMRunner:
    """Compile a model once with omc, then run the executable with parameter overrides.

    omc: path of the omc executable (macOS default install:
         '/Applications/OpenModelica/build_cmake/install_cmake/bin/omc' or
         '/opt/openmodelica/bin/omc'; Linux: 'omc').
    """

    def __init__(self, mo_file, model, workdir=None, omc=None, libs=()):
        self.mo_file = os.path.abspath(mo_file)
        self.model = model
        self.workdir = os.path.abspath(workdir or os.path.join(os.path.dirname(self.mo_file), 'om_build'))
        self.omc = omc or os.environ.get('OMC_PATH') or shutil.which('omc') or _find_omc()
        self.libs = libs
        os.makedirs(self.workdir, exist_ok=True)
        self.exe = os.path.join(self.workdir, model + ('.exe' if os.name == 'nt' else ''))

    def _omc(self, script):
        path = os.path.join(self.workdir, '_run.mos')
        with open(path, 'w') as fh:
            fh.write(script)
        out = subprocess.run([self.omc, path], cwd=self.workdir, capture_output=True, text=True)
        if out.returncode != 0 or 'Error' in out.stdout:
            msgs = [l for l in out.stdout.splitlines() if 'Error' in l or 'error' in l]
            if msgs:
                warnings.warn('omc: ' + ' | '.join(msgs[:5]))
        return out.stdout

    def build(self, linearize_python=True):
        """Compile the model (needed once, and after editing the .mo file)."""
        libs = ''.join(f'loadModel({l}); getErrorString();\n' for l in self.libs)
        opt = 'setCommandLineOptions("--linearizationDumpLanguage=python"); getErrorString();\n' if linearize_python else ''
        log = self._omc(f'{libs}{opt}loadFile("{self.mo_file}"); getErrorString();\n'
                        f'buildModel({self.model}); getErrorString();\n')
        if not os.path.exists(self.exe):
            raise RuntimeError('build failed:\n' + log[-3000:])
        return log

    def simulate(self, overrides=None, result=None, stop_time=None, tol=None, flags=()):
        """Run the compiled model. overrides: {'jn': 1e-4, 'mesh1.c': 2000, ...}.
        Returns an OMResult."""
        if not os.path.exists(self.exe):
            self.build()
        result = result or f'{self.model}_res.mat'
        ov = dict(overrides or {})
        if stop_time is not None:
            ov['stopTime'] = stop_time
        if tol is not None:
            ov['tolerance'] = tol
        cmd = [self.exe, f'-r={result}']
        if ov:
            cmd.append('-override=' + ','.join(f'{k}={v}' for k, v in ov.items()))
        cmd += list(flags)
        out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
        _check_overrides(out.stdout)
        if 'simulation finished successfully' not in out.stdout:
            raise RuntimeError('simulation failed:\n' + out.stdout[-3000:] + out.stderr[-2000:])
        return OMResult(os.path.join(self.workdir, result))

    def linearize(self, t_lin, overrides=None):
        """Linearise the model at time t_lin (simulate to t_lin, then build A).
        Returns dict(A, x0, states).  Requires build(linearize_python=True)."""
        if not os.path.exists(self.exe):
            self.build()
        lin = os.path.join(self.workdir, 'linearized_model.py')
        if os.path.exists(lin):
            os.remove(lin)
        cmd = [self.exe, f'-l={t_lin}', '-r=_lin_res.mat']
        if overrides:
            cmd.append('-override=' + ','.join(f'{k}={v}' for k, v in overrides.items()))
        out = subprocess.run(cmd, cwd=self.workdir, capture_output=True, text=True)
        _check_overrides(out.stdout)
        if not os.path.exists(lin):
            raise RuntimeError('linearisation produced no linearized_model.py (build with linearize_python=True)')
        ns = {}
        exec(open(lin).read(), ns)
        n, m, p, x0, u0, A, B, C, D, states, inputs, outputs = ns['linearized_model']()
        res = OMResult(os.path.join(self.workdir, '_lin_res.mat'))
        return dict(A=np.array(A, float), x0=np.array(x0, float), states=list(states), result=res, t_lin=t_lin)

    def modes(self, t_lin, overrides=None):
        """Linearise and return a Modes object (eigenfrequencies, damping, mode shapes)."""
        lin = self.linearize(t_lin, overrides)
        state = contact_state(lin['result'])
        open_ = [k for k, v in state.items() if v in ('separated', 'free in clearance')]
        if open_:
            warnings.warn(f'at t_lin = {t_lin} s these contacts are OPEN and do not appear in the modes: {open_}. '
                          'Choose a time where the system is loaded (contacts closed).')
        modes = Modes.from_linearization(lin['A'], lin['states'], lin['result'])
        modes.contact_state = state
        modes.t_lin = t_lin
        return modes


def _check_overrides(stdout):
    """OpenModelica silently keeps the old value when a parameter cannot be overridden
    (it is bound to another parameter, or evaluated at compile time). Make that loud."""
    bad = re.findall(r'not possible to override the following quantity: (\S+)', stdout)
    if bad:
        raise ValueError(f'OpenModelica ignored overrides of {bad}. Expose them as top-level parameters '
                         'of the model (e.g. parameter Real clr_idler; ... bearing(clr=clr_idler)) and override those.')


# =============================================================================
# 3. Signal tools
# =============================================================================
def _window_mask(t, window):
    if window is None:
        return np.ones_like(t, bool)
    t0 = -np.inf if window[0] is None else window[0]
    t1 = np.inf if window[1] is None else window[1]
    return (t >= t0) & (t <= t1)


def _uniform(t, y, n=None):
    """Remove duplicate event points and resample on a uniform grid."""
    t, i = np.unique(t, return_index=True)
    y = y[i]
    n = n or len(t)
    tu = np.linspace(t[0], t[-1], n)
    return tu, np.interp(tu, t, y)


def spectrum(t, y, window=None, detrend=True):
    """Single-sided amplitude spectrum vs frequency [Hz] (Hann window, amplitude-corrected)."""
    m = _window_mask(t, window)
    tu, yu = _uniform(t[m], y[m])
    if detrend:
        yu = yu - yu.mean()
    w = np.hanning(len(yu))
    Y = np.fft.rfft(yu*w)
    f = np.fft.rfftfreq(len(yu), tu[1]-tu[0])
    return f, 2*np.abs(Y)/w.sum()


def order_spectrum(t, y, ref_angle, window=None, samples_per_rev=512, detrend=True):
    """Order spectrum: resample y on a uniform grid of the reference ANGLE [rad]
    (e.g. driver or crank angle), then FFT.  Order 1 = once per reference revolution.
    Robust against speed fluctuations, the standard tool for rotating machinery."""
    m = _window_mask(t, window)
    t_, i = np.unique(t[m], return_index=True)
    ang = np.abs(ref_angle[m][i] - ref_angle[m][i][0])
    yy = y[m][i]
    o = np.argsort(ang)
    ang, yy = ang[o], yy[o]
    nrev = ang[-1]/(2*np.pi)
    if nrev < 1:
        warnings.warn('order spectrum over less than one reference revolution')
    n = max(64, int(nrev*samples_per_rev))
    ag = np.linspace(0, ang[-1], n)
    yg = np.interp(ag, ang, yy)
    if detrend:
        yg = yg - yg.mean()
    w = np.hanning(n)
    Y = np.fft.rfft(yg*w)
    orders = np.fft.rfftfreq(n, (ag[1]-ag[0])/(2*np.pi))
    return orders, 2*np.abs(Y)/w.sum()


def _onsets(mask):
    """Number of False->True transitions (contact onsets, impacts)."""
    mask = np.asarray(mask, bool)
    return int(np.sum(~mask[:-1] & mask[1:]))


def _frac(mask, t):
    """Time fraction where mask is True (time weighted, robust to non-uniform output)."""
    dt = np.diff(t)
    return float(np.sum(dt*mask[:-1])/max(np.sum(dt), 1e-30))


def _stats(y, t, prefix, unit, scale=1.0):
    y = y*scale
    dt = np.diff(t)
    mean = float(np.sum(dt*0.5*(y[1:]+y[:-1]))/max(np.sum(dt), 1e-30))
    rms_dyn = float(np.sqrt(np.sum(dt*(0.5*(y[1:]+y[:-1])-mean)**2)/max(np.sum(dt), 1e-30)))
    return {f'{prefix}_mean_{unit}': mean, f'{prefix}_max_{unit}': float(y.max()),
            f'{prefix}_min_{unit}': float(y.min()), f'{prefix}_pp_{unit}': float(y.max()-y.min()),
            f'{prefix}_rms_dyn_{unit}': rms_dyn}


# =============================================================================
# 4. Component registry
# =============================================================================
@dataclass
class ComponentType:
    name: str
    signature: tuple          # relative variable names that must exist
    exclude: tuple = ()       # relative names that must NOT exist
    metrics: callable = None  # f(comp, ctx) -> dict
    plot: callable = None     # f(comp, ctx, ax_list) -> None
    nplots: int = 1
    group: str = 'other'      # 'gear', 'bearing', 'support', 'torsion', 'body', 'load', 'source'
    priority: int = 0


REGISTRY: list[ComponentType] = []


def register(ctype: ComponentType):
    """Register a component type (new Modelica components -> add a signature + metrics)."""
    REGISTRY[:] = [c for c in REGISTRY if c.name != ctype.name] + [ctype]
    REGISTRY.sort(key=lambda c: -c.priority)


@dataclass
class Component:
    name: str
    ctype: ComponentType
    res: OMResult

    def __getitem__(self, rel):
        return self.res[f'{self.name}.{rel}']

    def has(self, rel):
        return self.res.has(f'{self.name}.{rel}')

    def p(self, rel, default=np.nan):
        return self.res.value(f'{self.name}.{rel}') if self.has(rel) else default


@dataclass
class Context:
    """What metrics/plots need besides the component: time window, x-axis, reference angle."""
    t: np.ndarray
    mask: np.ndarray
    x: np.ndarray            # x-axis for plots (reference angle in deg or time)
    xlabel: str
    ref_angle: np.ndarray    # rad, for order spectra (None -> time spectra)
    ref_name: str


# =============================================================================
# 5. System: discovery + report
# =============================================================================
class System:
    """All recognised components of one simulation result."""

    def __init__(self, res: OMResult, reference=None):
        """reference: name of the angle variable used as x-axis and for order
        spectra (e.g. 'driver.frame.phi' or 'crankGear.phi').  Default: the
        first ConstantSpeed source found, else the time axis."""
        self.res = res
        self.components: list[Component] = []
        taken = set()
        inst = sorted(res.instances(), key=lambda s: (s.count('.'), s))
        for ctype in REGISTRY:
            for name in inst:
                if name in taken:
                    continue
                if all(res.has(f'{name}.{v}') for v in ctype.signature) and \
                        not any(res.has(f'{name}.{v}') for v in ctype.exclude):
                    # skip connector sub-instances (e.g. 'mesh.frame_a')
                    if re.search(r'\.frame(_[a-z])?$', name):
                        continue
                    self.components.append(Component(name, ctype, res))
                    taken.add(name)
        self.components.sort(key=lambda c: (c.ctype.group, c.name))
        self.reference = reference
        if reference is None:
            src = self.by_type('ConstantSpeed')
            if src:
                self.reference = f'{src[0].name}.frame.phi'

    def by_type(self, tname):
        return [c for c in self.components if c.ctype.name == tname]

    def by_group(self, group):
        return [c for c in self.components if c.ctype.group == group]

    def __getitem__(self, name):
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)

    def summary(self):
        rows = [f'{c.name:28s} {c.ctype.name:24s} ({c.ctype.group})' for c in self.components]
        ref = self.reference or 'time'
        return f'{len(self.components)} components, reference = {ref}\n' + '\n'.join(rows)

    def context(self, window=None):
        t = self.res.time
        m = _window_mask(t, window)
        if self.reference and self.res.has(self.reference):
            ang = self.res[self.reference]
            return Context(t, m, np.rad2deg(np.abs(ang - ang[0])), f'{self.reference} [deg]', ang, self.reference)
        return Context(t, m, t, 'time [s]', None, 'time')

    def report(self, window=None, revs=None):
        """Compute metrics for all components.
        window: (t0, t1) in s;  revs: (r0, r1) in reference revolutions (alternative)."""
        if revs is not None and self.reference:
            ang = np.abs(self.res[self.reference] - self.res[self.reference][0])/(2*np.pi)
            t = self.res.time
            t0 = None if revs[0] is None else float(np.interp(revs[0], ang, t))
            t1 = None if revs[1] is None else float(np.interp(revs[1], ang, t))
            window = (t0, t1)
        ctx = self.context(window)
        return Report(self, ctx, window)


class Report:
    """Metrics of all components (+ figures on demand)."""

    def __init__(self, system: System, ctx: Context, window):
        self.system = system
        self.ctx = ctx
        self.window = window
        self.metrics = {}
        for c in system.components:
            if c.ctype.metrics is None:
                continue
            try:
                self.metrics[c.name] = {'type': c.ctype.name, **c.ctype.metrics(c, ctx)}
            except Exception as e:  # a broken component must not kill the report
                warnings.warn(f'metrics for {c.name} failed: {e}')

    # -- tables ---------------------------------------------------------------
    def table(self, ctype=None):
        """Metrics as a pandas DataFrame (one row per component; one type -> clean columns)."""
        rows = {k: v for k, v in self.metrics.items() if ctype is None or v['type'] == ctype}
        if pd is None:
            return rows
        return pd.DataFrame(rows).T

    def tables(self):
        types = sorted({v['type'] for v in self.metrics.values()})
        return {t: self.table(t).drop(columns='type').astype(float) for t in types}

    def to_csv(self, folder):
        os.makedirs(folder, exist_ok=True)
        for t, df in self.tables().items():
            df.to_csv(os.path.join(folder, f'metrics_{t}.csv'))

    # -- figures ----------------------------------------------------------------
    def figures(self, groups=None):
        """One figure per component that has a plot function. Returns {name: fig}."""
        figs = {}
        for c in self.system.components:
            if c.ctype.plot is None or (groups and c.ctype.group not in groups):
                continue
            n = c.ctype.nplots
            fig, axs = plt.subplots(1, n, figsize=(5.2*n, 3.4), squeeze=False)
            try:
                c.ctype.plot(c, self.ctx, list(axs[0]))
                fig.suptitle(f'{c.name}  ({c.ctype.name})', fontsize=10)
                fig.tight_layout()
                figs[c.name] = fig
            except Exception as e:
                plt.close(fig)
                warnings.warn(f'plot for {c.name} failed: {e}')
        return figs

    def torsional_overview(self):
        """System-level figure: speed fluctuation of every rotating body + torques of all torsion elements."""
        ctx = self.ctx
        bodies = [c for c in self.system.by_type('Body')]
        tors = self.system.by_type('TorsionSpringDamper')
        fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))
        for c in bodies:
            w = c['w']; m = ctx.mask
            wm = np.mean(w[m])
            axs[0].plot(ctx.x[m], (w[m]-wm)/abs(wm)*100 if abs(wm) > 1e-9 else w[m], lw=.7, label=c.name)
        axs[0].set_ylabel('speed deviation from mean [%]'); axs[0].set_xlabel(ctx.xlabel); axs[0].legend(fontsize=7)
        for c in tors:
            axs[1].plot(ctx.x[ctx.mask], c['tau'][ctx.mask], lw=.7, label=c.name)
        axs[1].set_ylabel('torque [N m]'); axs[1].set_xlabel(ctx.xlabel); axs[1].legend(fontsize=7)
        for c in tors:
            if ctx.ref_angle is not None:
                o, a = order_spectrum(ctx.t, c['tau'], ctx.ref_angle, (ctx.t[ctx.mask][0], ctx.t[ctx.mask][-1]))
                axs[2].plot(o, a, lw=.8, label=c.name); axs[2].set_xlabel(f'order of {ctx.ref_name}')
            else:
                f, a = spectrum(ctx.t, c['tau'], (ctx.t[ctx.mask][0], ctx.t[ctx.mask][-1]))
                axs[2].plot(f, a, lw=.8, label=c.name); axs[2].set_xlabel('frequency [Hz]')
        axs[2].set_xlim(0, 40 if ctx.ref_angle is not None else None)
        axs[2].set_ylabel('torque amplitude [N m]'); axs[2].legend(fontsize=7)
        fig.suptitle('Torsional overview', fontsize=10); fig.tight_layout()
        return fig

    def force_overview(self):
        """System-level bar chart: peak resultant force of every force-carrying component."""
        names, peak, mean = [], [], []
        for n, m in self.metrics.items():
            key = next((k for k in m if k.startswith('F_res_max_N') or k == 'F_tooth_max_N'), None)
            if key:
                names.append(n); peak.append(m[key])
                mean.append(abs(m.get('F_res_mean_N', m.get('F_normal_mean_N', np.nan))))
        fig, ax = plt.subplots(figsize=(max(6, 0.9*len(names)+2), 3.6))
        x = np.arange(len(names))
        ax.bar(x-0.2, peak, 0.4, label='peak'); ax.bar(x+0.2, mean, 0.4, label='mean')
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right'); ax.set_ylabel('force [N]')
        ax.legend(); ax.set_title('Peak and mean force per component (analysis window)', fontsize=10)
        fig.tight_layout()
        return fig

    # -- export -----------------------------------------------------------------
    def to_html(self, path, title=None, modes=None, extra_figs=None):
        """Single self-contained HTML report (tables + embedded PNG figures)."""
        def img(fig):
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90); plt.close(fig)
            return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'
        res = self.system.res
        w = self.window or (None, None)
        h = [f'<html><head><meta charset="utf-8"><title>{title or "Post-processing report"}</title>',
             '<style>body{font-family:sans-serif;max-width:1400px;margin:auto;padding:20px}'
             'table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}'
             'th{background:#eee}img{max-width:100%}h2{border-bottom:1px solid #999}</style></head><body>',
             f'<h1>{title or "Post-processing report"}</h1>',
             f'<p>Result: <code>{os.path.basename(res.path)}</code> &middot; t = {res.time[0]:g}..{res.time[-1]:g} s &middot; '
             f'analysis window: {w[0]}..{w[1]} s &middot; reference: {self.ctx.ref_name}</p>',
             f'<h2>Components found</h2><pre>{self.system.summary()}</pre>']
        for t, df in self.tables().items():
            h.append(f'<h2>{t}</h2>' + df.T.to_html(float_format=lambda v: f'{v:.4g}'))
        h.append('<h2>System overview</h2>' + img(self.force_overview()))
        if self.system.by_type('Body'):
            h.append(img(self.torsional_overview()))
        if modes is not None:
            h.append('<h2>Modes (linearised)</h2>' + modes.table().to_html(float_format=lambda v: f'{v:.4g}'))
            h.append(img(modes.plot_energy()))
        h.append('<h2>Components</h2>')
        for n, fig in self.figures().items():
            h.append(f'<h3>{n}</h3>' + img(fig))
        for f in (extra_figs or []):
            h.append(img(f))
        h.append('</body></html>')
        with open(path, 'w') as fh:
            fh.write('\n'.join(h))
        return path


def sweep(runner: OMRunner, param, values, window=None, revs=None, overrides=None, keys=None):
    """Run the compiled model for several values of one top-level parameter.
    Returns (table, reports): table has one column per value, rows = (component, metric)."""
    reports, labels = [], []
    for v in values:
        ov = dict(overrides or {}); ov[param] = v
        res = runner.simulate(overrides=ov, result=f'sweep_{param}_{v:g}.mat')
        reports.append(System(res).report(window=window, revs=revs)); labels.append(v)
    return compare_reports(reports, labels, keys), reports


def compare_reports(reports, labels, keys=None):
    """Side-by-side table of selected metrics for several runs (e.g. a parameter sweep).
    keys: list of (component, metric) or None for everything."""
    data = {}
    for rep, lab in zip(reports, labels):
        col = {}
        for comp, m in rep.metrics.items():
            for k, v in m.items():
                if k == 'type':
                    continue
                if keys is None or (comp, k) in keys:
                    col[(comp, k)] = v
        data[lab] = col
    return pd.DataFrame(data) if pd is not None else data


# =============================================================================
# 6. Modal analysis from the linearised model
# =============================================================================
class Modes:
    """Eigenfrequencies, damping and mode shapes from the state matrix A.

    Mode classification uses the kinetic-energy distribution of each mode over
    the degrees of freedom (m*v^2 for translations, J*w^2 for rotations):
    'torsional' if rotations hold most of the energy, else 'lateral'."""

    def __init__(self, freqs, zetas, shapes, dofs, energy):
        self.f = freqs; self.zeta = zetas; self.shapes = shapes; self.dofs = dofs; self.energy = energy

    @classmethod
    def from_linearization(cls, A, states, res: OMResult | None = None):
        ev, V = np.linalg.eig(A)
        keep = ev.imag > 1e-6                       # one of each conjugate pair, oscillatory only
        ev, V = ev[keep], V[:, keep]
        o = np.argsort(ev.imag); ev, V = ev[o], V[:, o]
        f = ev.imag/(2*np.pi)
        zeta = -ev.real/np.abs(ev)
        # displacement states and their velocity states: names like gear1_u / gear1_du, gear1_phi / gear1_w
        names = list(states)
        # OpenModelica writes state names with '.' replaced by '_': map back to instances
        inst_map = {}
        if res is not None:
            for i_ in res.instances():
                inst_map[i_.replace('.', '_')] = i_

        def par(inst, p):
            full = inst_map.get(inst, inst)
            return res.value(f'{full}.{p}') if res is not None and res.has(f'{full}.{p}') else 1.0
        disp, vel, inertia, kinds = [], [], [], []
        for i, s in enumerate(names):
            inst, _, var = s.rpartition('_')
            if var in ('u', 'v'):
                vname, kind, iner = f'{inst}_d{var}', 'trans', par(inst, 'm')
            elif var == 'phi':
                vname, kind, iner = f'{inst}_w', 'rot', par(inst, 'J')
            else:
                continue
            if vname in names:
                disp.append(s); vel.append(names.index(vname)); inertia.append(iner); kinds.append(kind)
        Vv = V[vel, :]
        E = (np.array(inertia)[:, None]*np.abs(Vv)**2)
        E = E/E.sum(axis=0, keepdims=True)
        shapes = V[[names.index(d) for d in disp], :]
        dofs = [(inst_map.get(d.rpartition('_')[0], d.rpartition('_')[0]) + '.' + d.rpartition('_')[2], k)
                for d, k in zip(disp, kinds)]
        return cls(f, zeta, shapes, dofs, E)

    def table(self, top=3):
        rows = []
        for j in range(len(self.f)):
            e = self.energy[:, j]
            rot = sum(e[i] for i, (d, k) in enumerate(self.dofs) if k == 'rot')
            o = np.argsort(e)[::-1][:top]
            rows.append({'f_Hz': self.f[j], 'zeta_pct': 100*self.zeta[j],
                         'type': 'torsional' if rot > 0.5 else 'lateral', 'torsional_energy_pct': 100*rot,
                         'dominant_dofs': ', '.join(f'{self.dofs[i][0]} {100*e[i]:.0f}%' for i in o)})
        return pd.DataFrame(rows) if pd is not None else rows

    def plot_energy(self, max_modes=12):
        n = min(max_modes, len(self.f))
        fig, ax = plt.subplots(figsize=(max(7, 0.7*n+3), 0.28*len(self.dofs)+1.8))
        im = ax.imshow(self.energy[:, :n]*100, aspect='auto', cmap='viridis', vmin=0, vmax=100)
        ax.set_yticks(range(len(self.dofs))); ax.set_yticklabels([d for d, k in self.dofs], fontsize=8)
        ax.set_xticks(range(n)); ax.set_xticklabels([f'{v:.0f}' for v in self.f[:n]], rotation=45, fontsize=8)
        ax.set_xlabel('mode frequency [Hz]'); ax.set_title('Kinetic-energy distribution per mode [%]', fontsize=10)
        plt.colorbar(im, ax=ax); fig.tight_layout()
        return fig


def contact_state(res: OMResult, t=None):
    """State of every gear mesh / bearing contact at time t (default: end of the run).
    Linearised modes are only meaningful for the contact state at the linearisation time."""
    sysm = System(res)
    i = -1 if t is None else int(np.searchsorted(res.time, t))
    out = {}
    for c in sysm.by_type('GearMesh2D'):
        a, b = c['dA'][i] > 0, c['dB'][i] > 0
        out[c.name] = 'flanks A+B' if a and b else ('flank A' if a else ('flank B' if b else 'separated'))
    for c in sysm.by_type('RadialBearingClearance'):
        out[c.name] = 'in contact' if c['r'][i] > c.p('clr') else 'free in clearance'
    return out


def campbell(modes: Modes, orders, rpm=(0, 10000), speed_ratio=1.0, fmax=None, ax=None, only=None):
    """Campbell diagram: eigenfrequencies (horizontal) vs excitation orders
    f = order * rpm/60 * speed_ratio.  only='torsional' or 'lateral' filters the modes.
    Returns (fig, table of resonance speeds inside the rpm range)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    r = np.linspace(*rpm, 200)
    fmax = fmax or 1.25*max(orders)*rpm[1]/60*speed_ratio
    kinds = list(modes.table()['type']) if pd is not None else ['?']*len(modes.f)
    sel = [j for j in range(len(modes.f)) if only is None or kinds[j] == only]
    rows = []
    for j in sel:
        f = modes.f[j]
        if f <= fmax:
            ax.axhline(f, color='C0', lw=.8, alpha=.7)
            ax.text(rpm[1], f, f' {f:.0f} Hz', va='center', fontsize=7)
    for o in orders:
        ax.plot(r, o*r/60*speed_ratio, color='C3', lw=.8)
        ax.text(r[-1], min(o*r[-1]/60*speed_ratio, fmax), f'{o:g}', color='C3', fontsize=7)
        for j in sel:
            f = modes.f[j]
            n_res = f*60/(o*speed_ratio)
            if rpm[0] <= n_res <= rpm[1]:
                rows.append({'order': o, 'f_Hz': f, 'mode_type': kinds[j], 'rpm_resonance': n_res})
    ax.set_ylim(0, fmax); ax.set_xlim(*rpm); ax.set_xlabel('speed [rpm]'); ax.set_ylabel('frequency [Hz]')
    ax.set_title('Campbell diagram (linearised, contact closed)', fontsize=10)
    tab = pd.DataFrame(rows).sort_values('rpm_resonance') if (pd is not None and rows) else rows
    return fig, tab


# =============================================================================
# 7. Metrics and plots per component type
# =============================================================================
def _x(ctx):
    return ctx.x[ctx.mask]


# ---- GearMesh2D ------------------------------------------------------------------
def gear_metrics(c, ctx):
    m, t = ctx.mask, ctx.t[ctx.mask]
    FA, FB, dA, dB = c['FA'][m], c['FB'][m], c['dA'][m], c['dB'][m]
    rb1, rb2, alpha = c.p('rb1'), c.p('rb2'), c.p('alpha')
    cA, cB = dA > 0, dB > 0
    Fn = FA - FB                                   # signed normal force (A positive)
    out = {}
    out.update(_stats(Fn, t, 'F_normal', 'N'))
    out['F_tooth_max_N'] = float(max(FA.max(), FB.max()))
    out['F_A_max_N'] = float(FA.max()); out['F_B_max_N'] = float(FB.max())
    Fm = abs(out['F_normal_mean_N'])
    out['dyn_factor_Kv'] = out['F_tooth_max_N']/Fm if Fm > 1e-9 else np.nan
    out['contact_A_pct'] = 100*_frac(cA, t); out['contact_B_pct'] = 100*_frac(cB, t)
    out['separated_pct'] = 100*_frac(~cA & ~cB, t); out['double_flank_pct'] = 100*_frac(cA & cB, t)
    out['impacts_A'] = _onsets(cA); out['impacts_B'] = _onsets(cB)
    # flank changes: A -> B or B -> A through a separation
    side = np.where(cA, 1, np.where(cB, -1, 0)); s = side[side != 0]
    out['flank_changes'] = int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0
    out['penetration_max_um'] = 1e6*float(max(dA.max(), dB.max(), 0))
    jn = c.p('jn')
    out['backlash_n_um'] = 1e6*jn
    out['free_play_max_um'] = 1e6*float(max((-dA).max(), (-dB).max()))
    # dynamic transmission error along the line of action (rotation part only)
    if c.has('frame_a.phi') and c.has('frame_b.phi'):
        dte = rb1*c['frame_a.phi'][m] + rb2*c['frame_b.phi'][m]
        out['DTE_pp_um'] = 1e6*float(dte.max()-dte.min())
        out['DTE_rms_dyn_um'] = 1e6*float(np.std(dte))
    # damping energy in the mesh
    if c.has('c'):
        cc = c.p('c')
        tt, ddA = np.unique(ctx.t[m], return_index=True)
        vA = np.gradient(dA[ddA], tt); vB = np.gradient(dB[ddA], tt)
        P = cc*(vA**2*(dA[ddA] > 0) + vB**2*(dB[ddA] > 0))
        E = float(_trapz(P, tt))
        out['damping_energy_J'] = E; out['damping_power_mean_W'] = E/max(tt[-1]-tt[0], 1e-30)
    return out


def gear_plot(c, ctx, axs):
    m, x = ctx.mask, _x(ctx)
    axs[0].plot(x, c['FA'][m], lw=.7, label='F_A'); axs[0].plot(x, c['FB'][m], lw=.7, label='F_B')
    axs[0].set_ylabel('tooth force [N]'); axs[0].legend(fontsize=7); axs[0].set_xlabel(ctx.xlabel)
    ca = np.cos(c.p('alpha'))
    axs[1].plot(x, -c['dA'][m]/ca*1e3, lw=.7, label='lash A'); axs[1].plot(x, -c['dB'][m]/ca*1e3, lw=.7, label='lash B')
    axs[1].axhline(0, color='k', lw=.5); axs[1].set_ylabel('lash on pitch circle [mm]'); axs[1].legend(fontsize=7)
    axs[1].set_xlabel(ctx.xlabel)
    Ft = (c['FA'] - c['FB'])*ca
    if ctx.ref_angle is not None:
        o, a = order_spectrum(ctx.t, Ft, ctx.ref_angle, (ctx.t[m][0], ctx.t[m][-1]))
        axs[2].plot(o, a, lw=.8); axs[2].set_xlabel(f'order of {ctx.ref_name}')
        zmax = max(c.p('z1', 40), c.p('z2', 40))
        axs[2].set_xlim(0, 2.2*zmax)
    else:
        f, a = spectrum(ctx.t, Ft, (ctx.t[m][0], ctx.t[m][-1])); axs[2].plot(f, a, lw=.8); axs[2].set_xlabel('frequency [Hz]')
    axs[2].set_ylabel('tangential force amplitude [N]'); axs[2].set_yscale('log')


# ---- RadialBearingClearance -----------------------------------------------------------
def bearing_metrics(c, ctx):
    m, t = ctx.mask, ctx.t[ctx.mask]
    Fx, Fy = c['Fx'][m], c['Fy'][m]               # force on the outer ring (gear)
    F = np.hypot(Fx, Fy)
    r, clr = c['r'][m], c.p('clr')
    out = {}
    out.update(_stats(F, t, 'F_res', 'N'))
    out.update(_stats(Fx, t, 'Fx', 'N')); out.update(_stats(Fy, t, 'Fy', 'N'))
    i = int(np.argmax(F))
    out['F_res_max_dir_deg'] = float(np.rad2deg(np.arctan2(-Fy[i], -Fx[i])))   # direction of load on the shaft
    out['clearance_um'] = 1e6*clr
    out['eccentricity_max_um'] = 1e6*float(r.max())
    out['clearance_use_max_pct'] = 100*float(r.max()/clr) if clr > 0 else np.nan
    contact = r > clr
    out['contact_pct'] = 100*_frac(contact, t); out['impacts'] = _onsets(contact)
    out['penetration_max_um'] = 1e6*float(max((r-clr).max(), 0))
    if c.has('Tf') and c.has('frame_b.phi'):
        tt, i_ = np.unique(t, return_index=True)
        w = np.gradient(c['frame_b.phi'][m][i_], tt)
        P = -c['Tf'][m][i_]*w                       # friction power loss >= 0
        out['friction_power_mean_W'] = float(_trapz(P, tt)/max(tt[-1]-tt[0], 1e-30))
    return out


def bearing_plot(c, ctx, axs):
    m, x = ctx.mask, _x(ctx)
    Fx, Fy = c['Fx'][m], c['Fy'][m]
    axs[0].plot(x, Fx, lw=.7, label='Fx'); axs[0].plot(x, Fy, lw=.7, label='Fy'); axs[0].plot(x, np.hypot(Fx, Fy), 'k', lw=.6, label='|F|')
    axs[0].set_ylabel('force on outer ring [N]'); axs[0].legend(fontsize=7); axs[0].set_xlabel(ctx.xlabel)
    axs[1].plot(-Fx, -Fy, lw=.5); axs[1].set_aspect('equal', 'datalim'); axs[1].set_title('load orbit on shaft [N]', fontsize=9)
    axs[1].set_xlabel('Fx'); axs[1].set_ylabel('Fy')
    rx, ry, clr = c['rx'][m]*1e6, c['ry'][m]*1e6, c.p('clr')*1e6
    axs[2].plot(rx, ry, lw=.5); th = np.linspace(0, 2*np.pi, 200)
    axs[2].plot(clr*np.cos(th), clr*np.sin(th), 'r--', lw=.8, label='clearance')
    axs[2].set_aspect('equal', 'datalim'); axs[2].set_title('journal orbit [um]', fontsize=9); axs[2].legend(fontsize=7)


# ---- supports: NonlinearSupport2D (Fx, Fy) and SpringDamper2D (fx, fy) -----------------------------
def _support_common(c, ctx, Fx, Fy, ux, uy):
    m, t = ctx.mask, ctx.t[ctx.mask]
    F = np.hypot(Fx[m], Fy[m])
    out = {}
    out.update(_stats(F, t, 'F_res', 'N'))
    out.update(_stats(Fx[m], t, 'Fx', 'N')); out.update(_stats(Fy[m], t, 'Fy', 'N'))
    if ux is not None:
        out['disp_x_pp_um'] = 1e6*float(np.ptp(ux[m])); out['disp_y_pp_um'] = 1e6*float(np.ptp(uy[m]))
        out['disp_res_max_um'] = 1e6*float(np.hypot(ux[m], uy[m]).max())
    return out


def nlsupport_metrics(c, ctx):
    ux = c['frame.u'] if c.has('frame.u') else None
    uy = c['frame.v'] if c.has('frame.v') else None
    out = _support_common(c, ctx, c['Fx'], c['Fy'], ux, uy)
    return out


def spring2d_metrics(c, ctx):
    ux = c['frame_b.u'] - c['frame_a.u'] if c.has('frame_b.u') and c.has('frame_a.u') else None
    uy = c['frame_b.v'] - c['frame_a.v'] if c.has('frame_b.v') and c.has('frame_a.v') else None
    return _support_common(c, ctx, c['fx'], c['fy'], ux, uy)


def _support_plot(c, ctx, axs, Fx, Fy):
    m, x = ctx.mask, _x(ctx)
    axs[0].plot(x, Fx[m], lw=.7, label='Fx'); axs[0].plot(x, Fy[m], lw=.7, label='Fy')
    axs[0].set_ylabel('support force [N]'); axs[0].legend(fontsize=7); axs[0].set_xlabel(ctx.xlabel)
    axs[1].plot(Fx[m], Fy[m], lw=.5); axs[1].set_aspect('equal', 'datalim'); axs[1].set_title('force orbit [N]', fontsize=9)


def nlsupport_plot(c, ctx, axs):
    _support_plot(c, ctx, axs, c['Fx'], c['Fy'])


def spring2d_plot(c, ctx, axs):
    _support_plot(c, ctx, axs, c['fx'], c['fy'])


# ---- TorsionSpringDamper -------------------------------------------------------------
def torsion_metrics(c, ctx):
    m, t = ctx.mask, ctx.t[ctx.mask]
    tau = c['tau'][m]
    out = _stats(tau, t, 'T', 'Nm')
    k = c.p('k')
    out['twist_mean_deg'] = float(np.rad2deg(out['T_mean_Nm']/k)) if k > 0 else np.nan
    out['twist_pp_deg'] = float(np.rad2deg(out['T_pp_Nm']/k)) if k > 0 else np.nan
    out['T_dyn_factor'] = out['T_max_Nm']/abs(out['T_mean_Nm']) if abs(out['T_mean_Nm']) > 1e-9 else np.nan
    if ctx.ref_angle is not None:
        o, a = order_spectrum(ctx.t, c['tau'], ctx.ref_angle, (t[0], t[-1]))
        sel = (o > 0.25) & (o < 50)
        if sel.any():
            k_ = np.argmax(a[sel])
            out['dominant_order'] = float(o[sel][k_]); out['dominant_order_amp_Nm'] = float(a[sel][k_])
    return out


def torsion_plot(c, ctx, axs):
    m, x = ctx.mask, _x(ctx)
    axs[0].plot(x, c['tau'][m], lw=.7); axs[0].set_ylabel('torque [N m]'); axs[0].set_xlabel(ctx.xlabel)
    if ctx.ref_angle is not None:
        o, a = order_spectrum(ctx.t, c['tau'], ctx.ref_angle, (ctx.t[m][0], ctx.t[m][-1]))
        axs[1].bar(o, a, width=0.4*(o[1]-o[0])*4); axs[1].set_xlim(0, 30); axs[1].set_xlabel(f'order of {ctx.ref_name}')
    else:
        f, a = spectrum(ctx.t, c['tau'], (ctx.t[m][0], ctx.t[m][-1])); axs[1].plot(f, a); axs[1].set_xlabel('frequency [Hz]')
    axs[1].set_ylabel('torque amplitude [N m]')


# ---- Body / PointMass ---------------------------------------------------------------------
def body_metrics(c, ctx):
    m, t = ctx.mask, ctx.t[ctx.mask]
    out = {}
    if c.has('w'):
        w = c['w'][m]
        wm = float(np.mean(w))
        out['speed_mean_rpm'] = wm*30/np.pi
        out['speed_pp_rpm'] = float(np.ptp(w))*30/np.pi
        out['irregularity_pct'] = 100*float(np.ptp(w)/abs(wm)) if abs(wm) > 1e-9 else np.nan
        tt, i_ = np.unique(t, return_index=True)
        acc = np.gradient(w[i_], tt)
        out['ang_acc_max_rad_s2'] = float(np.max(np.abs(acc)))
        if c.has('J'):
            out['inertia_torque_max_Nm'] = float(np.max(np.abs(acc))*c.p('J'))
    u, v = c['u'][m], c['v'][m]
    out['disp_res_max_um'] = 1e6*float(np.hypot(u, v).max())
    out['disp_x_pp_um'] = 1e6*float(np.ptp(u)); out['disp_y_pp_um'] = 1e6*float(np.ptp(v))
    return out


def body_plot(c, ctx, axs):
    m, x = ctx.mask, _x(ctx)
    if c.has('w'):
        w = c['w'][m]; axs[0].plot(x, w*30/np.pi, lw=.7); axs[0].set_ylabel('speed [rpm]')
    axs[0].set_xlabel(ctx.xlabel)
    axs[1].plot(c['u'][m]*1e6, c['v'][m]*1e6, lw=.5); axs[1].set_aspect('equal', 'datalim')
    axs[1].set_title('centre orbit [um]', fontsize=9)


# ---- loads ----------------------------------------------------------------------------------
def load_metrics(c, ctx):
    m, t = ctx.mask, ctx.t[ctx.mask]
    return _stats(c['T'][m] if c.has('T') else -c['frame.t'][m], t, 'T', 'Nm')


# ---- registration (priority: more specific signatures first) -------------------------------------
register(ComponentType('GearMesh2D', ('FA', 'FB', 'dA', 'dB', 'rb1', 'rb2'), metrics=gear_metrics, plot=gear_plot,
                       nplots=3, group='gear', priority=100))
register(ComponentType('RadialBearingClearance', ('Fn', 'Fx', 'Fy', 'rx', 'ry', 'r', 'clr'), metrics=bearing_metrics,
                       plot=bearing_plot, nplots=3, group='bearing', priority=90))
register(ComponentType('NonlinearSupport2D', ('Fx', 'Fy', 'dampScale'), exclude=('Fn',), metrics=nlsupport_metrics,
                       plot=nlsupport_plot, nplots=2, group='support', priority=80))
register(ComponentType('SpringDamper2D', ('fx', 'fy', 'kx'), metrics=spring2d_metrics, plot=spring2d_plot,
                       nplots=2, group='support', priority=70))
register(ComponentType('TorsionSpringDamper', ('tau', 'k', 'd'), metrics=torsion_metrics, plot=torsion_plot,
                       nplots=2, group='torsion', priority=60))
register(ComponentType('Body', ('u', 'v', 'phi', 'w', 'm', 'J'), metrics=body_metrics, plot=body_plot,
                       nplots=2, group='body', priority=50))
register(ComponentType('PointMass', ('u', 'v', 'du', 'dv', 'm'), exclude=('w',), metrics=body_metrics, plot=body_plot,
                       nplots=2, group='body', priority=40))
register(ComponentType('SineAngleTorque', ('T', 'A', 'Omega'), metrics=load_metrics, group='load', priority=30))
register(ComponentType('ConstantTorque', ('T', 'frame.t'), exclude=('A',), metrics=load_metrics, group='load', priority=20))
register(ComponentType('ConstantSpeed', ('Omega', 'frame.phi'), exclude=('A',), group='source', priority=10))
