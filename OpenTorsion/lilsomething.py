import numpy as np
import opentorsion as ot

def generator_torque(rpm):
    rated_T = 2.9e6
    if rpm < 4:
        return 0
    elif rpm < 15:
        m = (0.5 - 0.125) / (15 - 4) * rated_T
        b = 0.5 * rated_T - m * 15
        return m * rpm + b
    elif rpm < 22:
        P = rated_T * 15
        return P / rpm
    else:
        return 0

def get_windmill_excitation(rpm):
    f_s = rpm
    vs = np.array([4, 6, 8, 10, 12, 14, 16])
    omegas = 2 * np.pi * vs * f_s
    rated_T = 2.9e6
    amplitudes = np.array([0.0018, 0.0179, 0.0024, 0.0034, 0.0117, 0.0018, 0.0011]) * generator_torque(rpm)
    amplitudes[4] += rated_T * 0.0176
    return omegas, amplitudes

def _vibratory_torque_compat(assembly, shafts, excitations, omegas, k_shafts, C):
    """Call vibratory_torque across versions; fall back to ss_response if needed."""
    # Try: new API (with C)
    try:
        return assembly.vibratory_torque(excitations, omegas, k_shafts, C)
    except TypeError:
        pass
    # Try: older API (no C)
    try:
        return assembly.vibratory_torque(excitations, omegas, k_shafts)
    except TypeError:
        pass
    # Fallback: compute from steady-state response
    try:
        U, _ = assembly.ss_response(excitations, omegas, C)
    except TypeError:
        U, _ = assembly.ss_response(excitations, omegas)  # very old API
    vt = np.zeros((len(k_shafts), omegas.shape[0]), dtype="complex128")
    for i, sh in enumerate(shafts):
        # Shaft connects nodes sh.nl -> sh.nr
        vt[i] = k_shafts[i] * (U[sh.nr] - U[sh.nl])
    return vt

import inspect

class _ExcWrap:
    """Minimal wrapper to satisfy older OpenTorsion that expects .U and .omegas."""
    def __init__(self, U, omegas):
        self.U = U
        self.omegas = omegas

def vibratory_torque_compat(assembly, excitations, omegas, k_shafts, C=None):
    """
    Works with:
      - New API: vibratory_torque(excitations, omegas, k_shafts, C=None)
      - Old API: vibratory_torque(periodicExcitation, C, C_func=None) expecting .U and .omegas
    """
    try:
        sig = inspect.signature(assembly.vibratory_torque)
        if "k_shafts" in sig.parameters:
            # Newer versions accept matrices + k_shafts (+ optional C)
            return assembly.vibratory_torque(excitations, omegas, k_shafts, C)
    except Exception:
        pass

    # Older versions: wrap into an object with .U and .omegas
    exc = _ExcWrap(excitations, omegas)
    try:
        return assembly.vibratory_torque(exc, C)
    except TypeError:
        # Some very old builds also take a C_func slot
        return assembly.vibratory_torque(exc, C, None)

def forced_response():
    # Parameters of the mechanical model
    k1 = 3.67e8   # Nm/rad
    k2 = 5.496e9  # Nm/rad
    J1 = 1e7      # kgm^2
    J2 = 5770     # kgm^2
    J3 = 97030    # kgm^2

    # Creating assembly
    shafts, disks = [], []
    disks.append(ot.Disk(0, J1))
    shafts.append(ot.Shaft(0, 1, None, None, k=k1, I=0))
    disks.append(ot.Disk(1, J2))
    shafts.append(ot.Shaft(1, 2, None, None, k=k2, I=0))
    disks.append(ot.Disk(2, J3))

    assembly = ot.Assembly(shafts, disk_elements=disks)

    M, K = assembly.M, assembly.K
    C = assembly.C_modal(M, K, xi=0.02)  # we’ll use this if your API supports it

    # Modal analysis
    _A, _B = assembly.state_matrix(C)
    omegas_undamped, omegas_damped, damping_ratios = assembly.modal_analysis()
    print("Eigenfrequencies: ", omegas_undamped.round(3))

    plot_tools = ot.Plots(assembly)
    plot_tools.plot_assembly()
    plot_tools.plot_eigenmodes(modes=3)
    plot_tools.plot_campbell(frequency_range_rpm=[0, 300], num_modes=2)

    # Steady-state forced response analysis
    VT_element1 = []
    VT_element2 = []

    rpm_grid = np.linspace(0.1, 25, 5000)
    k_vec = np.array([k1, k2])

    for rpm in rpm_grid:
        omegas, amplitudes = get_windmill_excitation(rpm)
        excitations = np.zeros((M.shape[0], omegas.shape[0]), dtype="complex128")
        excitations[2] = amplitudes  # Excitation acts on the generator side

        # <- robust call that works across library versions
        T_vib = vibratory_torque_compat(assembly, excitations, omegas, np.array([k1, k2]), C)
        #T_vib = _vibratory_torque_compat(assembly, shafts, excitations, omegas, k_vec, C)

        VT_element1.append(np.sum(np.abs(T_vib[0])))
        VT_element2.append(np.sum(np.abs(T_vib[1])))

    T_e = np.array([np.array(VT_element1), np.array(VT_element2)])
    plot_tools.torque_response_plot(rpm_grid, T_e, show_plot=True)

forced_response()