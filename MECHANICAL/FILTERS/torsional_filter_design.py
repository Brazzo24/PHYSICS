# Create a reusable Python module for torsional vibration filter analysis
"""

Frequency-domain analysis and design helpers for torsional driveline "filtering".
Implements Kelvin–Voigt and Standard Linear Solid (Zener) couplings, multi-stage chains,
optional tuned absorbers, transfer ratio T_transmitted / T_input, and mode estimates.

Units:
- Inertias: kg·m^2
- Stiffness: N·m/rad   (convert from N·m/deg by multiplying by 180/π)
- Damping: N·m·s/rad
- Frequency: rad/s (use ω = 2π·Hz). Engine order ω = n * 2π * RPM/60.

Author: ChatGPT
License: MIT
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.linalg import eigh
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def deg_to_rad(x_deg: float) -> float:
    return np.deg2rad(x_deg)


def rad_to_deg(x_rad: float) -> float:
    return np.rad2deg(x_rad)


@dataclass
class Element:
    """A torsional coupling between nodes i and j.
    If j is None, the element is to ground (θ_ground=0).
    type_: 'kv' or 'sls'
    For 'kv': params = {'k':..., 'c':...}
    For 'sls': params = {'k0':..., 'k_inf':..., 'tau':..., 'c_par':...}
    label: optional name for reporting.
    """
    i: int
    j: Optional[int]
    type_: str
    params: Dict[str, float]
    label: Optional[str] = None

    def complex_stiffness(self, omega: float) -> complex:
        if self.type_ == 'kv':
            k = float(self.params.get('k', 0.0))
            c = float(self.params.get('c', 0.0))
            return k + 1j * omega * c
        elif self.type_ == 'sls':
            k0 = float(self.params['k0'])
            k_inf = float(self.params['k_inf'])
            tau = float(self.params['tau'])
            c_par = float(self.params.get('c_par', 0.0))
            k_star = k_inf + (k0 - k_inf) / (1.0 + 1j * omega * tau)
            return k_star + 1j * omega * c_par
        else:
            raise ValueError(f"Unknown element type '{self.type_}'")


class TorsionalSystem:
    """N-DOF torsional chain model with frequency-dependent couplings.

    Nodes have inertias J[k]. Elements connect node pairs (i, j) or to ground (j=None).
    Input torque is applied at input_node as a complex amplitude (default 1.0 N·m).
    The 'transmission_element' is the element index across which transmitted torque is measured.
    """

    def __init__(self, inertias: ArrayLike):
        self.J = np.asarray(inertias, dtype=float).copy()
        self.N = int(self.J.size)
        self.elements: List[Element] = []
        self.input_node: int = 0
        self.transmission_element: Optional[int] = None

    # --- element builders ---
    def add_kv(self, i: int, j: Optional[int], k: float, c: float = 0.0, label: Optional[str] = None) -> int:
        e = Element(i=i, j=j, type_='kv', params={'k': k, 'c': c}, label=label)
        self.elements.append(e)
        return len(self.elements) - 1

    def add_sls(self, i: int, j: Optional[int], k0: float, k_inf: float, tau: float, c_par: float = 0.0,
                label: Optional[str] = None) -> int:
        e = Element(i=i, j=j, type_='sls', params={'k0': k0, 'k_inf': k_inf, 'tau': tau, 'c_par': c_par}, label=label)
        self.elements.append(e)
        return len(self.elements) - 1

    def add_absorber(self, node: int, J_a: float, k: float, c: float = 0.0, label: Optional[str] = None) -> Tuple[int, int]:
        """Create a branch absorber: new node with inertia J_a, connected to 'node' via KV(k,c)."""
        new_index = self.N
        # extend inertias
        self.J = np.append(self.J, J_a)
        self.N += 1
        idx = self.add_kv(node, new_index, k=k, c=c, label=label or f"absorber@{node}")
        return new_index, idx

    # --- assembly ---
    def dynamic_stiffness(self, omega: float) -> np.ndarray:
        """Assemble global dynamic stiffness D(ω) = -ω²J + Σ_e K*_e(ω)·L_e^T L_e,
        where L_e encodes relative angle (θ_i - θ_j). Returns NxN complex matrix.
        """
        D = - (omega ** 2) * np.diag(self.J).astype(complex)
        for e in self.elements:
            Kstar = e.complex_stiffness(omega)
            if e.j is None:  # to ground
                D[e.i, e.i] += Kstar
            else:
                i, j = e.i, e.j
                D[i, i] += Kstar
                D[j, j] += Kstar
                D[i, j] -= Kstar
                D[j, i] -= Kstar
        return D

    def _apply_fix(self, M: np.ndarray, f: np.ndarray, fix_node: Optional[int]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Optionally eliminate a fixed node (θ=0). Returns reduced M,f and a scatter map for reconstruction."""
        if fix_node is None:
            return M, f, None
        keep = [k for k in range(self.N) if k != fix_node]
        Mr = M[np.ix_(keep, keep)]
        fr = f[keep]
        scatter = np.array(keep, dtype=int)
        return Mr, fr, scatter

    def solve_theta(self, omega: float, T_in: complex = 1.0, input_node: Optional[int] = None,
                    fix_node: Optional[int] = None) -> np.ndarray:
        """Solve for nodal angles θ given a harmonic input torque at input_node."""
        D = self.dynamic_stiffness(omega)
        node = self.input_node if input_node is None else input_node
        f = np.zeros(self.N, dtype=complex)
        f[node] = T_in
        Dr, fr, scatter = self._apply_fix(D, f, fix_node)
        theta_r = np.linalg.solve(Dr, fr)
        theta = np.zeros(self.N, dtype=complex)
        if scatter is None:
            theta = theta_r
        else:
            theta[scatter] = theta_r
            # fixed node already zero
        return theta

    def element_torque(self, e_idx: int, omega: float, theta: np.ndarray) -> complex:
        """Complex torque in element e, positive from i→j."""
        e = self.elements[e_idx]
        Kstar = e.complex_stiffness(omega)
        if e.j is None:
            return Kstar * (theta[e.i] - 0.0)
        else:
            return Kstar * (theta[e.i] - theta[e.j])

    def transfer_ratio(self, omega: float, T_in: complex = 1.0, input_node: Optional[int] = None,
                       transmission_element: Optional[int] = None, fix_node: Optional[int] = None) -> float:
        """|T_transmitted / T_in| at frequency ω."""
        e_idx = self.transmission_element if transmission_element is None else transmission_element
        if e_idx is None:
            raise ValueError("transmission_element not set.")
        theta = self.solve_theta(omega, T_in=T_in, input_node=input_node, fix_node=fix_node)
        T_t = self.element_torque(e_idx, omega, theta)
        return np.abs(T_t / T_in)

    # --- sweeps & utilities ---
    def sweep(self, omegas: ArrayLike, **kw) -> np.ndarray:
        """Vectorized transfer magnitude over an array of ω values."""
        mags = []
        for w in np.asarray(omegas, dtype=float):
            mags.append(self.transfer_ratio(w, **kw))
        return np.asarray(mags)

    def order_sweep(self, rpm: float, orders: ArrayLike, **kw) -> Tuple[np.ndarray, np.ndarray]:
        """Compute |T_t/T_in| vs engine order at given RPM.
        Returns (orders, magnitudes).
        """
        orders = np.asarray(orders, dtype=float)
        omega_engine = 2.0 * np.pi * rpm / 60.0
        omegas = orders * omega_engine
        mags = self.sweep(omegas, **kw)
        return orders, mags

    def order_map(self, rpm_grid: ArrayLike, orders: ArrayLike, **kw) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a 2D map of |T_t/T_in| on an RPM × order grid.
        Returns (RPMs, orders, magnitudes[RPM,order]).
        """
        rpms = np.asarray(rpm_grid, dtype=float)
        ords = np.asarray(orders, dtype=float)
        Z = np.zeros((rpms.size, ords.size), dtype=float)
        for i, rpm in enumerate(rpms):
            omega_engine = 2.0 * np.pi * rpm / 60.0
            omegas = ords * omega_engine
            Z[i, :] = self.sweep(omegas, **kw)
        return rpms, ords, Z

    def modal_frequencies(self, kind: str = 'k0', fix_node: Optional[int] = None) -> np.ndarray:
        """Estimate undamped natural frequencies (Hz) via generalized eigenproblem.
        kind='k0' uses static stiffness (ω→0): KV k + SLS k0.
        kind='kinf' uses high-frequency stiffness: KV k + SLS k_inf.
        """
        # Build real stiffness proxy
        K = np.zeros((self.N, self.N), dtype=float)
        for e in self.elements:
            if e.type_ == 'kv':
                k = float(e.params.get('k', 0.0))
            elif e.type_ == 'sls':
                k = float(e.params['k0'] if kind == 'k0' else e.params['k_inf'])
            else:
                continue
            if e.j is None:
                K[e.i, e.i] += k
            else:
                i, j = e.i, e.j
                K[i, i] += k
                K[j, j] += k
                K[i, j] -= k
                K[j, i] -= k

        J = np.diag(self.J)

        if fix_node is not None:
            keep = [k for k in range(self.N) if k != fix_node]
            K = K[np.ix_(keep, keep)]
            J = J[np.ix_(keep, keep)]

        if _HAS_SCIPY:
            w2, _ = eigh(K, J)
        else:  # fallback: numpy eig (may produce small complex noise)
            w2, _ = np.linalg.eig(np.linalg.pinv(J) @ K)
            w2 = np.real(w2)

        # filter small/negative numerical artifacts
        w2 = np.clip(w2, a_min=0.0, a_max=None)
        w = np.sqrt(w2)
        f = w / (2.0 * np.pi)
        f = np.sort(f)
        # drop near-zero rigid-body
        f = f[f > 1e-6]
        return f


# --- handy design formulas for 2-mass ---
def two_mass_target_k(I1: float, I2: float, f_n_target_hz: float) -> float:
    """Return k to place the 2-mass mode at f_n (Hz): ω_n^2 = k*(1/I1 + 1/I2)."""
    omega_n = 2.0 * np.pi * f_n_target_hz
    return (omega_n ** 2) / (1.0 / I1 + 1.0 / I2)


def two_mass_c_for_zeta(I1: float, I2: float, k: float, zeta: float) -> float:
    """Return c for target damping ratio ζ using reduced inertia I_eq."""
    I_eq = (I1 * I2) / (I1 + I2)
    return 2.0 * zeta * np.sqrt(k * I_eq)


# --- quick-start examples (not executed unless this file is run directly) ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example 1: Two-mass with soft coupling, measure torque across soft element
    I1, I2 = 0.15, 0.20      # kg·m^2
    k_soft = 1200.0          # N·m/rad
    c_soft = 8.0             # N·m·s/rad

    sys1 = TorsionalSystem([I1, I2])
    e_soft = sys1.add_kv(0, 1, k=k_soft, c=c_soft, label="soft coupling")
    sys1.transmission_element = e_soft
    sys1.input_node = 0

    rpm = 3000.0
    orders = np.linspace(0.5, 20, 400)
    ords, mags = sys1.order_sweep(rpm, orders)
    plt.figure()
    plt.semilogy(ords, mags)
    plt.xlabel("Engine order")
    plt.ylabel("|T_t/T_in|")
    plt.title("Two-mass low-pass behavior")
    plt.grid(True, which='both')
    plt.show()

    # Example 2: Two-stage (engine - soft - mid inertia - soft - gearbox)
    I_mid = 0.05
    sys2 = TorsionalSystem([I1, I_mid, I2])
    e1 = sys2.add_kv(0, 1, k=1000.0, c=6.0, label="stage1")
    e2 = sys2.add_kv(1, 2, k=1500.0, c=6.0, label="stage2")
    sys2.transmission_element = e2  # torque seen by gearbox
    sys2.input_node = 0

    ords2, mags2 = sys2.order_sweep(rpm, orders)
    plt.figure()
    plt.semilogy(ords, mags, label="one-stage")
    plt.semilogy(ords2, mags2, label="two-stage")
    plt.xlabel("Engine order")
    plt.ylabel("|T_t/T_in|")
    plt.title("Added roll-off with two-stage coupling")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    # Example 3: Add a tuned absorber (notch) on engine node for a target order
    sys3 = TorsionalSystem([I1, I2])
    e_soft3 = sys3.add_kv(0, 1, k=k_soft, c=c_soft, label="soft")
    sys3.transmission_element = e_soft3
    sys3.input_node = 0

    target_order = 2.0
    omega_engine = 2.0 * np.pi * rpm / 60.0
    omega_target = target_order * omega_engine

    J_a = 0.002
    k_a = (omega_target ** 2) * J_a
    node_a, _ = sys3.add_absorber(0, J_a=J_a, k=k_a, c=0.1, label="absorber")

    ords3, mags3 = sys3.order_sweep(rpm, orders)
    plt.figure()
    plt.semilogy(ords, mags, label="baseline")
    plt.semilogy(ords3, mags3, label=f"with absorber @{target_order:.1f} order")
    plt.xlabel("Engine order")
    plt.ylabel("|T_t/T_in|")
    plt.title("Notch via tuned torsional absorber")
    plt.grid(True, which='both')
    plt.legend()
    plt.show()

    # Example 4: Viscoelastic (Zener) coupling to localize damping near resonance
    sys4 = TorsionalSystem([I1, I2])
    e_ve = sys4.add_sls(0, 1, k0=900.0, k_inf=1500.0, tau=1/400.0, c_par=1.5, label="Zener")
    sys4.transmission_element = e_ve
    sys4.input_node = 0

    ords4, mags4 = sys4.order_sweep(rpm, orders)
    plt.figure()
    plt.semilogy(ords4, mags4)
    plt.xlabel("Engine order")
    plt.ylabel("|T_t/T_in|")
    plt.title("Standard Linear Solid coupling")
    plt.grid(True, which='both')
    plt.show()



