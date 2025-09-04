"""
Planar 3-DoF Parallel Manipulator (2D Stewart-analog) — Learning Model
---------------------------------------------------------------------

Goal: a minimal, understandable sandbox for the core ideas behind a 6-DoF
Stewart platform, but in 2D with only 3 DoF: x (surge), z (heave), and θ (pitch).
Three prismatic legs connect a fixed base to a moving rigid body (the "platform").

What you get:
- Clean geometry with 3 base anchors and 3 platform anchors in 2D
- Exact leg kinematics (lengths, unit directions)
- Jacobian J mapping platform twist [vx, vz, ω] → leg length rates Ldot
- Wrench ↔ leg force mapping via a 3×3 matrix A
- Rigid-body dynamics in plane (mass m, inertia Iy about out-of-plane axis)
- Simple PD controller in SE(2) to track a pose/trajectory
- Simple demo trajectory (heave sine + small pitch step)

This is intentionally compact and readable; extend it once the concepts click.

Dependencies: numpy, matplotlib

Author: ChatGPT — Aug 2025
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

# ---------------- Utility ----------------

def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

# ---------------- Geometry ----------------

@dataclass
class PlanarGeometry:
    base_pts_W: np.ndarray  # (3,2) base anchors in world frame
    plat_pts_P: np.ndarray  # (3,2) platform anchors in platform frame P

    @staticmethod
    def symmetric(R_base: float = 1.0, R_top: float = 0.6) -> "PlanarGeometry":
        # Equilateral triangle for base
        ang = np.deg2rad(np.array([90, 210, 330]))
        base = np.c_[R_base*np.cos(ang), R_base*np.sin(ang)]
        # Slightly smaller/rotated triangle on platform
        ang_t = ang + np.deg2rad(20)
        topP = np.c_[R_top*np.cos(ang_t), R_top*np.sin(ang_t)]
        return PlanarGeometry(base, topP)

# ---------------- Params & Model ----------------

@dataclass
class PlanarParams:
    m: float           # kg
    Iy: float          # kg m^2 (about out-of-plane y-axis)
    com_P: np.ndarray  # (2,) COM in platform frame
    g: float = 9.81
    force_limits: Tuple[float, float] = (-1.8e4, 1.8e4)
    stroke_limits: Tuple[float, float] = (0.5, 1.6)  # meters

class PlanarParallel3DoF:
    def __init__(self, geom: PlanarGeometry, p: PlanarParams,
                 x0: float = 0.0, z0: float = 1.0, th0: float = 0.0,
                 vx0: float = 0.0, vz0: float = 0.0, om0: float = 0.0):
        self.geom = geom
        self.p = p
        self.x = x0
        self.z = z0
        self.th = th0
        self.vx = vx0
        self.vz = vz0
        self.om = om0

    # ----- Kinematics -----
    def top_pts_W(self) -> np.ndarray:
        R = rot2(self.th)
        return np.vstack([self.x, self.z]).T + (R @ self.geom.plat_pts_P.T).T

    def com_W(self) -> np.ndarray:
        return np.array([self.x, self.z]) + rot2(self.th) @ self.p.com_P

    def leg_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        top = self.top_pts_W()
        base = self.geom.base_pts_W
        Lvec = top - base           # (3,2)
        L = np.linalg.norm(Lvec, axis=1)
        u = Lvec / L[:, None]
        return Lvec, L, u

    def jacobian(self) -> np.ndarray:
        """J maps platform twist [vx, vz, ω]^T to leg rates Ldot.
        Row i: [u_ix, u_iz, (p_i_W × u_i)_y], with 2D cross giving scalar out-of-plane.
        """
        R = rot2(self.th)
        _, _, u = self.leg_vectors()
        J = np.zeros((3,3))
        for i in range(3):
            p_i_W = R @ self.geom.plat_pts_P[i]
            ui = u[i]
            J[i, 0:2] = ui
            # 2D moment (about COM by default handled in wrench map); for velocity mapping we use about body origin
            J[i, 2] = p_i_W[0]*ui[1] - p_i_W[1]*ui[0]  # scalar z-component of cross
        return J

    # ----- Wrench from leg forces -----
    def wrench_from_forces(self, f: np.ndarray) -> Tuple[np.ndarray, float]:
        _, _, u = self.leg_vectors()
        top = self.top_pts_W()
        rC = self.com_W()
        F = np.sum(u * f[:, None], axis=0)
        tau = 0.0
        for i in range(3):
            arm = top[i] - rC
            # scalar out-of-plane moment (y)
            tau += arm[0]*(u[i][1]*f[i]) - arm[1]*(u[i][0]*f[i])
        return F, tau

    def solve_forces_for_wrench(self, F_des: np.ndarray, tau_des: float) -> np.ndarray:
        # Build 3×3 A: columns are leg force directions/arms
        _, _, u = self.leg_vectors()
        top = self.top_pts_W()
        rC = self.com_W()
        A = np.zeros((3,3))
        for i in range(3):
            A[0:2, i] = u[i]
            arm = top[i] - rC
            A[2, i] = arm[0]*u[i][1] - arm[1]*u[i][0]
        b = np.array([F_des[0], F_des[1], tau_des])
        f, *_ = np.linalg.lstsq(A, b, rcond=None)
        fmin, fmax = self.p.force_limits
        return np.clip(f, fmin, fmax)

    # ----- Dynamics -----
    def dynamics(self, f: np.ndarray) -> Tuple[np.ndarray, float]:
        F_act, tau_act = self.wrench_from_forces(f)
        Fg = np.array([0.0, -self.p.m*self.p.g])
        a = (F_act + Fg)/self.p.m
        alpha = (tau_act)/self.p.Iy - 0.0*self.om  # add rot. damping as needed
        return a, alpha

    # ----- Control (PD in SE(2)) -----
    def control_pd(self, x_d: float, z_d: float, th_d: float,
                   vx_d: float=0.0, vz_d: float=0.0, om_d: float=0.0,
                   ax_d: float=0.0, az_d: float=0.0, al_d: float=0.0,
                   Kp_pos=np.diag([2e4, 3e4]), Kd_pos=np.diag([2e3, 3e3]),
                   Kp_att=2e4, Kd_att=2e3) -> np.ndarray:
        e_p = np.array([x_d - self.x, z_d - self.z])
        e_v = np.array([vx_d - self.vx, vz_d - self.vz])
        e_th = th_d - self.th
        e_om = om_d - self.om

        F_des = Kp_pos @ e_p + Kd_pos @ e_v + self.p.m*np.array([ax_d, az_d])
        F_des += np.array([0.0, self.p.m*self.p.g])  # gravity compensation
        tau_des = Kp_att*e_th + Kd_att*e_om + self.p.Iy*al_d

        return self.solve_forces_for_wrench(F_des, tau_des)

    # ----- Integration -----
    def step(self, f: np.ndarray, dt: float):
        a, alpha = self.dynamics(f)
        self.vx += a[0]*dt
        self.vz += a[1]*dt
        self.om += alpha*dt
        self.x += self.vx*dt
        self.z += self.vz*dt
        self.th += self.om*dt

        # very gentle stroke constraint nudge
        _, L, u = self.leg_vectors()
        Lmin, Lmax = self.p.stroke_limits
        if np.any(L < Lmin) or np.any(L > Lmax):
            corr = np.zeros(2)
            base = self.geom.base_pts_W
            top = self.top_pts_W()
            for i in range(3):
                if L[i] < Lmin:
                    corr -= u[i]*(Lmin - L[i])*0.2
                elif L[i] > Lmax:
                    corr += u[i]*(L[i] - Lmax)*0.2
            self.x += corr[0]
            self.z += corr[1]

    def simulate(self, T: float, dt: float,
                 traj: Callable[[float], Tuple[float,float,float,float,float,float,float,float,float]]):
        N = int(np.round(T/dt))
        log = {
            't': np.zeros(N),
            'x': np.zeros(N), 'z': np.zeros(N), 'th': np.zeros(N),
            'vx': np.zeros(N), 'vz': np.zeros(N), 'om': np.zeros(N),
            'f': np.zeros((N,3)), 'L': np.zeros((N,3))
        }
        for k in range(N):
            t = k*dt
            xd, zd, thd, vxd, vzd, omd, axd, azd, ald = traj(t)
            f = self.control_pd(xd, zd, thd, vxd, vzd, omd, axd, azd, ald)
            self.step(f, dt)

            log['t'][k] = t
            log['x'][k] = self.x
            log['z'][k] = self.z
            log['th'][k] = self.th
            log['vx'][k] = self.vx
            log['vz'][k] = self.vz
            log['om'][k] = self.om
            log['f'][k] = f
            _, L, _ = self.leg_vectors()
            log['L'][k] = L
        return log

# ---------------- Demo trajectory ----------------

def make_demo_traj(z0: float=1.0):
    def traj(t: float):
        xd = 0.0
        zd = z0 + 0.02*np.sin(2*np.pi*1.0*t)  # 2 cm heave @ 1 Hz
        thd = np.deg2rad(0.0)
        if t >= 2.0:
            thd = np.deg2rad(2.0)  # small pitch step
        return xd, zd, thd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return traj

# ---------------- Quick start ----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    geom = PlanarGeometry.symmetric(R_base=1.0, R_top=0.6)
    p = PlanarParams(m=250.0, Iy=120.0, com_P=np.array([0.0, 0.05]),
                     force_limits=(-2.0e4, 2.0e4), stroke_limits=(0.7, 1.6))
    sys = PlanarParallel3DoF(geom, p, x0=0.0, z0=1.05, th0=0.0)

    T, dt = 5.0, 0.001
    log = sys.simulate(T, dt, make_demo_traj(z0=1.05))

    t = log['t']
    plt.figure(); plt.plot(t, log['z']); plt.xlabel('t [s]'); plt.ylabel('z [m]'); plt.title('Heave')
    plt.figure(); plt.plot(t, np.rad2deg(log['th'])); plt.xlabel('t [s]'); plt.ylabel('pitch [deg]'); plt.title('Pitch')
    plt.figure(); plt.plot(t, log['f']); plt.xlabel('t [s]'); plt.ylabel('leg forces [N]'); plt.title('Leg forces')
    plt.show()
