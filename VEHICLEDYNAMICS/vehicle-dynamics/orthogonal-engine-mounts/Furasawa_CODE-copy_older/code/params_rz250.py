"""Reconstructed parameter set for the paper's example machine.

SAE 830228 describes a water-cooled two-stroke parallel twin with a 180 deg
crank and cylinders inclined 24 deg forward -- i.e. an RZ250/RD350LC-class
engine.  The paper does NOT publish the numbers, so everything here is a
plausible reconstruction chosen to land on the natural frequencies it *does*
publish (Table 2).

Treat this file as the template for your own machine: replace the numbers,
keep the structure.  Nothing else in the notebook or in ``orthomount.py``
hard-codes any of it.

Provenance of each block is marked:
    [paper]   stated in SAE 830228
    [spec]    published RZ250 specification
    [est]     engineering estimate
    [fit]     obtained by fitting to the paper's Table 2 (see notebook S8)
"""

import numpy as np

import orthomount as om

D = np.deg2rad

# ---------------------------------------------------------------------------
# Engine as a rigid body
# ---------------------------------------------------------------------------
MASS = 40.0          # [kg]     engine + gearbox unit                    [est]
I_XI = 0.95          # [kg m^2] principal inertia, axis nearest x        [est]
I_ETA = 1.30         # [kg m^2] principal inertia about y (pitch)        [est]
I_ZETA = 1.60        # [kg m^2] principal inertia, axis nearest z        [est]
ALPHA = D(30.0)      # principal inertia axis inclination              [paper]

# ---------------------------------------------------------------------------
# Crank / cylinder geometry
# ---------------------------------------------------------------------------
BORE = 0.054         # [m]                                              [spec]
STROKE = 0.054       # [m]                                              [spec]
ROD_LENGTH = 0.100   # [m]                                               [est]
CYL_TILT = D(24.0)   # cylinders inclined forward from vertical        [paper]
CYL_SPACING = 0.070  # [m] distance between cylinder centrelines         [est]
M_RECIP = 0.20       # [kg] piston + rings + pin + small end             [est]
M_ROT = 0.00         # [kg] rotating unbalance at crank radius           [est]
BALANCE_FACTOR = 0.0 # counterweights: 0 = none, so the couple is full   [est]
CRANK_PHASES = (0.0, np.pi)   # 180 deg twin                           [paper]

# ---------------------------------------------------------------------------
# Mount layout  -- 4 cylindrical bushes, two front two rear, narrow laterally
# ---------------------------------------------------------------------------
# All positions relative to the engine CG, global axes (x fwd, y left, z up).
X_FRONT = 0.2305     # [m]                                               [fit]
X_REAR = -0.2602     # [m]                                               [fit]
Z_FRONT = -0.1155    # [m]                                               [fit]
Z_REAR = 0.1316      # [m]                                               [fit]
Y_HALF = 0.0498      # [m] half the lateral spacing -- deliberately small [fit]
K_FRONT_AXIAL = 1.276e6    # [N/m] along the bush axis                   [fit]
K_FRONT_RADIAL = 4.728e6   # [N/m]                                       [fit]
K_REAR_AXIAL = 0.664e6     # [N/m]                                       [fit]
K_REAR_RADIAL = 4.282e6    # [N/m]                                       [fit]
BUSH_TILT = 0.6442         # [rad] bush axis rotated about y             [fit]
LOSS_FACTOR = 0.15         # rubber hysteretic damping                   [est]


def engine_body() -> om.RigidBody:
    return om.RigidBody.from_principal(MASS, I_XI, I_ETA, I_ZETA, ALPHA)


def engine() -> om.Engine:
    cyls = [om.Cylinder(crank_phase=ph,
                        y=sy * CYL_SPACING / 2.0,
                        axis_tilt=CYL_TILT,
                        m_recip=M_RECIP,
                        m_rot=M_ROT)
            for ph, sy in zip(CRANK_PHASES, (+1, -1))]
    return om.Engine(cylinders=cyls, crank_radius=STROKE / 2.0,
                     rod_length=ROD_LENGTH, balance_factor=BALANCE_FACTOR)


def mounts() -> list[om.Mount]:
    out = []
    R = om.rot_from_euler_zyx(ry=BUSH_TILT)
    for x, z, ka, kr, tag in ((X_FRONT, Z_FRONT, K_FRONT_AXIAL, K_FRONT_RADIAL, "front"),
                              (X_REAR, Z_REAR, K_REAR_AXIAL, K_REAR_RADIAL, "rear")):
        for sy, side in ((+1, "L"), (-1, "R")):
            out.append(om.Mount(position=[x, sy * Y_HALF, z],
                                stiffness=[kr, ka, kr],   # bush axis along y
                                orientation=R,
                                loss_factor=LOSS_FACTOR,
                                name=f"{tag}-{side}"))
    return out


def system() -> om.MountSystem:
    return om.MountSystem(engine_body(), mounts())


# The paper's Table 2, for comparison.
TABLE2 = [
    # measured [Hz], predicted [Hz], mode
    (35.0, 34.5, "Roll"),
    (50.5, 47.0, "Lateral"),
    (76.5, 73.8, "Yaw"),
    (100.0, 107.5, "Fore/Aft"),
    (109.0, 103.9, "Bounce"),
    (180.0, 166.4, "Pitch"),
]
