import numpy as np
import matplotlib.pyplot as plt

# For sparse N-DOF system
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# For modal analysis
from scipy.linalg import eigh

###############################################################################
# 1. Basic Utility: Dynamic Stiffness
###############################################################################
def dynamic_stiffness(m, c, k, w):
    """
    Computes the complex dynamic stiffness for a single DOF:
        K_d(w) = k - w^2 * m + j * w * c
    """
    return k - (w**2)*m + 1j*w*c

###############################################################################
# 2. 4-DOF System (Direct Matrix Assembly and Solve)
###############################################################################
def compute_4DOF_response(m1, c1, k1, m2, c2, k2, m3, c3, k3, m4, c4, k4, w):
    """
    Solves for X1, X2, X3, X4 given a base velocity excitation at mass 1.
    Assembles the 4x4 system with correct coupling.
    """
    # Compute dynamic stiffness terms
    Kd1 = dynamic_stiffness(m1, c1, k1, w)
    Kd2 = dynamic_stiffness(m2, c2, k2, w)
    Kd3 = dynamic_stiffness(m3, c3, k3, w)
    Kd4 = dynamic_stiffness(m4, c4, k4, w)

    # Base velocity excitation (1 m/s) -> convert to displacement in freq domain
    V_base = 1.0
    X_base = V_base / (1j * w)

    # 4x4 System matrix
    M = np.array([
        [Kd1 + k2,  -k2,      0,       0      ],
        [   -k2,    Kd2 + k3, -k3,     0      ],
        [    0,     -k3,     Kd3 + k4, -k4    ],
        [    0,      0,      -k4,     Kd4    ]
    ])

    # Force only on the first mass from base motion
    F_input = -(1j*w*c1 + k1) * X_base
    RHS = np.array([F_input, 0, 0, 0])

    # Solve for displacements
    X1, X2, X3, X4 = np.linalg.solve(M, RHS)
    return X1, X2, X3, X4


###############################################################################
# MAIN SCRIPT / DEMO
###############################################################################
if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1) PARAMETERS FOR 4-DOF EXAMPLE
    # ---------------------------------------------------------
    # Lower + upper masses
    m1, m2, m3, m4 = 1.0, 1.0, 1.5, 0.8
    c1, c2, c3, c4 = 10.0, 2.0, 5.0, 3.0
    k1, k2, k3, k4 = 2000.0, 12000.0, 3000.0, 3500.0

    # Frequency range
    f_min, f_max = 0.1, 35.0
    num_points = 1000
    f_vals = np.linspace(f_min, f_max, num_points)
    w_vals = 2.0 * np.pi * f_vals

    # ---------------------------------------------------------
    # 2) COMPUTE 4-DOF RESPONSE AND PLOT
    # ---------------------------------------------------------
    # Pre-allocate
    X1_vals_4DOF = np.zeros_like(w_vals, dtype=complex)
    X2_vals_4DOF = np.zeros_like(w_vals, dtype=complex)
    X3_vals_4DOF = np.zeros_like(w_vals, dtype=complex)
    X4_vals_4DOF = np.zeros_like(w_vals, dtype=complex)

    for i, w in enumerate(w_vals):
        X1_vals_4DOF[i], X2_vals_4DOF[i], X3_vals_4DOF[i], X4_vals_4DOF[i] = compute_4DOF_response(
            m1, c1, k1, m2, c2, k2, m3, c3, k3, m4, c4, k4, w
        )

    # Acceleration = -w^2 * X
    A1_vals_4DOF = -w_vals**2 * X1_vals_4DOF
    A2_vals_4DOF = -w_vals**2 * X2_vals_4DOF
    A3_vals_4DOF = -w_vals**2 * X3_vals_4DOF
    A4_vals_4DOF = -w_vals**2 * X4_vals_4DOF

    # Plot acceleration magnitudes
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(f_vals, np.abs(A1_vals_4DOF), label="Mass 1 Accel")
    ax.plot(f_vals, np.abs(A2_vals_4DOF), label="Mass 2 Accel")
    ax.plot(f_vals, np.abs(A3_vals_4DOF), label="Mass 3 Accel")
    ax.plot(f_vals, np.abs(A4_vals_4DOF), label="Mass 4 Accel")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.set_title("4-DOF System Acceleration Responses")
    ax.legend()
    ax.grid(True)
    plt.show()