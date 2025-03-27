# main.py

import numpy as np
from simulation import run_simulation
from FDcalculations import forced_response_postprocessing
from comparison import *
from compare_modes import *

def main():
    # Define the frequency range and forcing vector
    f_min, f_max, num_points = 0.1, 400.0, 10000
    f_vals = np.linspace(f_min, f_max, num_points)
    
    # Define common forcing vector (adjust as needed)
    # Suppose each run uses the same F_ext
    m_common = np.array([1.21e-2, 3.95e-4, 7.92e-4, 1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3, 2.73e-1, 2.69e+1])
    N = len(m_common)
    F_ext = np.zeros(N, dtype=complex)
    F_ext[0] = 1.0

    # Define two different parameter sets

    m_run1 = np.array([1.21e-2, 3.95e-4, 7.92e-4, 1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3, 2.73e-1, 2.69e+1]),
    c_run1 = np.array([0.05]*9),
    k_run1 = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5, 2.72e4, 4.97e3, 7.73e2, 8.57e2])
    
    
    m_run2 = np.array([1.21e-2, 3.95e-4, 7.92e-4, 1.02e-3, 1.42e-3, 1.12e-4, 1.22e-3, 1.35e-3, 2.73e-1, 2.69e+1]),
    c_run2 = np.array([0.05]*9),
    k_run2 = np.array([2.34e4, 1.62e5, 1.11e3, 1.10e5, 1.10e5, 2.72e4, 4.97e3, 7.73e2, 8.57e2])

    f_min, f_max, num_points = 0.1, 400.0, 10000
    f_vals = np.linspace(f_min, f_max, num_points)

    forced_response1 = forced_response_postprocessing(m_run1, c_run1, k_run1, f_vals, F_ext)
    forced_response2 = forced_response_postprocessing(m_run2, c_run2, k_run2, f_vals, F_ext)

    compare_forced_response_overview(forced_response1, forced_response2, f_vals, m_run1)
    # Assuming modal_energy_analysis returns a list of modal energy dictionaries.



if __name__ == "__main__":
    main()
