# simulation.py

import numpy as np
from FDcalculations import forced_response_postprocessing, free_vibration_analysis_free_chain, modal_energy_analysis
from plotting import plot_forced_response_overview, plot_modal_energy_overview

def run_simulation(m, c_inter, k_inter, f_vals, F_ext):
    results = {}
    
    # Forced response
    response = forced_response_postprocessing(m, c_inter, k_inter, f_vals, F_ext)
    results['forced_response'] = response
    
    # Free vibration analysis (and modal energy analysis)
    f_n, eigvecs, M_free, K_free = free_vibration_analysis_free_chain(m, k_inter)
    results['natural_frequencies'] = f_n
    results['eigvecs'] = eigvecs
    
    # Modal energy analysis
    modal_energies = modal_energy_analysis(m, k_inter, f_n, eigvecs, M_free)
    results['modal_energies'] = modal_energies
    
    return results
