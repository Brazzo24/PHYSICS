import numpy as np
import matplotlib.pyplot as plt

def create_engine_excitation(f_vals, harmonics, engine_speed_rpm):
    """
    Creates frequency-dependent excitation torque spectrum based on engine orders.

    Parameters:
    - f_vals: array of frequencies [Hz]
    - harmonics: list of tuples [(order, amplitude), ...]
    - engine_speed_rpm: Engine speed in RPM
    
    Returns:
    - excitation_array: array of excitation amplitude values mapped to each frequency in f_vals
    """

    excitation = np.zeros_like(f_vals, dtype=complex)
    engine_freq_hz = engine_speed_rpm / 60.0
    for order, real, imag in harmonics:
        freq = order * engine_freq_hz
        idx = np.argmin(np.abs(f_vals - freq))  # closest frequency bin
        magnitude = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)
        excitation[idx] = magnitude * np.exp(1j * phase)
    return excitation



