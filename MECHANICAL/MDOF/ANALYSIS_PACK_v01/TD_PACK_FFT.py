import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def compute_fft(signal, t):
    """
    Compute the FFT of a 1-D signal using a Hanning window.
    
    Parameters:
        signal (array): 1D array (e.g., acceleration for a single DOF)
        t (array): time vector corresponding to the signal
        
    Returns:
        freqs (array): frequencies (only positive frequencies)
        fft_mag (array): magnitude of the FFT (absolute value)
    """
    N = len(signal)
    dt = t[1] - t[0]         # sampling time step
    window = np.hanning(N)   # Hanning window
    signal_win = signal * window
    fft_result = np.fft.fft(signal_win)
    freqs = np.fft.fftfreq(N, d=dt)
    
    # Only keep positive frequencies
    pos_idx = np.where(freqs >= 0)
    return freqs[pos_idx], np.abs(fft_result[pos_idx])


