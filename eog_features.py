"""Preprocessing + features, copied 1:1 from the doctor's train.py
(HusseinYounis/ocular-fatigue-detection) so live data goes through exactly
the same steps as the training data.

The original train.py loads the CSV and trains as soon as it is imported,
so it cannot be imported from a live script. That is why these functions
live in their own file.
"""
import numpy as np
from scipy import signal as sp_signal
from scipy.signal import find_peaks

FS = 1000        # Hz, same as the BIOPAC recording
WINDOW_S = 10    # seconds per window
WINDOW_N = FS * WINDOW_S

FEATURE_COLS = [
    "Blink_Rate_bpm", "Mean_Blink_Amplitude", "OCI",
    "RMS", "Std_Dev", "Dominant_Freq_Hz", "Energy_0.1_0.5Hz",
]


def preprocess(sig, fs=FS):
    """Remove DC offset, then 4th-order Butterworth band-pass 0.3-12 Hz (SOS)."""
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.mean(sig)
    sos = sp_signal.butter(4, [0.3, 12.0], btype="band", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, sig)


def extract_features(win, fs=FS):
    dur_s = len(win) / fs
    z = (win - np.mean(win)) / (np.std(win) + 1e-9)
    peaks, _ = find_peaks(z, prominence=2.5, distance=int(0.30 * fs))
    n_blinks = len(peaks)

    blink_rate_bpm = (n_blinks / dur_s) * 60
    mean_blink_amp = float(np.mean(z[peaks])) if n_blinks > 0 else 0.0
    oci = min((n_blinks * 0.2 / dur_s) * 100.0, 100.0)

    # NOTE: RMS and Std_Dev are in the units of the input (mV at the
    # electrodes in the dataset). They are NOT amplitude-invariant, so live
    # data must be converted to electrode millivolts before this call.
    rms = float(np.sqrt(np.mean(win ** 2)))
    std_dev = float(np.std(win))

    freqs = np.fft.rfftfreq(len(win), d=1.0 / fs)
    psd = np.abs(np.fft.rfft(z)) ** 2
    dominant_freq = float(freqs[np.argmax(psd[1:]) + 1])
    blink_band = (freqs >= 0.1) & (freqs <= 0.5)
    energy_blink = float(np.sum(psd[blink_band]))

    return {
        "Blink_Rate_bpm":       blink_rate_bpm,
        "Mean_Blink_Amplitude": mean_blink_amp,
        "OCI":                  oci,
        "RMS":                  rms,
        "Std_Dev":              std_dev,
        "Dominant_Freq_Hz":     dominant_freq,
        "Energy_0.1_0.5Hz":     energy_blink,
    }
