"""Signal-quality check for one 10 s window, shared by server.py and analyze_recordings.py.

Limits were measured on real recordings with the BioAmp EXG Pill + ESP32:
  clean rested windows : clipping <= 1.8 %, raw std 104-474 mV, mains hum 0-1.7 %
  no person wearing    : clipping ~40 %,   raw std ~1100 mV,   mains hum 30-42 %
  poor electrode contact:                                       mains hum 11-88 %
  person C (2026-11-23), visually good signal:                  mains hum 6-33 %
The 0.3-12 Hz band-pass removes the hum completely (<0.1 % left), so moderate hum does not
change the features; the limit only rejects windows where hum dominates the raw signal.
"""
import numpy as np

ADC_LOW_MV, ADC_HIGH_MV = 150, 3100   # outside this the ESP32 ADC is clipping
Q_MAX_CLIP_PCT = 5.0
Q_MIN_STD_MV = 3.0
Q_MAX_STD_MV = 800.0
Q_MAX_MAINS_PCT = 40.0                # 45-65 Hz power as % of raw power (was 10, too strict)


def quality_problems(raw_window_adc, feats, fs=1000):
    """Reasons this window is not usable (empty list = OK).

    raw_window_adc: raw ESP32 samples in mV at the pin (unfiltered).
    feats: features of the same window (from eog_features.extract_features).
    """
    w = np.asarray(raw_window_adc, dtype=float)
    problems = []
    clip = float(np.mean((w <= ADC_LOW_MV) | (w >= ADC_HIGH_MV)) * 100)
    std = float(np.std(w))
    if clip > Q_MAX_CLIP_PCT:
        problems.append(f"signal hits the ADC limits {clip:.0f}% of the time (electrode off or loose?)")
    if std < Q_MIN_STD_MV:
        problems.append("flat signal (sensor not connected?)")
    elif std > Q_MAX_STD_MV:
        problems.append("very large swings (electrode off, loose, or moving?)")
    if std >= Q_MIN_STD_MV:
        spec = np.abs(np.fft.rfft(w - w.mean())) ** 2
        freqs = np.fft.rfftfreq(len(w), d=1.0 / fs)
        total = spec[freqs >= 0.3].sum()
        mains = float(spec[(freqs >= 45) & (freqs <= 65)].sum() / total * 100) if total > 0 else 0.0
        if mains > Q_MAX_MAINS_PCT:
            problems.append(f"{mains:.0f}% mains hum (poor electrode contact? press the electrodes, "
                            "check the reference electrode)")
    if feats.get("Blink_Rate_bpm", 0) == 0:
        problems.append("no blinks detected")
    return problems
