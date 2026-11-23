import os
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.signal import find_peaks
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Segment boundaries from dataset README (seconds).
# Basal = resting baseline (no fatigue); participant sessions = sustained fixation (fatigue).
SEGMENTS = [
    ("basal",         0,   60,  0),
    ("participant_1", 68,  220, 1),
    ("participant_2", 291, 374, 1),
    ("participant_3", 442, 498, 1),
    ("participant_4", 532, 622, 1),
]

FS = 1000       # sampling frequency (Hz)
WINDOW_S = 10   # window length in seconds
WINDOW_N = WINDOW_S * FS

script_dir = os.path.dirname(os.path.abspath(__file__))
candidate_dirs = [
    os.path.join(os.getcwd(), "dataset"),
    os.path.join(script_dir, "dataset"),
    script_dir,
    os.getcwd(),
]
signal_file_path = next(
    (
        os.path.join(directory, "EOG_raw_signal.csv")
        for directory in candidate_dirs
        if os.path.isfile(os.path.join(directory, "EOG_raw_signal.csv"))
    ),
    None,
)
if signal_file_path is None:
    checked_paths = [
        os.path.join(directory, "EOG_raw_signal.csv")
        for directory in candidate_dirs
    ]
    raise FileNotFoundError(
        "Could not find EOG_raw_signal.csv. Checked:\n"
        + "\n".join(checked_paths)
    )
dataset_dir = os.path.dirname(signal_file_path)

print(f"Reading data from: {signal_file_path}")
df = pd.read_csv(signal_file_path, encoding="latin1")


# ---------- Signal preprocessing ----------

def preprocess(sig, fs=FS):
    """Remove DC offset and apply 4th-order Butterworth bandpass (0.3-12 Hz).
    SOS form is used for numerical stability at the low-frequency cutoff."""
    sig = sig - np.mean(sig)
    sos = sp_signal.butter(4, [0.3, 12.0], btype="band", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, sig)


# ---------- Feature extraction ----------

def extract_features(win, fs=FS):
    dur_s = len(win) / fs

    # Blink detection: z-score with physiological constraints
    # prominence=2.5 -> only peaks >2.5 SD above local baseline
    # distance=300 ms -> minimum plausible inter-blink interval
    z = (win - np.mean(win)) / (np.std(win) + 1e-9)
    peaks, _ = find_peaks(z, prominence=2.5, distance=int(0.30 * fs))
    n_blinks = len(peaks)

    blink_rate_bpm = (n_blinks / dur_s) * 60
    mean_blink_amp = float(np.mean(z[peaks])) if n_blinks > 0 else 0.0

    # OCI: % of window time estimated as eyes-closed (200 ms assumed per blink)
    oci = min((n_blinks * 0.2 / dur_s) * 100.0, 100.0)

    rms = float(np.sqrt(np.mean(win ** 2)))
    std_dev = float(np.std(win))

    freqs = np.fft.rfftfreq(len(win), d=1.0 / fs)
    # Use z-score signal for PSD so energy features are amplitude-invariant
    psd = np.abs(np.fft.rfft(z)) ** 2
    dominant_freq = float(freqs[np.argmax(psd[1:]) + 1])  # skip DC bin
    blink_band = (freqs >= 0.1) & (freqs <= 0.5)
    energy_blink = float(np.sum(psd[blink_band]))

    return {
        "Blink_Rate_bpm":       blink_rate_bpm,
        "Mean_Blink_Amplitude": mean_blink_amp,
        "OCI":                  oci,
        "RMS":                  rms,
        "Std_Dev":              std_dev,
        "Dominant_Freq_Hz":     dominant_freq,
        "Energy_0.1_0.5Hz":    energy_blink,
    }


FEATURE_COLS = [
    "Blink_Rate_bpm", "Mean_Blink_Amplitude", "OCI",
    "RMS", "Std_Dev", "Dominant_Freq_Hz", "Energy_0.1_0.5Hz",
]


# ---------- Step 1: Segment-aware windowing and labelling ----------

print("Extracting features from labelled segments...")
rows = []
for seg_name, t_start, t_end, label in SEGMENTS:
    mask = (df["time_s"] >= t_start) & (df["time_s"] < t_end)
    sig = preprocess(df.loc[mask, "Amp_mV"].values)
    n_wins = len(sig) // WINDOW_N
    for i in range(n_wins):
        win = sig[i * WINDOW_N : (i + 1) * WINDOW_N]
        feat = extract_features(win)
        feat["Segment"] = seg_name
        feat["Fatigue_Label"] = label
        rows.append(feat)

features_df = pd.DataFrame(rows)

print(f"Total windows: {len(features_df)}")
print("Class distribution (0=no fatigue, 1=fatigue):")
print(features_df["Fatigue_Label"].value_counts().to_string())

output_path = os.path.join(dataset_dir, "Labeled_EOG_Dataset.csv")
features_df.to_csv(output_path, index=False)
print(f"\nLabeled dataset saved to: {output_path}")
print(features_df.drop(columns="Segment").head())


# ---------- Step 2: Train and compare models ----------

X = features_df[FEATURE_COLS].values
y = features_df["Fatigue_Label"].values

# Normalise so SVM and distance-based models are not dominated by scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified 3-fold CV - more robust than a single split for this dataset size
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

models = {
    "Random Forest":          RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Support Vector Machine": SVC(kernel="rbf", class_weight="balanced", random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(random_state=42),
}

best_name, best_score, best_model = "", 0.0, None

print("\n--- Model Comparison (3-fold stratified cross-validation) ---")
for name, mdl in models.items():
    scores = cross_val_score(mdl, X_scaled, y, cv=cv, scoring="f1_weighted")
    mean_f1 = scores.mean()
    print(f"Model: {name} | Weighted F1: {mean_f1:.4f} (+/- {scores.std():.4f})")
    if mean_f1 > best_score:
        best_score = mean_f1
        best_name = name
        best_model = mdl

print(f"\n[BEST] Best model: {best_name} | Weighted F1 = {best_score:.4f}")

# Final held-out evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nDetailed classification report (held-out 20% test set):")
print(classification_report(y_test, y_pred, zero_division=0))


# ---------- Step 3: Export the best model for deployment ----------
# The model above was fit on 80% only, and on features scaled with a scaler
# fitted outside it. For deployment we refit a fresh copy of the best model
# on ALL windows, inside a Pipeline together with its own StandardScaler.
# Then live code only needs:  pipeline.predict(raw_features)
import json
import joblib
import sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline

deploy_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", clone(best_model)),
])
X_df = features_df[FEATURE_COLS]          # keep column names, the pipeline remembers them
deploy_pipeline.fit(X_df, y)

basal = features_df[features_df["Segment"] == "basal"]
bundle = {
    "pipeline":      deploy_pipeline,       # scaler + model in one object
    "model_name":    best_name,
    "cv_f1_weighted": float(best_score),
    "features":      FEATURE_COLS,          # exact order the pipeline expects
    "fs":            FS,                    # live signal must be sampled at this rate
    "window_s":      WINDOW_S,
    "filter":        {"type": "butter", "order": 4, "band_hz": [0.3, 12.0],
                      "form": "sos", "zero_phase": True, "dc_removal": "mean"},
    "blink_detection": {"prominence_z": 2.5, "min_distance_s": 0.30},
    "units":         "mV at the electrodes (divide ADC mV by amplifier gain)",
    "labels":        {0: "no fatigue", 1: "fatigue"},
    # used by live_classify.py for --gain auto and for range warnings
    "basal_rms_median": float(basal["RMS"].median()),
    "train_min":     features_df[FEATURE_COLS].min().to_dict(),
    "train_max":     features_df[FEATURE_COLS].max().to_dict(),
    "sklearn_version": sklearn.__version__,  # load with the same version
}

model_path = "eog_model.joblib"
joblib.dump(bundle, model_path)

# Human-readable copy of the settings (everything except the model itself)
meta = {k: v for k, v in bundle.items() if k != "pipeline"}
with open("eog_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)

print(f"\n[EXPORT] {best_name} refit on all {len(y)} windows -> {model_path}")
print("[EXPORT] settings -> eog_model_meta.json")

# Quick self-check: reload and predict the first window
_loaded = joblib.load(model_path)
_pred = _loaded["pipeline"].predict(X_df.iloc[:1])[0]
print(f"[EXPORT] reload OK, first window predicted as {_loaded['labels'][int(_pred)]}")
