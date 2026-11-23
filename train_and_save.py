"""Train the doctor's model and SAVE it so the live script can use it.

Same preprocessing, 10 s windows, 7 features, scaler and models as train.py.
What is different, because the data is very unbalanced (1 minute of rest vs
~6 minutes of fixation):

  * Overlapping windows for the REST class (a new window every 2 s instead
    of every 10 s). This gives more rest windows, but they are the SAME
    minute of signal, not new information.
  * Honest cross-validation: each segment is cut into 20 s blocks and
    windows never cross a block. All windows from one block stay together
    in train or in test (StratifiedGroupKFold), so overlapping copies of a
    signal cannot leak from train into test.
  * Reports balanced accuracy, recall per class and a confusion matrix
    (weighted F1 hides a model that just says "fatigue" every time).
    The best model is picked by balanced accuracy.
  * Optional extra data from your own recordings (live_classify.py --record):
        --rest  me_rest1.csv me_rest2.csv   -> label 0
        --tired me_tired.csv                -> label 1
    These need --gain (the number printed by live_classify --gain auto).
  * The final model is refit on ALL windows and saved to eog_model.joblib.

  * Baseline-relative mode (--baseline-relative): every window's features are
    divided by the median features of the basal (rest) minute, and the model
    is trained on these ratios. Live, the server divides each person's
    features by THEIR OWN recorded rest baseline. --hp-fc 1.0 first passes the
    dataset through a 1st-order high-pass that imitates the BioAmp EXG Pill's
    built-in filter (estimated from blink shape).

Usage:
  python train_and_save.py --csv EOG_raw_signal.csv
  python train_and_save.py --csv EOG_raw_signal.csv --rest me_rest.csv --gain 950
  python train_and_save.py --baseline-relative --hp-fc 1.0 --model svm \\
      --out eog_model_baseline.joblib --features-csv Labeled_EOG_Dataset_baseline.csv
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from eog_features import FEATURE_COLS, FS, WINDOW_N, WINDOW_S, extract_features, preprocess

SEGMENTS = [
    ("basal",         0,   60,  0),
    ("participant_1", 68,  220, 1),
    ("participant_2", 291, 374, 1),
    ("participant_3", 442, 498, 1),
    ("participant_4", 532, 622, 1),
]
BLOCK_S = 20            # CV block length (seconds)
STEP_REST_S = 2         # window step for rest (overlapping)
STEP_TIRED_S = 10       # window step for fatigue (no overlap, like train.py)
EPS = 1e-9


def ratio_features(feats, base):
    """Divide each feature column by the baseline median (guarding against 0)."""
    out = feats[FEATURE_COLS].astype(float).copy()
    for c in FEATURE_COLS:
        b = float(base[c])
        out[c] = out[c] / (b if abs(b) > EPS else EPS)
    return out


def windows_from_signal(sig_mv, label, source, step_s):
    """Filter a whole segment, cut it in 20 s blocks, then into 10 s windows."""
    sig = preprocess(sig_mv)
    block_n, step_n = BLOCK_S * FS, step_s * FS
    rows = []
    for b in range(len(sig) // block_n):
        block = sig[b * block_n:(b + 1) * block_n]
        for start in range(0, block_n - WINDOW_N + 1, step_n):
            f = extract_features(block[start:start + WINDOW_N])
            f.update(Fatigue_Label=label, Source=source, Group=f"{source}#b{b}")
            rows.append(f)
    return rows


def load_own_recording(path, gain, invert=False):
    """Recording CSV (t_us,adc_mV; from the server's Record button) -> mV at the electrodes."""
    d = pd.read_csv(path)
    mv = d.iloc[:, 1].to_numpy(dtype=float)
    fs_real = 1e6 / np.median(np.diff(d.iloc[:, 0].to_numpy()))
    if abs(fs_real - FS) > 50:
        print(f"  !! {path}: sampling rate {fs_real:.0f} Hz, expected {FS}")
    sig = (mv - mv.mean()) / gain
    return -sig if invert else sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/EOG_raw_signal.csv")
    ap.add_argument("--out", default="eog_model.joblib")
    ap.add_argument("--rest", nargs="*", default=[], help="your own rest recordings (label 0)")
    ap.add_argument("--tired", nargs="*", default=[], help="your own tired recordings (label 1)")
    ap.add_argument("--gain", type=float, help="needed with --rest/--tired")
    ap.add_argument("--invert", action="store_true",
                    help="flip your own recordings (same as EOG_INVERT=1 on the server)")
    ap.add_argument("--baseline-relative", action="store_true",
                    help="train on features divided by the basal-minute medians")
    ap.add_argument("--hp-fc", type=float, default=None,
                    help="pass the dataset through a 1st-order high-pass at this frequency (Hz) first")
    ap.add_argument("--model", choices=["best", "svm", "rf", "gb"], default="best",
                    help="model to save (default: best balanced accuracy)")
    ap.add_argument("--features-csv", default="Labeled_EOG_Dataset.csv",
                    help="where to write the per-window feature table")
    args = ap.parse_args()
    if (args.rest or args.tired) and not args.gain:
        ap.error("--rest/--tired need --gain")
    if args.baseline_relative and (args.rest or args.tired):
        ap.error("--baseline-relative cannot be combined with --rest/--tired yet "
                 "(own recordings would need their own baselines)")
    if args.hp_fc is not None and not args.hp_fc > 0:
        ap.error("--hp-fc must be > 0")

    df = pd.read_csv(args.csv, encoding="latin1")
    amp = df["Amp_mV"].to_numpy(dtype=float)
    if args.hp_fc:
        # causal 1st-order high-pass on the whole recording, like the analog filter in the EXG Pill
        amp = sosfilt(butter(1, args.hp_fc, btype="high", fs=FS, output="sos"), amp - amp.mean())
        print(f"dataset passed through a 1st-order {args.hp_fc:g} Hz high-pass")
    rows = []
    for name, t0, t1, label in SEGMENTS:
        m = ((df["time_s"] >= t0) & (df["time_s"] < t1)).to_numpy()
        step = STEP_REST_S if label == 0 else STEP_TIRED_S
        rows += windows_from_signal(amp[m], label, name, step)
    for paths, label, step in ((args.rest, 0, STEP_REST_S), (args.tired, 1, STEP_TIRED_S)):
        for p in paths:
            rows += windows_from_signal(load_own_recording(p, args.gain, args.invert), label, f"own:{p}", step)

    feats = pd.DataFrame(rows)
    feats.to_csv(args.features_csv, index=False)
    basal_medians = feats.loc[feats["Source"] == "basal", FEATURE_COLS].median()
    print("windows per class (0 = rest, 1 = fatigue):")
    print(feats["Fatigue_Label"].value_counts().sort_index().to_string())
    print("windows per source:")
    print(feats["Source"].value_counts().to_string())

    Xdf = ratio_features(feats, basal_medians) if args.baseline_relative else feats[FEATURE_COLS]
    if args.baseline_relative:
        print("baseline-relative: features divided by basal medians",
              {c: round(float(basal_medians[c]), 4) for c in FEATURE_COLS})
    X = Xdf.values
    y = feats["Fatigue_Label"].values
    groups = feats["Group"].values
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    print("\n--- 3-fold CV by 20 s blocks (no overlap leak) ---")
    print("A model that always says 'fatigue' gets balanced accuracy 0.50.")
    best_name, best_score = None, -1.0
    cv_scores = {}
    for name, mdl in models.items():
        pipe = make_pipeline(StandardScaler(), mdl)  # scaler fitted inside each fold
        pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups)
        bal = balanced_accuracy_score(y, pred)
        cv_scores[name] = bal
        r0, r1 = recall_score(y, pred, average=None, labels=[0, 1])
        cm = confusion_matrix(y, pred, labels=[0, 1])
        print(f"\n{name}\n  balanced accuracy = {bal:.2f}"
              f"\n  recall rest = {r0:.2f}   recall fatigue = {r1:.2f}"
              f"\n  confusion [rows = true 0/1, cols = predicted 0/1]:\n    {cm[0]}\n    {cm[1]}")
        if bal > best_score:
            best_name, best_score = name, bal
    print(f"\n[BEST] {best_name}  balanced accuracy = {best_score:.2f}")
    if args.model != "best":
        chosen = {"svm": "Support Vector Machine", "rf": "Random Forest", "gb": "Gradient Boosting"}[args.model]
        if chosen != best_name:
            print(f"[CHOSEN] {chosen} (requested with --model {args.model})")
        best_name = chosen
        best_score = cv_scores[chosen]
    if not args.rest:
        print("Note: all rest windows come from ONE minute at the start of the recording,\n"
              "so the model may be learning 'start of recording' rather than 'rested'.\n"
              "Adding your own rest recordings (--rest) is the real fix.")

    pipeline = make_pipeline(StandardScaler(), models[best_name]).fit(Xdf, y)
    scaler, model = pipeline[0], pipeline[-1]

    ranges = feats.groupby("Fatigue_Label")[FEATURE_COLS].agg(["min", "median", "max"])
    print("\nfeature ranges per class:\n", ranges.T.to_string())

    basal = feats[feats["Source"] == "basal"]
    joblib.dump({
        "model": model, "model_name": best_name, "scaler": scaler,
        "pipeline": pipeline,   # scaler + model in one, used by server.py and predict_sensor.py
        "labels": {0: "no fatigue", 1: "fatigue"},
        "features": FEATURE_COLS, "fs": FS, "window_s": WINDOW_S,
        # ranges in the space the model sees (ratios when baseline_relative)
        "train_min": Xdf.min().to_dict(),
        "train_max": Xdf.max().to_dict(),
        # median RMS of the dataset's rest minute, used for gain calibration
        "basal_rms_median": float(basal["RMS"].median()),
        "cv_balanced_accuracy": best_score,
        # baseline-relative mode: live features must be divided by the person's own rest medians
        "baseline_relative": bool(args.baseline_relative),
        "basal_medians": {c: float(basal_medians[c]) for c in FEATURE_COLS},
        "hp_fc": args.hp_fc,
    }, args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
