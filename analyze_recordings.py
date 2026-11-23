"""Evaluate the fatigue models offline on recordings from the server's Record buttons.

For each person, the recordings are cut into 10 s windows (as live), every window
gets the same signal-quality check as the server, and:
  * the baseline model (eog_model_baseline.joblib) divides features by the person's
    own rest baseline (BASELINE_S of clean REST windows), like the live baseline step;
  * the old model (eog_model.joblib) uses absolute features (needs the gain).
The baseline is taken twice: from the START and from the END of the rest data,
to show how much the result depends on which minutes are the baseline.

File labels come from the file name: "tired" -> fatigue, anything else -> rest.

Usage (one --person per person; list that person's files in time order):
  python analyze_recordings.py \\
      --person you  recordings/esp32-01_rest_20261122_124717.csv recordings/esp32-01_rest_20261122_124947.csv \\
      --person B    recordings/esp32-01_rest_20261122_174325.csv recordings/B_tired_1.csv

Writes analysis/window_results.csv (one row per window) and prints a summary.
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eog_features import FEATURE_COLS, extract_features, preprocess
from eog_quality import quality_problems

HERE = Path(__file__).resolve().parent
EPS = 1e-9


def file_windows(path, gain, sign, fs, win_n):
    """10 s non-overlapping windows of one recording: features + quality."""
    d = pd.read_csv(path)
    x = d.iloc[:, 1].to_numpy(dtype=float)              # adc_mV
    if len(x) < win_n:
        return []
    filtered = preprocess(sign * (x - x.mean()) / gain, fs=fs)
    label = 1 if "tired" in Path(path).name.lower() else 0
    rows = []
    for k, s in enumerate(range(0, len(x) - win_n + 1, win_n)):
        f = extract_features(filtered[s:s + win_n], fs=fs)
        rows.append({"file": Path(path).name, "label": label, "t_start_s": s / fs, "window_in_file": k,
                     "problems": "; ".join(quality_problems(x[s:s + win_n], f, fs=fs)), **f})
    return rows


def ratios(F, med):
    return pd.DataFrame({c: F[c] / (med[c] if abs(med[c]) > EPS else EPS) for c in FEATURE_COLS})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--person", nargs="+", action="append", required=True, metavar=("NAME", "FILE"),
                    help="person name followed by that person's recordings in time order")
    ap.add_argument("--baseline-model", default=str(HERE / "eog_model_baseline.joblib"))
    ap.add_argument("--old-model", default=str(HERE / "eog_model.joblib"))
    ap.add_argument("--baseline-s", type=int, default=120, help="baseline length in seconds (default 120)")
    ap.add_argument("--gain", type=float, help="default: calibration.json")
    ap.add_argument("--invert", choices=["yes", "no"], help="default: calibration.json (yes)")
    ap.add_argument("--out", default=str(HERE / "analysis" / "window_results.csv"))
    args = ap.parse_args()

    cal = json.loads((HERE / "calibration.json").read_text()) if (HERE / "calibration.json").is_file() else {}
    gain = args.gain or cal.get("gain")
    if not gain:
        ap.error("no gain: pass --gain or create calibration.json with the Calibrate button")
    invert = (args.invert == "yes") if args.invert else bool(cal.get("invert", True))
    sign = -1.0 if invert else 1.0

    newb, oldb = joblib.load(args.baseline_model), joblib.load(args.old_model)
    if not newb.get("baseline_relative"):
        ap.error(f"{args.baseline_model} is not a baseline-relative model")
    fs, win_n = int(newb["fs"]), int(round(newb["fs"] * newb["window_s"]))
    n_base = args.baseline_s * fs // win_n
    print(f"gain {gain:.0f}, invert={invert}, baseline {args.baseline_s} s = {n_base} windows, "
          f"models: {Path(args.baseline_model).name} vs {Path(args.old_model).name}\n")

    all_rows, summary = [], []
    for spec in args.person:
        if len(spec) < 2:
            ap.error("--person needs a name and at least one file")
        name, files = spec[0], spec[1:]
        rows = []
        for f in files:
            if not os.path.isfile(f):
                ap.error(f"file not found: {f}")
            rows += file_windows(f, gain, sign, fs, win_n)
        W = pd.DataFrame(rows)
        W.insert(0, "person", name)
        W["clean"] = W["problems"] == ""
        print(f"{name}: {len(W)} windows ({(W.label == 0).sum()} rest, {(W.label == 1).sum()} tired), "
              f"{(~W.clean).sum()} rejected by the quality check")
        for reason, n in W.loc[~W.clean, "problems"].value_counts().items():
            print(f"    {n} x {reason}")

        old_pred = np.full(len(W), np.nan)
        if W.clean.any():
            old_pred[W.clean.to_numpy()] = oldb["pipeline"].predict(W.loc[W.clean, FEATURE_COLS])
        W["old_pred"] = old_pred

        rest_clean = W.index[(W.label == 0) & W.clean].tolist()
        for strategy in ("start", "end"):
            base_idx = rest_clean[:n_base] if strategy == "start" else rest_clean[-n_base:]
            col = f"new_pred_baseline_{strategy}"
            W[col] = np.nan
            if len(base_idx) < max(3, n_base // 2):
                print(f"    baseline from {strategy}: not enough clean rest windows ({len(base_idx)})")
                continue
            med = W.loc[base_idx, FEATURE_COLS].median()
            test = W.index[W.clean & ~W.index.isin(base_idx)]
            if len(test):
                W.loc[test, col] = newb["pipeline"].predict(ratios(W.loc[test], med)[FEATURE_COLS])
            W.loc[base_idx, col] = -1                     # marks baseline windows
            for lab, lab_name in ((0, "rest"), (1, "tired")):
                t = W.loc[test][W.loc[test, "label"] == lab]
                if len(t) == 0:
                    continue
                correct = lambda p: float(np.mean(p == lab)) if len(p) else np.nan
                summary.append({
                    "person": name, "baseline": strategy, "true": lab_name, "test windows": len(t),
                    "new model correct": f"{int((t[col] == lab).sum())}/{len(t)} ({correct(t[col]) * 100:.0f}%)",
                    "old model correct": f"{int((t['old_pred'] == lab).sum())}/{len(t)} "
                                         f"({correct(t['old_pred']) * 100:.0f}%)",
                })
        all_rows.append(W)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
    print("\n" + (pd.DataFrame(summary).to_string(index=False) if summary else "no test windows"))
    print("\n'correct' = rest windows called 'no fatigue', tired windows called 'fatigue'.")
    if not any(s["true"] == "tired" for s in summary):
        print("No tired recordings yet: this only measures false alarms at rest, not fatigue detection.")
    print(f"per-window results: {out}")


if __name__ == "__main__":
    main()
