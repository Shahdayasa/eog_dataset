"""Predict fatigue from an ESP32 EOG CSV using the exported model.

The CSV must contain ``time_s`` and ``value`` columns. Supply
``--electrode-mv-per-count``: the calibrated conversion from one raw ``value``
unit to millivolts at the electrodes, including the analog amplifier gain.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eog_features import extract_features, preprocess


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "test_20261117_193811.csv",
        help="sensor CSV (default: test_20261117_193811.csv beside this script)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "eog_model.joblib",
        help="exported model bundle (default: eog_model.joblib beside this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="prediction CSV (default: <input name>_predictions.csv)",
    )
    parser.add_argument(
        "--electrode-mv-per-count",
        type=float,
        required=True,
        help=(
            "calibrated electrode mV per unit in the CSV's value column, "
            "including ADC conversion and amplifier gain"
        ),
    )
    args = parser.parse_args()

    if not np.isfinite(args.electrode_mv_per_count) or args.electrode_mv_per_count <= 0:
        parser.error("--electrode-mv-per-count must be a finite positive number")
    if not args.input.is_file():
        parser.error(f"input CSV not found: {args.input}")
    if not args.model.is_file():
        parser.error(f"model file not found: {args.model}")

    bundle = joblib.load(args.model)
    required_keys = {"pipeline", "features", "fs", "window_s"}
    missing_keys = required_keys.difference(bundle)
    if missing_keys:
        raise ValueError(f"model bundle is missing required keys: {sorted(missing_keys)}")

    fs = int(bundle["fs"])
    window_s = float(bundle["window_s"])
    window_n = int(round(fs * window_s))
    if fs <= 0 or window_s <= 0 or window_n <= 0:
        raise ValueError("model bundle contains an invalid sampling rate or window length")

    df = pd.read_csv(args.input)
    missing_columns = {"time_s", "value"}.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"input CSV is missing required columns: {sorted(missing_columns)}"
        )

    times = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("input CSV contains missing or non-numeric time_s/value data")
    if np.any(np.diff(times) < 0):
        raise ValueError("time_s values must be in chronological order")

    # Multiple ADC samples can have the same timestamp due to timestamp rounding.
    samples = pd.DataFrame({"time_s": times, "value": values})
    samples = samples.groupby("time_s", as_index=False, sort=True)["value"].mean()
    times = samples["time_s"].to_numpy(dtype=float)
    values = samples["value"].to_numpy(dtype=float)
    if len(times) < 2 or times[-1] <= times[0]:
        raise ValueError("input CSV must contain at least two distinct timestamps")

    duration_s = float(times[-1] - times[0])
    sample_count = int(np.floor(duration_s * fs)) + 1
    if sample_count < window_n:
        available_s = sample_count / fs
        raise ValueError(
            f"{args.input} contains only {duration_s:.3f} seconds "
            f"({available_s:.3f} seconds at {fs} Hz); the exported model "
            f"requires {window_n} samples ({window_s:g} seconds) per "
            "prediction. No prediction was produced."
        )

    uniform_times = times[0] + np.arange(sample_count, dtype=float) / fs
    uniform_values = np.interp(uniform_times, times, values)

    # Convert raw sensor units into the electrode-mV units used for training.
    signal_mv = (uniform_values - np.mean(uniform_values)) * args.electrode_mv_per_count
    filtered = preprocess(signal_mv, fs=fs)

    feature_names = list(bundle["features"])
    expected_output = args.output or args.input.with_name(
        f"{args.input.stem}_predictions.csv"
    )
    predictions = []
    for window_index, start in enumerate(range(0, len(filtered) - window_n + 1, window_n)):
        window = filtered[start : start + window_n]
        features = extract_features(window, fs=fs)
        feature_frame = pd.DataFrame(
            [[features[name] for name in feature_names]],
            columns=feature_names,
        )
        prediction = int(bundle["pipeline"].predict(feature_frame)[0])
        labels = bundle.get("labels", {})
        predictions.append(
            {
                "window_start_s": float(times[0] + window_index * window_s),
                "window_end_s": float(times[0] + (window_index + 1) * window_s),
                "predicted_label": prediction,
                "predicted_class": labels.get(prediction, str(prediction)),
                **features,
            }
        )

    if not predictions:
        raise ValueError("no complete prediction window could be formed from the input")

    result = pd.DataFrame(predictions)
    result.to_csv(expected_output, index=False)
    print(f"Model: {bundle.get('model_name', 'unknown')}")
    print(f"Predictions saved to: {expected_output}")
    print(result[["window_start_s", "window_end_s", "predicted_class"]].to_string(index=False))
    print("\nWindow counts by predicted class:")
    print(result["predicted_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
