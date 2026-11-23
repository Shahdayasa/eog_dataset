"""Server: receives raw EOG samples from the ESP32 and predicts fatigue.

The ESP32 POSTs batches of samples to /ingest. Every 10 s of continuous
signal (one model window) the server runs the SAME preprocessing, features
and model as training (eog_features.py + eog_model.joblib) and stores the
result. Live page: /view?device=esp32-01&key=<API key>

Wire format of POST /ingest
    Headers:  X-API-Key    shared secret (env EOG_API_KEY)
              X-Device-Id  any short name, e.g. "esp32-01"
              X-Seq        index of the first sample in this batch
                           (counts every sample since the ESP32 booted)
    Body:     N little-endian uint16 values = ADC input in millivolts
              (analogReadMilliVolts on the ESP32), N <= 5000

If X-Seq does not continue the previous batch (lost batch, reboot), the
buffer is cleared, because a window must be continuous signal.

Gain calibration (/calibrate): the user rests for 60 s; the gain is chosen so
that the median resting RMS matches the training data's rest minute
(basal_rms_median in the model bundle). The applied gain is saved to
calibration.json and used on the next start.

Recording (/record): saves the incoming signal to recordings/*.csv
(t_us,adc_mV), the format train_and_save.py --rest/--tired reads.

Environment variables
    EOG_API_KEY    required. Shared secret, also put it in the ESP32 sketch.
    EOG_AMP_GAIN   BioAmp gain: electrode mV = ADC mV / gain. Required unless
                   calibration.json exists (a calibrated gain overrides it).
    EOG_MODEL      optional. Path to the model bundle (default eog_model.joblib).
                   eog_model_baseline.joblib = baseline-relative model: each person first
                   records a rested baseline (/baseline), live features are divided by it,
                   and windows with a bad signal are not classified.
    EOG_BASELINE_S optional. Baseline length in seconds (default 120).
    EOG_INVERT     optional. 1 = flip the signal (use when blinks are downward dips).

Run:  uvicorn server:app --host 0.0.0.0 --port 8000
"""
import hmac
import json
import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from sklearn.pipeline import make_pipeline

from eog_features import extract_features, preprocess
from eog_quality import ADC_HIGH_MV, ADC_LOW_MV, quality_problems

HERE = Path(__file__).resolve().parent
MAX_BATCH_N = 5000          # samples per POST (5 s at 1000 Hz)
HISTORY_LEN = 360           # predictions kept per device (1 h at 10 s)
CALIB_S = 60                # resting time for gain calibration
CALIB_FILE = HERE / "calibration.json"
RECORD_DIR = HERE / "recordings"


def _required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"environment variable {name} must be set")
    return value


def _valid_gain(g):
    return isinstance(g, (int, float)) and math.isfinite(g) and g > 0


API_KEY = _required_env("EOG_API_KEY")

# Blinks must point UP, as in the training data. If your electrodes are wired the
# other way round (blinks are downward dips), set EOG_INVERT=1.
INVERT = os.environ.get("EOG_INVERT", "0").strip().lower() in ("1", "true", "yes")
SIGN = -1.0 if INVERT else 1.0

MODEL_PATH = Path(os.environ.get("EOG_MODEL", HERE / "eog_model.joblib"))
BUNDLE = joblib.load(MODEL_PATH)
_missing = {"features", "fs", "window_s"}.difference(BUNDLE)
if _missing:
    raise RuntimeError(f"model bundle is missing keys: {sorted(_missing)}")
if "pipeline" in BUNDLE:
    PREDICTOR = BUNDLE["pipeline"]
elif {"scaler", "model"} <= set(BUNDLE):           # older train_and_save.py bundles
    PREDICTOR = make_pipeline(BUNDLE["scaler"], BUNDLE["model"])
else:
    raise RuntimeError("model bundle has neither 'pipeline' nor 'scaler' + 'model'")

FS = int(BUNDLE["fs"])
WINDOW_S = float(BUNDLE["window_s"])
WINDOW_N = int(round(FS * WINDOW_S))
CONTEXT_N = 2 * WINDOW_N    # filter over 20 s (like the 20 s training blocks), predict on the last 10 s
FEATURES = list(BUNDLE["features"])
LABELS = BUNDLE.get("labels", {0: "no fatigue", 1: "fatigue"})
BASAL_RMS = BUNDLE.get("basal_rms_median")
TRAIN_MIN, TRAIN_MAX = BUNDLE.get("train_min", {}), BUNDLE.get("train_max", {})

# Baseline-relative models (train_and_save.py --baseline-relative): live features are
# divided by the person's own rest medians, recorded with the "Record baseline" step.
BASELINE_MODE = bool(BUNDLE.get("baseline_relative"))
BASELINE_S = int(os.environ.get("EOG_BASELINE_S", "120"))
MIN_BASELINE_WINDOWS = max(3, BASELINE_S // 20)   # at least half of the 10 s windows must be clean
EPS = 1e-9
if BASELINE_MODE and BASELINE_S < 3 * WINDOW_S:
    raise RuntimeError("EOG_BASELINE_S must be at least 3 windows (30 s)")

# Signal-quality limits per 10 s window live in eog_quality.py (shared with analyze_recordings.py).

# Gain: calibration.json (from the Calibrate button) wins over EOG_AMP_GAIN.
_gain_lock = threading.Lock()
AMP_GAIN, GAIN_SOURCE = None, None
if CALIB_FILE.is_file():
    _cal = json.loads(CALIB_FILE.read_text())
    if _valid_gain(_cal.get("gain")):
        AMP_GAIN, GAIN_SOURCE = float(_cal["gain"]), f"calibration.json ({_cal.get('saved_at', '?')})"
if AMP_GAIN is None:
    AMP_GAIN = float(_required_env("EOG_AMP_GAIN"))
    GAIN_SOURCE = "EOG_AMP_GAIN"
    if not _valid_gain(AMP_GAIN):
        raise RuntimeError("EOG_AMP_GAIN must be a finite positive number")
print(f"[server] gain {AMP_GAIN:g} from {GAIN_SOURCE}, invert={INVERT}, model={MODEL_PATH.name}, "
      f"baseline_relative={BASELINE_MODE}")


def _label_name(label):
    return str(LABELS.get(label, LABELS.get(str(label), label)))


class DeviceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.buf = deque(maxlen=CONTEXT_N)  # ADC millivolts
        self.expected_seq = None
        self.since_pred = 0                 # samples received since the last window
        self.gaps = 0
        self.samples_total = 0
        self.last_seen = None
        self.history = deque(maxlen=HISTORY_LEN)
        # calibration
        self.calib = None                   # list of arrays while collecting
        self.calib_n = 0
        self.calib_result = None
        # recording
        self.rec_file = None
        self.rec_path = None
        self.rec_label = None
        self.rec_n = 0
        self.rec_gaps = 0
        self.rec_started = None
        # personal rest baseline (baseline-relative models)
        self.base_collect = None            # list of arrays while recording the baseline
        self.base_n = 0
        self.baseline = None                # {"medians": {...}, ...} once recorded
        self.base_error = None


_devices = {}
_devices_lock = threading.Lock()


def _get_device(device_id, create=True):
    with _devices_lock:
        if create:
            return _devices.setdefault(device_id, DeviceState())
        dev = _devices.get(device_id)
    if dev is None:
        raise HTTPException(status_code=404, detail="unknown device (is the ESP32 sending?)")
    return dev


def _check_key(key):
    if not key or not hmac.compare_digest(key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid API key")


def _out_of_range(feats):
    return [k for k, v in feats.items()
            if k in TRAIN_MIN and not (TRAIN_MIN[k] <= v <= TRAIN_MAX[k])]


def _quality_problems(raw_window_adc, feats):
    """Reasons this 10 s window is not usable (empty list = OK)."""
    return quality_problems(raw_window_adc, feats, fs=FS)


def _ratio_features(feats, medians):
    return {n: feats[n] / (medians[n] if abs(medians[n]) > EPS else EPS) for n in FEATURES}


def _window_features(adc_mv, gain):
    """Features of the last 10 s of adc_mv (filtered over the whole context)."""
    signal_mv = SIGN * np.asarray(adc_mv, dtype=float) / gain   # electrode mV, as in training
    window = preprocess(signal_mv, fs=FS)[-WINDOW_N:]
    return extract_features(window, fs=FS)


def _predict(adc_mv, end_seq, baseline=None):
    """adc_mv: up to CONTEXT_N samples ending at the window end."""
    with _gain_lock:
        gain = AMP_GAIN
    feats = _window_features(adc_mv, gain)
    problems = _quality_problems(np.asarray(adc_mv)[-WINDOW_N:], feats)

    model_input, label, status = None, None, "ok"
    if BASELINE_MODE:
        if problems:
            status = "bad_signal"
        elif baseline is None:
            status = "needs_baseline"
        else:
            model_input = _ratio_features(feats, baseline["medians"])
    else:
        model_input = feats            # old model: always predicts; quality is only reported
    if model_input is not None:
        frame = pd.DataFrame([[model_input[n] for n in FEATURES]], columns=FEATURES)
        label = int(PREDICTOR.predict(frame)[0])

    names = {"bad_signal": "bad signal", "needs_baseline": "needs baseline"}
    return {
        "received_at": time.time(),
        "window_start_s": (end_seq - WINDOW_N) / FS,   # device time since boot
        "window_end_s": end_seq / FS,
        "status": status,
        "predicted_label": label,
        "predicted_class": _label_name(label) if label is not None else names[status],
        "quality_problems": problems,
        "gain": gain,
        "features": feats,
        "model_input": model_input,    # ratios in baseline mode
        "out_of_training_range": _out_of_range(model_input) if model_input else [],
    }


def _compute_baseline(adc_mv, gain):
    """Median features of the clean 10 s windows of a rest recording."""
    adc_mv = np.asarray(adc_mv, dtype=float)
    filtered = preprocess(SIGN * adc_mv / gain, fs=FS)
    rows, rejected = [], []
    for s in range(0, len(adc_mv) - WINDOW_N + 1, WINDOW_N):
        f = extract_features(filtered[s:s + WINDOW_N], fs=FS)
        p = _quality_problems(adc_mv[s:s + WINDOW_N], f)
        (rejected.append(p) if p else rows.append(f))
    total = len(rows) + len(rejected)
    if len(rows) < MIN_BASELINE_WINDOWS:
        reasons = sorted({r for p in rejected for r in p})
        return None, (f"only {len(rows)} of {total} windows were clean (need {MIN_BASELINE_WINDOWS}). "
                      f"Problems: {'; '.join(reasons) or 'none'}. Check the electrodes and repeat.")
    med = pd.DataFrame(rows)[FEATURES].median()
    return {"medians": {n: float(med[n]) for n in FEATURES}, "clean_windows": len(rows),
            "total_windows": total, "finished_at": time.time(), "gain": gain}, None


def _compute_calibration(adc_mv):
    """Gain that makes the median resting RMS equal the training rest RMS."""
    adc_mv = np.asarray(adc_mv, dtype=float)
    filt = preprocess(adc_mv, fs=FS)                 # sign does not matter for RMS
    step = 2 * FS                                    # 10 s windows every 2 s, as for training rest
    rms = [float(np.sqrt(np.mean(filt[s:s + WINDOW_N] ** 2)))
           for s in range(0, len(filt) - WINDOW_N + 1, step)]
    rms_med = float(np.median(rms))
    clipped = float(np.mean((adc_mv <= ADC_LOW_MV) | (adc_mv >= ADC_HIGH_MV)) * 100)
    result = {
        "finished_at": time.time(),
        "seconds": len(adc_mv) / FS,
        "rest_rms_adc_mv": rms_med,
        "train_rest_rms_mv": BASAL_RMS,
        "suggested_gain": rms_med / BASAL_RMS if BASAL_RMS and rms_med > 0 else None,
        "clipped_percent": clipped,
        "warnings": [],
    }
    if clipped > 1:
        result["warnings"].append(f"{clipped:.1f}% of samples hit the ADC limits; blink gently and repeat")
    if rms_med <= 0:
        result["warnings"].append("flat signal; check the electrodes and the BioAmp wiring")
    if len(rms) > 1 and np.max(rms) > 3 * np.min(rms):
        result["warnings"].append("signal size changed a lot during the minute (movement?); consider repeating")
    return result


def _close_recording(dev):
    if dev.rec_file is not None:
        dev.rec_file.close()
    info = {"path": str(dev.rec_path), "label": dev.rec_label,
            "seconds": dev.rec_n / FS, "gaps": dev.rec_gaps}
    dev.rec_file = dev.rec_path = dev.rec_label = dev.rec_started = None
    dev.rec_n = dev.rec_gaps = 0
    return info


app = FastAPI(title="EOG fatigue server")


@app.get("/health")
def health():
    return {"ok": True, "model": BUNDLE.get("model_name", "unknown"), "model_file": MODEL_PATH.name,
            "fs": FS, "window_s": WINDOW_S, "amp_gain": AMP_GAIN, "gain_source": GAIN_SOURCE,
            "invert": INVERT, "baseline_relative": BASELINE_MODE, "baseline_s": BASELINE_S,
            "hp_fc": BUNDLE.get("hp_fc")}


@app.post("/ingest")
async def ingest(
    request: Request,
    x_api_key: str = Header(None),
    x_device_id: str = Header(...),
    x_seq: int = Header(...),
):
    _check_key(x_api_key)
    if not 0 < len(x_device_id) <= 64:
        raise HTTPException(status_code=400, detail="X-Device-Id must be 1-64 characters")
    if x_seq < 0:
        raise HTTPException(status_code=400, detail="X-Seq must be >= 0")

    body = await request.body()
    if not body or len(body) % 2 or len(body) // 2 > MAX_BATCH_N:
        raise HTTPException(status_code=400, detail=f"body must be 1-{MAX_BATCH_N} uint16 samples")
    raw_u16 = np.frombuffer(body, dtype="<u2")
    samples = raw_u16.astype(float)

    dev = _get_device(x_device_id)
    new_predictions = []
    with dev.lock:
        gap = dev.expected_seq is not None and x_seq != dev.expected_seq
        if gap:
            dev.gaps += 1
        if gap or dev.expected_seq is None:
            dev.buf.clear()
            dev.since_pred = 0
            if dev.calib is not None:            # calibration needs one continuous minute
                dev.calib, dev.calib_n = [], 0
            if dev.base_collect is not None:     # so does the baseline
                dev.base_collect, dev.base_n = [], 0
            if dev.rec_file is not None and gap:
                dev.rec_gaps += 1

        dev.buf.extend(samples)
        dev.expected_seq = x_seq + len(samples)
        dev.since_pred += len(samples)
        dev.samples_total += len(samples)
        dev.last_seen = time.time()

        if dev.rec_file is not None:
            t_us = (x_seq + np.arange(len(raw_u16), dtype=np.int64)) * (1_000_000 // FS)
            np.savetxt(dev.rec_file, np.column_stack([t_us, raw_u16.astype(np.int64)]),
                       fmt="%d", delimiter=",")
            dev.rec_n += len(raw_u16)

        if dev.calib is not None:
            dev.calib.append(samples)
            dev.calib_n += len(samples)
            if dev.calib_n >= CALIB_S * FS:
                data = np.concatenate(dev.calib)[: CALIB_S * FS]
                dev.calib, dev.calib_n = None, 0
                dev.calib_result = _compute_calibration(data)

        if dev.base_collect is not None:
            dev.base_collect.append(samples)
            dev.base_n += len(samples)
            if dev.base_n >= BASELINE_S * FS:
                data = np.concatenate(dev.base_collect)[: BASELINE_S * FS]
                dev.base_collect, dev.base_n = None, 0
                with _gain_lock:
                    gain = AMP_GAIN
                dev.baseline, dev.base_error = _compute_baseline(data, gain)

        if dev.since_pred >= WINDOW_N:
            buf = np.fromiter(dev.buf, dtype=float, count=len(dev.buf))
            while dev.since_pred >= WINDOW_N:
                # The window ends this many samples before the newest sample.
                extra = dev.since_pred - WINDOW_N
                end = len(buf) - extra
                result = _predict(buf[max(0, end - CONTEXT_N):end], dev.expected_seq - extra, dev.baseline)
                dev.history.append(result)
                new_predictions.append(result)
                dev.since_pred -= WINDOW_N

        buffered_s = min(dev.since_pred, len(dev.buf)) / FS

    latest = new_predictions[-1] if new_predictions else None
    return {
        "ok": True,
        "gap_detected": gap,
        "seconds_until_next_prediction": round(WINDOW_S - buffered_s, 3),
        "prediction": None if latest is None else latest["predicted_class"],
    }


@app.get("/devices")
def devices(x_api_key: str = Header(None)):
    _check_key(x_api_key)
    with _devices_lock:
        items = list(_devices.items())
    return [
        {"device_id": d, "last_seen": s.last_seen, "samples_total": s.samples_total, "gaps": s.gaps}
        for d, s in items
    ]


@app.get("/raw/{device_id}")
def raw(device_id: str, x_api_key: str = Header(None)):
    """Diagnostics: the buffered raw ADC millivolts (up to the last 20 s)."""
    _check_key(x_api_key)
    dev = _get_device(device_id, create=False)
    with dev.lock:
        return {"fs": FS, "amp_gain": AMP_GAIN, "invert": INVERT, "adc_mv": [int(v) for v in dev.buf]}


@app.get("/latest/{device_id}")
def latest(device_id: str, limit: int = 1, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    dev = _get_device(device_id, create=False)
    limit = max(1, min(limit, HISTORY_LEN))
    with dev.lock:
        history = list(dev.history)[-limit:]
        return {"device_id": device_id, "gaps": dev.gaps, "last_seen": dev.last_seen, "predictions": history}


# ── Calibration ──────────────────────────────────────────────────────────────
@app.post("/calibrate/{device_id}/start")
def calibrate_start(device_id: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    if not BASAL_RMS:
        raise HTTPException(status_code=400, detail="model bundle has no basal_rms_median; cannot calibrate")
    dev = _get_device(device_id, create=False)
    with dev.lock:
        dev.calib, dev.calib_n, dev.calib_result = [], 0, None
    return {"ok": True, "seconds": CALIB_S}


@app.get("/calibrate/{device_id}")
def calibrate_status(device_id: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    dev = _get_device(device_id, create=False)
    with dev.lock:
        return {"running": dev.calib is not None, "collected_s": dev.calib_n / FS, "total_s": CALIB_S,
                "result": dev.calib_result, "current_gain": AMP_GAIN, "gain_source": GAIN_SOURCE}


@app.post("/gain")
def set_gain(value: float, x_api_key: str = Header(None)):
    """Apply a gain now and save it to calibration.json for the next start."""
    global AMP_GAIN, GAIN_SOURCE
    _check_key(x_api_key)
    if not _valid_gain(value):
        raise HTTPException(status_code=400, detail="gain must be a finite positive number")
    saved_at = datetime.now().isoformat(timespec="seconds")
    CALIB_FILE.write_text(json.dumps({"gain": value, "saved_at": saved_at,
                                      "invert": INVERT, "model": MODEL_PATH.name}, indent=2))
    with _gain_lock:
        AMP_GAIN, GAIN_SOURCE = float(value), f"calibration.json ({saved_at})"
    return {"ok": True, "amp_gain": AMP_GAIN, "saved_to": str(CALIB_FILE)}


# ── Personal baseline (baseline-relative models) ─────────────────────────────
@app.post("/baseline/{device_id}/start")
def baseline_start(device_id: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    if not BASELINE_MODE:
        raise HTTPException(status_code=400, detail="the loaded model is not baseline-relative")
    dev = _get_device(device_id, create=False)
    with dev.lock:
        # a new baseline usually means a new person: drop the old one right away
        dev.base_collect, dev.base_n, dev.baseline, dev.base_error = [], 0, None, None
    return {"ok": True, "seconds": BASELINE_S}


@app.get("/baseline/{device_id}")
def baseline_status(device_id: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    dev = _get_device(device_id, create=False)
    with dev.lock:
        return {"enabled": BASELINE_MODE, "running": dev.base_collect is not None,
                "collected_s": dev.base_n / FS, "total_s": BASELINE_S,
                "baseline": dev.baseline, "error": dev.base_error}


# ── Recording ────────────────────────────────────────────────────────────────
@app.post("/record/{device_id}/start")
def record_start(device_id: str, label: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    if label not in ("rest", "tired"):
        raise HTTPException(status_code=400, detail="label must be 'rest' or 'tired'")
    dev = _get_device(device_id, create=False)
    RECORD_DIR.mkdir(exist_ok=True)
    safe_dev = re.sub(r"[^A-Za-z0-9_-]", "_", device_id)
    path = RECORD_DIR / f"{safe_dev}_{label}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with dev.lock:
        if dev.rec_file is not None:
            _close_recording(dev)
        dev.rec_file = open(path, "w", newline="")
        dev.rec_file.write("t_us,adc_mV\n")
        dev.rec_path, dev.rec_label, dev.rec_started = path, label, time.time()
        dev.rec_n = dev.rec_gaps = 0
    return {"ok": True, "path": str(path)}


@app.post("/record/{device_id}/stop")
def record_stop(device_id: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    dev = _get_device(device_id, create=False)
    with dev.lock:
        if dev.rec_file is None:
            raise HTTPException(status_code=400, detail="not recording")
        return {"ok": True, **_close_recording(dev)}


@app.get("/record/{device_id}")
def record_status(device_id: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    dev = _get_device(device_id, create=False)
    with dev.lock:
        recording = dev.rec_file is not None
        current = ({"path": str(dev.rec_path), "label": dev.rec_label, "seconds": dev.rec_n / FS,
                    "gaps": dev.rec_gaps} if recording else None)
    files = sorted(RECORD_DIR.glob("*.csv")) if RECORD_DIR.is_dir() else []
    return {"recording": recording, "current": current,
            "files": [{"name": f.name, "kb": round(f.stat().st_size / 1024)} for f in files]}


# ── Live page ────────────────────────────────────────────────────────────────
@app.get("/view", response_class=HTMLResponse)
def view():
    """Live view. Open /view?device=esp32-01&key=<API key>."""
    return VIEW_HTML


VIEW_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EOG Live</title>
<style>
 body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222;max-width:1300px}
 canvas{width:100%;height:240px;background:#fff;border:1px solid #ddd;border-radius:6px}
 .row{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
 .tile{background:#fff;border:1px solid #ddd;border-radius:6px;padding:10px 14px;min-width:130px}
 .tile b{display:block;font-size:22px}
 .panel{background:#fff;border:1px solid #ddd;border-radius:6px;padding:12px 16px;flex:1;min-width:300px}
 .panel h3{margin:0 0 8px}
 button{font-size:15px;padding:7px 14px;border-radius:6px;border:1px solid #888;background:#f3f3f3;cursor:pointer}
 button.go{background:#2471a3;color:#fff;border-color:#2471a3}
 button.stop{background:#c0392b;color:#fff;border-color:#c0392b}
 button:disabled{opacity:.5;cursor:default}
 .fatigue{color:#c0392b}.ok{color:#1e8449}.warn{color:#b9770e}
 table{border-collapse:collapse;background:#fff;font-size:14px}
 td,th{border:1px solid #ddd;padding:4px 8px;text-align:right}
 progress{width:100%;height:16px}
 #err{color:#c0392b}
 small{color:#666}
</style></head><body>
<h2>EOG live &mdash; <span id="dev"></span></h2>
<div id="err"></div>
<div class="row">
 <div class="tile">Prediction<b id="pred">&ndash;</b><small id="oor"></small></div>
 <div class="tile">Signal RMS (mV)<b id="rms">&ndash;</b><small>training: 0.010&ndash;0.124</small></div>
 <div class="tile">Blinks / min<b id="blinks">&ndash;</b><small>training: 6&ndash;66</small></div>
 <div class="tile">ADC range (mV)<b id="range">&ndash;</b><small>OK: 150&ndash;3100</small></div>
 <div class="tile">Gain<b id="gain">&ndash;</b><small id="gainsrc"></small></div>
</div>

<div class="row" id="baseRow" style="display:none">
 <div class="panel">
  <h3>0. Record personal baseline <small id="modelName"></small></h3>
  <p><small>This model compares each person with their own rested signal. Do this once per person / session:
  rested, sitting still, eyes open, blinking naturally, <b>not looking at this screen</b>. Predictions start after it.</small></p>
  <button class="go" id="baseBtn" onclick="baseStart()">Record baseline</button>
  <div id="baseProg" style="display:none;margin-top:8px"><progress id="baseBar" value="0"></progress> <span id="baseTxt"></span></div>
  <div id="baseRes" style="margin-top:8px"></div>
 </div>
</div>

<div class="row">
 <div class="panel">
  <h3>1. Calibrate gain</h3>
  <p><small>Sit relaxed, eyes open, blink normally, don't move or talk for 60 s.</small></p>
  <button class="go" id="calBtn" onclick="calStart()">Start 60 s calibration</button>
  <div id="calProg" style="display:none;margin-top:8px"><progress id="calBar" max="60" value="0"></progress> <span id="calTxt"></span></div>
  <div id="calRes" style="margin-top:8px"></div>
 </div>
 <div class="panel">
  <h3>2. Record training data</h3>
  <p><small>Record 3&ndash;5 minutes of each. "Rested": after sleep / a break. "Tired": end of a long day or after long screen work.</small></p>
  <button class="go" id="recRest" onclick="recStart('rest')">Record RESTED</button>
  <button class="go" id="recTired" onclick="recStart('tired')">Record TIRED</button>
  <button class="stop" id="recStop" onclick="recStop()" disabled>Stop</button>
  <div id="recNow" style="margin-top:8px"></div>
  <div id="recFiles" style="margin-top:8px"></div>
 </div>
</div>

<p>Signal, last 20 s (mV at the ESP32 pin)<span id="inv"></span>:</p>
<canvas id="c" width="1200" height="240"></canvas>
<p>Last predictions:</p>
<table><thead><tr><th>Window (s)</th><th>Result</th><th>RMS</th><th>Blinks/min</th><th>Gain</th></tr></thead><tbody id="hist"></tbody></table>
<script>
const q=new URLSearchParams(location.search), dev=q.get('device')||'esp32-01', key=q.get('key')||'';
const D=encodeURIComponent(dev), $=id=>document.getElementById(id);
$('dev').textContent=dev;
const H={'X-API-Key':key};
async function api(p,method){const r=await fetch(p,{method:method||'GET',headers:H});
 const j=await r.json().catch(()=>({}));if(!r.ok)throw new Error(p+': '+(j.detail||('HTTP '+r.status)));return j;}
function draw(v){
 const c=$('c'),g=c.getContext('2d'),W=c.width,Hh=c.height;
 g.clearRect(0,0,W,Hh); if(v.length<2)return;
 let mn=Infinity,mx=-Infinity; for(const x of v){if(x<mn)mn=x;if(x>mx)mx=x;} if(mx-mn<10){mn-=5;mx+=5;}
 const step=Math.max(1,Math.floor(v.length/W));
 g.strokeStyle='#eee';for(let s=0;s<=20;s++){const x=W*s/20;g.beginPath();g.moveTo(x,0);g.lineTo(x,Hh);g.stroke();}
 g.strokeStyle='#2471a3';g.lineWidth=1;g.beginPath();
 for(let i=0;i<v.length;i+=step){const x=i/(v.length-1)*W,y=Hh-(v[i]-mn)/(mx-mn)*(Hh-10)-5;i?g.lineTo(x,y):g.moveTo(x,y);}
 g.stroke();
}
let HEALTH={};
api('/health').then(h=>{HEALTH=h;
 $('modelName').textContent='('+h.model_file+')';
 $('baseRow').style.display=h.baseline_relative?'flex':'none';
 $('baseBtn').textContent='Record '+(h.baseline_s/60).toFixed(h.baseline_s%60?1:0)+' min baseline';
 $('baseBar').max=h.baseline_s;}).catch(e=>{$('err').textContent=e.message;});
function cls(x){return x.predicted_label===1?'fatigue':(x.predicted_label===0?'ok':'warn');}
async function baseStart(){if(!confirm('Start a new baseline? The current one (if any) is replaced.'))return;
 try{await api('/baseline/'+D+'/start','POST');$('baseRes').innerHTML='';}catch(e){$('err').textContent=e.message;}}
async function calStart(){try{await api('/calibrate/'+D+'/start','POST');$('calRes').innerHTML='';}catch(e){$('err').textContent=e.message;}}
async function applyGain(v){if(!confirm('Use gain '+v.toFixed(0)+' from now on?'))return;
 try{await api('/gain?value='+v,'POST');}catch(e){$('err').textContent=e.message;}}
async function recStart(l){try{await api('/record/'+D+'/start?label='+l,'POST');}catch(e){$('err').textContent=e.message;}}
async function recStop(){try{const r=await api('/record/'+D+'/stop','POST');
 $('recNow').innerHTML='Saved <b>'+r.path.split(/[\\\\/]/).pop()+'</b> ('+r.seconds.toFixed(0)+' s)';}catch(e){$('err').textContent=e.message;}}
async function tick(){
 try{
  const raw=await api('/raw/'+D);
  draw(raw.invert ? raw.adc_mv.map(v=>3300-v) : raw.adc_mv);   // as the model sees it
  $('inv').textContent=raw.invert?' (flipped: EOG_INVERT=1)':'';
  if(raw.adc_mv.length){let mn=Infinity,mx=-Infinity;for(const x of raw.adc_mv){if(x<mn)mn=x;if(x>mx)mx=x;}
   $('range').textContent=mn+' \\u2013 '+mx; $('range').className=(mn<=150||mx>=3100)?'warn':'';}
  const lat=await api('/latest/'+D+'?limit=10'), p=lat.predictions;
  if(p.length){const L=p[p.length-1];
   $('pred').textContent=L.predicted_class;$('pred').className=cls(L);
   $('rms').textContent=L.features.RMS.toFixed(3);
   $('blinks').textContent=L.features.Blink_Rate_bpm.toFixed(0);
   const qp=L.quality_problems||[];
   $('oor').innerHTML=qp.length?'<span class="warn">&#9888; '+qp.join('; ')+'</span>':
    (L.out_of_training_range.length?'outside training range: '+L.out_of_training_range.join(', '):'');}
  $('hist').innerHTML=p.slice().reverse().map(x=>
   `<tr><td>${x.window_start_s.toFixed(0)}&ndash;${x.window_end_s.toFixed(0)}</td><td class="${cls(x)}">${x.predicted_class}</td><td>${x.features.RMS.toFixed(3)}</td><td>${x.features.Blink_Rate_bpm.toFixed(0)}</td><td>${x.gain.toFixed(0)}</td></tr>`).join('');

  if(HEALTH.baseline_relative){
   const b=await api('/baseline/'+D);
   $('baseBtn').disabled=b.running; $('baseProg').style.display=b.running?'block':'none';
   if(b.running){$('baseBar').value=b.collected_s;$('baseTxt').textContent=b.collected_s.toFixed(0)+' / '+b.total_s+' s';}
   else if(b.error){$('baseRes').innerHTML='<span class="warn">&#9888; Baseline failed: '+b.error+'</span>';}
   else if(b.baseline){const m=b.baseline.medians;
    $('baseRes').innerHTML='&#10003; Baseline recorded at '+new Date(b.baseline.finished_at*1000).toLocaleTimeString()+
     ' ('+b.baseline.clean_windows+'/'+b.baseline.total_windows+' clean windows): '+
     'blinks '+m.Blink_Rate_bpm.toFixed(0)+'/min, RMS '+m.RMS.toFixed(3)+' mV';}
   else{$('baseRes').innerHTML='<span class="warn">No baseline yet &mdash; predictions show "needs baseline".</span>';}
  }

  const cal=await api('/calibrate/'+D);
  $('gain').textContent=cal.current_gain.toFixed(0); $('gainsrc').textContent=cal.gain_source;
  $('calBtn').disabled=cal.running; $('calProg').style.display=cal.running?'block':'none';
  if(cal.running){$('calBar').value=cal.collected_s;$('calTxt').textContent=cal.collected_s.toFixed(0)+' / '+cal.total_s+' s';}
  if(cal.result&&!cal.running){const r=cal.result;
   $('calRes').innerHTML='Your resting RMS: <b>'+r.rest_rms_adc_mv.toFixed(1)+' mV</b> at the ADC<br>'+
    'Suggested gain: <b>'+(r.suggested_gain?r.suggested_gain.toFixed(0):'?')+'</b> '+
    (r.suggested_gain?'<button onclick="applyGain('+r.suggested_gain+')">Apply</button>':'')+
    r.warnings.map(w=>'<div class="warn">&#9888; '+w+'</div>').join('');}

  const rec=await api('/record/'+D);
  $('recRest').disabled=$('recTired').disabled=rec.recording; $('recStop').disabled=!rec.recording;
  if(rec.recording){const c=rec.current;
   $('recNow').innerHTML='<b class="fatigue">&#9679; REC</b> '+c.label.toUpperCase()+' &mdash; '+
    Math.floor(c.seconds/60)+':'+String(Math.floor(c.seconds%60)).padStart(2,'0')+(c.gaps?' <span class="warn">('+c.gaps+' gaps)</span>':'');}
  $('recFiles').innerHTML=rec.files.length?'<small>Saved in recordings/: '+rec.files.map(f=>f.name).join(', ')+'</small>':'';
  $('err').textContent='';
 }catch(e){$('err').textContent=e.message;}
}
tick(); setInterval(tick,1000);
</script></body></html>"""
