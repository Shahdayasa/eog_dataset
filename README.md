# EOG fatigue detection: ESP32 live system

Live eye-fatigue detection from EOG (electrooculography). An ESP32 samples a BioAmp EXG Pill at 1000 Hz and streams the raw signal over WiFi to a Python server. Every 10 s the server filters the signal, extracts 7 eye features and predicts **fatigue / no fatigue** with a model trained on a BIOPAC EOG dataset.

```
electrodes → BioAmp EXG Pill → ESP32 (GPIO34, 1000 Hz) ──WiFi/HTTP──▶ server.py ──▶ live web page
```

## Hardware
- ESP32 dev board (ESP32-D0WD-V3), Arduino core 3.x
- Upside Down Labs **BioAmp EXG Pill**, powered at 3.3 V, output on **GPIO34**
- 3 gel electrodes (2 around the eye, 1 reference)

**Safety:** when electrodes are connected to a person, the laptop must run **on battery** (charger unplugged), or the ESP32 must be powered from a power bank.

## Files
| File | Purpose |
|---|---|
| `esp32/eog_stream/eog_stream.ino` | ESP32 sketch: samples at 1000 Hz, POSTs 1 s batches to `/ingest` |
| `server.py` | FastAPI server: ingest, quality check, calibration, recording, baseline, prediction, live page `/view` |
| `eog_features.py` | Preprocessing (0.3–12 Hz Butterworth band-pass) and the 7 features, identical for training and live |
| `eog_quality.py` | Signal-quality check per 10 s window (ADC clipping, flat/huge signal, mains hum, no blinks) |
| `train_and_save.py` | Trains and saves the models (`eog_model.joblib`, `eog_model_baseline.joblib`) |
| `analyze_recordings.py` | Offline evaluation of both models on recordings made with the server's Record buttons |
| `train_export.py`, `predict_sensor.py` | Earlier offline training/prediction scripts |
| `dataset/` | BIOPAC EOG dataset used for training (`EOG_raw_signal.csv`) |
| `recordings/` | ESP32 recordings (`t_us, adc_mV`), rest and tired |
| `analysis/` | Per-window results from `analyze_recordings.py` |
| `calibration.json` | Amplifier gain estimate (6953) and polarity |

## Features (per 10 s window)
Blink rate, mean blink amplitude, OCI, RMS, standard deviation, dominant frequency, energy 0.1–0.5 Hz.

## Two models
- **`eog_model.joblib` (default)**: SVM on absolute features. It needs the exact amplifier gain, and in tests it called rested people "fatigue".
- **`eog_model_baseline.joblib` (recommended)**: SVM on features divided by the person's **own 2 min rested baseline**. The dataset is filtered with a simulated 1 Hz high-pass to match the EXG Pill. Because it uses ratios, it does not depend on the gain. Cross-validated balanced accuracy on the dataset: 0.91.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements-server.txt
```

## Run the server (Windows PowerShell)
```powershell
$env:EOG_API_KEY = "choose-a-secret"
$env:EOG_INVERT  = "1"                           # EXG Pill blinks are inverted vs the dataset
$env:EOG_MODEL   = "eog_model_baseline.joblib"
.venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000
```
Live page: `http://localhost:8000/view?device=esp32-01&key=<your key>`

## ESP32
Open `esp32/eog_stream/eog_stream.ino` in the Arduino IDE and fill in `WIFI_SSID`, `WIFI_PASS`, `SERVER_URL` (the laptop's IP, port 8000) and `API_KEY` (same as `EOG_API_KEY`). The WiFi must be 2.4 GHz, and the laptop firewall must allow port 8000.

## Measurement protocol
1. Put on fresh electrodes and wait until the page shows no quality warnings.
2. Ask the person's KSS sleepiness score (1–9).
3. Rested: click **Record RESTED** and **Record 2 min baseline**. The person sits still, looks at the wall and doesn't talk, for 4–5 min.
4. Tiring task: 60–90 min of screen work.
5. Ask the KSS score again, then click **Record TIRED** for 4–5 min.
6. Evaluate offline:
   ```powershell
   .venv\Scripts\python.exe analyze_recordings.py --person C recordings\<rest>.csv recordings\<tired>.csv
   ```

## Results so far
| Test | Baseline model | Old model |
|---|---|---|
| Rest windows called "no fatigue" (3 people) | 42–100% (depends on which 2 min are the baseline; most results 75–100%) | 0–8% |
| Tired windows called "fatigue" (person C, after 30 min of screen work) | 0/24 | 24/24 (but it says "fatigue" for everything) |

For person C, the features barely changed between rest and tired (ratios ~1.0). In the dataset, fatigue shows about 2× blink amplitude and RMS. Fatigue detection is therefore **not yet verified**; a stronger fatigue session with KSS scores is needed.

## Credits
Preprocessing and features follow the supervisor's code and dataset: [HusseinYounis/ocular-fatigue-detection](https://github.com/HusseinYounis/ocular-fatigue-detection).
