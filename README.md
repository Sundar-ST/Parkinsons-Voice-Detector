## Parkinsons Voice Detection — Setup & Run

This project records a short voice sample, extracts features with Praat/Parselmouth, and predicts Parkinson's risk using a trained scikit-learn model.

## Requirements
- Python 3.10 (required for the Parselmouth wheel on Windows)
- (See `requirements.txt` for exact pinned versions)

## Quick setup (Windows PowerShell)
1. Open PowerShell in the repository root (e.g. `c:\Users\hp\Documents\Parkinsons-Voice-Detection`).

2. Create and activate a virtual environment

```powershell
python -m venv .\parkinsons
# Activate (PowerShell)
# Activate (PowerShell)
.\parkinsons\Scripts\Activate.ps1
# Upgrade pip and wheel
python -m pip install --upgrade pip wheel
```

3. Install pinned dependencies

```powershell
pip install -r .\requirements.txt
```

This will fetch and install all packages including Parselmouth from PyPI automatically.

## Quick setup with Anaconda (alternative)

If you're using Anaconda or Miniconda instead of standard Python:

1. Create a conda environment with Python 3.10

```powershell
conda create -n parkinsons python=3.10
conda activate parkinsons
```

2. Install dependencies from `requirements.txt`

```powershell
pip install -r .\requirements.txt
```

Or, if you prefer conda for all packages (except Parselmouth, which is fetched via pip):

```powershell
conda install flask numpy pandas scikit-learn joblib librosa scipy resampy audioread soundfile numba sounddevice -y
pip install praat_parselmouth @ https://files.pythonhosted.org/packages/83/d0/a5383230ed55c262fb6d774828c7e9465939472f45acf4571d953cb247e2/praat_parselmouth-0.4.6-cp310-cp310-win_amd64.whl
```

Then proceed to "Verify core imports" or run the app.

## Verify core imports
Run a quick import check to ensure critical packages are available:

```powershell
python -c "import numpy, pandas, sklearn, joblib, flask, sounddevice, librosa, parselmouth; print('OK: imports success')"
```

If that prints `OK: imports success` without ImportError, you should be ready to run the app.

## Verify installation (automated)

Alternatively, run the automated `verify_env.py` script to check Python version, installed packages, and print actionable messages:

```powershell
python .\verify_env.py
```

This script will:
- Print your Python version and architecture.
- Warn if Python is not 3.10 (required for the Parselmouth wheel).
- Test imports of all core packages (numpy, pandas, sklearn, joblib, flask, sounddevice, librosa, parselmouth, scipy).
- Print package versions if imports succeed.
- Suggest commands to fix any missing or broken installs.

Exit code is 0 on success, non-zero if any imports fail.

## Run the app

```powershell
python .\app.py
```

Open http://127.0.0.1:5000/ in your browser and use the web UI to record and run a prediction.

## Notes & Troubleshooting
- Parselmouth wheel compatibility: the Parselmouth package is fetched from PyPI as a prebuilt wheel for CPython 3.10 (win_amd64). You must use Python 3.10. If you need support for a different Python version, check PyPI for a matching wheel.
- Audio capture: `mvp_core.py` uses `sounddevice`. If you experience audio capture errors, check microphone privacy/settings and ensure no other app is locking the device.
- Build errors when installing packages:
  - If you see compilation errors (e.g., when installing `numba` or `soundfile`), install Visual C++ Build Tools or use prebuilt wheels.
- If you prefer `pyaudio` instead of `sounddevice`, you'll need a matching prebuilt PyAudio wheel (Windows) and modify `mvp_core.py` accordingly.

## Rationale for pinned packages (why included)
- Flask==2.2.5 — app server used by `app.py`.
- numpy==1.24.3 — numeric arrays and conversions.
- pandas==1.5.3 — data loading and preprocessing in `train_model.py`.
- scikit-learn==1.1.3 — training and model API (RandomForestClassifier, scaler).
- joblib==1.2.0 — saving/loading model and scaler.
- librosa==0.9.2 — audio resampling, normalization, trimming.
- scipy==1.10.1 — `scipy.io.wavfile` used to write temporary wav for Parselmouth analysis.
- resampy==0.3.1, audioread==2.1.9, soundfile==0.12.1, numba==0.56.4 — supporting audio processing libraries used by librosa.
- sounddevice==0.4.6 — live audio recording in `mvp_core.py`.
- Parselmouth wheel — Praat bindings used to extract pitch, jitter, shimmer, and harmonicity features.

## Optional extras
- `matplotlib` — for plotting/visualization during analysis or debugging.
- `jupyter` — if you use notebooks for exploration.
- `pyaudio` — alternate live capture library (requires different install path on Windows).

## Reproducibility
- `requirements.txt` pins exact version numbers for reproducible installs. If you need a different Python version, update or replace the Parselmouth wheel accordingly.

## Files of interest
- `app.py` — Flask server and live prediction endpoint.
- `mvp_core.py` — audio recording, preprocessing, and feature extraction (Parselmouth + Praat calls).
- `train_model.py` — script to train and save the `parkinsons_model.pkl` and `feature_scaler.pkl`.

---
If you want, I can also:
- add a small `verify_env.py` script that programmatically runs the import checks and prints helpful messages, or
- pin different package versions tailored to a newer Python if you plan to upgrade. Which would you prefer next?
