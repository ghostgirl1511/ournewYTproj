# THE DEEZER HIT PREDICTOR

**Predicting Song Popularity from Audio Features**

A small Flask web app: a user uploads one MP3, the app extracts **93 librosa audio
features** from the first 30 seconds and returns a relative
`deezer_popularity_score` between 0 and 1 using a pre-trained **Strict Audio**
Random-Forest model.

> This is a relative prediction based on audio characteristics learned from the
> Deezer dataset. It is **not** a guarantee that a song will become commercially
> successful.

The app does **not** call the Deezer API and does **not** train anything at
runtime. It uses only audio extracted from the uploaded file.

---

## How it works

1. Upload an `.mp3` (max 20 MB by default).
2. The file is decoded to **mono at 22 050 Hz**, and the **first 30 seconds** are used
   (training used Deezer's 30-second previews — same `_compute_features` code path).
3. **93 librosa features** are extracted (MFCC, chroma, spectral, tonnetz, ZCR, RMS,
   tempo, mel, harmonic/percussive). `gain` and `duration_seconds` are **not** used —
   this is the Strict Audio model.
4. The features are ordered into the exact training order from `feature_columns.json`,
   verified to be length 93, and passed to the model.
5. The predicted score is clipped to `[0, 1]` and shown.

The temporary upload is always deleted after processing.

### Why `feature_columns.json` exists
The model was trained on a NumPy array, so the pickle does **not** store feature
names/order. `feature_columns.json` records the exact 93-feature order recovered from
the training pipeline (the alphabetically-sorted librosa columns produced by
`build_dataset.py`, minus `gain`/`duration_seconds`). The app reorders the extracted
features to match it before predicting. **Do not reorder or edit this file.**

---

## Project files

| File | Purpose |
|---|---|
| `app.py` | Flask app: upload, validate, extract, predict, friendly errors |
| `audio_features.py` | Source-of-truth librosa extraction (shared with training) |
| `feature_columns.json` | The exact 93-feature order + sample rate / segment length |
| `templates/index.html` | Upload UI (Deezer-inspired colors) |
| `static/style.css` | Styling |
| `models/…strict_audio_model.pkl` | The trained model (see "Model file" below) |
| `verify_app.py` | Local check: model loads + 93-feature vector + prediction |
| `requirements.txt`, `Procfile`, `render.yaml`, `Dockerfile`, `.gitignore` | Deployment |

---

## Run locally

```bash
cd webapp
python -m venv .venv && . .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Put the model in place (see "Model file"):
#   webapp/models/random_forest_deezer_10000_strict_audio_model.pkl

# Verify everything is wired correctly (no server needed):
python verify_app.py
# optional: test the full decode path with a real file
python verify_app.py path/to/song.mp3

# Run the app:
python app.py            # http://localhost:5000
# or, production-style:
gunicorn app:app --bind 0.0.0.0:5000
```

> **MP3 decoding:** locally you need an MP3 decoder for librosa. `ffmpeg` is the most
> reliable (`winget install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg`).
> The provided `Dockerfile` installs it for production.

---

## Model file (important — it is ~145 MB)

The Strict Audio model `random_forest_deezer_10000_strict_audio_model.pkl` is about
**145 MB**, which exceeds GitHub's 100 MB per-file limit. Pick one option:

- **Option A — host externally (recommended).** Upload the `.pkl` somewhere with a
  direct download link (e.g. a GitHub *Release* asset, S3, Hugging Face, Google Drive
  direct link). Set the `MODEL_URL` env var on Render; the app downloads it once on
  first start into `models/`. (`models/*.pkl` is git-ignored by default.)
- **Option B — commit via Git LFS.** `git lfs install`, `git lfs track "*.pkl"`, then
  remove the `models/*.pkl` line from `.gitignore` and commit the file with LFS.

Either way the app loads it from `MODEL_PATH`
(default `models/random_forest_deezer_10000_strict_audio_model.pkl`).

---

## Deploy on Render (Docker — recommended)

Docker guarantees `ffmpeg`/`libsndfile` are present for reliable MP3 decoding.

1. Commit this `webapp/` folder to your GitHub repo.
2. In Render, the existing web service → **Settings**:
   - **Runtime:** Docker (Render auto-detects the `Dockerfile`), or use the included
     `render.yaml` (Blueprint).
   - **Health check path:** `/health`
   - **Instance type:** at least **Starter / 512 MB+** — the model needs RAM to load;
     the free tier may OOM.
3. **Environment variables:**
   - `MODEL_URL` = direct download URL of the `.pkl` (Option A), **or** skip it if you
     committed the model via Git LFS (Option B).
   - `MAX_UPLOAD_MB` = `20` (optional).
4. Deploy. Verify `https://<your-service>/health` returns
   `{"status":"ok","model_ready":true,"n_features":93}`.

### Non-Docker (native) alternative
If you keep the existing service on a native Python runtime instead of Docker:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
- You must ensure an MP3 decoder is available. Native Render images may lack `ffmpeg`;
  if MP3 decoding fails, switch to the Docker option above. `soundfile` (libsndfile) is
  installed via pip and handles WAV/FLAC, but MP3 support depends on the system codec —
  Docker + ffmpeg is the dependable path.

---

## Error handling
Friendly messages (no stack traces) are shown for: non-MP3 files, empty/corrupted
files, undecodable/too-short audio, failed feature extraction, oversized uploads, and a
missing/unavailable model.
