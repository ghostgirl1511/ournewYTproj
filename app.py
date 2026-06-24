"""
THE DEEZER HIT PREDICTOR — web application.

Predicts a relative audio-based popularity score (0..1) for an uploaded MP3,
using the pre-trained Strict Audio Random-Forest model (93 librosa features).

The Deezer API is NOT used at prediction time. No model is trained here.
"""

import gc
import json
import logging
import os
import tempfile

import numpy as np
from flask import Flask, render_template, request

from audio_features import extract_features_from_file, SAMPLE_RATE, PREVIEW_SECONDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deezer_hit_predictor")

# ── configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "feature_columns.json")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "random_forest_deezer_10000_strict_audio_model.pkl"),
)
# Optional: if the model is hosted externally (it is ~145 MB), provide a direct
# download URL and it will be fetched once on first start into MODEL_PATH.
MODEL_URL = os.environ.get("MODEL_URL", "").strip()

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".mp3"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

DISCLAIMER = (
    "This is a relative prediction based on audio characteristics learned from "
    "the Deezer dataset. It is not a guarantee that the song will become "
    "commercially successful."
)

# ── load feature order (recovered from the training pipeline) ───────────────
with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
    _meta = json.load(f)
FEATURE_ORDER = _meta["feature_order"]
N_FEATURES = _meta["n_features"]
assert len(FEATURE_ORDER) == N_FEATURES == 93, "feature_columns.json is inconsistent"


def _maybe_download_model() -> None:
    """Fetch the model from MODEL_URL once if it is not present on disk."""
    if os.path.exists(MODEL_PATH) or not MODEL_URL:
        return
    import urllib.request
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    logger.info("Downloading model from MODEL_URL ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    logger.info("Model downloaded to %s", MODEL_PATH)


def _load_model():
    """Load the trained Strict Audio model once. Returns None if unavailable."""
    try:
        _maybe_download_model()
        if not os.path.exists(MODEL_PATH):
            logger.error("Model file not found at %s", MODEL_PATH)
            return None
        import joblib
        model = joblib.load(MODEL_PATH)
        if getattr(model, "n_features_in_", None) != 93:
            logger.error("Loaded model expects %s features, not 93", getattr(model, "n_features_in_", "?"))
            return None
        logger.info("Model loaded (%s features).", model.n_features_in_)
        return model
    except Exception:
        logger.exception("Failed to load model")
        return None


MODEL = _load_model()


class PredictionError(Exception):
    """User-facing error with a friendly message (no stack trace exposed)."""


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def _build_feature_vector(feats: dict) -> np.ndarray:
    """Order the extracted feature dict into the exact training order and verify."""
    missing = [c for c in FEATURE_ORDER if c not in feats]
    if missing:
        raise PredictionError("Feature extraction was incomplete for this file. "
                              "Please try a different MP3.")
    vec = np.array([feats[c] for c in FEATURE_ORDER], dtype=float)
    if vec.shape[0] != 93:
        raise PredictionError("Internal feature-vector size mismatch. Prediction aborted.")
    if not np.all(np.isfinite(vec)):
        raise PredictionError("The audio produced invalid feature values "
                              "(it may be silent or corrupted).")
    return vec.reshape(1, -1)


def _predict_from_upload(file_storage) -> float:
    """Validate, extract, and predict. Always removes the temp file."""
    if MODEL is None:
        raise PredictionError("The prediction model is currently unavailable. "
                              "Please try again later.")

    filename = file_storage.filename or ""
    if not filename or not _allowed(filename):
        raise PredictionError("Please upload a valid .mp3 file.")

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        file_storage.save(tmp_path)

        if os.path.getsize(tmp_path) == 0:
            raise PredictionError("The uploaded file is empty.")

        try:
            feats = extract_features_from_file(tmp_path, max_seconds=PREVIEW_SECONDS)
        except Exception:
            logger.exception("Feature extraction crashed")
            raise PredictionError("Could not analyze this audio file. "
                                  "It may be corrupted or in an unsupported format.")
        finally:
            # Delete upload immediately after extraction — don't hold it on disk
            # while the (heavier) model prediction runs.
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    logger.warning("Could not delete temp file %s", tmp_path)
                finally:
                    tmp_path = None

        if feats is None:
            raise PredictionError("Could not decode this audio file. Make sure it is "
                                  "a valid MP3 of at least a few seconds.")

        X = _build_feature_vector(feats)
        del feats  # 93-key dict no longer needed once vector is built
        score = float(MODEL.predict(X)[0])
        del X
        gc.collect()
        return max(0.0, min(1.0, score))  # clip to [0, 1]
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("Could not delete temp file %s", tmp_path)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", disclaimer=DISCLAIMER,
                           max_mb=MAX_UPLOAD_MB, model_ready=MODEL is not None,
                           score=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    error = None
    score = None
    try:
        if "audio" not in request.files or request.files["audio"].filename == "":
            raise PredictionError("No file selected. Please choose an MP3 file.")
        score = _predict_from_upload(request.files["audio"])
    except PredictionError as e:
        error = str(e)
    except Exception:
        logger.exception("Unexpected error during prediction")
        error = "An unexpected error occurred while processing your file."
    return render_template("index.html", disclaimer=DISCLAIMER, max_mb=MAX_UPLOAD_MB,
                           model_ready=MODEL is not None, score=score, error=error)


@app.errorhandler(413)
def too_large(_e):
    return render_template("index.html", disclaimer=DISCLAIMER, max_mb=MAX_UPLOAD_MB,
                           model_ready=MODEL is not None,
                           error=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB."), 413


@app.route("/health")
def health():
    return {"status": "ok", "model_ready": MODEL is not None, "n_features": N_FEATURES}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
