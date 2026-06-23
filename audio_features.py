"""
Audio Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~
Download a Deezer 30-second preview MP3 and extract librosa audio features,
returning a flat dictionary of aggregated statistics (mean + std) suitable
for a single DataFrame row.

Requires:  librosa, numpy, requests
           + ffmpeg installed on the system (for MP3 decoding)
"""

import logging
import os
import tempfile
import warnings
from typing import Any

import librosa
import numpy as np
import requests

logger = logging.getLogger(__name__)

TIMEOUT = 20  # seconds for downloading preview
SAMPLE_RATE = 22050  # standard librosa default
PREVIEW_SECONDS = 30  # training used Deezer 30s previews; uploads use first 30s


def _download_preview(preview_url: str) -> np.ndarray | None:
    """
    Download an MP3 preview from Deezer and return it as a numpy audio array.

    Uses librosa.load() which handles MP3 decoding via audioread/ffmpeg.
    Returns None if the download fails or the URL is empty.
    """
    if not preview_url:
        return None

    tmp_path = None
    try:
        resp = requests.get(preview_url, timeout=TIMEOUT)
        resp.raise_for_status()

        # Save to a temp file — librosa.load() needs a file path for MP3
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        # Load with librosa (handles MP3 via audioread/ffmpeg)
        audio_data, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        return audio_data

    except Exception as e:
        logger.warning("Failed to download/decode preview %s: %s", preview_url, e)
        return None

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass



def _safe_mean_std(feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std along the time axis (axis=1) for a 2-D feature matrix."""
    return np.mean(feature_matrix, axis=1), np.std(feature_matrix, axis=1)


def extract_features(preview_url: str) -> dict[str, float] | None:
    """
    Download a 30-second preview and extract a comprehensive set of audio
    features using librosa.

    Parameters
    ----------
    preview_url : str
        URL to the Deezer 30-second MP3 preview.

    Returns
    -------
    dict or None
        Flat dictionary of feature names → float values, or None if extraction fails.
    """
    y = _download_preview(preview_url)
    if y is None or len(y) == 0:
        return None
    try:
        return _compute_features(y, SAMPLE_RATE)
    except Exception as e:
        logger.error("Feature extraction failed for %s: %s", preview_url, e)
        return None


def extract_features_from_file(
    file_path: str, max_seconds: int = PREVIEW_SECONDS
) -> dict[str, float] | None:
    """
    Extract the SAME librosa feature dictionary from a LOCAL audio file.

    Used by the web app for user uploads. Mirrors the training-time
    preprocessing exactly: load as mono at SAMPLE_RATE (22050 Hz) and take the
    first ``max_seconds`` seconds (training used Deezer's 30-second previews,
    so we use a 30-second segment of the upload). Feature *computation* is
    identical to the preview path because both call ``_compute_features``.

    Returns the feature dict, or None if the file cannot be decoded.
    Raises on feature-computation errors (caller should handle).
    """
    try:
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=max_seconds)
    except Exception as e:
        logger.warning("Failed to load audio file %s: %s", file_path, e)
        return None
    if y is None or len(y) == 0:
        return None
    return _compute_features(y, SAMPLE_RATE)


def _compute_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """
    Compute the librosa feature dictionary from a mono audio array.

    Single source of truth shared by both the Deezer-preview path
    (``extract_features``) and the web-app upload path
    (``extract_features_from_file``). The feature names produced here are the
    exact set used to train the models.
    """
    features: dict[str, float] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ── MFCCs (13 coefficients) ──────────────────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean, mfcc_std = _safe_mean_std(mfccs)
        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = float(mfcc_mean[i])
            features[f"mfcc_{i+1}_std"] = float(mfcc_std[i])

        # ── Chroma (12 pitch classes) ────────────────────────────────
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean, chroma_std = _safe_mean_std(chroma)
        for i in range(12):
            features[f"chroma_{i+1}_mean"] = float(chroma_mean[i])
            features[f"chroma_{i+1}_std"] = float(chroma_std[i])

        # ── Spectral Centroid ────────────────────────────────────────
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(spec_cent))
        features["spectral_centroid_std"] = float(np.std(spec_cent))

        # ── Spectral Bandwidth ───────────────────────────────────────
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
        features["spectral_bandwidth_std"] = float(np.std(spec_bw))

        # ── Spectral Rolloff ─────────────────────────────────────────
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spec_rolloff))

        # ── Spectral Contrast (7 bands) ──────────────────────────────
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        sc_mean, sc_std = _safe_mean_std(spec_contrast)
        for i in range(len(sc_mean)):
            features[f"spectral_contrast_{i+1}_mean"] = float(sc_mean[i])
            features[f"spectral_contrast_{i+1}_std"] = float(sc_std[i])

        # ── Zero Crossing Rate ───────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        # ── RMS Energy ───────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        # ── Tempo ────────────────────────────────────────────────────
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(np.atleast_1d(tempo)[0])

        # ── Tonnetz (6 tonal centroids) ──────────────────────────────
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tn_mean, tn_std = _safe_mean_std(tonnetz)
        for i in range(6):
            features[f"tonnetz_{i+1}_mean"] = float(tn_mean[i])
            features[f"tonnetz_{i+1}_std"] = float(tn_std[i])

        # ── Mel Spectrogram (summary) ────────────────────────────────
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features["mel_mean"] = float(np.mean(mel_db))
        features["mel_std"] = float(np.std(mel_db))

        # ── Harmonic / Percussive Ratio ──────────────────────────────
        y_harm, y_perc = librosa.effects.hpss(y)
        features["harmonic_mean"] = float(np.mean(np.abs(y_harm)))
        features["harmonic_std"] = float(np.std(np.abs(y_harm)))
        features["percussive_mean"] = float(np.mean(np.abs(y_perc)))
        features["percussive_std"] = float(np.std(np.abs(y_perc)))

    return features


# ── quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Test with a well-known Deezer track preview
    test_url = "https://cdns-preview-d.dzcdn.net/stream/c-deda7fa9316d9e9e880d2c6207e92260-8.mp3"
    print(f"Extracting features from: {test_url}")
    result = extract_features(test_url)
    if result:
        print(f"Extracted {len(result)} features:")
        print(json.dumps(result, indent=2))
    else:
        print("Feature extraction failed.")
