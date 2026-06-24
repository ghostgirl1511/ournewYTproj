"""
Audio Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~
Download a Deezer 30-second preview MP3 and extract librosa audio features,
returning a flat dictionary of aggregated statistics (mean + std) suitable
for a single DataFrame row.

Requires:  librosa, numpy, requests
           + ffmpeg installed on the system (for MP3 decoding)
"""

import ctypes
import gc
import logging
import os
import platform
import subprocess
import tempfile
import warnings

import librosa
import numpy as np
import requests

logger = logging.getLogger(__name__)

TIMEOUT = 20  # seconds for downloading preview
SAMPLE_RATE = 22050  # standard librosa default
PREVIEW_SECONDS = 30  # training used Deezer 30s previews; uploads use first 30s


def _malloc_trim() -> None:
    """
    Return freed pages to the OS on Linux (Render's container environment).

    Python's pymalloc and ptmalloc hold freed numpy memory in internal arenas
    and never give it back to the OS, so the process RSS stays high even after
    del + gc.collect(). malloc_trim(0) forces an immediate OS-level trim.
    Safe no-op on macOS/Windows.
    """
    if platform.system() == "Linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def _load_audio_30s(file_path: str) -> np.ndarray | None:
    """
    Decode exactly the first PREVIEW_SECONDS of audio via an ffmpeg subprocess.

    This is the primary loader used by the web app. Key properties:

    * ``-t PREVIEW_SECONDS`` tells ffmpeg's muxer to stop at 30 s at the
      *input demux* level — no audio beyond that is ever decoded or buffered
      in Python, regardless of how long the source file is.  A 5-minute MP3
      at 128 kbps decoded with librosa.load() can temporarily occupy > 100 MB
      of PCM; this path uses < 5 MB.
    * ``-ac 1 -ar 22050 -f f32le`` gives us mono, 22050 Hz, float32 LE PCM
      directly — no resampling or mixing happens in Python.
    * stdout is consumed once into a bytes buffer; ``np.frombuffer(...).copy()``
      converts it to a numpy array and immediately drops the bytes reference.

    Returns None if ffmpeg is absent or the file cannot be decoded, so the
    caller can fall back to librosa.load().
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v", "error",
                "-i", file_path,
                "-t", str(PREVIEW_SECONDS),   # hard stop at 30 s (muxer level)
                "-ac", "1",                   # mono
                "-ar", str(SAMPLE_RATE),      # 22050 Hz
                "-f", "f32le",                # raw float32 little-endian PCM
                "pipe:1",                     # write to stdout
            ],
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError:
        logger.debug("ffmpeg not found; falling back to librosa.load()")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out decoding %s", file_path)
        return None
    except Exception as e:
        logger.warning("ffmpeg subprocess error for %s: %s", file_path, e)
        return None

    if result.returncode != 0 or not result.stdout:
        logger.warning(
            "ffmpeg decode failed (rc=%d) for %s: %s",
            result.returncode, file_path,
            result.stderr[:300].decode("utf-8", errors="replace"),
        )
        return None

    # np.frombuffer gives a read-only view into result.stdout (bytes).
    # .copy() makes it writable and lets the bytes object be GC'd immediately.
    y = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return y if len(y) > 0 else None


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

    Audio loading strategy (two-stage):

    1. **ffmpeg subprocess** (primary) — ``_load_audio_30s`` decodes exactly
       ``PREVIEW_SECONDS`` seconds at the muxer level.  Peak memory is < 5 MB
       regardless of source file length.  This is the path used on Render where
       ffmpeg is installed via the Dockerfile.

    2. **librosa.load fallback** — used when ffmpeg is absent.  ``duration``
       is passed to limit the read, but some libsndfile backends (e.g. when
       soundfile handles MP3) may still buffer the full decoded file before
       slicing, so this path is less memory-safe for long uploads.

    After feature extraction the audio array is deleted and malloc_trim is
    called so that freed C-heap pages are returned to the OS immediately.
    """
    # ── Stage 1: ffmpeg (hard 30-s muxer limit, < 5 MB peak) ────────────────
    y = _load_audio_30s(file_path)

    # ── Stage 2: librosa fallback (duration-limited, but less reliable for MP3)
    if y is None:
        try:
            y, _ = librosa.load(
                file_path, sr=SAMPLE_RATE, mono=True, duration=float(max_seconds)
            )
        except Exception as e:
            logger.warning("Failed to load audio file %s: %s", file_path, e)
            return None

    if y is None or len(y) == 0:
        return None

    try:
        return _compute_features(y, SAMPLE_RATE)
    finally:
        del y
        gc.collect()
        _malloc_trim()


def _compute_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """
    Compute the librosa feature dictionary from a mono audio array.

    Single source of truth shared by both the Deezer-preview path
    (``extract_features``) and the web-app upload path
    (``extract_features_from_file``). The feature names produced here are the
    exact set used to train the models.

    Memory note: intermediate arrays are deleted immediately after use and
    gc.collect() is called at natural boundaries to keep peak RSS low.
    HPSS is run once and its outputs reused for both the harmonic/percussive
    stats and tonnetz (effects.harmonic is just hpss()[0], so this is
    numerically identical to the original and produces the same 93 features).
    """
    features: dict[str, float] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # ── MFCCs (13 coefficients) ──────────────────────────────────
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std  = np.std(mfccs,  axis=1)
        del mfccs
        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = float(mfcc_mean[i])
            features[f"mfcc_{i+1}_std"]  = float(mfcc_std[i])
        del mfcc_mean, mfcc_std

        # ── Chroma (12 pitch classes) ────────────────────────────────
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std(chroma,  axis=1)
        del chroma
        for i in range(12):
            features[f"chroma_{i+1}_mean"] = float(chroma_mean[i])
            features[f"chroma_{i+1}_std"]  = float(chroma_std[i])
        del chroma_mean, chroma_std

        gc.collect()

        # ── Spectral Centroid ────────────────────────────────────────
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(spec_cent))
        features["spectral_centroid_std"]  = float(np.std(spec_cent))
        del spec_cent

        # ── Spectral Bandwidth ───────────────────────────────────────
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
        features["spectral_bandwidth_std"]  = float(np.std(spec_bw))
        del spec_bw

        # ── Spectral Rolloff ─────────────────────────────────────────
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))
        features["spectral_rolloff_std"]  = float(np.std(spec_rolloff))
        del spec_rolloff

        # ── Spectral Contrast (7 bands) ──────────────────────────────
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        sc_mean = np.mean(spec_contrast, axis=1)
        sc_std  = np.std(spec_contrast,  axis=1)
        del spec_contrast
        for i in range(len(sc_mean)):
            features[f"spectral_contrast_{i+1}_mean"] = float(sc_mean[i])
            features[f"spectral_contrast_{i+1}_std"]  = float(sc_std[i])
        del sc_mean, sc_std

        # ── Zero Crossing Rate ───────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"]  = float(np.std(zcr))
        del zcr

        # ── RMS Energy ───────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"]  = float(np.std(rms))
        del rms

        # ── Mel Spectrogram (summary) ────────────────────────────────
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        del mel
        features["mel_mean"] = float(np.mean(mel_db))
        features["mel_std"]  = float(np.std(mel_db))
        del mel_db

        gc.collect()

        # ── Tempo ────────────────────────────────────────────────────
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(np.atleast_1d(tempo)[0])

        # ── HPSS once — harmonic used for both tonnetz and harmonic stats ──
        # effects.harmonic(y) is identical to hpss(y)[0]; running hpss once
        # avoids a full duplicate STFT + median-filter + iSTFT cycle (~90 MB peak).
        y_harm, y_perc = librosa.effects.hpss(y)

        # ── Harmonic / Percussive stats ──────────────────────────────
        features["harmonic_mean"]   = float(np.mean(np.abs(y_harm)))
        features["harmonic_std"]    = float(np.std(np.abs(y_harm)))
        features["percussive_mean"] = float(np.mean(np.abs(y_perc)))
        features["percussive_std"]  = float(np.std(np.abs(y_perc)))
        del y_perc

        # ── Tonnetz (6 tonal centroids) ──────────────────────────────
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        del y_harm
        tn_mean = np.mean(tonnetz, axis=1)
        tn_std  = np.std(tonnetz,  axis=1)
        del tonnetz
        for i in range(6):
            features[f"tonnetz_{i+1}_mean"] = float(tn_mean[i])
            features[f"tonnetz_{i+1}_std"]  = float(tn_std[i])
        del tn_mean, tn_std

    gc.collect()
    _malloc_trim()
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
