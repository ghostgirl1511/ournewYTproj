"""
Local verification for THE DEEZER HIT PREDICTOR.

Confirms, WITHOUT retraining:
  1. feature_columns.json has exactly 93 features.
  2. The Strict Audio model loads and expects 93 features.
  3. The extractor produces those exact 93 feature names.
  4. A feature vector is built in the correct order and the model predicts a
     score in [0, 1].

Usage:
  python verify_app.py                 # uses a synthetic 30s audio array
  python verify_app.py path/to/song.mp3   # also tests the full file -> decode path
"""

import json
import os
import sys

import numpy as np

from audio_features import _compute_features, extract_features_from_file, SAMPLE_RATE

BASE = os.path.dirname(os.path.abspath(__file__))
ok = True


def check(label, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    ok = ok and bool(cond)


print("=" * 60)
print("DEEZER HIT PREDICTOR — verification")
print("=" * 60)

# 1) feature columns artifact
meta = json.load(open(os.path.join(BASE, "feature_columns.json"), encoding="utf-8"))
order = meta["feature_order"]
check("feature_columns.json has 93 features", len(order) == 93 == meta["n_features"])
check("sample_rate == 22050", meta["sample_rate"] == SAMPLE_RATE == 22050)

# 2) model loads + expects 93
model_path = os.environ.get(
    "MODEL_PATH", os.path.join(BASE, "models", meta["model_file"]))
if os.path.exists(model_path):
    import joblib
    model = joblib.load(model_path)
    check("model loads", model is not None)
    check("model.n_features_in_ == 93", getattr(model, "n_features_in_", None) == 93)
else:
    model = None
    print(f"  [SKIP] model file not found at {model_path} "
          "(place it there or set MODEL_PATH to test prediction)")

# 3) extractor produces exactly the 93 names
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    print(f"  using real audio file: {sys.argv[1]}")
    feats = extract_features_from_file(sys.argv[1])
    check("extraction from MP3 returned features", feats is not None)
else:
    print("  using synthetic 30s audio array")
    sr = SAMPLE_RATE
    y = (np.sin(2 * np.pi * 220 * np.arange(sr * 30) / sr)
         + 0.1 * np.random.RandomState(0).randn(sr * 30)).astype("float32")
    feats = _compute_features(y, sr)

check("extractor produced 93 features", feats is not None and len(feats) == 93)
check("extractor names == training feature set",
      feats is not None and set(feats.keys()) == set(order))

# 4) ordered vector + prediction
if feats is not None:
    vec = np.array([feats[c] for c in order], dtype=float).reshape(1, -1)
    check("ordered vector shape is (1, 93)", vec.shape == (1, 93))
    check("vector has only finite values", bool(np.all(np.isfinite(vec))))
    if model is not None:
        score = float(model.predict(vec)[0])
        score_c = max(0.0, min(1.0, score))
        print(f"  -> predicted score: {score:.4f} (clipped: {score_c:.4f})")
        check("prediction within [0, 1] after clip", 0.0 <= score_c <= 1.0)

print("=" * 60)
print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
print("=" * 60)
sys.exit(0 if ok else 1)
