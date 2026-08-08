# features.py
#
# ── TRAIN/SERVE PARITY — READ BEFORE EDITING ────────────────────────────
# This file's clean_text() / extract_features() MUST be byte-for-byte
# identical to the versions in retrain_with_malaysian.py (and mirrored in
# ablate_was_translated_feature.py, test_translation_fingerprint.py,
# ablate_boilerplate_feature.py, test_malaysian_inference.py) — that is
# the ONLY script that actually produces best_model.pt's weights, and the
# fc1 layer's extra-feature slots were fit against exactly this 6-value
# definition, in exactly this order, normalized against exactly this
# corpus. app.py imports clean_text/get_features_tensor from THIS file for
# every production request, so any drift here silently corrupts every
# prediction's engineered-feature input — the model still runs and returns
# a confident-looking score, it's just conditioning on garbage in 6 of its
# input dimensions.
#
# BUG HISTORY (do not reintroduce):
#   v1: this file mirrored the Dart TextCleaner.extractFeatures() feature
#       set (exclamation/question count, 'h'-letter count, ...) — wrong
#       feature space for a Python-trained model to begin with.
#   v2 ("fixed" 2026-07-21): recalibrated FEATURE_MEAN/FEATURE_STD via
#       recompute_normalization.py, but recompute_normalization.py computed
#       stats for a THIRD, newly-invented feature definition (char length,
#       avg word length, special-char ratio, ...) that matches neither the
#       Dart code nor retrain_with_malaysian.py. Diagnosed via
#       diagnose_features.py / length_stress_test.py, but those diagnostic
#       scripts also imported from this file, so they were themselves
#       measuring the wrong feature space — v2 "fixed" the internal
#       consistency of this file without checking it against the actual
#       training script, so the train/serve mismatch persisted.
#   v3 (this version): extract_features()/clean_text() now match
#       retrain_with_malaysian.py exactly. Normalization now prefers the
#       EXACT mean/std saved by that training run (feature_stats_malaysian.npz)
#       instead of a separately-recomputed approximation, since that's the
#       one artifact guaranteed to match what best_model.pt actually saw.
#
# DEPLOYMENT NOTE: feature_stats_malaysian.npz must ship alongside
# best_model.pt (same directory as this file / MODEL_PATH in app.py). It is
# written automatically by retrain_with_malaysian.py every time it's run.
# If you retrain, redeploy both files together.

import os
import re

import numpy as np


# ── Boilerplate stripping — must match retrain_with_malaysian.py exactly ──
# audit_malaysian_data.py found these are Sebenarnya-scraper/translation
# template artifacts correlated with the FAKE label (57% start with
# "According to,", 17.2% contain "TRUTH:", 8.7% contain "SOURCE:"/"SUMBER:",
# vs ~0-0.8% in REAL) — not genuine truthfulness signal. Stripped at both
# training and serving time so the model can't use them as a shortcut.
BOILERPLATE_PATTERNS = [
    r"^according to,?\s*",
    r"\btruth:\s*",
    r"\b(?:source|sumber)\s*:\s*",
]


def _strip_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


# ── Text cleaning — must match retrain_with_malaysian.py exactly ──────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _strip_boilerplate(text)
    return text


# ── Feature extraction — must match retrain_with_malaysian.py exactly ─────
# Feature index mapping (matches training):
#   0 — word count
#   1 — exclamation mark count
#   2 — question mark count
#   3 — uppercase ratio (fraction of chars that are uppercase)
#   4 — literal "http" substring count (crude URL-presence signal)
#   5 — digit ratio (fraction of chars that are digits)
#
# NOTE: this is computed on text that clean_text() has already lowercased,
# so features 3 will typically read ~0 and feature 4 will typically read 0
# (any "http" was already stripped) for real inputs — that's expected and
# matches training-time behaviour exactly; do not "fix" this by reordering
# clean+extract, since that would once again diverge from training.
def extract_features(text: str) -> list[float]:
    words = text.split()
    return [
        float(len(words)),
        float(text.count("!")),
        float(text.count("?")),
        (sum(1 for c in text if c.isupper()) / max(len(text), 1)),
        float(text.count("http")),
        (sum(c.isdigit() for c in text) / max(len(text), 1)),
    ]


# ── Normalization stats ────────────────────────────────────────────────────
# Preferred source: feature_stats_malaysian.npz, written by
# retrain_with_malaysian.py during the run that actually produced the
# currently-deployed best_model.pt. This is loaded lazily (not at import
# time) so importing this module never crashes app.py's startup; the error
# only surfaces if inference is attempted before the stats file exists.
_STATS_FILENAME = "feature_stats_malaysian.npz"
_stats_cache = None


def _load_stats():
    global _stats_cache
    if _stats_cache is not None:
        return _stats_cache

    search_paths = [
        _STATS_FILENAME,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _STATS_FILENAME),
    ]
    for path in search_paths:
        if os.path.exists(path):
            data = np.load(path)
            mean = data["mean"].astype(np.float64).tolist()
            std = data["std"].astype(np.float64).tolist()
            if len(mean) != 6 or len(std) != 6:
                raise RuntimeError(
                    f"{path} has {len(mean)} features, expected 6 — this "
                    "file doesn't match the current 6-feature definition "
                    "in features.py. Re-run retrain_with_malaysian.py to "
                    "regenerate it."
                )
            _stats_cache = (mean, std)
            return _stats_cache

    raise RuntimeError(
        f"{_STATS_FILENAME} not found (looked in: {search_paths}). This "
        "file is written by retrain_with_malaysian.py during training and "
        "is REQUIRED for inference to match what best_model.pt was trained "
        "on. Copy it from your training run into the same directory as "
        "app.py / best_model.pt, then restart. Do not substitute the "
        "constants from an older version of this file or from "
        "recompute_normalization.py's output unless you have verified "
        "they were computed with the extract_features() defined above."
    )


def normalize_features(features: list[float]) -> list[float]:
    """Z-score normalization using the training run's mean/std."""
    mean, std = _load_stats()
    normalized = []
    for i, val in enumerate(features):
        s = std[i] if std[i] >= 1e-8 else 1e-8
        normalized.append((val - mean[i]) / s)
    return normalized


def get_features_tensor(text: str):
    """Returns normalized features as a [1, 6] float32 torch tensor."""
    import torch
    raw = extract_features(text)
    normd = normalize_features(raw)
    return torch.tensor([normd], dtype=torch.float32)  # [1, 6]