# features.py
#
# NOTE: previously claimed to mirror TextCleaner.extractFeatures() /
# _normalizeFeatures() from text_cleaner.dart, but the FEATURE_MEAN /
# FEATURE_STD constants were calibrated for the DART feature definitions
# (exclamation count, question count, 'h'-letter count, etc.) while this
# file computes a DIFFERENT set of features (char length, word count,
# avg word length, digit ratio, special char ratio). Applying one
# feature space's normalization stats to another produced runaway
# z-scores for word count (e.g. z=346.8 for a 363-word article), which
# alone was enough to swing fake_score by +0.60.
#
# FEATURE_MEAN / FEATURE_STD below are PLACEHOLDERS. Run
# recompute_normalization.py against your actual training corpus and
# paste the printed values in before using this file for training or
# inference. Do NOT ship with the placeholders below.

import re
import math


# ── Normalization constants ────────────────────────────────────────────────
# Recomputed 2026-07-21 via recompute_normalization.py against the actual
# training corpus (malaysian_real/fake, ISOT, FakeNewsNet gossipcop+
# politifact, LIAR train+valid — test.tsv excluded, held out for
# ablate_was_translated_feature.py). N=81,725 examples.
#
# Replaces the previous constants, which were calibrated for the Dart
# TextCleaner.extractFeatures() definitions (exclamation count, question
# count, etc.) and were being incorrectly applied to this file's different
# feature definitions (char length, word count, avg word length, ...).
FEATURE_MEAN = [
    1453.2574854695627,     # 0 — text length (char count)
    238.47681859895994,     # 1 — word count
    5.207607207444535,      # 2 — average word length
    1.3622813393934964e-06, # 3 — uppercase ratio (~0, post-lowercase)
    0.008407261939760002,   # 4 — digit ratio
    0.026219840799482624,   # 5 — special char ratio
]

FEATURE_STD = [
    2011.0109126631844,     # 0 — text length (char count)
    328.2695703267409,      # 1 — word count
    1.2111691332123182,     # 2 — average word length
    0.0001846318832623726,  # 3 — uppercase ratio
    0.01745010515823723,    # 4 — digit ratio
    0.016922137951888225,   # 5 — special char ratio
]

_PLACEHOLDER_GUARD = any(v is None for v in FEATURE_MEAN + FEATURE_STD)


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_features(text: str) -> list[float]:
    """
    Extracts 6 raw numeric features from text.

    Feature index mapping:
      0 — text length (character count)
      1 — word count
      2 — average word length
      3 — uppercase ratio  (after clean → always 0.0 since we lowercased)
      4 — digit ratio
      5 — special char ratio
    """
    if not text:
        return [0.0] * 6

    length       = float(len(text))
    words        = text.split()
    word_count   = float(len(words))
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0
    digit_count   = sum(1 for c in text if c.isdigit())
    upper_count   = sum(1 for c in text if c.isupper())
    special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())

    digit_ratio   = digit_count   / length if length > 0 else 0.0
    upper_ratio   = upper_count   / length if length > 0 else 0.0
    special_ratio = special_count / length if length > 0 else 0.0

    return [
        length,        # 0
        word_count,    # 1
        avg_word_len,  # 2
        upper_ratio,   # 3
        digit_ratio,   # 4
        special_ratio  # 5
    ]


def normalize_features(features: list[float]) -> list[float]:
    """Z-score normalization using training-set mean/std."""
    if _PLACEHOLDER_GUARD:
        raise RuntimeError(
            "FEATURE_MEAN / FEATURE_STD in features.py still contain "
            "placeholder None values. Run recompute_normalization.py "
            "against your training corpus and paste the results into "
            "features.py before running inference or training."
        )
    normalized = []
    for i, val in enumerate(features):
        std = FEATURE_STD[i] if FEATURE_STD[i] >= 1e-8 else 1e-8
        normalized.append((val - FEATURE_MEAN[i]) / std)
    return normalized


def get_features_tensor(text: str):
    """Returns normalized features as a [1, 6] float32 torch tensor."""
    import torch
    raw   = extract_features(text)
    normd = normalize_features(raw)
    return torch.tensor([normd], dtype=torch.float32)   # [1, 6]