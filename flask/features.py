# features.py
# Mirrors TextCleaner.extractFeatures() and _normalizeFeatures() from model_service.dart.
# MUST produce identical values to the Dart implementation.

import re
import math


# ── Normalization constants — copied from model_service.dart ──────────────────
FEATURE_MEAN = [
    319.2340087890625,
    0.30629757046699524,
    0.4853785037994385,
    0.0,
    5.2002982556587085e-05,
    0.008210545405745506,
]

FEATURE_STD = [
    349.07989501953125,
    1.0457844734191895,
    1.3806873559951782,
    0.0,                  # std=0 → clamped to 1e-8 to avoid div/0
    0.009309545159339905,
    0.01442260853946209,
]


def clean_text(text: str) -> str:
    """Basic text cleaning — mirrors TextCleaner.clean()."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_features(text: str) -> list[float]:
    """
    Extracts 6 raw numeric features from text.
    Must match TextCleaner.extractFeatures() in text_cleaner.dart exactly.

    Feature index mapping:
      0 — text length (character count)
      1 — word count
      2 — average word length
      3 — digit ratio
      4 — uppercase ratio  (after clean → always 0.0 since we lowercased)
      5 — special char ratio
    """
    if not text:
        return [0.0] * 6

    length      = float(len(text))
    words       = text.split()
    word_count  = float(len(words))
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0
    digit_count  = sum(1 for c in text if c.isdigit())
    upper_count  = sum(1 for c in text if c.isupper())
    special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())

    digit_ratio   = digit_count  / length if length > 0 else 0.0
    upper_ratio   = upper_count  / length if length > 0 else 0.0
    special_ratio = special_count / length if length > 0 else 0.0

    return [
        length,       # 0
        word_count,   # 1
        avg_word_len, # 2
        upper_ratio,  # 3  (0.0 after lowercasing — matches Dart behaviour)
        digit_ratio,  # 4
        special_ratio # 5
    ]


def normalize_features(features: list[float]) -> list[float]:
    """Z-score normalization using training set mean/std."""
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