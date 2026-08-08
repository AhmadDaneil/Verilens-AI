"""
Fallback: approximate FEATURE_MEAN / FEATURE_STD for features.py.

── SUPERSEDED PRIMARY SOURCE ──────────────────────────────────────────────
features.py now loads its normalization stats directly from
feature_stats_malaysian.npz — the exact mean/std that
retrain_with_malaysian.py computed and saved during the run that produced
the currently-deployed best_model.pt. That file is the correct source of
truth and should be used whenever it's available; this script is only a
fallback for regenerating an *approximation* if that npz has been lost.

── BUG HISTORY (for context, not still true) ──────────────────────────────
An earlier version of this script (and of features.py) used a feature
definition — char length, word count, avg word length, uppercase ratio,
digit ratio, special char ratio — that did NOT match what
retrain_with_malaysian.py actually trained best_model.pt on (word count,
exclamation count, question count, uppercase ratio, 'http' count, digit
ratio). Both features.py and this script have since been corrected to use
that same training-matching extract_features() (imported from features.py,
so fixing it there fixes it here automatically). If you see either file
using char-length/avg-word-length/special-char-ratio again, that's a
regression — revert it.

USAGE (only if feature_stats_malaysian.npz is unavailable):
  1. Edit the CONFIG section below to point at your actual data files —
     ideally the SAME corpus retrain_with_malaysian.py trains on, including
     its replay-buffer sampling, for the closest possible match.
  2. Run:  python recompute_normalization.py
  3. This is an approximation of the real training-run stats, not a
     replacement for them. Prefer copying the actual
     feature_stats_malaysian.npz from your training run whenever possible.
"""
import glob
import numpy as np
import pandas as pd

from features import extract_features, clean_text

# ── CONFIG: point these at your real dataset files ─────────────────────────
# Add/remove paths as needed. Each entry needs a path + which column(s)
# hold the article text. Set to None / remove entries you don't have.
CONFIG = {
    "malaysian_real":            "../data/malaysian/malaysian_real.csv",
    "malaysian_fake":            "../data/malaysian/malaysian_fake.csv",
    # NOTE: malaysian_real_ms_raw.csv deliberately excluded — it's the
    # pre-translation raw Malay text underlying malaysian_real.csv, and
    # including both would double-count the same underlying articles.
    # malaysian_fake_en.csv also excluded for now — verify whether it's
    # a duplicate/superset of malaysian_fake.csv before adding it; if it's
    # a distinct additional fake set, add it as its own entry below.
    "isot_true":                 "../data/isot/True.csv",
    "isot_fake":                 "../data/isot/Fake.csv",
    "fakenewsnet_gossipcop_real": "../data/fakenewsnet/gossipcop_real.csv",
    "fakenewsnet_gossipcop_fake": "../data/fakenewsnet/gossipcop_fake.csv",
    "fakenewsnet_politifact_real": "../data/fakenewsnet/politifact_real.csv",
    "fakenewsnet_politifact_fake": "../data/fakenewsnet/politifact_fake.csv",
    "liar_train":                 "../data/liar/train.tsv",
    "liar_valid":                 "../data/liar/valid.tsv",
    # NOTE: deliberately NOT including liar/test.tsv here, to keep it as a
    # clean held-out set for ablate_was_translated_feature.py, same
    # discipline used when building the replay buffer.
}

TEXT_COL_CANDIDATES  = ["text", "content", "article", "body", "statement", "text_original_ms"]
TITLE_COL_CANDIDATES = ["title", "headline"]


def _find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def load_texts_from_csv(path):
    """Auto-detect text/title columns and return a list of article strings."""
    try:
        if path.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t", header=None, quoting=3, on_bad_lines="skip")
            # LIAR format has no header; text is typically column index 2
            if df.shape[1] > 2:
                return df[2].dropna().astype(str).tolist()
            return []
        df = pd.read_csv(path, on_bad_lines="skip")
    except FileNotFoundError:
        print(f"  [skip] {path} not found")
        return []
    except Exception as e:
        print(f"  [skip] {path} failed to load: {e}")
        return []

    text_col  = _find_col(df, TEXT_COL_CANDIDATES)
    title_col = _find_col(df, TITLE_COL_CANDIDATES)

    if text_col is None and title_col is None:
        print(f"  [skip] {path}: no recognizable text/title column found (columns: {list(df.columns)})")
        return []

    texts = []
    for _, row in df.iterrows():
        parts = []
        if title_col and pd.notna(row.get(title_col)):
            parts.append(str(row[title_col]))
        if text_col and pd.notna(row.get(text_col)):
            parts.append(str(row[text_col]))
        combined = " ".join(parts).strip()
        if combined:
            texts.append(combined)
    return texts


def main():
    all_texts = []
    print("Loading corpus...")
    for name, path in CONFIG.items():
        if not path:
            continue
        texts = load_texts_from_csv(path)
        print(f"  {name:20s} -> {len(texts)} rows from {path}")
        all_texts.extend(texts)

    if not all_texts:
        print("\nNo data loaded. Edit CONFIG paths at the top of this script and re-run.")
        return

    print(f"\nTotal texts: {len(all_texts)}")
    print("Computing raw features for every example (this may take a minute)...")

    raw_feature_rows = []
    for i, text in enumerate(all_texts):
        cleaned = clean_text(text)
        feats = extract_features(cleaned)
        raw_feature_rows.append(feats)
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(all_texts)}")

    arr = np.array(raw_feature_rows, dtype=np.float64)  # [N, 6]

    means = arr.mean(axis=0)
    stds  = arr.std(axis=0)

    feature_names = [
        "text length (char count)",
        "word count",
        "average word length",
        "uppercase ratio (post-lowercase, always 0)",
        "digit ratio",
        "special char ratio",
    ]

    print("\n=== Computed statistics (current, correct feature definitions) ===")
    for i, name in enumerate(feature_names):
        print(f"  [{i}] {name:45s} mean={means[i]:.6f}  std={stds[i]:.6f}")

    print("\n=== Paste these into features.py, replacing FEATURE_MEAN / FEATURE_STD ===\n")
    print("FEATURE_MEAN = [")
    for m in means:
        print(f"    {m},")
    print("]")
    print()
    print("FEATURE_STD = [")
    for s in stds:
        print(f"    {s},")
    print("]")

    print("\nSanity check -- normalize a few percentiles of word count to confirm sane range:")
    word_counts = arr[:, 1]
    for pct in [10, 50, 90, 99]:
        wc = np.percentile(word_counts, pct)
        z = (wc - means[1]) / (stds[1] if stds[1] >= 1e-8 else 1e-8)
        print(f"  p{pct} word_count={wc:.0f}  ->  z-score={z:+.3f}")


if __name__ == "__main__":
    main()