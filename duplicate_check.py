# duplicate_check.py
#
# Checks for duplicate/near-duplicate rows:
#   1. within each file
#   2. across malaysian_fake.csv vs malaysian_fake_en.csv
#
# Handles both exact text duplicates and near-duplicates (e.g. if
# malaysian_fake_en.csv is a translated version of the same source
# articles, which won't match on exact string comparison).

import re
import sys
import pandas as pd
from difflib import SequenceMatcher

FILE_A = r"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\malaysian_fake.csv"
FILE_B = r"C:\dev\python\project\Scamshield-AI Testing\data\malaysian\malaysian_fake_en.csv"
TEXT_COL_CANDIDATES = ["text", "content", "article", "body"]

NEAR_DUP_THRESHOLD = 0.85   # similarity ratio above which we call it a near-dup
BLOCK_LEN_TOLERANCE = 0.15  # only compare pairs within +/-15% char length (for speed)


def find_text_col(df: pd.DataFrame) -> str:
    for c in TEXT_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found. Columns present: {list(df.columns)}")


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)   # strip punctuation for comparison
    return text


def exact_duplicates(df: pd.DataFrame, col: str, label: str):
    normed = df[col].map(normalize)
    dup_mask = normed.duplicated(keep=False)
    n_dupes = dup_mask.sum()
    n_groups = normed[dup_mask].nunique()
    print(f"[{label}] exact duplicate rows: {n_dupes} "
          f"(across {n_groups} distinct duplicate groups), "
          f"out of {len(df)} total rows")
    return normed


def cross_file_exact_overlap(normed_a: pd.Series, normed_b: pd.Series):
    set_a, set_b = set(normed_a), set(normed_b)
    overlap = set_a & set_b
    print(f"\n[cross-file] exact text overlap between the two files: "
          f"{len(overlap)} shared normalized texts "
          f"({len(overlap)/max(len(set_a),1)*100:.1f}% of {FILE_A}'s unique texts, "
          f"{len(overlap)/max(len(set_b),1)*100:.1f}% of {FILE_B}'s unique texts)")
    return overlap


def near_duplicates_sample(texts_a, texts_b, max_pairs=20000, sample_cap=3000):
    """
    Blocked near-duplicate search: only compares pairs whose normalized
    length is within BLOCK_LEN_TOLERANCE of each other, to keep this
    roughly O(n log n) instead of O(n^2). Reports a sample of matches;
    for very large files, cap sample_cap and increase if you have time
    to spare.
    """
    a = list(texts_a)[:sample_cap]
    b = list(texts_b)[:sample_cap]
    a_sorted = sorted(range(len(a)), key=lambda i: len(a[i]))
    b_sorted = sorted(range(len(b)), key=lambda i: len(b[i]))

    matches = []
    comparisons = 0
    bj_start = 0
    for ai in a_sorted:
        la = len(a[ai])
        if la == 0:
            continue
        lo = la * (1 - BLOCK_LEN_TOLERANCE)
        hi = la * (1 + BLOCK_LEN_TOLERANCE)
        for bi in b_sorted:
            lb = len(b[bi])
            if lb < lo:
                continue
            if lb > hi:
                break
            comparisons += 1
            if comparisons > max_pairs:
                print(f"  (stopped after {max_pairs} comparisons — "
                      f"raise max_pairs or sample_cap for a fuller sweep)")
                return matches
            ratio = SequenceMatcher(None, a[ai], b[bi]).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                matches.append((ai, bi, ratio))
    return matches


if __name__ == "__main__":
    try:
        df_a = pd.read_csv(FILE_A)
        df_b = pd.read_csv(FILE_B)
    except FileNotFoundError as e:
        print(f"Could not find file: {e}. Update FILE_A/FILE_B paths at the top of the script.")
        sys.exit(1)

    col_a = find_text_col(df_a)
    col_b = find_text_col(df_b)
    print(f"{FILE_A}: {len(df_a)} rows, text column = '{col_a}'")
    print(f"{FILE_B}: {len(df_b)} rows, text column = '{col_b}'\n")

    normed_a = exact_duplicates(df_a, col_a, FILE_A)
    normed_b = exact_duplicates(df_b, col_b, FILE_B)
    overlap  = cross_file_exact_overlap(normed_a, normed_b)

    print(f"\nRunning near-duplicate sweep (threshold={NEAR_DUP_THRESHOLD}, "
          f"this may take a bit on large files)...")
    only_a = normed_a[~normed_a.isin(overlap)]
    only_b = normed_b[~normed_b.isin(overlap)]
    matches = near_duplicates_sample(only_a.tolist(), only_b.tolist())

    print(f"\nnear-duplicate cross-file pairs found (excluding exact matches): {len(matches)}")
    if matches:
        print("sample (up to 5):")
        for ai, bi, ratio in matches[:5]:
            print(f"  similarity={ratio:.3f}")
            print(f"    A: {only_a.iloc[ai][:100]}...")
            print(f"    B: {only_b.iloc[bi][:100]}...")

    total_dupe_rate = (len(overlap) + len(matches)) / max(len(normed_a), 1)
    print(f"\nEstimated total overlap rate (exact + near-dup) relative to {FILE_A}: "
          f"{total_dupe_rate*100:.1f}%")