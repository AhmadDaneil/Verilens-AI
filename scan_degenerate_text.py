"""
scan_degenerate_text.py
------------------------
Full-dataset scan for degenerate repeated-token/phrase text -- the
"Summary: Summary: Summary: ..." pattern found by hand in one
sinarharian-berita row is a classic LLM decoding-loop artifact (almost
certainly from the NLLB translation step this project's real-side Malay
pipeline uses). If it happened once, it likely happened elsewhere in the
same pipeline, silently -- this scans EVERY row in both CSVs rather than
relying on spot-checking a handful of short samples.

Detection approach: for each row, split into whitespace tokens and check
whether any short n-gram (1-4 tokens) repeats consecutively enough times
to be almost certainly a decoding-loop rather than natural repetition
(natural English/Malay text essentially never repeats the same 1-4 word
phrase back-to-back 4+ times).

Usage:
    python scan_degenerate_text.py
"""

import re
import pandas as pd

REAL_CSV = "data/malaysian/malaysian_real.csv"
FAKE_CSV = "data/malaysian/malaysian_fake_en.csv"

MIN_CONSECUTIVE_REPEATS = 4  # e.g. "Summary: Summary: Summary: Summary:" = 4x
NGRAM_SIZES = (1, 2, 3, 4)


def find_degenerate_repeat(text: str):
    """Returns (repeated_phrase, repeat_count) if a degenerate consecutive
    repeat is found, else None. Checks 1-4 word n-grams."""
    tokens = text.split()
    n = len(tokens)
    if n < MIN_CONSECUTIVE_REPEATS * 2:
        return None

    for gram_size in NGRAM_SIZES:
        i = 0
        while i + gram_size * MIN_CONSECUTIVE_REPEATS <= n:
            gram = tokens[i:i + gram_size]
            repeats = 1
            j = i + gram_size
            while j + gram_size <= n and tokens[j:j + gram_size] == gram:
                repeats += 1
                j += gram_size
            if repeats >= MIN_CONSECUTIVE_REPEATS:
                return (" ".join(gram), repeats)
            i += 1
    return None


def scan_csv(path: str, label: str, text_col: str = "text", source_col: str = "source"):
    print(f"\n{'=' * 70}")
    print(f"Scanning {label}: {path}")
    print("=" * 70)

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  (file not found, skipping)")
        return

    has_source = source_col in df.columns
    flagged = []

    for i, row in df.iterrows():
        text = str(row.get(text_col, ""))
        result = find_degenerate_repeat(text)
        if result:
            phrase, count = result
            flagged.append({
                "row": i,
                "source": row.get(source_col, "(no source col)") if has_source else "(no source col)",
                "phrase": phrase,
                "repeats": count,
                "word_count": len(text.split()),
                "url": row.get("url", ""),
            })

    print(f"  Total rows scanned: {len(df)}")
    print(f"  Degenerate rows found: {len(flagged)} ({len(flagged) / max(len(df), 1) * 100:.2f}%)")

    if flagged:
        if has_source:
            by_source = pd.DataFrame(flagged)["source"].value_counts()
            print(f"\n  Breakdown by source:")
            for src, cnt in by_source.items():
                print(f"    {src}: {cnt}")

        print(f"\n  Examples (up to 10):")
        for item in flagged[:10]:
            print(f"    [row {item['row']}] source={item['source']}  "
                  f"'{item['phrase']}' x{item['repeats']}  "
                  f"({item['word_count']} words total)  {item['url']}")

    return flagged


if __name__ == "__main__":
    real_flagged = scan_csv(REAL_CSV, "REAL-class data")
    fake_flagged = scan_csv(FAKE_CSV, "FAKE-class data")

    total_flagged = len(real_flagged or []) + len(fake_flagged or [])
    print(f"\n{'=' * 70}")
    print(f"TOTAL degenerate rows across both files: {total_flagged}")
    if total_flagged > 0:
        print("These rows are not real article content -- they're decoding-loop")
        print("garbage and should be removed before retraining, not just from")
        print("the specific source where the first one was spotted by hand.")
    print("=" * 70)