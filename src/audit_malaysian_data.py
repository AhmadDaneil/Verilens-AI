"""
audit_malaysian_data.py
--------------------------
Diagnoses two suspected root causes behind the model's suspiciously
perfect validation scores and poor real-world performance:

  A) GENRE SHORTCUT: real articles are full news stories, fake articles
     are short Sebenarnya-style bulletins. The model may be learning
     "long narrative vs short templated bulletin" instead of
     truthfulness.

  B) VERDICT MISLABELING: Sebenarnya.my publishes both "this claim is
     FALSE" and "this claim is actually TRUE, here's confirmation"
     posts. If the scraper labeled every Sebenarnya post as fake=0
     without checking the post's own verdict, some genuinely TRUE
     content is contaminating the fake-class training data.

  C) PER-SOURCE BREAKDOWN (added): malaysian_real.csv now accumulates
     rows from several different scrapers/discovery methods over time
     (collect_real_incremental.py's "thestar"/"bernama"/"fmt"/"malaymail"
     tags, and scrape_malaysian_news.py's "thestar_bulk"/"bernama_bulk"/
     "fmt_bulk"/"malaymail_bulk" tags -- suffixed "_bulk" specifically to
     keep them distinguishable in the same CSV). Part A/B only compare
     REAL-as-a-whole vs FAKE-as-a-whole, which can't catch a newly added
     batch skewing length/structure differently from the rest of the
     real-class pool -- e.g. if one discovery method (Bing RSS decode,
     paginated scrape, etc.) happens to pull shorter stubs, boilerplate-
     heavy pages, or extraction artifacts that the others don't. This
     section groups by the `source` column so a new batch's stats are
     visible on their own before they get blended into a retrain.

Usage:
    python audit_malaysian_data.py
"""

import pandas as pd
import numpy as np

REAL_CSV = "data/malaysian/malaysian_real.csv"
FAKE_CSV = "data/malaysian/malaysian_fake_en.csv"


def _clean_text_series(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    """astype(str) on a column with real NaNs silently turns them into the
    literal string "nan", which quietly distorts length stats instead of
    surfacing a data problem. Filter genuinely missing text out first and
    report how many rows that was."""
    missing = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
    if missing.sum() > 0:
        print(f"  ⚠️ {missing.sum()} row(s) have missing/empty '{text_col}' -- "
              f"excluded from stats below.")
    return df.loc[~missing, text_col].astype(str)


def print_length_stats(df: pd.DataFrame, label: str, text_col: str = "text"):
    texts = _clean_text_series(df, text_col)
    lengths_words = texts.apply(lambda t: len(t.split()))
    lengths_chars = texts.apply(len)

    print(f"--- {label} (n={len(df)}) ---")
    print(f"  Word count  — mean: {lengths_words.mean():.0f}  "
          f"median: {lengths_words.median():.0f}  "
          f"min: {lengths_words.min()}  max: {lengths_words.max()}")
    print(f"  Char count  — mean: {lengths_chars.mean():.0f}  "
          f"median: {lengths_chars.median():.0f}")
    print()
    return lengths_words


def check_boilerplate(df: pd.DataFrame, label: str, text_col: str = "text"):
    patterns = {
        "starts with 'According to'": r"^\s*according to",
        "contains 'SOURCE:' / 'SUMBER:'": r"\b(source|sumber)\s*:",
        "contains 'TRUTH:' (translated SEBENARNYA)": r"\btruth\s*:",
        "contains 'official statement'": r"official statement",
    }
    print(f"--- Boilerplate pattern rates: {label} ---")
    texts = _clean_text_series(df, text_col).str.lower()
    for name, pattern in patterns.items():
        pct = texts.str.contains(pattern, regex=True, na=False).mean() * 100
        print(f"  {name}: {pct:.1f}%")
    print()


def check_verdict_mislabeling(df: pd.DataFrame):
    """
    Scans the ORIGINAL Malay text (before translation) for verdict
    keywords. Sebenarnya-style fact-checks typically state their verdict
    explicitly in Malay:
      - "adalah palsu" / "disahkan palsu"  -> confirmed FALSE
      - "adalah benar" / "disahkan benar"  -> confirmed TRUE  <- these
        should NOT be in your fake-class training data if present
      - "mengelirukan"                     -> misleading (ambiguous,
        arguably still fake-adjacent, but worth knowing the split)
    """
    if "text_original_ms" not in df.columns:
        print("⚠️  No 'text_original_ms' column found — did you use the "
              "translated CSV that keeps the original Malay text? "
              "Skipping verdict-mislabeling check.\n")
        return

    text = df["text_original_ms"].astype(str).str.lower()

    false_pattern = r"(adalah|disahkan)\s+palsu"
    true_pattern  = r"(adalah|disahkan)\s+benar"
    misleading_pattern = r"mengelirukan"

    is_false = text.str.contains(false_pattern, regex=True, na=False)
    is_true  = text.str.contains(true_pattern, regex=True, na=False)
    is_misleading = text.str.contains(misleading_pattern, regex=True, na=False)
    unmarked = ~(is_false | is_true | is_misleading)

    n = len(df)
    print("--- Verdict keyword scan (fake-class data, original Malay text) ---")
    print(f"  Explicitly marked FALSE (palsu):        {is_false.sum():>5} ({is_false.mean()*100:.1f}%)")
    print(f"  Explicitly marked TRUE (benar):          {is_true.sum():>5} ({is_true.mean()*100:.1f}%)  <- potential mislabels")
    print(f"  Marked MISLEADING (mengelirukan):        {is_misleading.sum():>5} ({is_misleading.mean()*100:.1f}%)")
    print(f"  No clear verdict keyword found:          {unmarked.sum():>5} ({unmarked.mean()*100:.1f}%)")
    print()

    if is_true.sum() > 0:
        print(f"⚠️  Found {is_true.sum()} rows in your FAKE-class training data "
              f"that appear to confirm a claim as TRUE, not false. These are "
              f"likely mislabeled and should be removed or re-labeled before "
              f"retraining. Showing up to 3 examples:\n")
        examples = df[is_true][["text_original_ms"]].head(3)
        for i, row in examples.iterrows():
            print(f"  [row {i}] {row['text_original_ms'][:200]}...\n")
    else:
        print("✅ No explicit 'confirmed TRUE' keyword matches found in the "
              "fake-class data — verdict mislabeling doesn't look like the "
              "issue, at least not via this keyword pattern. (Note: this is "
              "a heuristic keyword scan, not exhaustive — Sebenarnya may "
              "phrase true-verdicts differently than assumed here.)\n")


def print_per_source_breakdown(df: pd.DataFrame, text_col: str = "text"):
    """
    Groups real-class rows by their `source` tag and prints length +
    boilerplate stats per source, so a newly added batch (e.g.
    thestar_bulk/bernama_bulk/fmt_bulk from a fixed scraper) can be
    checked for skew against the rest of the pool BEFORE it gets blended
    into a retrain, rather than only ever seeing REAL-as-a-whole averages.
    """
    if "source" not in df.columns:
        print("⚠️  No 'source' column found — skipping per-source breakdown.\n")
        return

    print("=" * 60)
    print("C) PER-SOURCE BREAKDOWN (real-class data)")
    print("=" * 60)

    boilerplate_patterns = {
        "starts with 'According to'": r"^\s*according to",
        "contains 'SOURCE:' / 'SUMBER:'": r"\b(source|sumber)\s*:",
    }

    summary_rows = []
    for source_name, group in df.groupby("source"):
        texts = _clean_text_series(group, text_col)
        if len(texts) == 0:
            continue
        word_counts = texts.apply(lambda t: len(t.split()))
        texts_lower = texts.str.lower()

        row = {
            "source": source_name,
            "n": len(group),
            "mean_words": round(word_counts.mean(), 0),
            "median_words": round(word_counts.median(), 0),
            "min_words": int(word_counts.min()),
        }
        for name, pattern in boilerplate_patterns.items():
            row[name] = round(
                texts_lower.str.contains(pattern, regex=True, na=False).mean() * 100, 1
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("n", ascending=False)
    print(summary_df.to_string(index=False))
    print()

    # Flag sources whose mean length deviates sharply from the overall
    # real-class mean -- the same >2x heuristic used for the REAL vs FAKE
    # comparison in Part A, applied within the real class across sources.
    overall_mean = summary_df["mean_words"].mean()
    outliers = summary_df[
        (summary_df["mean_words"] > overall_mean * 2)
        | (summary_df["mean_words"] < overall_mean * 0.5)
    ]
    if len(outliers) > 0:
        print("⚠️  Source(s) with length noticeably different from the "
              "cross-source average — worth a manual spot-check before "
              "retraining on them:")
        for _, row in outliers.iterrows():
            print(f"    {row['source']}: mean {row['mean_words']:.0f} words "
                  f"vs overall average {overall_mean:.0f}")
        print()
    else:
        print("✅ No source stands out as a length outlier relative to the "
              "others.\n")


def main():
    print("Loading data...\n")
    real_df = pd.read_csv(REAL_CSV)
    fake_df = pd.read_csv(FAKE_CSV)

    print("=" * 60)
    print("A) LENGTH / STRUCTURE COMPARISON")
    print("=" * 60)
    real_lengths = print_length_stats(real_df, "REAL")
    fake_lengths = print_length_stats(fake_df, "FAKE")

    ratio = real_lengths.mean() / max(fake_lengths.mean(), 1)
    print(f"Real-class articles are ~{ratio:.1f}x longer than fake-class "
          f"on average.")
    if ratio > 2:
        print("⚠️  This is a large enough gap that length/structure alone "
              "could be a usable shortcut for the model — independent of "
              "truthfulness.\n")
    else:
        print("Gap is modest — less likely to be a dominant shortcut on "
              "its own.\n")

    check_boilerplate(real_df, "REAL")
    check_boilerplate(fake_df, "FAKE")

    print("=" * 60)
    print("B) VERDICT MISLABELING CHECK")
    print("=" * 60)
    check_verdict_mislabeling(fake_df)

    print_per_source_breakdown(real_df)


if __name__ == "__main__":
    main()