"""
inspect_flagged_rows.py
------------------------
Follow-up to audit_malaysian_data.py's two flagged findings:

  1. 5 rows in malaysian_fake_en.csv matched a "confirmed TRUE" verdict
     keyword. audit_malaysian_data.py only prints the first 200 chars of
     each row, which cuts off before the matched phrase -- this script
     prints context AROUND the actual match so you can tell a genuine
     mislabel from a false-positive keyword match (e.g. a claim that's
     part-true/part-false, or a negation the regex didn't account for).

  2. sinarharian-berita (n=96, mean 91 words) and sinarharian-latest
     (n=30, mean 87 words) are real-class sources with FAKE-class-like
     length -- dramatically shorter than every other real source
     (335-474 words). Prints full text for a handful of rows from each
     so you can tell genuine short news briefs from a broken extraction
     (e.g. only a teaser/headline captured instead of the full body).

Usage:
    python inspect_flagged_rows.py
"""

import re
import pandas as pd

REAL_CSV = "data/malaysian/malaysian_real.csv"
FAKE_CSV = "data/malaysian/malaysian_fake_en.csv"

CONTEXT_CHARS = 150  # chars shown before/after the matched keyword


def inspect_verdict_mislabels():
    print("=" * 70)
    print("1) VERDICT-MISLABEL CANDIDATES — full context around the match")
    print("=" * 70)

    df = pd.read_csv(FAKE_CSV)
    if "text_original_ms" not in df.columns:
        print("⚠️  No 'text_original_ms' column -- can't run this check.\n")
        return

    text = df["text_original_ms"].astype(str)
    true_pattern = re.compile(r"(adalah|disahkan)\s+benar", re.IGNORECASE)

    matches_found = 0
    for i, row_text in text.items():
        m = true_pattern.search(row_text)
        if not m:
            continue
        matches_found += 1
        start = max(0, m.start() - CONTEXT_CHARS)
        end = min(len(row_text), m.end() + CONTEXT_CHARS)
        context = row_text[start:end]
        # highlight the matched phrase itself
        matched_phrase = row_text[m.start():m.end()]

        print(f"\n[row {i}]  matched: \"{matched_phrase}\"")
        print(f"  ...{context}...")
        print("-" * 70)

    print(f"\nTotal matches: {matches_found}")
    print("For each one above, decide: genuine 'confirmed TRUE' mislabel "
          "(remove/relabel) vs false-positive keyword match (leave as is, "
          "maybe tighten the regex). There's no automatic way to tell the "
          "difference reliably -- Malay negation and multi-claim posts can "
          "both trip the simple keyword pattern.\n")


def inspect_sinarharian_samples(n_per_source: int = 5):
    print("=" * 70)
    print("2) SINARHARIAN OUTLIER SPOT-CHECK — full text, shortest rows first")
    print("=" * 70)

    df = pd.read_csv(REAL_CSV)
    if "source" not in df.columns:
        print("⚠️  No 'source' column -- can't run this check.\n")
        return

    for source_name in ["sinarharian-berita", "sinarharian-latest"]:
        subset = df[df["source"] == source_name].copy()
        if len(subset) == 0:
            print(f"(no rows found for source='{source_name}')\n")
            continue

        subset["word_count"] = subset["text"].astype(str).apply(lambda t: len(t.split()))
        shortest = subset.sort_values("word_count").head(n_per_source)

        print(f"\n--- {source_name} (n={len(subset)} total, showing {len(shortest)} shortest) ---")
        for i, row in shortest.iterrows():
            url = row.get("url", "(no url)")
            print(f"\n  [row {i}]  {row['word_count']} words  —  {url}")
            print(f"  full text: {row['text']}")
            print("  " + "-" * 60)

    print("\nFor each sample above: does the text look like a genuinely "
          "complete short news brief, or does it look cut off / like only "
          "a headline+teaser was captured? If it's the latter, that's an "
          "extraction bug in whatever scraper/collector tagged rows with "
          "this source, not a real content difference -- worth checking "
          "that scraper's article-body selector against the site's current "
          "HTML structure (same category of bug as the thestar.com.my "
          "popup-scraping issue found earlier in this project).\n")


if __name__ == "__main__":
    inspect_verdict_mislabels()
    inspect_sinarharian_samples()