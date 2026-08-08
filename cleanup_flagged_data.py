"""
cleanup_flagged_data.py
------------------------
Removes two categories of confirmed-bad rows found during this audit
pass, before they get blended into a retrain:

  1. DEGENERATE TEXT: rows where a short phrase repeats consecutively
     4+ times (LLM decoding-loop artifacts, e.g. "Summary: Summary: ..."
     or "official" x202) -- re-detected here via the same logic as
     scan_degenerate_text.py rather than trusting hardcoded row numbers,
     so this stays correct even if the CSVs change between runs.
     Confirmed: 5 rows in malaysian_real.csv (all sinarharian-berita),
     15 rows in malaysian_fake_en.csv.

  2. VERDICT MISLABELS: 3 specific rows in malaysian_fake_en.csv
     manually confirmed (by reading full context) to actually say a
     claim IS true, not debunk it as false -- genuine contamination,
     not false-positive keyword matches:
       - row 901:  "...complaints from the public are true and
         well-founded" (confirming a complaint, not debunking a hoax)
       - row 1709: "...the SMS shortcode 66628 [is] genuine" (confirms
         an SMS as authentic, opposite of a scam debunk)
       - row 1733: "...the account number is genuine, registered to..."
         (confirms an account as authentic, opposite of a scam debunk)
     NOT removed (reviewed and kept):
       - row 1454: false-positive keyword match -- actual text says
         "NOT confirmed true" (negation the simple regex missed)
       - row 31: genuinely ambiguous compound claim (a circulating
         document's attribution was false even though the policy it
         described was real) -- left as fake, flagged for a human call
         rather than auto-removed.
     Removal is guarded by a text-content check, not just row-index
     trust: if the row at that index no longer contains the expected
     matched phrase (meaning the CSV changed since this was written),
     the script stops with an error instead of silently deleting the
     wrong row.

Both files are backed up (.bak) before being overwritten, and writes
are atomic (write to .tmp, then os.replace), matching the pattern
already used elsewhere in this project (collect_real_incremental.py,
scrape_malaysian_news.py's _merge_real_results).

Usage:
    python cleanup_flagged_data.py            # dry run -- report only
    python cleanup_flagged_data.py --apply     # actually remove + save
"""

import os
import sys
import shutil
import pandas as pd

from scan_degenerate_text import find_degenerate_repeat

REAL_CSV = "data/malaysian/malaysian_real.csv"
FAKE_CSV = "data/malaysian/malaysian_fake_en.csv"

# (row_index, expected substring that must still be present at that row
#  for the removal to proceed -- safety check against index drift)
CONFIRMED_MISLABELS = [
    (901, "adalah benar dan berasas"),
    (1709, "shortcode 66628 adalah benar"),
    (1733, "adalah benar dan didaftarkan"),
]


def find_degenerate_rows(df: pd.DataFrame, text_col: str = "text") -> list[int]:
    flagged = []
    for i, row in df.iterrows():
        text = str(row.get(text_col, ""))
        if find_degenerate_repeat(text):
            flagged.append(i)
    return flagged


def backup_and_write(df: pd.DataFrame, path: str, apply: bool):
    if not apply:
        return
    if os.path.exists(path):
        backup_path = path + ".bak"
        shutil.copy2(path, backup_path)
        print(f"  Backed up existing file -> {backup_path}")
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)
    print(f"  Saved -> {path}")


def clean_real_csv(apply: bool):
    print("=" * 70)
    print(f"REAL-class data: {REAL_CSV}")
    print("=" * 70)
    df = pd.read_csv(REAL_CSV)
    before = len(df)

    degenerate_idx = find_degenerate_rows(df)
    print(f"  Degenerate rows found: {len(degenerate_idx)} -> {degenerate_idx}")

    df_clean = df.drop(index=degenerate_idx)
    after = len(df_clean)
    print(f"  Rows: {before} -> {after} ({before - after} removed)")

    backup_and_write(df_clean, REAL_CSV, apply)
    print()


def clean_fake_csv(apply: bool):
    print("=" * 70)
    print(f"FAKE-class data: {FAKE_CSV}")
    print("=" * 70)
    df = pd.read_csv(FAKE_CSV)
    before = len(df)

    degenerate_idx = find_degenerate_rows(df)
    print(f"  Degenerate rows found: {len(degenerate_idx)} -> {degenerate_idx}")

    # Safety-checked verdict-mislabel removals
    mislabel_idx = []
    text_col = df["text_original_ms"] if "text_original_ms" in df.columns else None
    for idx, expected_substring in CONFIRMED_MISLABELS:
        if idx not in df.index:
            print(f"  ⚠️ ERROR: row {idx} no longer exists in the CSV -- "
                  f"skipping this removal. File may have changed since "
                  f"this script was written. Verify manually.")
            continue
        if text_col is None:
            print(f"  ⚠️ ERROR: no 'text_original_ms' column -- can't "
                  f"safety-check row {idx}. Skipping removal.")
            continue
        actual_text = str(text_col.loc[idx])
        if expected_substring not in actual_text:
            print(f"  ⚠️ ERROR: row {idx} does not contain expected text "
                  f"'{expected_substring}' -- index may have drifted. "
                  f"REFUSING to remove this row automatically. "
                  f"Verify manually before proceeding.")
            sys.exit(1)
        mislabel_idx.append(idx)

    print(f"  Verdict-mislabel rows confirmed and removed: {mislabel_idx}")

    all_remove_idx = sorted(set(degenerate_idx) | set(mislabel_idx))
    df_clean = df.drop(index=all_remove_idx)
    after = len(df_clean)
    print(f"  Rows: {before} -> {after} ({before - after} removed: "
          f"{len(degenerate_idx)} degenerate + {len(mislabel_idx)} mislabel, "
          f"minus any overlap)")

    backup_and_write(df_clean, FAKE_CSV, apply)
    print()


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    if not apply:
        print("DRY RUN -- no files will be modified. Re-run with --apply "
              "to actually remove rows and save.\n")

    clean_real_csv(apply)
    clean_fake_csv(apply)

    if not apply:
        print("This was a dry run. Review the counts above, then re-run "
              "with:  python cleanup_flagged_data.py --apply")