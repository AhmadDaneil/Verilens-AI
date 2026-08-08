#!/usr/bin/env python3
"""
translate_real_to_english.py

VeriLens AI — translation-fingerprint fix (Option 2)

Purpose
-------
The root-cause analysis found the classifier had learned "translated text ->
fake" instead of genuine truthfulness, because every fake-news example in
training was NLLB-translated from Malay and zero real-news examples were.
This script closes that gap by translating collected Malay-language real
news into English via NLLB and merging it into malaysian_real.csv, with
explicit provenance tagging so the translated/native split can be audited
and re-checked (see ablate_was_translated_feature.py).

Input
-----
A CSV of collected Malay real news (default:
data/malaysian/malaysian_real_ms_raw.csv), as produced by
collect_malay_real_for_translation.py. Expected to contain at least a text
column and optionally a title column. If --text-col/--title-col aren't
given, common column names are auto-detected (text/content/body/
article_text/full_text, title/headline). If nothing matches, the script
prints the actual columns it found so you can pass them explicitly.

Output
------
data/malaysian/malaysian_real.csv is updated (created if missing) with the following
provenance columns on EVERY row (not just newly translated ones), so the
whole file stays self-describing:

    was_translated     bool   True if this row's English text came from NLLB
                               translation of Malay source text.
    text_original_ms   str    The original Malay text, if was_translated is
                               True; empty string otherwise.

Existing rows in data/malaysian/malaysian_real.csv that don't already have these columns
are backfilled with was_translated=False, text_original_ms="" (i.e. treated
as native English real news), so the merge never silently mislabels
pre-existing data.

Usage
-----
    python src\translate_real_to_english.py

(defaults now match your project layout — data/malaysian/*.csv — so no
flags are required for a normal run). Override if needed, e.g.:

    python src\translate_real_to_english.py --limit 5   # quick smoke test
    python src\translate_real_to_english.py --text-col article_body

Requires: pandas, torch, transformers (sentencepiece for NLLB tokenizer)
    pip install pandas torch transformers sentencepiece
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/malaysian/malaysian_real_ms_raw.csv",
                   help="CSV of collected Malay real news "
                        "(default: data/malaysian/malaysian_real_ms_raw.csv, "
                        "matching collect_malay_real_for_translation.py's output)")
    p.add_argument("--output", default="data/malaysian/malaysian_real.csv",
                   help="Target CSV to merge translated rows into "
                        "(default: data/malaysian/malaysian_real.csv)")
    p.add_argument("--text-col", default=None,
                   help="Name of the body-text column in --input. If omitted, auto-detects "
                        "from common names (text, content, body, article_text, full_text).")
    p.add_argument("--title-col", default=None,
                   help="Name of the title column in --input, if present. If omitted, "
                        "auto-detects from common names (title, headline).")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M",
                   help="NLLB model checkpoint to use (default: facebook/nllb-200-distilled-600M)")
    p.add_argument("--src-lang", default="zsm_Latn",
                   help="NLLB FLORES-200 code for source language (default: zsm_Latn = Standard Malay)")
    p.add_argument("--tgt-lang", default="eng_Latn",
                   help="NLLB FLORES-200 code for target language (default: eng_Latn = English)")
    p.add_argument("--batch-size", type=int, default=8, help="Translation batch size (default: 8)")
    p.add_argument("--max-length", type=int, default=512, help="Max token length for translation (default: 512)")
    p.add_argument("--device", default=None,
                   help="Force device ('cuda', 'cpu', 'mps'). Default: auto-detect.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of rows to translate (useful for a quick test run)")
    p.add_argument("--dedup-col", default="url",
                   help="Column to dedup on when merging into --output, if present (default: url)")
    return p.parse_args()


def load_translator(model_name: str, device: str | None):
    """Lazy-import heavy deps so --help works without torch/transformers installed."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[translate] loading {model_name} on {device} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


def translate_batch(texts, tokenizer, model, device, src_lang, tgt_lang, max_length):
    import torch

    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # Newer transformers: convert_tokens_to_ids works for NLLB lang codes directly.
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=max_length,
            num_beams=4,
        )

    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


TEXT_COL_CANDIDATES = ["text", "content", "body", "article_text", "full_text", "full_article_text"]
TITLE_COL_CANDIDATES = ["title", "headline"]


def autodetect_col(columns, candidates, kind):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    print(f"ERROR: could not auto-detect a {kind} column. "
          f"Available columns: {list(columns)}. "
          f"Pass --{'text-col' if kind == 'text' else 'title-col'} explicitly.", file=sys.stderr)
    sys.exit(1)


def main():
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        print("This is expected to be the output of collect_malay_real_for_translation.py.", file=sys.stderr)
        sys.exit(1)

    df_ms = pd.read_csv(in_path)

    text_col = args.text_col
    if text_col is None:
        text_col = autodetect_col(df_ms.columns, TEXT_COL_CANDIDATES, "text")
        print(f"[translate] auto-detected text column: '{text_col}'", file=sys.stderr)
    elif text_col not in df_ms.columns:
        print(f"ERROR: --text-col '{text_col}' not found in {in_path}. "
              f"Available columns: {list(df_ms.columns)}", file=sys.stderr)
        sys.exit(1)
    args.text_col = text_col

    title_col = args.title_col
    if title_col is None:
        lower_map = {c.lower(): c for c in df_ms.columns}
        title_col = next((lower_map[c] for c in TITLE_COL_CANDIDATES if c in lower_map), None)
        if title_col:
            print(f"[translate] auto-detected title column: '{title_col}'", file=sys.stderr)
    elif title_col not in df_ms.columns:
        print(f"ERROR: --title-col '{title_col}' not found in {in_path}. "
              f"Available columns: {list(df_ms.columns)}", file=sys.stderr)
        sys.exit(1)
    args.title_col = title_col

    if args.limit:
        df_ms = df_ms.head(args.limit).copy()

    df_ms = df_ms[df_ms[args.text_col].notna() & (df_ms[args.text_col].str.strip() != "")].reset_index(drop=True)
    print(f"[translate] {len(df_ms)} Malay real-news rows to translate", file=sys.stderr)

    if len(df_ms) == 0:
        print("[translate] nothing to translate, exiting.", file=sys.stderr)
        sys.exit(0)

    tokenizer, model, device = load_translator(args.model, args.device)

    has_title = args.title_col is not None and args.title_col in df_ms.columns

    translated_titles = []
    translated_texts = []

    # Translate body text
    texts = df_ms[args.text_col].astype(str).tolist()
    for batch in chunked(texts, args.batch_size):
        out = translate_batch(batch, tokenizer, model, device, args.src_lang, args.tgt_lang, args.max_length)
        translated_texts.extend(out)
        print(f"[translate] {len(translated_texts)}/{len(texts)} texts done", file=sys.stderr)

    # Translate title, if present
    if has_title:
        titles = df_ms[args.title_col].fillna("").astype(str).tolist()
        for batch in chunked(titles, args.batch_size):
            out = translate_batch(batch, tokenizer, model, device, args.src_lang, args.tgt_lang, args.max_length)
            translated_titles.extend(out)
        print(f"[translate] {len(translated_titles)}/{len(titles)} titles done", file=sys.stderr)

    # Build the new rows with provenance tagging
    new_rows = pd.DataFrame({
        args.text_col: translated_texts,
        "text_original_ms": df_ms[args.text_col].astype(str).values,
        "was_translated": True,
    })
    if has_title:
        new_rows[args.title_col] = translated_titles

    # Carry over any other metadata columns from the source (url, date, source, etc.)
    for col in df_ms.columns:
        if col not in new_rows.columns and col != args.text_col:
            new_rows[col] = df_ms[col].values

    # Merge into malaysian_real.csv
    out_path = Path(args.output)
    if out_path.exists():
        df_real = pd.read_csv(out_path)
        # Backfill provenance columns on pre-existing rows so nothing is
        # silently mislabeled by the merge.
        if "was_translated" not in df_real.columns:
            df_real["was_translated"] = False
        else:
            df_real["was_translated"] = df_real["was_translated"].fillna(False)
        if "text_original_ms" not in df_real.columns:
            df_real["text_original_ms"] = ""
        else:
            df_real["text_original_ms"] = df_real["text_original_ms"].fillna("")

        merged = pd.concat([df_real, new_rows], ignore_index=True, sort=False)
    else:
        merged = new_rows
        print(f"[translate] {out_path} did not exist, creating fresh.", file=sys.stderr)

    if args.dedup_col in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=[args.dedup_col], keep="last").reset_index(drop=True)
        after = len(merged)
        if before != after:
            print(f"[translate] deduped on '{args.dedup_col}': {before} -> {after} rows", file=sys.stderr)

    merged.to_csv(out_path, index=False)

    n_translated = int(merged["was_translated"].sum())
    n_total = len(merged)
    print(f"[translate] wrote {out_path}: {n_total} total rows, "
          f"{n_translated} translated ({n_translated / n_total:.1%}), "
          f"{n_total - n_translated} native English real news", file=sys.stderr)
    print("[translate] Next: back up best_model.pt -> retrain -> run "
          "ablate_was_translated_feature.py to confirm the translation-fingerprint gap has closed.",
          file=sys.stderr)


if __name__ == "__main__":
    main()