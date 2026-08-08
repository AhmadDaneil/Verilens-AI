"""
translate_fake_to_english.py
------------------------------
Translates data/malaysian/malaysian_fake.csv from Bahasa Malaysia to
English using facebook/nllb-200-distilled-600M, so the fake-class text
is in the same language as the real-class text (Malay Mail, Bernama,
Borneo Post are all English). This removes the language-leakage bug
that caused the last fine-tune to hit 100% F1 in epoch 1 by learning
"is this Malay or English" instead of "is this true or false."

Prerequisites:
    pip install transformers sentencepiece torch tqdm pandas

Note: facebook/nllb-200-distilled-600M is CC-BY-NC licensed (research /
non-commercial use). Fine for building/testing your training pipeline;
swap to a commercially-licensed MT API before any commercial deployment.

Usage:
    python translate_fake_to_english.py
"""

import os
import re
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

INPUT_CSV  = "data/malaysian/malaysian_fake.csv"
OUTPUT_CSV = "data/malaysian/malaysian_fake_en.csv"

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG   = "zsm_Latn"  # Standard Malay
TGT_LANG   = "eng_Latn"  # English

MAX_CHUNK_CHARS = 800  # keep well under the model's 512-token limit per chunk


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split on sentence boundaries so we don't cut mid-sentence, while
    keeping each chunk under the model's effective length limit.
    NLLB degrades on inputs beyond ~512 tokens, and these articles are
    often much longer than that.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


def main():
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Rows to translate: {len(df)}")

    print(f"\nLoading {MODEL_NAME} (first run downloads ~2.4GB, cached after)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        max_length=512,
    )
    print("✅ Model loaded\n")

    # Resume support: if OUTPUT_CSV already exists (partial run), skip
    # rows already translated instead of starting over.
    already_done = {}
    if os.path.exists(OUTPUT_CSV):
        prior = pd.read_csv(OUTPUT_CSV)
        already_done = dict(zip(prior["url"], prior["text"]))
        print(f"Resuming — {len(already_done)} rows already translated\n")

    translated_texts = []
    failed_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Translating"):
        url = row.get("url", "")

        if url in already_done:
            translated_texts.append(already_done[url])
            continue

        original_text = str(row.get("text", ""))
        if not original_text.strip():
            translated_texts.append("")
            continue

        try:
            chunks = chunk_text(original_text)
            translated_chunks = [
                translator(chunk)[0]["translation_text"] for chunk in chunks
            ]
            translated_texts.append(" ".join(translated_chunks))
        except Exception as e:
            print(f"\n  ⚠️  Failed to translate row ({url}): {e}")
            translated_texts.append("")  # keep row alignment, mark as empty
            failed_count += 1

        # Periodic checkpoint save every 100 rows, so a crash partway
        # through 1,898 articles doesn't lose everything already done.
        if len(translated_texts) % 100 == 0:
            checkpoint_df = df.iloc[: len(translated_texts)].copy()
            checkpoint_df["text_original_ms"] = checkpoint_df["text"]
            checkpoint_df["text"] = translated_texts
            tmp_path = OUTPUT_CSV + ".tmp"
            checkpoint_df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, OUTPUT_CSV)

    df["text_original_ms"] = df["text"]  # keep original for reference/audit
    df["text"] = translated_texts

    tmp_path = OUTPUT_CSV + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, OUTPUT_CSV)

    empty_count = sum(1 for t in translated_texts if not t.strip())
    print(f"\n✅ Done. Saved → {OUTPUT_CSV}")
    print(f"   Translated: {len(translated_texts) - empty_count}")
    print(f"   Failed/empty: {empty_count}")
    if failed_count:
        print(
            f"   ⚠️  {failed_count} rows failed translation and were left "
            f"empty — filter these out before training (df[df['text'].str.len() > 0])"
        )


if __name__ == "__main__":
    main()