"""
ablate_was_translated_feature.py
---------------------------------
Confirms whether the translation-fingerprint shortcut ("translated text ->
fake") has actually been closed after adding translated Malaysian real news
to the training set.

Two-part check:

  PART 1 — Raw-data label leakage check
      Does `was_translated` still perfectly (or near-perfectly) predict
      `label` in the raw training data? If real news is now a healthy mix
      of translated/native (like fake news already was), the leakage should
      be gone at the data level, independent of any model.

  PART 2 — Retrained-model behavior check
      Run the actual best_model.pt (HybridModel: DistilBERT+BiLSTM+6
      engineered features) on four held-out slices:
          native fake   (e.g. ISOT/LIAR/FakeNewsNet)
          translated fake (malaysian_fake_en.csv)
          native real   (a native-English real-news set)
          translated real (malaysian_real.csv, was_translated==True rows)
      and compares native-vs-translated accuracy within each class. If the
      shortcut is gone, native and translated accuracy for a given class
      should be close (originally: 80.0% native fake vs 99.78% translated
      fake — a ~20pt gap that should shrink dramatically).

  ── TRAIN/EVAL OVERLAP FIX (previous version) ──────────────────────────
      v2 of retrain_with_malaysian.py added a "native replay buffer" that
      samples 500 fake + 500 real rows directly out of the full ISOT /
      FakeNewsNet / LIAR(train+valid) pools and trains on them. This
      script's native-fake / native-real slices were previously built from
      those SAME full files, with no exclusion — so a slice of "native"
      eval data had actually been trained on, inflating native accuracy
      and understating the true native/translated gap.

      Fix: this script loads a replay-buffer manifest (a CSV listing
      exactly which native rows were sampled into training, written by the
      patched retrain script) and drops any matching rows from the
      native-fake / native-real eval slices before scoring.

  ── TRAIN/EVAL OVERLAP FIX (this version) — OTHER SIDE OF THE COMPARISON ──
      The same overlap problem exists on the translated side, and it was
      NOT fixed by the replay-manifest change above. retrain_with_malaysian.py
      trains on a large majority of malaysian_fake_en.csv (capped by
      MAX_IMBALANCE_RATIO, but still hundreds of rows) and most of
      malaysian_real.csv, then this script was evaluating "translated fake"
      / "translated real" directly against those SAME raw CSVs with zero
      exclusion. That meant the translated-side accuracy numbers were very
      likely inflated by the model having directly memorized much of what
      it was being scored against — while the native side, after the fix
      above, was a genuinely held-out check. Comparing a contaminated
      number against a clean one is not a fair native-vs-translated gap.

      Fix: this script now also loads malaysian_val_manifest.csv (written
      by retrain_with_malaysian.py right after its own train_test_split —
      see companion NOTE at the bottom of this file) and RESTRICTS the
      translated-fake / translated-real eval slices to only the rows in
      that held-out val split, instead of scoring against the full raw
      CSVs. If no manifest is found, the script still runs but prints a
      loud warning, the same way it already does for the replay manifest.

Native data sources (defaults match your confirmed project paths)
-------------------------------------------------------------------
  ISOT:         data/isot/Fake.csv, data/isot/True.csv
                (title+text columns, combined)
  FakeNewsNet:  data/fakenewsnet/{gossipcop,politifact}_{fake,real}.csv
                NOTE: the standard FakeNewsNet release only ships
                id/news_url/title/tweet_ids — no full article body. If your
                files are the stock version, this slice tests on short
                titles rather than full text, which limits how directly
                comparable its accuracy is to the other (full-text) slices.
                The script auto-detects a fuller text column if you've
                scraped article bodies into these files; otherwise it
                warns and falls back to title.
  LIAR:         data/liar/test.tsv by default (held-out split only —
                train.tsv/valid.tsv are excluded by default since
                best_model.pt may have already been trained on them
                before the Malaysian work started; pass --liar-tsv
                explicitly to include them if that's not the case for you).
                6-way labels are binarized: pants-fire/false/barely-true
                -> fake, half-true/mostly-true/true -> real.

Usage
-----
    python ablate_was_translated_feature.py
    # (defaults now match your confirmed paths — override individual
    # sources with --isot-fake, --fakenewsnet-fake, --liar-tsv, etc. if needed)

    python ablate_was_translated_feature.py --sample-size 200   # quick smoke test

    python ablate_was_translated_feature.py --replay-manifest replay_buffer_manifest.csv
    # (default path — override if retrain script wrote it somewhere else)

    python ablate_was_translated_feature.py --malaysian-val-manifest malaysian_val_manifest.csv
    # (default path — override if retrain script wrote it somewhere else)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
from transformers import DistilBertTokenizer, DistilBertModel


# ── Reproduced EXACTLY from retrain_with_malaysian.py ──────────────────────
# These must match training precisely, or this ablation tests a different
# feature space than what's actually deployed in best_model.pt.

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text):
    words = text.split()
    return np.array([
        len(words),
        text.count("!"),
        text.count("?"),
        sum(1 for c in text if c.isupper()) / max(len(text), 1),
        text.count("http"),
        sum(c.isdigit() for c in text) / max(len(text), 1),
    ], dtype=np.float32)


class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert     = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bilstm   = nn.LSTM(768, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout  = nn.Dropout(0.3)
        self.fc1      = nn.Linear(1024 + 6, 512)
        self.batchnorm= nn.BatchNorm1d(512)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq         = outputs.last_hidden_state
        lstm_out, _ = self.bilstm(seq)
        mean_pool   = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled      = torch.cat((mean_pool, max_pool), dim=1)
        combined    = torch.cat((pooled, extra_features), dim=1)
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc2(x)


# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ISOT
    p.add_argument("--isot-fake", default="data/isot/Fake.csv", help="ISOT fake news CSV")
    p.add_argument("--isot-real", default="data/isot/True.csv", help="ISOT real news CSV")

    # FakeNewsNet (standard repo format: id,news_url,title,tweet_ids — no full body by default)
    p.add_argument("--fakenewsnet-fake", nargs="+",
                    default=["data/fakenewsnet/gossipcop_fake.csv", "data/fakenewsnet/politifact_fake.csv"],
                    help="FakeNewsNet fake news CSV(s)")
    p.add_argument("--fakenewsnet-real", nargs="+",
                    default=["data/fakenewsnet/gossipcop_real.csv", "data/fakenewsnet/politifact_real.csv"],
                    help="FakeNewsNet real news CSV(s)")

    # LIAR (tsv, no header, 6-way label needs binarizing). Default to test.tsv only
    # (held-out split) to avoid measuring the model on data it may have already
    # been trained on via train.tsv/valid.tsv.
    p.add_argument("--liar-tsv", nargs="+", default=["data/liar/test.tsv"],
                    help="LIAR tsv split(s) to use as native data (default: test.tsv only, "
                         "to avoid train-set leakage into this evaluation)")

    p.add_argument("--translated-fake-csv", default="data/malaysian/malaysian_fake_en.csv",
                   help="Translated Malaysian fake news CSV (matches retrain_with_malaysian.py)")
    p.add_argument("--translated-real-csv", default="data/malaysian/malaysian_real.csv",
                   help="Translated+native Malaysian real news CSV (will filter to was_translated==True)")
    p.add_argument("--text-col", default="text", help="Preferred text column name for generic CSVs (default: text)")
    p.add_argument("--model-path", default="best_model.pt", help="Path to trained model weights")
    p.add_argument("--feature-stats", default="feature_stats_malaysian.npz",
                   help="Path to feature_mean/feature_std npz saved during training")
    p.add_argument("--max-length", type=int, default=128, help="Tokenizer max length (default: 128, matches training)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sample-size", type=int, default=None,
                   help="Optional cap per NATIVE source file (not per combined slice), for a quick smoke test")

    # ── replay-buffer manifest, to prevent train/eval overlap on the NATIVE side ──
    p.add_argument("--replay-manifest", default="replay_buffer_manifest.csv",
                    help="CSV written by retrain_with_malaysian.py listing exactly which native "
                         "rows (text + source) were sampled into the replay buffer and trained on. "
                         "Any matching rows are dropped from the native-fake/native-real eval "
                         "slices below, so this script never scores the model on data it was just "
                         "trained on. If not found, the script still runs but prints a warning "
                         "since results may be optimistically biased.")
    p.add_argument("--skip-replay-filter", action="store_true",
                    help="Skip replay-buffer filtering even if a manifest is found. Not recommended "
                         "— only use this for debugging.")

    # ── NEW: Malaysian val manifest, to prevent train/eval overlap on the TRANSLATED side ──
    p.add_argument("--malaysian-val-manifest", default="malaysian_val_manifest.csv",
                    help="CSV written by retrain_with_malaysian.py listing the held-out "
                         "val split (text, label, is_replay). Restricts the "
                         "translated-fake/translated-real eval slices below to rows this "
                         "run never trained on — mirrors --replay-manifest's fix for the "
                         "native slices, closing the same overlap gap on the other side "
                         "of the native-vs-translated comparison. If not found, the script "
                         "still runs but prints a warning since translated-side accuracy "
                         "may be inflated by train/eval overlap.")
    p.add_argument("--skip-val-filter", action="store_true",
                    help="Skip Malaysian val-manifest filtering even if found. Not "
                         "recommended — only use this for debugging.")
    return p.parse_args()


TEXT_COL_CANDIDATES = ["text", "content", "body", "article_text", "full_text", "full_article_text"]
TITLE_COL_CANDIDATES = ["title", "headline"]


def autodetect_text_col(df, prefer_title_fallback=True):
    """Returns a column name to use as text, or None. Prefers body-text
    columns over title; falls back to title if that's all there is
    (e.g. stock FakeNewsNet CSVs, which only ship title/news_url)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in TEXT_COL_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand], False  # (col, is_title_only)
    if prefer_title_fallback:
        for cand in TITLE_COL_CANDIDATES:
            if cand in lower_map:
                return lower_map[cand], True
    return None, False


def load_generic_csv(path, true_label, source_name, sample_size=None):
    p = Path(path)
    if not p.exists():
        print(f"  ⚠️  {path} not found — skipping ({source_name}).")
        return None
    df = pd.read_csv(p)
    col, is_title_only = autodetect_text_col(df)
    if col is None:
        print(f"  ⚠️  No usable text/title column found in {path} "
              f"(columns: {list(df.columns)}) — skipping ({source_name}).")
        return None
    if is_title_only:
        print(f"  ⚠️  {path} only has a title column ('{col}'), no full article body. "
              f"Using title text — this slice will test on much shorter text than the "
              f"model was trained on, which limits how comparable this result is. "
              f"({source_name})")
    out = df[[col]].rename(columns={col: "text"})
    out["text"] = out["text"].astype(str)
    out = out[out["text"].str.strip() != ""].reset_index(drop=True)
    if sample_size:
        out = out.sample(n=min(sample_size, len(out)), random_state=42).reset_index(drop=True)
    out["true_label"] = true_label
    out["_source"] = source_name
    return out


def load_isot(fake_path, real_path, sample_size=None):
    frames = []
    for path, label, name in [(fake_path, 0, "isot_fake"), (real_path, 1, "isot_real")]:
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  {path} not found — skipping ({name}).")
            continue
        df = pd.read_csv(p)
        if "text" in df.columns:
            text_col = "text"
        else:
            col, _ = autodetect_text_col(df)
            if col is None:
                print(f"  ⚠️  No usable text column in {path} (columns: {list(df.columns)}) — skipping.")
                continue
            text_col = col
        # ISOT commonly has separate title + text; combine for fuller context if both exist.
        if "title" in df.columns and text_col != "title":
            combined = (df["title"].fillna("").astype(str) + ". " + df[text_col].fillna("").astype(str)).str.strip()
        else:
            combined = df[text_col].fillna("").astype(str)
        out = pd.DataFrame({"text": combined})
        out = out[out["text"].str.strip() != ""].reset_index(drop=True)
        if sample_size:
            out = out.sample(n=min(sample_size, len(out)), random_state=42).reset_index(drop=True)
        out["true_label"] = label
        out["_source"] = name
        frames.append(out)
    return frames


def load_liar(tsv_paths, sample_size=None):
    """LIAR tsv has no header. Standard 14-column layout:
    id, label, statement, subject, speaker, speaker_job, state, party,
    barely_true_ct, false_ct, half_true_ct, mostly_true_ct, pants_fire_ct, context
    Binarized: {pants-fire, false, barely-true} -> fake(0); {half-true, mostly-true, true} -> real(1)."""
    FAKE_LABELS = {"pants-fire", "false", "barely-true"}
    REAL_LABELS = {"half-true", "mostly-true", "true"}
    frames = []
    for path in tsv_paths:
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  {path} not found — skipping (liar).")
            continue
        df = pd.read_csv(p, sep="\t", header=None)
        if df.shape[1] < 3:
            print(f"  ⚠️  {path} doesn't look like standard LIAR tsv "
                  f"({df.shape[1]} columns, expected 14) — skipping.")
            continue
        df = df.rename(columns={1: "raw_label", 2: "text"})
        df["raw_label"] = df["raw_label"].astype(str).str.strip().str.lower()
        df = df[df["raw_label"].isin(FAKE_LABELS | REAL_LABELS)]
        df["true_label"] = df["raw_label"].apply(lambda x: 0 if x in FAKE_LABELS else 1)
        out = df[["text", "true_label"]].copy()
        out["text"] = out["text"].astype(str)
        out = out[out["text"].str.strip() != ""].reset_index(drop=True)
        if sample_size:
            out = out.sample(n=min(sample_size, len(out)), random_state=42).reset_index(drop=True)
        out["_source"] = f"liar:{p.name}"
        frames.append(out)
    return frames


# ── replay-buffer manifest loading + filtering (NATIVE side, existing) ─────
def load_replay_manifest(path):
    """Loads the manifest of native rows that went into the replay buffer.

    Expected format: a CSV with at least a 'text' column (the raw text of
    each sampled row, before clean_text() is applied — this function
    normalizes it the same way clean_text() does, so exact preprocessing
    differences don't matter). An optional 'source' column is used only
    for reporting.

    Returns a set of clean_text()-normalized strings, or None if the
    manifest doesn't exist / can't be read.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  ⚠️  Could not read replay manifest at {path}: {e}")
        return None
    if "text" not in df.columns:
        print(f"  ⚠️  Replay manifest at {path} has no 'text' column "
              f"(columns: {list(df.columns)}) — cannot filter, ignoring manifest.")
        return None
    normalized = df["text"].astype(str).map(clean_text)
    texts = set(normalized[normalized.str.strip() != ""])
    print(f"  📋 Loaded replay-buffer manifest: {len(texts)} unique native rows "
          f"that were trained on (will be excluded from native eval slices).")
    return texts


def filter_out_replay_rows(df, replay_texts, slice_name):
    """Drops any row from df whose clean_text() matches something in the
    replay-buffer manifest, so eval never scores the model on rows it was
    just trained on."""
    if df is None or len(df) == 0 or not replay_texts:
        return df
    normalized = df["text"].astype(str).map(clean_text)
    mask = ~normalized.isin(replay_texts)
    removed = (~mask).sum()
    filtered = df[mask].reset_index(drop=True)
    if removed:
        print(f"  🧹 {slice_name}: removed {removed} row(s) also present in the "
              f"replay-buffer manifest (train/eval overlap) — {len(df)} → {len(filtered)}")
    return filtered


# ── NEW: Malaysian val-manifest loading + restricting (TRANSLATED side) ────
def load_malaysian_val_manifest(path):
    """Loads the held-out Malaysian val split written by retrain_with_malaysian.py.

    Unlike the replay manifest (which lists TRAINED-on rows to exclude),
    this lists HELD-OUT rows to restrict to — the translated-fake/
    translated-real slices should only be scored on text from this set,
    not on the full raw CSVs (which include rows the model trained on).

    Returns a set of clean_text()-normalized strings, or None if the
    manifest doesn't exist / can't be read.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  ⚠️  Could not read Malaysian val manifest at {path}: {e}")
        return None
    if "text" not in df.columns:
        print(f"  ⚠️  Malaysian val manifest at {path} has no 'text' column "
              f"(columns: {list(df.columns)}) — cannot filter, ignoring manifest.")
        return None
    normalized = df["text"].astype(str).map(clean_text)
    texts = set(normalized[normalized.str.strip() != ""])
    print(f"  📋 Loaded Malaysian val manifest: {len(texts)} unique held-out rows "
          f"(translated-fake/translated-real slices will be restricted to these).")
    return texts


def restrict_to_val_rows(df, val_texts, slice_name):
    """Keeps only rows whose clean_text() appears in the held-out val manifest,
    so translated-fake/translated-real eval never scores the model on rows
    it was trained on — same purpose as filter_out_replay_rows(), but this
    KEEPS matches (val set) instead of DROPPING matches (trained-on set)."""
    if df is None or len(df) == 0 or not val_texts:
        return df
    normalized = df["text"].astype(str).map(clean_text)
    mask = normalized.isin(val_texts)
    kept = mask.sum()
    filtered = df[mask].reset_index(drop=True)
    print(f"  🎯 {slice_name}: restricted to held-out val rows — {len(df)} → {kept}")
    return filtered


# ── PART 1: raw-data label leakage check ────────────────────────────────────
def check_raw_data_leakage(args):
    print("=" * 70)
    print("PART 1 — Raw-data label leakage check (was_translated vs label)")
    print("=" * 70)

    rows = []

    def load_and_tag(path, label, default_translated):
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  {path} not found — skipping this slice in Part 1.")
            return None
        df = pd.read_csv(p)
        if "was_translated" not in df.columns:
            print(f"  ⚠️  {path} has no 'was_translated' column — "
                  f"assuming was_translated={default_translated} for all rows "
                  f"(per project history). Verify this assumption.")
            df["was_translated"] = default_translated
        df["was_translated"] = df["was_translated"].fillna(default_translated).astype(bool)
        df["label"] = label
        return df[["was_translated", "label"]]

    # Fake news: per the original finding, every fake example was translated.
    fake_df = load_and_tag(args.translated_fake_csv, label=0, default_translated=True)
    # Real news: this is the file the fix was applied to — should now be mixed.
    real_df = load_and_tag(args.translated_real_csv, label=1, default_translated=False)

    for df in (fake_df, real_df):
        if df is not None:
            rows.append(df)

    if not rows:
        print("  ❌ No data available for Part 1 — check your CSV paths.")
        return None

    combined = pd.concat(rows, ignore_index=True)
    crosstab = pd.crosstab(combined["label"], combined["was_translated"])
    crosstab.index = ["Fake (0)", "Real (1)"]
    print("\nCrosstab — rows=label, cols=was_translated:")
    print(crosstab)

    # Phi coefficient: correlation between two binary variables.
    # |phi| close to 1.0 means was_translated ~ perfectly predicts label
    # (i.e. the shortcut is fully present at the data level).
    # |phi| close to 0.0 means no data-level leakage.
    try:
        a = crosstab.loc["Fake (0)", False] if False in crosstab.columns else 0
        b = crosstab.loc["Fake (0)", True] if True in crosstab.columns else 0
        c = crosstab.loc["Real (1)", False] if False in crosstab.columns else 0
        d = crosstab.loc["Real (1)", True] if True in crosstab.columns else 0
        n = a + b + c + d
        numerator = (a * d) - (b * c)
        denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        phi = numerator / denom if denom != 0 else float("nan")
        print(f"\nPhi coefficient (label vs was_translated): {phi:.3f}")
        if abs(phi) > 0.7:
            print("  ❌ Still strong data-level leakage — the fix likely hasn't "
                  "broken the correlation enough.")
        elif abs(phi) > 0.3:
            print("  ⚠️  Moderate leakage remains — worth collecting more "
                  "translated real news to balance further.")
        else:
            print("  ✅ Low correlation — was_translated no longer strongly "
                  "predicts label at the data level.")
    except Exception as e:
        print(f"  Could not compute phi coefficient: {e}")

    return crosstab


# ── PART 2: retrained-model behavior check ──────────────────────────────────
def load_model_and_tokenizer(args, device):
    if not Path(args.model_path).exists():
        print(f"❌ Model not found at {args.model_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.feature_stats).exists():
        print(f"❌ Feature stats not found at {args.feature_stats}", file=sys.stderr)
        sys.exit(1)

    stats = np.load(args.feature_stats)
    feature_mean, feature_std = stats["mean"], stats["std"]

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = HybridModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    return model, tokenizer, feature_mean, feature_std


def predict_slice(df, text_col, model, tokenizer, feature_mean, feature_std, device, max_length, batch_size):
    """Returns predicted labels (0=fake, 1=real) for every row in df."""
    texts = df[text_col].astype(str).tolist()
    preds = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            cleaned = [clean_text(t) for t in batch_texts]

            encoding = tokenizer(
                cleaned, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            feats = np.array([extract_features(t) for t in cleaned])
            feats = (feats - feature_mean) / (feature_std + 1e-8)
            extra_features = torch.tensor(feats, dtype=torch.float32).to(device)

            outputs = model(input_ids, attention_mask, extra_features)
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(batch_preds.tolist())

    return np.array(preds)


def check_model_behavior(args):
    print("\n" + "=" * 70)
    print("PART 2 — Retrained-model behavior check (native vs translated)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, tokenizer, feature_mean, feature_std = load_model_and_tokenizer(args, device)

    # ── load replay-buffer manifest so we can exclude trained-on rows
    # from the NATIVE slices ────────────────────────────────────────────
    replay_texts = None
    if args.skip_replay_filter:
        print("  ⚠️  --skip-replay-filter set: NOT filtering native slices against the "
              "replay-buffer manifest. Results may include train/eval overlap.")
    else:
        replay_texts = load_replay_manifest(args.replay_manifest)
        if replay_texts is None:
            print(f"  ⚠️  No replay-buffer manifest found at '{args.replay_manifest}'. "
                  f"If retrain_with_malaysian.py's native replay buffer sampled from the "
                  f"same ISOT/FakeNewsNet/LIAR files used below, native accuracy here may be "
                  f"inflated by train/eval overlap (~500 fake + ~500 real rows were trained on). "
                  f"Treat this run's fake/real-class gaps as provisional until the "
                  f"retrain script is patched to write this manifest (see file header).")

    # ── NEW: load Malaysian val manifest so translated slices are also
    # restricted to genuinely held-out rows (same fix as the replay
    # manifest, applied to the other side of the native-vs-translated
    # comparison) ────────────────────────────────────────────────────────
    malaysian_val_texts = None
    if args.skip_val_filter:
        print("  ⚠️  --skip-val-filter set: NOT restricting translated slices to the "
              "held-out Malaysian val split. Results may include train/eval overlap.")
    else:
        malaysian_val_texts = load_malaysian_val_manifest(args.malaysian_val_manifest)
        if malaysian_val_texts is None:
            print(f"  ⚠️  No Malaysian val manifest found at '{args.malaysian_val_manifest}'. "
                  f"The translated-fake/translated-real slices below will evaluate on the "
                  f"FULL raw CSVs, which include rows retrain_with_malaysian.py trained on — "
                  f"translated accuracy may be inflated by train/eval overlap. Treat this "
                  f"run's gaps as provisional until the retrain script is patched to write "
                  f"this manifest (see companion NOTE at bottom of this file).")

    def load_translated_slice(path, true_label, filter_translated=None):
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  {path} not found — skipping this slice.")
            return None
        df = pd.read_csv(p)
        if args.text_col not in df.columns:
            print(f"  ⚠️  '{args.text_col}' not in {path} (columns: {list(df.columns)}) — skipping.")
            return None
        if filter_translated is not None and "was_translated" in df.columns:
            df = df[df["was_translated"].fillna(False).astype(bool) == filter_translated]
        df = df[df[args.text_col].notna() & (df[args.text_col].astype(str).str.strip() != "")]
        if args.sample_size:
            df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        df = df.reset_index(drop=True)
        out = pd.DataFrame({"text": df[args.text_col].astype(str), "true_label": true_label})
        return out

    # ── Native fake: ISOT fake + FakeNewsNet fake (gossipcop/politifact) + LIAR fake-labeled statements
    print("\nBuilding native FAKE slice from ISOT + FakeNewsNet + LIAR...")
    native_fake_frames = []
    isot_frames = load_isot(args.isot_fake, args.isot_real, sample_size=args.sample_size)
    native_fake_frames += [f for f in isot_frames if len(f) and f["true_label"].iloc[0] == 0]
    native_real_frames_isot = [f for f in isot_frames if len(f) and f["true_label"].iloc[0] == 1]

    for path in args.fakenewsnet_fake:
        d = load_generic_csv(path, true_label=0, source_name=f"fakenewsnet:{Path(path).name}",
                              sample_size=args.sample_size)
        if d is not None:
            native_fake_frames.append(d[["text", "true_label"]])

    liar_frames = load_liar(args.liar_tsv, sample_size=args.sample_size)
    native_fake_frames += [f[["text", "true_label"]] for f in liar_frames if len(f) and (f["true_label"] == 0).any()]

    # ── Native real: ISOT real + FakeNewsNet real + LIAR real-labeled statements
    print("Building native REAL slice from ISOT + FakeNewsNet + LIAR...")
    native_real_frames = list(native_real_frames_isot)
    for path in args.fakenewsnet_real:
        d = load_generic_csv(path, true_label=1, source_name=f"fakenewsnet:{Path(path).name}",
                              sample_size=args.sample_size)
        if d is not None:
            native_real_frames.append(d[["text", "true_label"]])
    native_real_frames += [f[["text", "true_label"]] for f in liar_frames if len(f) and (f["true_label"] == 1).any()]

    def combine(frames, label):
        # LIAR frames may contain both labels mixed in one df; filter per-target-label here.
        parts = [f[f["true_label"] == label] for f in frames if len(f)]
        if not parts:
            return None
        return pd.concat(parts, ignore_index=True)

    native_fake_df = combine(native_fake_frames, 0)
    native_real_df = combine(native_real_frames, 1)

    # ── drop any rows the replay buffer trained on (NATIVE side) ──────────
    native_fake_df = filter_out_replay_rows(native_fake_df, replay_texts, "native fake")
    native_real_df = filter_out_replay_rows(native_real_df, replay_texts, "native real")

    # ── NEW: restrict translated slices to the held-out Malaysian val
    # split (TRANSLATED side) ──────────────────────────────────────────────
    translated_fake_df = load_translated_slice(args.translated_fake_csv, true_label=0)
    translated_fake_df = restrict_to_val_rows(translated_fake_df, malaysian_val_texts, "translated fake")

    translated_real_df = load_translated_slice(args.translated_real_csv, true_label=1, filter_translated=True)
    translated_real_df = restrict_to_val_rows(translated_real_df, malaysian_val_texts, "translated real")

    slices = {
        "native fake":      native_fake_df,
        "translated fake":  translated_fake_df,
        "native real":      native_real_df,
        "translated real":  translated_real_df,
    }

    results = {}
    for name, df in slices.items():
        if df is None or len(df) == 0:
            print(f"\n{name}: no data, skipped")
            continue
        preds = predict_slice(df, "text", model, tokenizer, feature_mean, feature_std,
                               device, args.max_length, args.batch_size)
        acc = (preds == df["true_label"].values).mean()
        results[name] = acc
        print(f"\n{name}: n={len(df)}, accuracy={acc:.4f} ({acc*100:.2f}%)")

    print("\n" + "-" * 70)
    if "native fake" in results and "translated fake" in results:
        gap_fake = abs(results["native fake"] - results["translated fake"])
        print(f"Fake-class gap (native vs translated): {gap_fake*100:.2f}pp "
              f"(was ~19.78pp before the fix: 80.00% vs 99.78%; "
              f"47.74pp after v1: 51.57% vs 99.32%)")
    if "native real" in results and "translated real" in results:
        gap_real = abs(results["native real"] - results["translated real"])
        print(f"Real-class gap (native vs translated): {gap_real*100:.2f}pp")

    if replay_texts is None and not args.skip_replay_filter:
        print("\n⚠️  Reminder: no replay-buffer manifest was found, so the native slices above "
              "may still include rows the model was trained on. These numbers should be "
              "treated as an upper bound on native accuracy, not a clean read.")
    if malaysian_val_texts is None and not args.skip_val_filter:
        print("\n⚠️  Reminder: no Malaysian val manifest was found, so the translated slices above "
              "may still include rows the model was trained on. These numbers should be "
              "treated as an upper bound on translated accuracy, not a clean read.")
    if translated_real_df is not None and 0 < len(translated_real_df) < 20:
        print(f"\n⚠️  Note: translated real n={len(translated_real_df)} after restricting to the "
              f"held-out val split — this is a small sample, so its accuracy number is noisy/"
              f"low-confidence until translate_real_to_english.py grows the translated-real "
              f"pool further.")

    print("-" * 70)
    print("Interpretation: gaps should shrink toward single digits if the "
          "translation-fingerprint shortcut has been broken. A gap still in "
          "the teens-plus range means the model likely still leans on "
          "translated-ness as a proxy for the label.")

    return results


def main():
    args = parse_args()
    check_raw_data_leakage(args)
    check_model_behavior(args)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────
# NOTE — companion patch needed in retrain_with_malaysian.py
# ─────────────────────────────────────────────────────────────────────────
# 1) NATIVE-side manifest (already applied in earlier work): right after
#    the native replay buffer is sampled, write replay_buffer_manifest.csv
#    listing exactly which native rows were trained on, so this script can
#    drop them from the native-fake/native-real eval slices.
#
# 2) TRANSLATED-side manifest (this patch): right after
#    retrain_with_malaysian.py's own train_test_split() call, add:
#
#     train_df, val_df = train_test_split(
#         my_df, test_size=0.15, random_state=42, stratify=strat_key)
#
#     val_df[["text", "label", "is_replay"]].to_csv("malaysian_val_manifest.csv", index=False)
#     print(f"💾 Saved malaysian_val_manifest.csv ({len(val_df)} rows) — the held-out val "
#           f"split (Malaysian + replay), so ablate_was_translated_feature.py can restrict "
#           f"its translated-fake/translated-real eval slices to genuinely held-out rows.")
#
# This must go in retrain_with_malaysian.py itself (NOT in ablate_boilerplate_feature.py
# or any other ablation script) — val_df here is the one with the correct is_replay
# column and the correct pool composition (post-downsampling, post-replay-mixing).
# A val_df built independently inside another script, from a different-sized pool,
# will not describe what the deployed model actually trained on.