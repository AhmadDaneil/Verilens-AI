"""
retrain_with_malaysian.py  (patched v4 — Malaysian val-split manifest export)
--------------------------
Fine-tunes the existing best_model.pt on Malaysian news data.
Does NOT retrain from scratch — adapts the existing model so it
keeps its English fake-news knowledge but gains Malaysian context.

Prerequisites:
    - best_model.pt exists (your current trained model)
    - data/malaysian/malaysian_real.csv exists
    - data/malaysian/malaysian_fake_en.csv exists
    - data/isot/Fake.csv, data/isot/True.csv exist
    - data/fakenewsnet/{gossipcop,politifact}_{fake,real}.csv exist
    - data/liar/train.tsv, data/liar/valid.tsv exist
    - All original training dependencies installed

Usage:
    python retrain_with_malaysian.py

CHANGES vs v1 (patched):
    1. Added early sanity checks after loading + after length filtering so
       an empty class fails with a clear message instead of a cryptic
       sklearn "resulting train set will be empty" error deep in
       train_test_split.
    2. Replaced the hard n = min(1500, len(real), len(fake)) downsample
       with a capped-ratio approach (MAX_IMBALANCE_RATIO) so you don't
       throw away thousands of valid fake-class rows just because the
       real-class collector ran short — the existing WeightedRandomSampler
       already handles imbalance during training.

CHANGES vs v2 — NATIVE REPLAY BUFFER:
    Root cause found by ablate_was_translated_feature.py after v1: this
    script trained ONLY on Malaysian data (100% translated fake, mixed
    native/translated real). With zero native-style "fake" examples in the
    whole fine-tuning run, the model had no signal to avoid using
    translated-ness as a fake-detection shortcut, and also had no reason
    to retain whatever native-English fake-detection ability best_model.pt
    already had before this Malaysian work started. Result: native fake
    accuracy collapsed to 51.57% (near coin-flip) while translated fake
    stayed at 99.32% — a 47.74pp gap, worse than before this fix.

    Fix: mix a REPLAY BUFFER of native English fake + real examples
    (ISOT + FakeNewsNet + LIAR) in alongside the Malaysian data, so every
    epoch the model sees native-style examples of BOTH classes, not just
    translated fake / mixed real.

    IMPORTANT — train/eval separation: LIAR replay only draws from
    train.tsv + valid.tsv, NEVER test.tsv. ablate_was_translated_feature.py
    defaults to test.tsv as its held-out native-fake check; if this script
    trained on those same rows, that ablation would silently start
    measuring performance on data the model had already seen, which would
    hide the exact shortcut it's meant to catch. Keep this separation if
    you adjust either script.

    FakeNewsNet note: the standard release only ships title/news_url, no
    full article body. Those rows are shorter than the rest of the replay
    buffer (and than Malaysian full-article text) — this is a genuine
    limitation of that dataset, not a bug here. ISOT (which has full
    title+text) is preferred as the larger share of the replay buffer for
    that reason.

CHANGES vs v3 — REPLAY BUFFER MANIFEST:
    Even with the train/eval separation above, ISOT and FakeNewsNet had no
    such split: this script samples 500 fake + 500 real rows straight out
    of the SAME full ISOT/FakeNewsNet files that
    ablate_was_translated_feature.py uses (by default) as its "native"
    eval slices. That meant part of the ablation's native-accuracy numbers
    could be measuring rows the model had just been trained on here,
    inflating native accuracy and understating the true native/translated
    gap — silently, since nothing recorded which rows were sampled.

    Fix: build_native_replay_buffer() tags every candidate row with a
    `source` column, and after sampling, this script writes
    replay_buffer_manifest.csv (text, source, label) listing EXACTLY the
    500+500 rows that went into training. ablate_was_translated_feature.py
    (patched alongside this) loads that manifest and drops any matching
    rows from its native-fake / native-real eval slices before scoring, so
    the two scripts can no longer silently overlap on ISOT/FakeNewsNet
    rows the way they could before. LIAR's existing train/valid-vs-test
    split is untouched — it already had no overlap.

CHANGES vs v4 — MALAYSIAN VAL-SPLIT MANIFEST (this version):
    The replay-buffer manifest above fixed train/eval overlap on the
    NATIVE side of ablate_was_translated_feature.py's native-vs-translated
    comparison, but the TRANSLATED side had the identical problem and was
    never fixed: this script trains on most of malaysian_fake_en.csv
    (capped by MAX_IMBALANCE_RATIO, but still hundreds of rows) and most
    of malaysian_real.csv, while ablate_was_translated_feature.py's
    "translated fake" / "translated real" slices evaluated directly
    against those same raw CSVs with no exclusion at all. That meant the
    translated-side accuracy numbers were likely inflated by direct
    memorization, while the native side (after the v3 fix) was a
    genuinely held-out check — an unfair comparison between a
    contaminated number and a clean one.

    Fix: right after this script's own train_test_split() call, it now
    writes malaysian_val_manifest.csv (text, label, is_replay) — the
    actual held-out validation rows from THIS run's real pool composition
    (post-downsampling, post-replay-mixing, correct is_replay tagging).
    ablate_was_translated_feature.py (patched alongside this) loads that
    manifest and RESTRICTS its translated-fake / translated-real slices to
    only those held-out rows, instead of scoring against the full raw
    CSVs. This must live here rather than in any other script (e.g. an
    ablation script's own independent train_test_split) — only this
    script's val_df reflects what the deployed model actually trained on.

Usage:
    python retrain_with_malaysian.py
"""

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ── Replay buffer config ───────────────────────────────────────────────────
# How many native examples per class to mix into training. Kept modest
# relative to the Malaysian data so this run still primarily adapts to
# Malaysian style, while giving the model enough native signal per class
# to avoid re-learning translated-ness as a shortcut. Tune upward if the
# gap in ablate_was_translated_feature.py doesn't close enough; tune down
# if Malaysian-specific performance regresses.
REPLAY_FAKE_PER_CLASS = 500
REPLAY_REAL_PER_CLASS = 500

# Where the manifest of sampled replay-buffer rows gets written. Must match
# the --replay-manifest default (or override) in ablate_was_translated_feature.py.
REPLAY_MANIFEST_PATH = "replay_buffer_manifest.csv"

# NEW: where the held-out Malaysian val split gets written. Must match
# the --malaysian-val-manifest default (or override) in
# ablate_was_translated_feature.py.
MALAYSIAN_VAL_MANIFEST_PATH = "malaysian_val_manifest.csv"

ISOT_FAKE_PATH = "data/isot/Fake.csv"
ISOT_REAL_PATH = "data/isot/True.csv"
FAKENEWSNET_FAKE_PATHS = [
    "data/fakenewsnet/gossipcop_fake.csv",
    "data/fakenewsnet/politifact_fake.csv",
]
FAKENEWSNET_REAL_PATHS = [
    "data/fakenewsnet/gossipcop_real.csv",
    "data/fakenewsnet/politifact_real.csv",
]
# NOTE: train.tsv + valid.tsv only — test.tsv is reserved for
# ablate_was_translated_feature.py's held-out native check. Do not add
# test.tsv here without also changing what that script treats as held-out.
LIAR_TSV_PATHS = ["data/liar/train.tsv", "data/liar/valid.tsv"]


# ── Reproduce original feature extraction exactly ─────────────────────────────
# CHANGES vs v5 — BOILERPLATE STRIPPING + LENGTH-SHORTCUT MITIGATION:
#     audit_malaysian_data.py, run against the actual malaysian_real.csv /
#     malaysian_fake_en.csv, confirmed two dataset-level shortcuts:
#       1. Length: REAL articles average 242 words vs FAKE's 87 (~2.8x) --
#          independently confirmed by length_stress_test.py to swing
#          fake_score from 0.68 (20 words) to 0.09 (363 words) on the SAME
#          real article, with diagnose_features.py ruling out the
#          engineered features as the cause. See HybridDataset below for
#          the mitigation (train-time random truncation of REAL examples).
#       2. Boilerplate: 57.0% of FAKE-class text starts with "According
#          to," vs 0.8% of REAL; 17.2% contains a translated "TRUTH:"
#          marker vs 0%; 8.7% contains a "SOURCE:"/"SUMBER:" marker vs 0%.
#          These are Sebenarnya-scraper/translation template artifacts, not
#          truthfulness signal -- stripped below in clean_text() so the
#          model can't use them as a giveaway.
#     IMPORTANT: features.py (production inference, imported by app.py) has
#     the SAME boilerplate patterns added to ITS clean_text() for
#     train/serve parity. If you change BOILERPLATE_PATTERNS here, change
#     it there too -- this exact kind of divergence is what caused the
#     original feature-vector train/serve mismatch this project just fixed.
#     Also found: 5 rows in malaysian_fake_en.csv are Sebenarnya posts that
#     actually CONFIRM a claim as true ("adalah benar"/"disahkan benar"),
#     not debunk it -- genuinely mislabeled. Removed in
#     malaysian_fake_en_cleaned.csv; point FAKE_CSV loading at that file
#     (or overwrite the original after reviewing) before retraining.
BOILERPLATE_PATTERNS = [
    r"^according to,?\s*",       # leading "According to," / "According to"
    r"\btruth:\s*",              # translated "TRUTH:" marker (Sebenarnya template)
    r"\b(?:source|sumber)\s*:\s*", # "SOURCE:" / "SUMBER:" marker
]


def strip_boilerplate(text):
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = strip_boilerplate(text)
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


# ── Reproduce original model architecture exactly ─────────────────────────────
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert     = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bilstm   = nn.LSTM(768, 256, num_layers=1,
                                 batch_first=True, bidirectional=True)
        self.dropout  = nn.Dropout(0.3)
        self.fc1      = nn.Linear(1024 + 6, 512)
        self.batchnorm= nn.BatchNorm1d(512)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs       = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)
        seq           = outputs.last_hidden_state
        lstm_out, _   = self.bilstm(seq)
        mean_pool     = torch.mean(lstm_out, dim=1)
        max_pool, _   = torch.max(lstm_out, dim=1)
        pooled        = torch.cat((mean_pool, max_pool), dim=1)
        combined      = torch.cat((pooled, extra_features), dim=1)
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc2(x)


# ── Dataset ───────────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Length-shortcut mitigation (see v5 note above). Truncation range chosen to
# overlap the actual FAKE-class distribution measured by audit_malaysian_data.py
# (min 8, median 65, mean 87 words) so truncated REAL examples land in the
# same length neighborhood as real fake-class examples, not just "shorter".
#
# v6 — TUNED DOWN: v5's settings (prob=0.5, min=20 words) fixed the pure
# length-swing artifact (length_stress_test.py: 60pp range -> 8.9pp range)
# but the retrained model then confidently misclassified a genuine,
# investigative-style real article as FAKE at every length -- a new,
# different failure. Root cause is most likely that real-class data is
# already scarce (236 unique Malaysian real rows total), and aggressively
# truncating half of them down to as few as 20 words further shrinks the
# effective full-length training signal for an already-narrow class,
# without adding genuine topical diversity (it's still the same 236 base
# articles, just shorter slices of them). Tuning down here is a partial,
# quick mitigation; the more durable fix is collecting more real Malaysian
# data (see collect_real_incremental.py / scrape_malaysian_news.py),
# especially investigative/political-scandal-style pieces, not just
# announcement-style wire content -- do that before pushing augmentation
# strength back up.
REAL_TRUNCATE_PROB      = 0.25   # was 0.5
REAL_TRUNCATE_MIN_WORDS = 60     # was 20 -- keep enough content to still be informative
REAL_TRUNCATE_MAX_WORDS = 180    # unchanged


class HybridDataset(Dataset):
    def __init__(self, texts, labels, feature_mean, feature_std,
                 augment=False, seed=42):
        self.texts        = texts
        self.labels        = labels
        self.feature_mean = feature_mean
        self.feature_std  = feature_std
        # augment=True only for the TRAIN split -- val_loader must see
        # untouched text so its accuracy/F1 numbers reflect real performance,
        # not the augmentation. This mirrors the same discipline already
        # used elsewhere in this script for train/eval separation
        # (replay-buffer manifest, malaysian_val_manifest.csv).
        self.augment = augment
        self._rng    = random.Random(seed)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = clean_text(str(self.texts[idx]))
        label = self.labels[idx]

        # ── Length-shortcut mitigation (train-time only) ───────────────────
        # audit_malaysian_data.py: REAL averages 242 words vs FAKE's 87
        # (~2.8x). length_stress_test.py confirmed this alone can swing
        # fake_score from 0.68 to 0.09 on the SAME article, with
        # diagnose_features.py ruling out the engineered features as the
        # cause -- i.e. DistilBERT+BiLSTM itself was likely learning
        # "short == fake". Randomly truncating a fraction of REAL examples
        # into the FAKE-class length range forces the model to keep
        # recognizing genuinely-real content even when it's short, instead
        # of using raw length as a proxy label.
        if self.augment and label == 1 and self._rng.random() < REAL_TRUNCATE_PROB:
            words = text.split()
            if len(words) > REAL_TRUNCATE_MIN_WORDS:
                cutoff = self._rng.randint(
                    REAL_TRUNCATE_MIN_WORDS,
                    min(REAL_TRUNCATE_MAX_WORDS, len(words)),
                )
                text = " ".join(words[:cutoff])

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        features = extract_features(text)
        features = (features - self.feature_mean) / (self.feature_std + 1e-8)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": torch.tensor(features),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ── Native replay buffer loaders ───────────────────────────────────────────
TEXT_COL_CANDIDATES = ["text", "content", "body", "article_text", "full_text", "full_article_text"]
TITLE_COL_CANDIDATES = ["title", "headline"]


def _autodetect_text_col(df):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in TEXT_COL_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand], False
    for cand in TITLE_COL_CANDIDATES:
        if cand in lower_map:
            return lower_map[cand], True
    return None, False


def load_isot_replay():
    frames = []
    for path, label, name in [(ISOT_FAKE_PATH, 0, "isot_fake"), (ISOT_REAL_PATH, 1, "isot_real")]:
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  {path} not found — skipping ({name}) in replay buffer.")
            continue
        df = pd.read_csv(p)
        text_col, _ = _autodetect_text_col(df) if "text" not in df.columns else ("text", False)
        if text_col is None:
            print(f"  ⚠️  No usable text column in {path} — skipping ({name}).")
            continue
        if "title" in df.columns and text_col != "title":
            combined = (df["title"].fillna("").astype(str) + ". " + df[text_col].fillna("").astype(str)).str.strip()
        else:
            combined = df[text_col].fillna("").astype(str)
        out = pd.DataFrame({"text": combined, "label": label})
        out = out[out["text"].str.strip() != ""].reset_index(drop=True)
        # Tag source so sampled rows can be recorded in the replay manifest.
        out["source"] = name
        frames.append(out)
        print(f"  loaded {len(out)} rows from {name}")
    return frames


def load_fakenewsnet_replay(paths, label, name_prefix):
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  {path} not found — skipping in replay buffer.")
            continue
        df = pd.read_csv(p)
        text_col, is_title_only = _autodetect_text_col(df)
        if text_col is None:
            print(f"  ⚠️  No usable text/title column in {path} — skipping.")
            continue
        if is_title_only:
            print(f"  ⚠️  {path} only has title text (no full article body) — "
                  f"using title, shorter than the rest of the replay buffer.")
        out = pd.DataFrame({"text": df[text_col].fillna("").astype(str), "label": label})
        out = out[out["text"].str.strip() != ""].reset_index(drop=True)
        # Tag source so sampled rows can be recorded in the replay manifest.
        out["source"] = f"{name_prefix}:{p.name}"
        frames.append(out)
        print(f"  loaded {len(out)} rows from {name_prefix}:{p.name}")
    return frames


def load_liar_replay(paths):
    """LIAR tsv, no header, 14 columns. Binarizes 6-way truthfulness labels:
    {pants-fire, false, barely-true} -> fake(0); {half-true, mostly-true, true} -> real(1)."""
    FAKE_LABELS = {"pants-fire", "false", "barely-true"}
    REAL_LABELS = {"half-true", "mostly-true", "true"}
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  ⚠️  {path} not found — skipping (liar) in replay buffer.")
            continue
        df = pd.read_csv(p, sep="\t", header=None)
        if df.shape[1] < 3:
            print(f"  ⚠️  {path} doesn't look like standard LIAR tsv — skipping.")
            continue
        df = df.rename(columns={1: "raw_label", 2: "text"})
        df["raw_label"] = df["raw_label"].astype(str).str.strip().str.lower()
        df = df[df["raw_label"].isin(FAKE_LABELS | REAL_LABELS)]
        df["label"] = df["raw_label"].apply(lambda x: 0 if x in FAKE_LABELS else 1)
        out = df[["text", "label"]].copy()
        out["text"] = out["text"].astype(str)
        out = out[out["text"].str.strip() != ""].reset_index(drop=True)
        # Tag source so sampled rows can be recorded in the replay manifest.
        # (LIAR replay only ever draws from train/valid — see LIAR_TSV_PATHS note —
        # so this can never collide with ablate_was_translated_feature.py's
        # test.tsv-only default, but we still tag it for a complete manifest.)
        out["source"] = f"liar:{p.name}"
        frames.append(out)
        print(f"  loaded {len(out)} rows from liar:{p.name}")
    return frames


def build_native_replay_buffer(fake_per_class, real_per_class, seed=42,
                                manifest_path=REPLAY_MANIFEST_PATH):
    print("\n📦 Building native replay buffer (ISOT + FakeNewsNet + LIAR)...")

    print(" fake sources:")
    fake_frames = []
    isot_frames = load_isot_replay()
    fake_frames += [f for f in isot_frames if len(f) and (f["label"] == 0).all()]
    fake_frames += load_fakenewsnet_replay(FAKENEWSNET_FAKE_PATHS, label=0, name_prefix="fakenewsnet")
    liar_frames = load_liar_replay(LIAR_TSV_PATHS)
    fake_frames += [f[f["label"] == 0] for f in liar_frames if len(f) and (f["label"] == 0).any()]

    print(" real sources:")
    real_frames = [f for f in isot_frames if len(f) and (f["label"] == 1).all()]
    real_frames += load_fakenewsnet_replay(FAKENEWSNET_REAL_PATHS, label=1, name_prefix="fakenewsnet")
    real_frames += [f[f["label"] == 1] for f in liar_frames if len(f) and (f["label"] == 1).any()]

    fake_pool = pd.concat(fake_frames, ignore_index=True) if fake_frames else pd.DataFrame(columns=["text", "label", "source"])
    real_pool = pd.concat(real_frames, ignore_index=True) if real_frames else pd.DataFrame(columns=["text", "label", "source"])

    print(f"\n  Native fake pool: {len(fake_pool)} available, sampling {min(fake_per_class, len(fake_pool))}")
    print(f"  Native real pool: {len(real_pool)} available, sampling {min(real_per_class, len(real_pool))}")

    if len(fake_pool) == 0 or len(real_pool) == 0:
        print("  ⚠️  Replay buffer is missing a whole class — check the paths above. "
              "Continuing WITHOUT replay for the missing class (the original shortcut "
              "may not be fixed this run).")

    fake_sample = fake_pool.sample(n=min(fake_per_class, len(fake_pool)), random_state=seed) if len(fake_pool) else fake_pool
    real_sample = real_pool.sample(n=min(real_per_class, len(real_pool)), random_state=seed) if len(real_pool) else real_pool

    # ── Write a manifest of exactly which native rows were sampled into
    # training, so ablate_was_translated_feature.py can exclude these same
    # rows from its "native fake" / "native real" eval slices and avoid
    # silently scoring the model on data it was just trained on. ───────────
    manifest_df = pd.concat(
        [fake_sample[["text", "source", "label"]], real_sample[["text", "source", "label"]]],
        ignore_index=True,
    ) if (len(fake_sample) or len(real_sample)) else pd.DataFrame(columns=["text", "source", "label"])

    try:
        manifest_df.to_csv(manifest_path, index=False)
        print(f"  💾 Saved {manifest_path} ({len(manifest_df)} rows) — "
              f"ablate_was_translated_feature.py will exclude these from native eval.")
    except Exception as e:
        print(f"  ⚠️  Could not write replay-buffer manifest to {manifest_path}: {e}\n"
              f"     ablate_was_translated_feature.py will not be able to exclude these "
              f"rows from its native eval slices — its native accuracy numbers may be "
              f"inflated by train/eval overlap until this is fixed.")

    replay = pd.concat([fake_sample, real_sample], ignore_index=True)
    replay["is_replay"] = True
    return replay


# ── Load Malaysian data ───────────────────────────────────────────────────────
print("\n📂 Loading Malaysian data...")
real_df = pd.read_csv("data/malaysian/malaysian_real.csv")
# v5: use the cleaned file — audit_malaysian_data.py found 5 rows here were
# genuinely mislabeled (Sebenarnya posts confirming a claim as TRUE, not
# debunking it). See malaysian_fake_en_cleaned.csv / mislabeled_rows_review.csv.
# If you haven't generated this file yet, review mislabeled_rows_review.csv
# then either use malaysian_fake_en_cleaned.csv directly or remove those same
# rows from malaysian_fake_en.csv and point back at that filename.
fake_df = pd.read_csv("data/malaysian/malaysian_fake_en_cleaned.csv")  # translated to English

if "extraction_method" in real_df.columns:
    before = len(real_df)
    real_df = real_df[real_df["extraction_method"] == "full_article"].reset_index(drop=True)
    print(f"Filtered to full_article rows only: {before} -> {len(real_df)}")

real_df["label"] = 1
fake_df["label"] = 0
real_df["is_replay"] = False
fake_df["is_replay"] = False

print(f"Raw counts — real: {len(real_df)} | fake: {len(fake_df)}")


# ── GUARD 1: both classes must be non-empty before we do anything else ────────
MIN_PER_CLASS = 30  # below this, fine-tuning is unlikely to help and may hurt

if len(real_df) == 0 or len(fake_df) == 0:
    raise SystemExit(
        f"\n❌ Cannot fine-tune: one class is completely empty "
        f"(real={len(real_df)}, fake={len(fake_df)}).\n"
        f"Run collect_rss.py again and confirm BOTH malaysian_real.csv and "
        f"malaysian_fake_en.csv have rows before retraining."
    )

if min(len(real_df), len(fake_df)) < MIN_PER_CLASS:
    print(
        f"⚠️  Warning: smallest class has only {min(len(real_df), len(fake_df))} "
        f"rows (< {MIN_PER_CLASS}). Fine-tuning will run but results will be "
        f"noisy/unreliable. Consider collecting more before trusting the "
        f"output model."
    )

# ── Balance without wasting data ───────────────────────────────────────────────
# Cap how imbalanced the raw Malaysian pool is allowed to be; the
# WeightedRandomSampler below handles residual imbalance during training.
MAX_IMBALANCE_RATIO = 5
HARD_CAP_PER_CLASS   = 1500

minority_n = min(len(real_df), len(fake_df))
majority_cap = min(HARD_CAP_PER_CLASS, minority_n * MAX_IMBALANCE_RATIO)

if len(real_df) > majority_cap:
    real_df = real_df.sample(n=majority_cap, random_state=42)
elif len(real_df) > HARD_CAP_PER_CLASS:
    real_df = real_df.sample(n=HARD_CAP_PER_CLASS, random_state=42)

if len(fake_df) > majority_cap:
    fake_df = fake_df.sample(n=majority_cap, random_state=42)
elif len(fake_df) > HARD_CAP_PER_CLASS:
    fake_df = fake_df.sample(n=HARD_CAP_PER_CLASS, random_state=42)

# ── Mix in the native replay buffer ─────────────────────────────────────────
replay_df = build_native_replay_buffer(REPLAY_FAKE_PER_CLASS, REPLAY_REAL_PER_CLASS)

my_df = pd.concat(
    [real_df[["text", "label", "is_replay"]], fake_df[["text", "label", "is_replay"]], replay_df],
    ignore_index=True,
)
my_df = my_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Remove very short texts
my_df = my_df[my_df["text"].str.len() > 100].reset_index(drop=True)

print(f"\nCombined pool after adding replay buffer + length filter: {len(my_df)}")
print("By label:")
print(my_df["label"].value_counts())
print("By label x is_replay (Malaysian vs native replay mix per class):")
print(my_df.groupby(["label", "is_replay"]).size())

# ── GUARD 2: the length filter above can also zero out a class ────────────────
label_counts = my_df["label"].value_counts()
if len(my_df) == 0 or label_counts.min() < 2:
    raise SystemExit(
        f"\n❌ After filtering out texts <100 chars, one class has fewer than "
        f"2 rows left ({label_counts.to_dict()}). Cannot stratify-split this."
    )

# ── Train / val split ─────────────────────────────────────────────────────────
# Stratify on label x is_replay so BOTH the Malaysian and native replay
# portions of each class are represented in validation, not just whichever
# happens to survive a plain label-only stratified split.
strat_key = my_df["label"].astype(str) + "_" + my_df["is_replay"].astype(str)
train_df, val_df = train_test_split(
    my_df, test_size=0.15, random_state=42, stratify=strat_key)

# ── NEW (v4): export the held-out val split so ablate_was_translated_feature.py
# can restrict its translated-fake/translated-real eval slices to rows this
# run never trained on — mirrors replay_buffer_manifest.csv's fix for the
# native side, closing the same overlap gap on the other side of the
# native-vs-translated comparison. Must be written HERE (this script's own
# val_df), not inside any other script — this is the only val_df that
# reflects what best_model.pt actually trained on this run. ────────────────
val_df[["text", "label", "is_replay"]].to_csv(MALAYSIAN_VAL_MANIFEST_PATH, index=False)
print(f"💾 Saved {MALAYSIAN_VAL_MANIFEST_PATH} ({len(val_df)} rows) — the held-out val split "
      f"(Malaysian + replay), so ablate_was_translated_feature.py can restrict its "
      f"translated-fake/translated-real eval slices to genuinely held-out rows.")

train_texts  = train_df["text"].tolist()
train_labels = train_df["label"].tolist()
val_texts    = val_df["text"].tolist()
val_labels   = val_df["label"].tolist()

# Compute feature statistics from this training set (Malaysian + replay mix)
all_feats    = np.array([extract_features(clean_text(t)) for t in train_texts])
feature_mean = all_feats.mean(axis=0)
feature_std  = all_feats.std(axis=0)

print(f"\nTrain: {len(train_texts)} | Val: {len(val_texts)}")

np.savez("feature_stats_malaysian.npz", mean=feature_mean, std=feature_std)
print("💾 Saved feature_mean/feature_std → feature_stats_malaysian.npz")

# ── Weighted sampler to handle any class imbalance ────────────────────────────
class_counts  = [train_labels.count(0), train_labels.count(1)]
class_weights = [1.0 / c for c in class_counts]
sample_weights= [class_weights[l] for l in train_labels]
sampler       = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader  = DataLoader(
    HybridDataset(train_texts, train_labels, feature_mean, feature_std, augment=True),
    batch_size=8,
    sampler=sampler,
    drop_last=True,
)
val_loader    = DataLoader(
    HybridDataset(val_texts, val_labels, feature_mean, feature_std, augment=False),
    batch_size=16,
)

# ── Also track val performance split by is_replay, so you can see directly
# whether native (replay) vs Malaysian performance are converging, without
# waiting for a separate ablation run. ────────────────────────────────────
val_is_replay = val_df["is_replay"].tolist()

# ── Load existing model ───────────────────────────────────────────────────────
print("\n📦 Loading existing best_model.pt...")
model = HybridModel().to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
print("✅ Model loaded")

# ── Freeze strategy: freeze all BERT except last 2 transformer layers ─────────
for param in model.bert.parameters():
    param.requires_grad = False

for param in model.bert.transformer.layer[-2:].parameters():
    param.requires_grad = True

for param in model.bilstm.parameters():
    param.requires_grad = True
for param in model.fc1.parameters():
    param.requires_grad = True
for param in model.fc2.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ── Training setup ────────────────────────────────────────────────────────────
criterion   = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, 1.0]).to(device),
    label_smoothing=0.15,
)
optimizer   = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5,
    weight_decay=0.01,
)

EPOCHS      = 5
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids, attention_mask, extra_features)
            preds          = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return acc, f1, all_preds, all_labels


def evaluate_by_replay_split(preds, labels, is_replay_flags):
    """Break down val accuracy by native-replay vs Malaysian, per class, so
    you can see directly in training logs whether the native/translated gap
    is closing — a quick sanity signal ahead of the full ablation script."""
    preds = np.array(preds)
    labels = np.array(labels)
    flags = np.array(is_replay_flags)

    for label_val, label_name in [(0, "Fake"), (1, "Real")]:
        for is_replay, group_name in [(False, "Malaysian"), (True, "Native replay")]:
            mask = (labels == label_val) & (flags == is_replay)
            n = mask.sum()
            if n == 0:
                continue
            acc = (preds[mask] == labels[mask]).mean()
            print(f"    {label_name:5s} / {group_name:14s}: n={n:4d}  acc={acc:.4f}")


# ── Train ─────────────────────────────────────────────────────────────────────
best_val_f1 = 0.0

print(f"\n🚀 Fine-tuning for {EPOCHS} epochs on Malaysian data + native replay buffer...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_features = batch["extra_features"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, extra_features)
        loss    = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    val_acc, val_f1, preds, labels_list = evaluate(val_loader)

    print(f"\nEpoch {epoch+1}")
    print(f"  Loss: {total_loss/len(train_loader):.4f}")
    print(f"  Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    print(classification_report(labels_list, preds,
                                  target_names=["Fake", "Real"]))
    print("  Breakdown by source (native replay vs Malaysian) — "
          "watch for a persistent gap here, same signal as "
          "ablate_was_translated_feature.py:")
    evaluate_by_replay_split(preds, labels_list, val_is_replay)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  💾 New best model saved (F1={val_f1:.4f})")

print(f"\n✅ Fine-tuning complete. Best Val F1: {best_val_f1:.4f}")
print("best_model.pt has been updated with Malaysian news adaptation + native replay.")
print(f"\nNext: run ablate_was_translated_feature.py to confirm the native/translated "
      f"fake-class gap has actually closed. It will automatically load "
      f"{REPLAY_MANIFEST_PATH} (written above) to exclude the exact rows this run "
      f"trained on from its native-fake / native-real eval slices, "
      f"{MALAYSIAN_VAL_MANIFEST_PATH} (also written above) to restrict its translated-fake / "
      f"translated-real eval slices to genuinely held-out rows, and it still uses "
      f"LIAR test.tsv (which this script never trains on) as an additional held-out check.")
print("If it looks good: push best_model.pt to Hugging Face Space.")