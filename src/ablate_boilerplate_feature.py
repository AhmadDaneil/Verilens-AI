"""
ablate_boilerplate_feature.py
------------------------------
Tests whether the fine-tuned model is relying on fake-class boilerplate
markers ("According to,", translated "TRUTH:" tags) instead of real
content understanding.

From audit_malaysian_data.py findings:
    - 57% of fake-class text starts with "According to,"
    - 17% contains a translated "TRUTH:" marker

Mirrors the structure of ablate_word_count_feature.py: evaluate the model
normally, then re-evaluate with the boilerplate phrases stripped out of
the input text (leaving everything else intact), and compare.

A large accuracy/F1 drop after stripping boilerplate confirms the model
is using these markers as a shortcut rather than genuine truthfulness
signal.

VAL SPLIT — now restricted to malaysian_val_manifest.csv (written by
retrain_with_malaysian.py), same fix pattern as
ablate_was_translated_feature.py uses for its translated-fake/translated-
real slices. my_df is still built independently from real_df + fake_df
with no downsampling/replay buffer (since replay-buffer native text never
contains these markers and isn't relevant here), but it is then filtered
down to only rows whose text also appears in the manifest — i.e. rows the
deployed best_model.pt actually held out during its most recent training
run. Rows that were capped out of training (fake_df isn't capped by
MAX_IMBALANCE_RATIO here, unlike retrain_with_malaysian.py) or that landed
in the train split are naturally excluded by this filter, since they won't
match anything in the manifest.

Usage:
    python ablate_boilerplate_feature.py
    python ablate_boilerplate_feature.py --malaysian-val-manifest path/to/malaysian_val_manifest.csv
"""

import re
import argparse
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Must match retrain_with_malaysian.py exactly ──────────────────────────────
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

# ── Boilerplate patterns to strip for the ablated pass ────────────────────────
# Applied AFTER clean_text() (lowercased), so patterns are lowercase.
BOILERPLATE_PATTERNS = [
    r"^according to,?\s*",     # leading "According to," / "According to"
    r"\btruth:\s*",            # translated "TRUTH:" marker, anywhere in text
]

def strip_boilerplate(text):
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def load_malaysian_val_manifest(path):
    manifest = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing expected column(s): {missing}")
    return manifest


def restrict_to_val_rows(df, manifest, text_col="text", label_col="label"):
    """Restrict df to rows whose text also appears in the held-out manifest.
    Mirrors ablate_was_translated_feature.py's restrict_to_val_rows()."""
    manifest_texts = set(manifest[text_col].astype(str))
    before = len(df)
    mask = df[text_col].astype(str).isin(manifest_texts)
    restricted = df[mask].reset_index(drop=True)
    after = len(restricted)

    # Sanity check: labels should agree between df and manifest for matched rows.
    manifest_label_by_text = dict(zip(manifest[text_col].astype(str), manifest[label_col]))
    mismatches = sum(
        1 for t, l in zip(restricted[text_col].astype(str), restricted[label_col])
        if manifest_label_by_text.get(t) != l
    )
    if mismatches:
        print(f"  ⚠️  {mismatches} row(s) matched by text but disagree on label "
              f"with the manifest — investigate before trusting results.")

    print(f"  🎯 restricted to held-out val rows — {before} → {after}")
    return restricted


class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert     = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bilstm   = torch.nn.LSTM(768, 256, num_layers=1,
                                       batch_first=True, bidirectional=True)
        self.dropout   = torch.nn.Dropout(0.3)
        self.fc1       = torch.nn.Linear(1024 + 6, 512)
        self.batchnorm = torch.nn.BatchNorm1d(512)
        self.relu      = torch.nn.ReLU()
        self.fc2       = torch.nn.Linear(512, 2)

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


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class EvalDataset(Dataset):
    def __init__(self, texts, labels, feature_mean, feature_std, strip_bp=False):
        self.texts = texts
        self.labels = labels
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.strip_bp = strip_bp

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(str(self.texts[idx]))
        if self.strip_bp:
            text = strip_boilerplate(text)
        label = self.labels[idx]

        encoding = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=128, return_tensors="pt",
        )
        features = extract_features(text)
        features = (features - self.feature_mean) / (self.feature_std + 1e-8)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": torch.tensor(features),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask, extra_features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, f1, all_preds, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--malaysian-val-manifest", default="malaysian_val_manifest.csv",
                         help="Path to malaysian_val_manifest.csv written by "
                              "retrain_with_malaysian.py (default: malaysian_val_manifest.csv "
                              "in the current directory, matching where retrain writes it)")
    args = parser.parse_args()

    print("📂 Loading data...")
    real_df = pd.read_csv("data/malaysian/malaysian_real.csv")
    fake_df = pd.read_csv("data/malaysian/malaysian_fake_en.csv")

    if "extraction_method" in real_df.columns:
        before = len(real_df)
        real_df = real_df[real_df["extraction_method"] == "full_article"].reset_index(drop=True)
        print(f"Filtered to full_article rows only: {before} -> {len(real_df)}")

    real_df["label"] = 1
    fake_df["label"] = 0

    my_df = pd.concat([real_df, fake_df], ignore_index=True)
    my_df = my_df[my_df["text"].str.len() > 100].reset_index(drop=True)

    print(f"\n📋 Loading Malaysian val manifest: {args.malaysian_val_manifest}")
    val_manifest = load_malaysian_val_manifest(args.malaysian_val_manifest)
    print(f"  {len(val_manifest)} unique held-out rows in manifest "
          f"(includes native replay-buffer rows, which won't match anything in my_df)")

    val_df = restrict_to_val_rows(my_df, val_manifest)

    val_texts = val_df["text"].tolist()
    val_labels = val_df["label"].tolist()
    print(f"Validation set: {len(val_texts)} examples "
          f"({val_labels.count(1)} real, {val_labels.count(0)} fake)")

    # Report how many val examples actually contain the boilerplate,
    # so the ablation's effect size is interpretable.
    hits = sum(1 for t in val_texts if re.search(
        r"according to,|truth:", clean_text(t), re.IGNORECASE))
    print(f"Val examples containing boilerplate markers: {hits}/{len(val_texts)}")

    print("\n📦 Loading model + feature stats...")
    stats = np.load("feature_stats_malaysian.npz")
    feature_mean, feature_std = stats["mean"], stats["std"]

    model = HybridModel().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    print("✅ Loaded feature_stats_malaysian.npz")

    baseline_loader = DataLoader(
        EvalDataset(val_texts, val_labels, feature_mean, feature_std, strip_bp=False),
        batch_size=16)
    ablated_loader = DataLoader(
        EvalDataset(val_texts, val_labels, feature_mean, feature_std, strip_bp=True),
        batch_size=16)

    print("\n" + "=" * 60)
    print("BASELINE — boilerplate phrases intact")
    print("=" * 60)
    acc0, f10, preds0, labels0 = evaluate(model, baseline_loader)
    print(f"Accuracy: {acc0:.4f} | F1: {f10:.4f}")
    print(classification_report(labels0, preds0, target_names=["Fake", "Real"]))

    print("=" * 60)
    print("ABLATED — 'According to,' and 'TRUTH:' markers stripped")
    print("=" * 60)
    acc1, f11, preds1, labels1 = evaluate(model, ablated_loader)
    print(f"Accuracy: {acc1:.4f} | F1: {f11:.4f}")
    print(classification_report(labels1, preds1, target_names=["Fake", "Real"]))

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    drop = acc0 - acc1
    print(f"Accuracy drop from stripping boilerplate: {drop:.4f} ({drop*100:.1f} pp)")
    if drop > 0.05:
        print("⚠️  Large drop — model is likely relying on boilerplate "
              "markers as a shortcut, not genuine content understanding.")
    else:
        print("✅ Small drop — boilerplate markers don't appear to be a "
              "dominant shortcut.")