"""
Verilens — Model Testing / Evaluation Script
=============================================
Loads a trained best_model.pt and evaluates it on a labeled CSV -- either
your own held-out test.csv (in-domain check) or a separate external dataset
(out-of-domain / cross-country generalization check).

USAGE:
    python test_verilens.py --data data/verilens_clean/test.csv
    python test_verilens.py --data data/external/uk_news_labeled.csv --title-col title --text-col text --label-col label

Input CSV requirements:
  - A text column (article body). If a title column is also given, they are
    concatenated the same way as during training ("title. text").
  - A label column with 1 = real, 0 = fake.
    If your external dataset uses different label values (e.g. "real"/"fake"
    strings, or 1=fake/0=real), map them to 1=real/0=fake BEFORE passing the
    file in, or edit the --label-map arguments below.

Outputs:
  - Accuracy, F1, precision, recall, confusion matrix printed to console.
  - predictions_<inputfilename>.csv saved next to this script with a
    per-article predicted label and confidence, for manual error review.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to labeled CSV to evaluate on")
parser.add_argument("--title-col", default="title", help="Title column name (set to '' if none)")
parser.add_argument("--text-col", default="text", help="Body text column name")
parser.add_argument("--label-col", default="label", help="Label column name (1=real, 0=fake)")
parser.add_argument("--model-path", default="best_model.pt")
parser.add_argument("--stats-path", default="feature_norm_stats.npz",
                     help="Feature normalization stats saved by train_verilens.py")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-length", type=int, default=128)
args = parser.parse_args()

# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------
df = pd.read_csv(args.data)

if args.label_col not in df.columns:
    raise ValueError(
        f"Label column '{args.label_col}' not found. Columns present: {list(df.columns)}"
    )

df[args.text_col] = df[args.text_col].fillna("")
if args.title_col and args.title_col in df.columns:
    df[args.title_col] = df[args.title_col].fillna("")
    combined_text = (df[args.title_col] + ". " + df[args.text_col]).str.strip()
else:
    combined_text = df[args.text_col]

texts = combined_text.tolist()
labels = df[args.label_col].astype(int).tolist()

print(f"Loaded {len(texts)} rows from {args.data}")
print("Label distribution:", pd.Series(labels).value_counts().to_dict())

# ---------------------------------------------------------------------
# LOAD SAVED FEATURE NORMALIZATION STATS (must match training)
# ---------------------------------------------------------------------
if not os.path.exists(args.stats_path):
    raise FileNotFoundError(
        f"'{args.stats_path}' not found. Re-run the (already-updated) "
        f"train_verilens.py first -- it now saves this file automatically "
        f"after computing feature_mean/feature_std."
    )
stats = np.load(args.stats_path)
feature_mean = stats["mean"]
feature_std = stats["std"]

# ---------------------------------------------------------------------
# TEXT CLEANING + FEATURES (identical to training script -- do not change
# independently of train_verilens.py or the two will drift out of sync)
# ---------------------------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text):
    words = text.split()
    text_length = len(words)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    url_count = text.count("http")
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return np.array([
        text_length, exclamation_count, question_count,
        uppercase_ratio, url_count, digit_ratio
    ], dtype=np.float32)


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


class EvalDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(str(self.texts[idx]))
        label = self.labels[idx]

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        features = extract_features(text)
        features = (features - feature_mean) / (feature_std + 1e-8)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": torch.tensor(features, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


eval_loader = DataLoader(EvalDataset(texts, labels), batch_size=args.batch_size)

# ---------------------------------------------------------------------
# MODEL (identical architecture to training script)
# ---------------------------------------------------------------------

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bilstm = nn.LSTM(
            input_size=768, hidden_size=256, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1024 + 6, 512)
        self.batchnorm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        mean_pool = torch.mean(lstm_output, dim=1)
        max_pool, _ = torch.max(lstm_output, dim=1)
        pooled_output = torch.cat((mean_pool, max_pool), dim=1)
        combined = torch.cat((pooled_output, extra_features), dim=1)
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc2(x)


print(f"Loading model weights from {args.model_path} ...")
model = HybridModel().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# ---------------------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------------------
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_features = batch["extra_features"].to(device)
        batch_labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, extra_features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(real)
        all_labels.extend(batch_labels.cpu().numpy())

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="binary")
precision = precision_score(all_labels, all_preds, average="binary")
recall = recall_score(all_labels, all_preds, average="binary")
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "=" * 50)
print(f"RESULTS on {args.data}")
print("=" * 50)
print(f"Accuracy:  {acc:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("\nConfusion matrix (rows=true, cols=predicted, order=[fake, real]):")
print(cm)
print("\nFull classification report:")
print(classification_report(all_labels, all_preds, target_names=["fake", "real"]))

# ---------------------------------------------------------------------
# SAVE PER-ARTICLE PREDICTIONS FOR ERROR REVIEW
# ---------------------------------------------------------------------
out_name = f"predictions_{os.path.splitext(os.path.basename(args.data))[0]}.csv"
result_df = pd.DataFrame({
    "text": texts,
    "true_label": all_labels,
    "predicted_label": all_preds,
    "prob_real": all_probs,
    "correct": np.array(all_labels) == np.array(all_preds),
})
result_df.to_csv(out_name, index=False)
print(f"\nPer-article predictions saved to {out_name}")