import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import re

from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

############################################
# LOAD DATA
############################################

def load_liar_all():
    train_df = pd.read_csv("data/liar/train.tsv", sep="\t", header=None)
    valid_df = pd.read_csv("data/liar/valid.tsv", sep="\t", header=None)
    test_df  = pd.read_csv("data/liar/test.tsv", sep="\t", header=None)

    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    df = df[[1, 2]]
    df.columns = ["label", "text"]

    fake_labels = ["false", "barely-true", "pants-fire"]
    df["label"] = df["label"].apply(lambda x: 0 if x in fake_labels else 1)

    return df


def load_isot_all():
    fake_df = pd.read_csv("data/isot/Fake.csv")
    true_df = pd.read_csv("data/isot/True.csv")

    fake_df["label"] = 0
    true_df["label"] = 1

    return pd.concat([fake_df[["text","label"]], true_df[["text","label"]]], ignore_index=True)


def load_fakenewsnet_df(root_path):
    texts, labels = [], []

    files = [
        ("gossipcop_real.csv", 1),
        ("gossipcop_fake.csv", 0),
        ("politifact_real.csv", 1),
        ("politifact_fake.csv", 0),
    ]

    for filename, label in files:
        path = os.path.join(root_path, filename)

        if os.path.exists(path):
            df = pd.read_csv(path)

            col = "text" if "text" in df.columns else \
                  "title" if "title" in df.columns else \
                  "content" if "content" in df.columns else None

            if col is None:
                continue

            for t in df[col].fillna(""):
                if len(str(t).strip()) > 0:
                    texts.append(t)
                    labels.append(label)

    return pd.DataFrame({"text": texts, "label": labels})


print("Loading datasets...")
liar_df = load_liar_all()
isot_df = load_isot_all()
fnn_df  = load_fakenewsnet_df("data/fakenewsnet")

############################################
# SPLIT DATA (FIXED)
############################################

fnn_train, fnn_external = train_test_split(
    fnn_df,
    test_size=0.5,
    stratify=fnn_df["label"],
    random_state=42
)

combined_df = pd.concat([liar_df, isot_df, fnn_train], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, temp_df = train_test_split(
    combined_df,
    test_size=0.2,
    stratify=combined_df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

# 🔥 IMPORTANT FIX
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

############################################
# TEXT PROCESSING
############################################

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
    return np.array([
        len(text.split()),
        text.count("!"),
        text.count("?"),
        sum(1 for c in text if c.isupper()) / max(len(text),1),
        text.count("http"),
        sum(c.isdigit() for c in text) / max(len(text),1)
    ], dtype=np.float32)

############################################
# DATASET
############################################

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(str(self.texts[idx]))

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        features = extract_features(text)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "features": torch.tensor(features, dtype=torch.float),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 🔥 IMPORTANT FIX (.tolist())
train_loader = DataLoader(
    NewsDataset(train_df["text"].tolist(), train_df["label"].tolist()),
    batch_size=16, shuffle=True
)

val_loader = DataLoader(
    NewsDataset(val_df["text"].tolist(), val_df["label"].tolist()),
    batch_size=16
)

test_loader = DataLoader(
    NewsDataset(test_df["text"].tolist(), test_df["label"].tolist()),
    batch_size=16
)

############################################
# MODELS
############################################

class DistilBERT_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768+6, 2)

    def forward(self, ids, mask, feats):
        x = self.bert(ids, attention_mask=mask).last_hidden_state[:,0]
        x = torch.cat((x, feats), dim=1)
        return self.fc(x)


class DistilBERT_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024+6, 2)

    def forward(self, ids, mask, feats):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x,_ = self.lstm(x)

        mean_pool = torch.mean(x, dim=1)
        max_pool,_ = torch.max(x, dim=1)

        x = torch.cat((mean_pool, max_pool, feats), dim=1)
        return self.fc(x)

############################################
# TRAIN / EVAL
############################################

def evaluate(model, loader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for b in loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            feats = b["features"].to(device)
            y = b["labels"].to(device)

            out = model(ids, mask, feats)
            p = torch.argmax(out, dim=1)

            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    return accuracy_score(labels, preds), f1_score(labels, preds)


def train_model(model, name):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()

        for b in tqdm(train_loader):
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            feats = b["features"].to(device)
            y = b["labels"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(ids,mask,feats), y)
            loss.backward()
            optimizer.step()

    acc, f1 = evaluate(model, test_loader)
    print(f"{name} → Acc: {acc:.4f}, F1: {f1:.4f}")
    return acc, f1

############################################
# RUN BOTH MODELS
############################################

print("\n=== BASELINE ===")
base_acc, base_f1 = train_model(DistilBERT_Baseline(), "DistilBERT")

print("\n=== HYBRID ===")
hyb_acc, hyb_f1 = train_model(DistilBERT_BiLSTM(), "DistilBERT+BiLSTM")

############################################
# FINAL COMPARISON
############################################

print("\n📊 FINAL COMPARISON")
print("-----------------------------------")
print(f"DistilBERT        → F1: {base_f1:.4f}")
print(f"DistilBERT+BiLSTM → F1: {hyb_f1:.4f}")
print(f"Improvement       → {hyb_f1 - base_f1:.4f}")