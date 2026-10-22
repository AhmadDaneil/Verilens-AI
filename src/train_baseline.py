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
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

############################################
# 1 LOAD PRE-CLEANED, PRE-SPLIT DATASET
############################################
# train.csv / val.csv / test.csv were produced by clean_verilens_data.py from
# the ISOT True.csv / Fake.csv source. They are already: leakage-stripped
# (Reuters byline / brand-name removed, site boilerplate removed), deduped,
# class-balanced, shuffled, and stratified-split 80/10/10.
# Columns: title, text, label (1 = real, 0 = fake).
#
# LIAR and FakeNewsNet are intentionally NOT included here: LIAR is short
# fact-checked claims (a different task from full-article classification),
# and FakeNewsNet only provides titles/URLs, not article bodies. If you want
# an external, out-of-distribution generalization check later, that's where
# a source like FakeNewsNet would come back in -- but it would need full
# article text scraped from its URLs first.

DATA_DIR = "data/verilens_clean"

def load_split(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    # Combine title + body into a single input string
    df["text"] = (df["title"] + ". " + df["text"]).str.strip()
    return df[["text", "label"]]

print("Loading cleaned train/val/test splits...")
train_df = load_split("train")
val_df = load_split("val")
test_df = load_split("test")

print("Train:", len(train_df), train_df["label"].value_counts().to_dict())
print("Val:", len(val_df), val_df["label"].value_counts().to_dict())
print("Test:", len(test_df), test_df["label"].value_counts().to_dict())

############################################
# 2️ FEATURE EXTRACTION
############################################

def clean_text(text):

    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove html
    text = re.sub(r"<.*?>", "", text)

    # remove usernames
    text = re.sub(r"@\w+", "", text)

    # remove hashtags symbol only
    text = re.sub(r"#", "", text)

    # remove multiple punctuation
    text = re.sub(r"[!?]{2,}", "!", text)

    # remove extra spaces
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
        text_length,
        exclamation_count,
        question_count,
        uppercase_ratio,
        url_count,
        digit_ratio
    ], dtype=np.float32)


train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()

val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()

test_texts = test_df["text"].tolist()
test_labels = test_df["label"].tolist()

all_train_features = np.array([extract_features(t) for t in train_texts])
feature_mean = all_train_features.mean(axis=0)
feature_std  = all_train_features.std(axis=0)

# Save normalization stats so test_verilens.py uses the exact same scaling
# the model was trained on -- recomputing mean/std from a different dataset
# would feed the model features on the wrong scale and hurt accuracy.
np.savez("feature_norm_stats.npz", mean=feature_mean, std=feature_std)

############################################
# 3️ DATASET CLASS
############################################

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_vocabulary(".")

class HybridDataset(Dataset):

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
            max_length=128,
            return_tensors="pt"
        )

        features = extract_features(text)
        features = (features - feature_mean) / (feature_std + 1e-8)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "extra_features": torch.tensor(features),
            "labels": torch.tensor(label, dtype=torch.long)
        }


train_loader = DataLoader(HybridDataset(train_texts, train_labels), batch_size=16, shuffle=True)
val_loader   = DataLoader(HybridDataset(val_texts, val_labels), batch_size=16)
test_loader  = DataLoader(HybridDataset(test_texts, test_labels), batch_size=16)

############################################
# 4️ MODEL
############################################

class HybridModel(nn.Module):

    def __init__(self):
        super(HybridModel, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1024 + 6, 512)

        self.batchnorm = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        # (batch, seq_len, 768)

        lstm_output, _ = self.bilstm(sequence_output)
        # (batch, seq_len, 512)

        mean_pool = torch.mean(lstm_output, dim=1)
        max_pool, _ = torch.max(lstm_output, dim=1)

        pooled_output = torch.cat((mean_pool, max_pool), dim=1)
        # shape: (batch, 1024)

        combined = torch.cat((pooled_output, extra_features), dim=1)

        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        logits = self.fc2(x)

        return logits


model = HybridModel().to(device)


############################################
# 5️ TRAINING SETUP
############################################
# Class weights left at 1:1 since train.csv is already balanced by
# clean_verilens_data.py. Adjust only if you re-split away from that balance.

class_weights = torch.tensor([1.0, 1.0]).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

############################################
# 6️ EVALUATION FUNCTION (DEBUG ENABLED)
############################################

def evaluate(loader, debug=False):

    model.eval()
    all_preds = []
    all_labels = []

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

    if debug:
        print("\nPrediction distribution:")
        print(pd.Series(all_preds).value_counts())

    return accuracy_score(all_labels, all_preds), \
           f1_score(all_labels, all_preds, average="binary")

############################################
# 7 TRAIN
############################################

best_val_f1 = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_features = batch["extra_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, extra_features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    val_acc, val_f1 = evaluate(val_loader)

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", total_loss/len(train_loader))
    print("Validation Accuracy:", val_acc)
    print("Validation F1:", val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pt")

############################################
# 8 FINAL TEST
############################################

model.load_state_dict(torch.load("best_model.pt"))

test_acc, test_f1 = evaluate(test_loader, debug=True)

print("\n🔥 FINAL TEST RESULTS")
print("Final Test Accuracy:", test_acc)
print("Final Test F1:", test_f1)