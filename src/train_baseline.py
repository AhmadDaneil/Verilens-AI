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
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

############################################
# 1 LOAD DATASETS
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

    fake_df = fake_df[["text", "label"]]
    true_df = true_df[["text", "label"]]

    return pd.concat([fake_df, true_df], ignore_index=True)


def load_fakenewsnet_df(root_path):

    texts = []
    labels = []

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

            print("\nLoaded:", filename)
            print("Columns:", df.columns)

            # Detect correct column
            if "text" in df.columns:
                text_column = "text"
            elif "title" in df.columns:
                text_column = "title"
            elif "content" in df.columns:
                text_column = "content"
            else:
                print("❌ No usable text column in", filename)
                continue

            for text in df[text_column].fillna(""):
                if len(str(text).strip()) > 0:
                    texts.append(text)
                    labels.append(label)

        else:
            print("❌ Missing file:", filename)

    return pd.DataFrame({
        "text": texts,
        "label": labels
    })

print("Loading LIAR...")
liar_df = load_liar_all()

print("Loading ISOT...")
isot_df = load_isot_all()

print("Loading FakeNewsNet...")
fakenewsnet_df = load_fakenewsnet_df("data/fakenewsnet")

print("FakeNewsNet total:", len(fakenewsnet_df))
print(fakenewsnet_df["label"].value_counts())


############################################
# 2 SPLIT FAKENEWSNET (50% TRAIN / 50% EXTERNAL)
############################################

fnn_train_df, fnn_external_df = train_test_split(
    fakenewsnet_df,
    test_size=0.5,
    random_state=42,
    stratify=fakenewsnet_df["label"]
)

print("FakeNewsNet Train:", len(fnn_train_df))
print("FakeNewsNet External:", len(fnn_external_df))


############################################
# 3 COMBINE ONLY LIAR + ISOT
############################################

combined_df = pd.concat(
    [liar_df, isot_df, fnn_train_df],
    ignore_index=True
)

combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Training samples:", len(combined_df))
print(combined_df["label"].value_counts())

############################################
# 4️ SPLIT DATA
############################################

train_df, temp_df = train_test_split(
    combined_df,
    test_size=0.2,
    random_state=42,
    stratify=combined_df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

############################################
# 5️ FEATURE EXTRACTION
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

############################################
# 6️ DATASET CLASS
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
# 7️ MODEL
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
# 8️ TRAINING SETUP
############################################

class_weights = torch.tensor([1.2, 1.0]).to(device)

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
# 9️ EVALUATION FUNCTION (DEBUG ENABLED)
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
# 10 TRAIN
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
# 11 FINAL TEST
############################################

model.load_state_dict(torch.load("best_model.pt"))

test_acc, test_f1 = evaluate(test_loader)

print("\n🔥 FINAL TEST RESULTS")
print("Final Test Accuracy:", test_acc)
print("Final Test F1:", test_f1)

############################################
# DOMAIN ADAPTATION (FakeNewsNet Fine-Tuning)
############################################

print("\nStarting FakeNewsNet fine-tuning...")

fnn_texts = fnn_train_df["text"].tolist()
fnn_labels = fnn_train_df["label"].tolist()

fnn_loader = DataLoader(
    HybridDataset(fnn_texts, fnn_labels),
    batch_size=8,
    shuffle=True
)

for param in model.bert.parameters():
    param.requires_grad = False

for param in model.bert.transformer.layer[-2:].parameters():
    param.requires_grad = True

# Lower learning rate for fine-tuning
optimizer = AdamW(model.parameters(), lr=5e-6)

epochs = 4

for epoch in range(epochs):

    model.train()
    total_loss = 0
    

    for batch in tqdm(fnn_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_features = batch["extra_features"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, extra_features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nFine-tune Epoch {epoch+1}")
    print("Train Loss:", total_loss/len(fnn_loader))

############################################
# 12 EXTERNAL TEST AFTER DOMAIN ADAPTATION
############################################

external_texts = fnn_external_df["text"].tolist()
external_labels = fnn_external_df["label"].tolist()

external_loader = DataLoader(
    HybridDataset(external_texts, external_labels),
    batch_size=16
)

external_acc, external_f1 = evaluate(external_loader)

print("\n🌍 EXTERNAL RESULTS AFTER DOMAIN ADAPTATION")
print("External Accuracy:", external_acc)
print("External F1:", external_f1)