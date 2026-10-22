# get_feature_stats.py
import numpy as np
import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[!?]{2,}', '!', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(text):
    words = text.split()
    text_length      = len(words)
    exclamation      = text.count('!')
    question         = text.count('?')
    uppercase_ratio  = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    url_count        = text.count('http')
    digit_ratio      = sum(c.isdigit() for c in text) / max(len(text), 1)
    return np.array([
        text_length, exclamation, question,
        uppercase_ratio, url_count, digit_ratio
    ], dtype=np.float32)

# Load your training data
import pandas as pd

def load_liar_all():
    train_df = pd.read_csv("data/liar/train.tsv", sep="\t", header=None)
    valid_df = pd.read_csv("data/liar/valid.tsv", sep="\t", header=None)
    test_df  = pd.read_csv("data/liar/test.tsv",  sep="\t", header=None)
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    df = df[[1, 2]]
    df.columns = ["label", "text"]
    return df

def load_isot_all():
    fake_df = pd.read_csv("data/isot/Fake.csv")
    true_df = pd.read_csv("data/isot/True.csv")
    fake_df["label"] = 0
    true_df["label"] = 1
    return pd.concat([
        fake_df[["text", "label"]],
        true_df[["text", "label"]]
    ], ignore_index=True)

print("Loading datasets...")
liar_df = load_liar_all()
isot_df = load_isot_all()
combined_df = pd.concat([liar_df, isot_df], ignore_index=True)

print(f"Total samples: {len(combined_df)}")

# Extract features from all training texts
print("Extracting features...")
all_features = np.array([
    extract_features(clean_text(str(text)))
    for text in combined_df["text"].tolist()
])

# Calculate mean and std
feature_mean = all_features.mean(axis=0)
feature_std  = all_features.std(axis=0)

print("\n✅ Feature statistics calculated!")
print(f"\nfeature_mean = {feature_mean.tolist()}")
print(f"feature_std  = {feature_std.tolist()}")
print("\nCopy these values into model_service.dart")