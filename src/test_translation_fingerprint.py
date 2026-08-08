"""
test_translation_fingerprint.py
--------------------------------
Tests whether the fine-tuned model is keying on "this text was machine-
translated" rather than "this text is false."

Rationale:
    Every fake-class example the model was fine-tuned on went through
    NLLB Malay->English translation. Every real-class example is native
    English. Machine-translated text has systematic stylistic tells
    (word choice, sentence rhythm, occasional artifacts like mistranslated
    proper nouns) that are completely independent of truthfulness. A model
    could hit near-perfect validation scores by learning "translated ->
    fake" instead of learning anything about misinformation.

Method:
    Evaluate the current best_model.pt on a held-out sample of NATIVE
    ENGLISH fake news (LIAR / ISOT / FakeNewsNet - never translated) and
    compare its accuracy and average fake-class confidence against its
    performance on the translated Malaysian fake-class set.

    If the model relies on translation artifacts as a shortcut, we'd
    expect:
        - Much lower accuracy / recall on native-English fake news
          (it "reads real" because it isn't translated)
        - Or much lower average fake-confidence on native-English fake
          news even where the prediction is still technically correct

    If the model has learned genuine content signal, accuracy and
    confidence should be roughly comparable across both fake-class
    sources.

Usage:
    python test_translation_fingerprint.py
"""

import re
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
N_SAMPLES_PER_SOURCE = 150   # cap per dataset so no single source dominates
RANDOM_STATE = 42

def load_native_english_fake():
    """
    Returns a DataFrame with a 'text' column containing NATIVE ENGLISH
    fake-class examples, pulled from LIAR / ISOT / FakeNewsNet - i.e.
    the datasets the ORIGINAL (pre-Malaysian) model was trained on.
    These were never passed through NLLB translation.
    """
    frames = []

    # --- ISOT (Fake.csv has 'text' column, all fake) -----------------------
    try:
        isot = pd.read_csv("data/isot/Fake.csv")
        isot_text_col = "text" if "text" in isot.columns else isot.columns[1]
        df = pd.DataFrame({"text": isot[isot_text_col].astype(str)})
        df["source"] = "ISOT"
        frames.append(df)
    except FileNotFoundError:
        print("⚠️  ISOT Fake.csv not found - skipping (adjust path in script if you have it)")

    # --- LIAR (label column has multiple fake-ish classes) ------------------
    try:
        liar = pd.read_csv("data/liar/train.tsv", sep="\t", header=None)
        # Standard LIAR columns: id, label, statement, subject, speaker, ...
        liar.columns = ["id", "label", "statement"] + [f"col{i}" for i in range(liar.shape[1] - 3)]
        fake_labels = ["false", "pants-fire", "barely-true"]
        liar_fake = liar[liar["label"].isin(fake_labels)]
        df = pd.DataFrame({"text": liar_fake["statement"].astype(str)})
        df["source"] = "LIAR"
        frames.append(df)
    except FileNotFoundError:
        print("⚠️  LIAR train.tsv not found - skipping (adjust path in script if you have it)")

    # --- FakeNewsNet ------------------------------------------------------
    # NOTE: this release only has a 'title' column, not full article text.
    # Titles are short (~10-15 words) vs ISOT's full articles and the
    # translated Malaysian fake set (~140 words avg). To avoid re-introducing
    # the same length confound as Root Cause #2, this is kept as its own
    # separately-labeled source (FakeNewsNet-title) rather than blended
    # into the general "native English fake" pool - so if it behaves
    # differently in results, that's attributable to length, not translation.
    try:
        fnn_frames = []
        for fname in ["gossipcop_fake.csv", "politifact_fake.csv"]:
            fnn = pd.read_csv(f"data/fakenewsnet/{fname}")
            df = pd.DataFrame({"text": fnn["title"].astype(str)})
            df["source"] = "FakeNewsNet-title"
            fnn_frames.append(df)
        frames.extend(fnn_frames)
    except FileNotFoundError:
        print("⚠️  FakeNewsNet files not found - skipping (adjust path in script if you have it)")

    if not frames:
        raise SystemExit(
            "\n❌ No native-English fake-news source files were found.\n"
            "Edit load_native_english_fake() in this script to point at "
            "wherever your original LIAR / ISOT / FakeNewsNet CSVs live "
            "(the same files the ORIGINAL pre-Malaysian model trained on)."
        )

    combined = pd.concat(frames, ignore_index=True)
    # Only apply the >100-char filter to full-article sources; FakeNewsNet
    # titles are inherently short and would be entirely wiped out by it.
    is_title_source = combined["source"] == "FakeNewsNet-title"
    long_enough = combined["text"].str.len() > 100
    combined = combined[is_title_source | long_enough].reset_index(drop=True)

    # Sample evenly-ish across sources, capped, for a manageable eval set
    sampled = (
        combined.groupby("source", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), N_SAMPLES_PER_SOURCE), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    return sampled


def load_translated_malaysian_fake():
    """Translated Malaysian fake-class set - the one the model WAS fine-tuned on."""
    df = pd.read_csv("data/malaysian/malaysian_fake_en.csv")
    df = df[df["text"].astype(str).str.len() > 100].reset_index(drop=True)
    if len(df) > N_SAMPLES_PER_SOURCE * 3:
        df = df.sample(n=N_SAMPLES_PER_SOURCE * 3, random_state=RANDOM_STATE).reset_index(drop=True)
    df["source"] = "Malaysian (translated)"
    return df[["text", "source"]]


# ═══════════════════════════════════════════════════════════════════════════
# Model / feature code - must match retrain_with_malaysian.py exactly
# ═══════════════════════════════════════════════════════════════════════════
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


class InferDataset(Dataset):
    def __init__(self, texts, feature_mean, feature_std):
        self.texts = texts
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(str(self.texts[idx]))
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
        }


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds, all_fake_conf = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_features = batch["extra_features"].to(device)
        outputs = model(input_ids, attention_mask, extra_features)
        probs = torch.softmax(outputs, dim=1)  # [:,0]=fake, [:,1]=real (label 0=fake,1=real per training code)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_fake_conf.extend(probs[:, 0].cpu().numpy())  # confidence assigned to FAKE class
    return np.array(all_preds), np.array(all_fake_conf)


def report_for_source(name, preds, fake_conf, true_label=0):
    """true_label=0 means every example in this set is actually FAKE."""
    acc = (preds == true_label).mean()
    print(f"\n--- {name} (n={len(preds)}, all true label = FAKE) ---")
    print(f"  Accuracy (correctly flagged as FAKE): {acc:.4f}")
    print(f"  Mean fake-class confidence:            {fake_conf.mean():.4f}")
    print(f"  Median fake-class confidence:          {np.median(fake_conf):.4f}")
    print(f"  % predicted REAL (i.e. missed):        {(preds != true_label).mean()*100:.1f}%")
    return acc, fake_conf.mean()


if __name__ == "__main__":
    print("📂 Loading native-English fake-news sample (LIAR / ISOT / FakeNewsNet)...")
    native_df = load_native_english_fake()
    print(f"Loaded {len(native_df)} native-English fake examples:")
    print(native_df["source"].value_counts())

    print("\n📂 Loading translated Malaysian fake-news sample...")
    translated_df = load_translated_malaysian_fake()
    print(f"Loaded {len(translated_df)} translated Malaysian fake examples")

    print("\n📦 Loading model + feature stats...")
    stats = np.load("feature_stats_malaysian.npz")
    feature_mean, feature_std = stats["mean"], stats["std"]

    model = HybridModel().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    print("✅ Model loaded")

    native_loader = DataLoader(
        InferDataset(native_df["text"].tolist(), feature_mean, feature_std), batch_size=16)
    translated_loader = DataLoader(
        InferDataset(translated_df["text"].tolist(), feature_mean, feature_std), batch_size=16)

    native_preds, native_conf = predict(model, native_loader)
    translated_preds, translated_conf = predict(model, translated_loader)

    print("\n" + "=" * 65)
    print("TRANSLATION-FINGERPRINT TEST RESULTS")
    print("=" * 65)
    acc_native, conf_native = report_for_source("NATIVE ENGLISH fake news (never translated)",
                                                  native_preds, native_conf)
    acc_trans, conf_trans = report_for_source("TRANSLATED Malaysian fake news (NLLB output)",
                                                translated_preds, translated_conf)

    # Per-source breakdown for native set, since it's a mix of 3 datasets
    print("\n--- Breakdown by native source ---")
    native_df = native_df.reset_index(drop=True)
    for src in native_df["source"].unique():
        idx = native_df.index[native_df["source"] == src].to_numpy()
        report_for_source(f"  {src}", native_preds[idx], native_conf[idx])

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    acc_gap = acc_trans - acc_native
    conf_gap = conf_trans - conf_native
    print(f"Accuracy gap (translated - native):   {acc_gap:+.4f} ({acc_gap*100:+.1f} pp)")
    print(f"Fake-confidence gap (translated - native): {conf_gap:+.4f}")

    if acc_gap > 0.15 or conf_gap > 0.15:
        print(
            "\n⚠️  Large gap — the model performs much better on translated "
            "fake news than on native-English fake news. This is strong "
            "evidence it has learned a 'translated text -> fake' shortcut "
            "rather than genuine truthfulness signal. This would also "
            "explain the real-world failures on genuine Malaysian-English "
            "articles: the model may be using 'native-sounding English' as "
            "a proxy for 'real', regardless of actual content."
        )
    else:
        print(
            "\n✅ Small gap — the model performs comparably on native and "
            "translated fake news. Translation fingerprint does not appear "
            "to be a dominant shortcut."
        )