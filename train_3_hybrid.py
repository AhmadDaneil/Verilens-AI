"""
============================================================
VeriLens Ablation Study — Model 3: Hybrid (DistilBERT + Bi-LSTM)
============================================================
This is the full proposed VeriLens architecture:
  DistilBERT → Token Embeddings → Bi-LSTM → Mean+Max Pooling →
  Concat Linguistic Features → Dense → Classifier

Identical dataset, splits, epochs, and early stopping as
the DistilBERT-Only and Bi-LSTM-Only scripts.
Dataset : LIAR (official splits) + ISOT Fake News Dataset
Outputs : best_hybrid.pt  +  results_hybrid.txt
============================================================
"""

import os, re, time, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── Reproducibility ───────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Hyperparameters (shared across ALL 3 models) ──────────
MAX_LEN       = 128
BATCH_SIZE    = 16
LR            = 2e-5
EPOCHS        = 6
PATIENCE      = 2
LSTM_HIDDEN   = 128         # Bi-LSTM hidden units per direction
LSTM_LAYERS   = 2
FREEZE_LAYERS = 3           # freeze bottom N DistilBERT transformer blocks
HIDDEN_DIM    = 384         # dense head size
LINGUISTIC_DIM = 6          # number of handcrafted features

# ─── Paths — adjust to your local setup ────────────────────
LIAR_TRAIN = "data/liar/train.tsv"
LIAR_VALID = "data/liar/valid.tsv"
LIAR_TEST  = "data/liar/test.tsv"
ISOT_FAKE  = "data/isot/Fake.csv"
ISOT_REAL  = "data/isot/True.csv"

SAVE_MODEL  = "best_hybrid.pt"
SAVE_RESULT = "results_hybrid.txt"

# ─── LIAR label mapping ────────────────────────────────────
LIAR_FAKE_LABELS = {"false", "pants-fire", "barely-true"}
LIAR_REAL_LABELS = {"true", "mostly-true", "half-true"}


# ══════════════════════════════════════════════════════════════
# 1.  LINGUISTIC FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_linguistic_features(text: str) -> list:
    """
    6 handcrafted features that capture fake-news writing style:
      1. Caps ratio         — ALL-CAPS signals sensationalism
      2. Punctuation ratio  — excessive ! ? . signals emotion
      3. Exclamation count  — clickbait indicator
      4. Question count     — rhetorical doubt
      5. Avg word length    — formal vs informal vocabulary
      6. Sentiment proxy    — ratio of strongly emotive words
    Returns a list of 6 floats, all in [0, 1] or normalised range.
    """
    words = text.split()
    n     = max(len(words), 1)
    chars = max(len(text), 1)

    # 1. Caps ratio
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / n

    # 2. Punctuation ratio
    punct_count = sum(1 for c in text if c in "!?.,;:")
    punct_ratio = min(punct_count / chars, 1.0)

    # 3. Exclamation count (normalised by length)
    excl = text.count("!")
    excl_norm = min(excl / n, 1.0)

    # 4. Question count
    ques = text.count("?")
    ques_norm = min(ques / n, 1.0)

    # 5. Average word length (normalised to [0,1] assuming max 15)
    avg_wlen = min(sum(len(w) for w in words) / n / 15.0, 1.0)

    # 6. Emotive word ratio (simple lexicon)
    emotive = {"shocking", "breaking", "urgent", "exclusive", "exposed",
               "horrible", "terrible", "amazing", "unbelievable", "scandal",
               "outrage", "disgusting", "incredible", "bombshell", "fake",
               "hoax", "lie", "fraud", "corrupt", "conspiracy"}
    emo_ratio = sum(1 for w in words if w.lower() in emotive) / n

    return [caps_ratio, punct_ratio, excl_norm, ques_norm, avg_wlen, emo_ratio]


# ══════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_liar(path):
    df = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2],
                     names=["label", "text"])
    df = df.dropna(subset=["label", "text"])
    texts, labels = [], []
    for _, row in df.iterrows():
        lbl = str(row["label"]).strip().lower()
        if lbl in LIAR_FAKE_LABELS:
            labels.append(0)
        elif lbl in LIAR_REAL_LABELS:
            labels.append(1)
        else:
            continue
        texts.append(str(row["text"]).strip())
    return texts, labels


def load_isot(fake_path, real_path):
    fake_df = pd.read_csv(fake_path)[["text"]].dropna()
    real_df = pd.read_csv(real_path)[["text"]].dropna()
    fake_df["label"] = 0
    real_df["label"] = 1
    combined = pd.concat([fake_df, real_df]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return combined["text"].tolist(), combined["label"].tolist()


def combine_datasets():
    print("Loading LIAR dataset …")
    liar_train_t, liar_train_l = load_liar(LIAR_TRAIN)
    liar_valid_t, liar_valid_l = load_liar(LIAR_VALID)
    liar_test_t,  liar_test_l  = load_liar(LIAR_TEST)

    print("Loading ISOT dataset …")
    isot_t, isot_l = load_isot(ISOT_FAKE, ISOT_REAL)

    n = len(isot_t)
    n_train = int(0.8 * n); n_valid = int(0.1 * n)
    isot_train_t, isot_train_l = isot_t[:n_train],                 isot_l[:n_train]
    isot_valid_t, isot_valid_l = isot_t[n_train:n_train+n_valid],  isot_l[n_train:n_train+n_valid]
    isot_test_t,  isot_test_l  = isot_t[n_train+n_valid:],         isot_l[n_train+n_valid:]

    train_texts  = liar_train_t + isot_train_t
    train_labels = liar_train_l + isot_train_l
    valid_texts  = liar_valid_t + isot_valid_t
    valid_labels = liar_valid_l + isot_valid_l
    test_texts   = liar_test_t  + isot_test_t
    test_labels  = liar_test_l  + isot_test_l

    print(f"  Train: {len(train_texts):,}  |  Valid: {len(valid_texts):,}  |  Test: {len(test_texts):,}")
    return (train_texts, train_labels,
            valid_texts, valid_labels,
            test_texts,  test_labels)


# ══════════════════════════════════════════════════════════════
# 3.  DATASET
# ══════════════════════════════════════════════════════════════

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels    = torch.tensor(labels, dtype=torch.long)
        self.ling_feats = torch.tensor(
            [extract_linguistic_features(t) for t in texts],
            dtype=torch.float
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "ling_feats":     self.ling_feats[idx],
            "label":          self.labels[idx],
        }


# ══════════════════════════════════════════════════════════════
# 4.  MODEL — Hybrid DistilBERT + Bi-LSTM
# ══════════════════════════════════════════════════════════════

class HybridDistilBertBiLSTM(nn.Module):
    """
    Full VeriLens architecture:
      DistilBERT encoder
        → all token embeddings (not just [CLS])
        → Bi-LSTM (sequential modelling)
        → Mean + Max pooling
        → concatenate 6 linguistic features
        → Dense head
        → binary classifier
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Freeze bottom FREEZE_LAYERS transformer blocks
        for i in range(FREEZE_LAYERS):
            for param in self.distilbert.transformer.layer[i].parameters():
                param.requires_grad = False

        bert_hidden = self.distilbert.config.hidden_size  # 768

        self.bilstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if LSTM_LAYERS > 1 else 0.0,
        )

        lstm_out_dim = LSTM_HIDDEN * 2            # bidirectional
        pooled_dim   = lstm_out_dim * 2           # mean + max concat
        fused_dim    = pooled_dim + LINGUISTIC_DIM

        self.dropout    = nn.Dropout(0.3)
        self.norm       = nn.LayerNorm(fused_dim)
        self.fc1        = nn.Linear(fused_dim, HIDDEN_DIM)
        self.act        = nn.GELU()
        self.dropout2   = nn.Dropout(0.2)
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, input_ids, attention_mask, ling_feats):
        # ── DistilBERT ──
        bert_out    = self.distilbert(input_ids=input_ids,
                                      attention_mask=attention_mask)
        token_embs  = bert_out.last_hidden_state          # (B, seq, 768)
        token_embs  = self.dropout(token_embs)

        # ── Bi-LSTM ──
        lstm_out, _ = self.bilstm(token_embs)             # (B, seq, 2*LSTM_HIDDEN)

        # ── Pooling (mask padding) ──
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq, 1)
        # Mean pooling (only non-pad positions)
        sum_emb   = (lstm_out * mask_expanded).sum(dim=1)
        count     = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pool = sum_emb / count                       # (B, 2*LSTM_HIDDEN)
        # Max pooling (mask pad with -inf before max)
        masked_lstm = lstm_out + (1 - mask_expanded) * (-1e9)
        max_pool, _ = masked_lstm.max(dim=1)              # (B, 2*LSTM_HIDDEN)

        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, 4*LSTM_HIDDEN)

        # ── Fuse linguistic features ──
        fused = torch.cat([pooled, ling_feats], dim=1)    # (B, fused_dim)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        x     = self.act(self.fc1(fused))
        x     = self.dropout2(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lf    = batch["ling_feats"].to(device)
            lbls  = batch["label"].to(device)
            logits = model(ids, mask, lf)
            loss   = criterion(logits, lbls)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="weighted")
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, f1, prec, rec, all_preds, all_labels


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  MODEL 3 — Hybrid: DistilBERT + Bi-LSTM (Proposed)")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    (train_t, train_l,
     valid_t, valid_l,
     test_t,  test_l) = combine_datasets()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print("Tokenising and extracting linguistic features …")
    train_ds = NewsDataset(train_t, train_l, tokenizer)
    valid_ds = NewsDataset(valid_t, valid_l, tokenizer)
    test_ds  = NewsDataset(test_t,  test_l,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model     = HybridDistilBertBiLSTM().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_f1, patience_cnt = 0.0, 0
    log_lines = [
        "MODEL: Hybrid DistilBERT + Bi-LSTM (Proposed Architecture)\n",
        f"Hyperparameters: MAX_LEN={MAX_LEN}, BATCH={BATCH_SIZE}, LR={LR}, "
        f"EPOCHS={EPOCHS}, PATIENCE={PATIENCE}, LSTM_HIDDEN={LSTM_HIDDEN}, "
        f"FREEZE_LAYERS={FREEZE_LAYERS}\n\n"
    ]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader, 1):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lf    = batch["ling_feats"].to(device)
            lbls  = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(ids, mask, lf)
            loss   = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if step % 200 == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch} | Step {step}/{len(train_loader)} | "
                      f"Loss {epoch_loss/step:.4f} | Elapsed {elapsed:.0f}s")

        avg_train_loss = epoch_loss / len(train_loader)
        _, val_acc, val_f1, val_prec, val_rec, _, _ = evaluate(model, valid_loader, device, criterion)
        scheduler.step(val_f1)

        line = (f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
                f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f}")
        print(line)
        log_lines.append(line + "\n")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_MODEL)
            print(f"  ✅ Best model saved (F1={best_f1:.4f})")
            log_lines.append(f"  ✅ Best model saved (F1={best_f1:.4f})\n")
        else:
            patience_cnt += 1
            msg = f"  No improvement. Patience: {patience_cnt}/{PATIENCE}"
            print(msg)
            log_lines.append(msg + "\n")
            if patience_cnt >= PATIENCE:
                print("  Early stopping triggered.")
                log_lines.append("  Early stopping triggered.\n")
                break

    # ── Final test evaluation ──
    print("\nLoading best model for final test evaluation …")
    model.load_state_dict(torch.load(SAVE_MODEL, map_location=device))
    _, test_acc, test_f1, test_prec, test_rec, test_preds, test_lbls = \
        evaluate(model, test_loader, device, criterion)

    report = classification_report(test_lbls, test_preds,
                                   target_names=["FAKE", "REAL"])

    summary = (
        f"\n{'='*50}\n"
        f"FINAL TEST RESULTS — Hybrid DistilBERT + Bi-LSTM\n"
        f"{'='*50}\n"
        f"Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)\n"
        f"Test F1 Score  : {test_f1:.4f} ({test_f1*100:.2f}%)\n"
        f"Test Precision : {test_prec:.4f}\n"
        f"Test Recall    : {test_rec:.4f}\n\n"
        f"Classification Report:\n{report}\n"
    )
    print(summary)
    log_lines.append(summary)

    with open(SAVE_RESULT, "w", encoding="utf-8") as f:
        f.writelines(log_lines)
    print(f"Results saved → {SAVE_RESULT}")


if __name__ == "__main__":
    train()
