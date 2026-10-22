"""
============================================================
VeriLens Ablation Study — Model 2: Bi-LSTM ONLY
============================================================
Pure Bi-LSTM with GloVe / learnable word embeddings.
Identical dataset, splits, epochs and early stopping as
the DistilBERT and Hybrid scripts.
Dataset : LIAR (official splits) + ISOT Fake News Dataset
Outputs : best_bilstm_only.pt  +  results_bilstm_only.txt
============================================================
"""

import os, time, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter

# ─── Reproducibility ───────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Hyperparameters (shared across ALL 3 models) ──────────
MAX_LEN       = 128
BATCH_SIZE    = 16
LR            = 1e-3        # higher LR suitable for LSTM (no pretrained weights)
EPOCHS        = 6
PATIENCE      = 2
VOCAB_SIZE    = 50_000      # top-N words kept
EMBED_DIM     = 300         # embedding dimension
LSTM_HIDDEN   = 128         # Bi-LSTM hidden units (× 2 bidirectional)
LSTM_LAYERS   = 2
DROPOUT       = 0.3
HIDDEN_DIM    = 384         # dense head size (mirrors DistilBERT scripts)

# ─── Paths — adjust to your local setup ────────────────────
LIAR_TRAIN = "data/liar/train.tsv"
LIAR_VALID = "data/liar/valid.tsv"
LIAR_TEST  = "data/liar/test.tsv"
ISOT_FAKE  = "data/isot/Fake.csv"
ISOT_REAL  = "data/isot/True.csv"

SAVE_MODEL  = "best_bilstm_only.pt"
SAVE_RESULT = "results_bilstm_only.txt"

# ─── LIAR label mapping ────────────────────────────────────
LIAR_FAKE_LABELS = {"false", "pants-fire", "barely-true"}
LIAR_REAL_LABELS = {"true", "mostly-true", "half-true"}


# ══════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (same helpers as other scripts)
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
# 2.  VOCABULARY + TOKENISER (simple whitespace)
# ══════════════════════════════════════════════════════════════

PAD_IDX = 0
UNK_IDX = 1


def build_vocab(texts):
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    tokens = text.lower().split()[:MAX_LEN]
    ids = [vocab.get(tok, UNK_IDX) for tok in tokens]
    # Pad / truncate to MAX_LEN
    ids = ids + [PAD_IDX] * (MAX_LEN - len(ids))
    return ids


# ══════════════════════════════════════════════════════════════
# 3.  DATASET
# ══════════════════════════════════════════════════════════════

class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.ids    = [encode(t, vocab) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.ids[idx],    dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════
# 4.  MODEL — Bi-LSTM ONLY  (no DistilBERT)
# ══════════════════════════════════════════════════════════════

class BiLSTMOnly(nn.Module):
    """
    Learnable embedding → Bi-LSTM (stacked) → mean + max pooling →
    Dense → classifier.
    Architecture mirrors the LSTM branch of the Hybrid model so that
    the component contribution can be fairly ablated.
    """
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM,
                                      padding_idx=PAD_IDX)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.embedding.weight.data[PAD_IDX].fill_(0)

        self.bilstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        lstm_out_dim = LSTM_HIDDEN * 2  # bidirectional
        # Mean + Max pooling → concatenate
        pooled_dim   = lstm_out_dim * 2

        self.dropout    = nn.Dropout(DROPOUT)
        self.norm       = nn.LayerNorm(pooled_dim)
        self.fc1        = nn.Linear(pooled_dim, HIDDEN_DIM)
        self.act        = nn.GELU()
        self.dropout2   = nn.Dropout(0.2)
        self.classifier = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, input_ids):
        # input_ids : (B, seq_len)
        emb = self.embedding(input_ids)            # (B, seq_len, EMBED_DIM)
        emb = self.dropout(emb)

        lstm_out, _ = self.bilstm(emb)             # (B, seq_len, 2*LSTM_HIDDEN)

        # Mask padding tokens before pooling
        # (all-zero embedding rows are padding, but we pool over positions 0..seq_len)
        # Mean pooling
        mean_pool = lstm_out.mean(dim=1)           # (B, 2*LSTM_HIDDEN)
        # Max pooling
        max_pool, _ = lstm_out.max(dim=1)          # (B, 2*LSTM_HIDDEN)

        x = torch.cat([mean_pool, max_pool], dim=1)  # (B, 4*LSTM_HIDDEN)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════
# 5.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for ids, lbls in loader:
            ids  = ids.to(device)
            lbls = lbls.to(device)
            logits = model(ids)
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
    print("  MODEL 2 — Bi-LSTM ONLY")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Data ──
    (train_t, train_l,
     valid_t, valid_l,
     test_t,  test_l) = combine_datasets()

    print("Building vocabulary …")
    vocab = build_vocab(train_t)
    print(f"  Vocabulary size: {len(vocab):,}")

    train_ds = NewsDataset(train_t, train_l, vocab)
    valid_ds = NewsDataset(valid_t, valid_l, vocab)
    test_ds  = NewsDataset(test_t,  test_l,  vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model / optimiser ──
    model     = BiLSTMOnly(len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_f1, patience_cnt = 0.0, 0
    log_lines = [
        "MODEL: Bi-LSTM Only\n",
        f"Hyperparameters: MAX_LEN={MAX_LEN}, BATCH={BATCH_SIZE}, LR={LR}, "
        f"EPOCHS={EPOCHS}, PATIENCE={PATIENCE}, VOCAB_SIZE={VOCAB_SIZE}, "
        f"EMBED_DIM={EMBED_DIM}, LSTM_HIDDEN={LSTM_HIDDEN}, LSTM_LAYERS={LSTM_LAYERS}\n\n"
    ]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()
        for step, (ids, lbls) in enumerate(train_loader, 1):
            ids  = ids.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            logits = model(ids)
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
            print(f"  [SAVED] Best model saved (F1={best_f1:.4f})")
            log_lines.append(f"  [SAVED] Best model saved (F1={best_f1:.4f})\n")
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
        f"FINAL TEST RESULTS — Bi-LSTM Only\n"
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