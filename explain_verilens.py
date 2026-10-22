"""
Verilens -- Explainability (LIME)
==================================
Loads the trained best_model.pt, predicts whether an English news article is
real or fake, and explains WHY by highlighting the words that pushed the
decision toward "real" (green) or "fake" (red). Output is a self-contained
HTML page you can open in a browser, show in a presentation, or embed in the app.

SETUP (once):
    pip install lime

USAGE:
    python explain_verilens.py --text "Paste the article text here ..."
    python explain_verilens.py --file article.txt --title "Headline here"
    python explain_verilens.py --csv data/verilens_clean/test.csv --row 12
    python explain_verilens.py --file article.txt --num-samples 1500 --open

Needs best_model.pt and feature_norm_stats.npz (created by train_verilens.py)
in the folder you run it from, or pass --model-path / --stats-path.

USE FROM YOUR APP:
    from explain_verilens import VerilensExplainer, build_html
    explainer = VerilensExplainer("best_model.pt", "feature_norm_stats.npz")
    result = explainer.explain(article_text, title=headline)
    html_page = build_html(result)

HOW TO READ THE RESULT
    LIME removes random words from the article many times, asks the model
    for its prediction each time, then fits a simple model on those results.
    A word's weight = how much its presence shifted the prediction.
    Green = pushes toward REAL, red = pushes toward FAKE.
    It explains what the model relies on, NOT whether the article is true.
"""

import argparse
import html
import os
import re
import warnings
import webbrowser

import numpy as np

CLASS_NAMES = ["fake", "real"]  # index 0 = fake, 1 = real (same as training)


# ---------------------------------------------------------------------
# Text cleaning + features: IDENTICAL to train_verilens.py.
# Keep these in sync with the training script or explanations will be wrong.
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
    return np.array(
        [text_length, exclamation_count, question_count,
         uppercase_ratio, url_count, digit_ratio],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------
class VerilensExplainer:
    def __init__(self, model_path="best_model.pt",
                 stats_path="feature_norm_stats.npz",
                 max_length=128, batch_size=32):
        # Heavy imports live here so the HTML/LIME helpers can be reused
        # without loading PyTorch.
        import torch
        import torch.nn as nn
        from transformers import DistilBertTokenizer, DistilBertModel

        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"'{stats_path}' not found. It is created by train_verilens.py."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"'{model_path}' not found. It is created by train_verilens.py."
            )
        stats = np.load(stats_path)
        self.feature_mean = stats["mean"]
        self.feature_std = stats["std"]

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        class HybridModel(nn.Module):  # identical architecture to training
            def __init__(self):
                super().__init__()
                self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
                self.bilstm = nn.LSTM(768, 256, num_layers=1,
                                      batch_first=True, bidirectional=True)
                self.dropout = nn.Dropout(0.3)
                self.fc1 = nn.Linear(1024 + 6, 512)
                self.batchnorm = nn.BatchNorm1d(512)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(512, 2)

            def forward(self, input_ids, attention_mask, extra_features):
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                lstm_out, _ = self.bilstm(out.last_hidden_state)
                pooled = torch.cat((lstm_out.mean(dim=1), lstm_out.max(dim=1)[0]), dim=1)
                x = self.dropout(torch.cat((pooled, extra_features), dim=1))
                x = self.relu(self.batchnorm(self.fc1(x)))
                return self.fc2(x)

        self.model = HybridModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_proba(self, texts):
        """list[str] -> array (N, 2) of [P(fake), P(real)]. This is the function LIME calls."""
        torch = self.torch
        probs = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [clean_text(str(t)) for t in texts[i:i + self.batch_size]]
            enc = self.tokenizer(chunk, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            feats = np.array([extract_features(t) for t in chunk])
            feats = (feats - self.feature_mean) / (self.feature_std + 1e-8)
            with torch.no_grad():
                logits = self.model(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                    torch.tensor(feats, dtype=torch.float32).to(self.device),
                )
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def explain(self, text, title="", num_samples=1000, num_features=15, max_words=100):
        return explain_with_lime(self.predict_proba, text, title,
                                 num_samples, num_features, max_words)


# ---------------------------------------------------------------------
# LIME (model-agnostic: only needs a predict_proba function)
# ---------------------------------------------------------------------
def explain_with_lime(predict_proba, text, title="", num_samples=1000,
                      num_features=15, max_words=100):
    from lime.lime_text import LimeTextExplainer

    full_text = (f"{title}. {text}" if title else text).strip()
    if not full_text:
        raise ValueError("Empty article text.")

    # The model only reads ~128 tokens (about 100 words), so only that part
    # can influence the decision and only that part is explained.
    matches = list(re.finditer(r"\S+", full_text))
    if len(matches) > max_words:
        cut = matches[max_words - 1].end()
        analysed, remainder = full_text[:cut], full_text[cut:]
    else:
        analysed, remainder = full_text, ""

    final_probs = predict_proba([full_text])[0]  # verdict = what the app would say
    pred_idx = int(np.argmax(final_probs))

    # Explain the model's LOG-ODDS (log P(real) - log P(fake)) rather than the
    # probability. The model was trained with label smoothing, so its
    # probability flattens out near ~95% and barely moves when single words
    # are removed, which makes probability-based weights collapse to ~0.
    # Log-odds keeps those small differences visible.
    def logodds_fn(texts):
        p = np.asarray(predict_proba(list(texts)))
        z = np.log(p[:, 1] + 1e-9) - np.log(p[:, 0] + 1e-9)
        return np.column_stack([-z, z])

    explainer = LimeTextExplainer(class_names=CLASS_NAMES, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # LIME warns that [-z, z] does not sum to 1; intended
        exp = explainer.explain_instance(
            analysed, logodds_fn, labels=(1,),
            num_features=num_features, num_samples=num_samples,
        )
    # weight > 0 pushes toward REAL, weight < 0 pushes toward FAKE
    weights = {str(w): float(v) for w, v in exp.as_list(label=1)}

    # Fidelity: R^2 of LIME's simple local model vs the real model.
    try:
        fidelity = float(exp.score[1] if isinstance(exp.score, dict) else exp.score)
    except Exception:
        fidelity = None
    n_words = len(matches)

    return {
        "prediction": CLASS_NAMES[pred_idx],
        "prob_fake": float(final_probs[0]),
        "prob_real": float(final_probs[1]),
        "analysed_text": analysed,
        "remainder_text": remainder,
        "weights": weights,
        "num_samples": num_samples,
        "max_words": max_words,
        "fidelity": fidelity,
        "n_words": n_words,
    }


# ---------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------
def _highlight(analysed_text, weights):
    max_abs = max((abs(w) for w in weights.values()), default=1e-9) or 1e-9
    parts = []
    for tok in re.split(r"(\W+)", analysed_text):
        if not tok:
            continue
        w = weights.get(tok)
        safe = html.escape(tok)
        if w is None:
            parts.append(safe)
            continue
        alpha = 0.18 + 0.62 * min(abs(w) / max_abs, 1.0)
        rgb = "34,160,70" if w > 0 else "214,60,60"
        direction = "real" if w > 0 else "fake"
        parts.append(
            f'<span class="w" style="background:rgba({rgb},{alpha:.2f})" '
            f'title="pushes toward {direction} ({w:+.3f})">{safe}</span>'
        )
    return "".join(parts)


def build_html(result):
    pred = result["prediction"]
    conf = result["prob_real"] if pred == "real" else result["prob_fake"]
    weights = result["weights"]
    ranked = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)
    max_abs = max((abs(w) for _, w in ranked), default=1e-9) or 1e-9

    rows = []
    for word, w in ranked:
        width = int(100 * abs(w) / max_abs)
        cls = "real" if w > 0 else "fake"
        rows.append(
            f'<tr><td class="word">{html.escape(word)}</td>'
            f'<td><div class="bar {cls}" style="width:{width}%"></div></td>'
            f'<td class="num">{w:+.3f}</td>'
            f'<td>{"REAL" if w > 0 else "FAKE"}</td></tr>'
        )

    notes = []
    if result.get("n_words", 99) < 20:
        notes.append("This text has fewer than 20 words. The model was trained on full articles of at least 20 words, "
                     "so this prediction is not reliable.")
    fid = result.get("fidelity")
    if fid is not None and fid < 0.5:
        notes.append(f"Low explanation fidelity (R&sup2; = {fid:.2f}): the model's decision is spread across many words, "
                     "so the highlights are only a rough guide.")
    notes_html = "".join(f'<div class="warn">{n}</div>' for n in notes)
    fid_txt = f" Explanation fidelity R&sup2; = {fid:.2f}." if fid is not None else ""

    remainder = result["remainder_text"]
    remainder_html = (
        f'<p class="muted"><b>Not analysed</b> (beyond the first ~{result["max_words"]} words the model reads): '
        f'{html.escape(remainder[:400])}{"..." if len(remainder) > 400 else ""}</p>'
        if remainder else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verilens - Explanation</title>
<style>
:root {{ --bg:#fff; --fg:#1d2330; --card:#f5f6f9; --muted:#667085; --line:#e1e4ea; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14171f; --fg:#e8eaf0; --card:#1e232e; --muted:#9aa3b5; --line:#2c3342; }} }}
body {{ font-family:system-ui,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--fg);
       max-width:860px; margin:2rem auto; padding:0 1rem; line-height:1.7; }}
.verdict {{ border-radius:14px; padding:1.1rem 1.4rem; color:#fff; margin-bottom:1.4rem; }}
.verdict.real {{ background:#1f8a47; }} .verdict.fake {{ background:#c93a3a; }}
.verdict h1 {{ margin:0; font-size:1.6rem; }} .verdict p {{ margin:.2rem 0 0; opacity:.92; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:1rem 1.3rem; margin-bottom:1.2rem; }}
h2 {{ font-size:1.05rem; margin:.1rem 0 .6rem; }}
.w {{ border-radius:4px; padding:1px 3px; }}
.legend span {{ display:inline-block; padding:2px 10px; border-radius:4px; margin-right:.6rem; font-size:.85rem; }}
table {{ width:100%; border-collapse:collapse; font-size:.92rem; }}
td {{ padding:.25rem .4rem; border-bottom:1px solid var(--line); }}
td.word {{ font-weight:600; width:28%; }} td.num {{ text-align:right; color:var(--muted); width:14%; }}
.bar {{ height:10px; border-radius:5px; }} .bar.real {{ background:#22a046; }} .bar.fake {{ background:#d63c3c; }}
.muted {{ color:var(--muted); font-size:.88rem; }}
.warn {{ background:#fff3cd; color:#664d03; border-radius:8px; padding:.6rem .9rem; margin-bottom:1rem; font-size:.9rem; }}
</style></head><body>
<div class="verdict {pred}">
  <h1>Prediction: {pred.upper()}</h1>
  <p>Model confidence {conf*100:.1f}% &nbsp;|&nbsp; P(real) {result["prob_real"]*100:.1f}% &nbsp;|&nbsp; P(fake) {result["prob_fake"]*100:.1f}%</p>
</div>
{notes_html}
<div class="card">
  <h2>Why? Words that drove the decision</h2>
  <p class="legend">
    <span style="background:rgba(34,160,70,.55)">pushes toward REAL</span>
    <span style="background:rgba(214,60,60,.55)">pushes toward FAKE</span>
    <span class="muted">darker = stronger influence. Hover a word for its weight.</span>
  </p>
  <div>{_highlight(result["analysed_text"], weights)}</div>
  {remainder_html}
</div>

<div class="card">
  <h2>Top influential words <span class="muted">(weight = shift in log-odds of REAL vs FAKE)</span></h2>
  <table>{"".join(rows)}</table>
</div>

<p class="muted">Explanation method: LIME ({result["num_samples"]} perturbed samples).{fid_txt} It shows which words the
model relies on to reach its prediction; it does not verify whether the article's claims are true.
Verilens is a decision aid, so please cross-check important claims with trusted sources.</p>
</body></html>"""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Explain a Verilens prediction with LIME.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Article text (in quotes)")
    src.add_argument("--file", help="Path to a .txt file containing the article")
    src.add_argument("--csv", help="CSV with title/text columns (e.g. test.csv); use with --row")
    p.add_argument("--row", type=int, default=0, help="Row index when using --csv")
    p.add_argument("--title", default="", help="Optional headline")
    p.add_argument("--out", default="explanation.html")
    p.add_argument("--num-samples", type=int, default=1000,
                   help="LIME samples. Fewer is faster on CPU; use 2000+ for a final, more stable result.")
    p.add_argument("--num-features", type=int, default=15)
    p.add_argument("--max-words", type=int, default=100)
    p.add_argument("--model-path", default="best_model.pt")
    p.add_argument("--stats-path", default="feature_norm_stats.npz")
    p.add_argument("--open", action="store_true", help="Open the HTML in your browser when done")
    args = p.parse_args()

    title = args.title
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        row = df.iloc[args.row]
        text = str(row["text"])
        title = str(row["title"]) if "title" in df.columns and not title else title
        if "label" in df.columns:
            print(f"True label of row {args.row}: {'real' if int(row['label']) == 1 else 'fake'}")

    print("Loading model ...")
    explainer = VerilensExplainer(args.model_path, args.stats_path)
    print(f"Running LIME with {args.num_samples} samples (a few minutes on CPU) ...")
    result = explainer.explain(text, title=title, num_samples=args.num_samples,
                               num_features=args.num_features, max_words=args.max_words)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_html(result))

    conf = result["prob_real"] if result["prediction"] == "real" else result["prob_fake"]
    print(f"\nPrediction: {result['prediction'].upper()} (confidence {conf*100:.1f}%)")
    print("Top words (+ toward REAL, - toward FAKE):")
    for w, v in sorted(result["weights"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]:
        print(f"  {v:+.4f}  {w}")
    if result["n_words"] < 20:
        print("\nWARNING: fewer than 20 words. The model was trained on full articles (20+ words); "
              "this prediction is not reliable.")
    if result["fidelity"] is not None:
        print(f"\nExplanation fidelity (R^2): {result['fidelity']:.2f}"
              + ("  <- low: highlights are only a rough guide" if result["fidelity"] < 0.5 else ""))
    print(f"\nSaved: {args.out}")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
