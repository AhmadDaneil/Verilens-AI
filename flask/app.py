# app.py
# ScamShield Flask REST API
# Endpoint: POST /predict  { "text": "..." }
# Response: { "is_fake": bool, "fake_score": float, "real_score": float, "confidence": float }

import os
import time
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS

from model     import load_model
from tokenizer import Tokenizer
from features  import clean_text, get_features_tensor

from flask import Flask, request, jsonify
import requests as req
from newspaper import Article
from bs4 import BeautifulSoup
import re

import urllib.request
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_URL  = os.environ.get("MODEL_URL", "")
MODEL_PATH = "/tmp/best_model.pt"
PORT       = int(os.environ.get("PORT", 5000))
DEBUG      = os.environ.get("DEBUG", "false").lower() == "true"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Download model if not cached ──────────────────────────────────────────────
def ensure_model():
    # ── Local dev: use existing file if present ───────────────────────────
    local_path = r"C:\dev\python\project\Scamshield-AI Testing\best_model.pt"
    if os.path.exists(local_path):
        print(f"✅ Using local model: {local_path}")
        global MODEL_PATH
        MODEL_PATH = local_path
        return

    # ── Production (Render): download from Hugging Face ───────────────────
    if os.path.exists(MODEL_PATH):
        print("✅ Model already cached")
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL environment variable not set")
    print("⬇️  Downloading model from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded")

ensure_model()

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"📦 Loading model...")
t0    = time.time()
model = load_model(MODEL_PATH, DEVICE)   # <-- MODEL_PATH not CHECKPOINT_PATH
tok   = Tokenizer(max_length=128)
print(f"✅ Model ready in {time.time() - t0:.1f}s")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow Flutter app on any origin

# ── Load model once at startup ────────────────────────────────────────────────
print(f"   Device: {DEVICE}")
t0    = time.time()
model = load_model(MODEL_PATH, DEVICE)
tok   = Tokenizer(max_length=128)
print(f"✅ Model ready in {time.time() - t0:.1f}s")


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)}), 200


# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    # ── 1. Validate request ───────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: text"}), 400

    raw_text = str(data["text"]).strip()
    if not raw_text:
        return jsonify({"error": "text field is empty"}), 400

    try:
        t_start = time.time()

        # ── 2. Preprocess ─────────────────────────────────────────────────────
        cleaned    = clean_text(raw_text)
        tokens     = tok.tokenize(cleaned)
        input_ids  = tokens["input_ids"].to(DEVICE)       # [1, 128]
        attn_mask  = tokens["attention_mask"].to(DEVICE)  # [1, 128]
        features   = get_features_tensor(cleaned).to(DEVICE)  # [1, 6]

        # ── 3. Inference ──────────────────────────────────────────────────────
        with torch.no_grad():
            logits = model(input_ids, attn_mask, features)   # [1, 2]
            probs  = F.softmax(logits, dim=1)[0]             # [2]

        fake_score = float(probs[0])
        real_score = float(probs[1])
        is_fake    = fake_score > real_score
        confidence = fake_score if is_fake else real_score

        elapsed_ms = int((time.time() - t_start) * 1000)

        # ── 4. Response ───────────────────────────────────────────────────────
        return jsonify({
            "is_fake":    is_fake,
            "fake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
            "confidence": round(confidence, 4),
            "label":      "SCAM"    if is_fake else "REAL",
            "elapsed_ms": elapsed_ms,
        }), 200

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500
    
    # ── URL Prediction endpoint ───────────────────────────────────────────────────
@app.route("/predict_url", methods=["POST"])
def predict_url():
    # ── 1. Validate request ───────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing required field: url"}), 400

    url = str(data["url"]).strip()
    if not url:
        return jsonify({"error": "url field is empty"}), 400

    # Basic URL validation
    url_pattern = re.compile(
        r'^https?://'                        # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(url):
        return jsonify({"error": "Invalid URL format"}), 400

    try:
        t_start = time.time()

        # ── 2. Fetch & extract article text ──────────────────────────────────
        article_title    = ""
        article_text     = ""
        extraction_method = ""

        # Method A: newspaper3k (best for news sites)
        try:
            article = Article(url, request_timeout=10)
            article.download()
            article.parse()
            article_title     = article.title or ""
            article_text      = article.text  or ""
            extraction_method = "newspaper3k"
        except Exception:
            pass  # fall through to Method B

        # Method B: BeautifulSoup fallback
        if not article_text.strip():
            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                    )
                }
                resp = req.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                # Remove nav, ads, scripts
                for tag in soup(["script", "style", "nav", "footer",
                                  "header", "aside", "form"]):
                    tag.decompose()

                article_title     = soup.title.string if soup.title else ""
                # Prefer <article> or <main> tags; fall back to <p> tags
                body = soup.find("article") or soup.find("main")
                if body:
                    paragraphs = body.find_all("p")
                else:
                    paragraphs = soup.find_all("p")

                article_text      = " ".join(p.get_text(" ", strip=True)
                                             for p in paragraphs)
                extraction_method = "beautifulsoup"
            except Exception as fetch_err:
                return jsonify({
                    "error": f"Could not fetch URL: {str(fetch_err)}"
                }), 422

        # ── 3. Guard: make sure we have enough text ───────────────────────────
        article_text = article_text.strip()
        if len(article_text) < 50:
            return jsonify({
                "error": "Not enough article text found at this URL. "
                         "The page may be paywalled, JavaScript-rendered, "
                         "or not a news article."
            }), 422

        # Truncate to avoid token overflow (keep first ~2000 chars like a headline + lede)
        text_for_model = article_text[:2000]

        # ── 4. Reuse existing predict pipeline ───────────────────────────────
        cleaned   = clean_text(text_for_model)
        tokens    = tok.tokenize(cleaned)
        input_ids = tokens["input_ids"].to(DEVICE)
        attn_mask = tokens["attention_mask"].to(DEVICE)
        features  = get_features_tensor(cleaned).to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attn_mask, features)
            probs  = F.softmax(logits, dim=1)[0]

        fake_score = float(probs[0])
        real_score = float(probs[1])
        is_fake    = fake_score > real_score
        confidence = fake_score if is_fake else real_score

        elapsed_ms = int((time.time() - t_start) * 1000)

        # ── 5. Response ───────────────────────────────────────────────────────
        return jsonify({
            "is_fake":            is_fake,
            "fake_score":         round(fake_score, 4),
            "real_score":         round(real_score, 4),
            "confidence":         round(confidence, 4),
            "label":              "SCAM" if is_fake else "REAL",
            "elapsed_ms":         elapsed_ms,
            # Bonus metadata — useful to show in Flutter UI
            "article_title":      article_title,
            "article_snippet":    article_text[:300],   # first 300 chars for preview
            "source_url":         url,
            "extraction_method":  extraction_method,
            "text_length":        len(article_text),
        }), 200

    except Exception as e:
        app.logger.exception("URL prediction failed")
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)