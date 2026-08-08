# app.py
# VeriLens Flask REST API

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import re
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from lime.lime_text import LimeTextExplainer

from model     import load_model
from tokenizer import Tokenizer
from features  import clean_text, get_features_tensor

import requests as req
from newspaper import Article
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_model.pt"
PORT       = int(os.environ.get("PORT", 7860))
DEBUG      = os.environ.get("DEBUG", "false").lower() == "true"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAKE_THRESHOLD = 0.75
REAL_THRESHOLD = 0.40

# Rotate user agents to reduce bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

ensure_model()
print(f"🚀 Starting VeriLens API")
print(f"📂 Model Path: {MODEL_PATH}")
print(f"🖥 Device: {DEVICE}")
print(f"⚖️  Thresholds: FAKE>{FAKE_THRESHOLD} | REAL<{REAL_THRESHOLD} | else UNCERTAIN")

app = Flask(__name__)
CORS(app)

print("📦 Loading model...")
t0    = time.time()
model = load_model(MODEL_PATH, DEVICE)
tok   = Tokenizer(max_length=128)
print(f"✅ Model loaded in {time.time() - t0:.1f}s")

explainer = LimeTextExplainer(class_names=["real", "fake"])


# ── Verdict helper ────────────────────────────────────────────────────────────
# Friendlier, less "technical" wording for end users (sprint-eval feedback:
# the word "Real" read as too technical). `label` is kept unchanged for any
# existing frontend code that switches on it; `display_label` /
# `display_description` are the new user-facing strings.
DISPLAY_LABELS = {
    "FAKE":       {"display_label": "Likely False",   "display_description": "This looks like misinformation."},
    "REAL":       {"display_label": "Likely Genuine",  "display_description": "This looks like credible reporting."},
    "UNCERTAIN":  {"display_label": "Not Sure",         "display_description": "We couldn't confidently verify this one — read carefully."},
}

def _get_verdict(fake_score: float) -> dict:
    if fake_score > FAKE_THRESHOLD:
        verdict = {
            "is_fake":    True,
            "label":      "FAKE",
            "verdict":    "FAKE NEWS",
            "confidence": round(fake_score, 4),
        }
    elif fake_score < REAL_THRESHOLD:
        verdict = {
            "is_fake":    False,
            "label":      "REAL",
            "verdict":    "REAL NEWS",
            "confidence": round(1.0 - fake_score, 4),
        }
    else:
        distance_from_center = abs(fake_score - 0.5)
        verdict = {
            "is_fake":    fake_score >= 0.5,
            "label":      "UNCERTAIN",
            "verdict":    "UNCERTAIN",
            "confidence": round(0.5 + distance_from_center, 4),
        }
    verdict.update(DISPLAY_LABELS[verdict["label"]])
    return verdict


# ── Article extraction ────────────────────────────────────────────────────────
def _extract_article(url: str) -> tuple[str, str, str]:
    """
    Try multiple extraction strategies.
    Returns (title, text, method) — text may be empty if all fail.
    """
    title  = ""
    text   = ""
    method = ""

    # ── Strategy 1: newspaper3k ───────────────────────────────────────────
    for ua in USER_AGENTS:
        try:
            article = Article(url, request_timeout=15, browser_user_agent=ua)
            article.download()
            article.parse()
            if article.text and len(article.text.strip()) > 100:
                return article.title or "", article.text.strip(), "newspaper3k"
        except Exception:
            continue

    # ── Strategy 2: BeautifulSoup with multiple selectors ─────────────────
    for ua in USER_AGENTS:
        try:
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0",
            }
            resp = req.get(url, headers=headers, timeout=15, allow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Get title
            if soup.title:
                title = soup.title.string or ""

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "noscript", "iframe",
                              "figure", ".advertisement", ".ad", ".social-share"]):
                tag.decompose()

            # Try content selectors in priority order
            content_selectors = [
                "article",
                "[itemprop='articleBody']",
                ".article-body",
                ".article-content",
                ".story-body",
                ".entry-content",
                ".post-content",
                ".content-body",
                ".news-body",
                "main",
            ]

            extracted = ""
            for selector in content_selectors:
                container = soup.select_one(selector)
                if container:
                    paragraphs = container.find_all("p")
                    candidate  = " ".join(
                        p.get_text(" ", strip=True)
                        for p in paragraphs
                        if len(p.get_text(strip=True)) > 30  # skip short nav snippets
                    )
                    if len(candidate) > 150:
                        extracted = candidate
                        break

            # Fallback: all <p> tags
            if not extracted:
                paragraphs = soup.find_all("p")
                extracted  = " ".join(
                    p.get_text(" ", strip=True)
                    for p in paragraphs
                    if len(p.get_text(strip=True)) > 30
                )

            if len(extracted) > 100:
                return title, extracted.strip(), "beautifulsoup"

        except Exception:
            continue

    # ── Strategy 3: meta description fallback (very short but better than nothing)
    try:
        headers = {"User-Agent": USER_AGENTS[0]}
        resp    = req.get(url, headers=headers, timeout=10)
        soup    = BeautifulSoup(resp.text, "html.parser")

        meta_desc = (
            soup.find("meta", attrs={"name": "description"}) or
            soup.find("meta", attrs={"property": "og:description"})
        )
        og_title = soup.find("meta", attrs={"property": "og:title"})

        if og_title:
            title = og_title.get("content", "") or ""
        if meta_desc:
            content = meta_desc.get("content", "") or ""
            if len(content) > 50:
                return title, content.strip(), "meta_description"
    except Exception:
        pass

    return title, text, method


# ── Model helpers ─────────────────────────────────────────────────────────────
def _model_predict_batch(texts):
    results = []
    for raw in texts:
        cleaned   = clean_text(raw)
        tokens    = tok.tokenize(cleaned)
        input_ids = tokens["input_ids"].to(DEVICE)
        attn_mask = tokens["attention_mask"].to(DEVICE)
        features  = get_features_tensor(cleaned).to(DEVICE)
        with torch.no_grad():
            logits = model(input_ids, attn_mask, features)
            probs  = F.softmax(logits, dim=1)[0]
        results.append([float(probs[1]), float(probs[0])])
    return np.array(results, dtype=np.float32)


def _get_highlights(text, num_features=20, num_samples=None):
    if num_samples is None:
        word_count = len(text.split())
        num_samples = 100 if word_count < 50 else (200 if word_count < 150 else 300)

    try:
        exp = explainer.explain_instance(
            text, _model_predict_batch,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),
        )
    except Exception as e:
        print(f"⚠️ LIME failed: {e}")
        return []

    word_weights = dict(exp.as_list(label=1))
    if not word_weights:
        return []

    all_scores = [float(np.clip(0.5 + w, 0.0, 1.0)) for w in word_weights.values()]
    if len(all_scores) < 2:
        fake_threshold, real_threshold = 0.52, 0.48
    else:
        mean = np.mean(all_scores)
        std  = np.std(all_scores)
        fake_threshold = min(max(mean + 0.5 * std, 0.51), 0.65)
        real_threshold = max(min(mean - 0.5 * std, 0.49), 0.35)

    highlights = []
    for token in re.findall(r'\S+|\s+', text):
        stripped = token.strip()
        if not stripped:
            highlights.append({"text": token, "score": 0.5, "label": "neutral"})
            continue
        weight     = word_weights.get(stripped.lower(), 0.0)
        fs         = float(np.clip(0.5 + weight, 0.0, 1.0))
        label      = "fake" if fs >= fake_threshold else ("real" if fs <= real_threshold else "neutral")
        highlights.append({"text": token, "score": round(fs, 4), "label": label})

    return highlights


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "app": "VeriLens AI API",
        "status": "running",
        "endpoints": ["/health", "/predict", "/predict_url"],
        "thresholds": {
            "fake":      f"fake_score > {FAKE_THRESHOLD}",
            "real":      f"fake_score < {REAL_THRESHOLD}",
            "uncertain": f"{REAL_THRESHOLD} \u2264 fake_score \u2264 {FAKE_THRESHOLD}",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data     = request.get_json(silent=True)
    raw_text = str(data.get("text", "")).strip() if data else ""
    if not raw_text:
        return jsonify({"error": "Missing or empty field: text"}), 400

    try:
        t_start   = time.time()
        cleaned   = clean_text(raw_text)
        tokens    = tok.tokenize(cleaned)
        input_ids = tokens["input_ids"].to(DEVICE)
        attn_mask = tokens["attention_mask"].to(DEVICE)
        features  = get_features_tensor(cleaned).to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attn_mask, features)
            probs  = F.softmax(logits, dim=1)[0]

        fake_score = float(probs[0])
        real_score = float(probs[1])
        verdict    = _get_verdict(fake_score)
        highlights = _get_highlights(raw_text)
        elapsed_ms = int((time.time() - t_start) * 1000)

        return jsonify({
            "is_fake":            verdict["is_fake"],
            "fake_score":         round(fake_score, 4),
            "real_score":         round(real_score, 4),
            "confidence":         verdict["confidence"],
            "label":              verdict["label"],
            "display_label":      verdict["display_label"],
            "display_description": verdict["display_description"],
            "verdict":            verdict["verdict"],
            "elapsed_ms":         elapsed_ms,
            "highlights":         highlights,
        }), 200

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_url", methods=["POST"])
def predict_url():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json(silent=True)
    url  = str(data.get("url", "")).strip() if data else ""
    if not url:
        return jsonify({"error": "Missing or empty field: url"}), 400

    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(url):
        return jsonify({"error": "Invalid URL format"}), 400

    try:
        t_start = time.time()

        article_title, article_text, extraction_method = _extract_article(url)

        if len(article_text) < 50:
            return jsonify({
                "error": (
                    "Could not extract enough text from this URL. "
                    "The page may be paywalled, JavaScript-rendered, "
                    "or blocking automated access. "
                    "Try copying the article text and pasting it directly."
                )
            }), 422

        text_for_model = article_text[:2000]

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
        verdict    = _get_verdict(fake_score)
        highlights = _get_highlights(text_for_model)
        elapsed_ms = int((time.time() - t_start) * 1000)

        return jsonify({
            "is_fake":                verdict["is_fake"],
            "fake_score":             round(fake_score, 4),
            "real_score":             round(real_score, 4),
            "confidence":             verdict["confidence"],
            "label":                  verdict["label"],
            "display_label":          verdict["display_label"],
            "display_description":    verdict["display_description"],
            "verdict":                verdict["verdict"],
            "elapsed_ms":             elapsed_ms,
            "article_title":          article_title,
            "article_snippet":        article_text[:300],
            "article_text_for_model": text_for_model,
            "source_url":             url,
            "extraction_method":      extraction_method,
            "text_length":            len(article_text),
            "highlights":             highlights,
        }), 200

    except Exception as e:
        app.logger.exception("URL prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)