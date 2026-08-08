"""
test_malaysian_inference.py
------------------------------
Standalone smoke test: loads the fine-tuned best_model.pt and runs it
against a handful of real Malaysian-English news articles, applying the
same tiered verdict logic (_get_verdict) described in app.py:

    fake_score > 0.75           -> FAKE
    fake_score < 0.40           -> REAL
    0.40 <= fake_score <= 0.75  -> UNCERTAIN

This is the test that actually matters: your validation set during
training was drawn from the same narrow source pool as training data
(3 real outlets, Sebenarnya-format fake claims). This script checks
behavior on the kind of input that started the whole project — genuine
Malaysian-English government/news text that the ORIGINAL model wrongly
flagged FAKE.

IMPORTANT — feature normalization stats:
This script loads feature_stats_malaysian.npz, which is only produced
by the UPDATED retrain_with_malaysian_patched.py (previous runs didn't
save it). If that file doesn't exist yet, you're looking at stats from
an old run that didn't persist them — re-run the retrain script once
first so this test uses the exact stats the current best_model.pt was
actually trained with.

Usage:
    python test_malaysian_inference.py
    (edit SAMPLE_ARTICLES below with your own text, or paste real
    examples from the original bug reports)
"""

import re
import numpy as np
import torch
import torch.nn as nn
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── EDIT THIS: real article URLs to test against ──────────────────────────
# The script fetches and extracts these at runtime — same idea as your
# app's own _extract_article(), so this test exercises a realistic path
# instead of hand-typed text. If a site blocks scripted requests (FMT is
# known to, via Cloudflare), fetch_article_text() will return None and
# that entry gets skipped with a warning — paste that one manually into
# a plain string entry instead if you want it included.
#
# v2 — STYLE-TAGGED, EXPANDED: length_stress_test.py found the retrained
# model confidently (85-90%) misclassifies a real, investigative/scandal-
# style article as FAKE regardless of length — while training-set val
# accuracy on Malaysian real content is 97%+. That gap suggests a style/
# topic blind spot (only 236 unique real Malaysian rows total, likely
# skewed toward announcement-style wire content) rather than a remaining
# length-shortcut bug. Added "style" tags below and pulled in verified,
# recent (2026) articles from thestar.com.my / malaymail.com — both
# already represented in the training scrape (collect_real_incremental.py /
# scrape_malaysian_news.py), so STYLE is the controlled variable here, not
# an unfamiliar source domain. Results are broken down by style at the end
# of main() so you can see whether the miss is isolated or systemic.
SAMPLE_URLS = [
    {
        "label_hint": "should be REAL",
        "style": "announcement",
        "url": "https://www.thestar.com.my/news/nation/2026/07/17/network-school-in-forest-city-told-to-shut-down-says-johor-mb",
    },
    {
        "label_hint": "should be REAL",
        "style": "announcement",
        "url": "https://penangtoday.my/news/8720/former-tvb-actor-dickson-lee-finds-peace-in-penang-retirement/",
    },
    {
        "label_hint": "should be REAL (likely blocked — FMT uses bot detection)",
        "style": "announcement",
        "url": "https://www.freemalaysiatoday.com/category/nation/2026/07/17/vote-ph-in-ge16-or-zahid-will-be-pm-says-pua",
    },
    {
        "label_hint": "should be REAL",
        "style": "business/economic",
        "url": "https://www.thestar.com.my/news/nation/2026/07/24/malaysias-economy-increasingly-competitive-and-business-friendly",
    },
    {
        "label_hint": "should be REAL",
        "style": "court/legal",
        "url": "https://www.thestar.com.my/news/nation/2026/06/30/syed-saddiq-arrives-at-federal-court-ahead-of-final-appeal-verdict",
    },
    {
        "label_hint": "should be REAL",
        "style": "investigative/scandal",
        "url": "https://www.malaymail.com/news/malaysia/2025/03/02/macc-chief-defends-investigation-into-malaysiakini-reporter-over-alleged-bribe-says-initially-asked-for-more-than-rm20000/168448",
    },
    {
        "label_hint": "should be REAL",
        "style": "investigative/scandal",
        "url": "https://www.malaymail.com/news/malaysia/2025/12/29/malaysias-courtroom-drama-in-2025-najib-rosmah-dnaa-rulings-and-more/202432",
    },
    # v3 — QUOTE-DENSITY TEST: length_stress_test.py's failing example
    # (Azam Baki) had a question-mark z-score of +2.43 -- far outside the
    # typical range -- and was dense with first-person defensive quotes and
    # rhetorical questions ("Does this mean...?" repeated three times).
    # investigative/scandal as a STYLE tested fine above (both malaymail
    # samples scored confidently REAL), so these two isolate quote-density/
    # rhetorical-question-density specifically as the variable, independent
    # of topic -- both are heavily-quoted political defensive statements,
    # one containing an explicit embedded rhetorical question ("Has the
    # Johor government not helped?").
    {
        "label_hint": "should be REAL",
        "style": "quote_heavy",
        "url": "https://www.malaymail.com/news/malaysia/2026/07/04/anwar-defends-pakatans-johor-push-this-is-about-helping-the-people/226342",
    },
    {
        "label_hint": "should be REAL",
        "style": "quote_heavy",
        "url": "https://www.malaymail.com/news/malaysia/2026/06/07/anwar-says-leadership-requires-patience-as-he-defends-stance-amid-uec-criticism/222858",
    },
]

# Optional: paste one row's text from malaysian_fake_en.csv here as a
# sanity check that the model still catches obvious fakes. Leave as None
# to skip this check.
FAKE_SANITY_CHECK_TEXT = "According to a statement by the Ministry of Finance (MOF): Based on the formula of the Automatic Pricing Mechanism (APM), the retail price of unsubsidized RON95 increased by 5 cents, while the retail price of unsubsidized diesel increased by 10 cents for the period from July 16 to July 22, 2026. The retail price of unsubsidized RON97 was maintained. The price of petroleum products was affected by the movement of the world oil market in the previous week. While the conflict has not reached a final resolution, the price of petroleum products is expected to continue to be hampered by global market uncertainty. While the country's fuel supply remains adequate for the time being, the MADANI Government urges the people to continue to practice fuel consumption in a conservative manner. The retail price of petrol and diesel without subsidies for the period 16 July to 22 July 2026 is as follows: Retail price The retained subsidies of petroleum prices are as follows: Through BUDI MADANI RON95 (BUD195) and BUDI MADANI Diesel (BUDI Diesel), eligible Malaysians can continue to purchase RON95 and diesel subsidized using MyCard, subject to the respective eligibility and monthly eligibility balances.",
# Example — run this in your main env first, then paste one result here:
#   python -c "import pandas as pd; df=pd.read_csv('data/malaysian/malaysian_fake_en.csv'); print(df['text'].iloc[0])"


def fetch_article_text(url: str) -> str | None:
    """Scoped CSS-selector extraction, matching app.py's _extract_article()
    and scrape_malaysian_news.py's fetch_article_text().

    PATCHED: the previous version of this function did a naive
    soup.find_all("p") across the ENTIRE page -- exactly the bug that
    caused the original false-positive this whole investigation started
    from (a thestar.com.my rewards popup's <p> tags getting pulled into
    the "article" text alongside the real content). That fix was already
    applied to collect_real_incremental.py and scrape_malaysian_news.py,
    but this file had drifted out of parity with it. Restored here: try a
    scoped article-body container first, only fall back to page-wide <p>
    tags (with nav/footer/etc stripped) if no scoped container is found.
    """
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                )
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Could not fetch {url}: {type(e).__name__}: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")

    # Strip elements that commonly contain popup/nav/footer text before
    # we even try extraction, same discipline as fetch_article_text() in
    # scrape_malaysian_news.py.
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    for selector in ["article", "[itemprop='articleBody']",
                      ".article-body", ".story-body",
                      ".entry-content", ".content-body", "main"]:
        container = soup.select_one(selector)
        if container:
            paras = container.find_all("p")
            text = " ".join(
                p.get_text(" ", strip=True) for p in paras
                if len(p.get_text(strip=True)) > 30
            )
            if len(text) >= 200:
                return text

    # Fallback: page-wide <p> tags, but nav/footer/popup elements were
    # already stripped above, and short fragments (likely UI chrome, not
    # article prose) are filtered out.
    paras = soup.find_all("p")
    text = " ".join(
        p.get_text(" ", strip=True) for p in paras
        if len(p.get_text(strip=True)) > 30
    )

    if len(text) < 200:
        print(f"  ⚠️  Extracted text suspiciously short ({len(text)} chars) for {url}")
        return None

    return text

# ── Reproduce EXACT preprocessing from retrain_with_malaysian.py ──────────
# v2: added boilerplate stripping (According to,/TRUTH:/SOURCE:/SUMBER:) to
# match retrain_with_malaysian.py and features.py — this script's clean_text
# had fallen out of parity with those after that fix. Doesn't change much for
# the REAL-article samples below (that boilerplate is a Sebenarnya/fake-class
# artifact), but keep it in sync regardless — this exact kind of silent
# drift between "the training script" and "everything downstream that
# reimplements its preprocessing" is what caused the original feature-vector
# train/serve mismatch this project tracked down earlier.
BOILERPLATE_PATTERNS = [
    r"^according to,?\s*",
    r"\btruth:\s*",
    r"\b(?:source|sumber)\s*:\s*",
]


def strip_boilerplate(text):
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = strip_boilerplate(text)
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


# ── Reproduce EXACT model architecture from retrain_with_malaysian.py ─────
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert    = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bilstm  = nn.LSTM(768, 256, num_layers=1,
                                batch_first=True, bidirectional=True)
        self.dropout  = nn.Dropout(0.3)
        self.fc1      = nn.Linear(1024 + 6, 512)
        self.batchnorm= nn.BatchNorm1d(512)
        self.relu     = nn.ReLU()
        self.fc2      = nn.Linear(512, 2)

    def forward(self, input_ids, attention_mask, extra_features):
        outputs       = self.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)
        seq           = outputs.last_hidden_state
        lstm_out, _   = self.bilstm(seq)
        mean_pool     = torch.mean(lstm_out, dim=1)
        max_pool, _   = torch.max(lstm_out, dim=1)
        pooled        = torch.cat((mean_pool, max_pool), dim=1)
        combined      = torch.cat((pooled, extra_features), dim=1)
        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return self.fc2(x)


def get_verdict(fake_score: float) -> str:
    """Mirrors app.py's _get_verdict() tiered thresholds."""
    if fake_score > 0.75:
        return "FAKE"
    elif fake_score < 0.40:
        return "REAL"
    else:
        return "UNCERTAIN"


def main():
    print("Loading tokenizer + model...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model = HybridModel().to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    print("✅ Model loaded\n")

    print("Fetching article text from URLs...")
    sample_articles = []
    for item in SAMPLE_URLS:
        print(f"  {item['url']}")
        text = fetch_article_text(item["url"])
        if text:
            sample_articles.append({
                "label_hint": item["label_hint"],
                "style": item["style"],
                "text": text,
            })
        else:
            print(f"    Skipped (fetch failed) — paste this one manually if needed")

    if FAKE_SANITY_CHECK_TEXT:
        sample_articles.append({
            "label_hint": "should be FAKE (sanity check)",
            "style": "fake_sanity_check",
            "text": FAKE_SANITY_CHECK_TEXT,
        })

    if not sample_articles:
        print("❌ No articles available to test — check fetch errors above.")
        return

    print(f"\n{len(sample_articles)} article(s) ready to test\n")

    try:
        stats = np.load("feature_stats_malaysian.npz")
        feature_mean, feature_std = stats["mean"], stats["std"]
        print("✅ Loaded feature_stats_malaysian.npz (Malaysian fine-tune stats)\n")
    except FileNotFoundError:
        print(
            "⚠️  feature_stats_malaysian.npz not found — this means "
            "best_model.pt was trained BEFORE the patch that saves these "
            "stats. Re-run retrain_with_malaysian_patched.py once so this "
            "test actually matches how the current model was trained. "
            "Falling back to stats computed from these sample articles "
            "only, which is NOT representative and will give unreliable "
            "results.\n"
        )
        feats = np.array([
            extract_features(clean_text(a["text"])) for a in sample_articles
        ])
        feature_mean, feature_std = feats.mean(axis=0), feats.std(axis=0)

    results = []  # (style, label_hint, verdict, fake_score) for the summary below

    for i, article in enumerate(sample_articles, 1):
        text = clean_text(article["text"])

        encoding = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=128, return_tensors="pt",
        )
        features = extract_features(text)
        features = (features - feature_mean) / (feature_std + 1e-8)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(
                encoding["input_ids"].to(device),
                encoding["attention_mask"].to(device),
                features_tensor,
            )
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        fake_score, real_score = probs[0], probs[1]
        verdict = get_verdict(fake_score)
        results.append((article["style"], article["label_hint"], verdict, fake_score))

        print(f"=== Article {i} [{article['style']}] ({article['label_hint']}) ===")
        print(f"  Text (first 100 chars): {article['text'][:100]}...")
        print(f"  Fake score: {fake_score:.4f} | Real score: {real_score:.4f}")
        print(f"  Verdict: {verdict}")
        print()

    # ── Per-style breakdown ─────────────────────────────────────────────────
    # This is the whole point of the v2 expansion: is a REAL misclassification
    # isolated to one style (e.g. investigative/scandal) or systemic across
    # all of them? A style-isolated pattern points at real-class topical
    # diversity gaps in training data; a systemic one points elsewhere
    # (thresholds, remaining preprocessing mismatch, etc).
    print("=" * 65)
    print("BREAKDOWN BY STYLE")
    print("=" * 65)
    by_style = {}
    for style, label_hint, verdict, fake_score in results:
        by_style.setdefault(style, []).append((label_hint, verdict, fake_score))

    for style, rows in by_style.items():
        expected_real = [r for r in rows if "should be REAL" in r[0]]
        if expected_real:
            correct = sum(1 for _, v, _ in expected_real if v == "REAL")
            avg_fake_score = sum(fs for _, _, fs in expected_real) / len(expected_real)
            print(f"  {style:22s} n={len(expected_real)}  "
                  f"correctly REAL: {correct}/{len(expected_real)}  "
                  f"avg fake_score={avg_fake_score:.3f}")
        else:
            for label_hint, verdict, fake_score in rows:
                print(f"  {style:22s} {label_hint} -> {verdict} "
                      f"(fake_score={fake_score:.3f})")


if __name__ == "__main__":
    main()