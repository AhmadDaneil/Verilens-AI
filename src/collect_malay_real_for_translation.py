"""
collect_malay_real_for_translation.py
---------------------------------------
Collects genuine (real) news articles written in Bahasa Malaysia, to be
translated to English via NLLB (translate_real_to_english.py) and merged
into malaysian_real.csv.

WHY THIS EXISTS:
    test_translation_fingerprint.py confirmed the model learned
    "translated text -> fake" as a shortcut: 80.0% accuracy on native
    ISOT fake news vs 99.78% on translated Malaysian fake news. This
    happened because translation was 100% correlated with the fake
    label — every fake example was translated, zero real examples were.
    The fix: add translated REAL examples too, so "was this translated"
    stops predicting the label.

Mirrors collect_real_incremental.py's structure: feed dict, atomic
write via os.replace, extraction_method tagging (full_article vs
snippet_only), URL-based dedupe.

IMPORTANT — VERIFY FEED URLS BEFORE RUNNING:
    The BM (Bahasa Malaysia) feed path below is a best guess based on
    the pattern of your confirmed-live EN feed
    (bernama.com/en/rssfeed.php -> bernama.com/bm/rssfeed.php). It is
    NOT independently confirmed the way your EN feeds were. Test first:
        curl -I https://www.bernama.com/bm/rssfeed.php
    Malay Mail is primarily an English-language outlet — it's omitted
    here since it likely has no meaningful BM feed. If Bernama BM
    doesn't yield enough volume alone, consider adding Berita Harian or
    Utusan Malaysia feeds (unverified paths, add if you have them).

Usage:
    python collect_malay_real_for_translation.py
    # run again later to accumulate more, same as collect_real_incremental.py
"""

import csv
import os
import time
import random
import requests
import feedparser
from bs4 import BeautifulSoup

OUTPUT_CSV = "data/malaysian/malaysian_real_ms_raw.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

MIN_ARTICLE_WORDS = 150  # same threshold as collect_real_incremental.py

# ── VERIFY BEFORE RUNNING (see note above) ─────────────────────────────────
MALAY_FEED_ATTEMPTS = {
    "sinarharian-berita": {
        "urls": [
            "https://www.sinarharian.com.my/rssFeed/211",
        ],
    },
    "sinarharian-latest": {
        "urls": [
            "https://www.sinarharian.com.my/rssFeed/0",
        ],
    },
    "bernama-bm": {
        "urls": [
            "https://www.bernama.com/bm/rssfeed.php",  # currently 500ing — kept in case transient
        ],
    },
}

URL_COLUMN_CANDIDATES = ["url", "link", "URL", "Link", "article_url", "source_url"]


def _find_url_column(fieldnames):
    for candidate in URL_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate
    return None


def load_existing():
    seen_urls = set()
    rows = []
    existing_fieldnames = []

    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            url_col = _find_url_column(existing_fieldnames)
            for row in reader:
                rows.append(row)
                if url_col:
                    seen_urls.add((row.get(url_col) or "").strip())

    return rows, seen_urls, existing_fieldnames


def fetch_feed(url: str) -> list:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ {url} — {type(e).__name__}: {e}")
        return []

    parsed = feedparser.parse(resp.content)
    if parsed.bozo:
        print(f"  ⚠️  Parse issue on {url}: {parsed.bozo_exception}")
    return parsed.entries


def fetch_full_article(url: str):
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️  Full-article fetch failed for {url}: {type(e).__name__}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    scope = soup.find("article") or soup
    paragraphs = scope.find_all("p")
    text = " ".join(p.get_text(strip=True) for p in paragraphs)

    if len(text.split()) < MIN_ARTICLE_WORDS:
        return None
    return text


def main():
    existing_rows, seen_urls, existing_fieldnames = load_existing()
    print(f"Existing collected: {len(existing_rows)} articles\n")

    new_rows = []

    for source_name, cfg in MALAY_FEED_ATTEMPTS.items():
        print(f"=== {source_name} ===")
        source_new = 0
        for url in cfg["urls"]:
            print(f"Fetching: {url}")
            entries = fetch_feed(url)
            print(f"  Found {len(entries)} entries")
            if len(entries) == 0:
                print(f"  ⚠️  Zero entries — feed URL likely wrong/dead. "
                      f"Verify manually with curl -I {url}")

            for entry in entries:
                link = entry.get("link", "")
                if not link or link in seen_urls:
                    continue

                title = entry.get("title", "")
                full_text = fetch_full_article(link)

                if full_text:
                    text = f"{title}. {full_text}"
                    extraction_method = "full_article"
                else:
                    summary = entry.get("summary", "") or entry.get("description", "")
                    text = f"{title}. {summary}".strip()
                    extraction_method = "snippet_only"

                if len(text) < 100:
                    continue

                new_rows.append({
                    "title": title,
                    "text": text,
                    "url": link,
                    "source": source_name,
                    "extraction_method": extraction_method,
                })
                seen_urls.add(link)
                source_new += 1
                time.sleep(random.uniform(0.5, 1.5))

            time.sleep(random.uniform(0.5, 1.5))

        print(f"  → {source_new} new articles from {source_name}\n")

    all_rows = existing_rows + new_rows
    fieldnames = list(existing_fieldnames) if existing_fieldnames else \
        ["title", "text", "url", "source", "extraction_method"]
    for col in ["title", "text", "url", "source", "extraction_method"]:
        if col not in fieldnames:
            fieldnames.append(col)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    tmp_path = OUTPUT_CSV + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp_path, OUTPUT_CSV)  # atomic — same fix as collect_rss.py's crash bug

    full_count = sum(1 for r in new_rows if r.get("extraction_method") == "full_article")
    print(f"✅ Total collected: {len(all_rows)} (+{len(new_rows)} this run, "
          f"{full_count} full_article)")
    print(f"Saved → {OUTPUT_CSV}")

    if len(all_rows) == 0:
        print("\n❌ Zero articles collected. The BM feed URL is unverified — "
              "check it manually before assuming this script is broken.")


if __name__ == "__main__":
    main()