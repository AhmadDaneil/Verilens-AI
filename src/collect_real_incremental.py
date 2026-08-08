"""
collect_real_incremental.py
-----------------------------
Grows data/malaysian/malaysian_real.csv over time instead of overwriting
it each run. Two strategies combined, because RSS pagination support is
inconsistent and undocumented across these sites:

  1. PAGINATION ATTEMPTS
     Tries common WordPress-style pagination params (?paged=N) on feeds
     that are WordPress-backed (Borneo Post). Custom-built feeds (Bernama's
     rssfeed.php, Malay Mail's feed) likely don't support any pagination
     param at all — the request will just silently return the same latest
     batch. That's fine: the dedupe step below means a repeated/identical
     page costs you nothing, it just won't add new rows.

  2. ACCUMULATION ACROSS RUNS (the reliable fallback)
     News sites publish new articles continuously. Running this script
     daily (or a few times a day) and appending only NEW articles
     (deduped by URL) will grow your real-news corpus steadily even if
     pagination turns out to be a dead end for some/all of these feeds.
     This is the strategy to actually rely on to hit 500 real articles.

  3. FMT fallback merge (newspaper3k scraping, since FMT blocks RSS)
     Reuses collect_fmt() from fmt_fallback_snippet.py if present.

Usage:
    python collect_real_incremental.py
    # run this again tomorrow, and the day after — it appends, not overwrites
"""

import csv
import os
import time
import random
import requests
import feedparser
from bs4 import BeautifulSoup

OUTPUT_CSV = "data/malaysian/malaysian_real.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

MIN_ARTICLE_WORDS = 150  # below this, treat as a failed/partial extraction

# Each entry: (base_url, list_of_paginated_variants_to_also_try)
# Variants beyond the base are speculative — dedupe makes trying them safe.
REAL_FEED_ATTEMPTS = {
    "malaymail": {
        "urls": [
            "https://www.malaymail.com/feed/rss/malaysia",
        ],
    },
    "bernama": {
        "urls": [
            "https://www.bernama.com/en/rssfeed.php",
        ],
    },
    "theborneopost": {
        "urls": [
            "https://www.theborneopost.com/feed/",
            "https://www.theborneopost.com/feed/?paged=2",
            "https://www.theborneopost.com/feed/?paged=3",
        ],
    },
    "thestar": {
        # Confirmed live via thestar.com.my/RSS. Nation section chosen for
        # topical overlap with existing sources; add World/Business/etc.
        # from the same feed index if you want more volume/diversity later.
        "urls": [
            "https://www.thestar.com.my/rss/News/Nation",
        ],
    },
}


# Column names to check, in priority order, in case the existing CSV
# used a different name than "url" for the article link.
URL_COLUMN_CANDIDATES = ["url", "link", "URL", "Link", "article_url", "source_url"]


def _find_url_column(fieldnames) -> str | None:
    for candidate in URL_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate
    return None


def load_existing():
    """Load already-collected URLs so we never store duplicates."""
    seen_urls = set()
    rows = []
    existing_fieldnames = []

    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            url_col = _find_url_column(existing_fieldnames)

            print(f"Existing CSV columns: {existing_fieldnames}")
            if url_col is None:
                print(
                    "  ⚠️  Could not find a URL column among "
                    f"{URL_COLUMN_CANDIDATES} — dedupe will NOT work this "
                    "run and everything will be treated as new. Check your "
                    "CSV's actual column name and add it to "
                    "URL_COLUMN_CANDIDATES above."
                )
            else:
                print(f"  Using '{url_col}' as the URL column for dedupe")

            for row in reader:
                rows.append(row)
                if url_col:
                    seen_urls.add((row.get(url_col) or "").strip())

    if rows:
        sample = list(seen_urls)[:3]
        print(f"  Sample of {len(seen_urls)} seen URLs: {sample}\n")

    return rows, seen_urls, existing_fieldnames


def fetch_feed(url: str) -> list:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ {url} — {type(e).__name__}: {e}")
        return []

    parsed = feedparser.parse(resp.content)
    if parsed.bozo:
        print(f"  ⚠️  Parse issue on {url}: {parsed.bozo_exception}")
    return parsed.entries


def fetch_full_article(url: str) -> str | None:
    """
    Fetches the actual article body from its link, instead of relying on
    the RSS feed's short teaser/summary. This matters because production
    inference (_extract_article() in app.py) sees full article text, not
    RSS snippets — training on snippets alone creates a train/serve
    mismatch where the model has never seen what a real full-length
    article actually looks like.

    v2 — SCOPED SELECTORS (was: blind "grab every <p> tag on the page"):
    That naive approach is what caused test_malaysian_inference.py to
    return thestar.com.my's dismissible "Stay signed in to save your
    points... Nexus rewards" popup instead of the actual article — that
    popup's text lives in <p> tags too, just not inside the article body.
    thestar.com.my is NOT actually paywalled (verified directly — the full
    article is present in the raw HTML), so this was purely an extraction
    scoping bug, not a real access problem. Now mirrors app.py's
    _extract_article() Strategy 2: try likely article-body containers
    first (works for thestar's markup and most other outlets), and only
    fall back to "every <p> tag on the page" if none of those are found.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️  Full-article fetch failed for {url}: {type(e).__name__}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")

    # Strip obvious noise before extracting, so a stray nav/footer <p> can't
    # sneak into even a correctly-scoped container.
    for tag in soup(["script", "style", "nav", "footer", "header",
                      "aside", "form", "noscript", "iframe", "figure"]):
        tag.decompose()

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
    ]
    for selector in content_selectors:
        container = soup.select_one(selector)
        if container:
            paragraphs = container.find_all("p")
            text = " ".join(
                p.get_text(strip=True) for p in paragraphs
                if len(p.get_text(strip=True)) > 30  # skip short nav/promo snippets
            )
            if len(text.split()) >= MIN_ARTICLE_WORDS:
                return text

    # Fallback: no scoped container matched (or it was too short) — try
    # every <p> tag on the page, same as before. Riskier (can pick up
    # popups/promo widgets like thestar's Nexus rewards banner), so this
    # is a last resort, not the first attempt.
    paragraphs = soup.find_all("p")
    text = " ".join(
        p.get_text(strip=True) for p in paragraphs
        if len(p.get_text(strip=True)) > 30
    )

    if len(text.split()) < MIN_ARTICLE_WORDS:
        return None  # too short — likely nav/UI text, not the real article

    return text


def main():
    existing_rows, seen_urls, existing_fieldnames = load_existing()
    print(f"Existing collected: {len(existing_rows)} articles\n")

    new_rows = []

    for source_name, cfg in REAL_FEED_ATTEMPTS.items():
        print(f"=== {source_name} ===")
        source_new = 0
        for url in cfg["urls"]:
            print(f"Fetching: {url}")
            entries = fetch_feed(url)
            print(f"  Found {len(entries)} entries")

            for entry in entries:
                link = entry.get("link", "")
                if not link or link in seen_urls:
                    continue  # duplicate — either re-fetched same page,
                              # or already collected in a previous run

                title = entry.get("title", "")

                full_text = fetch_full_article(link)
                if full_text:
                    text = f"{title}. {full_text}"
                    extraction_method = "full_article"
                else:
                    # Fallback: RSS snippet only. Still saved so we don't
                    # lose the row entirely, but flagged via the
                    # extraction_method column so these can be filtered
                    # out or re-fetched later rather than silently mixed
                    # in with full articles.
                    summary = entry.get("summary", "") or entry.get("description", "")
                    text = f"{title}. {summary}".strip()
                    extraction_method = "snippet_only"

                if len(text) < 100:
                    continue  # too short to be useful training text either way

                new_rows.append({
                    "title": title,
                    "text": text,
                    "url": link,
                    "source": source_name,
                    "extraction_method": extraction_method,
                })
                seen_urls.add(link)
                source_new += 1

                time.sleep(random.uniform(0.5, 1.5))  # polite delay per-article fetch

            time.sleep(random.uniform(0.5, 1.5))  # be polite between requests

        print(f"  → {source_new} new articles from {source_name}\n")

    # ── Optional: merge FMT fallback if that module is available ──────────
    try:
        from fmt_fallback_snippet import collect_fmt
        print("=== freemalaysiatoday (newspaper3k fallback) ===")
        fmt_articles = collect_fmt()
        fmt_new = 0
        for art in fmt_articles:
            if art["url"] in seen_urls:
                continue
            new_rows.append({
                "title": art["title"],
                "text": art["text"],
                "url": art["url"],
                "source": "freemalaysiatoday",
                "extraction_method": "full_article",
            })
            seen_urls.add(art["url"])
            fmt_new += 1
        print(f"  → {fmt_new} new articles from FMT\n")
    except ImportError:
        print("(fmt_fallback_snippet.py not found — skipping FMT fallback)\n")

    # ── Write back: existing + new ─────────────────────────────────────────
    all_rows = existing_rows + new_rows

    # Union of every column seen, old CSV or new rows, preserving old
    # columns first so nothing about the existing schema shifts unexpectedly.
    new_fieldnames = ["title", "text", "url", "source", "extraction_method"]
    fieldnames = list(existing_fieldnames)
    for col in new_fieldnames:
        if col not in fieldnames:
            fieldnames.append(col)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    tmp_path = OUTPUT_CSV + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    # Atomic replace: if anything above failed, the original OUTPUT_CSV
    # is untouched. This is what the earlier crash was missing — that
    # crash truncated OUTPUT_CSV directly and lost the previous 50 rows.
    os.replace(tmp_path, OUTPUT_CSV)

    full_count = sum(1 for r in new_rows if r.get("extraction_method") == "full_article")
    snippet_count = len(new_rows) - full_count
    print(f"✅ Total real articles now: {len(all_rows)} (+{len(new_rows)} this run)")
    print(f"   This run: {full_count} full articles, {snippet_count} snippet-only fallbacks")
    if snippet_count > 0:
        print(
            f"   ⚠️  {snippet_count} rows are snippet-only (full-article fetch "
            f"failed) — consider filtering these out before training with:\n"
            f"   df[df['extraction_method'] == 'full_article']"
        )
    print(f"Saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()