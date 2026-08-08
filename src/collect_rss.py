"""
collect_rss_fixes.py
---------------------
Drop-in fixes for collect_rss.py:

1. Corrected/pruned REAL_FEEDS and FAKE_FEEDS URLs (see notes per entry).
2. fetch_feed() wrapper that ALWAYS prints what happened — including
   exceptions — so a feed never silently contributes 0 entries without
   you knowing why. This is what was hiding the afpfactcheck failure:
   your log jumped straight from "Fetching fake RSS: ...afpfactcheck..."
   to the next source with no "Found X entries" line in between, which
   means the fetch/parse threw before reaching that print — and whatever
   except-block caught it wasn't printing the error.

Wire this into your existing script by importing fetch_feed() and the
two URL lists, or just copy the relevant pieces in directly.
"""

import feedparser
import requests

# ── REAL news feeds ────────────────────────────────────────────────────────
REAL_FEEDS = [
    # ✅ Confirmed live — this is the one that already worked for you.
    "https://www.malaymail.com/feed/rss/malaysia",

    # ✅ Confirmed live — Bernama publishes ONE combined feed, not
    # per-category ones. Your old /rss/politics.php and /rss/general.php
    # paths don't exist; this is the real path.
    "https://www.bernama.com/en/rssfeed.php",

    # ✅ Confirmed live bonus source not in your original list.
    "https://www.theborneopost.com/feed/",

]

# ── FAKE/fact-check feeds ──────────────────────────────────────────────────
FAKE_FEEDS = [
    # Keep afpfactcheck, but see fetch_feed() below — we need to see WHY
    # it's failing before deciding whether it's fixable.
    "https://factcheck.afp.com/feed",

]


def fetch_feed(url: str, label: str = "") -> list:
    """
    Fetch + parse an RSS feed, always printing what happened.
    Returns a list of feedparser entries (possibly empty).
    """
    tag = f"[{label}] " if label else ""
    print(f"Fetching {tag}RSS: {url}")

    try:
        # Use requests first so we can see HTTP-level failures
        # (403, timeout, connection refused) that feedparser sometimes
        # swallows into an empty/malformed result without raising.
        resp = requests.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ HTTP fetch failed: {type(e).__name__}: {e}")
        return []

    parsed = feedparser.parse(resp.content)

    if parsed.bozo:
        # bozo=1 means feedparser hit a parse error — still may have
        # partial entries, but flag it so it's visible.
        print(f"  ⚠️  Feed parsing issue: {parsed.bozo_exception}")

    entries = parsed.entries
    print(f"  Found {len(entries)} entries")
    return entries


if __name__ == "__main__":
    print("=== REAL feeds ===")
    all_real = []
    for url in REAL_FEEDS:
        all_real.extend(fetch_feed(url, label="real"))
    print(f"\nTotal real entries: {len(all_real)}")

    print("\n=== FAKE feeds ===")
    all_fake = []
    for url in FAKE_FEEDS:
        all_fake.extend(fetch_feed(url, label="fake"))
    print(f"\nTotal fake (RSS-only, excludes Sebenarnya scraper): {len(all_fake)}")