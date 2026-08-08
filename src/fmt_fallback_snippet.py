"""
Drop-in addition for collect_rss.py — scrapes FMT category pages instead of
RSS, since freemalaysiatoday.com returns bot-detection blocks on scripted
RSS/HTTP requests (confirmed independently: even a plain GET to
/feed/ and /feeds/rss/headlines/ gets blocked at the edge, not just feedparser).

Uses newspaper3k's build() to walk category pages and extract articles,
mirroring the same 3-strategy extraction already used in app.py's
_extract_article().
"""

import time
import random
from newspaper import build, Article

FMT_CATEGORIES = [
    "https://www.freemalaysiatoday.com/category/nation/",
    "https://www.freemalaysiatoday.com/category/highlight/",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]


def collect_fmt(max_articles_per_category: int = 40) -> list[dict]:
    collected = []
    for cat_url in FMT_CATEGORIES:
        print(f"Building FMT category: {cat_url}")
        try:
            paper = build(
                cat_url,
                memoize_articles=False,
                browser_user_agent=random.choice(USER_AGENTS),
                request_timeout=15,
            )
        except Exception as e:
            print(f"  Failed to build category page: {e}")
            continue

        print(f"  Found {len(paper.articles)} article links")
        count = 0
        for art in paper.articles:
            if count >= max_articles_per_category:
                break
            try:
                art.download()
                time.sleep(random.uniform(0.5, 1.5))  # polite delay
                art.parse()
                if len(art.text) < 200:
                    continue  # likely a stub/paywall/nav page, skip
                collected.append({
                    "title": art.title,
                    "text": art.text,
                    "url": art.url,
                    "source": "freemalaysiatoday",
                    "label": "real",
                })
                count += 1
            except Exception:
                continue  # skip broken links, don't kill the whole run

        print(f"  Collected {count} articles from this category")

    return collected


if __name__ == "__main__":
    results = collect_fmt()
    print(f"\nTotal FMT collected: {len(results)}")