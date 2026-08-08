"""
scrape_malaysian_news.py
------------------------
Scrapes Malaysian real news from The Star, NST, Bernama, FMT
and Malaysian fake/hoax news from Sebenarnya.my (government fact-check portal).

Output: data/malaysian/malaysian_real.csv
        data/malaysian/malaysian_fake.csv

Each CSV has two columns: text, label (1=real, 0=fake)

Usage:
    pip install requests beautifulsoup4 newspaper3k pandas tqdm
    python scrape_malaysian_news.py
"""

import os
import csv
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
import xml.etree.ElementTree as ET
import urllib.parse

os.makedirs("data/malaysian", exist_ok=True)

HEADERS_LIST = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/17.4 Safari/605.1.15",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
    },
]

def get_headers():
    return random.choice(HEADERS_LIST)

def fetch_article_text(url: str, min_length: int = 100) -> str:
    """Try newspaper3k first, then BeautifulSoup fallback."""
    # Method 1: newspaper3k
    try:
        article = Article(url, request_timeout=15,
                          browser_user_agent=get_headers()["User-Agent"])
        article.download()
        article.parse()
        if article.text and len(article.text.strip()) > min_length:
            return article.text.strip()
    except Exception:
        pass

    # Method 2: BeautifulSoup
    try:
        resp = requests.get(url, headers=get_headers(), timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer",
                          "header", "aside", "form", "noscript"]):
            tag.decompose()

        for selector in ["article", "[itemprop='articleBody']",
                          ".article-body", ".story-body",
                          ".entry-content", ".content-body", "main"]:
            container = soup.select_one(selector)
            if container:
                paras = container.find_all("p")
                text  = " ".join(p.get_text(" ", strip=True)
                                 for p in paras if len(p.get_text(strip=True)) > 30)
                if len(text) > min_length:
                    return text.strip()

        # all <p> fallback
        paras = soup.find_all("p")
        text  = " ".join(p.get_text(" ", strip=True)
                         for p in paras if len(p.get_text(strip=True)) > 30)
        if len(text) > min_length:
            return text.strip()
    except Exception:
        pass

    return ""


# ── URL DISCOVERY HELPER (Bing News RSS, for FMT only) ──────────────────────
#
# FMT was rebuilt on Next.js: /category/nation/, /wp-json/, /wp-sitemap.xml
# all 404 to an identical not-found shell, and even the homepage serves
# what looks like a stale cached snapshot rather than live content on a
# plain GET. No reliable direct-access method was found for FMT specifically
# (unlike The Star, which turned out to have a working /news/latest page).
#
# Bing News RSS does return correct, current FMT article URLs -- verified
# directly, e.g. freemalaysiatoday.com/category/nation/2026/08/04/... --
# but it's flaky: some requests get a fake-XML bot-challenge response
# (status 200, content-type: application/xml, but the body is actually JS
# interstitial garbage, not real RSS). retries + a real-RSS sanity check
# (body must start with "<?xml" and contain at least one <item>) work
# around that instead of trusting a single request.

def _unwrap_bing_link(raw_url: str) -> str:
    """Bing News RSS <link> values are wrapped in a apiclick.aspx
    click-tracking redirect with the real destination URL-encoded in a
    `url=` query param -- decode it so we store/fetch the actual article
    URL, not Bing's tracking link."""
    parsed = urllib.parse.urlparse(raw_url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "url" in qs and qs["url"]:
        return qs["url"][0]
    return raw_url  # not wrapped -- already a direct URL


def _discover_fmt_via_bing_news_rss(max_urls: int = 100, retries: int = 3) -> set[str]:
    """Discover recent FMT article URLs via Bing News RSS, with retries
    to work around Bing's occasional fake-XML bot-challenge response."""
    site_domain = "freemalaysiatoday.com"
    rss_url = f"https://www.bing.com/news/search?q=site%3A{site_domain}&format=rss"
    urls = set()

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(rss_url, headers=get_headers(), timeout=15)
            body = resp.text.lstrip()
            if not body.startswith("<?xml"):
                print(f"  ⚠️ Bing News RSS attempt {attempt}: not real RSS (likely bot-challenge), retrying...")
                time.sleep(random.uniform(2.0, 4.0))
                continue

            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            if not items:
                print(f"  ⚠️ Bing News RSS attempt {attempt}: 0 items, retrying...")
                time.sleep(random.uniform(2.0, 4.0))
                continue

            for item in items:
                link_el = item.find("link")
                if link_el is None or not link_el.text:
                    continue
                real_url = _unwrap_bing_link(link_el.text.strip())
                if site_domain in real_url:
                    urls.add(real_url)
                if len(urls) >= max_urls:
                    break
            break  # got real results, no need to retry further
        except Exception as e:
            print(f"  ⚠️ Bing News RSS attempt {attempt} error: {e}")
            time.sleep(random.uniform(2.0, 4.0))

    return urls


# ── REAL NEWS SCRAPERS ────────────────────────────────────────────────────────

def scrape_thestar(max_articles: int = 300) -> list[tuple[str, str]]:
    """Scrape The Star news articles from /news/latest (paginated).

    (Old sitemap/category-page discovery retired: news-sitemap.xml is a
    confirmed 404, and /news/nation's article list is JS/AJAX-rendered,
    not present in static HTML no matter how it's fetched. /news/latest
    is different -- verified server-rendered with real dated article
    links in the raw HTML, paginated via ?pgno=N.)
    """
    print("\n📰 Scraping The Star...")
    texts = []

    # /news/latest is server-rendered with real links in static HTML and
    # paginates via ?pgno=N (confirmed: /news/nation is JS/AJAX-rendered
    # and never contains article links no matter how it's fetched, but
    # /news/latest does -- verified directly, real dated article URLs
    # present in the raw response).
    article_urls = set()
    pages_to_check = max(1, min(10, max_articles // 20 + 1))
    for pgno in range(1, pages_to_check + 1):
        page_url = "https://www.thestar.com.my/news/latest" + (
            f"?pgno={pgno}" if pgno > 1 else ""
        )
        try:
            resp = requests.get(page_url, headers=get_headers(), timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("https://www.thestar.com.my/") and href.count("/") >= 6:
                    # article URLs look like .../<section>/<subsection>/YYYY/MM/DD/slug
                    article_urls.add(href)
                elif href.startswith("/") and href.count("/") >= 5 and any(c.isdigit() for c in href):
                    article_urls.add("https://www.thestar.com.my" + href)
        except Exception as e:
            print(f"  ⚠️ Error {page_url}: {e}")
        if len(article_urls) >= max_articles:
            break
        time.sleep(random.uniform(0.5, 1.5))

    print(f"  Found {len(article_urls)} URLs")

    for url in tqdm(list(article_urls)[:max_articles], desc="The Star"):
        text = fetch_article_text(url)
        if text and len(text) > 150:
            texts.append((url, text))
        time.sleep(random.uniform(1.0, 2.5))  # polite crawling

    print(f"  ✅ Collected {len(texts)} articles")
    return texts


def scrape_bernama(max_articles: int = 300) -> list[tuple[str, str]]:
    """Scrape Bernama (official Malaysian news agency).

    (Old news.php?cat=X scheme retired -- confirmed it now 200s into a
    77-byte meta-refresh redirect straight to the homepage. Current
    category listings live at /en/<section>/ with no query string;
    current article URLs look like /en/general/news.php?id=NNNNNNN.)
    """
    print("\n📰 Scraping Bernama...")
    texts = []

    base_urls = [
        "https://www.bernama.com/en/general/",
        "https://www.bernama.com/en/politics/",
        "https://www.bernama.com/en/business/",
    ]

    article_urls = set()

    for base_url in base_urls:
        try:
            resp = requests.get(base_url, headers=get_headers(), timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)
            for link in links:
                href = link["href"]
                # Current article URLs contain /<section>/news.php?id=NNNN
                if "news.php?id=" in href:
                    if href.startswith("http"):
                        article_urls.add(href)
                    elif href.startswith("/"):
                        article_urls.add("https://www.bernama.com" + href)
        except Exception as e:
            print(f"  ⚠️ Error {base_url}: {e}")

    print(f"  Found {len(article_urls)} URLs")

    for url in tqdm(list(article_urls)[:max_articles], desc="Bernama"):
        text = fetch_article_text(url)
        if text and len(text) > 150:
            texts.append((url, text))
        time.sleep(random.uniform(1.0, 2.5))

    print(f"  ✅ Collected {len(texts)} articles")
    return texts


def scrape_fmt(max_articles: int = 300) -> list[tuple[str, str]]:
    """Scrape Free Malaysia Today, discovered via Bing News RSS.

    (Old category-page scraper retired: FMT rebuilt on Next.js and
    /category/nation/ etc. no longer serve article lists in static HTML
    -- confirmed 404 with an identical not-found shell body across
    /category/nation/, /wp-json/, and /wp-sitemap.xml. Even the homepage
    serves what looks like a stale cached snapshot on a plain GET, so no
    direct-access method was found here -- Bing News RSS is best-effort,
    with retries to handle its occasional bot-challenge response.)
    """
    print("\n📰 Scraping FMT...")
    texts = []

    article_urls = _discover_fmt_via_bing_news_rss(max_urls=max_articles)
    print(f"  Found {len(article_urls)} URLs")

    for url in tqdm(list(article_urls)[:max_articles], desc="FMT"):
        text = fetch_article_text(url)
        if text and len(text) > 150:
            texts.append((url, text))
        time.sleep(random.uniform(1.0, 2.5))

    print(f"  ✅ Collected {len(texts)} articles")
    return texts


def scrape_malaymail(max_articles: int = 200) -> list[tuple[str, str]]:
    """Scrape Malay Mail."""
    print("\n📰 Scraping Malay Mail...")
    texts = []

    base_urls = [
        "https://www.malaymail.com/section/malaysia",
        "https://www.malaymail.com/section/malaysia/government",
        "https://www.malaymail.com/section/malaysia/politics",
    ]

    article_urls = set()

    for base_url in base_urls:
        try:
            resp = requests.get(base_url, headers=get_headers(), timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)
            for link in links:
                href = link["href"]
                if "malaymail.com/news" in href:
                    article_urls.add(href)
        except Exception as e:
            print(f"  ⚠️ Error {base_url}: {e}")

    for url in tqdm(list(article_urls)[:max_articles], desc="Malay Mail"):
        text = fetch_article_text(url)
        if text and len(text) > 150:
            texts.append((url, text))
        time.sleep(random.uniform(1.0, 2.5))

    print(f"  ✅ Collected {len(texts)} articles")
    return texts


# ── FAKE NEWS SCRAPERS ────────────────────────────────────────────────────────

def scrape_sebenarnya(max_articles: int = 500) -> list[str]:
    """
    Scrape Sebenarnya.my — Malaysian government fact-check portal.
    Each entry is a verified FAKE/HOAX claim with explanation.
    We use the hoax claim text as fake samples.
    """
    print("\n🚨 Scraping Sebenarnya.my (fake news)...")
    texts = []

    for page in tqdm(range(1, 50), desc="Sebenarnya.my"):
        try:
            url  = f"https://sebenarnya.my/page/{page}/"
            resp = requests.get(url, headers=get_headers(), timeout=15)

            if resp.status_code == 404:
                break

            soup  = BeautifulSoup(resp.text, "html.parser")
            posts = soup.find_all("article")

            for post in posts:
                # Get the post URL
                link = post.find("a", href=True)
                if not link:
                    continue

                post_url = link["href"]

                try:
                    post_resp = requests.get(
                        post_url, headers=get_headers(), timeout=15)
                    post_soup = BeautifulSoup(post_resp.text, "html.parser")

                    # Get the claim/hoax text — usually in .entry-content
                    content = post_soup.find(
                        class_=["entry-content", "post-content",
                                 "article-content", "td-post-content"])
                    if content:
                        paras = content.find_all("p")
                        text  = " ".join(
                            p.get_text(" ", strip=True)
                            for p in paras[:5]          # first 5 paras = the hoax claim
                            if len(p.get_text(strip=True)) > 30
                        )
                        if len(text) > 100:
                            texts.append(text)

                    time.sleep(random.uniform(0.5, 1.5))

                except Exception:
                    continue

            time.sleep(random.uniform(1.0, 2.0))

            if len(texts) >= max_articles:
                break

        except Exception as e:
            print(f"  ⚠️ Page {page} error: {e}")
            continue

    print(f"  ✅ Collected {len(texts)} fake claims")
    return texts


def scrape_cekfakta(max_articles: int = 200) -> list[str]:
    """Scrape CekFakta Malaysia fact-check site for fake claims."""
    print("\n🚨 Scraping CekFakta (fake news)...")
    texts = []

    for page in tqdm(range(1, 30), desc="CekFakta"):
        try:
            url  = f"https://cekfakta.com/page/{page}/"
            resp = requests.get(url, headers=get_headers(), timeout=15)

            if resp.status_code == 404:
                break

            soup  = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)

            article_urls = [
                l["href"] for l in links
                if "cekfakta.com" in l["href"] and "/20" in l["href"]
            ]

            for article_url in article_urls[:10]:
                text = fetch_article_text(article_url, min_length=80)
                if text:
                    texts.append(text)
                time.sleep(random.uniform(0.5, 1.5))

            if len(texts) >= max_articles:
                break

        except Exception:
            continue

    print(f"  ✅ Collected {len(texts)} fake claims")
    return texts


# ── MAIN ──────────────────────────────────────────────────────────────────────
#
# v2 — MERGE INSTEAD OF OVERWRITE (was a real data-loss bug):
#   The original main() below called `real_df.to_csv("data/malaysian/
#   malaysian_real.csv", index=False)` unconditionally -- that WIPES OUT
#   whatever's already in that file, including everything
#   collect_real_incremental.py has accumulated across runs. It also wrote
#   a different schema (`text, label` only) than what's actually there
#   (`title, text, url, source, extraction_method, was_translated,
#   text_original_ms`) and than what retrain_with_malaysian.py expects
#   (it filters on `extraction_method == "full_article"`, which didn't
#   exist in the old output at all). Fixed below: loads the existing CSV,
#   dedupes new rows by URL against it (same pattern
#   collect_real_incremental.py already uses), and appends with a
#   schema-compatible row shape.
#
#   FAKE-side scraping (scrape_sebenarnya / scrape_cekfakta) is
#   DELIBERATELY NOT WIRED IN below. Both sites are Malay/Indonesian-
#   language fact-check portals -- these functions scrape the raw claim
#   text with no translation step, unlike the NLLB-translation pipeline
#   collect_malay_real_for_translation.py uses for the REAL side. Merging
#   that raw non-English text directly into malaysian_fake_en.csv (whose
#   name promises English) would silently reintroduce a language-
#   consistency bug of the same kind already fixed elsewhere in this
#   project. If you want to use these, translate their output first and
#   merge through a dedicated script mirroring the real-side translation
#   pipeline -- don't just uncomment the block below as-is.
#
#   v3 — SCRAPER FIXES (this pass):
#   The Star, Bernama, and FMT all had drifted from their live site
#   structures since this script was first written (confirmed via a
#   standalone diagnostic run -- see diagnose_scrapers.py):
#     - Bernama: news.php?cat=X now 200s into a meta-refresh redirect to
#       the homepage -- that scheme is retired. Switched to the current
#       /en/<section>/ listing pages and news.php?id=NNNN article pattern.
#       CONFIRMED WORKING in production (6 URLs found, 2-6 new per run).
#     - The Star: news-sitemap.xml is 404, and /news/nation's article
#       list is JS/AJAX-rendered -- confirmed dead no matter how it's
#       fetched, real links never appear in static HTML. Tried Google
#       News RSS first (dead end: links wrapped in a redirect needing JS
#       to resolve, and news.google.com blocks automated access outright
#       per robots.txt) then Bing News RSS (worked but flaky -- Bing
#       intermittently serves a fake-XML bot-challenge instead of real
#       results). Neither was needed in the end: /news/latest turned out
#       to be server-rendered with real dated article links in static
#       HTML, paginated via ?pgno=N. Switched to that -- no third-party
#       dependency, no flakiness. NOT YET CONFIRMED against production,
#       verify Found N URLs > 0 on the next run.
#     - FMT: site rebuilt on Next.js; /category/nation/, /wp-json/, and
#       /wp-sitemap.xml all 404 to an identical not-found shell, and even
#       the homepage serves what looks like a stale cached snapshot on a
#       plain GET. No working direct-access method was found here (unlike
#       The Star). Left on Bing News RSS as best-effort, now with retries
#       and real-RSS validation (body must start with "<?xml" and contain
#       >=1 <item>) to ride out Bing's occasional bot-challenge response,
#       plus decoding the real article URL out of Bing's apiclick.aspx
#       tracking-link wrapper instead of storing/fetching through it.
#       Still the least reliable of the three -- if it keeps coming back
#       empty or thin, the next step is inspecting FMT's actual internal
#       API via browser devtools (Network tab) rather than continuing to
#       lean on a third-party proxy.
#   fetch_article_text() itself was untouched -- it was already using
#   scoped CSS-selector extraction, so no popup-scraping risk here.

import os as _os


REAL_CSV_PATH = "data/malaysian/malaysian_real.csv"
URL_COLUMN_CANDIDATES = ["url", "link", "URL", "Link", "article_url", "source_url"]


def _find_url_column(fieldnames):
    for candidate in URL_COLUMN_CANDIDATES:
        if candidate in fieldnames:
            return candidate
    return None


def _merge_real_results(scraped_by_source: dict) -> None:
    """
    scraped_by_source: {"thestar": [(url, text), ...], "bernama": [...], ...}
    Merges into REAL_CSV_PATH, deduping by URL against what's already there,
    same discipline as collect_real_incremental.py's load_existing()/atomic
    write. Never overwrites existing rows.
    """
    existing_rows = []
    seen_urls = set()
    existing_fieldnames = ["title", "text", "url", "source", "extraction_method",
                            "was_translated", "text_original_ms"]

    if _os.path.exists(REAL_CSV_PATH):
        with open(REAL_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or existing_fieldnames
            url_col = _find_url_column(existing_fieldnames)
            for row in reader:
                existing_rows.append(row)
                if url_col:
                    seen_urls.add((row.get(url_col) or "").strip())

    print(f"\nExisting real-class rows: {len(existing_rows)}")

    new_rows = []
    for source_name, pairs in scraped_by_source.items():
        source_new = 0
        for url, text in pairs:
            if not url or url in seen_urls:
                continue
            new_rows.append({
                "title": "",
                "text": text,
                "url": url,
                "source": source_name,
                "extraction_method": "full_article",  # fetch_article_text()
                                                        # already enforces a
                                                        # min-length floor
                "was_translated": "False",
                "text_original_ms": "",
            })
            seen_urls.add(url)
            source_new += 1
        print(f"  {source_name:12s} -> {source_new} new (of {len(pairs)} scraped)")

    all_rows = existing_rows + new_rows
    fieldnames = list(existing_fieldnames)
    for col in ["title", "text", "url", "source", "extraction_method",
                "was_translated", "text_original_ms"]:
        if col not in fieldnames:
            fieldnames.append(col)

    _os.makedirs(_os.path.dirname(REAL_CSV_PATH), exist_ok=True)
    tmp_path = REAL_CSV_PATH + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    _os.replace(tmp_path, REAL_CSV_PATH)  # atomic, same as collect_real_incremental.py

    print(f"\n✅ Total real articles now: {len(all_rows)} (+{len(new_rows)} this run)")
    print(f"Saved → {REAL_CSV_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("Malaysian News Data Scraper (bulk, real-class only)")
    print("=" * 60)

    scraped_by_source = {
        "thestar_bulk":   scrape_thestar(max_articles=400),
        "bernama_bulk":   scrape_bernama(max_articles=400),
        "fmt_bulk":       scrape_fmt(max_articles=300),
        "malaymail_bulk": scrape_malaymail(max_articles=200),
    }
    # Suffixed "_bulk" to distinguish from collect_real_incremental.py's
    # "thestar"/"bernama"/"fmt"/"malaymail" source tags in the same CSV,
    # since these come from a different scraping strategy (sitemap/section-
    # page bulk scrape vs RSS) even when hitting the same outlet.

    total_scraped = sum(len(v) for v in scraped_by_source.values())
    print(f"\n✅ Total scraped this run (pre-dedupe): {total_scraped}")

    _merge_real_results(scraped_by_source)

    print("\n" + "=" * 60)
    print("DONE. FAKE-class scraping was skipped (see comment above main() —")
    print("scrape_sebenarnya/scrape_cekfakta need a translation step before")
    print("merging into malaysian_fake_en.csv, not wired in here).")
    print("Next step: run retrain_with_malaysian.py once real-class count")
    print("has grown meaningfully.")
    print("=" * 60)