"""
Standalone diagnostic for the Malaysian news scraper targets.
No project dependencies -- run directly with: python diagnose_scrapers.py

Prints, for each URL: status code, response length, content-type,
number of <a href> tags found, and a short snippet -- so we can see
exactly what your machine is receiving.
"""

import requests
from bs4 import BeautifulSoup

# A realistic browser header set -- some sites (Bernama especially) are
# known to blackhole requests that don't look like a real browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

TARGETS = {
    "The Star (news/nation)": "https://www.thestar.com.my/news/nation",
    "The Star (sitemap - known dead)": "https://www.thestar.com.my/sitemap/news-sitemap.xml",
    "The Star (Bing News RSS)": "https://www.bing.com/news/search?q=site%3Athestar.com.my&format=rss",
    "FMT (category/nation)": "https://www.freemalaysiatoday.com/category/nation/",
    "FMT (wp-json REST API)": "https://www.freemalaysiatoday.com/wp-json/wp/v2/posts?per_page=5",
    "FMT (wp-sitemap)": "https://www.freemalaysiatoday.com/wp-sitemap.xml",
    "FMT (Bing News RSS)": "https://www.bing.com/news/search?q=site%3Afreemalaysiatoday.com&format=rss",
    "Bing News RSS (control - generic query)": "https://www.bing.com/news/search?q=malaysia&format=rss",
    "Bernama (news.php cat=malaysia)": "https://www.bernama.com/en/news.php?cat=malaysia",
    "Bernama (homepage - baseline)": "https://www.bernama.com/en/",
}


def diagnose(name: str, url: str, timeout: int = 20):
    print("=" * 70)
    print(f"{name}\n  {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
    except requests.exceptions.ConnectTimeout:
        print("  RESULT: connection TIMEOUT (blocked or unreachable)")
        return
    except requests.exceptions.SSLError as e:
        print(f"  RESULT: SSL error -- {e}")
        return
    except requests.exceptions.RequestException as e:
        print(f"  RESULT: request error -- {e}")
        return

    print(f"  status: {resp.status_code}   final_url: {resp.url}")
    print(f"  content-type: {resp.headers.get('Content-Type')}")
    print(f"  content-length: {len(resp.content)} bytes")
    print(f"  server header: {resp.headers.get('Server')}")

    # Basic bot-wall smell test
    lowered = resp.text.lower()
    for marker in ["cloudflare", "captcha", "access denied", "just a moment", "cf-browser-verification"]:
        if marker in lowered:
            print(f"  ⚠️ possible bot-protection marker found: '{marker}'")

    is_rss_like = (
        "xml" in resp.headers.get("Content-Type", "")
        or url.endswith(".xml")
        or "format=rss" in url
        or resp.text.lstrip()[:100].lower().find("<rss") != -1
    )

    if is_rss_like:
        try:
            soup = BeautifulSoup(resp.text, "lxml-xml")
            locs = soup.find_all("loc")
            items = soup.find_all("item")
            links = soup.find_all("link")
            print(f"  <loc> tags: {len(locs)}   <item> tags: {len(items)}   <link> tags: {len(links)}")
            if items:
                first_link = items[0].find("link")
                print(f"  sample item link: {first_link.text.strip() if first_link is not None else '(none)'}")
            elif locs:
                print(f"  sample loc: {locs[0].text.strip()}")
        except Exception as e:
            print(f"  ⚠️ XML parse failed: {e}")
    else:
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)
        print(f"  <a href> tags found: {len(links)}")
        sample_links = [l["href"] for l in links[:5]]
        for s in sample_links:
            print(f"    sample href: {s}")

    print(f"  first 200 chars of body:\n    {resp.text[:200]!r}")


if __name__ == "__main__":
    for name, url in TARGETS.items():
        diagnose(name, url)
        print()