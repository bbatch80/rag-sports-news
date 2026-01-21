"""
News scraper for sports RSS feeds.

Key concepts:
- feedparser: Parses RSS/Atom feeds into Python dicts
- BeautifulSoup: Extracts text content from HTML
- Rate limiting: Be respectful to news sites
"""

import time
from dataclasses import dataclass
from datetime import datetime

import feedparser
import requests
from bs4 import BeautifulSoup


@dataclass
class Article:
    """Structured article data for downstream processing."""
    title: str
    content: str
    url: str
    source: str
    published: str


# RSS feed URLs - these are public feeds from major sports sites
RSS_FEEDS = {
    # CBS Sports - these work well (server-rendered content)
    "cbs_sports": "https://www.cbssports.com/rss/headlines/",
    "cbs_nba": "https://www.cbssports.com/rss/headlines/nba/",
    "cbs_nfl": "https://www.cbssports.com/rss/headlines/nfl/",
    "cbs_ncaam": "https://www.cbssports.com/rss/headlines/college-basketball/",
    # ESPN - often blocked (JavaScript-rendered)
    "espn": "https://www.espn.com/espn/rss/news",
    "espn_nfl": "https://www.espn.com/espn/rss/nfl/news",
    "espn_nba": "https://www.espn.com/espn/rss/nba/news",
    # BBC Sport - international focus
    "bbc_sport": "https://feeds.bbci.co.uk/sport/rss.xml",
}

# Headers needed for some feeds that block basic requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch_feed(feed_url: str) -> list[dict]:
    """
    Parse an RSS feed and return entry metadata.

    feedparser handles the XML parsing and normalizes different
    RSS/Atom formats into a consistent structure.

    Note: We fetch with requests first because some sites block
    direct feedparser requests based on User-Agent.
    """
    # Fetch with proper headers, then parse
    try:
        response = requests.get(feed_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch feed {feed_url}: {e}")
        return []

    feed = feedparser.parse(response.content)

    entries = []
    for entry in feed.entries:
        entries.append({
            "title": entry.get("title", ""),
            "url": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": entry.get("summary", ""),
        })

    return entries


def extract_article_content(url: str) -> str | None:
    """
    Fetch a URL and extract the main article text.

    This is intentionally simple - production scrapers would need
    site-specific extraction rules. We use common patterns that
    work for most news sites.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, and nav elements
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Try common article container patterns
    article = (
        soup.find("article") or
        soup.find("div", class_="article-body") or
        soup.find("div", class_="story-body") or
        soup.find("div", {"id": "article-body"})
    )

    if article:
        paragraphs = article.find_all("p")
    else:
        # Fallback: get all paragraphs from body
        paragraphs = soup.find_all("p")

    # Join paragraph text, filtering out short/empty ones
    text_parts = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if len(text) > 50:  # Skip short paragraphs (likely not content)
            text_parts.append(text)

    return "\n\n".join(text_parts) if text_parts else None


def scrape_feed(feed_name: str, max_articles: int = 10, delay: float = 1.0) -> list[Article]:
    """
    Scrape articles from a named RSS feed.

    Args:
        feed_name: Key from RSS_FEEDS dict
        max_articles: Limit how many articles to fetch (be respectful)
        delay: Seconds between requests (rate limiting)

    Returns:
        List of Article objects with full content
    """
    if feed_name not in RSS_FEEDS:
        raise ValueError(f"Unknown feed: {feed_name}. Available: {list(RSS_FEEDS.keys())}")

    feed_url = RSS_FEEDS[feed_name]
    print(f"Fetching feed: {feed_name}")

    entries = fetch_feed(feed_url)
    print(f"Found {len(entries)} entries, processing up to {max_articles}")

    articles = []
    for entry in entries[:max_articles]:
        print(f"  Fetching: {entry['title'][:50]}...")

        content = extract_article_content(entry["url"])

        if content and len(content) > 200:  # Skip articles with little content
            articles.append(Article(
                title=entry["title"],
                content=content,
                url=entry["url"],
                source=feed_name,
                published=entry["published"],
            ))
            print(f"    ✓ Got {len(content)} chars")
        else:
            print(f"    ✗ Skipped (insufficient content)")

        time.sleep(delay)  # Rate limiting

    return articles


def scrape_all_feeds(max_per_feed: int = 5) -> list[Article]:
    """Scrape articles from all configured feeds."""
    all_articles = []

    for feed_name in RSS_FEEDS:
        try:
            articles = scrape_feed(feed_name, max_articles=max_per_feed)
            all_articles.extend(articles)
            print(f"Got {len(articles)} articles from {feed_name}\n")
        except Exception as e:
            print(f"Error scraping {feed_name}: {e}\n")

    return all_articles


# CLI for testing
if __name__ == "__main__":
    print("=== Sports News Scraper ===\n")

    # Test with just ESPN to start
    articles = scrape_feed("espn", max_articles=3)

    print(f"\n=== Scraped {len(articles)} articles ===\n")

    for i, article in enumerate(articles, 1):
        print(f"{i}. {article.title}")
        print(f"   Source: {article.source}")
        print(f"   URL: {article.url}")
        print(f"   Content preview: {article.content[:200]}...")
        print()
