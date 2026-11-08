#!/usr/bin/env python3
"""
news_tools.py - RSS/Atom news tools for agentic workflows.

* Reads feeds from `feeds.txt` located NEXT TO THIS FILE (hard-coded path).
* One function per category: get_world_news(), get_ai_news(), get_tech..., etc.
* Generic get_news(category), search_news(keywords), list_categories().
* Every function returns plain JSON-serializable dicts (safe to hand to an LLM).
* TOOL_SCHEMAS + dispatch_tool_call() let you plug straight into an LLM tool loop.
* No third-party dependencies (Python 3.8+).

Quick start:
    from news_tools import get_ai_news, get_news, search_news
    result = get_ai_news(limit=5, since_hours=24)
    for a in result["articles"]:
        print(a["title"], a["link"])

CLI:
    python news_tools.py                 # list categories + feed counts
    python news_tools.py ai 5            # 5 latest AI articles
    python news_tools.py search "openai,nvidia"
"""

import html
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config (hard-coded)
# --------------------------------------------------------------------------- #
FEEDS_FILE = Path(__file__).resolve().parent / "feeds.txt"
USER_AGENT = "Mozilla/5.0 (compatible; news_tools/1.0)"
REQUEST_TIMEOUT = 15          # seconds per feed
CACHE_TTL = 300               # seconds; avoids re-fetching in agent loops
MAX_WORKERS = 8
DEFAULT_LIMIT = 10
SUMMARY_CHARS = 300           # keep payloads small for LLM context

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "media": "http://search.yahoo.com/mrss/",
}

_cache = {}  # url -> (timestamp, [articles])


# --------------------------------------------------------------------------- #
# feeds.txt loading
# --------------------------------------------------------------------------- #
def load_feeds():
    """Parse feeds.txt into {category: [urls]}. Raises if the file is missing."""
    if not FEEDS_FILE.exists():
        raise FileNotFoundError(
            f"feeds.txt not found at {FEEDS_FILE}. Put it next to news_tools.py."
        )
    feeds, current = {}, None
    for raw in FEEDS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[([^\]]+)\]", line)
        if m:
            current = m.group(1).strip().lower()
            feeds.setdefault(current, [])
        elif current is not None:
            feeds[current].append(line)
    return feeds


def list_categories():
    """List available news categories and how many feeds each has."""
    feeds = load_feeds()
    return {"categories": {c: len(u) for c, u in feeds.items()}}


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _clean(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _parse_date(value):
    if not value:
        return None
    value = value.strip()
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _text(elem, path):
    found = elem.find(path, NS)
    return found.text.strip() if found is not None and found.text else ""


def _parse_feed(xml_bytes, feed_url):
    root = ET.fromstring(xml_bytes)
    out = []

    channel = root.find("channel")  # RSS 2.0
    if channel is not None:
        source = _text(channel, "title") or feed_url
        for item in channel.findall("item"):
            date = _parse_date(_text(item, "pubDate") or _text(item, "dc:date"))
            media = item.find("media:content", NS)
            out.append({
                "title": _clean(_text(item, "title")),
                "link": _text(item, "link"),
                "summary": _clean(_text(item, "description")),
                "author": _text(item, "dc:creator") or _text(item, "author"),
                "published": date.isoformat() if date else "",
                "image": media.get("url", "") if media is not None else "",
                "source": source,
            })
        return out

    if root.tag.endswith("feed"):  # Atom
        source = _text(root, "atom:title") or feed_url
        for entry in root.findall("atom:entry", NS):
            link = ""
            for l in entry.findall("atom:link", NS):
                if l.get("rel", "alternate") == "alternate":
                    link = l.get("href", "")
                    break
            date = _parse_date(_text(entry, "atom:published") or _text(entry, "atom:updated"))
            author = entry.find("atom:author/atom:name", NS)
            out.append({
                "title": _clean(_text(entry, "atom:title")),
                "link": link,
                "summary": _clean(_text(entry, "atom:summary") or _text(entry, "atom:content")),
                "author": author.text.strip() if author is not None and author.text else "",
                "published": date.isoformat() if date else "",
                "image": "",
                "source": source,
            })
        return out

    raise ValueError("unrecognized feed format")


def _fetch_feed(url):
    """Return (articles, error). Uses a small TTL cache."""
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1], None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            articles = _parse_feed(resp.read(), url)
        _cache[url] = (now, articles)
        return articles, None
    except Exception as exc:
        return [], f"{url}: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Core query logic
# --------------------------------------------------------------------------- #
def _collect(urls, limit, keywords, since_hours, label):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(_fetch_feed, urls))

    errors = [e for _, e in results if e]
    articles, seen = [], set()
    for feed_articles, _ in results:
        for a in feed_articles:
            key = a["link"] or a["title"]
            if key and key not in seen:
                seen.add(key)
                articles.append(dict(a))  # copy so cache stays untouched

    if keywords:
        kws = [k.strip().lower() for k in keywords if k.strip()]
        articles = [
            a for a in articles
            if any(k in (a["title"] + " " + a["summary"]).lower() for k in kws)
        ]

    if since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
        articles = [
            a for a in articles
            if a["published"] and datetime.fromisoformat(a["published"]) >= cutoff
        ]

    articles.sort(key=lambda a: a["published"], reverse=True)
    total = len(articles)
    articles = articles[:limit] if limit else articles

    for a in articles:
        if len(a["summary"]) > SUMMARY_CHARS:
            a["summary"] = a["summary"][:SUMMARY_CHARS].rstrip() + "..."

    return {
        "category": label,
        "count": len(articles),
        "total_matches": total,
        "articles": articles,
        "errors": errors,
    }


def _norm_keywords(keywords):
    """Accept a list or a comma-separated string (LLMs send both)."""
    if not keywords:
        return None
    if isinstance(keywords, str):
        return [k for k in keywords.split(",")]
    return list(keywords)


def get_news(category, limit=DEFAULT_LIMIT, keywords=None, since_hours=None):
    """Get latest news for any category defined in feeds.txt."""
    feeds = load_feeds()
    category = category.strip().lower()
    if category not in feeds:
        return {
            "category": category, "count": 0, "total_matches": 0, "articles": [],
            "errors": [f"Unknown category '{category}'. Available: {sorted(feeds)}"],
        }
    return _collect(feeds[category], limit, _norm_keywords(keywords), since_hours, category)


def search_news(keywords, limit=DEFAULT_LIMIT, since_hours=None, categories=None):
    """Search ALL categories (or a subset) for articles matching any keyword."""
    feeds = load_feeds()
    if categories:
        if isinstance(categories, str):
            categories = [c.strip() for c in categories.split(",")]
        feeds = {c: feeds[c.lower()] for c in categories if c.lower() in feeds}
    urls = list(dict.fromkeys(u for lst in feeds.values() for u in lst))
    return _collect(urls, limit, _norm_keywords(keywords), since_hours, "search")


def get_all_news(limit_per_category=3, since_hours=None):
    """A short briefing: the top N articles from every category."""
    return {
        cat: get_news(cat, limit=limit_per_category, since_hours=since_hours)["articles"]
        for cat in load_feeds()
    }


# --------------------------------------------------------------------------- #
# One function per category
# --------------------------------------------------------------------------- #
CATEGORY_DESCRIPTIONS = {
    "world": "international / world news and geopolitics",
    "us": "United States national news",
    "politics": "politics, government, elections and policy",
    "business": "business news, companies, industry and the economy",
    "finance": "markets, stocks, investing, banking and personal finance",
    "technology": "general technology and gadgets news",
    "ai": "artificial intelligence, machine learning and AI lab announcements",
    "science": "scientific research and discoveries",
    "health": "health, medicine and public health",
    "sports": "sports scores, results and news",
    "entertainment": "movies, TV, music, celebrities and culture",
    "environment": "climate, energy and environmental news",
    "cybersecurity": "security breaches, vulnerabilities, malware and infosec",
    "space": "space exploration, NASA, rockets and astronomy",
    "gaming": "video games and gaming industry news",
    "crypto": "cryptocurrency and blockchain news",
}


def _make_category_function(category, description):
    def fn(limit=DEFAULT_LIMIT, keywords=None, since_hours=None):
        return get_news(category, limit=limit, keywords=keywords, since_hours=since_hours)

    fn.__name__ = f"get_{category}_news"
    fn.__qualname__ = fn.__name__
    fn.__doc__ = (
        f"Get the latest {description}.\n\n"
        "Args:\n"
        "    limit: max articles to return (default 10)\n"
        "    keywords: optional list or comma-separated string; keep articles matching any\n"
        "    since_hours: optional; only articles published in the last N hours"
    )
    return fn


get_world_news = _make_category_function("world", CATEGORY_DESCRIPTIONS["world"])
get_us_news = _make_category_function("us", CATEGORY_DESCRIPTIONS["us"])
get_politics_news = _make_category_function("politics", CATEGORY_DESCRIPTIONS["politics"])
get_business_news = _make_category_function("business", CATEGORY_DESCRIPTIONS["business"])
get_finance_news = _make_category_function("finance", CATEGORY_DESCRIPTIONS["finance"])
get_technology_news = _make_category_function("technology", CATEGORY_DESCRIPTIONS["technology"])
get_ai_news = _make_category_function("ai", CATEGORY_DESCRIPTIONS["ai"])
get_science_news = _make_category_function("science", CATEGORY_DESCRIPTIONS["science"])
get_health_news = _make_category_function("health", CATEGORY_DESCRIPTIONS["health"])
get_sports_news = _make_category_function("sports", CATEGORY_DESCRIPTIONS["sports"])
get_entertainment_news = _make_category_function("entertainment", CATEGORY_DESCRIPTIONS["entertainment"])
get_environment_news = _make_category_function("environment", CATEGORY_DESCRIPTIONS["environment"])
get_cybersecurity_news = _make_category_function("cybersecurity", CATEGORY_DESCRIPTIONS["cybersecurity"])
get_space_news = _make_category_function("space", CATEGORY_DESCRIPTIONS["space"])
get_gaming_news = _make_category_function("gaming", CATEGORY_DESCRIPTIONS["gaming"])
get_crypto_news = _make_category_function("crypto", CATEGORY_DESCRIPTIONS["crypto"])


# --------------------------------------------------------------------------- #
# Agent integration: tool registry, schemas, dispatcher
# --------------------------------------------------------------------------- #
TOOL_FUNCTIONS = {f"get_{c}_news": globals()[f"get_{c}_news"] for c in CATEGORY_DESCRIPTIONS}
TOOL_FUNCTIONS.update({
    "get_news": get_news,
    "search_news": search_news,
    "get_all_news": get_all_news,
    "list_categories": list_categories,
})

_COMMON_PROPS = {
    "limit": {"type": "integer", "description": "Max articles to return (default 10)."},
    "keywords": {
        "type": "array", "items": {"type": "string"},
        "description": "Only keep articles whose title/summary contains any of these.",
    },
    "since_hours": {
        "type": "number",
        "description": "Only articles published within the last N hours.",
    },
}


def tool_schemas():
    """Tool definitions in Anthropic format (name / description / input_schema).
    For OpenAI, wrap each as {"type": "function", "function": {name, description,
    "parameters": input_schema}}."""
    schemas = []
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        schemas.append({
            "name": f"get_{cat}_news",
            "description": f"Get the latest {desc}. Returns articles with title, link, summary, source, published time.",
            "input_schema": {"type": "object", "properties": dict(_COMMON_PROPS)},
        })
    try:
        cats = sorted(load_feeds())
    except FileNotFoundError:
        cats = sorted(CATEGORY_DESCRIPTIONS)
    schemas += [
        {
            "name": "get_news",
            "description": "Get the latest news for any category in feeds.txt (use for categories without a dedicated tool).",
            "input_schema": {
                "type": "object",
                "properties": {"category": {"type": "string", "enum": cats}, **_COMMON_PROPS},
                "required": ["category"],
            },
        },
        {
            "name": "search_news",
            "description": "Search across all news categories for articles matching any keyword.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "limit": _COMMON_PROPS["limit"],
                    "since_hours": _COMMON_PROPS["since_hours"],
                    "categories": {"type": "array", "items": {"type": "string", "enum": cats},
                                   "description": "Optional: restrict search to these categories."},
                },
                "required": ["keywords"],
            },
        },
        {
            "name": "get_all_news",
            "description": "Get a quick briefing: the top few articles from every category.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit_per_category": {"type": "integer", "description": "Default 3."},
                    "since_hours": _COMMON_PROPS["since_hours"],
                },
            },
        },
        {
            "name": "list_categories",
            "description": "List the available news categories.",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    return schemas


TOOL_SCHEMAS = tool_schemas()


def dispatch_tool_call(name, tool_input=None):
    """Run a tool by name with a dict of arguments; returns a JSON string
    (put this straight into your tool_result message)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        return json.dumps(fn(**(tool_input or {})), ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_articles(result):
    for a in result["articles"]:
        print(f"[{a['published'][:16].replace('T', ' ')}] {a['source']}")
        print(f"  {a['title']}\n  {a['link']}\n")
    for e in result["errors"]:
        print(f"[warn] {e}", file=sys.stderr)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(json.dumps(list_categories(), indent=2))
    elif args[0] == "search" and len(args) > 1:
        _print_articles(search_news(args[1], limit=int(args[2]) if len(args) > 2 else 10))
    else:
        _print_articles(get_news(args[0], limit=int(args[1]) if len(args) > 1 else 10))