"""News collection and deterministic enrichment utilities."""

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import requests


@dataclass(frozen=True)
class NewsArticle:
    title: str
    url: str = ""
    source: str = ""
    description: str = ""
    published_at: str | None = None
    tone: float | None = None
    sentiment_label: str | None = None
    symbols: tuple[str, ...] = ()
    market_ids: tuple[str, ...] = ()


DEFAULT_COMPANY_ALIASES = {
    "RELIANCE": ("reliance", "reliance industries", "reliance ind"),
    "TCS": ("tcs", "tata consultancy services"),
    "INFY": ("infy", "infosys"),
    "HDFCBANK": ("hdfc bank", "hdfcbank"),
    "ICICIBANK": ("icici bank", "icicibank"),
    "SBIN": ("sbi", "state bank of india"),
}

INDIAN_LISTING_DIRECTORY = {
    "RELIANCE": {"name": "Reliance Industries Ltd", "exchanges": ["NSE", "BSE"]},
    "TCS": {"name": "Tata Consultancy Services Ltd", "exchanges": ["NSE", "BSE"]},
    "INFY": {"name": "Infosys Ltd", "exchanges": ["NSE", "BSE"]},
    "HDFCBANK": {"name": "HDFC Bank Ltd", "exchanges": ["NSE", "BSE"]},
    "ICICIBANK": {"name": "ICICI Bank Ltd", "exchanges": ["NSE", "BSE"]},
    "SBIN": {"name": "State Bank of India", "exchanges": ["NSE", "BSE"]},
}

EVENT_KEYWORDS = {
    "EARNINGS": ("earnings", "profit", "revenue", "quarterly results", "results"),
    "MERGER_ACQUISITION": ("merger", "acquire", "acquisition", "takeover"),
    "REGULATORY": (" sebi ", " rbi ", "regulator", "regulatory", "penalty", "ban"),
    "MANAGEMENT": ("appoints", "appointed", "resigns", "resignation", "ceo", "cfo"),
    "DIVIDEND": ("dividend", "buyback", "bonus", "split"),
    "CONTRACT": ("contract", "order win", "order worth", "deal", "partnership"),
    "LITIGATION": ("lawsuit", "court", "legal", "investigation", "fraud"),
    "MACRO": ("inflation", "interest rate", "repo rate", "gdp", "fed", "monsoon"),
}

POSITIVE_WORDS = {"beat", "growth", "profit", "surge", "strong", "upgrade", "record", "wins", "positive"}
NEGATIVE_WORDS = {"loss", "fall", "drop", "weak", "downgrade", "fraud", "penalty", "lawsuit", "decline"}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", value.lower())).strip()


def _article(row: NewsArticle | Mapping[str, Any]) -> NewsArticle:
    if isinstance(row, NewsArticle):
        return row
    title = str(row.get("title", "")).strip()
    if not title:
        raise ValueError("news article title is required")
    tone = row.get("tone")
    try:
        parsed_tone = float(tone) if tone is not None else None
    except (TypeError, ValueError):
        parsed_tone = None

    def string_values(value: Any) -> tuple[str, ...]:
        values = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        return tuple(str(item).strip() for item in values if str(item).strip())

    return NewsArticle(
        title=title,
        url=str(row.get("url", row.get("link", ""))),
        source=str(row.get("source", row.get("publisher", ""))),
        description=str(row.get("description", row.get("summary", ""))),
        published_at=str(row["published_at"]) if row.get("published_at") is not None else None,
        tone=parsed_tone,
        sentiment_label=str(row.get("sentiment_label", "")).upper() or None,
        symbols=string_values(row.get("symbols", row.get("symbol"))),
        market_ids=string_values(row.get("market_ids", row.get("market_id"))),
    )


def _fingerprint(article: NewsArticle) -> str:
    value = article.url.strip() or _normalize_text(article.title)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def deduplicate_news(articles: Iterable[NewsArticle | Mapping[str, Any]]) -> list[NewsArticle]:
    """Remove duplicate URLs and normalized headlines, preserving first-seen order."""
    result: list[NewsArticle] = []
    seen: set[str] = set()
    for row in articles:
        article = _article(row)
        fingerprint = _fingerprint(article)
        if fingerprint not in seen:
            seen.add(fingerprint)
            result.append(article)
    return result


def map_companies(title: str, description: str = "", aliases: Mapping[str, Iterable[str]] | None = None) -> list[str]:
    text = f"{title} {description}".lower()
    matches = []
    for symbol, names in (aliases or DEFAULT_COMPANY_ALIASES).items():
        if any(re.search(rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])", text) for name in names):
            matches.append(symbol)
    return matches


def classify_sentiment(title: str, description: str = "") -> dict[str, Any]:
    words = set(_normalize_text(f"{title} {description}").split())
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    score = (positive - negative) / max(positive + negative, 1)
    label = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
    return {"label": label, "score": score}


def classify_event(title: str, description: str = "") -> str:
    text = f" {_normalize_text(f'{title} {description}')} "
    for event, keywords in EVENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return event
    return "GENERAL"


def collect_news(
    records: Iterable[NewsArticle | Mapping[str, Any]],
    aliases: Mapping[str, Iterable[str]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize, deduplicate, and enrich news records from any feed adapter."""
    enriched = []
    for article in deduplicate_news(records):
        companies = map_companies(article.title, article.description, aliases)
        provider_symbols = {symbol.upper() for symbol in article.symbols}
        provider_listing_codes = [
            symbol.upper()
            for symbol in article.symbols
            if symbol.upper() in INDIAN_LISTING_DIRECTORY
        ]
        listing_codes = list(dict.fromkeys([
            *provider_listing_codes,
            *companies,
        ]))
        unmatched_symbols = [
            symbol for symbol in article.symbols
            if symbol.upper() not in INDIAN_LISTING_DIRECTORY
        ]
        listings = [
            {
                "symbol": code,
                **INDIAN_LISTING_DIRECTORY[code],
                "match_type": "provider" if code in provider_symbols else "headline",
            }
            for code in listing_codes
            if code in INDIAN_LISTING_DIRECTORY
        ]
        sentiment = classify_sentiment(article.title, article.description)
        if article.sentiment_label in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
            sentiment = {
                "label": article.sentiment_label,
                "score": {"POSITIVE": 1.0, "NEGATIVE": -1.0, "NEUTRAL": 0.0}[article.sentiment_label],
            }
        elif article.tone is not None:
            tone_score = max(-1.0, min(1.0, article.tone / 10))
            sentiment = {
                "label": "POSITIVE" if article.tone >= 1 else "NEGATIVE" if article.tone <= -1 else "NEUTRAL",
                "score": tone_score,
            }
        enriched.append({
            **asdict(article),
            "id": _fingerprint(article),
            "companies": companies,
            "listings": listings,
            "unmatched_symbols": unmatched_symbols,
            "sentiment": sentiment,
            "event": classify_event(article.title, article.description),
        })
    return enriched


def fetch_news_feed(records: Iterable[NewsArticle | Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Process supplied feed records; network adapters can call this function."""
    if records is None:
        return {"source": "news", "status": "not_implemented"}
    articles = collect_news(records)
    return {"source": "news", "status": "ready", "count": len(articles), "articles": articles}


GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_PROJECT_URL = "https://www.gdeltproject.org/"
NEWS_DATA_API_URL = "https://newsdata.io/api/1/market"
NEWS_DATA_SOURCE_URL = "https://newsdata.io/"
DEFAULT_NEWS_QUERY = "India (stocks OR shares OR business)"
NEWS_CACHE_TTL_SECONDS = 900
NEWS_CACHE_MAX_ITEMS = 64
_NEWS_CACHE: OrderedDict[tuple[str, str, int], tuple[float, dict[str, Any]]] = OrderedDict()


def _published_at(value: Any) -> str | None:
    if not value:
        return None
    raw = str(value)
    try:
        parsed = datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return parsed.isoformat()
    except ValueError:
        return raw


def fetch_market_news(
    query: str = "",
    limit: int = 25,
    newsdata_api_key: str = "",
) -> dict[str, Any]:
    """Fetch recent business news from NewsData.io or fall back to GDELT."""
    search_query = " ".join(query.split()) or DEFAULT_NEWS_QUERY
    if len(search_query) > 100:
        raise ValueError("query must be 100 characters or fewer")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")

    provider = "NewsData.io" if newsdata_api_key else "GDELT"
    cache_key = (provider, search_query.casefold(), limit)
    cached = _NEWS_CACHE.get(cache_key)
    now = time.monotonic()
    if cached:
        _NEWS_CACHE.move_to_end(cache_key)
        if now - cached[0] < NEWS_CACHE_TTL_SECONDS:
            return {**cached[1], "cache_status": "fresh"}

    try:
        if newsdata_api_key:
            response = requests.get(
                NEWS_DATA_API_URL,
                params={
                    "q": search_query,
                    "country": "in",
                    "language": "en",
                    "timezone": "Asia/Kolkata",
                    "size": min(limit, 10),
                    "removeduplicate": "1",
                },
                headers={"X-ACCESS-KEY": newsdata_api_key},
                timeout=8,
            )
        else:
            response = requests.get(
                GDELT_API_URL,
                params={
                    "query": search_query,
                    "mode": "artlist",
                    "format": "json",
                    "maxrecords": limit,
                    "sort": "datedesc",
                    "timespan": "7d",
                },
                timeout=8,
            )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        if cached:
            return {
                **cached[1],
                "cache_status": "stale",
                "cache_warning": f"{provider} is unavailable; showing a previously fetched result.",
            }
        raise
    if newsdata_api_key:
        if payload.get("status") != "success":
            raise requests.RequestException("NewsData.io returned an unsuccessful response")
        raw_articles = payload.get("results", [])
        records = [
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "source": item.get("source_name", ""),
                "description": item.get("description", ""),
                "published_at": item.get("pubDate"),
                "sentiment_label": item.get("sentiment"),
                "symbols": item.get("symbol", []),
                "market_ids": item.get("market_id", []),
            }
            for item in raw_articles
            if isinstance(item, Mapping) and item.get("title")
        ]
    else:
        raw_articles = payload.get("articles", [])
        records = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": item.get("domain", ""),
                "published_at": _published_at(item.get("seendate")),
                "tone": item.get("tone"),
            }
            for item in raw_articles
            if isinstance(item, Mapping) and item.get("title")
        ]
    articles = collect_news(records)
    counts = {
        label.lower(): sum(article["sentiment"]["label"] == label for article in articles)
        for label in ("POSITIVE", "NEGATIVE", "NEUTRAL")
    }
    if counts["positive"] > counts["negative"]:
        overall = "POSITIVE"
    elif counts["negative"] > counts["positive"]:
        overall = "NEGATIVE"
    else:
        overall = "MIXED" if counts["positive"] and counts["negative"] else "NEUTRAL"

    result = {
        "source": provider,
        "source_url": NEWS_DATA_SOURCE_URL if newsdata_api_key else GDELT_PROJECT_URL,
        "status": "ready",
        "query": search_query,
        "count": len(articles),
        "sentiment_counts": counts,
        "headline_bias": overall,
        "articles": articles,
        "notice": "Headline sentiment describes news text; it is not a price forecast or trading recommendation.",
    }
    if newsdata_api_key:
        result["notice"] += " NewsData.io free-tier coverage is delayed by 12 hours."
    _NEWS_CACHE[cache_key] = (time.monotonic(), result)
    _NEWS_CACHE.move_to_end(cache_key)
    while len(_NEWS_CACHE) > NEWS_CACHE_MAX_ITEMS:
        _NEWS_CACHE.popitem(last=False)
    return {**result, "cache_status": "fresh"}
