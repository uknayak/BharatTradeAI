"""AI Stock Picks news collection, persistence, and cutoff-aware sentiment."""

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError

from app.data.news import fetch_market_news
from app.db.database import SessionLocal
from app.services.ai_picks.news_models import AIPickNewsSync, AIStockNews, AIStockNewsSnapshot
from app.services.market.live_quotes import WATCHLIST

logger = logging.getLogger(__name__)
NEWS_LOOKBACK = timedelta(days=7)
SYNC_COOLDOWN = timedelta(minutes=15)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
_sync_lock = threading.Lock()
_legacy_archive_lock = threading.Lock()
_legacy_archive_complete = False
_legacy_archive_session_id: int | None = None

COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "RELIANCE": ("reliance", "reliance industries"),
    "HDFCBANK": ("hdfc bank", "hdfcbank"),
    "TCS": ("tcs", "tata consultancy services"),
    "INFY": ("infosys", "infy"),
    "ICICIBANK": ("icici bank", "icicibank"),
    "SBIN": ("state bank of india", "sbi"),
    "ITC": ("itc", "itc limited"),
    "LT": ("larsen and toubro", "larsen & toubro", "l&t"),
    "BHARTIARTL": ("bharti airtel", "airtel"),
    "AXISBANK": ("axis bank", "axisbank"),
}

EVENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("QUARTERLY_RESULTS", ("quarterly results", "quarter results", "q1 results", "q2 results", "q3 results", "q4 results", "earnings", "profit", "revenue")),
    ("MAJOR_ORDER", ("order worth", "wins order", "bags order", "secures order", "contract worth", "major order")),
    ("MERGER_ACQUISITION", ("merger", "acquisition", "acquires", "acquired", "takeover")),
    ("REGULATORY", (" sebi ", " rbi ", "regulatory", "regulator", "penalty", "compliance", "court order")),
    ("GOVERNMENT_POLICY", ("government policy", "government announces", "union budget", "cabinet approves", "ministry announces", "policy change", "import duty")),
    ("SECTOR_DEVELOPMENT", ("sector", "industry outlook", "industry-wide", "industry wide", "market-wide", "market wide")),
)


class NewsSyncCooldownError(Exception):
    """Raised when an AI Stock Picks news refresh is requested too soon."""


class NewsSyncInProgressError(Exception):
    """Raised when an AI Stock Picks news refresh is already running."""


def _db_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("cutoff must include a timezone.")
    return value.astimezone(timezone.utc)


def _parse_published_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=MARKET_TIMEZONE)
    return result.astimezone(timezone.utc)


def _matches_company(symbol: str, title: str, description: str, provider_symbols: Any) -> bool:
    searchable = f" {title} {description} ".casefold()
    for alias in COMPANY_ALIASES.get(symbol, ()):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])", searchable):
            return True
    values = provider_symbols if isinstance(provider_symbols, (list, tuple, set)) else [provider_symbols]
    valid_provider_symbols = {
        symbol,
        f"NSE{symbol}",
        f"BSE{symbol}",
        f"{symbol}NS",
        f"{symbol}BO",
    }
    return any(
        re.sub(r"[^A-Z0-9]", "", str(value).strip().upper()) in valid_provider_symbols
        for value in values
    )


def _event_category(title: str, description: str) -> str:
    content = f" {re.sub(r'[^a-z0-9&]+', ' ', f'{title} {description}'.casefold())} "
    for category, patterns in EVENT_PATTERNS:
        if any(pattern in content for pattern in patterns):
            return category
    return "GENERAL"


def _article_id(title: str) -> str:
    fingerprint = re.sub(r"\s+", " ", title.casefold()).strip()
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _news_rows_for_instrument(
    instrument: dict[str, str], provider_api_key: str
) -> tuple[list[dict[str, Any]], str, str | None]:
    symbol = instrument["symbol"]
    query = f'"{instrument["name"]}" India'
    response = fetch_market_news(query=query, limit=10, newsdata_api_key=provider_api_key)
    provider = str(response.get("source") or "Unknown")
    articles = response.get("articles")
    if not isinstance(articles, list):
        raise ValueError(f"{provider} returned an invalid article list for {symbol}.")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_urls: set[str] = set()
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        if not title:
            continue
        description = str(article.get("description") or "")
        symbols = article.get("symbols", ())
        if not _matches_company(symbol, title, description, symbols):
            continue
        published_at = _parse_published_at(article.get("published_at"))
        if published_at is None:
            continue

        url = str(article.get("url") or "")
        article_id = _article_id(title)
        normalized_url = url.strip().casefold().rstrip("/")
        if article_id in seen or (normalized_url and normalized_url in seen_urls):
            continue
        seen.add(article_id)
        if normalized_url:
            seen_urls.add(normalized_url)
        sentiment = article.get("sentiment")
        if not isinstance(sentiment, dict):
            sentiment = {}
        label = str(sentiment.get("label") or "NEUTRAL").upper()
        if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
            label = "NEUTRAL"
        try:
            score = float(sentiment.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(-1.0, min(1.0, score))
        rows.append(
            {
                "symbol": symbol,
                "article_id": article_id,
                "title": title,
                "description": description,
                "url": url or None,
                "source": str(article.get("source") or provider),
                "provider": provider,
                "published_at": published_at,
                "sentiment_label": label,
                "sentiment_score": score,
                "event_category": _event_category(title, description),
            }
        )
    return rows, provider, response.get("cache_warning")


def _persist_news(rows: list[dict[str, Any]], synced_at: datetime) -> None:
    db = SessionLocal()
    try:
        for item in rows:
            identity = AIStockNews.article_id == item["article_id"]
            if item["url"]:
                identity = or_(identity, AIStockNews.url == item["url"])
            row = db.query(AIStockNews).filter(AIStockNews.symbol == item["symbol"], identity).one_or_none()
            if row is None:
                row = AIStockNews(symbol=item["symbol"], article_id=item["article_id"])
                db.add(row)
            else:
                row.article_id = item["article_id"]
            for key, value in item.items():
                if key != "symbol":
                    setattr(row, key, _db_datetime(value) if key == "published_at" else value)
            row.retrieved_at = _db_datetime(synced_at)

            snapshot_values = {
                key: value
                for key, value in item.items()
                if key not in {"symbol", "article_id"}
            }
            snapshot_values["retrieved_at"] = _db_datetime(synced_at)
            content_hash = hashlib.sha256(
                json.dumps(
                    {
                        key: value.isoformat() if isinstance(value, datetime) else value
                        for key, value in snapshot_values.items()
                        if key != "retrieved_at"
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            snapshot_exists = (
                db.query(AIStockNewsSnapshot.id)
                .filter_by(
                    symbol=item["symbol"],
                    article_id=item["article_id"],
                    content_hash=content_hash,
                )
                .first()
            )
            if snapshot_exists is None:
                db.add(
                    AIStockNewsSnapshot(
                        symbol=item["symbol"],
                        article_id=item["article_id"],
                        content_hash=content_hash,
                        **snapshot_values,
                    )
                )

        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()


def archive_existing_news() -> int:
    """Create immutable retrieval snapshots for news stored before the archive rollout."""
    global _legacy_archive_complete, _legacy_archive_session_id
    with _legacy_archive_lock:
        session_id = id(SessionLocal)
        if _legacy_archive_complete and _legacy_archive_session_id == session_id:
            return 0
        db = SessionLocal()
        try:
            records = db.query(AIStockNews).all()
            legacy_rows = [
                {
                    "symbol": article.symbol,
                    "article_id": article.article_id,
                    "title": article.title,
                    "description": article.description,
                    "url": article.url,
                    "source": article.source,
                    "provider": article.provider,
                    "published_at": article.published_at,
                    "sentiment_label": article.sentiment_label,
                    "sentiment_score": article.sentiment_score,
                    "event_category": article.event_category,
                    "retrieved_at": article.retrieved_at,
                }
                for article in records
            ]
        except SQLAlchemyError:
            logger.exception("Could not read the legacy company-news archive.")
            raise
        finally:
            db.close()

        grouped_rows: dict[datetime, list[dict[str, Any]]] = {}
        for row in legacy_rows:
            retrieved_at = _db_datetime(row.pop("retrieved_at"))
            grouped_rows.setdefault(retrieved_at, []).append(row)
        for retrieved_at, rows in grouped_rows.items():
            _persist_news(rows, retrieved_at)
        _legacy_archive_complete = True
        _legacy_archive_session_id = session_id
        return len(legacy_rows)


def _update_sync_status(
    *, synced_at: datetime, status: str, provider: str | None, warnings: list[str]
) -> None:
    db = SessionLocal()
    try:
        sync = db.query(AIPickNewsSync).filter_by(id=1).one_or_none()
        if sync is None:
            sync = AIPickNewsSync(id=1)
            db.add(sync)
        sync.last_sync_at = _db_datetime(synced_at)
        sync.status = status
        sync.provider = provider
        sync.warning = "\n".join(warnings)[:4000] or None
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()


def sync_ai_pick_news(*, provider_api_key: str = "", manual: bool = True) -> dict[str, Any]:
    """Fetch and persist deduplicated company news via configured market-news sources."""
    if not _sync_lock.acquire(blocking=False):
        raise NewsSyncInProgressError("An AI Stock Picks news sync is already in progress.")
    try:
        synced_at = datetime.now(timezone.utc)
        db = SessionLocal()
        try:
            sync = db.query(AIPickNewsSync).filter_by(id=1).one_or_none()
            last_sync = sync.last_sync_at if sync else None
        finally:
            db.close()
        if manual and last_sync is not None:
            if last_sync.tzinfo is None:
                last_sync = last_sync.replace(tzinfo=timezone.utc)
            if synced_at - last_sync < SYNC_COOLDOWN:
                raise NewsSyncCooldownError("Company news was refreshed recently. Please try again later.")

        warnings: list[str] = []
        providers: set[str] = set()
        synced_symbols: list[str] = []
        provider_rate_limited = False
        for instrument in WATCHLIST:
            if instrument["type"] != "EQUITY":
                continue
            if provider_rate_limited:
                warnings.append(f'{instrument["symbol"]}: skipped after provider rate limit.')
                continue
            try:
                rows, provider, cache_warning = _news_rows_for_instrument(instrument, provider_api_key)
                _persist_news(rows, synced_at)
                providers.add(provider)
                synced_symbols.append(instrument["symbol"])
                if cache_warning:
                    warnings.append(f'{instrument["symbol"]}: {cache_warning}')
            except (requests.RequestException, ValueError, SQLAlchemyError) as exc:
                warnings.append(f'{instrument["symbol"]}: {str(exc) or exc.__class__.__name__}')
                if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429:
                    provider_rate_limited = True

        status = "ok" if not warnings else "partial" if synced_symbols else "error"
        provider_name = ", ".join(sorted(providers)) or (
            "NewsData.io" if provider_api_key else "GDELT"
        )
        _update_sync_status(
            synced_at=synced_at,
            status=status,
            provider=provider_name,
            warnings=warnings,
        )
        return {
            "status": status,
            "provider": provider_name,
            "syncedAt": synced_at.isoformat(),
            "syncedSymbols": synced_symbols,
            "warnings": warnings,
        }
    finally:
        _sync_lock.release()


def archive_news_intelligence_articles(response: dict[str, Any]) -> dict[str, Any]:
    """Persist company-matched News Intelligence articles for point-in-time retrieval."""
    articles = response.get("articles")
    if not isinstance(articles, list):
        raise ValueError("News Intelligence returned an invalid article list.")
    provider = str(response.get("source") or "Unknown")
    retrieved_at = _parse_published_at(response.get("retrieved_at")) or datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        title = str(article.get("title") or "").strip()
        published_at = _parse_published_at(article.get("published_at"))
        if not title or published_at is None:
            continue
        description = str(article.get("description") or "")
        symbols = article.get("symbols", ())
        sentiment = article.get("sentiment")
        sentiment = sentiment if isinstance(sentiment, dict) else {}
        try:
            score = float(sentiment.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(-1.0, min(1.0, score))
        for instrument in WATCHLIST:
            if instrument["type"] != "EQUITY" or not _matches_company(
                instrument["symbol"], title, description, symbols
            ):
                continue
            rows.append(
                {
                    "symbol": instrument["symbol"],
                    "article_id": _article_id(title),
                    "title": title,
                    "description": description,
                    "url": str(article.get("url") or "") or None,
                    "source": str(article.get("source") or provider),
                    "provider": provider,
                    "published_at": published_at,
                    "sentiment_label": str(sentiment.get("label") or "NEUTRAL").upper(),
                    "sentiment_score": score,
                    "event_category": str(article.get("event") or "GENERAL"),
                    "retrieved_at": retrieved_at,
                }
            )
    if rows:
        _persist_news(rows, retrieved_at)
    return {
        "archived_article_count": len({(row["symbol"], row["article_id"]) for row in rows}),
        "archived_symbols": sorted({row["symbol"] for row in rows}),
    }


def get_ai_pick_news(*, cutoff: datetime | None = None, limit: int = 10) -> dict[str, Any]:
    """Return company sentiment and articles published before a prediction cutoff."""
    cutoff_at = _to_utc(cutoff) if cutoff is not None else datetime.now(timezone.utc)
    earliest = cutoff_at - NEWS_LOOKBACK
    db = SessionLocal()
    try:
        records = (
            db.query(AIStockNews)
            .filter(
                AIStockNews.published_at < _db_datetime(cutoff_at),
                AIStockNews.published_at >= _db_datetime(earliest),
            )
            .order_by(AIStockNews.published_at.desc())
            .all()
        )
        sync = db.query(AIPickNewsSync).filter_by(id=1).one_or_none()
    finally:
        db.close()

    by_symbol: dict[str, list[AIStockNews]] = {}
    for record in records:
        by_symbol.setdefault(record.symbol, []).append(record)

    companies: list[dict[str, Any]] = []
    for instrument in WATCHLIST:
        if instrument["type"] != "EQUITY":
            continue
        articles = by_symbol.get(instrument["symbol"], [])
        if not articles:
            score = None
        else:
            score = sum(article.sentiment_score for article in articles) / len(articles)
            score = round(score, 4)
        label = (
            "NO_DATA"
            if score is None
            else "POSITIVE"
            if score > 0.05
            else "NEGATIVE"
            if score < -0.05
            else "NEUTRAL"
        )
        event_counts: dict[str, int] = {}
        for article in articles:
            event_counts[article.event_category] = event_counts.get(article.event_category, 0) + 1
        companies.append(
            {
                "symbol": instrument["symbol"],
                "companyName": instrument["name"],
                "sentiment": label,
                "sentimentScore": score,
                "articleCount": len(articles),
                "eventCounts": event_counts,
                "articles": [
                    {
                        "id": article.article_id,
                        "title": article.title,
                        "description": article.description,
                        "url": article.url,
                        "source": article.source,
                        "provider": article.provider,
                        "publishedAt": _as_utc(article.published_at).isoformat(),
                        "retrievedAt": _as_utc(article.retrieved_at).isoformat(),
                        "sentiment": article.sentiment_label,
                        "sentimentScore": article.sentiment_score,
                        "event": article.event_category,
                    }
                    for article in articles[:limit]
                ],
            }
        )

    return {
        "status": "ready" if records else "no_data",
        "cutoff": cutoff_at.isoformat(),
        "lookbackDays": NEWS_LOOKBACK.days,
        "provider": sync.provider if sync and sync.provider else "Configured news provider",
        "lastSyncedAt": _as_utc(sync.last_sync_at).isoformat() if sync else None,
        "syncStatus": sync.status if sync else "never",
        "syncWarning": sync.warning if sync else None,
        "companies": companies,
        "notice": "Company news and sentiment are descriptive context only, not a forecast or investment recommendation. Only items published before the cutoff are included.",
    }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
