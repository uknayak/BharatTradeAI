"""FinBERT inference and freshness/relevance-weighted stock sentiment."""

import math
import re
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.db.database import SessionLocal
from app.services.ai_picks.news import (
    COMPANY_ALIASES,
    NEWS_LOOKBACK,
    _to_utc,
)
from app.services.ai_picks.news_models import AIStockArticleSentiment, AIStockNews
from app.services.market.live_quotes import WATCHLIST
from app.services.sentiment.base import (
    FinancialSentimentModel,
    SentimentModelUnavailable,
    SentimentProbabilities,
)
from app.services.sentiment.registry import DEFAULT_MODEL_ID, get_sentiment_model

FRESHNESS_HALF_LIFE_HOURS = 48.0
_analysis_lock = threading.Lock()


class SentimentAnalysisInProgress(RuntimeError):
    """Raised when another analysis request is already in progress."""


def _article_text(article: AIStockNews) -> str:
    description = (article.description or "").strip()
    return f"{article.title.strip()}. {description}".strip()


def _relevance_score(article: AIStockNews) -> float:
    title = f" {article.title.casefold()} "
    description = f" {(article.description or '').casefold()} "
    aliases = COMPANY_ALIASES.get(article.symbol, ())
    if any(re.search(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])", title) for alias in aliases):
        return 1.0
    if any(
        re.search(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])", description)
        for alias in aliases
    ):
        return 0.85
    return 0.7


def _freshness_weight(published_at: datetime, cutoff: datetime) -> float:
    publication = published_at
    if publication.tzinfo is None:
        publication = publication.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (cutoff - publication.astimezone(timezone.utc)).total_seconds() / 3600)
    return math.exp(-math.log(2) * age_hours / FRESHNESS_HALF_LIFE_HOURS)


def _normalized_probabilities(
    prediction: SentimentProbabilities,
) -> dict[str, float]:
    values = {
        "positive": float(prediction.positive),
        "negative": float(prediction.negative),
        "neutral": float(prediction.neutral),
    }
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise SentimentModelUnavailable("The sentiment model returned invalid probabilities.")
    total = sum(values.values())
    if total <= 0:
        raise SentimentModelUnavailable("The sentiment model returned an empty probability distribution.")
    return {key: value / total for key, value in values.items()}


def _load_recent_data(cutoff: datetime, model_id: str) -> tuple[list[AIStockNews], dict[tuple[str, str], AIStockArticleSentiment]]:
    start = cutoff - NEWS_LOOKBACK
    db = SessionLocal()
    try:
        articles = (
            db.query(AIStockNews)
            .filter(
                AIStockNews.published_at < cutoff,
                AIStockNews.published_at >= start,
            )
            .order_by(AIStockNews.published_at.desc())
            .all()
        )
        identities = {(article.symbol, article.article_id) for article in articles}
        cached: dict[tuple[str, str], AIStockArticleSentiment] = {}
        if identities:
            article_ids = {article_id for _, article_id in identities}
            predictions = (
                db.query(AIStockArticleSentiment)
                .filter(
                    AIStockArticleSentiment.model_id == model_id,
                    AIStockArticleSentiment.article_id.in_(article_ids),
                )
                .all()
            )
            cached = {
                (prediction.symbol, prediction.article_id): prediction
                for prediction in predictions
                if (prediction.symbol, prediction.article_id) in identities
            }
    finally:
        db.close()

    unique_articles: list[AIStockNews] = []
    seen: set[tuple[str, str]] = set()
    seen_urls: set[tuple[str, str]] = set()
    for article in articles:
        identity = (article.symbol, article.article_id)
        url_identity = (article.symbol, article.url.casefold().rstrip("/")) if article.url else None
        if identity in seen or (url_identity is not None and url_identity in seen_urls):
            continue
        seen.add(identity)
        if url_identity is not None:
            seen_urls.add(url_identity)
        unique_articles.append(article)
    return unique_articles, cached


def _persist_predictions(
    articles: list[AIStockNews],
    predictions: list[SentimentProbabilities],
    model_id: str,
    analyzed_at: datetime,
) -> None:
    if len(articles) != len(predictions):
        raise SentimentModelUnavailable("The sentiment model returned an unexpected result count.")

    db = SessionLocal()
    try:
        for article, prediction in zip(articles, predictions):
            probabilities = _normalized_probabilities(prediction)
            label = max(probabilities, key=probabilities.get)
            row = (
                db.query(AIStockArticleSentiment)
                .filter_by(
                    symbol=article.symbol,
                    article_id=article.article_id,
                    model_id=model_id,
                )
                .one_or_none()
            )
            if row is None:
                row = AIStockArticleSentiment(
                    symbol=article.symbol,
                    article_id=article.article_id,
                    model_id=model_id,
                )
                db.add(row)
            row.label = label.upper()
            row.probabilities = probabilities
            row.sentiment_score = probabilities["positive"] - probabilities["negative"]
            row.relevance_score = _relevance_score(article)
            row.analyzed_at = analyzed_at
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()


def _company_summary(
    symbol: str,
    company_name: str,
    articles: list[AIStockNews],
    cached: dict[tuple[str, str], AIStockArticleSentiment],
    cutoff: datetime,
) -> dict[str, Any]:
    weighted = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0
    article_results: list[dict[str, Any]] = []
    event_counts: dict[str, int] = {}

    for article in articles:
        event_counts[article.event_category] = event_counts.get(article.event_category, 0) + 1
        prediction = cached.get((symbol, article.article_id))
        relevance = prediction.relevance_score if prediction else _relevance_score(article)
        freshness = _freshness_weight(article.published_at, cutoff)
        weight = relevance * freshness
        result: dict[str, Any] = {
            "id": article.article_id,
            "title": article.title,
            "description": article.description,
            "url": article.url,
            "source": article.source,
            "provider": article.provider,
            "publishedAt": _utc_iso(article.published_at),
            "retrievedAt": _utc_iso(article.retrieved_at),
            "event": article.event_category,
            "relevanceScore": round(relevance, 4),
            "freshnessWeight": round(freshness, 4),
            "sentiment": None,
            "probabilities": None,
            "sentimentScore": None,
            "analyzedAt": None,
        }
        if prediction is not None:
            probabilities = _normalized_mapping(prediction.probabilities)
            result.update(
                sentiment=prediction.label,
                probabilities=probabilities,
                sentimentScore=round(prediction.sentiment_score, 4),
                analyzedAt=_utc_iso(prediction.analyzed_at),
            )
            for label in weighted:
                weighted[label] += probabilities[label] * weight
            total_weight += weight
        article_results.append(result)

    if total_weight:
        probabilities = {label: round(value / total_weight, 6) for label, value in weighted.items()}
        score = round(probabilities["positive"] - probabilities["negative"], 6)
        sentiment = max(probabilities, key=probabilities.get).upper()
    else:
        probabilities = None
        score = None
        sentiment = "PENDING" if articles else "NO_DATA"

    return {
        "symbol": symbol,
        "companyName": company_name,
        "sentiment": sentiment,
        "sentimentScore": score,
        "probabilities": probabilities,
        "articleCount": len(articles),
        "analyzedArticleCount": sum(article["sentiment"] is not None for article in article_results),
        "eventCounts": event_counts,
        "articles": article_results,
    }


def _normalized_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise SentimentModelUnavailable("A stored sentiment probability record is invalid.")
    probabilities = {
        label: float(value.get(label, 0.0))
        for label in ("positive", "negative", "neutral")
    }
    if any(not math.isfinite(score) or score < 0 for score in probabilities.values()):
        raise SentimentModelUnavailable("A stored sentiment probability record is invalid.")
    total = sum(probabilities.values())
    if total <= 0:
        raise SentimentModelUnavailable("A stored sentiment probability record is empty.")
    return {label: score / total for label, score in probabilities.items()}


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def get_ai_pick_sentiment(
    *, cutoff: datetime | None = None, limit: int = 10, model_id: str = DEFAULT_MODEL_ID
) -> dict[str, Any]:
    cutoff_at = _to_utc(cutoff) if cutoff is not None else datetime.now(timezone.utc)
    articles, cached = _load_recent_data(cutoff_at, model_id)
    by_symbol: dict[str, list[AIStockNews]] = {}
    for article in articles:
        by_symbol.setdefault(article.symbol, []).append(article)

    companies = [
        _company_summary(
            instrument["symbol"],
            instrument["name"],
            by_symbol.get(instrument["symbol"], [])[:limit],
            cached,
            cutoff_at,
        )
        for instrument in WATCHLIST
        if instrument["type"] == "EQUITY"
    ]
    has_articles = any(company["articleCount"] for company in companies)
    incomplete = any(
        company["articleCount"] != company["analyzedArticleCount"]
        for company in companies
        if company["articleCount"]
    )
    analyzed = sum(company["analyzedArticleCount"] for company in companies)
    return {
        "status": "no_data" if not has_articles else "pending" if incomplete else "ready",
        "model": model_id,
        "cutoff": cutoff_at.isoformat(),
        "lookbackDays": NEWS_LOOKBACK.days,
        "freshnessHalfLifeHours": FRESHNESS_HALF_LIFE_HOURS,
        "aggregation": "freshness × company relevance weighted mean of article probabilities",
        "articleCount": sum(company["articleCount"] for company in companies),
        "analyzedArticleCount": analyzed,
        "companies": companies,
        "notice": "FinBERT probabilities describe the tone of published news. They are not stock-return probabilities, a trading recommendation, or a validated stock forecast. Only news published before the cutoff is included.",
    }


def analyze_ai_pick_sentiment(
    *,
    cutoff: datetime | None = None,
    limit: int = 10,
    model_id: str = DEFAULT_MODEL_ID,
    model: FinancialSentimentModel | None = None,
) -> dict[str, Any]:
    """Run the selected financial model for recent articles and persist probabilities."""
    if not _analysis_lock.acquire(blocking=False):
        raise SentimentAnalysisInProgress("A financial sentiment analysis is already running.")
    try:
        cutoff_at = _to_utc(cutoff) if cutoff is not None else datetime.now(timezone.utc)
        articles, cached = _load_recent_data(cutoff_at, model_id)
        by_symbol: dict[str, list[AIStockNews]] = {}
        for article in articles:
            by_symbol.setdefault(article.symbol, []).append(article)
        unique_articles = [
            article
            for instrument in WATCHLIST
            if instrument["type"] == "EQUITY"
            for article in by_symbol.get(instrument["symbol"], [])[:limit]
        ]
        missing = [
            article
            for article in unique_articles
            if (article.symbol, article.article_id) not in cached
        ]
        if missing:
            sentiment_model = model or get_sentiment_model(model_id)
            predictions = sentiment_model.predict([_article_text(article) for article in missing])
            _persist_predictions(missing, predictions, model_id, datetime.now(timezone.utc))
        return get_ai_pick_sentiment(cutoff=cutoff_at, limit=limit, model_id=model_id)
    finally:
        _analysis_lock.release()
