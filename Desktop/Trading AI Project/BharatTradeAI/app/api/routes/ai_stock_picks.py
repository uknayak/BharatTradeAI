from datetime import datetime

import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.ai_picks.news import (
    NewsSyncCooldownError,
    NewsSyncInProgressError,
    get_ai_pick_news,
    sync_ai_pick_news,
)
from app.services.ai_picks.sentiment_analysis import (
    SentimentAnalysisInProgress,
    analyze_ai_pick_sentiment,
    get_ai_pick_sentiment,
)
from app.services.ai_picks.prediction import (
    PredictionModelUnavailable,
    PredictionTrainingInProgress,
    get_stock_direction_prediction,
    get_stock_direction_predictions,
    get_stock_direction_watchlist_job,
    run_stock_direction_watchlist_job,
    start_stock_direction_watchlist_job,
    train_stock_direction_model,
)
from app.services.ai_picks.selection import build_ai_stock_opportunities
from app.services.sentiment.base import SentimentModelUnavailable
from config.settings import settings

router = APIRouter(prefix="/api/v1/ai-stock-picks", tags=["AI Stock Picks"])


def _parse_cutoff(cutoff: str | None) -> datetime | None:
    try:
        cutoff_at = datetime.fromisoformat(cutoff.replace("Z", "+00:00")) if cutoff else None
        if cutoff_at is not None and cutoff_at.utcoffset() is None:
            raise ValueError("cutoff must include a timezone.")
        return cutoff_at
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Cutoff must be an ISO 8601 datetime that includes a timezone.",
        ) from exc


def _company_response(response: dict, symbol: str | None) -> dict:
    if not symbol:
        return response
    normalized_symbol = symbol.upper()
    company = next(
        (item for item in response["companies"] if item["symbol"] == normalized_symbol),
        None,
    )
    if company is None:
        raise HTTPException(status_code=404, detail=f"Unsupported market symbol: {symbol}.")
    response["companies"] = [company]
    return response


@router.get("/news")
def ai_stock_pick_news(
    symbol: str | None = Query(default=None, max_length=50),
    cutoff: str | None = Query(default=None, max_length=50),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict:
    response = get_ai_pick_news(cutoff=_parse_cutoff(cutoff), limit=limit)
    return _company_response(response, symbol)


@router.post("/news/sync")
def sync_ai_stock_pick_news() -> dict:
    try:
        result = sync_ai_pick_news(provider_api_key=settings.news_data_api_key)
    except NewsSyncCooldownError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except NewsSyncInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if result["status"] == "error":
        raise HTTPException(
            status_code=502,
            detail="; ".join(result["warnings"]) or "Configured market news provider returned no results.",
        )
    return result


@router.get("/sentiment")
def ai_stock_pick_sentiment(
    symbol: str | None = Query(default=None, max_length=50),
    cutoff: str | None = Query(default=None, max_length=50),
    limit: int = Query(default=10, ge=1, le=500),
) -> dict:
    """Return stored FinBERT article probabilities and stock-level aggregates."""
    response = get_ai_pick_sentiment(cutoff=_parse_cutoff(cutoff), limit=limit)
    return _company_response(response, symbol)


@router.post("/sentiment/analyze")
def analyze_stock_pick_sentiment(
    cutoff: str | None = Query(default=None, max_length=50),
    limit: int = Query(default=10, ge=1, le=500),
) -> dict:
    """Run ProsusAI/finbert on stored, recent company news and persist its probabilities."""
    try:
        return analyze_ai_pick_sentiment(cutoff=_parse_cutoff(cutoff), limit=limit)
    except SentimentAnalysisInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SentimentModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/models/train/{symbol}")
def train_stock_pick_model(
    symbol: str,
    lstm_epochs: int = Query(default=30, ge=1, le=100),
) -> dict:
    """Train and chronologically evaluate the XGBoost/LSTM direction models."""
    try:
        return train_stock_direction_model(symbol, lstm_epochs=lstm_epochs)
    except PredictionTrainingInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PredictionModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/models/train", status_code=202)
def start_stock_pick_watchlist_training(
    background_tasks: BackgroundTasks,
    lstm_epochs: int = Query(default=30, ge=1, le=100),
) -> dict:
    """Queue missing watchlist model training without holding the HTTP request open."""
    job, created = start_stock_direction_watchlist_job()
    if created:
        background_tasks.add_task(
            run_stock_direction_watchlist_job,
            job["job_id"],
            lstm_epochs=lstm_epochs,
        )
    return job


@router.get("/models/train/jobs/{job_id}")
def stock_pick_watchlist_training_status(job_id: str) -> dict:
    """Return queued, running, or per-symbol results for a watchlist training job."""
    job = get_stock_direction_watchlist_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Watchlist training job was not found.")
    return job


@router.get("/predictions")
def stock_pick_predictions() -> dict:
    """Return live-watchlist quote symbols and only validated saved predictions."""
    return get_stock_direction_predictions()


@router.get("/opportunities")
def stock_pick_opportunities() -> dict:
    """Return fresh, validated model picks above 80%, ranked by probability."""
    return build_ai_stock_opportunities()


@router.get("/predictions/{symbol}")
def stock_pick_prediction(symbol: str) -> dict:
    """Return a calibrated next-regular-session direction prediction if validated."""
    try:
        return get_stock_direction_prediction(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PredictionModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
