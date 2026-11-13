from datetime import datetime

import requests
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

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
    evaluate_stock_news_feature_impact,
    get_stock_direction_prediction,
    get_stock_direction_predictions,
    get_stock_direction_watchlist_job,
    run_stock_direction_watchlist_job,
    start_stock_direction_watchlist_job,
    train_stock_direction_model,
)
from app.services.ai_picks.rag import NewsRagUnavailable, answer_company_news_question
from app.services.ai_picks.selection import (
    build_ai_stock_opportunities,
    queue_opportunity_explanations,
    run_opportunity_explanations,
)
from app.services.sentiment.base import SentimentModelUnavailable
from config.settings import settings

router = APIRouter(prefix="/api/v1/ai-stock-picks", tags=["AI Stock Picks"])


class GroundedNewsQuestion(BaseModel):
    symbol: str = Field(min_length=1, max_length=50)
    question: str = Field(min_length=1, max_length=500)
    cutoff: str | None = Field(default=None, max_length=50)
    limit: int = Field(default=5, ge=1, le=8)

    @field_validator("symbol", "question")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("This field must not be blank.")
        return normalized


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


@router.post("/news/grounded-analysis")
def grounded_company_news_analysis(request: GroundedNewsQuestion) -> dict:
    """Answer a company-news question from archived, cutoff-bounded sources."""
    try:
        return answer_company_news_question(
            symbol=request.symbol,
            question=request.question,
            cutoff=_parse_cutoff(request.cutoff),
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NewsRagUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models/news-impact/{symbol}")
def stock_pick_news_feature_impact(
    symbol: str,
    step: int = Query(default=60, ge=30, le=120),
) -> dict:
    """Compare point-in-time news and no-news XGBoost variants on matched future folds."""
    try:
        return evaluate_stock_news_feature_impact(symbol, step=step)
    except PredictionTrainingInProgress as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PredictionModelUnavailable as exc:
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
def stock_pick_opportunities(
    background_tasks: BackgroundTasks,
    threshold_percent: int = Query(default=80, ge=65, le=80),
) -> dict:
    """Return fresh, validated model picks above the selected probability threshold."""
    if threshold_percent not in (65, 70, 80):
        raise HTTPException(
            status_code=422,
            detail="Threshold must be 65, 70, or 80 percent.",
        )
    response = build_ai_stock_opportunities(
        include_explanations=False,
        threshold_percent=threshold_percent,
    )
    jobs = queue_opportunity_explanations(response["opportunities"])
    if jobs:
        background_tasks.add_task(run_opportunity_explanations, jobs)
    return response


@router.get("/predictions/{symbol}")
def stock_pick_prediction(symbol: str) -> dict:
    """Return a calibrated next-regular-session direction prediction if validated."""
    try:
        return get_stock_direction_prediction(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PredictionModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
