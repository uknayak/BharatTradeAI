"""Persisted-candle next-session direction training and inference."""

import logging
import math
import os
import re
import threading
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from app.db.database import SessionLocal
from app.db.models import MarketInstrument
from app.ml.ensemble import compare_news_feature_walk_forward, train_direction_ensemble
from app.ml.features import build_direction_features, build_direction_training_set
from app.ml.lstm import build_ohlcv_sequences, predict_lstm_probabilities
from app.ml.ensemble import _calibrate
from app.services.ai_picks.news_models import (
    AIStockArticleSentiment,
    AIStockNewsSnapshot,
)
from app.services.market.live_quotes import WATCHLIST
from app.services.market.models import Candle, MarketDataSync
from app.services.sentiment.registry import DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)
MODEL_DIRECTORY = Path(os.getenv("AI_STOCK_MODEL_DIR", "model_artifacts/ai_stock_picks"))
DAILY_TIMEFRAME = "1d"
INDIA_CLOSE_UTC = time(hour=10)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
HISTORY_READY_TIME = time(hour=16, minute=15)
FRESHNESS_HALF_LIFE_HOURS = 48
UPWARD_PROBABILITY_THRESHOLD = 0.8
_training_lock = threading.Lock()
_watchlist_job_lock = threading.Lock()
_watchlist_jobs: dict[str, dict[str, Any]] = {}


class PredictionTrainingInProgress(RuntimeError):
    """Raised when another training request is active."""


class PredictionModelUnavailable(RuntimeError):
    """Raised when model training or inference dependencies fail."""


def _instrument(symbol: str) -> dict[str, str] | None:
    tracked = next(
        (
            row
            for row in WATCHLIST
            if row["symbol"] == symbol.upper() and row.get("type") == "EQUITY"
        ),
        None,
    )
    if tracked is not None:
        return tracked
    db = SessionLocal()
    try:
        market_instrument = (
            db.query(MarketInstrument)
            .filter(
                MarketInstrument.symbol == symbol.upper(),
                MarketInstrument.exchange.in_(("NSE", "BSE")),
                MarketInstrument.is_active.is_(True),
                func.lower(MarketInstrument.instrument_type).in_(("stock", "equity")),
            )
            .first()
        )
    except SQLAlchemyError:
        logger.exception("Could not resolve the market instrument for %s.", symbol)
        raise
    finally:
        db.close()
    if market_instrument is None:
        return None
    return {
        "symbol": market_instrument.symbol,
        "name": market_instrument.symbol,
        "exchange": market_instrument.exchange.upper(),
        "type": "EQUITY",
    }


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _market_date(value: datetime) -> str:
    return _as_utc(value).date().isoformat()


def _decision_cutoff(value: datetime) -> datetime:
    return datetime.combine(_as_utc(value).date(), INDIA_CLOSE_UTC, tzinfo=timezone.utc)


def _news_was_available_by(
    article: AIStockNewsSnapshot,
    sentiment: AIStockArticleSentiment,
    cutoff: datetime,
) -> bool:
    return (
        _as_utc(article.published_at) < cutoff
        and _as_utc(article.retrieved_at) < cutoff
        and _as_utc(sentiment.analyzed_at) < cutoff
    )


def _is_complete_daily_bar(
    candle_time: datetime,
    last_synced_at: datetime | None,
    *,
    now: datetime | None = None,
) -> bool:
    market_now = _as_utc(now or datetime.now(timezone.utc)).astimezone(MARKET_TIMEZONE)
    market_bar_time = _as_utc(candle_time).astimezone(MARKET_TIMEZONE)
    if market_bar_time.date() != market_now.date():
        return True
    if market_now.time() < HISTORY_READY_TIME or last_synced_at is None:
        return False
    sync_time = _as_utc(last_synced_at).astimezone(MARKET_TIMEZONE)
    return sync_time.date() == market_now.date() and sync_time.time() >= HISTORY_READY_TIME


def _returns_by_date(rows: list[Candle]) -> dict[str, float]:
    previous_by_symbol: dict[str, float] = {}
    returns: dict[str, float] = {}
    for row in sorted(rows, key=lambda candle: (candle.symbol, candle.candle_time)):
        if row.close is None or row.close <= 0:
            continue
        previous = previous_by_symbol.get(row.symbol)
        if previous is not None and previous > 0:
            returns[_market_date(row.candle_time)] = row.close / previous - 1
        previous_by_symbol[row.symbol] = row.close
    return returns


def _load_context(db: Any, instrument: dict[str, str], candles: list[Candle]) -> dict[str, Any]:
    market_rows = (
        db.query(Candle)
        .filter(
            Candle.symbol == "NIFTY 50",
            Candle.exchange == "NSE",
            Candle.timeframe == DAILY_TIMEFRAME,
        )
        .order_by(Candle.candle_time.asc())
        .all()
    )
    market_returns = _returns_by_date(market_rows)

    company = (
        db.query(MarketInstrument)
        .filter(
            MarketInstrument.symbol == instrument["symbol"],
            MarketInstrument.exchange == instrument["exchange"],
            MarketInstrument.is_active.is_(True),
        )
        .one_or_none()
    )
    sector = company.sector if company is not None else None
    sector_returns: dict[str, float] = {}
    if sector:
        peers = (
            db.query(MarketInstrument.symbol)
            .filter(
                MarketInstrument.sector == sector,
                MarketInstrument.exchange == instrument["exchange"],
                MarketInstrument.is_active.is_(True),
                func.lower(MarketInstrument.instrument_type).in_(("stock", "equity")),
                MarketInstrument.symbol != instrument["symbol"],
            )
            .all()
        )
        peer_symbols = [row[0] for row in peers]
        if peer_symbols:
            peer_rows = (
                db.query(Candle)
                .filter(
                    Candle.symbol.in_(peer_symbols),
                    Candle.timeframe == DAILY_TIMEFRAME,
                )
                .order_by(Candle.candle_time.asc())
                .all()
            )
            by_symbol_date: dict[str, dict[str, float]] = {}
            previous: dict[str, float] = {}
            for row in peer_rows:
                if row.close is None or row.close <= 0:
                    continue
                old = previous.get(row.symbol)
                if old is not None and old > 0:
                    by_symbol_date.setdefault(_market_date(row.candle_time), {})[row.symbol] = row.close / old - 1
                previous[row.symbol] = row.close
            sector_returns = {
                day: sum(values.values()) / len(values)
                for day, values in by_symbol_date.items()
                if values
            }

    articles = (
        db.query(AIStockNewsSnapshot)
        .filter(AIStockNewsSnapshot.symbol == instrument["symbol"])
        .all()
    )
    article_scores = (
        db.query(AIStockArticleSentiment)
        .filter(
            AIStockArticleSentiment.symbol == instrument["symbol"],
            AIStockArticleSentiment.model_id == DEFAULT_MODEL_ID,
        )
        .all()
    )
    score_by_id = {row.article_id: row for row in article_scores}
    news_sentiment: dict[str, float] = {}
    for candle in candles:
        cutoff = _decision_cutoff(candle.candle_time)
        weighted_score = 0.0
        total_weight = 0.0
        for article in articles:
            sentiment = score_by_id.get(article.article_id)
            if sentiment is None or not _news_was_available_by(article, sentiment, cutoff):
                continue
            published_at = _as_utc(article.published_at)
            age_hours = max(0.0, (cutoff - published_at).total_seconds() / 3600)
            freshness = math.exp(-math.log(2) * age_hours / FRESHNESS_HALF_LIFE_HOURS)
            weight = sentiment.relevance_score * freshness
            weighted_score += sentiment.sentiment_score * weight
            total_weight += weight
        if total_weight:
            news_sentiment[_market_date(candle.candle_time)] = weighted_score / total_weight
    return {
        "news_sentiment": news_sentiment,
        "sector_returns": sector_returns,
        "market_returns": market_returns,
        "sector": sector,
    }


def _load_market_data(symbol: str) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    instrument = _instrument(symbol)
    if instrument is None:
        raise ValueError(f"Unsupported NSE equity symbol: {symbol}.")
    from app.services.ai_picks.news import archive_existing_news

    archive_existing_news()
    db = SessionLocal()
    try:
        rows = (
            db.query(Candle)
            .filter(
                Candle.symbol == instrument["symbol"],
                Candle.exchange == instrument["exchange"],
                Candle.timeframe == DAILY_TIMEFRAME,
            )
            .order_by(Candle.candle_time.asc())
            .all()
        )
        valid_rows = [
            row
            for row in rows
            if row.open is not None
            and row.high is not None
            and row.low is not None
            and row.close is not None
            and row.close > 0
        ]
        sync_state = (
            db.query(MarketDataSync)
            .filter_by(
                symbol=instrument["symbol"],
                exchange=instrument["exchange"],
                timeframe=DAILY_TIMEFRAME,
            )
            .one_or_none()
        )
        if valid_rows and not _is_complete_daily_bar(
            valid_rows[-1].candle_time,
            sync_state.last_synced_at if sync_state else None,
        ):
            logger.info(
                "Excluding incomplete current-session daily bar for %s.",
                instrument["symbol"],
            )
            valid_rows.pop()
        context = _load_context(db, instrument, valid_rows) if valid_rows else {
            "news_sentiment": {},
            "sector_returns": {},
            "market_returns": {},
            "sector": None,
        }
        candles = [
            {
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume or 0),
                "candle_time": row.candle_time,
            }
            for row in valid_rows
        ]
    except SQLAlchemyError:
        logger.exception("Could not load market history for %s.", symbol)
        raise
    finally:
        db.close()
    return instrument, candles, context


def _data_status(candles: list[dict[str, Any]]) -> dict[str, Any]:
    if not candles:
        return {
            "status": "insufficient_data",
            "message": "No valid daily OHLCV history is stored for this symbol.",
            "sample_count": 0,
            "latest_market_time": None,
        }
    latest = _as_utc(candles[-1]["candle_time"])
    age = datetime.now(timezone.utc) - latest
    if age.total_seconds() > 4 * 24 * 60 * 60:
        return {
            "status": "stale_data",
            "message": "The latest daily candle is more than four days old; sync market history before training.",
            "sample_count": len(candles),
            "latest_market_time": latest.isoformat(),
        }
    return {
        "status": "ready",
        "message": None,
        "sample_count": len(candles),
        "latest_market_time": latest.isoformat(),
    }


def evaluate_stock_news_feature_impact(symbol: str, *, step: int = 60) -> dict[str, Any]:
    """Run a paired chronological XGBoost ablation without changing production models."""
    if not _training_lock.acquire(blocking=False):
        raise PredictionTrainingInProgress("A stock prediction training job is already running.")
    try:
        instrument, candles, context = _load_market_data(symbol)
        data_status = _data_status(candles)
        response: dict[str, Any] = {
            "symbol": instrument["symbol"],
            "exchange": instrument["exchange"],
            "provider": "Yahoo Finance",
            "evaluation": "FinBERT news-feature ablation; production model unchanged",
            "market_data": data_status,
            "news_context": "available" if context["news_sentiment"] else "unavailable",
        }
        if data_status["status"] != "ready":
            return {**response, "status": data_status["status"], "message": data_status["message"]}

        dataset = build_direction_training_set(
            candles,
            news_sentiment=context["news_sentiment"],
            sector_returns=context["sector_returns"],
            market_returns=context["market_returns"],
        )
        sample_count = len(dataset["labels"])
        start_index = int(sample_count * 0.85)
        comparison = compare_news_feature_walk_forward(
            dataset["features"],
            dataset["labels"],
            start_index=start_index,
            step=step,
        )
        result = {**response, **comparison, "sample_count": sample_count}
        if comparison["status"] == "ready":
            first_test_candle = candles[dataset["indices"][start_index]]
            last_test_candle = candles[dataset["indices"][-1]]
            result["out_of_sample_window"] = {
                "input_start": _as_utc(first_test_candle["candle_time"]).isoformat(),
                "input_end": _as_utc(last_test_candle["candle_time"]).isoformat(),
                "predicted_sessions": comparison["with_news"]["sample_count"],
            }
        return result
    finally:
        _training_lock.release()


def _artifact_directory(root: Path, symbol: str) -> Path:
    safe_symbol = re.sub(r"[^A-Z0-9_-]", "_", symbol.upper())
    return root / safe_symbol


def _write_artifact(
    payload: dict[str, Any],
    symbol: str,
    artifact_directory: Path,
) -> tuple[str, str]:
    import joblib

    directory = _artifact_directory(artifact_directory, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
    model_path = directory / f"{version}.joblib"
    temporary_model = directory / f".{version}.tmp"
    joblib.dump({**payload, "model_version": version}, temporary_model)
    os.replace(temporary_model, model_path)
    latest_file = directory / "latest.txt"
    temporary_latest = directory / ".latest.tmp"
    temporary_latest.write_text(version, encoding="ascii")
    os.replace(temporary_latest, latest_file)
    return version, str(model_path)


def _latest_artifact(symbol: str, artifact_directory: Path) -> tuple[str, dict[str, Any]] | None:
    import joblib

    directory = _artifact_directory(artifact_directory, symbol)
    latest_file = directory / "latest.txt"
    if not latest_file.exists():
        return None
    version = latest_file.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[A-Za-z0-9TZ-]+", version):
        raise PredictionModelUnavailable("The model-version pointer is invalid.")
    model_path = directory / f"{version}.joblib"
    if not model_path.is_file():
        raise PredictionModelUnavailable("The latest model artifact is missing.")
    return version, joblib.load(model_path)


def train_stock_direction_model(
    symbol: str,
    *,
    artifact_directory: Path | None = None,
    lstm_epochs: int = 30,
) -> dict[str, Any]:
    if not _training_lock.acquire(blocking=False):
        raise PredictionTrainingInProgress("A stock prediction training job is already running.")
    try:
        instrument, candles, context = _load_market_data(symbol)
        data_status = _data_status(candles)
        response: dict[str, Any] = {
            "symbol": instrument["symbol"],
            "exchange": instrument["exchange"],
            "provider": "Yahoo Finance",
            "market_data": data_status,
            "sector_context": "available" if context["sector_returns"] else "unavailable",
            "market_context": "available" if context["market_returns"] else "unavailable",
            "news_context": "available" if context["news_sentiment"] else "unavailable",
        }
        if data_status["status"] != "ready":
            return {
                **response,
                "status": data_status["status"],
                "validation_status": data_status["status"],
                "message": data_status["message"],
            }

        dataset = build_direction_training_set(
            candles,
            news_sentiment=context["news_sentiment"],
            sector_returns=context["sector_returns"],
            market_returns=context["market_returns"],
        )
        latest_feature = build_direction_features(
            candles,
            news_sentiment=context["news_sentiment"],
            sector_returns=context["sector_returns"],
            market_returns=context["market_returns"],
        )[-1]
        if not all(math.isfinite(value) for value in latest_feature.values()):
            return {
                **response,
                "status": "insufficient_data",
                "validation_status": "insufficient_data",
                "message": "Historical data does not yet cover the required indicator warm-up period.",
                "sample_count": len(dataset["labels"]),
                "required_sample_count": 200,
            }
        trained = train_direction_ensemble(
            dataset["features"],
            dataset["labels"],
            candles,
            dataset["indices"],
            latest_features=latest_feature,
            lstm_epochs=lstm_epochs,
        )
        if trained["status"] != "ready":
            return {**response, **trained}

        model_data = trained.pop("artifacts")
        payload = {
            "symbol": instrument["symbol"],
            "exchange": instrument["exchange"],
            "model_version": None,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "latest_market_time": data_status["latest_market_time"],
            "latest": trained["latest"],
            "selection": trained["selection"],
            "metrics": trained["metrics"],
            "calibration_status": trained["calibration_status"],
            "model_data": model_data,
        }
        root = artifact_directory or MODEL_DIRECTORY
        version, _ = _write_artifact(payload, instrument["symbol"], root)
        payload["model_version"] = version
        trained["prediction"] = _prediction_payload(
            instrument,
            candles,
            payload,
            version,
            context,
        )
        return {
            **response,
            **trained,
            "model_version": version,
        }
    except (ImportError, ModuleNotFoundError) as exc:
        logger.exception("Prediction model dependency is unavailable for %s.", symbol)
        raise PredictionModelUnavailable(
            "XGBoost, scikit-learn, and PyTorch are required for prediction training."
        ) from exc
    except SQLAlchemyError:
        raise
    except PredictionModelUnavailable:
        raise
    except Exception as exc:
        logger.exception("Stock prediction training failed for %s.", symbol)
        raise PredictionModelUnavailable(f"Stock prediction training failed: {exc}") from exc
    finally:
        _training_lock.release()


def _prediction_payload(
    instrument: dict[str, str],
    candles: list[dict[str, Any]],
    payload: dict[str, Any],
    version: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    from app.ml.features import build_direction_features

    features = build_direction_features(
        candles,
        news_sentiment=context["news_sentiment"],
        sector_returns=context["sector_returns"],
        market_returns=context["market_returns"],
    )[-1]
    model_data = payload["model_data"]
    feature_names = model_data["feature_names"]
    matrix = [[features[name] for name in feature_names]]
    xgb_raw = model_data["xgboost"].predict_proba(matrix)[:, 1]
    xgb_probability = float(_calibrate(model_data["xgboost_calibrator"], xgb_raw)[0])
    sequence = build_ohlcv_sequences(candles, [len(candles) - 1], model_data["sequence_length"])
    lstm_raw = predict_lstm_probabilities(model_data["lstm"], sequence)
    lstm_probability = float(_calibrate(model_data["lstm_calibrator"], lstm_raw)[0])
    xgb_weight = float(model_data["xgboost_weight"])
    upward_probability = xgb_weight * xgb_probability + (1 - xgb_weight) * lstm_probability
    technical_indicators = {
        "rsi_14": features["rsi_14"] * 100,
        "volume_ratio": features["volume_ratio"],
        "macd_pct": features["macd_pct"],
        "macd_signal_pct": features["macd_signal_pct"],
        "ema_20_gap_pct": features["ema_20_gap"] * 100,
        "ema_50_gap_pct": features["ema_50_gap"] * 100,
        "sma_200_gap_pct": features["sma_200_gap"] * 100,
        "bollinger_position": features["bollinger_position"],
        "bollinger_width_pct": features["bollinger_width"] * 100,
        "atr_14_pct": features["atr_14_pct"] * 100,
        "volatility_20_daily_pct": features["volatility_20"] * 100,
    }
    historical_pattern = {
        "latest_daily_return_pct": features["return_1"] * 100,
        "momentum_5d_pct": features["momentum_5"] * 100,
        "momentum_20d_pct": features["momentum_20"] * 100,
    }
    return {
        "symbol": instrument["symbol"],
        "exchange": instrument["exchange"],
        "sector": context.get("sector"),
        "status": "ready",
        "validation_status": "out_of_sample_evaluated",
        "model_version": version,
        "prediction_for": "next_regular_trading_day_close_vs_previous_regular_close",
        "prediction_date": "next regular trading day",
        "last_close": candles[-1]["close"],
        "last_market_time": _as_utc(candles[-1]["candle_time"]).isoformat(),
        "history_candle_count": len(candles),
        "technical_indicators": technical_indicators,
        "historical_pattern": historical_pattern,
        "upward_probability": upward_probability,
        "downward_probability": 1 - upward_probability,
        "direction": "UP" if upward_probability >= 0.5 else "DOWN",
        "probability_threshold_met": upward_probability >= UPWARD_PROBABILITY_THRESHOLD,
        "threshold": UPWARD_PROBABILITY_THRESHOLD,
        "probability_note": (
            "This is a calibrated model probability, not a guarantee of accuracy. "
            "Historical out-of-sample metrics are reported separately."
        ),
        "model_probabilities": {
            "xgboost_upward": xgb_probability,
            "lstm_upward": lstm_probability,
        },
        "ensemble": {
            "used": payload["selection"]["ensemble_used"],
            "xgboost_weight": xgb_weight,
            "lstm_weight": 1 - xgb_weight,
        },
        "metrics": payload["metrics"],
        "calibration_status": payload["calibration_status"],
        "news_context": "available" if context["news_sentiment"] else "unavailable",
        "sector_context": "available" if context["sector_returns"] else "unavailable",
        "market_context": "available" if context["market_returns"] else "unavailable",
    }


def get_stock_direction_prediction(
    symbol: str,
    *,
    artifact_directory: Path | None = None,
) -> dict[str, Any]:
    instrument = _instrument(symbol)
    if instrument is None:
        raise ValueError(f"Unsupported NSE equity symbol: {symbol}.")
    root = artifact_directory or MODEL_DIRECTORY
    try:
        artifact = _latest_artifact(instrument["symbol"], root)
        if artifact is None:
            _, candles, context = _load_market_data(instrument["symbol"])
            data_status = _data_status(candles)
            return {
                "symbol": instrument["symbol"],
                "exchange": instrument["exchange"],
                "sector": context.get("sector"),
                "status": "not_trained" if data_status["status"] == "ready" else data_status["status"],
                "validation_status": "not_trained" if data_status["status"] == "ready" else data_status["status"],
                "message": "Train and evaluate the model before requesting a prediction."
                if data_status["status"] == "ready"
                else data_status["message"],
                "market_data": data_status,
            }
        version, payload = artifact
        _, candles, context = _load_market_data(instrument["symbol"])
        data_status = _data_status(candles)
        if data_status["status"] != "ready":
            return {
                "symbol": instrument["symbol"],
                "exchange": instrument["exchange"],
                "sector": context.get("sector"),
                "status": data_status["status"],
                "validation_status": data_status["status"],
                "message": data_status["message"],
                "market_data": data_status,
            }
        return _prediction_payload(instrument, candles, payload, version, context)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.exception("Prediction inference dependency is unavailable for %s.", symbol)
        raise PredictionModelUnavailable("Prediction inference dependencies are unavailable.") from exc
    except SQLAlchemyError:
        raise
    except PredictionModelUnavailable:
        raise
    except Exception as exc:
        logger.exception("Stock prediction inference failed for %s.", symbol)
        raise PredictionModelUnavailable(f"Stock prediction inference failed: {exc}") from exc


def get_stock_direction_predictions() -> dict[str, Any]:
    """Return real provider-watchlist stocks with model status, never sample forecasts."""
    predictions: list[dict[str, Any]] = []
    for instrument in WATCHLIST:
        if instrument.get("type") != "EQUITY":
            continue
        try:
            predictions.append(get_stock_direction_prediction(instrument["symbol"]))
        except PredictionModelUnavailable as exc:
            logger.exception("Could not retrieve prediction state for %s.", instrument["symbol"])
            predictions.append(
                {
                    "symbol": instrument["symbol"],
                    "exchange": instrument["exchange"],
                    "status": "unavailable",
                    "validation_status": "unavailable",
                    "message": str(exc),
                }
            )
    return {
        "status": "ready" if predictions else "no_data",
        "provider": "Yahoo Finance daily history + validated local models",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
        "notice": (
            "Quotes update from Yahoo Finance independently. Probabilities appear only for "
            "trained, calibrated models with fresh daily history and out-of-sample evaluation."
        ),
    }


def train_stock_direction_watchlist(
    *,
    lstm_epochs: int = 30,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train missing watchlist models and preserve each symbol's independent status."""
    outcomes: list[dict[str, Any]] = []

    def record(outcome: dict[str, Any]) -> None:
        outcomes.append(outcome)
        if on_progress is not None:
            on_progress(outcome)

    for instrument in WATCHLIST:
        if instrument.get("type") != "EQUITY":
            continue
        symbol = instrument["symbol"]
        try:
            existing = get_stock_direction_prediction(symbol)
            if existing["status"] == "ready":
                record({**existing, "training_action": "existing_model_reused"})
                continue
            trained = train_stock_direction_model(symbol, lstm_epochs=lstm_epochs)
            prediction = trained.get("prediction")
            record(
                {
                    "symbol": symbol,
                    "exchange": instrument["exchange"],
                    "status": trained["status"],
                    "validation_status": trained["validation_status"],
                    "training_action": "trained" if trained["status"] == "ready" else "not_trained",
                    "message": trained.get("message"),
                    "model_version": trained.get("model_version"),
                    "sample_count": trained.get("sample_count"),
                    "required_sample_count": trained.get("required_sample_count"),
                    "prediction": prediction,
                    "metrics": trained.get("metrics"),
                }
            )
        except PredictionTrainingInProgress as exc:
            record(
                {
                    "symbol": symbol,
                    "exchange": instrument["exchange"],
                    "status": "training_in_progress",
                    "validation_status": "training_in_progress",
                    "training_action": "not_trained",
                    "message": str(exc),
                }
            )
        except (PredictionModelUnavailable, SQLAlchemyError) as exc:
            logger.exception("Could not train watchlist model for %s.", symbol)
            record(
                {
                    "symbol": symbol,
                    "exchange": instrument["exchange"],
                    "status": "unavailable",
                    "validation_status": "unavailable",
                    "training_action": "failed",
                    "message": str(exc),
                }
            )
    ready_count = sum(row["status"] == "ready" for row in outcomes)
    return {
        "status": "ready" if ready_count == len(outcomes) and outcomes else "partial",
        "trained_count": sum(row["training_action"] == "trained" for row in outcomes),
        "reused_count": sum(row["training_action"] == "existing_model_reused" for row in outcomes),
        "failed_count": sum(row["training_action"] == "failed" for row in outcomes),
        "predictions": outcomes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "notice": (
            "Each probability is shown only when its saved model passed chronological calibration "
            "and out-of-sample evaluation. Insufficient-data and failed symbols remain without probabilities."
        ),
    }


def start_stock_direction_watchlist_job() -> tuple[dict[str, Any], bool]:
    """Create one background watchlist-training job, or return the active job."""
    with _watchlist_job_lock:
        active = next(
            (
                job
                for job in _watchlist_jobs.values()
                if job["status"] in ("queued", "running")
            ),
            None,
        )
        if active is not None:
            return {**active, "predictions": list(active["predictions"])}, False

        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "completed_count": 0,
            "total_count": sum(row.get("type") == "EQUITY" for row in WATCHLIST),
            "trained_count": 0,
            "reused_count": 0,
            "failed_count": 0,
            "predictions": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "message": "Watchlist model training is queued.",
        }
        _watchlist_jobs[job_id] = job
        return {**job, "predictions": []}, True


def get_stock_direction_watchlist_job(job_id: str) -> dict[str, Any] | None:
    with _watchlist_job_lock:
        job = _watchlist_jobs.get(job_id)
        return {**job, "predictions": list(job["predictions"])} if job else None


def run_stock_direction_watchlist_job(job_id: str, *, lstm_epochs: int = 30) -> None:
    with _watchlist_job_lock:
        job = _watchlist_jobs.get(job_id)
        if job is None:
            logger.error("Watchlist model-training job %s no longer exists.", job_id)
            return
        job["status"] = "running"
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        job["message"] = "Training and validating stock models."

    def record_progress(outcome: dict[str, Any]) -> None:
        with _watchlist_job_lock:
            job = _watchlist_jobs[job_id]
            job["predictions"].append(outcome)
            job["completed_count"] += 1
            job["trained_count"] += outcome.get("training_action") == "trained"
            job["reused_count"] += outcome.get("training_action") == "existing_model_reused"
            job["failed_count"] += outcome.get("training_action") == "failed"
            job["updated_at"] = datetime.now(timezone.utc).isoformat()

    try:
        result = train_stock_direction_watchlist(
            lstm_epochs=lstm_epochs,
            on_progress=record_progress,
        )
        with _watchlist_job_lock:
            job = _watchlist_jobs[job_id]
            job["status"] = result["status"]
            job["updated_at"] = result["updated_at"]
            job["message"] = result["notice"]
    except Exception as exc:
        logger.exception("Watchlist model-training job %s failed.", job_id)
        with _watchlist_job_lock:
            job = _watchlist_jobs[job_id]
            job["status"] = "failed"
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            job["message"] = str(exc)
