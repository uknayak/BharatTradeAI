"""Validated probability screening and explainable stock-pick ranking."""

import math
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from app.services.ai_picks.ollama_explainer import (
    OllamaExplanationUnavailable,
    explain_stock_analyses,
    unavailable_explanation,
)
from app.services.ai_picks.prediction import get_stock_direction_predictions
from app.services.ai_picks.sentiment_analysis import get_ai_pick_sentiment
from app.services.market.live_quotes import WATCHLIST, fetch_live_quotes

MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
MIN_UPWARD_PROBABILITY = 0.8
MAX_QUOTE_AGE = timedelta(minutes=15)
MAX_DAILY_BAR_AGE = timedelta(days=4)
NO_OPPORTUNITIES_MESSAGE = "No high-confidence opportunities found for the next trading day."


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _next_weekday(last_market_time: str) -> str:
    market_date = _parse_timestamp(last_market_time).astimezone(MARKET_TIMEZONE).date()
    target = market_date + timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target.isoformat()


def _risk_level(indicators: dict[str, Any]) -> tuple[str, str]:
    atr_pct = indicators["atr_14_pct"]
    volatility_pct = indicators["volatility_20_daily_pct"]
    if atr_pct >= 4 or volatility_pct >= 3:
        return "HIGH", "Recent range or daily volatility is elevated."
    if atr_pct >= 2 or volatility_pct >= 1.5:
        return "MEDIUM", "Recent range or daily volatility is moderate."
    return "LOW", "Recent range and daily volatility are comparatively contained."


def _technical_signals(indicators: dict[str, Any], pattern: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    signals: list[dict[str, str]] = []
    summary: dict[str, Any] = {}

    def add(signal_id: str, label: str, polarity: str) -> None:
        signals.append({"id": signal_id, "label": label, "polarity": polarity})

    ema20 = indicators["ema_20_gap_pct"]
    sma200 = indicators["sma_200_gap_pct"]
    if ema20 > 0 and sma200 > 0:
        add("trend", "Price is above the 20-day and 200-day moving averages.", "bullish")
        summary["trend"] = "bullish"
    elif ema20 < 0 and sma200 < 0:
        add("trend", "Price is below the 20-day and 200-day moving averages.", "bearish")
        summary["trend"] = "bearish"
    else:
        add("trend", "Moving-average trend is mixed.", "risk")
        summary["trend"] = "mixed"

    if indicators["macd_pct"] > indicators["macd_signal_pct"]:
        add("macd", "MACD is above its signal line.", "bullish")
        summary["macd"] = "bullish"
    else:
        add("macd", "MACD is at or below its signal line.", "bearish")
        summary["macd"] = "bearish"

    rsi = indicators["rsi_14"]
    if rsi >= 70:
        add("rsi", "RSI is in the historically overbought range.", "risk")
        summary["rsi"] = "overbought"
    elif rsi <= 30:
        add("rsi", "RSI is in the historically oversold range.", "risk")
        summary["rsi"] = "oversold"
    else:
        summary["rsi"] = "neutral"

    momentum = pattern["momentum_5d_pct"]
    if momentum > 0:
        add("momentum", "Five-session price momentum is positive.", "bullish")
        summary["momentum_5d"] = "positive"
    elif momentum < 0:
        add("momentum", "Five-session price momentum is negative.", "bearish")
        summary["momentum_5d"] = "negative"
    else:
        summary["momentum_5d"] = "flat"
    return signals, summary


def build_ai_stock_opportunities(*, now: datetime | None = None) -> dict[str, Any]:
    """Select only fresh, validated model outputs above the strict 80% threshold."""
    as_of = now or datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        raise ValueError("The opportunity-selection time must include a timezone.")
    as_of = as_of.astimezone(timezone.utc)
    quotes_response = fetch_live_quotes()
    predictions_response = get_stock_direction_predictions()
    sentiment_response = get_ai_pick_sentiment(cutoff=as_of)

    quotes_by_symbol = {
        quote["symbol"]: quote
        for quote in quotes_response.get("quotes", [])
        if quote.get("type") == "EQUITY"
    }
    names = {
        instrument["symbol"]: instrument["name"]
        for instrument in WATCHLIST
        if instrument.get("type") == "EQUITY"
    }
    sentiment_by_symbol = {
        company["symbol"]: company
        for company in sentiment_response.get("companies", [])
    }
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    predictions = predictions_response.get("predictions", [])

    for prediction in predictions:
        symbol = prediction.get("symbol")
        if not symbol:
            continue
        if (
            prediction.get("status") != "ready"
            or prediction.get("validation_status") != "out_of_sample_evaluated"
            or prediction.get("calibration_status") != "sigmoid_calibrated_on_chronological_holdout"
        ):
            exclusions.append({"symbol": symbol, "reason": "model_not_validated"})
            continue
        probability = prediction.get("upward_probability")
        if not isinstance(probability, (int, float)) or not MIN_UPWARD_PROBABILITY < probability <= 1:
            exclusions.append({"symbol": symbol, "reason": "below_strict_80_percent_threshold"})
            continue
        last_market_time = _parse_timestamp(prediction.get("last_market_time"))
        if (
            last_market_time is None
            or last_market_time > as_of + timedelta(minutes=2)
            or as_of - last_market_time > MAX_DAILY_BAR_AGE
        ):
            exclusions.append({"symbol": symbol, "reason": "daily_history_stale_or_invalid"})
            continue

        quote = quotes_by_symbol.get(symbol)
        quote_time = _parse_timestamp(quote.get("timestamp")) if quote else None
        if (
            quote is None
            or quote_time is None
            or quote_time > as_of + timedelta(minutes=2)
            or as_of - quote_time > MAX_QUOTE_AGE
            or not isinstance(quote.get("lastPrice"), (int, float))
            or not isinstance(quote.get("changePct"), (int, float))
            or not math.isfinite(quote["lastPrice"])
            or quote["lastPrice"] <= 0
            or not math.isfinite(quote["changePct"])
        ):
            exclusions.append({"symbol": symbol, "reason": "live_quote_stale_or_incomplete"})
            continue

        indicators = prediction.get("technical_indicators")
        pattern = prediction.get("historical_pattern")
        required_indicators = (
            "rsi_14",
            "volume_ratio",
            "macd_pct",
            "macd_signal_pct",
            "ema_20_gap_pct",
            "ema_50_gap_pct",
            "sma_200_gap_pct",
            "bollinger_position",
            "bollinger_width_pct",
            "atr_14_pct",
            "volatility_20_daily_pct",
        )
        required_patterns = ("latest_daily_return_pct", "momentum_5d_pct", "momentum_20d_pct")
        if (
            not isinstance(indicators, dict)
            or not isinstance(pattern, dict)
            or any(
                not isinstance(indicators.get(key), (int, float))
                or not math.isfinite(indicators[key])
                for key in required_indicators
            )
            or any(
                not isinstance(pattern.get(key), (int, float))
                or not math.isfinite(pattern[key])
                for key in required_patterns
            )
        ):
            exclusions.append({"symbol": symbol, "reason": "technical_history_incomplete"})
            continue

        company_sentiment = sentiment_by_symbol.get(symbol, {})
        sentiment = {
            "label": company_sentiment.get("sentiment", "NO_DATA"),
            "score": company_sentiment.get("sentimentScore"),
            "article_count": company_sentiment.get("articleCount", 0),
            "analyzed_article_count": company_sentiment.get("analyzedArticleCount", 0),
            "probabilities": company_sentiment.get("probabilities"),
        }
        news_signals = []
        if sentiment["label"] == "POSITIVE":
            news_signals.append({"id": "news_sentiment", "label": "FinBERT company-news sentiment is positive.", "polarity": "bullish"})
        elif sentiment["label"] == "NEGATIVE":
            news_signals.append({"id": "news_sentiment", "label": "FinBERT company-news sentiment is negative.", "polarity": "bearish"})

        risk_level, risk_summary = _risk_level(indicators)
        signals, technical_summary = _technical_signals(indicators, pattern)
        signals.extend(news_signals)
        signals.append({"id": "risk_level", "label": risk_summary, "polarity": "risk"})
        explanation_payload = {
            "symbol": symbol,
            "exchange": prediction["exchange"],
            "direction": prediction["direction"],
            "upward_probability": probability,
            "current_price": quote["lastPrice"],
            "daily_change_pct": quote["changePct"],
            "quote_source": quotes_response.get("provider", "Yahoo Finance"),
            "quote_timestamp": quote["timestamp"],
            "technical_indicators": indicators,
            "historical_pattern": pattern,
            "news_sentiment": sentiment,
            "risk_level": risk_level,
        }
        eligible.append(
            {
                "company_name": names.get(symbol, symbol),
                "exchange": prediction["exchange"],
                "symbol": symbol,
                "current_price": quote["lastPrice"],
                "daily_change_pct": quote["changePct"],
                "upward_probability": probability,
                "predicted_direction": prediction["direction"],
                "expected_price_range": None,
                "expected_price_range_status": "unsupported_by_direction_only_model",
                "news_sentiment": sentiment,
                "technical_analysis": {
                    "indicators": indicators,
                    "summary": technical_summary,
                    "historical_pattern": pattern,
                    "history_candle_count": prediction.get("history_candle_count"),
                    "last_market_time": prediction["last_market_time"],
                },
                "risk_classification": {"level": risk_level, "summary": risk_summary},
                "prediction_timestamp": as_of.isoformat(),
                "next_trading_day": _next_weekday(prediction["last_market_time"]),
                "next_trading_day_is_estimate": True,
                "calendar_note": "Weekday estimate only; an exchange holiday calendar is not configured.",
                "model_version": prediction["model_version"],
                "validation_status": prediction["validation_status"],
                "historical_test_metrics": prediction.get("metrics", {}).get("selected"),
                "gemma_explanation": None,
                "_signals": signals,
                "_explanation_payload": explanation_payload,
            }
        )

    eligible.sort(key=lambda stock: stock["upward_probability"], reverse=True)
    try:
        explanations = explain_stock_analyses(
            [
                (stock["_explanation_payload"], stock["_signals"])
                for stock in eligible
            ]
        )
    except OllamaExplanationUnavailable:
        explanations = [unavailable_explanation() for _ in eligible]
    for stock, explanation in zip(eligible, explanations):
        stock["gemma_explanation"] = explanation
    for stock in eligible:
        stock.pop("_signals")
        stock.pop("_explanation_payload")

    return {
        "status": "ready" if eligible else "no_opportunities",
        "provider": "Yahoo Finance + validated local ML + stored FinBERT + Ollama",
        "threshold": MIN_UPWARD_PROBABILITY,
        "threshold_comparison": "strictly greater than",
        "generated_at": as_of.isoformat(),
        "evaluated_count": len(predictions),
        "excluded_count": len(exclusions),
        "exclusion_reasons": exclusions,
        "model_statuses": predictions,
        "opportunities": eligible,
        "empty_message": None if eligible else NO_OPPORTUNITIES_MESSAGE,
        "notice": (
            "Probabilities come only from the calibrated, out-of-sample validated ML service. "
            "Gemma 3 selects from supplied signal IDs; prices, probabilities, indicators, and summaries "
            "are rendered from trusted application data. Yahoo quotes may be delayed."
        ),
    }
