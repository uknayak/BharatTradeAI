"""Grounded explanation selection using a local Ollama model."""

import json
import logging
from typing import Any

import requests

from config.settings import settings

logger = logging.getLogger(__name__)
OLLAMA_TIMEOUT_SECONDS = 180
MAX_EXPLANATION_TOKENS = 100


class OllamaExplanationUnavailable(RuntimeError):
    """Raised when Ollama cannot return a valid grounded explanation."""


def _signal_text(signal: dict[str, str]) -> str:
    return signal["label"]


def _explanation_from_evidence(
    payload: dict[str, Any],
    selected_ids: dict[str, list[str]],
    signals: dict[str, dict[str, str]],
) -> dict[str, Any]:
    bullish = [_signal_text(signals[signal_id]) for signal_id in selected_ids["bullish_signal_ids"]]
    bearish = [_signal_text(signals[signal_id]) for signal_id in selected_ids["bearish_signal_ids"]]
    risks = [_signal_text(signals[signal_id]) for signal_id in selected_ids["risk_signal_ids"]]

    direction = payload["direction"].lower()
    sentiment = payload["news_sentiment"]
    probability_note = (
        f"The validated model estimates a {payload['upward_probability']:.1%} upward probability. "
        "This is a calibrated model output, not a guarantee."
    )
    factors = []
    factors.append(
        "Bullish signals: " + "; ".join(bullish) + "."
        if bullish
        else "Gemma 3 selected no bullish evidence signals."
    )
    factors.append(
        "Bearish signals: " + "; ".join(bearish) + "."
        if bearish
        else "Gemma 3 selected no bearish evidence signals."
    )
    factors.append(
        "Risks: " + "; ".join(risks) + "."
        if risks
        else "Gemma 3 selected no risk evidence; use the displayed risk classification."
    )
    factors.append(
        f"News sentiment is {sentiment['label'].lower()} based on stored FinBERT analysis."
        if sentiment["label"] not in ("NO_DATA", "PENDING")
        else "Analyzed company-specific news sentiment is unavailable."
    )
    return {
        "status": "ready",
        "provider": "Ollama",
        "model": settings.model_name,
        "summary": f"The validated model currently signals {direction}. {probability_note}",
        "bullish_signals": bullish,
        "bearish_signals": bearish,
        "risks": risks,
        "analysis": " ".join(factors),
        "evidence_ids": selected_ids,
    }


def _request_facts(payload: dict[str, Any], signals: list[dict[str, str]]) -> dict[str, Any]:
    indicators = payload["technical_indicators"]
    pattern = payload["historical_pattern"]
    return {
        "symbol": payload["symbol"],
        "exchange": payload["exchange"],
        "prediction": {
            "direction": payload["direction"],
            "upward_probability": payload["upward_probability"],
        },
        "market_snapshot": {
            "current_price": payload["current_price"],
            "daily_change_pct": payload["daily_change_pct"],
            "source": payload.get("quote_source", "Yahoo Finance"),
            "timestamp": payload.get("quote_timestamp"),
        },
        "technical_indicator_summary": {
            key: indicators[key]
            for key in (
                "rsi_14",
                "macd_pct",
                "macd_signal_pct",
                "ema_20_gap_pct",
                "ema_50_gap_pct",
                "sma_200_gap_pct",
                "bollinger_position",
                "atr_14_pct",
                "volatility_20_daily_pct",
            )
            if key in indicators
        },
        "historical_pattern_summary": {
            key: pattern[key]
            for key in ("latest_daily_return_pct", "momentum_5d_pct", "momentum_20d_pct")
            if key in pattern
        },
        "news_sentiment": payload["news_sentiment"],
        "risk_level": payload["risk_level"],
        "available_signals": signals,
    }


def _validate_signal_selection(
    selection: Any,
    signal_map: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    if not isinstance(selection, dict) or set(selection) != {
        "bullish_signal_ids",
        "bearish_signal_ids",
        "risk_signal_ids",
    }:
        raise OllamaExplanationUnavailable("Ollama returned an unexpected explanation format.")

    selected_ids: dict[str, list[str]] = {}
    for field in ("bullish_signal_ids", "bearish_signal_ids", "risk_signal_ids"):
        values = selection[field]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or value not in signal_map for value in values)
            or len(values) != len(set(values))
        ):
            raise OllamaExplanationUnavailable("Ollama referenced an unknown or duplicate signal.")
        selected_ids[field] = values

    expected_polarities = {
        "bullish_signal_ids": "bullish",
        "bearish_signal_ids": "bearish",
        "risk_signal_ids": "risk",
    }
    for field, polarity in expected_polarities.items():
        if any(signal_map[signal_id]["polarity"] != polarity for signal_id in selected_ids[field]):
            raise OllamaExplanationUnavailable("Ollama assigned a signal to the wrong category.")
    return selected_ids


def explain_stock_analyses(
    candidates: list[tuple[dict[str, Any], list[dict[str, str]]]],
) -> list[dict[str, Any]]:
    """Ask Ollama once to select grounded evidence IDs for each qualified stock."""
    if not candidates:
        return []
    if settings.llm_provider.casefold() != "ollama":
        raise OllamaExplanationUnavailable(
            "The configured explanation provider is not Ollama."
        )
    signal_maps = {
        payload["symbol"]: {signal["id"]: signal for signal in signals}
        for payload, signals in candidates
    }
    request_payload = [
        _request_facts(payload, signals)
        for payload, signals in candidates
    ]
    prompt = (
        "You are a financial-analysis explanation layer. The supplied market data is authoritative; "
        "never change, infer, or add prices, news, indicators, probabilities, dates, or other facts. "
        "News is already summarized and must not be expanded. For each stock select only its "
        "available signal IDs for the matching stock. Put IDs into the matching "
        "bullish_signal_ids, bearish_signal_ids, or risk_signal_ids arrays according to their "
        "supplied polarity. A signal may be omitted, but must never be reassigned to another "
        "polarity. Use the exact id strings from available_signals; never copy a signal label into "
        "an ID array. For example, if the available signal has id \"risk_level\", return "
        "\"risk_level\", not its description. Do not write prose. Return JSON only "
        "with exactly one key, analyses, whose value is an array containing one object per input "
        "stock. Each object must contain exactly symbol, bullish_signal_ids, bearish_signal_ids, "
        "and risk_signal_ids. Analysis facts:\n"
        f"{json.dumps(request_payload, sort_keys=True, allow_nan=False)}"
    )
    base_url = settings.ollama_base_url.rstrip("/")
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": settings.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0, "num_predict": MAX_EXPLANATION_TOKENS, "num_ctx": 2048},
                "keep_alive": "5m",
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        raw_response = body.get("response") if isinstance(body, dict) else None
        if not isinstance(raw_response, str):
            raise OllamaExplanationUnavailable("Ollama returned no explanation selection.")
        parsed = json.loads(raw_response)
        if (
            not isinstance(parsed, dict)
            or set(parsed) != {"analyses"}
            or not isinstance(parsed["analyses"], list)
            or len(parsed["analyses"]) != len(candidates)
        ):
            raise OllamaExplanationUnavailable("Ollama returned an unexpected explanation format.")

        analyses_by_symbol: dict[str, dict[str, Any]] = {}
        for analysis in parsed["analyses"]:
            if (
                not isinstance(analysis, dict)
                or set(analysis)
                != {
                    "symbol",
                    "bullish_signal_ids",
                    "bearish_signal_ids",
                    "risk_signal_ids",
                }
                or not isinstance(analysis["symbol"], str)
                or analysis["symbol"] not in signal_maps
                or analysis["symbol"] in analyses_by_symbol
            ):
                raise OllamaExplanationUnavailable("Ollama returned an unknown or duplicate stock.")
            analyses_by_symbol[analysis["symbol"]] = _validate_signal_selection(
                {key: value for key, value in analysis.items() if key != "symbol"},
                signal_maps[analysis["symbol"]],
            )
        if set(analyses_by_symbol) != set(signal_maps):
            raise OllamaExplanationUnavailable("Ollama omitted a qualified stock.")
        return [
            _explanation_from_evidence(payload, analyses_by_symbol[payload["symbol"]], signal_maps[payload["symbol"]])
            for payload, _ in candidates
        ]
    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
        symbols = ", ".join(payload["symbol"] for payload, _ in candidates)
        logger.warning("Ollama stock explanation unavailable for %s: %s", symbols, exc)
        raise OllamaExplanationUnavailable(
            "Gemma 3 explanation is unavailable; the validated model results are unchanged."
        ) from exc


def explain_stock_analysis(
    payload: dict[str, Any],
    signals: list[dict[str, str]],
) -> dict[str, Any]:
    """Explain one stock with the same strict evidence validation as the batch API."""
    return explain_stock_analyses([(payload, signals)])[0]


def unavailable_explanation() -> dict[str, str]:
    return {
        "status": "unavailable",
        "provider": "Ollama",
        "model": settings.model_name,
        "summary": "Gemma 3 explanation is unavailable. No LLM-generated rationale is shown.",
    }
