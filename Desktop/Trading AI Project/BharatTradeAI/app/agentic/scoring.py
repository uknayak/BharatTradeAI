"""Deterministic weighted scoring for the agentic analysis outputs."""

from collections.abc import Mapping
from typing import Any


DEFAULT_WEIGHTS = {
    "technical": 0.20,
    "options": 0.20,
    "ml": 0.25,
    "market_regime": 0.15,
    "news": 0.10,
    "fundamental": 0.10,
}

SCORE_BANDS = (
    (30, "VERY WEAK"),
    (45, "WEAK"),
    (59, "NEUTRAL"),
    (74, "WATCH"),
    (84, "STRONG"),
    (100, "VERY STRONG"),
)


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _latest(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    for value in reversed(values):
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _latest_direction(values: Any) -> str | None:
    if not isinstance(values, list):
        return None
    return next((value.lower() for value in reversed(values) if isinstance(value, str)), None)


def _technical_score(result: Mapping[str, Any]) -> float | None:
    analysis = result.get("analysis")
    if not isinstance(analysis, Mapping):
        return None
    scores: list[float] = []
    rsi = _latest(analysis.get("rsi"))
    if rsi is not None:
        scores.append(_clamp(rsi))
    macd = analysis.get("macd")
    histogram = _latest(macd.get("histogram")) if isinstance(macd, Mapping) else None
    if histogram is not None:
        scores.append(65.0 if histogram > 0 else 35.0 if histogram < 0 else 50.0)
    supertrend = analysis.get("supertrend")
    direction = _latest_direction(supertrend.get("direction")) if isinstance(supertrend, Mapping) else None
    if direction is not None:
        scores.append(70.0 if direction == "bullish" else 30.0)
    breakout = _latest(analysis.get("volume_breakout"))
    if breakout is not None:
        scores.append(65.0 if breakout else 50.0)
    return _average(scores)


def _options_score(result: Mapping[str, Any]) -> float | None:
    analysis = result.get("analysis")
    pcr = analysis.get("pcr", {}).get("open_interest") if isinstance(analysis, Mapping) else None
    return _clamp(float(pcr) * 50) if isinstance(pcr, (int, float)) else None


def _ml_score(result: Mapping[str, Any]) -> float | None:
    prediction = result.get("prediction")
    probability = prediction.get("probability") if isinstance(prediction, Mapping) else None
    return _clamp(float(probability) * 100) if isinstance(probability, (int, float)) else None


def _market_score(result: Mapping[str, Any]) -> float | None:
    analysis = result.get("analysis")
    direction = analysis.get("direction") if isinstance(analysis, Mapping) else None
    return {"BULLISH": 100.0, "BEARISH": 0.0, "NEUTRAL": 50.0}.get(direction)


def _news_score(result: Mapping[str, Any]) -> float | None:
    articles = result.get("articles")
    if not isinstance(articles, list):
        return None
    sentiment = [article.get("sentiment", {}).get("score") for article in articles if isinstance(article, Mapping)]
    scores = [50 + float(value) * 50 for value in sentiment if isinstance(value, (int, float))]
    return _average(scores) if scores else 50.0


def _fundamental_score(result: Mapping[str, Any]) -> float | None:
    metrics = result.get("analysis")
    if not isinstance(metrics, Mapping):
        return None
    scores: list[float] = []
    if isinstance(metrics.get("profit"), (int, float)):
        scores.append(70.0 if metrics["profit"] > 0 else 30.0)
    if isinstance(metrics.get("roe"), (int, float)):
        scores.append(_clamp(50 + float(metrics["roe"])))
    growth = metrics.get("growth", {})
    if isinstance(growth, Mapping) and isinstance(growth.get("profit"), (int, float)):
        scores.append(_clamp(50 + float(growth["profit"])))
    debt = metrics.get("debt_to_equity")
    if isinstance(debt, (int, float)):
        scores.append(_clamp(80 - float(debt) * 20))
    valuation = metrics.get("valuation", {})
    pe = valuation.get("pe") if isinstance(valuation, Mapping) else None
    if isinstance(pe, (int, float)) and pe > 0:
        scores.append(70.0 if 10 <= pe <= 25 else 50.0 if pe <= 40 else 30.0)
    return _average(scores)


_SCORERS = {
    "technical": _technical_score,
    "options": _options_score,
    "ml": _ml_score,
    "market_regime": _market_score,
    "news": _news_score,
    "fundamental": _fundamental_score,
}


def score_band(score: float) -> str:
    for upper, label in SCORE_BANDS:
        if score <= upper:
            return label
    return SCORE_BANDS[-1][1]


def score_signal(state: Mapping[str, Any], weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Calculate a 0-100 score, renormalizing weights for available components."""
    applied_weights = dict(weights or DEFAULT_WEIGHTS)
    if set(applied_weights) != set(DEFAULT_WEIGHTS) or any(value < 0 for value in applied_weights.values()):
        raise ValueError("weights must contain exactly the six non-negative signal components")
    component_scores = {name: scorer(state.get("market" if name == "market_regime" else name, {})) for name, scorer in _SCORERS.items()}
    available = {name: value for name, value in component_scores.items() if value is not None}
    total_weight = sum(applied_weights[name] for name in available)
    score = sum(available[name] * applied_weights[name] for name in available) / total_weight if total_weight else 50.0
    score = round(_clamp(score), 2)
    return {
        "score": score,
        "band": score_band(score),
        "components": {name: round(value, 2) for name, value in available.items()},
        "weights": applied_weights,
        "available_components": list(available),
        "weight_source": "default_until_backtested" if weights is None else "configured",
    }