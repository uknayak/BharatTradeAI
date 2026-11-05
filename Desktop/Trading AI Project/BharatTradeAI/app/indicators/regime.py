"""Market-regime classification from a close or OHLCV series."""

from collections.abc import Iterable, Sequence
from math import sqrt
from statistics import mean, pstdev
from typing import Any, Mapping


def _close_values(data: Iterable[Mapping[str, Any]] | Sequence[float]) -> list[float]:
    rows = list(data)
    if not rows:
        raise ValueError("market data must contain at least one observation")
    if isinstance(rows[0], Mapping):
        return [float(row["close"]) for row in rows]  # type: ignore[index]
    return [float(value) for value in rows]  # type: ignore[arg-type]


def _ema(values: Sequence[float], period: int) -> float:
    current = mean(values[:period])
    multiplier = 2 / (period + 1)
    for value in values[period:]:
        current = (value - current) * multiplier + current
    return current


def classify_market_regime(
    data: Iterable[Mapping[str, Any]] | Sequence[float],
    period: int = 14,
    direction_threshold: float = 0.02,
    high_volatility_threshold: float = 0.015,
    low_volatility_threshold: float = 0.005,
    trend_efficiency_threshold: float = 0.35,
) -> dict[str, Any]:
    """Classify direction, volatility, and market structure.

    Volatility is the standard deviation of percentage returns. Structure uses
    directional efficiency: net movement divided by total absolute movement.
    Thresholds are configurable because they vary by timeframe and instrument.
    """
    close = _close_values(data)
    if period <= 1:
        raise ValueError("period must be greater than one")
    if len(close) <= period:
        raise ValueError("market data must contain more than period observations")
    returns = [(close[index] / close[index - 1]) - 1 for index in range(1, len(close))]
    window_close = close[-(period + 1):]
    window_returns = returns[-period:]
    net_return = (window_close[-1] / window_close[0]) - 1
    short_ema = _ema(window_close, max(2, period // 2))
    long_ema = _ema(window_close, period)
    volatility = pstdev(window_returns)
    total_movement = sum(abs(value) for value in window_returns)
    efficiency = abs(net_return) / total_movement if total_movement else 0.0

    if net_return >= direction_threshold and short_ema >= long_ema:
        direction = "BULLISH"
    elif net_return <= -direction_threshold and short_ema <= long_ema:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    volatility_label = "HIGH VOL" if volatility >= high_volatility_threshold else "LOW VOL" if volatility <= low_volatility_threshold else "NORMAL VOL"
    structure = "TRENDING" if efficiency >= trend_efficiency_threshold else "RANGE"
    return {
        "direction": direction,
        "volatility": volatility_label,
        "structure": structure,
        "regime": f"{direction} / {volatility_label} / {structure}",
        "metrics": {
            "period_return": net_return,
            "volatility": volatility,
            "annualized_volatility": volatility * sqrt(252),
            "directional_efficiency": efficiency,
            "short_ema": short_ema,
            "long_ema": long_ema,
        },
    }