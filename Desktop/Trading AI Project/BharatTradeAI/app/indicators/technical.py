"""Technical indicators calculated from OHLCV candles."""

from collections.abc import Iterable, Sequence
from math import sqrt
from typing import Any, Mapping


def _ema(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return result
    current = sum(values[:period]) / period
    result[period - 1] = current
    multiplier = 2 / (period + 1)
    for index in range(period, len(values)):
        current = (values[index] - current) * multiplier + current
        result[index] = current
    return result


def compute_ema(values: Sequence[float], period: int = 20) -> list[float | None]:
    return _ema(values, period)


def compute_rsi(values: Sequence[float] | None = None, period: int = 14) -> dict[str, Any] | list[float | None]:
    if values is None:
        return {"indicator": "rsi", "status": "not_implemented"}
    if period <= 0:
        raise ValueError("period must be positive")
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    gains = [max(values[index] - values[index - 1], 0.0) for index in range(1, len(values))]
    losses = [max(values[index - 1] - values[index], 0.0) for index in range(1, len(values))]
    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    def rsi() -> float:
        if average_loss == 0:
            return 100.0 if average_gain else 50.0
        return 100 - (100 / (1 + average_gain / average_loss))

    result[period] = rsi()
    for index in range(period, len(gains)):
        average_gain = (average_gain * (period - 1) + gains[index]) / period
        average_loss = (average_loss * (period - 1) + losses[index]) / period
        result[index + 1] = rsi()
    return result


def compute_macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float | None]]:
    fast_line = _ema(values, fast)
    slow_line = _ema(values, slow)
    macd = [fast_line[index] - slow_line[index] if fast_line[index] is not None and slow_line[index] is not None else None for index in range(len(values))]
    available = [value for value in macd if value is not None]
    signal_values = _ema(available, signal)
    signal_line: list[float | None] = [None] * len(values)
    offset = len(values) - len(available)
    for index, value in enumerate(signal_values):
        signal_line[index + offset] = value
    histogram = [macd[index] - signal_line[index] if macd[index] is not None and signal_line[index] is not None else None for index in range(len(values))]
    return {"macd": macd, "signal": signal_line, "histogram": histogram}


def compute_vwap(high: Sequence[float], low: Sequence[float], close: Sequence[float], volume: Sequence[float]) -> list[float | None]:
    price_volume = 0.0
    total_volume = 0.0
    result: list[float | None] = []
    for high_value, low_value, close_value, volume_value in zip(high, low, close, volume):
        price_volume += ((high_value + low_value + close_value) / 3) * volume_value
        total_volume += volume_value
        result.append(price_volume / total_volume if total_volume else None)
    return result


def _wilder(values: Sequence[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return result
    current = sum(values[:period]) / period
    result[period - 1] = current
    for index in range(period, len(values)):
        current = ((current * (period - 1)) + values[index]) / period
        result[index] = current
    return result


def compute_atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> list[float | None]:
    true_ranges = [high[0] - low[0]] if high else []
    true_ranges.extend(max(high[index] - low[index], abs(high[index] - close[index - 1]), abs(low[index] - close[index - 1])) for index in range(1, len(high)))
    return _wilder(true_ranges, period)


def compute_adx(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> dict[str, list[float | None]]:
    true_ranges = [high[0] - low[0]] if high else []
    plus_dm = [0.0]
    minus_dm = [0.0]
    for index in range(1, len(high)):
        up = high[index] - high[index - 1]
        down = low[index - 1] - low[index]
        true_ranges.append(max(high[index] - low[index], abs(high[index] - close[index - 1]), abs(low[index] - close[index - 1])))
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    atr = _wilder(true_ranges, period)
    plus = _wilder(plus_dm, period)
    minus = _wilder(minus_dm, period)
    plus_di = [100 * plus[index] / atr[index] if atr[index] else None for index in range(len(true_ranges))]
    minus_di = [100 * minus[index] / atr[index] if atr[index] else None for index in range(len(true_ranges))]
    dx = [100 * abs(plus_di[index] - minus_di[index]) / (plus_di[index] + minus_di[index]) if plus_di[index] is not None and minus_di[index] is not None and plus_di[index] + minus_di[index] else None for index in range(len(true_ranges))]
    valid_dx = [value for value in dx if value is not None]
    adx_values = _wilder(valid_dx, period)
    adx: list[float | None] = [None] * len(true_ranges)
    offset = len(true_ranges) - len(valid_dx)
    for index, value in enumerate(adx_values):
        adx[index + offset] = value
    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def compute_bollinger(values: Sequence[float], period: int = 20, deviations: float = 2.0) -> dict[str, list[float | None]]:
    middle: list[float | None] = [None] * len(values)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1:index + 1]
        average = sum(window) / period
        standard_deviation = sqrt(sum((value - average) ** 2 for value in window) / period)
        middle[index] = average
        upper[index] = average + deviations * standard_deviation
        lower[index] = average - deviations * standard_deviation
    return {"middle": middle, "upper": upper, "lower": lower}


def compute_supertrend(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 10, multiplier: float = 3.0) -> dict[str, list[float | str | None]]:
    atr = compute_atr(high, low, close, period)
    trend: list[float | None] = [None] * len(close)
    direction: list[str | None] = [None] * len(close)
    for index in range(len(close)):
        if atr[index] is None:
            continue
        midpoint = (high[index] + low[index]) / 2
        upper_band = midpoint + multiplier * atr[index]
        lower_band = midpoint - multiplier * atr[index]
        previous = trend[index - 1] if index else None
        trend[index] = lower_band if previous is None or close[index] >= previous else upper_band
        direction[index] = "bullish" if close[index] >= trend[index] else "bearish"
    return {"supertrend": trend, "direction": direction}


def compute_support_resistance(low: Sequence[float], high: Sequence[float], period: int = 20) -> dict[str, list[float | None]]:
    support: list[float | None] = [None] * len(low)
    resistance: list[float | None] = [None] * len(high)
    for index in range(period - 1, len(low)):
        support[index] = min(low[index - period + 1:index + 1])
        resistance[index] = max(high[index - period + 1:index + 1])
    return {"support": support, "resistance": resistance}


def compute_volume_breakout(close: Sequence[float], volume: Sequence[float], period: int = 20, multiplier: float = 1.5) -> list[bool | None]:
    result: list[bool | None] = [None] * len(close)
    for index in range(period, len(close)):
        average_volume = sum(volume[index - period:index]) / period
        result[index] = volume[index] > average_volume * multiplier and close[index] > close[index - 1]
    return result


def _rows(candles: Iterable[Mapping[str, Any]]) -> tuple[list[float], list[float], list[float], list[float]]:
    rows = list(candles)
    return ([float(row["high"]) for row in rows], [float(row["low"]) for row in rows], [float(row["close"]) for row in rows], [float(row.get("volume", 0)) for row in rows])


def compute_technical_indicators(candles: Iterable[Mapping[str, Any]], period: int = 14) -> dict[str, Any]:
    """Compute all Iteration 5 indicators and return their full time series."""
    high, low, close, volume = _rows(candles)
    if not close:
        raise ValueError("candles must contain at least one row")
    return {
        "rsi": compute_rsi(close, period),
        "ema": compute_ema(close, period),
        "macd": compute_macd(close),
        "vwap": compute_vwap(high, low, close, volume),
        "atr": compute_atr(high, low, close, period),
        "adx": compute_adx(high, low, close, period),
        "bollinger": compute_bollinger(close, period),
        "supertrend": compute_supertrend(high, low, close, period),
        "support_resistance": compute_support_resistance(low, high, period),
        "volume_breakout": compute_volume_breakout(close, volume, period),
    }
