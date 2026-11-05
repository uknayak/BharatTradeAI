"""Leakage-safe feature and label construction for OHLCV data."""

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from math import isfinite, log1p
from typing import Any


FEATURE_NAMES = ("return_1", "return_lookback", "range_pct", "volume_ratio", "volatility")
DIRECTION_FEATURE_NAMES = (
    "open_return",
    "high_return",
    "low_return",
    "return_1",
    "momentum_5",
    "momentum_20",
    "range_pct",
    "volume_log",
    "volume_ratio",
    "rsi_14",
    "macd_pct",
    "macd_signal_pct",
    "ema_20_gap",
    "ema_50_gap",
    "sma_200_gap",
    "bollinger_position",
    "bollinger_width",
    "atr_14_pct",
    "volatility_20",
    "news_sentiment",
    "news_available",
    "sector_return_1d",
    "sector_available",
    "market_return_1d",
    "market_available",
)


def _context_value(
    context: Mapping[str, float | None] | Iterable[float | None] | None,
    index: int,
    row: Mapping[str, Any],
) -> float | None:
    if context is None:
        return None
    if isinstance(context, Mapping):
        candle_date = row.get("candle_time", row.get("date"))
        if isinstance(candle_date, (date, datetime)):
            key = candle_date.isoformat()[:10]
        else:
            key = str(candle_date or "")[:10]
        value = context.get(key)
    else:
        values = list(context)
        value = values[index] if index < len(values) else None
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def build_direction_features(
    candles: Iterable[Mapping[str, Any]],
    *,
    news_sentiment: Mapping[str, float | None] | Iterable[float | None] | None = None,
    sector_returns: Mapping[str, float | None] | Iterable[float | None] | None = None,
    market_returns: Mapping[str, float | None] | Iterable[float | None] | None = None,
) -> list[dict[str, float]]:
    """Build end-of-day features; every rolling value uses only current/prior bars."""
    import numpy as np
    import pandas as pd

    rows = list(candles)
    if not rows:
        return []
    if news_sentiment is not None and not isinstance(news_sentiment, Mapping):
        news_sentiment = list(news_sentiment)
    if sector_returns is not None and not isinstance(sector_returns, Mapping):
        sector_returns = list(sector_returns)
    if market_returns is not None and not isinstance(market_returns, Mapping):
        market_returns = list(market_returns)
    frame = pd.DataFrame(
        {
            field: [float(row.get(field) or 0) for row in rows]
            for field in ("open", "high", "low", "close", "volume")
        }
    )
    close = frame["close"].replace(0, np.nan)
    previous_close = close.shift(1)
    returns = close.pct_change(fill_method=None)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    delta = close.diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50)
    middle = close.rolling(20, min_periods=20).mean()
    sma_200 = close.rolling(200, min_periods=200).mean()
    band_std = close.rolling(20, min_periods=20).std(ddof=0)
    upper_band = middle + 2 * band_std
    lower_band = middle - 2 * band_std
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()
    average_volume = frame["volume"].rolling(20, min_periods=20).mean()
    volatility_20 = returns.rolling(20, min_periods=20).std(ddof=0)
    features: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        price = float(frame["close"].iloc[index])
        previous = float(previous_close.iloc[index]) if pd.notna(previous_close.iloc[index]) else float("nan")
        news_value = _context_value(news_sentiment, index, row)
        sector_value = _context_value(sector_returns, index, row)
        market_value = _context_value(market_returns, index, row)
        volume = float(frame["volume"].iloc[index])
        avg_volume = float(average_volume.iloc[index]) if pd.notna(average_volume.iloc[index]) else float("nan")
        values = {
            "open_return": float(frame["open"].iloc[index]) / previous - 1 if previous > 0 else float("nan"),
            "high_return": float(frame["high"].iloc[index]) / previous - 1 if previous > 0 else float("nan"),
            "low_return": float(frame["low"].iloc[index]) / previous - 1 if previous > 0 else float("nan"),
            "return_1": float(returns.iloc[index]) if pd.notna(returns.iloc[index]) else float("nan"),
            "momentum_5": price / float(close.iloc[index - 5]) - 1 if index >= 5 and close.iloc[index - 5] > 0 else float("nan"),
            "momentum_20": price / float(close.iloc[index - 20]) - 1 if index >= 20 and close.iloc[index - 20] > 0 else float("nan"),
            "range_pct": (float(frame["high"].iloc[index]) - float(frame["low"].iloc[index])) / price if price > 0 else float("nan"),
            "volume_log": log1p(max(0.0, volume)),
            "volume_ratio": volume / avg_volume if avg_volume > 0 else float("nan"),
            "rsi_14": float(rsi.iloc[index]) / 100 if pd.notna(rsi.iloc[index]) else float("nan"),
            "macd_pct": float(macd.iloc[index]) / price if price > 0 and pd.notna(macd.iloc[index]) else float("nan"),
            "macd_signal_pct": float(macd_signal.iloc[index]) / price if price > 0 and pd.notna(macd_signal.iloc[index]) else float("nan"),
            "ema_20_gap": price / float(ema_20.iloc[index]) - 1 if pd.notna(ema_20.iloc[index]) and ema_20.iloc[index] > 0 else float("nan"),
            "ema_50_gap": price / float(ema_50.iloc[index]) - 1 if pd.notna(ema_50.iloc[index]) and ema_50.iloc[index] > 0 else float("nan"),
            "sma_200_gap": price / float(sma_200.iloc[index]) - 1 if pd.notna(sma_200.iloc[index]) and sma_200.iloc[index] > 0 else float("nan"),
            "bollinger_position": (price - float(lower_band.iloc[index])) / (float(upper_band.iloc[index]) - float(lower_band.iloc[index])) if pd.notna(upper_band.iloc[index]) and upper_band.iloc[index] != lower_band.iloc[index] else float("nan"),
            "bollinger_width": (float(upper_band.iloc[index]) - float(lower_band.iloc[index])) / price if price > 0 and pd.notna(upper_band.iloc[index]) else float("nan"),
            "atr_14_pct": float(atr.iloc[index]) / price if price > 0 and pd.notna(atr.iloc[index]) else float("nan"),
            "volatility_20": float(volatility_20.iloc[index]) if pd.notna(volatility_20.iloc[index]) else float("nan"),
            "news_sentiment": news_value or 0.0,
            "news_available": float(news_value is not None),
            "sector_return_1d": sector_value or 0.0,
            "sector_available": float(sector_value is not None),
            "market_return_1d": market_value or 0.0,
            "market_available": float(market_value is not None),
        }
        features.append(values)
    return features


def build_direction_training_set(
    candles: Iterable[Mapping[str, Any]],
    *,
    news_sentiment: Mapping[str, float | None] | Iterable[float | None] | None = None,
    sector_returns: Mapping[str, float | None] | Iterable[float | None] | None = None,
    market_returns: Mapping[str, float | None] | Iterable[float | None] | None = None,
) -> dict[str, Any]:
    rows = list(candles)
    features = build_direction_features(
        rows,
        news_sentiment=news_sentiment,
        sector_returns=sector_returns,
        market_returns=market_returns,
    )
    labels = build_labels([float(row["close"]) for row in rows])
    usable = [
        index
        for index, label in enumerate(labels)
        if label is not None and all(isfinite(value) for value in features[index].values())
    ]
    return {
        "features": [features[index] for index in usable],
        "labels": [int(labels[index]) for index in usable],
        "feature_names": list(DIRECTION_FEATURE_NAMES),
        "indices": usable,
    }


def build_features(candles: Iterable[Mapping[str, Any]] | None = None, lookback: int = 5) -> list[dict[str, float]] | dict[str, str]:
    if candles is None:
        return {"module": "features", "status": "not_implemented"}
    rows = list(candles)
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    closes = [float(row["close"]) for row in rows]
    volumes = [float(row.get("volume", 0)) for row in rows]
    features: list[dict[str, float]] = []
    for index in range(len(rows)):
        if index < lookback or closes[index - 1] == 0:
            features.append({name: float("nan") for name in FEATURE_NAMES})
            continue
        returns = [(closes[item] / closes[item - 1]) - 1 for item in range(index - lookback + 1, index + 1)]
        average_volume = sum(volumes[index - lookback:index]) / lookback
        features.append({
            "return_1": returns[-1],
            "return_lookback": (closes[index] / closes[index - lookback]) - 1,
            "range_pct": (float(rows[index]["high"]) - float(rows[index]["low"])) / closes[index],
            "volume_ratio": volumes[index] / average_volume if average_volume else 0.0,
            "volatility": (sum((value - sum(returns) / len(returns)) ** 2 for value in returns) / len(returns)) ** 0.5,
        })
    return features


def build_labels(closes: Iterable[float], horizon: int = 1, threshold: float = 0.0) -> list[int | None]:
    """Create forward-return labels without assigning a label beyond available data."""
    values = [float(value) for value in closes]
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    labels: list[int | None] = [None] * len(values)
    for index in range(len(values) - horizon):
        if values[index] == 0:
            continue
        labels[index] = int((values[index + horizon] / values[index]) - 1 > threshold)
    return labels


def build_training_set(candles: Iterable[Mapping[str, Any]], lookback: int = 5, horizon: int = 1, threshold: float = 0.0) -> dict[str, Any]:
    rows = list(candles)
    features = build_features(rows, lookback)
    labels = build_labels([float(row["close"]) for row in rows], horizon, threshold)
    usable = [index for index, label in enumerate(labels) if label is not None and not any(value != value for value in features[index].values())]
    return {"features": [features[index] for index in usable], "labels": [labels[index] for index in usable], "feature_names": list(FEATURE_NAMES), "indices": usable}
