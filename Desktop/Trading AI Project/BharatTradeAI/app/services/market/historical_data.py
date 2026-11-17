"""Daily OHLCV history from the same Yahoo chart endpoint used by live quotes."""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from app.db.database import SessionLocal
from app.services.market.live_quotes import (
    AI_STOCK_HISTORY_UNIVERSE as WATCHLIST,
    _fetch_chart_result,
    _number,
)
from app.services.market.models import Candle, MarketDataSync

TIMEFRAME = "1d"
HISTORY_RANGE = "5y"
INTRADAY_RANGES = {"1m": "1d", "5m": "5d", "15m": "1mo"}
INTRADAY_CACHE_TTL_SECONDS = 60
SYNC_COOLDOWN = timedelta(minutes=15)
_sync_lock = threading.Lock()
_intraday_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_intraday_cache_lock = threading.Lock()


class SyncCooldownError(Exception):
    """Raised when a manual history sync is requested too soon."""


class SyncInProgressError(Exception):
    """Raised when a history sync is already running."""


def _instrument(symbol: str) -> dict[str, str] | None:
    return next((row for row in WATCHLIST if row["symbol"] == symbol.upper()), None)


def _utc_datetime(epoch: Any) -> datetime | None:
    timestamp = _number(epoch)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _database_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _response_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_daily_bars(result: dict[str, Any]) -> list[dict[str, Any]]:
    indicators = result.get("indicators")
    quote_series = indicators.get("quote") if isinstance(indicators, dict) else None
    series = (
        quote_series[0]
        if isinstance(quote_series, list) and quote_series and isinstance(quote_series[0], dict)
        else {}
    )
    timestamps = result.get("timestamp")
    if not isinstance(timestamps, list):
        return []

    bars: list[dict[str, Any]] = []
    for index, timestamp in enumerate(timestamps):
        candle_time = _utc_datetime(timestamp)
        if candle_time is None:
            continue

        values: dict[str, Any] = {}
        for field in ("open", "high", "low", "close"):
            field_series = series.get(field)
            raw_value = field_series[index] if isinstance(field_series, list) and index < len(field_series) else None
            values[field] = _number(raw_value)
        if any(values[field] is None for field in ("open", "high", "low", "close")):
            continue

        volume_series = series.get("volume")
        raw_volume = volume_series[index] if isinstance(volume_series, list) and index < len(volume_series) else None
        volume_value = _number(raw_volume)
        volume = int(volume_value) if volume_value is not None and volume_value >= 0 else None
        bars.append({"candle_time": candle_time, **values, "volume": volume})

    return sorted(bars, key=lambda bar: bar["candle_time"])


def _fetch_daily_bars(instrument: dict[str, str]) -> list[dict[str, Any]]:
    result = _fetch_chart_result(
        instrument["yahoo_symbol"],
        {"range": HISTORY_RANGE, "interval": TIMEFRAME},
    )
    bars = _parse_daily_bars(result)
    if not bars:
        raise ValueError(f"No valid daily OHLCV data returned for {instrument['symbol']}.")
    return bars


def _persist_bars_and_sync(
    instrument: dict[str, str], bars: list[dict[str, Any]], synced_at: datetime
) -> None:
    db = SessionLocal()
    try:
        existing = (
            db.query(Candle)
            .filter(
                Candle.symbol == instrument["symbol"],
                Candle.exchange == instrument["exchange"],
                Candle.timeframe == TIMEFRAME,
            )
            .all()
        )
        by_time = {_database_datetime(row.candle_time): row for row in existing}

        for bar in bars:
            candle_time = _database_datetime(bar["candle_time"])
            candle = by_time.get(candle_time)
            if candle is None:
                candle = Candle(
                    symbol=instrument["symbol"],
                    exchange=instrument["exchange"],
                    timeframe=TIMEFRAME,
                    candle_time=candle_time,
                )
                db.add(candle)
            candle.open = bar["open"]
            candle.high = bar["high"]
            candle.low = bar["low"]
            candle.close = bar["close"]
            candle.volume = bar["volume"]

        sync_state = (
            db.query(MarketDataSync)
            .filter_by(
                symbol=instrument["symbol"],
                exchange=instrument["exchange"],
                timeframe=TIMEFRAME,
            )
            .one_or_none()
        )
        if sync_state is None:
            sync_state = MarketDataSync(
                symbol=instrument["symbol"],
                exchange=instrument["exchange"],
                timeframe=TIMEFRAME,
            )
            db.add(sync_state)
        sync_state.source = "Yahoo Finance"
        sync_state.status = "ok"
        sync_state.last_synced_at = synced_at
        sync_state.latest_market_time = bars[-1]["candle_time"]
        sync_state.rows_stored = len(bars)
        sync_state.error = None
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()


def _persist_sync_error(instrument: dict[str, str], message: str, synced_at: datetime) -> None:
    db = SessionLocal()
    try:
        sync_state = (
            db.query(MarketDataSync)
            .filter_by(
                symbol=instrument["symbol"],
                exchange=instrument["exchange"],
                timeframe=TIMEFRAME,
            )
            .one_or_none()
        )
        if sync_state is None:
            sync_state = MarketDataSync(
                symbol=instrument["symbol"],
                exchange=instrument["exchange"],
                timeframe=TIMEFRAME,
            )
            db.add(sync_state)
        sync_state.source = "Yahoo Finance"
        sync_state.status = "error"
        sync_state.last_synced_at = synced_at
        sync_state.error = message[:500]
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    finally:
        db.close()


def sync_daily_history(*, manual: bool = False) -> dict[str, Any]:
    """Refresh five years of daily bars for model warm-up and backtesting."""
    if not _sync_lock.acquire(blocking=False):
        raise SyncInProgressError("A daily market history sync is already in progress.")

    try:
        synced_at = datetime.now(timezone.utc)
        if manual:
            db = SessionLocal()
            try:
                last_sync = db.query(func.max(MarketDataSync.last_synced_at)).scalar()
            finally:
                db.close()
            if last_sync is not None:
                if last_sync.tzinfo is None:
                    last_sync = last_sync.replace(tzinfo=timezone.utc)
                if synced_at - last_sync < SYNC_COOLDOWN:
                    raise SyncCooldownError("Market history was synced recently. Please try again later.")

        warnings: list[str] = []
        synced_symbols: list[str] = []
        rate_limited = False
        for instrument in WATCHLIST:
            if rate_limited:
                message = "Skipped after Yahoo Finance rate-limited this sync."
                warnings.append(f"{instrument['symbol']}: {message}")
                _persist_sync_error(instrument, message, synced_at)
                continue
            try:
                bars = _fetch_daily_bars(instrument)
                _persist_bars_and_sync(instrument, bars, synced_at)
                synced_symbols.append(instrument["symbol"])
            except (requests.RequestException, ValueError, SQLAlchemyError) as exc:
                message = str(exc) or exc.__class__.__name__
                warnings.append(f"{instrument['symbol']}: {message}")
                if not isinstance(exc, SQLAlchemyError):
                    _persist_sync_error(instrument, message, synced_at)
                if (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code == 429
                ):
                    rate_limited = True

        status = "ok" if not warnings else "partial" if synced_symbols else "error"
        return {
            "status": status,
            "provider": "Yahoo Finance",
            "timeframe": TIMEFRAME,
            "range": HISTORY_RANGE,
            "syncedAt": synced_at.isoformat(),
            "syncedSymbols": synced_symbols,
            "warnings": warnings,
        }
    finally:
        _sync_lock.release()


def get_daily_history(symbol: str, limit: int = 100) -> dict[str, Any]:
    """Read persisted daily bars and calculate previous closes for display."""
    instrument = _instrument(symbol)
    if instrument is None:
        raise ValueError(f"Unsupported market symbol: {symbol}.")

    db = SessionLocal()
    try:
        rows = (
            db.query(Candle)
            .filter(
                Candle.symbol == instrument["symbol"],
                Candle.exchange == instrument["exchange"],
                Candle.timeframe == TIMEFRAME,
            )
            .order_by(Candle.candle_time.desc())
            .limit(limit + 1)
            .all()
        )
        sync_state = (
            db.query(MarketDataSync)
            .filter_by(
                symbol=instrument["symbol"],
                exchange=instrument["exchange"],
                timeframe=TIMEFRAME,
            )
            .one_or_none()
        )
    finally:
        db.close()

    ascending = list(reversed(rows))
    previous_close = ascending[0].close if len(rows) > limit else None
    bars = ascending[-limit:]
    candles: list[dict[str, Any]] = []
    for row in bars:
        candles.append(
            {
                "timestamp": _response_datetime(row.candle_time),
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "previousClose": previous_close,
                "volume": row.volume,
            }
        )
        previous_close = row.close

    latest_market_time = bars[-1].candle_time if bars else None
    latest_market_time = (
        latest_market_time.replace(tzinfo=timezone.utc)
        if latest_market_time is not None and latest_market_time.tzinfo is None
        else latest_market_time
    )
    freshness = "unknown"
    if latest_market_time is not None:
        freshness = (
            "stale"
            if datetime.now(timezone.utc) - latest_market_time.astimezone(timezone.utc) > timedelta(days=4)
            else "fresh"
        )
    return {
        "status": "ok" if candles else "no_data",
        "provider": "Yahoo Finance",
        "symbol": instrument["symbol"],
        "exchange": instrument["exchange"],
        "timeframe": TIMEFRAME,
        "range": HISTORY_RANGE,
        "count": len(candles),
        "lastSyncedAt": _response_datetime(sync_state.last_synced_at) if sync_state else None,
        "latestMarketTime": _response_datetime(latest_market_time),
        "freshness": freshness,
        "syncStatus": sync_state.status if sync_state else "never",
        "syncError": sync_state.error if sync_state else None,
        "candles": candles,
    }


def get_intraday_history(symbol: str, interval: str = "5m") -> dict[str, Any]:
    """Fetch a short intraday OHLCV window directly from the existing Yahoo adapter."""
    instrument = _instrument(symbol)
    if instrument is None:
        raise ValueError(f"Unsupported market symbol: {symbol}.")
    if interval not in INTRADAY_RANGES:
        raise ValueError("Intraday interval must be 1m, 5m, or 15m.")

    cache_key = (instrument["symbol"], interval)
    now_monotonic = time.monotonic()
    with _intraday_cache_lock:
        cached = _intraday_cache.get(cache_key)
        if cached and now_monotonic - cached[0] < INTRADAY_CACHE_TTL_SECONDS:
            return {
                **cached[1],
                "cacheAgeSeconds": max(0, int(now_monotonic - cached[0])),
            }

    try:
        result = _fetch_chart_result(
            instrument["yahoo_symbol"],
            {"range": INTRADAY_RANGES[interval], "interval": interval},
        )
        bars = _parse_daily_bars(result)
        if not bars:
            raise ValueError(f"No valid intraday OHLCV data returned for {instrument['symbol']}.")
    except (requests.RequestException, ValueError):
        if cached is None:
            raise
        return {
            **cached[1],
            "freshness": "stale",
            "warning": "Yahoo Finance is unavailable; showing the last successful intraday bars.",
            "cacheAgeSeconds": max(0, int(now_monotonic - cached[0])),
        }
    retrieved_at = datetime.now(timezone.utc)
    bars = [bar for bar in bars if bar["candle_time"].astimezone(timezone.utc) <= retrieved_at]
    latest = bars[-1]["candle_time"] if bars else None
    candles = [
        {
            "timestamp": _response_datetime(bar["candle_time"]),
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": bar["volume"],
        }
        for bar in bars
    ]
    if latest is None:
        freshness = "future_timestamp"
        provider_timestamp = None
        warning = "Yahoo Finance returned only future-dated intraday bars; they were excluded."
    else:
        provider_timestamp = _response_datetime(latest)
        age = retrieved_at - latest.astimezone(timezone.utc)
        freshness = "fresh" if age <= timedelta(minutes=15) else "stale"
        warning = "Yahoo Finance intraday bars may be delayed or unavailable; this is not an exchange feed." if freshness != "fresh" else None
    response = {
        "status": "ok",
        "provider": "Yahoo Finance",
        "symbol": instrument["symbol"],
        "exchange": instrument["exchange"],
        "interval": interval,
        "range": INTRADAY_RANGES[interval],
        "retrievedAt": retrieved_at.isoformat(),
        "providerTimestamp": provider_timestamp,
        "freshness": freshness,
        "warning": warning,
        "candles": candles,
    }
    with _intraday_cache_lock:
        _intraday_cache[cache_key] = (time.monotonic(), response)
    return response
