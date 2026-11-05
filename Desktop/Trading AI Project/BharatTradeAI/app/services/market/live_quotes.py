"""Best-effort public intraday quotes from Yahoo Finance's chart endpoint."""

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import requests

CACHE_TTL_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 10
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_FINANCE_URL = "https://finance.yahoo.com/"
WATCHLIST = (
    {"symbol": "NIFTY 50", "name": "NIFTY 50", "exchange": "NSE", "type": "INDEX", "yahoo_symbol": "^NSEI"},
    {"symbol": "BANKNIFTY", "name": "NIFTY BANK", "exchange": "NSE", "type": "INDEX", "yahoo_symbol": "^NSEBANK"},
    {"symbol": "RELIANCE", "name": "Reliance Industries", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "RELIANCE.NS"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "HDFCBANK.NS"},
    {"symbol": "TCS", "name": "Tata Consultancy Services", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "TCS.NS"},
    {"symbol": "INFY", "name": "Infosys", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "INFY.NS"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "ICICIBANK.NS"},
    {"symbol": "SBIN", "name": "State Bank of India", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "SBIN.NS"},
    {"symbol": "ITC", "name": "ITC", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "ITC.NS"},
    {"symbol": "LT", "name": "Larsen & Toubro", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "LT.NS"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "BHARTIARTL.NS"},
    {"symbol": "AXISBANK", "name": "Axis Bank", "exchange": "NSE", "type": "EQUITY", "yahoo_symbol": "AXISBANK.NS"},
)

_cache: tuple[float, dict[str, Any]] | None = None
_cache_lock = threading.Lock()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _market_timestamp(epoch: Any) -> str | None:
    timestamp = _number(epoch)
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _last_available(values: Any) -> tuple[int, float] | None:
    if not isinstance(values, list):
        return None
    for index in range(len(values) - 1, -1, -1):
        value = _number(values[index])
        if value is not None:
            return index, value
    return None


def _fetch_chart_result(yahoo_symbol: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{YAHOO_CHART_URL}/{quote(yahoo_symbol, safe='')}"
    response = requests.get(
        url,
        params=params,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict) or chart.get("error"):
        raise ValueError(f"No chart data returned for {yahoo_symbol}.")

    results = chart.get("result")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise ValueError(f"No chart data returned for {yahoo_symbol}.")
    return results[0]


def _fetch_quote(instrument: dict[str, str]) -> dict[str, Any]:
    result = _fetch_chart_result(instrument["yahoo_symbol"], {"range": "1d", "interval": "1m"})
    meta = result.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Quote details are missing for {instrument['symbol']}.")

    indicators = result.get("indicators")
    quote_series = indicators.get("quote") if isinstance(indicators, dict) else None
    series = quote_series[0] if isinstance(quote_series, list) and quote_series and isinstance(quote_series[0], dict) else {}
    timestamp_series = result.get("timestamp")
    timestamp_series = timestamp_series if isinstance(timestamp_series, list) else []

    last_close = _last_available(series.get("close"))
    last_price = _number(meta.get("regularMarketPrice"))
    if last_price is None and last_close is not None:
        last_price = last_close[1]
    if last_price is None:
        raise ValueError(f"Latest price is missing for {instrument['symbol']}.")

    previous_close = _number(meta.get("chartPreviousClose"))
    if previous_close is None:
        previous_close = _number(meta.get("previousClose"))
    if previous_close is not None and previous_close <= 0:
        previous_close = None

    change = last_price - previous_close if previous_close is not None else None
    change_pct = (change / previous_close * 100) if change is not None else None
    return {
        "symbol": instrument["symbol"],
        "name": instrument["name"],
        "exchange": instrument["exchange"],
        "type": instrument["type"],
        "lastPrice": last_price,
        "change": change,
        "changePct": change_pct,
        "open": _number(meta.get("regularMarketDayOpen")) or _number(meta.get("regularMarketOpen")),
        "high": _number(meta.get("regularMarketDayHigh")) or _number(meta.get("regularMarketHigh")),
        "low": _number(meta.get("regularMarketDayLow")) or _number(meta.get("regularMarketLow")),
        "previousClose": previous_close,
        "volume": _number(meta.get("regularMarketVolume")),
        "timestamp": _market_timestamp(meta.get("regularMarketTime"))
        or (_market_timestamp(timestamp_series[last_close[0]]) if last_close and last_close[0] < len(timestamp_series) else None),
    }


def fetch_live_quotes() -> dict[str, Any]:
    """Fetch a 1-minute intraday snapshot and cache it to limit public API polling."""
    global _cache

    with _cache_lock:
        now = time.monotonic()
        if _cache is not None and now - _cache[0] < CACHE_TTL_SECONDS:
            return _cache[1]

        quotes = []
        warnings = []
        for instrument in WATCHLIST:
            try:
                quotes.append(_fetch_quote(instrument))
            except (requests.RequestException, ValueError) as exc:
                warnings.append(f"{instrument['symbol']}: {exc}")

        result = {
            "status": "ok" if quotes else "error",
            "provider": "Yahoo Finance",
            "providerUrl": YAHOO_FINANCE_URL,
            "refreshedAt": datetime.now(timezone.utc).isoformat(),
            "quotes": quotes,
            "warnings": warnings,
        }
        if not quotes:
            result["message"] = "Yahoo Finance could not return any market quotes. Try again later."

        _cache = (time.monotonic(), result)
        return result
