"""Best-effort public intraday quotes from Yahoo Finance's chart endpoint."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import requests

CACHE_TTL_SECONDS = 60
AI_UNIVERSE_CACHE_TTL_SECONDS = 300
AI_QUOTE_FETCH_WORKERS = 8
REQUEST_TIMEOUT_SECONDS = 10
FUTURE_CLOCK_SKEW_TOLERANCE = timedelta(minutes=2)
QUOTE_FRESH_FOR = timedelta(minutes=5)
QUOTE_STALE_AFTER = timedelta(minutes=15)
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
_NIFTY_50_INSTRUMENTS = tuple(
    {
        "symbol": symbol,
        "name": name,
        "exchange": "NSE",
        "type": "EQUITY",
        "yahoo_symbol": f"{symbol}.NS",
    }
    for symbol, name in (
        ("ADANIENT", "Adani Enterprises"),
        ("ADANIPORTS", "Adani Ports and Special Economic Zone"),
        ("APOLLOHOSP", "Apollo Hospitals Enterprise"),
        ("ASIANPAINT", "Asian Paints"),
        ("AXISBANK", "Axis Bank"),
        ("BAJAJ-AUTO", "Bajaj Auto"),
        ("BAJFINANCE", "Bajaj Finance"),
        ("BAJAJFINSV", "Bajaj Finserv"),
        ("BEL", "Bharat Electronics"),
        ("BHARTIARTL", "Bharti Airtel"),
        ("CIPLA", "Cipla"),
        ("COALINDIA", "Coal India"),
        ("DRREDDY", "Dr. Reddy's Laboratories"),
        ("EICHERMOT", "Eicher Motors"),
        ("ETERNAL", "Eternal"),
        ("GRASIM", "Grasim Industries"),
        ("HCLTECH", "HCL Technologies"),
        ("HDFCBANK", "HDFC Bank"),
        ("HDFCLIFE", "HDFC Life Insurance"),
        ("HINDALCO", "Hindalco Industries"),
        ("HINDUNILVR", "Hindustan Unilever"),
        ("ICICIBANK", "ICICI Bank"),
        ("INDUSINDBK", "IndusInd Bank"),
        ("INFY", "Infosys"),
        ("ITC", "ITC"),
        ("JIOFIN", "Jio Financial Services"),
        ("JSWSTEEL", "JSW Steel"),
        ("KOTAKBANK", "Kotak Mahindra Bank"),
        ("LT", "Larsen & Toubro"),
        ("LTIM", "LTIMindtree"),
        ("M&M", "Mahindra & Mahindra"),
        ("MARUTI", "Maruti Suzuki India"),
        ("NESTLEIND", "Nestle India"),
        ("NTPC", "NTPC"),
        ("ONGC", "Oil and Natural Gas Corporation"),
        ("POWERGRID", "Power Grid Corporation of India"),
        ("RELIANCE", "Reliance Industries"),
        ("SBILIFE", "SBI Life Insurance"),
        ("SBIN", "State Bank of India"),
        ("SHRIRAMFIN", "Shriram Finance"),
        ("SUNPHARMA", "Sun Pharmaceutical Industries"),
        ("TATACONSUM", "Tata Consumer Products"),
        ("TMPV", "Tata Motors Passenger Vehicles"),
        ("TATASTEEL", "Tata Steel"),
        ("TCS", "Tata Consultancy Services"),
        ("TECHM", "Tech Mahindra"),
        ("TITAN", "Titan Company"),
        ("TRENT", "Trent"),
        ("ULTRACEMCO", "UltraTech Cement"),
        ("WIPRO", "Wipro"),
    )
)
_CORE_AI_SYMBOLS = (
    "RELIANCE",
    "HDFCBANK",
    "TCS",
    "INFY",
    "ICICIBANK",
    "SBIN",
    "ITC",
    "LT",
    "BHARTIARTL",
    "AXISBANK",
)
_NIFTY_50_BY_SYMBOL = {instrument["symbol"]: instrument for instrument in _NIFTY_50_INSTRUMENTS}
AI_STOCK_UNIVERSE = tuple(
    _NIFTY_50_BY_SYMBOL[symbol] for symbol in _CORE_AI_SYMBOLS
) + tuple(
    instrument
    for instrument in _NIFTY_50_INSTRUMENTS
    if instrument["symbol"] not in _CORE_AI_SYMBOLS
)
AI_STOCK_HISTORY_UNIVERSE = (WATCHLIST[0], WATCHLIST[1], *AI_STOCK_UNIVERSE)

_cache: tuple[float, dict[str, Any]] | None = None
_ai_universe_cache: tuple[float, dict[str, Any]] | None = None
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


def _quote_freshness(timestamp: str | None, now: datetime) -> tuple[str, str | None]:
    if timestamp is None:
        return "unavailable", "Provider did not return a market timestamp."
    market_time = datetime.fromisoformat(timestamp)
    if market_time.tzinfo is None:
        return "unavailable", "Provider timestamp did not include a timezone."
    age = now.astimezone(timezone.utc) - market_time.astimezone(timezone.utc)
    if age < timedelta(0):
        if -age <= FUTURE_CLOCK_SKEW_TOLERANCE:
            return "clock_skew", "Provider timestamp is slightly ahead; using server time for display."
        return "future_timestamp", "Provider timestamp is ahead of the server clock."
    if age <= QUOTE_FRESH_FOR:
        return "fresh", None
    if age <= QUOTE_STALE_AFTER:
        return "delayed", "Provider quote is more than five minutes old."
    return "stale", "Provider quote is more than fifteen minutes old."


def _fetch_quote(instrument: dict[str, str], *, now: datetime | None = None) -> dict[str, Any]:
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
    provider_timestamp = _market_timestamp(meta.get("regularMarketTime")) or (
        _market_timestamp(timestamp_series[last_close[0]])
        if last_close and last_close[0] < len(timestamp_series)
        else None
    )
    fetched_at = now or datetime.now(timezone.utc)
    timestamp_status, timestamp_warning = _quote_freshness(provider_timestamp, fetched_at)
    timestamp = (
        None if timestamp_status == "future_timestamp"
        else fetched_at.isoformat() if timestamp_status == "clock_skew"
        else provider_timestamp
    )
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
        "timestamp": timestamp,
        "providerTimestamp": provider_timestamp,
        "timestampStatus": timestamp_status,
        "timestampWarning": timestamp_warning,
    }


def fetch_live_quotes() -> dict[str, Any]:
    """Fetch a 1-minute intraday snapshot and cache it to limit public API polling."""
    global _cache

    with _cache_lock:
        now = time.monotonic()
        if _cache is not None and now - _cache[0] < CACHE_TTL_SECONDS:
            return _cache[1]

        result = _fetch_quotes_for(WATCHLIST)
        _cache = (time.monotonic(), result)
        return result


def fetch_ai_stock_quotes() -> dict[str, Any]:
    """Fetch the expanded AI universe less frequently to respect the public quote provider."""
    global _ai_universe_cache

    with _cache_lock:
        now = time.monotonic()
        if (
            _ai_universe_cache is not None
            and now - _ai_universe_cache[0] < AI_UNIVERSE_CACHE_TTL_SECONDS
        ):
            age = max(0, int(now - _ai_universe_cache[0]))
            return {
                **_ai_universe_cache[1],
                "cacheStatus": "fresh",
                "cacheAgeSeconds": age,
            }

        result = _fetch_quotes_for(
            AI_STOCK_UNIVERSE,
            include_unavailable=True,
        )
        _ai_universe_cache = (time.monotonic(), result)
        return result


def _unavailable_quote(instrument: dict[str, str], warning: str) -> dict[str, Any]:
    return {
        "symbol": instrument["symbol"],
        "name": instrument["name"],
        "exchange": instrument["exchange"],
        "type": instrument["type"],
        "lastPrice": None,
        "change": None,
        "changePct": None,
        "open": None,
        "high": None,
        "low": None,
        "previousClose": None,
        "volume": None,
        "timestamp": None,
        "providerTimestamp": None,
        "timestampStatus": "unavailable",
        "timestampWarning": warning,
    }


def _fetch_quotes_for(
    instruments: tuple[dict[str, str], ...],
    *,
    include_unavailable: bool = False,
) -> dict[str, Any]:
    quotes_by_symbol = {}
    warnings = []
    fetched_at = datetime.now(timezone.utc)

    def fetch(instrument: dict[str, str]) -> tuple[str, dict[str, Any] | None, str | None]:
        try:
            return instrument["symbol"], _fetch_quote(instrument, now=fetched_at), None
        except (requests.RequestException, ValueError) as exc:
            return instrument["symbol"], None, f"{instrument['symbol']}: {exc}"

    if include_unavailable:
        with ThreadPoolExecutor(max_workers=AI_QUOTE_FETCH_WORKERS) as executor:
            futures = {
                executor.submit(fetch, instrument): instrument
                for instrument in instruments
            }
            for future in as_completed(futures):
                instrument = futures[future]
                symbol, quote, warning = future.result()
                if warning:
                    warnings.append(warning)
                if quote is not None:
                    quotes_by_symbol[symbol] = quote
                elif include_unavailable:
                    quotes_by_symbol[symbol] = _unavailable_quote(instrument, warning or "Quote unavailable.")
    else:
        for instrument in instruments:
            symbol, quote, warning = fetch(instrument)
            if warning:
                warnings.append(warning)
            if quote is not None:
                quotes_by_symbol[symbol] = quote

    quotes = [
        quotes_by_symbol[instrument["symbol"]]
        for instrument in instruments
        if instrument["symbol"] in quotes_by_symbol
    ]
    successful_count = sum(1 for quote in quotes if quote["lastPrice"] is not None)

    result = {
        "status": "ok" if successful_count else "error",
        "provider": "Yahoo Finance",
        "providerUrl": YAHOO_FINANCE_URL,
        "refreshedAt": fetched_at.isoformat(),
        "cacheStatus": "fresh",
        "cacheAgeSeconds": 0,
        "lastSuccessfulRefresh": fetched_at.isoformat() if successful_count else None,
        "quotes": quotes,
        "warnings": warnings,
    }
    if not quotes:
        result["message"] = "Yahoo Finance could not return any market quotes. Try again later."
    return result
