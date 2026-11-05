"""Cached public IPO and market data from Xflot."""

import threading
import time
from typing import Any

import requests

X_FLOT_API_BASE = "https://www.xflot.com/api/public/market"
CACHE_TTL_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 10
_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()


def fetch_market_data(endpoint: str) -> dict[str, Any]:
    """Fetch one public endpoint, caching it to honor Xflot's polling limit."""
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(endpoint)
        if cached is not None and now - cached[0] < CACHE_TTL_SECONDS:
            return cached[1]

        response = requests.get(
            f"{X_FLOT_API_BASE}/{endpoint}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise ValueError("The Xflot market API returned an invalid response.")
        if not isinstance(data.get("disclaimer"), str) or not data["disclaimer"].strip():
            raise ValueError("The Xflot market API response did not include its disclaimer.")
        _cache[endpoint] = (time.monotonic(), data)
        return data
