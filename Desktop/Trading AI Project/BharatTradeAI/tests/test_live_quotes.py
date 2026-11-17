import pytest
import requests
from fastapi.testclient import TestClient

from app.main import app


def _chart_response(symbol, last_price, previous_close, timestamp=1793613672):
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": symbol,
                        "regularMarketPrice": last_price,
                        "regularMarketTime": timestamp,
                        "regularMarketDayOpen": previous_close - 10,
                        "regularMarketDayHigh": last_price + 5,
                        "regularMarketDayLow": previous_close - 15,
                        "regularMarketVolume": 123456,
                        "chartPreviousClose": previous_close,
                    },
                    "timestamp": [timestamp - 60, timestamp],
                    "indicators": {
                        "quote": [
                            {
                                "close": [last_price - 1, last_price],
                                "open": [previous_close, previous_close],
                                "high": [last_price, last_price + 5],
                                "low": [previous_close - 15, previous_close - 10],
                                "volume": [1000, 2000],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


def test_public_live_quotes_normalize_data_and_cache_snapshot(monkeypatch):
    from app.services.market import live_quotes

    calls = []

    class DummyResponse:
        def __init__(self, symbol):
            self.symbol = symbol

        def raise_for_status(self):
            return None

        def json(self):
            return _chart_response(self.symbol, 25000, 24900)

    def fake_get(url, params, headers, timeout):
        symbol = url.rsplit("/", 1)[-1]
        calls.append((symbol, params, headers, timeout))
        return DummyResponse(symbol)

    monkeypatch.setattr(live_quotes.requests, "get", fake_get)
    monkeypatch.setattr(live_quotes, "_cache", None)

    first = live_quotes.fetch_live_quotes()
    second = live_quotes.fetch_live_quotes()
    nifty = next(row for row in first["quotes"] if row["symbol"] == "NIFTY 50")

    assert first["status"] == "ok"
    assert first["provider"] == "Yahoo Finance"
    assert first["providerUrl"] == "https://finance.yahoo.com/"
    assert first["warnings"] == []
    assert len(first["quotes"]) == len(live_quotes.WATCHLIST)
    assert nifty["lastPrice"] == 25000
    assert nifty["change"] == 100
    assert nifty["changePct"] == pytest.approx(100 / 24900 * 100)
    assert nifty["open"] == 24890
    assert nifty["high"] == 25005
    assert nifty["low"] == 24885
    assert nifty["previousClose"] == 24900
    assert nifty["volume"] == 123456
    assert nifty["timestamp"] == "2026-11-02T10:01:12+00:00"
    assert all(params == {"range": "1d", "interval": "1m"} for _, params, _, _ in calls)
    assert all(headers["Accept"] == "application/json" and timeout == 10 for _, _, headers, timeout in calls)
    assert second is first
    assert len(calls) == len(live_quotes.WATCHLIST)


def test_public_live_quotes_return_partial_data_and_warnings(monkeypatch):
    from app.services.market import live_quotes

    def fake_get(url, params, headers, timeout):
        if url.endswith("/%5ENSEI"):
            raise requests.Timeout("provider timeout")

        symbol = url.rsplit("/", 1)[-1]

        class DummyResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return _chart_response(symbol, 100, 90)

        return DummyResponse()

    monkeypatch.setattr(live_quotes.requests, "get", fake_get)
    monkeypatch.setattr(live_quotes, "_cache", None)

    response = live_quotes.fetch_live_quotes()

    assert response["status"] == "ok"
    assert len(response["quotes"]) == len(live_quotes.WATCHLIST) - 1
    assert len(response["warnings"]) == 1
    assert response["warnings"][0].startswith("NIFTY 50:")


def test_ai_live_quotes_cover_the_50_stock_universe_and_cache_provider_requests(monkeypatch):
    from app.services.market import live_quotes

    live_quotes._ai_universe_cache = None
    requested_symbols = []

    def fake_quote(instrument, *, now):
        requested_symbols.append(instrument["symbol"])
        return {
            "symbol": instrument["symbol"],
            "name": instrument["name"],
            "exchange": instrument["exchange"],
            "type": instrument["type"],
            "lastPrice": 100.0,
            "changePct": 1.0,
            "timestamp": now.isoformat(),
        }

    monkeypatch.setattr(live_quotes, "_fetch_quote", fake_quote)

    first = live_quotes.fetch_ai_stock_quotes()
    second = live_quotes.fetch_ai_stock_quotes()

    assert len(live_quotes.AI_STOCK_UNIVERSE) == 50
    assert len({item["symbol"] for item in live_quotes.AI_STOCK_UNIVERSE}) == 50
    assert len(first["quotes"]) == 50
    assert len(requested_symbols) == 50
    assert second["cacheStatus"] == "fresh"
    assert second["cacheAgeSeconds"] >= 0
    assert len(requested_symbols) == 50


def test_ai_live_quotes_keep_symbols_visible_when_yahoo_quote_fails(monkeypatch):
    from app.services.market import live_quotes

    live_quotes._ai_universe_cache = None

    def fake_quote(instrument, *, now):
        if instrument["symbol"] == "ADANIENT":
            raise requests.Timeout("provider timeout")
        return {
            "symbol": instrument["symbol"],
            "name": instrument["name"],
            "exchange": instrument["exchange"],
            "type": instrument["type"],
            "lastPrice": 100.0,
            "changePct": 1.0,
            "timestamp": now.isoformat(),
        }

    monkeypatch.setattr(live_quotes, "_fetch_quote", fake_quote)

    response = live_quotes.fetch_ai_stock_quotes()
    unavailable = next(row for row in response["quotes"] if row["symbol"] == "ADANIENT")

    assert len(response["quotes"]) == 50
    assert unavailable["name"] == "Adani Enterprises"
    assert unavailable["lastPrice"] is None
    assert unavailable["timestampStatus"] == "unavailable"
    assert response["warnings"] == ["ADANIENT: provider timeout"]


def test_future_provider_quote_timestamp_is_not_exposed_as_market_time(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from app.services.market import live_quotes

    now = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)
    future_timestamp = int((now + timedelta(minutes=5)).timestamp())

    monkeypatch.setattr(
        live_quotes,
        "_fetch_chart_result",
        lambda *_: _chart_response("RELIANCE.NS", 100, 90, future_timestamp)["chart"]["result"][0],
    )

    quote = live_quotes._fetch_quote(live_quotes.WATCHLIST[2], now=now)

    assert quote["timestamp"] is None
    assert quote["lastPrice"] == 100
    assert quote["changePct"] == pytest.approx(10 / 90 * 100)
    assert quote["providerTimestamp"] == "2026-11-06T10:05:00+00:00"
    assert quote["timestampStatus"] == "future_timestamp"
    assert "ahead" in quote["timestampWarning"]


def test_slight_provider_clock_skew_keeps_quote_and_clamps_display_time(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from app.services.market import live_quotes

    now = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)
    provider_time = int((now + timedelta(seconds=30)).timestamp())
    monkeypatch.setattr(
        live_quotes,
        "_fetch_chart_result",
        lambda *_: _chart_response("RELIANCE.NS", 100, 90, provider_time)["chart"]["result"][0],
    )

    quote = live_quotes._fetch_quote(live_quotes.WATCHLIST[2], now=now)

    assert quote["lastPrice"] == 100
    assert quote["changePct"] == pytest.approx(10 / 90 * 100)
    assert quote["timestamp"] == now.isoformat()
    assert quote["providerTimestamp"] == "2026-11-06T10:00:30+00:00"
    assert quote["timestampStatus"] == "clock_skew"
    assert "server time" in quote["timestampWarning"]


def test_live_quotes_endpoint_returns_public_snapshot(monkeypatch):
    result = {
        "status": "ok",
        "provider": "Yahoo Finance",
        "providerUrl": "https://finance.yahoo.com/",
        "refreshedAt": "2026-11-02T10:00:00+00:00",
        "quotes": [],
        "warnings": [],
    }
    monkeypatch.setattr("app.api.routes.stocks.fetch_live_quotes", lambda: result)

    response = TestClient(app).get("/api/v1/stocks/live")

    assert response.status_code == 200
    assert response.json() == result


def test_live_quotes_endpoint_surfaces_total_provider_failure(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.stocks.fetch_live_quotes",
        lambda: {
            "status": "error",
            "provider": "Yahoo Finance",
            "message": "Yahoo Finance could not return any market quotes.",
        },
    )

    response = TestClient(app).get("/api/v1/stocks/live")

    assert response.status_code == 502
    assert "Yahoo Finance could not return" in response.json()["detail"]
