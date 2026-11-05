from datetime import datetime, timedelta, timezone

import pytest
import requests
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.main import MARKET_TIMEZONE, _next_history_sync, app
from app.services.market import historical_data


def _isolated_sessions(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(historical_data, "SessionLocal", sessions)
    return sessions


def _bar(day: int, close: float) -> dict:
    return {
        "candle_time": datetime(2026, 11, day, 10, tzinfo=timezone.utc),
        "open": close - 1,
        "high": close + 2,
        "low": close - 2,
        "close": close,
        "volume": 1200,
    }


def test_parse_daily_bars_skips_incomplete_rows_and_keeps_missing_volume():
    result = {
        "timestamp": [1793613600, 1793700000, 1793786400],
        "indicators": {
            "quote": [
                {
                    "open": [100, None, 102],
                    "high": [105, 106, 107],
                    "low": [98, 99, 100],
                    "close": [103, 104, 106],
                    "volume": [1000, 2000, None],
                }
            ]
        },
    }

    bars = historical_data._parse_daily_bars(result)

    assert len(bars) == 2
    assert bars[0]["close"] == 103
    assert bars[0]["volume"] == 1000
    assert bars[1]["close"] == 106
    assert bars[1]["volume"] is None


def test_history_scheduler_uses_weekday_after_close():
    now = datetime(2026, 11, 7, 18, 0, tzinfo=MARKET_TIMEZONE)

    assert _next_history_sync(now) == datetime(2026, 11, 9, 16, 15, tzinfo=MARKET_TIMEZONE)


def test_history_sync_upserts_bars_and_returns_previous_close(monkeypatch):
    sessions = _isolated_sessions(monkeypatch)
    instrument = historical_data.WATCHLIST[0]
    monkeypatch.setattr(historical_data, "WATCHLIST", (instrument,))
    bars = [_bar(2, 100), _bar(3, 110)]
    monkeypatch.setattr(historical_data, "_fetch_daily_bars", lambda _: bars)

    first = historical_data.sync_daily_history()
    bars[1] = _bar(3, 111)
    second = historical_data.sync_daily_history()
    result = historical_data.get_daily_history(instrument["symbol"], limit=10)

    db = sessions()
    try:
        rows = db.query(historical_data.Candle).filter_by(
            symbol=instrument["symbol"], timeframe="1d"
        ).all()
    finally:
        db.close()

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert len(rows) == 2
    assert result["count"] == 2
    assert result["candles"][0]["previousClose"] is None
    assert result["candles"][1]["previousClose"] == 100
    assert result["candles"][1]["close"] == 111
    assert result["syncStatus"] == "ok"
    assert result["latestMarketTime"] is not None
    assert result["freshness"] == "fresh"


def test_history_sync_persists_provider_errors_and_reports_partial_status(monkeypatch):
    _isolated_sessions(monkeypatch)
    instrument = historical_data.WATCHLIST[0]
    monkeypatch.setattr(historical_data, "WATCHLIST", (instrument,))

    def fail_fetch(_):
        raise ValueError("No valid daily OHLCV data returned.")

    monkeypatch.setattr(historical_data, "_fetch_daily_bars", fail_fetch)
    result = historical_data.sync_daily_history()
    history = historical_data.get_daily_history(instrument["symbol"])

    assert result["status"] == "error"
    assert result["warnings"]
    assert history["status"] == "no_data"
    assert history["syncStatus"] == "error"
    assert "No valid daily OHLCV" in history["syncError"]


def test_manual_history_sync_enforces_cooldown(monkeypatch):
    sessions = _isolated_sessions(monkeypatch)
    instrument = historical_data.WATCHLIST[0]
    db = sessions()
    db.add(
        historical_data.MarketDataSync(
            symbol=instrument["symbol"],
            exchange=instrument["exchange"],
            timeframe="1d",
            source="Yahoo Finance",
            status="ok",
            last_synced_at=datetime.now(timezone.utc) - timedelta(minutes=2),
            rows_stored=0,
        )
    )
    db.commit()
    db.close()

    with pytest.raises(historical_data.SyncCooldownError):
        historical_data.sync_daily_history(manual=True)


def test_history_sync_stops_after_provider_rate_limit(monkeypatch):
    _isolated_sessions(monkeypatch)
    instruments = historical_data.WATCHLIST[:2]
    monkeypatch.setattr(historical_data, "WATCHLIST", instruments)
    calls = []

    def rate_limited(instrument):
        calls.append(instrument["symbol"])
        response = requests.Response()
        response.status_code = 429
        raise requests.HTTPError("rate limited", response=response)

    monkeypatch.setattr(historical_data, "_fetch_daily_bars", rate_limited)
    result = historical_data.sync_daily_history()

    assert calls == [instruments[0]["symbol"]]
    assert result["status"] == "error"
    assert len(result["warnings"]) == 2
    assert "rate-limited" in result["warnings"][1]


def test_history_api_validates_symbols_and_sync_endpoint(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.stocks.get_daily_history",
        lambda symbol, limit: {"status": "ok", "symbol": symbol, "count": limit, "candles": []},
    )
    def fake_sync(*, manual: bool) -> dict:
        assert manual
        return {
            "status": "ok",
            "provider": "Yahoo Finance",
            "timeframe": "1d",
            "range": "5y",
            "syncedAt": "2026-11-03T10:00:00+00:00",
            "syncedSymbols": ["RELIANCE"],
            "warnings": [],
        }

    monkeypatch.setattr("app.api.routes.stocks.sync_daily_history", fake_sync)
    client = TestClient(app)

    history = client.get("/api/v1/stocks/history/RELIANCE?limit=12")
    sync = client.post("/api/v1/stocks/history/sync")

    assert history.status_code == 200
    assert history.json()["count"] == 12
    assert sync.status_code == 200
    assert sync.json()["syncedSymbols"] == ["RELIANCE"]


def test_history_api_translates_invalid_symbol(monkeypatch):
    def invalid_symbol(symbol, limit):
        assert limit > 0
        raise ValueError(f"Unsupported market symbol: {symbol}.")

    monkeypatch.setattr("app.api.routes.stocks.get_daily_history", invalid_symbol)
    response = TestClient(app).get("/api/v1/stocks/history/NOT-A-WATCHLIST-SYMBOL")

    assert response.status_code == 404
    assert "Unsupported market symbol" in response.json()["detail"]
