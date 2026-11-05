from datetime import datetime, timedelta, timezone

from app.services.market.nse_market import NSEMarketDataService


def test_save_quote_persists_market_record():
    service = NSEMarketDataService()
    result = service.save_quote(
        symbol="NIFTY",
        exchange="NSE",
        ltp=24500.0,
        close=24480.0,
        volume=1200000,
        percent_change=0.82,
        metadata={"source": "nse"},
    )

    assert result["status"] == "saved"
    assert result["symbol"] == "NIFTY"


def test_save_candle_persists_candle_record():
    service = NSEMarketDataService()
    candle_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    result = service.save_candle(
        symbol="RELIANCE",
        exchange="NSE",
        candle_time=candle_time,
        open_=2500.0,
        high=2525.0,
        low=2495.0,
        close=2515.0,
        volume=50000,
        timeframe="5m",
    )

    assert result["status"] == "saved"
    assert result["symbol"] == "RELIANCE"
