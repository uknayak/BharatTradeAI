from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import requests
from sqlalchemy.exc import OperationalError

from app.db.database import Base, SessionLocal, engine
from app.services.market.models import Candle, MarketQuote
from config.settings import settings

_MEMORY_STORE: Dict[str, List[Dict[str, Any]]] = {"quotes": [], "candles": []}


class NSEMarketDataService:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or settings.nse_base_url

    def fetch_index_snapshot(self, symbol: str = "NIFTY 50") -> Dict[str, Any]:
        try:
            response = requests.get(self.base_url, timeout=10)
            status = "online" if response.status_code == 200 else "offline"
            return {
                "source": "NSE",
                "symbol": symbol,
                "status": status,
                "response_code": response.status_code,
                "message": response.text[:200] if response.text else "",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "source": "NSE",
                "symbol": symbol,
                "status": "offline",
                "response_code": None,
                "message": str(exc),
            }

    def fetch_stock_quote(self, symbol: str) -> Dict[str, Any]:
        return {
            "source": "NSE",
            "symbol": symbol,
            "status": "not_connected",
            "message": "Live stock quote fetching requires the market data source and broker credentials to be configured.",
        }

    def fetch_candles(self, symbol: str, interval: str = "5m", limit: int = 50) -> List[Dict[str, Any]]:
        candle_time = datetime.now(timezone.utc)
        return [
            {
                "symbol": symbol,
                "timeframe": interval,
                "open": 100.0,
                "high": 103.0,
                "low": 98.5,
                "close": 101.5,
                "volume": 120000,
                "candle_time": (candle_time - timedelta(minutes=i * 5)).isoformat(),
            }
            for i in range(limit)
        ]

    def save_quote(
        self,
        symbol: str,
        exchange: str,
        ltp: float | None,
        close: float | None,
        volume: int | None,
        percent_change: float | None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "exchange": exchange,
            "instrument_type": "index" if symbol.upper() in {"NIFTY", "BANKNIFTY"} else "stock",
            "ltp": ltp,
            "close": close,
            "volume": volume,
            "percent_change": percent_change,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata_json": metadata or {},
        }

        try:
            Base.metadata.create_all(bind=engine)
        except Exception:
            pass

        db = SessionLocal()
        try:
            quote = MarketQuote(
                symbol=symbol,
                exchange=exchange,
                instrument_type=payload["instrument_type"],
                ltp=ltp,
                close=close,
                volume=volume,
                percent_change=percent_change,
                timestamp=datetime.now(timezone.utc),
                metadata_json=metadata or {},
            )
            db.add(quote)
            db.commit()
            db.refresh(quote)
            return {"status": "saved", "storage": "postgresql", "id": quote.id, "symbol": symbol}
        except OperationalError:
            try:
                Base.metadata.create_all(bind=engine)
                db.rollback()
                quote = MarketQuote(
                    symbol=symbol,
                    exchange=exchange,
                    instrument_type=payload["instrument_type"],
                    ltp=ltp,
                    close=close,
                    volume=volume,
                    percent_change=percent_change,
                    timestamp=datetime.now(timezone.utc),
                    metadata_json=metadata or {},
                )
                db.add(quote)
                db.commit()
                db.refresh(quote)
                return {"status": "saved", "storage": "postgresql", "id": quote.id, "symbol": symbol}
            except Exception:
                _MEMORY_STORE["quotes"].append(payload)
                return {"status": "queued", "storage": "memory", "symbol": symbol, "detail": "Database unavailable; market data queued in memory."}
        finally:
            db.close()

    def save_candle(
        self,
        symbol: str,
        exchange: str,
        candle_time: datetime,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        timeframe: str = "5m",
    ) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "candle_time": candle_time.isoformat(),
        }

        try:
            Base.metadata.create_all(bind=engine)
        except Exception:
            pass

        db = SessionLocal()
        try:
            candle = Candle(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                candle_time=candle_time,
            )
            db.add(candle)
            db.commit()
            db.refresh(candle)
            return {"status": "saved", "storage": "postgresql", "id": candle.id, "symbol": symbol}
        except OperationalError:
            try:
                Base.metadata.create_all(bind=engine)
                db.rollback()
                candle = Candle(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    candle_time=candle_time,
                )
                db.add(candle)
                db.commit()
                db.refresh(candle)
                return {"status": "saved", "storage": "postgresql", "id": candle.id, "symbol": symbol}
            except Exception:
                _MEMORY_STORE["candles"].append(payload)
                return {"status": "queued", "storage": "memory", "symbol": symbol, "detail": "Database unavailable; candle queued in memory."}
        finally:
            db.close()


def fetch_nifty_snapshot() -> Dict[str, Any]:
    service = NSEMarketDataService()
    return service.fetch_index_snapshot("NIFTY 50")


def fetch_banknifty_snapshot() -> Dict[str, Any]:
    service = NSEMarketDataService()
    return service.fetch_index_snapshot("BANKNIFTY")
