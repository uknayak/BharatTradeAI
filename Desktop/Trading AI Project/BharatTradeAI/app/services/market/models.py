from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.sql import func

from app.db.database import Base


class MarketQuote(Base):
    __tablename__ = "market_quotes"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(20), nullable=False, index=True)
    instrument_type = Column(String(20), nullable=False, default="stock")
    ltp = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    percent_change = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Candle(Base):
    __tablename__ = "market_candles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False, default="5m")
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    volume = Column(Integer, nullable=True)
    candle_time = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MarketDataSync(Base):
    __tablename__ = "market_data_syncs"
    __table_args__ = (
        UniqueConstraint("symbol", "exchange", "timeframe", name="uq_market_data_sync_symbol"),
    )

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    exchange = Column(String(20), nullable=False)
    timeframe = Column(String(20), nullable=False)
    source = Column(String(40), nullable=False)
    status = Column(String(20), nullable=False)
    last_synced_at = Column(DateTime(timezone=True), nullable=False)
    latest_market_time = Column(DateTime(timezone=True), nullable=True)
    rows_stored = Column(Integer, nullable=False, default=0)
    error = Column(String(500), nullable=True)
