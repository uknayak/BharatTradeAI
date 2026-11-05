from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from app.db.database import Base


class MarketInstrument(Base):
    __tablename__ = "market_instruments"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), unique=True, nullable=False, index=True)
    exchange = Column(String(20), default="NSE")
    instrument_type = Column(String(50), default="stock")
    sector = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)
    probability = Column(Float, nullable=True)
    confidence = Column(String(20), default="medium")
    expected_move = Column(Float, nullable=True)
    source = Column(String(50), default="system")
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
