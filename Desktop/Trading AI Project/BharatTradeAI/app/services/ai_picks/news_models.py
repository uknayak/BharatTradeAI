from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, UniqueConstraint

from app.db.database import Base


class AIStockNews(Base):
    __tablename__ = "ai_stock_news"
    __table_args__ = (
        UniqueConstraint("symbol", "article_id", name="uq_ai_stock_news_symbol_article"),
    )

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    article_id = Column(String(64), nullable=False)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    url = Column(Text, nullable=True)
    source = Column(String(200), nullable=False)
    provider = Column(String(50), nullable=False)
    published_at = Column(DateTime(timezone=True), nullable=False, index=True)
    retrieved_at = Column(DateTime(timezone=True), nullable=False)
    sentiment_label = Column(String(20), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    event_category = Column(String(40), nullable=False)


class AIPickNewsSync(Base):
    __tablename__ = "ai_pick_news_sync"

    id = Column(Integer, primary_key=True)
    last_sync_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(20), nullable=False)
    provider = Column(String(50), nullable=True)
    warning = Column(Text, nullable=True)


class AIStockArticleSentiment(Base):
    __tablename__ = "ai_stock_article_sentiment"
    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "article_id",
            "model_id",
            name="uq_ai_stock_article_sentiment_model",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    article_id = Column(String(64), nullable=False, index=True)
    model_id = Column(String(100), nullable=False)
    label = Column(String(20), nullable=False)
    probabilities = Column(JSON, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    relevance_score = Column(Float, nullable=False)
    analyzed_at = Column(DateTime(timezone=True), nullable=False)
