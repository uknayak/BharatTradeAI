from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.main import app
from app.services.ai_picks import sentiment_analysis
from app.services.ai_picks.news_models import AIStockArticleSentiment, AIStockNews
from app.services.sentiment.base import SentimentModelUnavailable, SentimentProbabilities
from app.services.sentiment.registry import DEFAULT_MODEL_ID


def _isolated_sessions(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(sentiment_analysis, "SessionLocal", sessions)
    return sessions


def _news(
    *,
    symbol: str,
    article_id: str,
    title: str,
    description: str,
    published_at: datetime,
    url: str | None = None,
) -> AIStockNews:
    return AIStockNews(
        symbol=symbol,
        article_id=article_id,
        title=title,
        description=description,
        url=url,
        source="Example Financial News",
        provider="GDELT",
        published_at=published_at,
        retrieved_at=datetime.now(timezone.utc),
        sentiment_label="NEUTRAL",
        sentiment_score=0,
        event_category="GENERAL",
    )


class _FakeModel:
    model_id = "test-financial-sentiment"

    def __init__(self):
        self.calls: list[list[str]] = []

    def predict(self, texts: list[str]) -> list[SentimentProbabilities]:
        self.calls.append(texts)
        return [
            SentimentProbabilities(positive=0.8, negative=0.1, neutral=0.1)
            if text.startswith("TCS secures")
            else SentimentProbabilities(positive=0.1, negative=0.8, neutral=0.1)
            for text in texts
        ]


def test_finbert_article_scores_are_cached_and_aggregated_by_freshness_and_relevance(monkeypatch):
    sessions = _isolated_sessions(monkeypatch)
    cutoff = datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
    db = sessions()
    db.add_all(
        [
            _news(
                symbol="TCS",
                article_id="newest",
                title="TCS secures a major order",
                description="Order announcement",
                published_at=cutoff - timedelta(hours=1),
                url="https://news.example/tcs-order",
            ),
            _news(
                symbol="TCS",
                article_id="same-url",
                title="Republished order headline",
                description="Syndicated duplicate",
                published_at=cutoff - timedelta(hours=2),
                url="https://news.example/tcs-order",
            ),
            _news(
                symbol="TCS",
                article_id="older",
                title="Quarterly results remain mixed",
                description="TCS reported its results",
                published_at=cutoff - timedelta(hours=48),
                url="https://news.example/tcs-results",
            ),
            _news(
                symbol="TCS",
                article_id="outdated",
                title="TCS old news",
                description="Old",
                published_at=cutoff - timedelta(days=8),
            ),
            _news(
                symbol="TCS",
                article_id="future",
                title="TCS future news",
                description="Future",
                published_at=cutoff,
            ),
        ]
    )
    db.commit()
    db.close()
    model = _FakeModel()

    result = sentiment_analysis.analyze_ai_pick_sentiment(
        cutoff=cutoff,
        model_id=model.model_id,
        model=model,
    )
    repeated = sentiment_analysis.analyze_ai_pick_sentiment(
        cutoff=cutoff,
        model_id=model.model_id,
        model=model,
    )
    tcs = next(company for company in result["companies"] if company["symbol"] == "TCS")
    newest_weight = sentiment_analysis._freshness_weight(cutoff - timedelta(hours=1), cutoff)
    older_weight = sentiment_analysis._freshness_weight(cutoff - timedelta(hours=48), cutoff) * 0.85
    expected_positive = (0.8 * newest_weight + 0.1 * older_weight) / (newest_weight + older_weight)

    assert result["status"] == "ready"
    assert result["model"] == model.model_id
    assert tcs["articleCount"] == 2
    assert tcs["analyzedArticleCount"] == 2
    assert tcs["sentiment"] == "POSITIVE"
    assert tcs["probabilities"]["positive"] == pytest.approx(expected_positive, abs=1e-5)
    assert tcs["sentimentScore"] == pytest.approx(
        tcs["probabilities"]["positive"] - tcs["probabilities"]["negative"]
    )
    assert [article["id"] for article in tcs["articles"]] == ["newest", "older"]
    assert tcs["articles"][1]["relevanceScore"] == 0.85
    assert model.calls and len(model.calls) == 1
    assert repeated["articleCount"] == result["articleCount"]
    db = sessions()
    try:
        stored = db.query(AIStockArticleSentiment).filter_by(model_id=model.model_id).count()
    finally:
        db.close()
    assert stored == 2


def test_finbert_analysis_returns_no_data_without_loading_model(monkeypatch):
    _isolated_sessions(monkeypatch)

    class ModelThatMustNotRun:
        model_id = DEFAULT_MODEL_ID

        def predict(self, texts):
            raise AssertionError("No inference should run without eligible news.")

    result = sentiment_analysis.analyze_ai_pick_sentiment(model=ModelThatMustNotRun())

    assert result["status"] == "no_data"
    assert result["articleCount"] == 0


def test_finbert_analysis_rejects_bad_model_probability_output(monkeypatch):
    sessions = _isolated_sessions(monkeypatch)
    now = datetime.now(timezone.utc)
    db = sessions()
    db.add(
        _news(
            symbol="TCS",
            article_id="bad-output",
            title="TCS reports quarterly earnings",
            description="Results",
            published_at=now - timedelta(hours=1),
        )
    )
    db.commit()
    db.close()

    class InvalidModel:
        model_id = "invalid"

        def predict(self, texts):
            return [SentimentProbabilities(positive=float("nan"), negative=0, neutral=0)]

    with pytest.raises(SentimentModelUnavailable, match="invalid probabilities"):
        sentiment_analysis.analyze_ai_pick_sentiment(
            cutoff=now,
            model_id="invalid",
            model=InvalidModel(),
        )


def test_finbert_sentiment_api_supports_symbol_and_explicit_cutoff(monkeypatch):
    cutoff = "2026-11-03T10:00:00+00:00"
    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.get_ai_pick_sentiment",
        lambda **kwargs: {
            "status": "ready",
            "model": DEFAULT_MODEL_ID,
            "cutoff": cutoff,
            "companies": [{"symbol": "TCS", "sentiment": "POSITIVE"}],
        },
    )

    response = TestClient(app).get(
        "/api/v1/ai-stock-picks/sentiment",
        params={"symbol": "tcs", "cutoff": cutoff},
    )

    assert response.status_code == 200
    assert response.json()["companies"] == [{"symbol": "TCS", "sentiment": "POSITIVE"}]


def test_finbert_analysis_api_returns_service_unavailable_for_model_load_failure(monkeypatch):
    def unavailable(**kwargs):
        raise SentimentModelUnavailable("Could not load ProsusAI/finbert.")

    monkeypatch.setattr("app.api.routes.ai_stock_picks.analyze_ai_pick_sentiment", unavailable)

    response = TestClient(app).post("/api/v1/ai-stock-picks/sentiment/analyze")

    assert response.status_code == 503
    assert "Could not load ProsusAI/finbert" in response.json()["detail"]
