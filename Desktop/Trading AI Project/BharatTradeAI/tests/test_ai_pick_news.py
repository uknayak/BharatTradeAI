from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.main import app
from app.services.ai_picks import news
from app.services.ai_picks.news_models import AIStockNews


def _isolated_sessions(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(news, "SessionLocal", sessions)
    return sessions


def test_ai_news_sync_maps_deduplicates_and_scores_per_company(monkeypatch):
    sessions = _isolated_sessions(monkeypatch)
    instrument = next(row for row in news.WATCHLIST if row["symbol"] == "RELIANCE")
    monkeypatch.setattr(news, "WATCHLIST", (instrument,))
    monkeypatch.setattr(
        news,
        "fetch_market_news",
        lambda **kwargs: {
            "source": "GDELT" if kwargs["newsdata_api_key"] == "" and kwargs["limit"] == 10 else "Unexpected",
            "articles": [
                {
                    "title": "Reliance wins a major order after quarterly results beat estimates",
                    "description": "Strong growth",
                    "url": "https://news.example/reliance-order",
                    "source": "Example Finance",
                    "published_at": "2026-11-03T08:00:00+00:00",
                    "sentiment": {"label": "POSITIVE", "score": 1.0},
                    "symbols": (),
                },
                {
                    "title": "Reliance wins a major order after quarterly results beat estimates",
                    "description": "Duplicate syndicated article",
                    "url": "https://another-news.example/republished-order",
                    "source": "Another publisher",
                    "published_at": "2026-11-03T08:00:00+00:00",
                    "sentiment": {"label": "POSITIVE", "score": 1.0},
                    "symbols": (),
                },
                {
                    "title": "Reliance faces regulatory penalty",
                    "description": "Negative report",
                    "url": "https://news.example/reliance-penalty",
                    "source": "Example Finance",
                    "published_at": "2026-11-03T09:00:00+00:00",
                    "sentiment": {"label": "NEGATIVE", "score": -1.0},
                    "symbols": (),
                },
                {
                    "title": "Reliance future outlook",
                    "url": "https://news.example/no-date",
                    "source": "Example Finance",
                    "published_at": None,
                    "sentiment": {"label": "POSITIVE", "score": 1.0},
                    "symbols": (),
                },
            ],
        },
    )

    sync_result = news.sync_ai_pick_news(manual=False)
    news.sync_ai_pick_news(manual=False)
    response = news.get_ai_pick_news(
        cutoff=datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
    )
    company = response["companies"][0]
    db = sessions()
    try:
        stored = db.query(AIStockNews).filter_by(symbol="RELIANCE").count()
    finally:
        db.close()

    assert sync_result["status"] == "ok"
    assert sync_result["provider"] == "GDELT"
    assert stored == 2
    assert company["symbol"] == "RELIANCE"
    assert company["articleCount"] == 2
    assert company["sentiment"] == "NEUTRAL"
    assert company["sentimentScore"] == 0
    assert company["eventCounts"] == {"QUARTERLY_RESULTS": 1, "REGULATORY": 1}
    assert company["articles"][0]["source"] == "Example Finance"
    assert company["articles"][0]["publishedAt"] < response["cutoff"]


def test_ai_news_event_classifier_covers_requested_event_categories():
    cases = {
        "quarterly results": "QUARTERLY_RESULTS",
        "company secures order": "MAJOR_ORDER",
        "merger announced": "MERGER_ACQUISITION",
        "regulatory penalty": "REGULATORY",
        "government policy change": "GOVERNMENT_POLICY",
        "sector outlook": "SECTOR_DEVELOPMENT",
    }

    for headline, expected in cases.items():
        assert news._event_category(headline, "") == expected


def test_ai_news_cutoff_excludes_at_and_after_cutoff(monkeypatch):
    sessions = _isolated_sessions(monkeypatch)
    cutoff = datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)
    db = sessions()
    for symbol, article_id, published_at in (
        ("TCS", "before", cutoff - timedelta(seconds=1)),
        ("TCS", "at", cutoff),
        ("TCS", "after", cutoff + timedelta(seconds=1)),
    ):
        db.add(
            AIStockNews(
                symbol=symbol,
                article_id=article_id,
                title=article_id,
                source="Example",
                provider="GDELT",
                published_at=published_at,
                retrieved_at=cutoff,
                sentiment_label="POSITIVE",
                sentiment_score=1,
                event_category="GENERAL",
            )
        )
    db.commit()
    db.close()

    result = news.get_ai_pick_news(cutoff=cutoff)
    tcs = next(company for company in result["companies"] if company["symbol"] == "TCS")

    assert [article["id"] for article in tcs["articles"]] == ["before"]
    assert tcs["articleCount"] == 1


def test_ai_news_sync_enforces_cooldown(monkeypatch):
    sessions = _isolated_sessions(monkeypatch)
    db = sessions()
    db.add(
        news.AIPickNewsSync(
            id=1,
            last_sync_at=datetime.now(timezone.utc) - timedelta(minutes=2),
            status="ok",
            provider="GDELT",
        )
    )
    db.commit()
    db.close()

    with pytest.raises(news.NewsSyncCooldownError):
        news.sync_ai_pick_news(manual=True)


def test_ai_pick_news_api_requires_timezone_for_cutoff():
    response = TestClient(app).get(
        "/api/v1/ai-stock-picks/news",
        params={"cutoff": "2026-11-03T10:00:00"},
    )

    assert response.status_code == 422
    assert "ISO 8601" in response.json()["detail"]
