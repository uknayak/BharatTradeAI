from app.agents.news_agent import analyze_news
from app.data.news import classify_event, classify_sentiment, collect_news, fetch_market_news, fetch_news_feed


def test_news_collection_deduplicates_maps_company_and_enriches_article():
    records = [
        {"title": "TCS reports record profit growth", "link": "https://example.test/tcs", "summary": "Strong quarterly results"},
        {"title": "TCS reports record profit growth", "link": "https://example.test/tcs", "summary": "Duplicate feed item"},
        {"title": "Reliance faces regulatory penalty", "url": "https://example.test/reliance"},
    ]

    result = collect_news(records)

    assert len(result) == 2
    assert result[0]["companies"] == ["TCS"]
    assert result[0]["sentiment"]["label"] == "POSITIVE"
    assert result[0]["event"] == "EARNINGS"
    assert result[1]["companies"] == ["RELIANCE"]
    assert result[1]["sentiment"]["label"] == "NEGATIVE"
    assert result[1]["event"] == "REGULATORY"


def test_news_helpers_and_agent_boundary():
    assert classify_sentiment("Company outlook is stable")["label"] == "NEUTRAL"
    assert classify_event("Board appoints a new CEO") == "MANAGEMENT"
    assert fetch_news_feed()["status"] == "not_implemented"
    result = analyze_news([{"title": "Infosys wins a strong contract", "url": "x"}])
    assert result["status"] == "ready"
    assert result["articles"][0]["companies"] == ["INFY"]


def test_market_news_adapter_uses_gdelt_tone_and_maps_articles(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "articles": [
                    {
                        "title": "Infosys reports strong profit growth",
                        "url": "https://example.test/infosys",
                        "domain": "example.test",
                        "seendate": "20261016T110000Z",
                        "tone": "3.5",
                    },
                    {
                        "title": "Reliance faces penalty over regulatory breach",
                        "url": "https://example.test/reliance",
                        "domain": "example.test",
                        "seendate": "20261016T100000Z",
                        "tone": "-4.0",
                    },
                ]
            }

    def fake_get(url, params, timeout):
        assert "api.gdeltproject.org" in url
        assert params["query"] == "Infosys India"
        assert params["maxrecords"] == 10
        assert timeout == 8
        return DummyResponse()

    monkeypatch.setattr("app.data.news.requests.get", fake_get)

    result = fetch_market_news("Infosys India", limit=10)

    assert result["source"] == "GDELT"
    assert result["count"] == 2
    assert result["headline_bias"] == "MIXED"
    assert result["sentiment_counts"] == {"positive": 1, "negative": 1, "neutral": 0}
    assert result["articles"][0]["companies"] == ["INFY"]
    assert result["articles"][0]["sentiment"]["label"] == "POSITIVE"
    assert result["articles"][0]["listings"][0]["match_type"] == "headline"
    assert result["articles"][0]["published_at"] == "2026-10-16T11:00:00+00:00"


def test_market_news_uses_newsdata_market_tickers_and_maps_sentiment(monkeypatch):
    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "success",
                "results": [
                    {
                        "title": "TCS reports improved quarterly revenue",
                        "link": "https://example.test/tcs-newsdata",
                        "source_name": "Example Finance",
                        "description": "Business update",
                        "pubDate": "2026-10-18 09:00:00",
                        "sentiment": "positive",
                        "symbol": ["tcs"],
                        "market_id": ["tcs-in-ns"],
                    }
                ],
            }

    def fake_get(url, params, headers, timeout):
        assert url == "https://newsdata.io/api/1/market"
        assert headers == {"X-ACCESS-KEY": "test-api-key"}
        assert params["country"] == "in"
        assert params["language"] == "en"
        assert params["q"] == "TCS"
        assert params["size"] == 10
        assert timeout == 8
        return DummyResponse()

    monkeypatch.setattr("app.data.news.requests.get", fake_get)

    result = fetch_market_news("TCS", limit=25, newsdata_api_key="test-api-key")

    assert result["source"] == "NewsData.io"
    assert result["count"] == 1
    assert result["articles"][0]["source"] == "Example Finance"
    assert result["articles"][0]["companies"] == ["TCS"]
    assert result["articles"][0]["sentiment"]["label"] == "POSITIVE"
    assert result["articles"][0]["listings"] == [
        {
            "symbol": "TCS",
            "name": "Tata Consultancy Services Ltd",
            "exchanges": ["NSE", "BSE"],
            "match_type": "provider",
        }
    ]
    assert result["articles"][0]["symbols"] == ("tcs",)
    assert result["articles"][0]["market_ids"] == ("tcs-in-ns",)
    assert result["articles"][0]["unmatched_symbols"] == []
    assert "12 hours" in result["notice"]


def test_market_news_uses_stale_cache_when_gdelt_times_out(monkeypatch):
    from requests import Timeout

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "articles": [
                    {
                        "title": "TCS reports strong profit growth",
                        "url": "https://example.test/tcs-cache",
                        "domain": "example.test",
                        "tone": 2.0,
                    }
                ]
            }

    calls = 0

    def fake_get(url, params, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return DummyResponse()
        raise Timeout("temporary upstream timeout")

    monkeypatch.setattr("app.data.news.requests.get", fake_get)
    monkeypatch.setattr("app.data.news.NEWS_CACHE_TTL_SECONDS", 0)

    first = fetch_market_news("TCS cache fallback test", limit=1)
    second = fetch_market_news("TCS cache fallback test", limit=1)

    assert first["cache_status"] == "fresh"
    assert second["cache_status"] == "stale"
    assert second["articles"][0]["companies"] == ["TCS"]
    assert calls == 2


def test_news_endpoint_exposes_gdelt_rate_limit(monkeypatch):
    import requests
    from fastapi.testclient import TestClient

    from app.main import app

    upstream_response = requests.Response()
    upstream_response.status_code = 429
    upstream_response.headers["Retry-After"] = "60"

    def fake_get(url, params, timeout):
        raise requests.HTTPError("rate limited", response=upstream_response)

    monkeypatch.setattr("app.data.news.requests.get", fake_get)
    monkeypatch.setattr("app.api.routes.news.settings.news_data_api_key", "")

    response = TestClient(app).get("/api/v1/news", params={"query": "rate-limit-route-test"})

    assert response.status_code == 503
    assert response.headers["retry-after"] == "60"
    assert "rate-limiting" in response.json()["detail"]