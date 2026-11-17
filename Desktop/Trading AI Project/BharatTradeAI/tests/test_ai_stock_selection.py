from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ai_picks import ollama_explainer, selection


NOW = datetime(2026, 11, 4, 5, 0, tzinfo=timezone.utc)
INDICATORS = {
    "rsi_14": 61.0,
    "volume_ratio": 1.2,
    "macd_pct": 0.015,
    "macd_signal_pct": 0.01,
    "ema_20_gap_pct": 2.0,
    "ema_50_gap_pct": 1.0,
    "sma_200_gap_pct": 4.0,
    "bollinger_position": 0.6,
    "bollinger_width_pct": 3.0,
    "atr_14_pct": 1.0,
    "volatility_20_daily_pct": 1.0,
}
PATTERN = {
    "latest_daily_return_pct": 0.4,
    "momentum_5d_pct": 1.2,
    "momentum_20d_pct": 3.0,
}


def _prediction(symbol: str, probability: float, *, market_time: datetime | None = None) -> dict:
    return {
        "symbol": symbol,
        "exchange": "NSE",
        "status": "ready",
        "validation_status": "out_of_sample_evaluated",
        "calibration_status": "sigmoid_calibrated_on_chronological_holdout",
        "upward_probability": probability,
        "direction": "UP",
        "last_market_time": (market_time or NOW - timedelta(hours=1)).isoformat(),
        "model_version": f"{symbol}-version",
        "history_candle_count": 1236,
        "technical_indicators": dict(INDICATORS),
        "historical_pattern": dict(PATTERN),
        "metrics": {"selected": {"sample_count": 150, "brier_score": 0.2}},
    }


def _quote(symbol: str, *, timestamp: datetime | None = None) -> dict:
    return {
        "symbol": symbol,
        "type": "EQUITY",
        "lastPrice": 123.45,
        "changePct": 1.25,
        "timestamp": (timestamp or NOW - timedelta(minutes=2)).isoformat(),
    }


def _sentiment(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "sentiment": "POSITIVE",
        "sentimentScore": 0.4,
        "articleCount": 2,
        "analyzedArticleCount": 2,
        "probabilities": {"positive": 0.6, "negative": 0.2, "neutral": 0.2},
    }


def test_opportunity_selection_requires_fresh_validated_probability_over_80(monkeypatch):
    predictions = [
        _prediction("RELIANCE", 0.9),
        _prediction("HDFCBANK", 0.81),
        _prediction("TCS", 0.8),
        {
            **_prediction("INFY", 0.95),
            "validation_status": "not_validated",
        },
        _prediction("SBIN", 0.99, market_time=NOW - timedelta(days=5)),
    ]
    monkeypatch.setattr(selection, "fetch_live_quotes", lambda: {
        "status": "ok",
        "quotes": [_quote(row["symbol"]) for row in predictions],
    })
    monkeypatch.setattr(selection, "get_stock_direction_predictions", lambda: {"predictions": predictions})
    monkeypatch.setattr(selection, "get_ai_pick_sentiment", lambda *, cutoff: {
        "companies": [_sentiment(row["symbol"]) for row in predictions],
    })
    monkeypatch.setattr(
        selection,
        "explain_stock_analyses",
        lambda candidates: [
            {
                "status": "ready",
                "provider": "Ollama",
                "model": "gemma3",
                "summary": "Grounded summary.",
            }
            for _ in candidates
        ],
    )

    result = selection.build_ai_stock_opportunities(now=NOW)

    assert result["status"] == "ready"
    assert [row["symbol"] for row in result["opportunities"]] == ["RELIANCE", "HDFCBANK"]
    assert result["opportunities"][0]["current_price"] == 123.45
    assert result["opportunities"][0]["expected_price_range"] is None
    assert result["opportunities"][0]["news_sentiment"]["label"] == "POSITIVE"
    assert result["opportunities"][0]["technical_analysis"]["history_candle_count"] == 1236
    assert result["opportunities"][0]["next_trading_day"] == "2026-11-05"
    assert {row["reason"] for row in result["exclusion_reasons"]} == {
        "below_strict_80_percent_threshold",
        "model_not_validated",
        "daily_history_stale_or_invalid",
    }


@pytest.mark.parametrize(
    ("threshold_percent", "expected_symbols"),
    [
        (65, ["A", "B", "C", "D", "E"]),
        (70, ["A", "B", "C"]),
        (80, ["A"]),
    ],
)
def test_opportunity_selection_applies_strict_threshold_and_probability_order(
    monkeypatch, threshold_percent, expected_symbols
):
    predictions = [
        _prediction("A", 0.81),
        _prediction("B", 0.80),
        _prediction("C", 0.71),
        _prediction("D", 0.70),
        _prediction("E", 0.66),
        _prediction("F", 0.65),
    ]
    monkeypatch.setattr(selection, "fetch_live_quotes", lambda: {
        "status": "ok",
        "quotes": [_quote(row["symbol"]) for row in predictions],
    })
    monkeypatch.setattr(selection, "get_stock_direction_predictions", lambda: {
        "predictions": predictions,
    })
    monkeypatch.setattr(selection, "get_ai_pick_sentiment", lambda *, cutoff: {"companies": []})

    result = selection.build_ai_stock_opportunities(
        now=NOW,
        include_explanations=False,
        threshold_percent=threshold_percent,
    )

    assert [row["symbol"] for row in result["opportunities"]] == expected_symbols
    assert [row["upward_probability"] for row in result["opportunities"]] == sorted(
        (row["upward_probability"] for row in result["opportunities"]),
        reverse=True,
    )
    assert result["threshold"] == threshold_percent / 100
    assert result["threshold_comparison"] == "strictly greater than"


def test_opportunity_selection_returns_required_empty_message_without_llm_call(monkeypatch):
    monkeypatch.setattr(selection, "fetch_live_quotes", lambda: {"status": "ok", "quotes": []})
    monkeypatch.setattr(selection, "get_stock_direction_predictions", lambda: {
        "predictions": [_prediction("RELIANCE", 0.8)],
    })
    monkeypatch.setattr(selection, "get_ai_pick_sentiment", lambda *, cutoff: {"companies": []})
    monkeypatch.setattr(
        selection,
        "explain_stock_analyses",
        lambda candidates: [] if not candidates else pytest.fail("No stocks should qualify."),
    )

    result = selection.build_ai_stock_opportunities(now=NOW)

    assert result["status"] == "no_opportunities"
    assert result["opportunities"] == []
    assert result["empty_message"] == selection.NO_OPPORTUNITIES_MESSAGE


def test_opportunity_screen_can_return_without_waiting_for_gemma(monkeypatch):
    prediction = _prediction("RELIANCE", 0.91)
    monkeypatch.setattr(selection, "fetch_live_quotes", lambda: {
        "status": "ok",
        "provider": "Yahoo Finance",
        "quotes": [_quote("RELIANCE")],
    })
    monkeypatch.setattr(selection, "get_stock_direction_predictions", lambda: {
        "predictions": [prediction],
    })
    monkeypatch.setattr(selection, "get_ai_pick_sentiment", lambda *, cutoff: {
        "companies": [_sentiment("RELIANCE")],
    })
    monkeypatch.setattr(
        selection,
        "explain_stock_analyses",
        lambda *_: pytest.fail("The non-blocking screen must not call Ollama."),
    )

    result = selection.build_ai_stock_opportunities(now=NOW, include_explanations=False)

    assert result["status"] == "ready"
    assert result["opportunities"][0]["gemma_explanation"]["status"] == "pending"
    assert result["opportunities"][0]["quote_source"] == "Yahoo Finance"


def test_opportunity_explanations_are_queued_and_cached(monkeypatch):
    selection._EXPLANATION_CACHE.clear()
    selection._EXPLANATION_PENDING.clear()
    prediction = _prediction("RELIANCE", 0.91)
    monkeypatch.setattr(selection, "fetch_live_quotes", lambda: {
        "status": "ok",
        "provider": "Yahoo Finance",
        "quotes": [_quote("RELIANCE")],
    })
    monkeypatch.setattr(selection, "get_stock_direction_predictions", lambda: {
        "predictions": [prediction],
    })
    monkeypatch.setattr(selection, "get_ai_pick_sentiment", lambda *, cutoff: {
        "companies": [_sentiment("RELIANCE")],
    })
    screened = selection.build_ai_stock_opportunities(now=NOW, include_explanations=False)
    opportunities = screened["opportunities"]
    jobs = selection.queue_opportunity_explanations(opportunities)

    assert len(jobs) == 1
    assert jobs[0][1]["symbol"] == "RELIANCE"
    assert jobs[0][1]["technical_indicators"] == INDICATORS
    monkeypatch.setattr(selection, "explain_stock_analyses", lambda candidates: [
        {
            "status": "ready",
            "provider": "Ollama",
            "model": "gemma3:latest",
            "summary": "Grounded output.",
        }
        for _ in candidates
    ])
    selection.run_opportunity_explanations(jobs)

    assert selection.queue_opportunity_explanations(opportunities) == []
    assert opportunities[0]["gemma_explanation"]["status"] == "ready"
    assert not selection._EXPLANATION_PENDING


def test_opportunity_selection_excludes_stale_intraday_quotes(monkeypatch):
    monkeypatch.setattr(selection, "fetch_live_quotes", lambda: {
        "status": "ok",
        "quotes": [_quote("RELIANCE", timestamp=NOW - timedelta(minutes=16))],
    })
    monkeypatch.setattr(selection, "get_stock_direction_predictions", lambda: {
        "predictions": [_prediction("RELIANCE", 0.91)],
    })
    monkeypatch.setattr(selection, "get_ai_pick_sentiment", lambda *, cutoff: {"companies": []})

    result = selection.build_ai_stock_opportunities(now=NOW)

    assert result["status"] == "no_opportunities"
    assert result["exclusion_reasons"] == [
        {"symbol": "RELIANCE", "reason": "live_quote_stale_or_incomplete"}
    ]


def test_gemma_explanation_only_uses_known_signal_labels_and_server_facts(monkeypatch):
    signals = [
        {"id": "trend", "label": "Price is above the 20-day moving average.", "polarity": "bullish"},
        {"id": "risk_level", "label": "Recent volatility is high.", "polarity": "risk"},
    ]
    payload = {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "direction": "UP",
        "upward_probability": 0.86,
        "current_price": 123.45,
        "daily_change_pct": 1.2,
        "technical_indicators": INDICATORS,
        "historical_pattern": PATTERN,
        "news_sentiment": {"label": "POSITIVE", "score": 0.5},
        "risk_level": "HIGH",
    }
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"analyses":[{"symbol":"RELIANCE",'
                    '"bullish_signal_ids":["trend"],'
                    '"bearish_signal_ids":[],"risk_signal_ids":["risk_level"]}]}'
                )
            }

    def fake_post(url, *, json, timeout):
        captured.update(url=url, request=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(ollama_explainer.requests, "post", fake_post)
    monkeypatch.setattr(ollama_explainer.settings, "llm_provider", "ollama")
    monkeypatch.setattr(ollama_explainer.settings, "ollama_base_url", "http://localhost:11434")
    monkeypatch.setattr(ollama_explainer.settings, "model_name", "gemma3")

    result = ollama_explainer.explain_stock_analysis(payload, signals)

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["request"]["model"] == "gemma3"
    assert captured["request"]["format"] == "json"
    assert captured["request"]["keep_alive"] == "5m"
    assert captured["request"]["options"]["num_ctx"] == 2048
    assert "technical_indicator_summary" in captured["request"]["prompt"]
    assert result["status"] == "ready"
    assert result["bullish_signals"] == [signals[0]["label"]]
    assert result["risks"] == [signals[1]["label"]]
    assert "86.0%" in result["summary"]
    assert "123.45" not in result["analysis"]


def test_gemma_explanation_rejects_unknown_signals_and_connection_errors(monkeypatch):
    payload = {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "direction": "UP",
        "upward_probability": 0.86,
        "current_price": 123.45,
        "daily_change_pct": 1.2,
        "technical_indicators": INDICATORS,
        "historical_pattern": PATTERN,
        "news_sentiment": {"label": "POSITIVE", "score": 0.5},
        "risk_level": "LOW",
    }
    signal = {"id": "trend", "label": "Trend is positive.", "polarity": "bullish"}

    class InvalidResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "response": (
                    '{"analyses":[{"symbol":"RELIANCE",'
                    '"bullish_signal_ids":["invented"],'
                    '"bearish_signal_ids":[],"risk_signal_ids":[]}]}'
                )
            }

    monkeypatch.setattr(ollama_explainer.requests, "post", lambda *args, **kwargs: InvalidResponse())
    monkeypatch.setattr(ollama_explainer.settings, "llm_provider", "ollama")
    with pytest.raises(ollama_explainer.OllamaExplanationUnavailable):
        ollama_explainer.explain_stock_analysis(payload, [signal])

    def unavailable(*args, **kwargs):
        raise ollama_explainer.requests.ConnectionError("Ollama is offline")

    monkeypatch.setattr(ollama_explainer.requests, "post", unavailable)
    with pytest.raises(ollama_explainer.OllamaExplanationUnavailable):
        ollama_explainer.explain_stock_analysis(payload, [signal])


def test_opportunities_api_exposes_selection(monkeypatch):
    expected = {
        "status": "no_opportunities",
        "threshold": 0.8,
        "opportunities": [],
        "empty_message": selection.NO_OPPORTUNITIES_MESSAGE,
    }
    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.build_ai_stock_opportunities",
        lambda **kwargs: expected,
    )

    response = TestClient(app).get("/api/v1/ai-stock-picks/opportunities")

    assert response.status_code == 200
    assert response.json()["empty_message"] == selection.NO_OPPORTUNITIES_MESSAGE
    assert response.json()["threshold"] == 0.8


def test_ai_stock_quotes_api_uses_expanded_universe(monkeypatch):
    from app.api.routes import ai_stock_picks

    expected = {
        "status": "ok",
        "provider": "Yahoo Finance",
        "quotes": [{"symbol": "ADANIENT"}, {"symbol": "RELIANCE"}],
    }
    monkeypatch.setattr(ai_stock_picks, "fetch_ai_stock_quotes", lambda: expected)

    response = TestClient(app).get("/api/v1/ai-stock-picks/quotes")

    assert response.status_code == 200
    assert response.json() == expected


def test_opportunities_api_accepts_only_supported_thresholds(monkeypatch):
    from app.api.routes import ai_stock_picks

    calls = []
    monkeypatch.setattr(
        ai_stock_picks,
        "build_ai_stock_opportunities",
        lambda **kwargs: calls.append(kwargs) or {
            "status": "no_opportunities",
            "opportunities": [],
            "empty_message": "No qualifying opportunities.",
        },
    )
    client = TestClient(app)

    response = client.get("/api/v1/ai-stock-picks/opportunities?threshold_percent=70")
    invalid_response = client.get("/api/v1/ai-stock-picks/opportunities?threshold_percent=75")

    assert response.status_code == 200
    assert calls == [{"include_explanations": False, "threshold_percent": 70}]
    assert invalid_response.status_code == 422


def test_opportunities_api_queues_gemma_after_screening(monkeypatch):
    from app.api.routes import ai_stock_picks

    calls = []
    opportunity = {"symbol": "RELIANCE"}
    expected = {"status": "ready", "opportunities": [opportunity]}
    jobs = [("job",)]
    monkeypatch.setattr(
        ai_stock_picks,
        "build_ai_stock_opportunities",
        lambda **kwargs: calls.append(("build", kwargs)) or expected,
    )
    monkeypatch.setattr(
        ai_stock_picks,
        "queue_opportunity_explanations",
        lambda opportunities: calls.append(("queue", opportunities)) or jobs,
    )
    monkeypatch.setattr(
        ai_stock_picks,
        "run_opportunity_explanations",
        lambda queued: calls.append(("run", queued)),
    )

    response = TestClient(app).get("/api/v1/ai-stock-picks/opportunities")

    assert response.status_code == 200
    assert calls == [
        ("build", {"include_explanations": False, "threshold_percent": 80}),
        ("queue", [opportunity]),
        ("run", jobs),
    ]
