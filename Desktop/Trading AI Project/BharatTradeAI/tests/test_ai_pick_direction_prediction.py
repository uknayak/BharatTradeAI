from datetime import datetime, timedelta, timezone
from math import sin

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.db.models import MarketInstrument
torch = pytest.importorskip("torch")
xgboost = pytest.importorskip("xgboost")

from app.ml.ensemble import train_direction_ensemble
from app.ml.features import build_direction_features, build_direction_training_set
from app.ml.lstm import build_ohlcv_sequences, predict_lstm_probabilities, train_lstm_model
from app.main import app
from app.services.ai_picks import prediction


def _candles(count: int = 500) -> list[dict]:
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for index in range(count):
        close = 100 + index * 0.05 + sin(index * 3.141592653589793 / 2) * 1.5
        rows.append(
            {
                "open": close - 0.2,
                "high": close + 1.5,
                "low": close - 1.5,
                "close": close,
                "volume": 100_000 + (index % 17) * 500,
                "candle_time": today - timedelta(days=count - index),
            }
        )
    return rows


def test_technical_features_are_finite_after_warmup_and_do_not_use_future_bars():
    candles = _candles(260)
    features = build_direction_features(candles)
    changed_future = [dict(row) for row in candles]
    changed_future[250]["close"] *= 1.5
    changed_future[250]["high"] *= 1.5
    later_features = build_direction_features(changed_future)

    assert features[198]["sma_200_gap"] != features[198]["sma_200_gap"]
    assert np.isfinite(list(features[220].values())).all()
    assert features[240] == later_features[240]
    assert {
        "rsi_14",
        "macd_pct",
        "ema_20_gap",
        "ema_50_gap",
        "sma_200_gap",
        "bollinger_width",
        "atr_14_pct",
        "news_sentiment",
        "sector_return_1d",
        "market_return_1d",
    }.issubset(features[220])


def test_direction_training_set_uses_context_and_forward_close_labels():
    candles = _candles(260)
    day = candles[220]["candle_time"].date().isoformat()
    dataset = build_direction_training_set(
        candles,
        news_sentiment={day: 0.4},
        sector_returns={day: 0.01},
        market_returns={day: -0.02},
    )
    feature_index = dataset["indices"].index(220)

    assert dataset["features"][feature_index]["news_sentiment"] == pytest.approx(0.4)
    assert dataset["features"][feature_index]["sector_return_1d"] == pytest.approx(0.01)
    assert dataset["features"][feature_index]["market_return_1d"] == pytest.approx(-0.02)
    assert dataset["labels"][feature_index] == int(candles[221]["close"] > candles[220]["close"])


def test_lstm_normalizes_training_sequences_and_returns_probabilities():
    sequences = np.random.default_rng(42).normal(size=(12, 30, 5)).astype(np.float32)
    labels = [0, 1] * 6
    model = train_lstm_model(sequences, labels, epochs=1)
    probabilities = predict_lstm_probabilities(model, sequences[:3])

    assert model["sequence_length"] == 30
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert build_ohlcv_sequences(_candles(40), [29, 39]).shape == (2, 30, 5)


def test_ensemble_returns_insufficient_status_before_training_when_history_is_short():
    candles = _candles(230)
    result = train_direction_ensemble(
        [{"close_return": float(index)} for index in range(199)],
        [index % 2 for index in range(199)],
        candles,
        list(range(199)),
        latest_features={"close_return": 0.01},
    )

    assert result["status"] == "insufficient_data"
    assert result["validation_status"] == "insufficient_data"
    assert result["required_sample_count"] == 200
    assert "latest" not in result


def test_training_service_reports_insufficient_history_without_saving_probability(tmp_path, monkeypatch):
    candles = _candles(100)
    context = {
        "news_sentiment": {},
        "sector_returns": {},
        "market_returns": {},
        "sector": None,
    }
    monkeypatch.setattr(
        prediction,
        "_load_market_data",
        lambda symbol: (prediction._instrument(symbol), candles, context),
    )

    result = prediction.train_stock_direction_model(
        "RELIANCE",
        artifact_directory=tmp_path,
    )

    assert result["status"] == "insufficient_data"
    assert result["validation_status"] == "insufficient_data"
    assert "prediction" not in result
    assert not (tmp_path / "RELIANCE" / "latest.txt").exists()


def test_model_resolves_active_bse_instruments_from_existing_market_metadata(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(prediction, "SessionLocal", sessions)
    db = sessions()
    db.add(
        MarketInstrument(
            symbol="500325",
            exchange="BSE",
            instrument_type="stock",
            sector="Energy",
            is_active=True,
        )
    )
    db.commit()
    db.close()

    instrument = prediction._instrument("500325")

    assert instrument is not None
    assert instrument["exchange"] == "BSE"


def test_current_session_daily_candle_requires_post_sync_confirmation():
    current_day_bar = datetime(2026, 11, 4, 9, 15, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    earlier_day_bar = datetime(2026, 11, 3, 9, 15, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    before_close = datetime(2026, 11, 4, 10, 10, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    after_close = datetime(2026, 11, 4, 17, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    assert not prediction._is_complete_daily_bar(
        current_day_bar,
        datetime(2026, 11, 3, 16, 20, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        now=before_close,
    )
    assert not prediction._is_complete_daily_bar(
        current_day_bar,
        datetime(2026, 11, 4, 15, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        now=after_close,
    )
    assert prediction._is_complete_daily_bar(
        current_day_bar,
        datetime(2026, 11, 4, 16, 16, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        now=after_close,
    )
    assert prediction._is_complete_daily_bar(
        earlier_day_bar,
        None,
        now=before_close,
    )


def test_watchlist_prediction_summary_has_real_model_status_for_each_equity(monkeypatch):
    symbols = [
        row["symbol"]
        for row in prediction.WATCHLIST
        if row.get("type") == "EQUITY"
    ]
    monkeypatch.setattr(
        prediction,
        "get_stock_direction_prediction",
        lambda symbol: {
            "symbol": symbol,
            "status": "not_trained",
            "validation_status": "not_trained",
        },
    )

    response = prediction.get_stock_direction_predictions()

    assert response["status"] == "ready"
    assert [row["symbol"] for row in response["predictions"]] == symbols
    assert all(row["status"] == "not_trained" for row in response["predictions"])
    assert all("upward_probability" not in row for row in response["predictions"])


def test_watchlist_training_reuses_models_and_reports_each_stock_status(monkeypatch):
    watchlist = [
        {"symbol": "RELIANCE", "exchange": "NSE", "type": "EQUITY"},
        {"symbol": "TCS", "exchange": "NSE", "type": "EQUITY"},
        {"symbol": "INFY", "exchange": "NSE", "type": "EQUITY"},
    ]
    monkeypatch.setattr(prediction, "WATCHLIST", watchlist)
    monkeypatch.setattr(
        prediction,
        "get_stock_direction_prediction",
        lambda symbol: {
            "symbol": symbol,
            "exchange": "NSE",
            "status": "ready" if symbol == "RELIANCE" else "not_trained",
            "validation_status": "out_of_sample_evaluated" if symbol == "RELIANCE" else "not_trained",
            "upward_probability": 0.61 if symbol == "RELIANCE" else None,
        },
    )

    def fake_train(symbol: str, *, lstm_epochs: int) -> dict:
        assert lstm_epochs == 3
        if symbol == "TCS":
            return {
                "status": "ready",
                "validation_status": "out_of_sample_evaluated",
                "model_version": "test-version",
                "prediction": {"symbol": symbol, "status": "ready", "upward_probability": 0.55},
            }
        return {
            "status": "insufficient_data",
            "validation_status": "insufficient_data",
            "message": "More history required.",
            "sample_count": 120,
            "required_sample_count": 200,
        }

    monkeypatch.setattr(prediction, "train_stock_direction_model", fake_train)

    result = prediction.train_stock_direction_watchlist(lstm_epochs=3)

    assert result["status"] == "partial"
    assert result["trained_count"] == 1
    assert result["reused_count"] == 1
    assert [row["symbol"] for row in result["predictions"]] == ["RELIANCE", "TCS", "INFY"]
    assert result["predictions"][0]["training_action"] == "existing_model_reused"
    assert result["predictions"][1]["prediction"]["upward_probability"] == 0.55
    assert result["predictions"][2]["status"] == "insufficient_data"
    assert "upward_probability" not in result["predictions"][2]


def test_watchlist_training_job_records_progress_and_final_results(monkeypatch):
    monkeypatch.setattr(
        prediction,
        "train_stock_direction_watchlist",
        lambda *, lstm_epochs, on_progress: (
            on_progress(
                {
                    "symbol": "TCS",
                    "status": "ready",
                    "training_action": "trained",
                    "prediction": {"symbol": "TCS", "status": "ready", "upward_probability": 0.52},
                }
            )
            or {
                "status": "ready",
                "trained_count": 1,
                "reused_count": 0,
                "failed_count": 0,
                "updated_at": "2026-11-04T00:00:00+00:00",
                "notice": "Validated.",
            }
        ),
    )
    job, created = prediction.start_stock_direction_watchlist_job()

    assert created is True
    prediction.run_stock_direction_watchlist_job(job["job_id"], lstm_epochs=3)
    completed = prediction.get_stock_direction_watchlist_job(job["job_id"])

    assert completed is not None
    assert completed["status"] == "ready"
    assert completed["completed_count"] == 1
    assert completed["trained_count"] == 1
    assert completed["predictions"][0]["prediction"]["upward_probability"] == 0.52


def test_chronological_xgboost_lstm_ensemble_is_calibrated_and_versioned(tmp_path, monkeypatch):
    candles = _candles()
    empty_context = {
        "news_sentiment": {},
        "sector_returns": {},
        "market_returns": {},
        "sector": None,
    }
    monkeypatch.setattr(
        prediction,
        "_load_market_data",
        lambda symbol: (prediction._instrument(symbol), candles, empty_context),
    )

    trained = prediction.train_stock_direction_model(
        "RELIANCE",
        artifact_directory=tmp_path,
        lstm_epochs=1,
    )
    predicted = prediction.get_stock_direction_prediction(
        "RELIANCE",
        artifact_directory=tmp_path,
    )

    assert trained["status"] == "ready"
    assert trained["validation_status"] == "out_of_sample_evaluated"
    assert sum(trained["split_sizes"].values()) == trained["sample_count"]
    assert trained["metrics"]["selected"]["sample_count"] > 0
    assert trained["metrics"]["selected"]["brier_score"] >= 0
    assert trained["metrics"]["walk_forward_xgboost"]["method"] == "expanding_window"
    assert trained["prediction"]["status"] == "ready"
    if trained["selection"]["ensemble_used"]:
        assert trained["selection"]["selected_validation_brier"] < min(
            trained["selection"]["xgboost_validation_brier"],
            trained["selection"]["lstm_validation_brier"],
        )
    assert predicted["model_version"] == trained["model_version"]
    assert predicted["upward_probability"] + predicted["downward_probability"] == pytest.approx(1.0)
    assert "not a guarantee" in predicted["probability_note"]
    assert (tmp_path / "RELIANCE" / "latest.txt").is_file()


def test_prediction_api_exposes_saved_status_and_training_metrics(monkeypatch):
    status = {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "status": "insufficient_data",
        "validation_status": "insufficient_data",
        "message": "More history required.",
    }

    def fake_prediction(symbol: str) -> dict:
        if symbol != "RELIANCE":
            raise ValueError(f"Unsupported NSE equity symbol: {symbol}.")
        return status

    def fake_train(symbol: str, *, lstm_epochs: int) -> dict:
        assert symbol == "RELIANCE"
        assert lstm_epochs == 4
        return {**status, "sample_count": 70, "required_sample_count": 200}

    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.get_stock_direction_predictions",
        lambda: {
            "status": "ready",
            "provider": "Yahoo Finance daily history + validated local models",
            "updated_at": "2026-11-04T00:00:00+00:00",
            "predictions": [status],
            "notice": "No placeholder probabilities.",
        },
    )
    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.get_stock_direction_prediction",
        fake_prediction,
    )
    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.train_stock_direction_model",
        fake_train,
    )
    client = TestClient(app)

    saved = client.get("/api/v1/ai-stock-picks/predictions/RELIANCE")
    all_stocks = client.get("/api/v1/ai-stock-picks/predictions")
    trained = client.post(
        "/api/v1/ai-stock-picks/models/train/RELIANCE?lstm_epochs=4"
    )
    unsupported = client.get("/api/v1/ai-stock-picks/predictions/UNKNOWN")

    assert saved.status_code == 200
    assert saved.json()["validation_status"] == "insufficient_data"
    assert all_stocks.status_code == 200
    assert all_stocks.json()["predictions"][0]["status"] == "insufficient_data"
    assert trained.status_code == 200
    assert trained.json()["sample_count"] == 70
    assert unsupported.status_code == 404


def test_watchlist_training_api_returns_per_symbol_results(monkeypatch):
    response = {
        "job_id": "test-job",
        "status": "queued",
        "completed_count": 0,
        "total_count": 3,
        "trained_count": 0,
        "reused_count": 0,
        "failed_count": 0,
        "updated_at": "2026-11-04T00:00:00+00:00",
        "message": "Queued.",
        "predictions": [],
    }
    completed = {
        **response,
        "status": "partial",
        "completed_count": 3,
        "predictions": [
            {"symbol": "RELIANCE", "status": "ready", "training_action": "existing_model_reused"},
            {"symbol": "TCS", "status": "ready", "training_action": "trained"},
            {"symbol": "INFY", "status": "insufficient_data", "training_action": "not_trained"},
        ],
    }
    background_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.start_stock_direction_watchlist_job",
        lambda: (response, True),
    )
    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.run_stock_direction_watchlist_job",
        lambda job_id, *, lstm_epochs: background_calls.append((job_id, lstm_epochs)),
    )
    monkeypatch.setattr(
        "app.api.routes.ai_stock_picks.get_stock_direction_watchlist_job",
        lambda job_id: completed if job_id == "test-job" else None,
    )

    client = TestClient(app)
    result = client.post(
        "/api/v1/ai-stock-picks/models/train?lstm_epochs=4"
    )
    status = client.get("/api/v1/ai-stock-picks/models/train/jobs/test-job")
    missing = client.get("/api/v1/ai-stock-picks/models/train/jobs/unknown")

    assert result.status_code == 202
    assert result.json()["status"] == "queued"
    assert background_calls == [("test-job", 4)]
    assert status.status_code == 200
    assert status.json()["status"] == "partial"
    assert status.json()["predictions"][1]["training_action"] == "trained"
    assert status.json()["predictions"][2]["status"] == "insufficient_data"
    assert missing.status_code == 404
