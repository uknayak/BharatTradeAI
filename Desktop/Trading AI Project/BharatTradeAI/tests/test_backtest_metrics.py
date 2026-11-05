from app.ml.backtest import evaluate_backtest, run_backtest


def test_backtest_reports_requested_years_metrics_and_signal_breakdown():
    records = [
        {"year": 2022, "prediction": 1, "actual": 1, "pnl": 10, "signal": "Strong Buy"},
        {"year": 2022, "prediction": 1, "actual": 0, "pnl": -5, "signal": "Strong Buy"},
        {"year": 2023, "prediction": 0, "actual": 1, "pnl": -2, "signal": "Watch"},
        {"year": 2024, "prediction": 1, "actual": 1, "pnl": 8, "signal": "Buy"},
        {"year": 2025, "prediction": 0, "actual": 0, "pnl": 4, "signal": "Strong Sell"},
        {"year": 2026, "prediction": 1, "actual": 1, "pnl": 6, "signal": "Buy"},
    ]

    result = evaluate_backtest(records)

    assert list(result["years"]) == ["2022", "2023", "2024", "2025", "2026"]
    assert result["overall"]["accuracy"] == 4 / 6
    assert result["overall"]["precision"] == 3 / 4
    assert result["overall"]["recall"] == 3 / 4
    assert result["overall"]["win_rate"] == 4 / 6
    assert result["signal_breakdown"]["Buy"] == {"count": 2, "win_rate": 1.0}
    assert result["signal_breakdown"]["Strong Sell"]["win_rate"] == 1.0


def test_backtest_without_records_is_explicitly_not_ready():
    assert run_backtest() == {"module": "backtest", "status": "no_data"}