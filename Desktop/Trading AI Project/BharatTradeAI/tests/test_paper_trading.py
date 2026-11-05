import pytest

from app.paper_trading import PaperTradingService


def test_paper_order_tracks_signal_without_live_execution_and_compares_outcome():
    service = PaperTradingService()

    order = service.place_order({"signal": "BUY", "score": 82}, "TCS", quantity=2, entry_price=100)
    assert order["order_id"] == "PAPER-000001"
    assert order["status"] == "OPEN"
    assert service.list_orders("OPEN")[0]["symbol"] == "TCS"

    closed = service.close_order(order["order_id"], exit_price=110)
    assert closed["status"] == "CLOSED"
    assert closed["pnl"] == 20
    assert closed["actual_direction"] == "BUY"
    assert closed["prediction_correct"] is True
    assert service.list_orders("OPEN") == []


def test_paper_trading_rejects_invalid_signals_and_duplicate_close():
    service = PaperTradingService()
    with pytest.raises(ValueError):
        service.place_order({"signal": "HOLD"}, "TCS", 1, 100)
    order = service.place_order({"signal": "SELL"}, "TCS", 1, 100)
    service.close_order(order["order_id"], 90)
    with pytest.raises(ValueError):
        service.close_order(order["order_id"], 80)