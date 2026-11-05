from app.services.market_data import get_market_status


def test_market_status_reports_source(monkeypatch):
    class DummyResponse:
        status_code = 200
        text = "ok"

    def fake_get(url, timeout=10):
        return DummyResponse()

    monkeypatch.setattr("app.services.market_data.requests.get", fake_get)

    result = get_market_status()

    assert result["source"] == "NSE"
    assert result["status"] == "online"
