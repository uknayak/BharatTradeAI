import requests
from fastapi.testclient import TestClient

from app.main import app


def test_market_routes_proxy_all_xflot_resources(monkeypatch):
    from app.data import ipo_market

    calls = []

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "disclaimer": "Xflot disclaimer"}

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return DummyResponse()

    monkeypatch.setattr(ipo_market.requests, "get", fake_get)
    monkeypatch.setattr(ipo_market, "_cache", {})

    client = TestClient(app)
    responses = [
        client.get("/api/v1/market/ipos"),
        client.get("/api/v1/market/ipos/example-ipo"),
        client.get("/api/v1/market/deals"),
        client.get("/api/v1/market/scoreboard"),
    ]

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["disclaimer"] == "Xflot disclaimer" for response in responses)
    assert [url for url, _ in calls] == [
        "https://www.xflot.com/api/public/market/ipos?limit=200",
        "https://www.xflot.com/api/public/market/ipos/example-ipo",
        "https://www.xflot.com/api/public/market/deals?limit=300",
        "https://www.xflot.com/api/public/market/scoreboard",
    ]
    assert all(timeout == 10 for _, timeout in calls)


def test_market_data_cache_limits_upstream_polling(monkeypatch):
    from app.data import ipo_market

    calls = 0

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "count": 0, "disclaimer": "Xflot disclaimer"}

    def fake_get(url, timeout):
        nonlocal calls
        calls += 1
        return DummyResponse()

    monkeypatch.setattr(ipo_market.requests, "get", fake_get)
    monkeypatch.setattr(ipo_market, "_cache", {})

    expected = {"ok": True, "count": 0, "disclaimer": "Xflot disclaimer"}
    assert ipo_market.fetch_market_data("scoreboard") == expected
    assert ipo_market.fetch_market_data("scoreboard") == expected
    assert calls == 1


def test_market_proxy_reports_upstream_errors_and_rejects_bad_slugs(monkeypatch):
    from app.data import ipo_market

    monkeypatch.setattr(ipo_market, "_cache", {})

    def fail_get(url, timeout):
        raise requests.Timeout("upstream timeout")

    monkeypatch.setattr(ipo_market.requests, "get", fail_get)
    client = TestClient(app)

    failed_upstream = client.get("/api/v1/market/scoreboard")
    invalid_slug = client.get("/api/v1/market/ipos/not_valid")

    assert failed_upstream.status_code == 502
    assert invalid_slug.status_code == 422
