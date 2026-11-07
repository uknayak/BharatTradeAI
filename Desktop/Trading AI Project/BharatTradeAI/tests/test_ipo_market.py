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

    first = ipo_market.fetch_market_data("scoreboard")
    second = ipo_market.fetch_market_data("scoreboard")
    assert first["ok"] is True
    assert first["dataFreshness"]["provider"] == "Xflot"
    assert first["dataFreshness"]["status"] == "fresh"
    assert first["dataFreshness"]["cacheAgeSeconds"] == 0
    assert second["dataFreshness"]["status"] == "fresh"
    assert second["dataFreshness"]["cacheAgeSeconds"] >= 0
    assert calls == 1


def test_market_data_returns_last_success_with_stale_status(monkeypatch):
    from app.data import ipo_market

    monkeypatch.setattr(ipo_market, "_cache", {})
    responses = iter([
        {"ok": True, "disclaimer": "Xflot disclaimer"},
        requests.Timeout("upstream timeout"),
    ])

    def fake_get(url, timeout):
        response = next(responses)
        if isinstance(response, Exception):
            raise response

        class PayloadResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return response

        return PayloadResponse()

    monkeypatch.setattr(ipo_market.requests, "get", fake_get)
    first = ipo_market.fetch_market_data("ipos")
    monkeypatch.setattr(ipo_market, "CACHE_TTL_SECONDS", 0)
    stale = ipo_market.fetch_market_data("ipos")

    assert first["dataFreshness"]["status"] == "fresh"
    assert stale["dataFreshness"]["status"] == "stale"
    assert "last successful" in stale["dataFreshness"]["warning"]


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
