from app.broker.factory import get_broker_client


def test_broker_factory_returns_upstox_client():
    client = get_broker_client("upstox")
    assert client.__class__.__name__ == "UpstoxBroker"


def test_upstox_client_has_required_methods():
    client = get_broker_client("upstox")
    for method_name in [
        "login",
        "get_access_token",
        "build_auth_url",
        "exchange_code_for_token",
        "get_profile",
        "get_funds",
        "get_holdings",
        "get_positions",
        "get_orders",
        "get_instruments",
    ]:
        assert hasattr(client, method_name)


def test_upstox_auth_url_builds_expected_url():
    client = get_broker_client("upstox", api_key="demo-key", redirect_uri="https://example.com/callback")
    url = client.build_auth_url()

    assert "https://api.upstox.com/v2/login/authorization/dialog" in url
    assert "client_id=demo-key" in url
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in url


def test_upstox_token_exchange_sets_access_token(monkeypatch):
    client = get_broker_client(
        "upstox",
        api_key="demo-key",
        api_secret="demo-secret",
        redirect_uri="https://example.com/callback",
        code="demo-code",
    )

    class DummyResponse:
        status_code = 200

        def json(self):
            return {"access_token": "live-token"}

        def raise_for_status(self):
            return None

    def fake_post(url, data, timeout):
        assert url == "https://api.upstox.com/v2/login/authorization/token"
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == "demo-code"
        return DummyResponse()

    monkeypatch.setattr("app.broker.common.requests.post", fake_post)

    result = client.exchange_code_for_token()

    assert result["status"] == "ok"
    assert client.access_token == "live-token"

