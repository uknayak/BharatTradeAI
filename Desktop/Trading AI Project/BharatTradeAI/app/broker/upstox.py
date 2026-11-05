from app.broker.common import BrokerClient


class UpstoxBroker(BrokerClient):
    provider_name = "upstox"
    base_url = "https://api.upstox.com/v2"
    auth_url_base = "https://api.upstox.com/v2/login/authorization/dialog"
    token_url = "https://api.upstox.com/v2/login/authorization/token"

    def get_profile(self):
        return self._request_json("/user/profile")

    def get_funds(self):
        return self._request_json("/user/get-funds")

    def get_holdings(self):
        return self._request_json("/portfolio/short-term-holdings")

    def get_positions(self):
        return self._request_json("/position")

    def get_orders(self):
        return self._request_json("/order")

    def get_instruments(self):
        return self._request_json("/instruments")
