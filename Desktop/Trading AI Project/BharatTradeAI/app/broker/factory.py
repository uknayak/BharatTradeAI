from app.broker.upstox import UpstoxBroker


def get_broker_client(provider: str = "upstox", **kwargs):
    provider_name = (provider or "upstox").lower()
    if provider_name == "upstox":
        return UpstoxBroker(**kwargs)
    raise ValueError(f"Unsupported broker provider: {provider}")
