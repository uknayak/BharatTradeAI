from typing import Any, Dict

from app.broker.factory import get_broker_client
from config.settings import settings


def build_broker_from_settings(provider: str | None = None) -> Any:
    provider_name = (provider or settings.broker_provider or "upstox").lower()

    kwargs: Dict[str, str] = {
        "api_key": getattr(settings, "upstox_api_key", ""),
        "api_secret": getattr(settings, "upstox_api_secret", ""),
        "redirect_uri": getattr(settings, "upstox_redirect_uri", ""),
        "code": getattr(settings, "upstox_code", ""),
        "access_token": getattr(settings, "upstox_access_token", ""),
    }

    return get_broker_client(provider_name, **kwargs)
