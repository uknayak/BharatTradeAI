import requests

from config.settings import settings


def get_market_status() -> dict:
    try:
        response = requests.get(settings.nse_base_url, timeout=10)
        status = "online" if response.status_code == 200 else "offline"
        return {
            "source": "NSE",
            "status": status,
            "response_code": response.status_code,
            "message": response.text[:200] if response.text else "",
        }
    except Exception as exc:  # pragma: no cover
        return {
            "source": "NSE",
            "status": "offline",
            "response_code": None,
            "message": str(exc),
        }
