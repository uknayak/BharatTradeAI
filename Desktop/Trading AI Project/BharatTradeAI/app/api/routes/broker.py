from fastapi import APIRouter

from app.broker.session import build_broker_from_settings

router = APIRouter(prefix="/api/v1", tags=["broker"])


@router.get("/broker/login")
def login_broker(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.login()


@router.get("/broker/token")
def broker_token(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.get_access_token()


@router.get("/broker/profile")
def broker_profile(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.get_profile()


@router.get("/broker/funds")
def broker_funds(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.get_funds()


@router.get("/broker/holdings")
def broker_holdings(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.get_holdings()


@router.get("/broker/positions")
def broker_positions(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.get_positions()


@router.get("/broker/orders")
def broker_orders(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.get_orders()


@router.get("/broker/instruments")
def broker_instruments(provider: str = "upstox") -> dict:
    client = build_broker_from_settings(provider)
    return client.get_instruments()
