import re
from typing import Any

import requests
from fastapi import APIRouter, HTTPException

from app.data.ipo_market import fetch_market_data

router = APIRouter(prefix="/api/v1/market", tags=["market"])


def _market_data(endpoint: str) -> dict[str, Any]:
    try:
        return fetch_market_data(endpoint)
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="The Xflot market data service could not be reached. Try again later.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="The Xflot market data service returned an invalid response.",
        ) from exc


@router.get("/ipos")
def list_ipos() -> dict[str, Any]:
    return _market_data("ipos?limit=200")


@router.get("/ipos/{slug}")
def get_ipo(slug: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-z0-9-]+", slug):
        raise HTTPException(status_code=422, detail="Invalid IPO slug.")
    return _market_data(f"ipos/{slug}")


@router.get("/deals")
def list_deals() -> dict[str, Any]:
    return _market_data("deals?limit=300")


@router.get("/scoreboard")
def get_scoreboard() -> dict[str, Any]:
    return _market_data("scoreboard")
