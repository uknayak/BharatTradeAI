from fastapi import APIRouter, HTTPException
from fastapi import Query

from app.services.market.historical_data import (
    SyncCooldownError,
    SyncInProgressError,
    get_daily_history,
    sync_daily_history,
)
from app.services.market.live_quotes import fetch_live_quotes

router = APIRouter(prefix="/api/v1/stocks", tags=["stocks"])


@router.get("/live")
def live_stock_quotes() -> dict:
    response = fetch_live_quotes()
    if response["status"] == "error":
        raise HTTPException(status_code=502, detail=response["message"])
    return response


@router.get("/history/{symbol}")
def daily_stock_history(symbol: str, limit: int = Query(default=100, ge=1, le=500)) -> dict:
    try:
        return get_daily_history(symbol, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/history/sync")
def sync_stock_history() -> dict:
    try:
        result = sync_daily_history(manual=True)
    except SyncCooldownError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SyncInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if result["status"] == "error":
        details = "; ".join(result["warnings"])
        raise HTTPException(
            status_code=502,
            detail=details or "Yahoo Finance history sync failed without returning market data.",
        )
    return result
