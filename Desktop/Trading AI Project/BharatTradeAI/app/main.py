import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI

from app.api.routes.broker import router as broker_router
from app.api.routes.ai_stock_picks import router as ai_stock_picks_router
from app.api.routes.db_status import router as db_status_router
from app.api.routes.health import router as health_router
from app.api.routes.market import router as market_router
from app.api.routes.news import router as news_router
from app.api.routes.stocks import router as stocks_router
from app.db.init_db import init_db
from app.services.market.historical_data import sync_daily_history

logger = logging.getLogger(__name__)
MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
HISTORY_SYNC_TIME = time(hour=16, minute=15)


def _next_history_sync(now: datetime) -> datetime:
    target = datetime.combine(now.date(), HISTORY_SYNC_TIME, tzinfo=MARKET_TIMEZONE)
    if target <= now:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


async def _market_history_scheduler() -> None:
    while True:
        now = datetime.now(MARKET_TIMEZONE)
        target = _next_history_sync(now)
        await asyncio.sleep((target - now).total_seconds())
        try:
            result = await asyncio.to_thread(sync_daily_history)
            if result["warnings"]:
                logger.warning(
                    "Scheduled Yahoo market history sync completed with warnings: %s",
                    "; ".join(result["warnings"]),
                )
        except Exception:
            logger.exception("Scheduled Yahoo daily market history sync failed.")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    task = asyncio.create_task(_market_history_scheduler())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="BharatTradeAI",
    description="Indian market intelligence and alert platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(db_status_router)
app.include_router(broker_router)
app.include_router(ai_stock_picks_router)
app.include_router(news_router)
app.include_router(market_router)
app.include_router(stocks_router)


@app.get("/")
def root() -> dict:
    return {
        "message": "Welcome to BharatTradeAI",
        "status": "running",
    }
