from redis import Redis
from redis.exceptions import RedisError

from config.settings import settings


def get_redis_client() -> Redis:
    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
    )


def check_redis() -> dict:
    client = get_redis_client()
    try:
        return {
            "status": "connected" if client.ping() else "unavailable",
            "detail": "redis ping successful",
        }
    except RedisError as exc:
        return {"status": "unavailable", "detail": str(exc)}
    except Exception as exc:  # pragma: no cover
        return {"status": "error", "detail": str(exc)}
