from flask import current_app
from redis import Redis


def get_redis_client() -> Redis:
    """Return a shared Redis client if available, otherwise create one."""
    client = getattr(current_app, "redis_client", None)
    if client is None:
        client = Redis.from_url(current_app.config["REDIS_URL"])
    return client
