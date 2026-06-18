"""Rate limiting configuration.

Three tiers based on cost/risk:
- RATE_AI_SUMMARIES:  Arbitrary AI summary generation (OpenAI cost / abuse risk)
- RATE_NEWS:          News endpoints (cached, but may hit Yahoo/Naver/OpenAI on miss)
- RATE_EXTERNAL_API:  Endpoints that proxy to Yahoo/Naver (IP ban risk)
- RATE_DEFAULT:       General endpoints
"""
from __future__ import annotations

import logging
import threading
import time

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

# Rate limit tiers (per IP)
RATE_AI_SUMMARIES = "10/minute"
RATE_NEWS = "30/minute"
RATE_EXTERNAL_API = "30/minute"
RATE_DEFAULT = "60/minute"

_RATE_UNITS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
}

_MISS_QUOTA_KEY_PREFIX = "sg:missq:"


def _parse_rate(rate: str) -> tuple[int, int]:
    count_text, unit_text = rate.strip().split("/", 1)
    count = int(count_text)
    unit = unit_text.strip().lower().rstrip("s")
    if unit not in _RATE_UNITS:
        raise ValueError(f"Unsupported rate unit: {unit_text}")
    return count, _RATE_UNITS[unit]


class _MissQuota:
    """Miss-only quota store with Redis support and in-memory fallback."""

    def __init__(self) -> None:
        self._redis = None
        self._fallback: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()
        self._init_redis()

    def _init_redis(self) -> None:
        from web.backend.core.config import settings

        url = settings.redis_url
        if not url:
            return

        try:
            import redis as redis_lib

            self._redis = redis_lib.Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._redis.ping()
        except Exception as exc:
            logger.warning("Miss quota Redis init failed, using in-memory fallback: %s", exc)
            self._redis = None

    def consume(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        window_slot = int(now // window_seconds)
        retry_after = max(1, int(window_seconds - (int(now) % window_seconds)))
        full_key = f"{_MISS_QUOTA_KEY_PREFIX}{key}:{window_slot}"

        if self._redis is not None:
            try:
                current = int(self._redis.incr(full_key))
                if current == 1:
                    self._redis.expire(full_key, retry_after)
                return current <= limit, retry_after
            except Exception as exc:
                logger.warning("Miss quota Redis consume failed for %s: %s", key, exc)

        return self._fallback_consume(full_key, limit, retry_after, now)

    def _fallback_consume(
        self,
        full_key: str,
        limit: int,
        retry_after: int,
        now: float,
    ) -> tuple[bool, int]:
        expires_at = now + retry_after
        with self._lock:
            entry = self._fallback.get(full_key)
            if entry is None or entry[0] <= now:
                current = 1
            else:
                expires_at = entry[0]
                current = entry[1] + 1
            self._fallback[full_key] = (expires_at, current)

            if len(self._fallback) > 1024:
                self._fallback = {
                    key: value for key, value in self._fallback.items() if value[0] > now
                }

        return current <= limit, max(1, int(expires_at - now))


_miss_quota_instance: _MissQuota | None = None
_miss_quota_lock = threading.Lock()


def _get_miss_quota() -> _MissQuota:
    global _miss_quota_instance
    if _miss_quota_instance is None:
        with _miss_quota_lock:
            if _miss_quota_instance is None:
                _miss_quota_instance = _MissQuota()
    return _miss_quota_instance


def enforce_miss_quota(
    request: Request,
    *,
    rate: str,
    scope: str,
    detail: str,
) -> None:
    """Enforce a quota only when the caller already knows this is a cache miss."""

    limit, window_seconds = _parse_rate(rate)
    identity = get_remote_address(request) or "unknown"
    allowed, retry_after = _get_miss_quota().consume(
        key=f"{scope}:{identity}",
        limit=limit,
        window_seconds=window_seconds,
    )
    if allowed:
        return

    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(retry_after)},
    )
