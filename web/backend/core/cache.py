"""Shared Redis cache with automatic in-memory fallback.

Uses Upstash Redis when UPSTASH_REDIS_URL is set.
Falls back to a local dict cache for local development.
"""
from __future__ import annotations

import json
import logging
import time
import threading
from typing import Any

logger = logging.getLogger(__name__)

_KEY_PREFIX = "sg:"


class _Cache:
    """Redis-backed cache with in-memory fallback."""

    def __init__(self) -> None:
        self._redis = None
        self._fallback: dict[str, tuple[float, float, Any]] = {}
        self._lock = threading.Lock()
        self._init_redis()

    def _init_redis(self) -> None:
        from web.backend.core.config import settings

        url = settings.redis_url
        if not url:
            logger.info("UPSTASH_REDIS_URL not set — using in-memory cache fallback")
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
            logger.info("Connected to Upstash Redis")
        except Exception as exc:
            logger.warning("Redis connection failed, falling back to in-memory: %s", exc)
            self._redis = None

    def get(self, key: str) -> Any | None:
        full_key = f"{_KEY_PREFIX}{key}"
        if self._redis is not None:
            try:
                raw = self._redis.get(full_key)
                if raw is None:
                    return None
                return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis GET failed for %s: %s", key, exc)
                return self._fallback_get(key)
        return self._fallback_get(key)

    def set(self, key: str, value: Any, ttl: int) -> None:
        if isinstance(value, set):
            value = list(value)
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        full_key = f"{_KEY_PREFIX}{key}"
        if self._redis is not None:
            try:
                self._redis.setex(full_key, ttl, serialized)
                return
            except Exception as exc:
                logger.warning("Redis SET failed for %s: %s", key, exc)
        self._fallback_set(key, value, ttl)

    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a glob pattern (e.g. 'kr:*'). Returns count."""
        full_pattern = f"{_KEY_PREFIX}{pattern}"
        deleted = 0
        if self._redis is not None:
            try:
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=full_pattern, count=100)
                    if keys:
                        self._redis.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
            except Exception as exc:
                logger.warning("Redis DELETE pattern failed for %s: %s", pattern, exc)
        # Also clear from fallback
        with self._lock:
            import fnmatch
            to_delete = [k for k in self._fallback if fnmatch.fnmatch(k, pattern)]
            for k in to_delete:
                del self._fallback[k]
            deleted += len(to_delete)
        return deleted

    def _fallback_get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._fallback.get(key)
            if entry is None:
                return None
            ts, ttl, data = entry
            if time.time() - ts > ttl:
                del self._fallback[key]
                return None
            return data

    def _fallback_set(self, key: str, value: Any, ttl: int) -> None:
        with self._lock:
            self._fallback[key] = (time.time(), ttl, value)


_instance: _Cache | None = None
_init_lock = threading.Lock()


def _get_cache() -> _Cache:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = _Cache()
    return _instance


def cache_get(key: str) -> Any | None:
    return _get_cache().get(key)


def cache_set(key: str, value: Any, ttl: int) -> None:
    _get_cache().set(key, value, ttl)


def cache_delete_pattern(pattern: str) -> int:
    return _get_cache().delete_pattern(pattern)
