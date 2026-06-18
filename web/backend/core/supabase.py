"""Supabase client singleton for the backend."""
from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from web.backend.core.config import settings


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Return a cached Supabase admin client (service-role key)."""
    url = settings.supabase_url_resolved
    key = settings.supabase_service_role_key_resolved
    if not url or not key:
        raise RuntimeError(
            "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars."
        )
    return create_client(url, key)
