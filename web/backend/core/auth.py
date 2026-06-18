"""FastAPI dependencies for Supabase JWT authentication."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from web.backend.core.supabase import get_supabase


async def get_current_user_id(request: Request) -> str:
    """Extract and verify user_id from Supabase JWT in Authorization header.

    Returns the user's UUID string.
    Raises 401 if missing or invalid.
    """
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = auth_header[7:]  # strip "Bearer "
    try:
        sb = get_supabase()
        user_resp = sb.auth.get_user(token)
        if user_resp is None or user_resp.user is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_resp.user.id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
