from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.utils.config import get_settings

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_header)) -> str | None:
    """Validate the ``X-API-Key`` header when ``API_KEY`` is configured.

    If no ``API_KEY`` is set in the environment the check is skipped,
    allowing unauthenticated local development.
    """
    settings = get_settings()
    if not settings.api_key:
        return None

    if not key or key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return key
