from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.analyzer import handle_query, handle_query_stream
from app.utils.rate_limit import limiter
from app.utils.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"], dependencies=[Depends(require_api_key)])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The financial research question")
    company: str | None = Field(None, description="Optional company filter")
    year: str | None = Field(None, description="Optional year filter (e.g. '2024')")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: str


@router.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query_documents(request: Request, body: QueryRequest):
    """Query ingested financial documents using RAG."""
    logger.info(
        "Query received: %.80s (company=%s, year=%s)",
        body.query,
        body.company,
        body.year,
    )

    try:
        result = await handle_query(
            body.query,
            company=body.company,
            year=body.year,
        )
    except Exception:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="Query processing failed. Please try again later.")

    return QueryResponse(**result)


async def _safe_stream(query: str, company: str | None, year: str | None) -> AsyncIterator[str]:
    """Wrap the streaming generator so errors become an SSE error event."""
    try:
        async for event in handle_query_stream(query, company=company, year=year):
            yield event
    except Exception:
        logger.exception("Error during streaming query")
        yield f"event: error\ndata: {json.dumps({'detail': 'Streaming failed. Please try again later.'})}\n\n"


@router.post("/query/stream")
@limiter.limit("10/minute")
async def query_documents_stream(request: Request, body: QueryRequest):
    """Stream the answer token-by-token via Server-Sent Events.

    **Event types:**

    | event     | data payload                              |
    |-----------|-------------------------------------------|
    | `token`   | JSON string - one text fragment            |
    | `sources` | JSON array  - retrieved source chunks      |
    | `done`    | JSON object - `{confidence, answer_length}`|
    | `error`   | JSON object - `{detail}` on failure        |
    """
    logger.info(
        "Stream query received: %.80s (company=%s, year=%s)",
        body.query,
        body.company,
        body.year,
    )

    return StreamingResponse(
        _safe_stream(body.query, body.company, body.year),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
