from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, AsyncIterator

from openai import AsyncOpenAI

from app.utils.config import get_settings

if TYPE_CHECKING:
    from app.services.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

ANALYST_SYSTEM_PROMPT = (
    "You are a senior financial analyst with deep expertise in corporate finance, "
    "equity research, and financial statement analysis.\n\n"
    "Rules:\n"
    "- Focus on financial insights, not mere summarization.\n"
    "- Include specific numbers, percentages, and dollar amounts when available.\n"
    "- Use concise bullet points for clarity.\n"
    "- If the provided context is insufficient to answer confidently, say so explicitly.\n"
    "- Never fabricate data or figures not present in the context.\n"
    "- When comparing companies, use a structured side-by-side format.\n"
)

QUERY_TEMPLATE = (
    "Answer the following query using ONLY the provided context.\n\n"
    "Context:\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Query: {query}\n"
)

COMPARISON_TEMPLATE = (
    "The user is asking for a comparison. Retrieve and contrast the key financial "
    "metrics for the companies mentioned.\n\n"
    "Structure your response as:\n"
    "- **Company A vs Company B**\n"
    "  - Revenue\n"
    "  - Growth\n"
    "  - Profitability\n"
    "  - Risks\n"
    "  - Key highlights\n\n"
    "Use the context below. If data for one company is missing, state that clearly.\n\n"
    "Context:\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Query: {query}\n"
)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta_parts = []
        if chunk.metadata.get("company"):
            meta_parts.append(f"Company: {chunk.metadata['company']}")
        if chunk.metadata.get("year"):
            meta_parts.append(f"Year: {chunk.metadata['year']}")
        if chunk.metadata.get("section"):
            meta_parts.append(f"Section: {chunk.metadata['section']}")
        header = " | ".join(meta_parts) if meta_parts else f"Chunk {i}"
        parts.append(f"[{header}]\n{chunk.text}")
    return "\n\n".join(parts)


def _estimate_confidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "low — no relevant documents found"
    avg_score = sum(c.score for c in chunks) / len(chunks)
    if avg_score >= 0.85:
        return "high"
    if avg_score >= 0.70:
        return "medium"
    return "low — retrieved documents may not be closely relevant"


async def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    is_comparison: bool = False,
) -> dict:
    """Call the LLM to produce a financial-analyst-grade answer.

    Returns ``{"answer": str, "sources": list[dict], "confidence": str}``.
    """
    settings = get_settings()
    client = _get_client()

    context = _build_context(chunks)
    template = COMPARISON_TEMPLATE if is_comparison else QUERY_TEMPLATE
    user_prompt = template.format(context=context, query=query)

    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content or ""
    except Exception:
        logger.exception("LLM call failed in generate_answer")
        raise

    sources = [
        {
            "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
            "score": round(c.score, 4),
            **{k: v for k, v in c.metadata.items() if k != "text"},
        }
        for c in chunks
    ]

    confidence = _estimate_confidence(chunks)
    logger.info("Generated answer (%d chars, confidence=%s)", len(answer), confidence)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


def _build_sources(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
            "score": round(c.score, 4),
            **{k: v for k, v in c.metadata.items() if k != "text"},
        }
        for c in chunks
    ]


async def generate_answer_stream(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    is_comparison: bool = False,
) -> AsyncIterator[str]:
    """Stream the LLM answer as Server-Sent Events.

    Yields SSE-formatted lines:
      - ``event: token``  with ``data: <text>``   for each streamed token
      - ``event: sources`` with ``data: <json>``   once at the end
      - ``event: done``    with ``data: <json>``   final metadata
    """
    settings = get_settings()
    client = _get_client()

    context = _build_context(chunks)
    template = COMPARISON_TEMPLATE if is_comparison else QUERY_TEMPLATE
    user_prompt = template.format(context=context, query=query)

    try:
        stream = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            stream=True,
        )
    except Exception:
        logger.exception("LLM streaming call failed")
        yield f"event: error\ndata: {json.dumps({'detail': 'LLM call failed'})}\n\n"
        return

    full_answer = []
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            full_answer.append(delta.content)
            yield f"event: token\ndata: {json.dumps(delta.content)}\n\n"

    sources = _build_sources(chunks)
    confidence = _estimate_confidence(chunks)

    yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
    yield f"event: done\ndata: {json.dumps({'confidence': confidence, 'answer_length': len(''.join(full_answer))})}\n\n"

    logger.info("Streamed answer (%d tokens, confidence=%s)", len(full_answer), confidence)
