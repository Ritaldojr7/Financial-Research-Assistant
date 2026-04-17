from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from app.utils.config import get_settings

if TYPE_CHECKING:
    from app.services.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

RERANK_PROMPT = (
    "You are a relevance scoring engine for financial document retrieval.\n\n"
    "Given a user query and a list of document chunks, score each chunk's "
    "relevance to the query on a scale of 0 to 10.\n\n"
    "Scoring guide:\n"
    "- 10: Directly and completely answers the query with specific data\n"
    "- 7-9: Highly relevant, contains key information for the query\n"
    "- 4-6: Partially relevant, mentions related topics\n"
    "- 1-3: Tangentially related\n"
    "- 0: Completely irrelevant\n\n"
    "Return ONLY a JSON array of objects with keys \"index\" (0-based) and \"score\".\n"
    "Example: [{\"index\": 0, \"score\": 8}, {\"index\": 1, \"score\": 3}]\n"
)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _build_rerank_input(query: str, chunks: list[RetrievedChunk]) -> str:
    parts = [f"Query: {query}\n\nDocuments:"]
    for i, chunk in enumerate(chunks):
        preview = chunk.text[:500]
        meta_bits = []
        if chunk.metadata.get("company"):
            meta_bits.append(chunk.metadata["company"])
        if chunk.metadata.get("year"):
            meta_bits.append(chunk.metadata["year"])
        tag = f" ({', '.join(meta_bits)})" if meta_bits else ""
        parts.append(f"\n[{i}]{tag}\n{preview}")
    return "\n".join(parts)


def _parse_scores(raw: str, n: int) -> list[float]:
    """Extract per-chunk scores from the LLM response, falling back gracefully."""
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        items = json.loads(raw[start:end])
        scores = [0.0] * n
        for item in items:
            idx = int(item["index"])
            if 0 <= idx < n:
                scores[idx] = float(item["score"])
        return scores
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse rerank scores, keeping original order: %s", exc)
        return [float(n - i) for i in range(n)]


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Re-rank *chunks* by LLM-judged relevance and return the best *top_k*.

    Uses a single GPT call that scores all chunks at once (listwise reranking).
    If the call fails for any reason, the original order is preserved.
    """
    if len(chunks) <= top_k:
        return chunks

    settings = get_settings()
    client = _get_client()
    user_msg = _build_rerank_input(query, chunks)

    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": RERANK_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        scores = _parse_scores(raw, len(chunks))
    except Exception:
        logger.exception("Reranking LLM call failed, returning original order")
        return chunks[:top_k]

    ranked = sorted(
        zip(chunks, scores),
        key=lambda pair: pair[1],
        reverse=True,
    )

    result = []
    for chunk, rerank_score in ranked[:top_k]:
        chunk.metadata["rerank_score"] = round(rerank_score, 1)
        result.append(chunk)

    logger.info(
        "Reranked %d → %d chunks (top rerank score=%.1f, lowest=%.1f)",
        len(chunks),
        len(result),
        result[0].metadata.get("rerank_score", 0),
        result[-1].metadata.get("rerank_score", 0),
    )
    return result
