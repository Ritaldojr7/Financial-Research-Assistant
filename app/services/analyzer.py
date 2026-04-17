from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator

from app.services.llm import generate_answer, generate_answer_stream
from app.services.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

COMPARISON_PATTERNS = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bcomparison\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdifference(?:s)? between\b", re.IGNORECASE),
    re.compile(r"\bhow does .+ compare\b", re.IGNORECASE),
]

COMPANY_EXTRACT_PATTERN = re.compile(
    r"\b(?:compare|comparing)\s+(.+?)\s+(?:and|vs\.?|versus|with|to)\s+(.+?)(?:\s+(?:in|for|during)\s+\d{4})?[?.!]?\s*$",
    re.IGNORECASE,
)


def is_comparison_query(query: str) -> bool:
    return any(p.search(query) for p in COMPARISON_PATTERNS)


def extract_companies(query: str) -> list[str]:
    """Best-effort extraction of two company names from a comparison query."""
    m = COMPANY_EXTRACT_PATTERN.search(query)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return []


async def handle_query(
    query: str,
    *,
    company: str | None = None,
    year: str | None = None,
) -> dict:
    """Unified entry point for both simple and comparison queries.

    Returns the dict produced by ``llm.generate_answer``.
    """
    comparison = is_comparison_query(query)

    if comparison:
        return await _handle_comparison(query, year=year)

    chunks = await retrieve(query, company=company, year=year)

    if not chunks:
        return {
            "answer": "No relevant documents were found for your query. Please ingest the relevant financial reports first.",
            "sources": [],
            "confidence": "low — no documents found",
        }

    return await generate_answer(query, chunks)


async def _handle_comparison(query: str, *, year: str | None = None) -> dict:
    """Retrieve chunks for each company mentioned and generate a comparison."""
    all_chunks = await _gather_comparison_chunks(query, year=year)

    if not all_chunks:
        return {
            "answer": "No relevant documents were found for the companies in your comparison query. Please ingest reports for both companies first.",
            "sources": [],
            "confidence": "low — no documents found",
        }

    return await generate_answer(query, all_chunks, is_comparison=True)


# ---------------------------------------------------------------------------
# Streaming variants
# ---------------------------------------------------------------------------

async def handle_query_stream(
    query: str,
    *,
    company: str | None = None,
    year: str | None = None,
) -> AsyncIterator[str]:
    """Streaming version of ``handle_query``. Yields SSE-formatted events."""
    comparison = is_comparison_query(query)

    if comparison:
        chunks = await _gather_comparison_chunks(query, year=year)
    else:
        chunks = await retrieve(query, company=company, year=year)

    if not chunks:
        no_docs = {
            "answer": "No relevant documents were found for your query. Please ingest the relevant financial reports first.",
            "sources": [],
            "confidence": "low — no documents found",
        }
        yield f"event: token\ndata: {json.dumps(no_docs['answer'])}\n\n"
        yield f"event: sources\ndata: {json.dumps([])}\n\n"
        yield f"event: done\ndata: {json.dumps({'confidence': no_docs['confidence'], 'answer_length': len(no_docs['answer'])})}\n\n"
        return

    async for event in generate_answer_stream(query, chunks, is_comparison=comparison):
        yield event


async def _gather_comparison_chunks(
    query: str, *, year: str | None = None
) -> list[RetrievedChunk]:
    """Retrieve chunks for a comparison query (shared by sync and stream paths)."""
    companies = extract_companies(query)
    all_chunks: list[RetrievedChunk] = []
    if len(companies) >= 2:
        for comp in companies[:2]:
            all_chunks.extend(await retrieve(query, company=comp, year=year))
    else:
        all_chunks = await retrieve(query, year=year)
    return all_chunks


# ---------------------------------------------------------------------------
# Basic evaluation helper
# ---------------------------------------------------------------------------


async def evaluate_retrieval(
    query: str,
    expected_keywords: list[str],
    *,
    company: str | None = None,
    year: str | None = None,
) -> dict:
    """Quick relevance check: what fraction of *expected_keywords* appear in
    the retrieved chunks. Useful for smoke-testing retrieval quality.

    Returns ``{"recall": float, "matched": list[str], "missing": list[str]}``.
    """
    chunks = await retrieve(query, company=company, year=year)
    combined_text = " ".join(c.text for c in chunks).lower()

    matched = [kw for kw in expected_keywords if kw.lower() in combined_text]
    missing = [kw for kw in expected_keywords if kw.lower() not in combined_text]
    recall = len(matched) / max(len(expected_keywords), 1)

    return {"recall": round(recall, 3), "matched": matched, "missing": missing}
