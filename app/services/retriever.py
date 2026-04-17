from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from pinecone import Pinecone, ServerlessSpec

from app.ingestion.chunker import DocumentChunk
from app.ingestion.embedder import embed_chunks, embed_query
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

_index = None


# ---------------------------------------------------------------------------
# Simple TTL LRU cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    value: object
    expires_at: float


class TTLCache:
    """Lightweight in-memory LRU cache with per-entry TTL."""

    def __init__(self, max_size: int = 128, ttl: float = 600):
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return entry.value

    def put(self, key: str, value: object):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + self._ttl)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)


_cache: TTLCache | None = None


def _get_cache() -> TTLCache:
    global _cache
    if _cache is None:
        s = get_settings()
        _cache = TTLCache(max_size=s.cache_max_size, ttl=s.cache_ttl_seconds)
    return _cache


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

def _get_index():
    global _index
    if _index is not None:
        return _index

    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info("Creating Pinecone index '%s'", settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    _index = pc.Index(settings.pinecone_index_name)
    return _index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


async def store_chunks(chunks: list[DocumentChunk]) -> int:
    """Embed chunks and upsert them into Pinecone. Returns count stored."""
    if not chunks:
        return 0

    embeddings = await embed_chunks(chunks)
    index = _get_index()

    vectors = []
    for chunk, emb in zip(chunks, embeddings):
        doc_id = hashlib.sha256(chunk.text.encode()).hexdigest()[:32]
        vectors.append({
            "id": doc_id,
            "values": emb,
            "metadata": {**chunk.metadata, "text": chunk.text},
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        await asyncio.to_thread(index.upsert, vectors=batch)

    logger.info("Upserted %d vectors into Pinecone", len(vectors))
    return len(vectors)


async def retrieve(
    query: str,
    *,
    company: str | None = None,
    year: str | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Semantic search against Pinecone, with optional metadata filters.

    When reranking is enabled, fetches ``rerank_initial_k`` candidates from
    Pinecone, re-ranks them with the LLM, and returns the best ``top_k``.
    """
    from app.services.reranker import rerank

    settings = get_settings()
    final_k = top_k or settings.retrieval_top_k
    fetch_k = settings.rerank_initial_k if settings.rerank_enabled else final_k

    cache_key = hashlib.md5(
        f"{query}|{company}|{year}|{final_k}|{settings.rerank_enabled}".encode()
    ).hexdigest()
    cached = _get_cache().get(cache_key)
    if cached is not None:
        logger.info("Cache HIT for query (key=%s)", cache_key[:8])
        return cached  # type: ignore[return-value]

    query_embedding = await embed_query(query)
    index = _get_index()

    metadata_filter: dict = {}
    if company:
        metadata_filter["company"] = {"$eq": company}
    if year:
        metadata_filter["year"] = {"$eq": year}

    results = await asyncio.to_thread(
        index.query,
        vector=query_embedding,
        top_k=fetch_k,
        include_metadata=True,
        filter=metadata_filter if metadata_filter else None,
    )

    chunks = [
        RetrievedChunk(
            text=match.metadata.pop("text", ""),
            score=match.score,
            metadata=match.metadata,
        )
        for match in results.matches
    ]

    logger.info("Retrieved %d candidates from Pinecone (score range %.3f–%.3f)",
                len(chunks),
                chunks[-1].score if chunks else 0,
                chunks[0].score if chunks else 0)

    if settings.rerank_enabled and len(chunks) > final_k:
        chunks = await rerank(query, chunks, final_k)

    _get_cache().put(cache_key, chunks)
    return chunks
