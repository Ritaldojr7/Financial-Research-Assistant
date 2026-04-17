from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from app.utils.config import get_settings

if TYPE_CHECKING:
    from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts using OpenAI."""
    settings = get_settings()
    client = _get_client()

    all_embeddings: list[list[float]] = []
    batch_size = 512

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        all_embeddings.extend([item.embedding for item in response.data])

    logger.info("Generated %d embeddings", len(all_embeddings))
    return all_embeddings


async def embed_query(query: str) -> list[float]:
    """Generate a single embedding for a search query."""
    embeddings = await embed_texts([query])
    return embeddings[0]


async def embed_chunks(chunks: list[DocumentChunk]) -> list[list[float]]:
    """Generate embeddings for a list of DocumentChunks."""
    texts = [c.text for c in chunks]
    return await embed_texts(texts)
