from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.routes.extract import router as extract_router
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router
from app.utils.config import get_settings
from app.utils.rate_limit import limiter


def _setup_logging():
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    logger = logging.getLogger(__name__)
    settings = get_settings()
    logger.info("Starting %s v%s", settings.app_title, settings.app_version)

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set — LLM calls will fail")
    if not settings.pinecone_api_key:
        logger.warning("PINECONE_API_KEY is not set — vector storage will fail")
    if not settings.api_key:
        logger.warning("API_KEY is not set — endpoints are unauthenticated")

    yield

    # --- Shutdown: close cached AsyncOpenAI clients ----------------------
    from app.services import extractor, llm, reranker
    from app.ingestion import embedder

    for mod in (llm, reranker, extractor, embedder):
        client = getattr(mod, "_client", None)
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

    logger.info("Shutting down %s", settings.app_title)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description="RAG-powered financial research assistant that analyses annual reports and earnings transcripts.",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logging.getLogger(__name__).exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Please try again later."},
        )

    app.include_router(query_router)
    app.include_router(ingest_router)
    app.include_router(extract_router)

    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "ok", "version": settings.app_version}

    return app


app = create_app()
