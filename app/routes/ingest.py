from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_loader import extract_text_from_upload
from app.services.retriever import store_chunks
from app.utils.rate_limit import limiter
from app.utils.auth import require_api_key
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"], dependencies=[Depends(require_api_key)])


class IngestResponse(BaseModel):
    message: str
    chunks_stored: int
    total_characters: int


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="PDF file to ingest"),
    company: str = Form("", description="Company name"),
    year: str = Form("", description="Fiscal year"),
):
    """Upload a PDF financial document, extract text, chunk, embed, and store in Pinecone."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    logger.info("Ingesting file=%s company=%s year=%s", file.filename, company, year)

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        settings = get_settings()
        if len(contents) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum upload size of {settings.max_upload_mb}MB",
            )

        text = await extract_text_from_upload(contents, file.filename)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("Failed to extract text from PDF")
        raise HTTPException(status_code=500, detail="PDF extraction failed. Please try again later.")

    chunks = chunk_text(text, company=company, year=year, source=file.filename)

    if not chunks:
        raise HTTPException(status_code=422, detail="No text chunks could be produced from the document")

    try:
        stored = await store_chunks(chunks)
    except Exception:
        logger.exception("Failed to store chunks in vector DB")
        raise HTTPException(status_code=500, detail="Vector storage failed. Please try again later.")

    return IngestResponse(
        message=f"Successfully ingested '{file.filename}'",
        chunks_stored=stored,
        total_characters=len(text),
    )
