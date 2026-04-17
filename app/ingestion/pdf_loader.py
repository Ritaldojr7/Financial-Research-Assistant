from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract all text from a PDF using pdfplumber.

    Returns the concatenated text of every page, separated by newlines.
    Raises ``ValueError`` when no text can be extracted.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pages_text: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages_text.append(text)
            else:
                logger.warning("No text extracted from page %d of %s", page_num, file_path.name)

    if not pages_text:
        raise ValueError(f"Could not extract any text from {file_path.name}")

    full_text = "\n\n".join(pages_text)
    logger.info(
        "Extracted %d characters from %d pages of %s",
        len(full_text),
        len(pages_text),
        file_path.name,
    )
    return full_text


async def extract_text_from_upload(contents: bytes, filename: str) -> str:
    """Extract text from in-memory PDF bytes (e.g. an uploaded file)."""
    import tempfile

    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        return extract_text_from_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
