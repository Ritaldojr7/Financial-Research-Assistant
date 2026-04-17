from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import tiktoken

from app.utils.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section heading patterns — ordered by specificity (longest match first)
# Covers 10-K, 10-Q, annual reports, and earnings call transcripts.
# ---------------------------------------------------------------------------

SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Management's Discussion and Analysis", re.compile(
        r"^#{0,3}\s*management.s discussion and analysis", re.I | re.M)),
    ("Risk Factors", re.compile(
        r"^#{0,3}\s*risk factors", re.I | re.M)),
    ("Financial Statements", re.compile(
        r"^#{0,3}\s*(?:consolidated |audited )?financial statements", re.I | re.M)),
    ("Notes to Financial Statements", re.compile(
        r"^#{0,3}\s*notes to (?:the )?(?:consolidated )?financial statements", re.I | re.M)),
    ("Income Statement", re.compile(
        r"^#{0,3}\s*(?:consolidated )?(?:statements? of (?:income|operations)|income statements?)", re.I | re.M)),
    ("Balance Sheet", re.compile(
        r"^#{0,3}\s*(?:consolidated )?(?:balance sheets?|statements? of financial (?:position|condition))", re.I | re.M)),
    ("Cash Flow", re.compile(
        r"^#{0,3}\s*(?:consolidated )?statements? of cash flows?", re.I | re.M)),
    ("Stockholders' Equity", re.compile(
        r"^#{0,3}\s*(?:consolidated )?statements? of (?:stockholders|shareholders).? equity", re.I | re.M)),
    ("Revenue", re.compile(
        r"^#{0,3}\s*(?:revenue|net (?:revenue|sales))(?:\s|$)", re.I | re.M)),
    ("Selected Financial Data", re.compile(
        r"^#{0,3}\s*selected (?:consolidated )?financial data", re.I | re.M)),
    ("Business Overview", re.compile(
        r"^#{0,3}\s*(?:business(?: overview)?|overview of (?:the )?(?:company|business))", re.I | re.M)),
    ("Executive Summary", re.compile(
        r"^#{0,3}\s*executive summary", re.I | re.M)),
    ("Forward-Looking Statements", re.compile(
        r"^#{0,3}\s*(?:cautionary )?(?:note )?(?:regarding |about )?forward.looking statements", re.I | re.M)),
    ("Properties", re.compile(
        r"^#{0,3}\s*properties(?:\s|$)", re.I | re.M)),
    ("Legal Proceedings", re.compile(
        r"^#{0,3}\s*legal proceedings", re.I | re.M)),
    ("Market Risk", re.compile(
        r"^#{0,3}\s*(?:quantitative and qualitative )?disclosures? about market risk", re.I | re.M)),
    ("Controls and Procedures", re.compile(
        r"^#{0,3}\s*controls and procedures", re.I | re.M)),
    ("Segment Information", re.compile(
        r"^#{0,3}\s*(?:segment|reportable segment) (?:information|results)", re.I | re.M)),
    ("Dividends", re.compile(
        r"^#{0,3}\s*dividends", re.I | re.M)),
    ("Acquisitions", re.compile(
        r"^#{0,3}\s*(?:acquisitions|business combinations)", re.I | re.M)),
    ("Guidance", re.compile(
        r"^#{0,3}\s*(?:financial )?(?:outlook|guidance)", re.I | re.M)),
    # Earnings-call transcript sections
    ("Prepared Remarks", re.compile(
        r"^#{0,3}\s*(?:prepared remarks|opening remarks)", re.I | re.M)),
    ("Q&A Session", re.compile(
        r"^#{0,3}\s*(?:q(?:uestion)?(?:\s*&\s*|\s+and\s+)a(?:nswer)?\s*(?:session)?)", re.I | re.M)),
]

# Matches lines that look like numeric table rows (3+ numbers/currency values)
_TABLE_ROW = re.compile(
    r"(?:[$€£¥]?\s*[\d,]+\.?\d*\s*%?\s*){3,}"
)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass
class DocumentChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0


@dataclass
class _Section:
    name: str
    text: str
    start_char: int


# ---------------------------------------------------------------------------
# Pass 1: Split into logical sections
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> list[_Section]:
    """Identify section headings and split the document at those boundaries."""
    # Find all heading positions
    boundaries: list[tuple[int, str]] = []
    for name, pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            boundaries.append((m.start(), name))

    if not boundaries:
        return [_Section(name="", text=text, start_char=0)]

    boundaries.sort(key=lambda b: b[0])

    # Deduplicate overlapping matches (keep first match at each position range)
    deduped: list[tuple[int, str]] = []
    for pos, name in boundaries:
        if not deduped or pos - deduped[-1][0] > 20:
            deduped.append((pos, name))

    sections: list[_Section] = []

    # Text before the first detected heading
    if deduped[0][0] > 0:
        preamble = text[: deduped[0][0]].strip()
        if preamble:
            sections.append(_Section(name="Preamble", text=preamble, start_char=0))

    for i, (pos, name) in enumerate(deduped):
        end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        section_text = text[pos:end].strip()
        if section_text:
            sections.append(_Section(name=name, text=section_text, start_char=pos))

    return sections


# ---------------------------------------------------------------------------
# Pass 2: Paragraph-and-table-aware chunking within a section
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, keeping table blocks together."""
    raw_parts = _PARAGRAPH_BREAK.split(text)
    paragraphs: list[str] = []
    table_buffer: list[str] = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        is_table = bool(_TABLE_ROW.search(part))

        if is_table:
            table_buffer.append(part)
        else:
            if table_buffer:
                paragraphs.append("\n".join(table_buffer))
                table_buffer = []
            paragraphs.append(part)

    if table_buffer:
        paragraphs.append("\n".join(table_buffer))

    return paragraphs


def _chunk_section(
    section: _Section,
    enc: tiktoken.Encoding,
    chunk_size: int,
    overlap: int,
    base_meta: dict,
) -> list[DocumentChunk]:
    """Chunk a single section, preferring paragraph boundaries."""
    paragraphs = _split_paragraphs(section.text)
    if not paragraphs:
        return []

    chunks: list[DocumentChunk] = []
    current_parts: list[str] = []
    current_tokens = 0

    def _flush(include_overlap_from: str | None = None):
        nonlocal current_parts, current_tokens
        if not current_parts:
            return
        chunk_text = "\n\n".join(current_parts)
        meta = {
            **base_meta,
            "section": section.name,
            "chunk_index": len(chunks),
        }
        token_count = enc.encode(chunk_text).__len__()
        chunks.append(DocumentChunk(text=chunk_text, metadata=meta, token_count=token_count))
        current_parts = []
        current_tokens = 0

        # Carry forward overlap from the tail of the previous chunk
        if include_overlap_from:
            overlap_tokens = enc.encode(include_overlap_from)
            if len(overlap_tokens) <= overlap:
                current_parts = [include_overlap_from]
                current_tokens = len(overlap_tokens)

    for para in paragraphs:
        para_tokens = len(enc.encode(para))

        # Single paragraph exceeds chunk_size — force-split it by tokens
        if para_tokens > chunk_size:
            _flush()
            tokens = enc.encode(para)
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                piece = enc.decode(tokens[start:end])
                meta = {
                    **base_meta,
                    "section": section.name,
                    "chunk_index": len(chunks),
                }
                chunks.append(DocumentChunk(text=piece, metadata=meta, token_count=end - start))
                start += chunk_size - overlap
            continue

        # Adding this paragraph would exceed the limit — flush first
        if current_tokens + para_tokens > chunk_size and current_parts:
            last_part = current_parts[-1]
            _flush(include_overlap_from=last_part)

        current_parts.append(para)
        current_tokens += para_tokens

    _flush()
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    *,
    company: str = "",
    year: str = "",
    source: str = "",
) -> list[DocumentChunk]:
    """Split a financial document into section-aware, paragraph-aligned chunks.

    Two-pass strategy:
      1. Detect section headings and split the document at those boundaries.
      2. Within each section, accumulate paragraphs up to ``chunk_size`` tokens,
         flushing at paragraph breaks and keeping table rows together.
    """
    settings = get_settings()
    enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = len(enc.encode(text))
    if total_tokens == 0:
        logger.warning("Empty text provided for chunking")
        return []

    base_meta = {
        "company": company,
        "year": year,
        "source": source,
    }

    sections = _split_into_sections(text)
    logger.info(
        "Detected %d sections in %s/%s: %s",
        len(sections),
        company or "unknown",
        year or "unknown",
        [s.name for s in sections],
    )

    all_chunks: list[DocumentChunk] = []
    for section in sections:
        section_chunks = _chunk_section(
            section, enc, settings.chunk_size, settings.chunk_overlap, base_meta
        )
        all_chunks.extend(section_chunks)

    # Re-index globally
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        "Created %d chunks (avg %d tokens) for %s/%s",
        len(all_chunks),
        total_tokens // max(len(all_chunks), 1),
        company or "unknown",
        year or "unknown",
    )
    return all_chunks
