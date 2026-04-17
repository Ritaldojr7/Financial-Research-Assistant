from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.services.retriever import retrieve
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# The JSON schema the LLM must follow. Described inline so the model sees it.
METRICS_SCHEMA = {
    "company": "string",
    "year": "string",
    "currency": "string (e.g. USD)",
    "income_statement": {
        "revenue": {"value": "number or null", "unit": "string (e.g. 'B', 'M')"},
        "cost_of_revenue": {"value": "number or null", "unit": "string"},
        "gross_profit": {"value": "number or null", "unit": "string"},
        "gross_margin_pct": "number or null",
        "operating_expenses": {"value": "number or null", "unit": "string"},
        "operating_income": {"value": "number or null", "unit": "string"},
        "operating_margin_pct": "number or null",
        "net_income": {"value": "number or null", "unit": "string"},
        "net_margin_pct": "number or null",
        "ebitda": {"value": "number or null", "unit": "string"},
        "ebitda_margin_pct": "number or null",
        "eps_basic": "number or null",
        "eps_diluted": "number or null",
        "rd_expenses": {"value": "number or null", "unit": "string"},
    },
    "balance_sheet": {
        "total_assets": {"value": "number or null", "unit": "string"},
        "total_liabilities": {"value": "number or null", "unit": "string"},
        "total_equity": {"value": "number or null", "unit": "string"},
        "cash_and_equivalents": {"value": "number or null", "unit": "string"},
        "total_debt": {"value": "number or null", "unit": "string"},
    },
    "cash_flow": {
        "operating_cash_flow": {"value": "number or null", "unit": "string"},
        "capital_expenditures": {"value": "number or null", "unit": "string"},
        "free_cash_flow": {"value": "number or null", "unit": "string"},
        "dividends_per_share": "number or null",
    },
    "key_ratios": {
        "debt_to_equity": "number or null",
        "current_ratio": "number or null",
        "return_on_equity_pct": "number or null",
        "return_on_assets_pct": "number or null",
    },
    "notes": "string — anything the model wants to flag (e.g. restated figures, fiscal year mismatch)",
}

EXTRACTION_SYSTEM_PROMPT = (
    "You are a financial data extraction engine. Your task is to extract "
    "structured financial metrics from document context and return ONLY valid JSON.\n\n"
    "Rules:\n"
    "- Extract exact numbers from the context. Do NOT estimate or calculate values not stated.\n"
    "- Use null for any metric not found in the context.\n"
    "- For monetary values, provide the number and its unit separately "
    "(e.g. value: 383.3, unit: 'B' for $383.3 billion).\n"
    "- Percentages should be plain numbers (e.g. 25.1 for 25.1%).\n"
    "- If EBITDA is not explicitly stated but D&A and operating income are, "
    "you may compute EBITDA = operating income + D&A and note it.\n"
    "- Put any caveats in the 'notes' field.\n"
    "- Return ONLY the JSON object, no markdown fences, no commentary.\n"
)

EXTRACTION_USER_TEMPLATE = (
    "Extract financial metrics for {company} ({year}) from the context below.\n\n"
    "Return JSON matching this schema:\n"
    "{schema}\n\n"
    "Context:\n"
    "---\n"
    "{context}\n"
    "---\n"
)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _build_context_for_extraction(chunks) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        meta = []
        if chunk.metadata.get("section"):
            meta.append(chunk.metadata["section"])
        tag = f" ({', '.join(meta)})" if meta else ""
        parts.append(f"[Chunk {i}{tag}]\n{chunk.text}")
    return "\n\n".join(parts)


def _safe_parse_json(raw: str) -> dict | None:
    """Strip markdown fences and parse the LLM's JSON output."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse extraction JSON: %s", exc)
        return None


async def extract_metrics(
    company: str,
    year: str,
) -> dict:
    """Retrieve relevant chunks and extract structured financial metrics.

    Returns a dict with:
      - ``metrics``: the structured JSON (or empty dict on failure)
      - ``sources``: list of source chunk summaries
      - ``extraction_quality``: 'full' | 'partial' | 'failed'
    """
    # Targeted retrieval: bias toward financial-data-heavy chunks
    financial_queries = [
        f"{company} {year} revenue net income operating income",
        f"{company} {year} balance sheet total assets liabilities equity",
        f"{company} {year} cash flow EBITDA earnings per share",
    ]

    all_chunks = []
    seen_ids: set[str] = set()
    for q in financial_queries:
        chunks = await retrieve(q, company=company, year=year)
        for c in chunks:
            chunk_id = c.text[:80]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                all_chunks.append(c)

    if not all_chunks:
        return {
            "metrics": {},
            "sources": [],
            "extraction_quality": "failed",
            "notes": f"No documents found for {company} ({year}). Please ingest the relevant financial reports first.",
        }

    context = _build_context_for_extraction(all_chunks)
    schema_str = json.dumps(METRICS_SCHEMA, indent=2)
    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        company=company,
        year=year,
        schema=schema_str,
        context=context,
    )

    settings = get_settings()
    client = _get_client()

    try:
        response = await client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=3000,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        logger.exception("Extraction LLM call failed")
        return {
            "metrics": {},
            "sources": [],
            "extraction_quality": "failed",
            "notes": "LLM call failed during extraction.",
        }

    metrics = _safe_parse_json(raw)
    if metrics is None:
        return {
            "metrics": {},
            "sources": [],
            "extraction_quality": "failed",
            "notes": "Could not parse LLM output as JSON.",
        }

    # Assess extraction quality by counting non-null leaf values
    null_count, total_count = _count_nulls(metrics)
    filled = total_count - null_count
    if filled == 0:
        quality = "failed"
    elif filled < total_count * 0.4:
        quality = "partial"
    else:
        quality = "full"

    sources = [
        {
            "text": c.text[:200] + ("..." if len(c.text) > 200 else ""),
            "score": round(c.score, 4),
            **{k: v for k, v in c.metadata.items() if k != "text"},
        }
        for c in all_chunks
    ]

    logger.info(
        "Extracted metrics for %s/%s: %d/%d fields populated (%s)",
        company, year, filled, total_count, quality,
    )

    return {
        "metrics": metrics,
        "sources": sources,
        "extraction_quality": quality,
    }


def _count_nulls(obj: dict | list | object, _counts: list | None = None) -> tuple[int, int]:
    """Recursively count null vs total leaf values, skipping string-type schema hints."""
    if _counts is None:
        _counts = [0, 0]  # [null_count, total_count]

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("notes", "company", "year", "currency", "unit"):
                continue
            _count_nulls(v, _counts)
    elif isinstance(obj, list):
        for item in obj:
            _count_nulls(item, _counts)
    else:
        if not isinstance(obj, str):
            _counts[1] += 1
            if obj is None:
                _counts[0] += 1

    return _counts[0], _counts[1]
