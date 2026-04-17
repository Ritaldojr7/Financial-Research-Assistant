from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.services.llm import _build_context, _estimate_confidence
from app.services.retriever import RetrievedChunk, retrieve
from app.utils.config import get_settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None

# Every metric the extractor knows about, with common aliases so the LLM
# can map varied document language to a canonical key.
SUPPORTED_METRICS: dict[str, list[str]] = {
    "revenue": ["total revenue", "net revenue", "net sales", "total net revenue"],
    "cost_of_revenue": ["cost of goods sold", "cogs", "cost of sales", "cost of revenue"],
    "gross_profit": ["gross profit", "gross margin amount"],
    "gross_margin_pct": ["gross margin %", "gross margin percentage"],
    "operating_income": ["operating income", "operating profit", "income from operations"],
    "operating_margin_pct": ["operating margin %"],
    "net_income": ["net income", "net profit", "net earnings"],
    "net_margin_pct": ["net margin %", "net profit margin"],
    "ebitda": ["ebitda"],
    "ebit": ["ebit", "earnings before interest and taxes"],
    "eps_basic": ["basic eps", "basic earnings per share"],
    "eps_diluted": ["diluted eps", "diluted earnings per share"],
    "total_assets": ["total assets"],
    "total_liabilities": ["total liabilities"],
    "total_equity": ["total equity", "stockholders equity", "shareholders equity"],
    "total_debt": ["total debt", "long-term debt", "total borrowings"],
    "cash_and_equivalents": ["cash and cash equivalents", "cash"],
    "free_cash_flow": ["free cash flow", "fcf"],
    "operating_cash_flow": ["cash from operations", "operating cash flow", "net cash from operating"],
    "capex": ["capital expenditures", "capex"],
    "dividends_per_share": ["dividends per share", "dps"],
    "shares_outstanding": ["shares outstanding", "weighted average shares"],
    "revenue_growth_pct": ["revenue growth %", "yoy revenue growth"],
    "rd_expense": ["r&d expense", "research and development"],
}

EXTRACTION_SYSTEM_PROMPT = (
    "You are a financial data extraction engine. Your job is to extract exact "
    "financial metrics from document context and return them as structured JSON.\n\n"
    "Rules:\n"
    "- Extract ONLY values explicitly stated in the context. Never estimate or calculate.\n"
    "- For each metric, return: value (numeric, in original units), formatted (human-readable "
    "string like '$383.3B'), period (e.g. 'FY2024', 'Q3 2024'), and source_section if identifiable.\n"
    "- If a metric is not found in the context, set value/formatted/period to null and add a "
    "'note' field explaining it was not found.\n"
    "- Numeric values should be raw numbers (e.g. 383300000000 not '383.3B').\n"
    "- Percentages should be decimals (e.g. 0.251 for 25.1%).\n"
    "- Return valid JSON only. No markdown fences, no commentary.\n"
)

EXTRACTION_USER_TEMPLATE = (
    "Extract the following financial metrics from the provided context.\n\n"
    "Metrics to extract:\n{metrics_list}\n\n"
    "Context:\n---\n{context}\n---\n\n"
    "Return a JSON object where each key is the metric name and the value is an object with:\n"
    '{{"value": <number|null>, "formatted": <string|null>, "period": <string|null>, '
    '"source_section": <string|null>, "note": <string|null>}}\n'
)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _build_metrics_list(requested: list[str] | None) -> list[str]:
    """Resolve requested metric keys, defaulting to all supported metrics."""
    if not requested:
        return list(SUPPORTED_METRICS.keys())
    valid = []
    for m in requested:
        key = m.lower().strip().replace(" ", "_")
        if key in SUPPORTED_METRICS:
            valid.append(key)
        else:
            logger.warning("Unknown metric requested: %s", m)
    return valid or list(SUPPORTED_METRICS.keys())


def _format_metrics_for_prompt(keys: list[str]) -> str:
    lines = []
    for key in keys:
        aliases = SUPPORTED_METRICS.get(key, [])
        alias_str = f" (also known as: {', '.join(aliases)})" if aliases else ""
        lines.append(f"- {key}{alias_str}")
    return "\n".join(lines)


def _parse_extraction(raw: str, keys: list[str]) -> dict[str, dict]:
    """Parse the LLM JSON output, filling missing metrics with null entries."""
    null_entry = {
        "value": None,
        "formatted": None,
        "period": None,
        "source_section": None,
        "note": "Not found in available documents",
    }

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse metrics JSON from LLM: %s", exc)
        return {k: {**null_entry, "note": "Extraction failed — LLM response was not valid JSON"} for k in keys}

    result: dict[str, dict] = {}
    for key in keys:
        entry = parsed.get(key)
        if entry and isinstance(entry, dict) and entry.get("value") is not None:
            result[key] = {
                "value": entry.get("value"),
                "formatted": entry.get("formatted"),
                "period": entry.get("period"),
                "source_section": entry.get("source_section"),
                "note": entry.get("note"),
            }
        else:
            result[key] = {**null_entry}
            if entry and isinstance(entry, dict) and entry.get("note"):
                result[key]["note"] = entry["note"]

    return result


async def extract_metrics(
    *,
    company: str | None = None,
    year: str | None = None,
    metrics: list[str] | None = None,
) -> dict:
    """Retrieve relevant chunks and extract structured financial metrics.

    Returns::

        {
            "company": str,
            "year": str,
            "metrics": { "<metric_key>": { value, formatted, period, source_section, note } },
            "sources": [...],
            "confidence": str,
            "metrics_found": int,
            "metrics_missing": int,
        }
    """
    settings = get_settings()
    client = _get_client()

    metric_keys = _build_metrics_list(metrics)

    search_terms = []
    if company:
        search_terms.append(company)
    search_terms.extend(["financial statements", "revenue", "income", "balance sheet"])
    query = " ".join(search_terms)

    chunks = await retrieve(query, company=company, year=year)

    if not chunks:
        null_entry = {
            "value": None, "formatted": None, "period": None,
            "source_section": None, "note": "No documents found",
        }
        return {
            "company": company or "",
            "year": year or "",
            "metrics": {k: {**null_entry} for k in metric_keys},
            "sources": [],
            "confidence": "low — no documents found",
            "metrics_found": 0,
            "metrics_missing": len(metric_keys),
        }

    context = _build_context(chunks)
    metrics_prompt = _format_metrics_for_prompt(metric_keys)
    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        metrics_list=metrics_prompt, context=context,
    )

    response = await client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )

    raw = response.choices[0].message.content or ""
    extracted = _parse_extraction(raw, metric_keys)

    sources = [
        {
            "text": c.text[:300] + ("..." if len(c.text) > 300 else ""),
            "score": round(c.score, 4),
            **{k: v for k, v in c.metadata.items() if k != "text"},
        }
        for c in chunks
    ]

    found = sum(1 for v in extracted.values() if v.get("value") is not None)
    missing = len(extracted) - found
    confidence = _estimate_confidence(chunks)

    logger.info(
        "Extracted %d/%d metrics for %s/%s (confidence=%s)",
        found, len(extracted), company or "?", year or "?", confidence,
    )

    return {
        "company": company or "",
        "year": year or "",
        "metrics": extracted,
        "sources": sources,
        "confidence": confidence,
        "metrics_found": found,
        "metrics_missing": missing,
    }
