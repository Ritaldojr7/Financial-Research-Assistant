from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.extractor import extract_metrics
from app.utils.rate_limit import limiter
from app.utils.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Extraction"], dependencies=[Depends(require_api_key)])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    company: str = Field(..., min_length=1, description="Company name (must match ingested metadata)")
    year: str = Field(..., min_length=4, max_length=4, description="Fiscal year (e.g. '2024')")


class MonetaryValue(BaseModel):
    value: float | None = None
    unit: str | None = None


class IncomeStatement(BaseModel):
    revenue: MonetaryValue = MonetaryValue()
    cost_of_revenue: MonetaryValue = MonetaryValue()
    gross_profit: MonetaryValue = MonetaryValue()
    gross_margin_pct: float | None = None
    operating_expenses: MonetaryValue = MonetaryValue()
    operating_income: MonetaryValue = MonetaryValue()
    operating_margin_pct: float | None = None
    net_income: MonetaryValue = MonetaryValue()
    net_margin_pct: float | None = None
    ebitda: MonetaryValue = MonetaryValue()
    ebitda_margin_pct: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None
    rd_expenses: MonetaryValue = MonetaryValue()


class BalanceSheet(BaseModel):
    total_assets: MonetaryValue = MonetaryValue()
    total_liabilities: MonetaryValue = MonetaryValue()
    total_equity: MonetaryValue = MonetaryValue()
    cash_and_equivalents: MonetaryValue = MonetaryValue()
    total_debt: MonetaryValue = MonetaryValue()


class CashFlow(BaseModel):
    operating_cash_flow: MonetaryValue = MonetaryValue()
    capital_expenditures: MonetaryValue = MonetaryValue()
    free_cash_flow: MonetaryValue = MonetaryValue()
    dividends_per_share: float | None = None


class KeyRatios(BaseModel):
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    return_on_equity_pct: float | None = None
    return_on_assets_pct: float | None = None


class FinancialMetrics(BaseModel):
    company: str | None = None
    year: str | None = None
    currency: str | None = None
    income_statement: IncomeStatement = IncomeStatement()
    balance_sheet: BalanceSheet = BalanceSheet()
    cash_flow: CashFlow = CashFlow()
    key_ratios: KeyRatios = KeyRatios()
    notes: str | None = None


class ExtractResponse(BaseModel):
    metrics: FinancialMetrics | dict
    sources: list[dict]
    extraction_quality: str = Field(
        ...,
        description="'full' (>= 40% fields populated), 'partial', or 'failed'",
    )


@router.post("/extract", response_model=ExtractResponse)
@limiter.limit("20/minute")
async def extract_financial_metrics(request: Request, body: ExtractRequest):
    """Extract structured financial metrics (revenue, profit, EBITDA, etc.)
    from ingested documents for a given company and fiscal year.

    Returns a typed JSON object following a standardised financial-data schema.
    """
    logger.info("Extraction requested: company=%s year=%s", body.company, body.year)

    try:
        result = await extract_metrics(body.company, body.year)
    except Exception:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail="Extraction failed. Please try again later.")

    raw_metrics = result.get("metrics", {})

    try:
        metrics = FinancialMetrics(**raw_metrics) if raw_metrics else FinancialMetrics()
    except Exception:
        logger.warning("Could not validate metrics into Pydantic model, returning raw dict")
        metrics = raw_metrics

    return ExtractResponse(
        metrics=metrics,
        sources=result.get("sources", []),
        extraction_quality=result.get("extraction_quality", "failed"),
    )
