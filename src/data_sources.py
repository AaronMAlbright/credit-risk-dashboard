"""
Data-source registry for public market and macro inputs.

The dashboard uses a mix of observed public data, public proxies, and
synthetic fallbacks. Keeping those distinctions in one place makes the app
easier to audit and prevents captions, freshness labels, and documentation
from drifting apart.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    series_id: str
    label: str
    provider: str
    column: str | None
    frequency: str
    quality: str
    transform: str
    limitation: str


SOURCES: dict[str, DataSource] = {
    "DGS10": DataSource(
        "DGS10", "10Y Treasury Yield", "FRED", "yield_10y", "daily",
        "observed", "Forward-filled up to 5 business days.",
        "Treasury market holidays create short gaps.",
    ),
    "DGS2": DataSource(
        "DGS2", "2Y Treasury Yield", "FRED", "yield_2y", "daily",
        "observed", "Forward-filled up to 5 business days.",
        "Treasury market holidays create short gaps.",
    ),
    "UNRATE": DataSource(
        "UNRATE", "Unemployment Rate", "FRED", "unemployment", "monthly",
        "observed", "Snapped to business-day spine and forward-filled up to 35 days.",
        "Monthly release, so current daily value is stale between releases.",
    ),
    "NFCI": DataSource(
        "NFCI", "NFCI (Liquidity)", "FRED", "nfci", "weekly",
        "observed", "Forward-filled up to 7 days.",
        "Weekly financial-conditions index, not a real-time tradeable price.",
    ),
    "BAMLH0A0HYM2": DataSource(
        "BAMLH0A0HYM2", "HY Credit Spread", "FRED/ICE BofA", "hy_spread", "daily",
        "observed_with_proxy_history",
        "ICE HY OAS where available; pre-ICE period can use scaled BAA10Y fallback.",
        "Pre-ICE history is a public proxy and should not be treated as bond-index truth.",
    ),
    "SP500": DataSource(
        "SP500", "S&P 500", "yfinance/FRED", "sp500", "daily",
        "observed", "Uses yfinance ^GSPC first, with FRED fallback.",
        "FRED SP500 history starts later than yfinance.",
    ),
    "VIXCLS": DataSource(
        "VIXCLS", "VIX", "FRED", "vix", "daily",
        "observed", "Forward-filled up to 5 business days.",
        "Holiday gaps are possible.",
    ),
    "T10YIE": DataSource(
        "T10YIE", "10Y Breakeven", "FRED", "breakeven_10y", "daily",
        "observed_with_synthetic_history",
        "Observed from 2003 onward; earlier rows can use a constant long-run assumption.",
        "Pre-2003 values are synthetic and flagged via breakeven_imputed.",
    ),
    "BAMLC0A0CM": DataSource(
        "BAMLC0A0CM", "IG OAS", "FRED/ICE BofA", "ig_spread", "daily",
        "observed_limited_history", "Forward-filled up to 5 business days.",
        "Public FRED availability is limited in this dataset.",
    ),
    "BAMLC0A4CBBB": DataSource(
        "BAMLC0A4CBBB", "BBB OAS", "FRED/ICE BofA", "bbb_spread", "daily",
        "observed_limited_history", "Forward-filled up to 5 business days.",
        "Public FRED availability is limited in this dataset.",
    ),
    "BAMLH0A0HYM2EY": DataSource(
        "BAMLH0A0HYM2EY", "HY Effective Yield", "FRED/ICE BofA", "hy_yield", "daily",
        "observed_limited_history", "Forward-filled up to 5 business days.",
        "Public FRED availability is limited in this dataset.",
    ),
    "BAMLC0A0CMEY": DataSource(
        "BAMLC0A0CMEY", "IG Effective Yield", "FRED/ICE BofA", "ig_yield", "daily",
        "observed_limited_history", "Forward-filled up to 5 business days.",
        "Public FRED availability is limited in this dataset.",
    ),
    "DRTSCILM": DataSource(
        "DRTSCILM", "SLOOS (C&I Tightening)", "FRED", "sloos_ci", "quarterly",
        "observed", "Forward-filled up to 95 days.",
        "Survey data; not a market price and updates quarterly.",
    ),
    "DRBLACBS": DataSource(
        "DRBLACBS", "Business Loan Delinquency Rate", "FRED", "ci_loan_delinquency",
        "quarterly", "observed", "Forward-filled up to 95 days.",
        "Quarterly bank-reported delinquency rate; lagging realized credit stress.",
    ),
    "CORBLACBS": DataSource(
        "CORBLACBS", "Business Charge-Off Rate", "FRED", "business_chargeoff_rate",
        "quarterly", "observed", "Forward-filled up to 95 days.",
        "Quarterly bank-reported charge-off rate; lagging realized credit losses.",
    ),
}


def fred_source_labels() -> dict[str, str]:
    """Return the series label mapping used by freshness checks."""
    return {series_id: source.label for series_id, source in SOURCES.items()}


def get_source(series_id: str) -> DataSource:
    """Return source metadata for a known series ID."""
    return SOURCES[series_id]


def source_by_column(column: str) -> DataSource | None:
    """Return source metadata for a generated data column, if registered."""
    for source in SOURCES.values():
        if source.column == column:
            return source
    return None


def column_quality(column: str) -> str:
    """Return source quality for a generated data column, or 'derived/unregistered'."""
    source = source_by_column(column)
    if source is None:
        return "derived/unregistered"
    return source.quality


def registered_columns() -> set[str]:
    """Return all non-null generated columns covered by source metadata."""
    return {source.column for source in SOURCES.values() if source.column is not None}


def source_rows() -> list[dict[str, str | None]]:
    """Return source metadata as table-ready dictionaries."""
    return [
        {
            "series_id": source.series_id,
            "label": source.label,
            "provider": source.provider,
            "column": source.column,
            "frequency": source.frequency,
            "quality": source.quality,
            "transform": source.transform,
            "limitation": source.limitation,
        }
        for source in SOURCES.values()
    ]


def format_source_note(*series_ids: str) -> str:
    """Build a concise UI caption for one or more source IDs."""
    parts = []
    for series_id in series_ids:
        source = get_source(series_id)
        parts.append(
            f"{source.label} (`{source.series_id}`): {source.quality}; "
            f"{source.frequency}; {source.transform}"
        )
    return " ".join(parts)
