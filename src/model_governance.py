"""
Model governance metadata for the dashboard.
"""

from __future__ import annotations

import pandas as pd


def governance_status_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Production decision score", "Legacy composite risk score", "Production", "Drives final_decision today."),
            ("Institutional channel score", "Six-channel credit framework", "Parallel / research", "Used for explanation and credit view; not yet production decision engine."),
            ("Spread decomposition", "OAS minus expected loss", "Research", "Uses regime-level PD and 40% recovery assumption."),
            ("Relative value", "Spread percentiles and ratios", "Research", "Public spread data only; not rating-bucket index data."),
            ("Rating buckets", "IG/BBB/HY/distressed proxies", "Proxy", "Requires institutional rating-bucket OAS for production use."),
            ("Refinancing wall", "Maturity-bucket framework", "Placeholder", "Needs issuer/index maturity schedule data."),
            ("Positioning playbook", "Regime-to-action mapping", "Research", "Action language, not executable trade instruction."),
        ],
        columns=["Area", "Method", "Status", "Governance Note"],
    )


def known_limitations_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Data", "Uses public macro/spread/ETF proxies instead of bond-level institutional datasets."),
            ("Expected loss", "Uses regime-level PD assumptions rather than issuer/rating/sector PD models."),
            ("Backtest", "Regime validation is historical and should be treated directionally."),
            ("Composite", "Legacy and institutional composites coexist; replacement requires disagreement analysis."),
            ("Macro releases", "Macro series are revised and released with lags."),
            ("Liquidity", "ETF proxies are imperfect substitutes for TRACE/bond-level liquidity."),
            ("Refinancing", "No true maturity wall unless issuer/index maturity buckets are supplied."),
        ],
        columns=["Topic", "Limitation"],
    )


def required_institutional_data_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Bond-level OAS", "Issuer/rating/sector spread curves and true spread-duration context."),
            ("TRACE liquidity", "Turnover, bid-ask, block liquidity, dealer balance sheet pressure."),
            ("Rating bucket histories", "AAA/AA/A/BBB/BB/B/CCC spreads, returns, migrations."),
            ("Issuer fundamentals", "Leverage, coverage, EBITDA margins, cash flow, maturity schedule."),
            ("Default history", "Rating/sector default rates and recovery rates by cycle."),
            ("CDX HY/IG", "Liquid hedge and market-implied credit beta benchmarks."),
            ("Primary market", "Issuance calendar, concessions, order book quality, refinancing activity."),
        ],
        columns=["Dataset", "Why It Matters"],
    )


def governance_markdown() -> str:
    return "\n".join(
        [
            "# Model Governance",
            "",
            "This project is a research-oriented macro-credit regime framework.",
            "The legacy composite remains the production decision score. The institutional channel score is currently a parallel explanatory framework.",
            "",
            "## Production Boundary",
            "- Production decision score: legacy `composite_risk_score_smooth` and `final_decision`.",
            "- Research/explanatory layer: channel score, spread decomposition, relative value, rating proxies, and refinancing framework.",
            "- Trade language is positioning guidance, not an executable recommendation.",
            "",
            "## Promotion Criteria",
            "- Compare legacy and institutional composites historically.",
            "- Document disagreement periods and failure cases.",
            "- Validate forward returns and spread changes out of sample.",
            "- Confirm assumptions with institutional datasets before using as a production score.",
        ]
    )

