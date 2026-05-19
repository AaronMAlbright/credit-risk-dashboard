"""
Compact model specification for the institutional credit framework.
"""

from __future__ import annotations

import pandas as pd

from src.credit_taxonomy import CHANNELS
from src.model_governance import known_limitations_table, required_institutional_data_table


def credit_model_spec_table() -> pd.DataFrame:
    rows = []
    for channel in CHANNELS:
        rows.append(
            {
                "channel": channel.name,
                "weight": channel.weight,
                "status": "Proxy" if channel.key in {"fundamentals", "technicals"} else "Observed",
                "observable_inputs": ", ".join(channel.columns[:3]),
                "purpose": channel.description,
            }
        )
    return pd.DataFrame(rows)


def validation_boundary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Production signal", "Legacy composite risk score and final decision", "Used for live decision support today."),
            ("Research signal", "Six-channel institutional credit framework", "Used for interpretation, memoing, and relative-value framing."),
            ("Proxy channels", "Fundamentals and technicals", "Derived from public market data until institutional datasets are available."),
            ("Production-grade upgrade", "Issuer fundamentals, TRACE liquidity, CDX, rating histories", "Required before replacing the production decision engine."),
        ],
        columns=["Layer", "Scope", "Interpretation"],
    )


def model_spec_markdown() -> str:
    return "\n".join(
        [
            "# Credit Model Spec",
            "",
            "The framework combines observed market channels with transparent proxies.",
            "That keeps the model interview-safe: it states what is observed, what is inferred, and what still needs institutional data.",
            "",
            "## Channel Structure",
            "- Macro Cycle: growth, labor, recession, and broad macro deterioration risk.",
            "- Rates And Liquidity: real-rate, curve, central-bank liquidity, and funding stress.",
            "- Credit Market: spread level, spread momentum, volatility, and credit-market stress.",
            "- Credit Fundamentals: default, leverage, lending standards, and profit-cycle risk.",
            "- Market Technicals: flows, issuance, ETF liquidity, and market-structure pressure.",
            "- Cross-Asset Confirmation: equity, volatility, FX, commodity, bank, and sovereign confirmation.",
            "",
            "## Governance Boundary",
            "- Production decision score: legacy composite risk score and final decision.",
            "- Institutional channel score: explanatory research layer.",
            "- Proxy channels: fundamentals and technicals until better data is wired in.",
            "",
            "## Promotion Criteria",
            "- Validate each channel against forward spread moves and drawdowns.",
            "- Compare the institutional layer to the legacy decision engine over full history.",
            "- Replace proxies only when issuer, liquidity, and flow data are available.",
        ]
    )


def model_spec_limitations() -> pd.DataFrame:
    return known_limitations_table()


def model_spec_institutional_data() -> pd.DataFrame:
    return required_institutional_data_table()
