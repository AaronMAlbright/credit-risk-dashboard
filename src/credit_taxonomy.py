"""
Credit risk channel taxonomy.

This module gives the dashboard a finance-native structure: indicators are
grouped by the economic channel they are intended to measure. It is deliberately
lightweight so it can be used by reports, Streamlit views, and tests without
pulling in plotting or data-fetching dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CreditChannel:
    key: str
    name: str
    weight: float
    description: str
    columns: tuple[str, ...]


CHANNELS: tuple[CreditChannel, ...] = (
    CreditChannel(
        key="macro_cycle",
        name="Macro Cycle",
        weight=0.20,
        description="Growth, labor, recession, and broad macro deterioration risk.",
        columns=(
            "macro_risk_score_smooth",
            "recession_probability",
            "macro_nowcast_score",
            "labor_warning_score",
        ),
    ),
    CreditChannel(
        key="rates_liquidity",
        name="Rates And Liquidity",
        weight=0.20,
        description="Real-rate, curve, central-bank liquidity, and funding stress.",
        columns=(
            "liquidity_regime_score_smooth",
            "treasury_stress_score_smooth",
            "funding_stress_score",
            "financial_conditions_score",
            "fed_liquidity_score",
        ),
    ),
    CreditChannel(
        key="credit_market",
        name="Credit Market",
        weight=0.25,
        description="Spread level, spread momentum, volatility, and credit-market stress.",
        columns=(
            "credit_market_risk_score_smooth",
            "spread_volatility_score",
            "credit_momentum_score",
            "fallen_angel_score",
            "distressed_debt_score",
        ),
    ),
    CreditChannel(
        key="fundamentals",
        name="Credit Fundamentals",
        weight=0.15,
        description="Default, leverage, lending standards, and profit-cycle risk.",
        columns=(
            "default_cycle_score",
            "corporate_leverage_score",
            "corporate_profit_cycle_score",
            "sloos_stress_score",
            "cds_implied_pd_score",
        ),
    ),
    CreditChannel(
        key="technicals",
        name="Market Technicals",
        weight=0.10,
        description="Flows, issuance, ETF liquidity, and market-structure pressure.",
        columns=(
            "etf_fund_flow_score",
            "primary_market_score",
            "etf_dislocation_score",
            "loan_market_score",
            "clo_stress_score",
        ),
    ),
    CreditChannel(
        key="cross_asset",
        name="Cross-Asset Confirmation",
        weight=0.10,
        description="Equity, volatility, FX, commodity, bank, and sovereign confirmation.",
        columns=(
            "cross_asset_divergence_score_smooth",
            "market_internals_score_smooth",
            "vol_regime_composite_score",
            "fx_commodity_score_smooth",
            "banking_stress_score_smooth",
        ),
    ),
)


def get_channel(key: str) -> CreditChannel:
    for channel in CHANNELS:
        if channel.key == key:
            return channel
    raise KeyError(f"Unknown credit channel: {key}")


def available_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def compute_channel_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute equal-weighted 0-100 channel scores from available component columns.

    Missing columns do not create scores; the returned DataFrame includes
    coverage metadata so callers can distinguish "low score" from "no data".
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)

    for channel in CHANNELS:
        cols = available_columns(df, channel.columns)
        score_col = f"{channel.key}_channel_score"
        coverage_col = f"{channel.key}_channel_coverage"

        if not cols:
            out[score_col] = pd.NA
            out[coverage_col] = 0.0
            continue

        out[score_col] = df[cols].mean(axis=1).clip(0, 100)
        out[coverage_col] = len(cols) / len(channel.columns)

    weighted_cols = []
    weighted_terms = []
    for channel in CHANNELS:
        score_col = f"{channel.key}_channel_score"
        if score_col in out.columns:
            valid = out[score_col].notna()
            if valid.any():
                weighted_cols.append(score_col)
                weighted_terms.append(out[score_col].fillna(0) * channel.weight)

    if weighted_terms:
        active_weight = sum(
            channel.weight
            for channel in CHANNELS
            if f"{channel.key}_channel_score" in weighted_cols
            and out[f"{channel.key}_channel_score"].notna().any()
        )
        out["institutional_credit_score"] = sum(weighted_terms) / active_weight
        out["institutional_credit_score"] = out["institutional_credit_score"].clip(0, 100)
    else:
        out["institutional_credit_score"] = pd.NA

    return out


def latest_channel_snapshot(df: pd.DataFrame) -> dict:
    scores = compute_channel_scores(df)
    if scores.empty:
        return {"available": False, "channels": [], "composite": None}

    latest = scores.iloc[-1]
    channels = []
    for channel in CHANNELS:
        score_col = f"{channel.key}_channel_score"
        coverage_col = f"{channel.key}_channel_coverage"
        score = latest.get(score_col)
        coverage = latest.get(coverage_col, 0.0)
        channels.append(
            {
                "key": channel.key,
                "name": channel.name,
                "weight": channel.weight,
                "score": None if pd.isna(score) else round(float(score), 1),
                "coverage": round(float(coverage), 2),
                "description": channel.description,
                "available_columns": available_columns(df, channel.columns),
            }
        )

    composite = latest.get("institutional_credit_score")
    return {
        "available": not pd.isna(composite),
        "composite": None if pd.isna(composite) else round(float(composite), 1),
        "channels": channels,
    }

