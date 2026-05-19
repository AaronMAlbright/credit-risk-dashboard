"""
Credit spread decomposition.

The goal is to connect observed OAS to expected credit loss and residual
spread compensation. This is intentionally simple and transparent: it provides
an interview-defensible framework without pretending public data can fully
separate default, liquidity, and risk premia.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_RECOVERY_RATE = 0.40

REGIME_DEFAULT_PD = {
    "Risk-On": 0.025,
    "Neutral": 0.040,
    "Caution": 0.075,
    "Risk-Off": 0.120,
}


def _as_spread_bps(series: pd.Series) -> pd.Series:
    """
    Normalize spread units to basis points.

    Existing data in this project often stores OAS as percentage points
    (e.g. 3.5 for 350 bps). Synthetic tests sometimes use bps directly.
    """
    s = pd.to_numeric(series, errors="coerce")
    median_abs = s.dropna().abs().median()
    if pd.notna(median_abs) and median_abs < 50:
        return s * 100.0
    return s


def expected_loss_bps(default_probability: float, recovery_rate: float = DEFAULT_RECOVERY_RATE) -> float:
    loss_given_default = 1.0 - recovery_rate
    return default_probability * loss_given_default * 10_000.0


def classify_excess_spread(excess_spread_bps: float) -> str:
    if pd.isna(excess_spread_bps):
        return "Unavailable"
    if excess_spread_bps >= 450:
        return "Very Cheap"
    if excess_spread_bps >= 300:
        return "Cheap"
    if excess_spread_bps >= 175:
        return "Fair"
    if excess_spread_bps >= 75:
        return "Rich"
    return "Very Rich"


def decompose_spreads(
    df: pd.DataFrame,
    spread_col: str = "hy_spread",
    regime_col: str = "final_decision",
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
    default_pd_by_regime: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add expected-loss and excess-spread columns.

    Output columns:
    - spread_oas_bps
    - default_probability
    - recovery_rate
    - expected_loss_bps
    - excess_spread_bps
    - spread_compensation_ratio
    - spread_valuation
    """
    if df.empty or spread_col not in df.columns:
        return pd.DataFrame(index=df.index)

    pd_map = {**REGIME_DEFAULT_PD, **(default_pd_by_regime or {})}
    out = pd.DataFrame(index=df.index)
    out["spread_oas_bps"] = _as_spread_bps(df[spread_col])

    if regime_col in df.columns:
        out["default_probability"] = df[regime_col].map(pd_map).fillna(pd_map["Neutral"])
    else:
        out["default_probability"] = pd_map["Neutral"]

    out["recovery_rate"] = recovery_rate
    out["expected_loss_bps"] = out["default_probability"].apply(
        lambda pd_: expected_loss_bps(float(pd_), recovery_rate)
    )
    out["excess_spread_bps"] = out["spread_oas_bps"] - out["expected_loss_bps"]
    out["spread_compensation_ratio"] = (
        out["spread_oas_bps"] / out["expected_loss_bps"].replace(0, pd.NA)
    )
    out["spread_valuation"] = out["excess_spread_bps"].apply(classify_excess_spread)
    return out


def latest_spread_snapshot(df: pd.DataFrame) -> dict:
    decomp = decompose_spreads(df)
    if decomp.empty:
        return {"available": False, "reason": "hy_spread column unavailable"}

    latest = decomp.dropna(subset=["spread_oas_bps"]).tail(1)
    if latest.empty:
        return {"available": False, "reason": "No valid spread observations"}

    row = latest.iloc[0]
    return {
        "available": True,
        "spread_oas_bps": round(float(row["spread_oas_bps"]), 1),
        "default_probability": round(float(row["default_probability"]), 4),
        "recovery_rate": round(float(row["recovery_rate"]), 2),
        "expected_loss_bps": round(float(row["expected_loss_bps"]), 1),
        "excess_spread_bps": round(float(row["excess_spread_bps"]), 1),
        "spread_compensation_ratio": round(float(row["spread_compensation_ratio"]), 2),
        "spread_valuation": row["spread_valuation"],
    }

