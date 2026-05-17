"""
Taylor Rule monitor.

The Taylor Rule (1993) suggests the appropriate Fed Funds rate:
  r* = r_neutral + π + 0.5(π - π*) + 0.5(y - y*)

where:
  r_neutral = 2.0% (long-run neutral real rate)
  π         = current inflation (CPI YoY)
  π*        = inflation target (2.0%)
  y - y*    = output gap proxy: -(unemployment - u*) × 2  (Okun's law)
  u*        = NAIRU estimate (4.0%)

Public API
----------
  TAYLOR_PARAMS     — dict of model parameters
  compute_taylor_rule(df) -> pd.DataFrame
    Adds columns: taylor_rate, policy_gap, policy_stance
  run_taylor_analysis(df) -> dict
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

TAYLOR_PARAMS: dict = {
    "r_neutral": 2.0,    # long-run neutral real rate (%)
    "pi_star": 2.0,      # Fed inflation target (%)
    "u_star": 4.0,       # NAIRU estimate (%)
    "okun_multiplier": 2.0,  # Okun's law coefficient mapping unemployment gap to output gap
    "alpha_pi": 0.5,     # weight on inflation gap
    "alpha_y": 0.5,      # weight on output gap
}

# Stance thresholds (policy_gap = actual Fed Funds - Taylor rate)
_STANCE_TOO_TIGHT = 1.0
_STANCE_SLIGHTLY_TIGHT = 0.25
_STANCE_SLIGHTLY_LOOSE = -0.25
_STANCE_TOO_LOOSE = -1.0

# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

_INFLATION_CANDIDATES = ("cpi_yoy", "inflation_yoy", "breakeven_10y", "cpi")
_FED_FUNDS_CANDIDATES = ("dff", "fed_funds_rate", "fed_funds")
_UNEMPLOYMENT_CANDIDATES = ("unemployment", "unrate")


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column name present in *df*, or None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_taylor_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Add Taylor Rule columns to *df* and return a copy.

    Added columns
    -------------
    taylor_rate   : float  — prescribed Fed Funds rate (%) per Taylor (1993)
    policy_gap    : float  — actual Fed Funds minus taylor_rate (% points)
    policy_stance : str    — categorical label for the gap

    If any required column (inflation, fed_funds, unemployment) is absent the
    function returns *df* unchanged so downstream callers can check for column
    presence rather than catching exceptions.

    Parameters
    ----------
    df : pd.DataFrame
        Dashboard-scored DataFrame.  Must contain date-indexed rows.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added columns.
    """
    df = df.copy()

    inflation_col = _resolve_column(df, _INFLATION_CANDIDATES)
    fed_funds_col = _resolve_column(df, _FED_FUNDS_CANDIDATES)
    unemployment_col = _resolve_column(df, _UNEMPLOYMENT_CANDIDATES)

    if inflation_col is None or fed_funds_col is None or unemployment_col is None:
        # Cannot compute; return unchanged so callers can detect missing columns
        return df

    p = TAYLOR_PARAMS
    r_neutral = p["r_neutral"]
    pi_star = p["pi_star"]
    u_star = p["u_star"]
    okun = p["okun_multiplier"]
    alpha_pi = p["alpha_pi"]
    alpha_y = p["alpha_y"]

    _pi_raw = df[inflation_col]
    # If CPI column is in level (e.g., 330 vs 3.3%), convert to YoY %
    if _pi_raw.median() > 30:
        pi = _pi_raw.pct_change(252).fillna(method="bfill") * 100
    else:
        pi = _pi_raw

    u = df[unemployment_col]         # current unemployment rate
    r_actual = df[fed_funds_col]     # actual Fed Funds rate

    # Output gap proxy via Okun's law: y - y* ≈ -(u - u*) * okun
    output_gap = -(u - u_star) * okun

    # Taylor Rule: r* = r_neutral + π + α_π*(π - π*) + α_y*(y - y*)
    taylor_rate = (
        r_neutral
        + pi
        + alpha_pi * (pi - pi_star)
        + alpha_y * output_gap
    )

    policy_gap = r_actual - taylor_rate

    df["taylor_rate"] = taylor_rate
    df["policy_gap"] = policy_gap
    df["policy_stance"] = policy_gap.apply(_classify_stance)

    return df


def _classify_stance(gap: float) -> str:
    """Map a policy gap value (actual - Taylor) to a qualitative stance label.

    Parameters
    ----------
    gap : float
        Fed Funds rate minus the Taylor-prescribed rate, in percentage points.

    Returns
    -------
    str
        One of: "Too Tight", "Slightly Tight", "Neutral",
                "Slightly Loose", "Too Loose".
    """
    if math.isnan(gap):
        return "Unknown"
    if gap > _STANCE_TOO_TIGHT:
        return "Too Tight"
    if gap > _STANCE_SLIGHTLY_TIGHT:
        return "Slightly Tight"
    if gap >= _STANCE_SLIGHTLY_LOOSE:
        return "Neutral"
    if gap >= _STANCE_TOO_LOOSE:
        return "Slightly Loose"
    return "Too Loose"


# ---------------------------------------------------------------------------
# High-level analysis entry point
# ---------------------------------------------------------------------------

def run_taylor_analysis(df: pd.DataFrame) -> dict:
    """Run the full Taylor Rule analysis and return a structured result dict.

    Parameters
    ----------
    df : pd.DataFrame
        Dashboard-scored DataFrame.

    Returns
    -------
    dict
        Keys:
          "df"        — DataFrame with taylor_rate, policy_gap, policy_stance
          "current"   — dict with latest observation values
          "available" — bool; False when required columns are absent
          "params"    — TAYLOR_PARAMS dict

    Example
    -------
    >>> result = run_taylor_analysis(scored_df)
    >>> if result["available"]:
    ...     print(result["current"]["stance"])
    """
    inflation_col = _resolve_column(df, _INFLATION_CANDIDATES)
    fed_funds_col = _resolve_column(df, _FED_FUNDS_CANDIDATES)
    unemployment_col = _resolve_column(df, _UNEMPLOYMENT_CANDIDATES)

    missing = [
        label
        for label, col in (
            ("inflation", inflation_col),
            ("fed_funds", fed_funds_col),
            ("unemployment", unemployment_col),
        )
        if col is None
    ]

    if missing:
        return {
            "df": df,
            "current": {},
            "available": False,
            "params": TAYLOR_PARAMS,
            "missing_columns": missing,
        }

    enriched = compute_taylor_rule(df)

    # Use the last non-NaN row for the "current" snapshot
    valid = enriched.dropna(subset=["taylor_rate", "policy_gap"])
    if valid.empty:
        return {
            "df": enriched,
            "current": {},
            "available": False,
            "params": TAYLOR_PARAMS,
            "missing_columns": [],
        }

    last = valid.iloc[-1]

    current: dict = {
        "fed_funds": float(last[fed_funds_col]),
        "taylor_rate": float(last["taylor_rate"]),
        "policy_gap": float(last["policy_gap"]),
        "stance": str(last["policy_stance"]),
        "inflation": float(last[inflation_col]),
        "unemployment": float(last[unemployment_col]),
    }

    return {
        "df": enriched,
        "current": current,
        "available": True,
        "params": TAYLOR_PARAMS,
        "missing_columns": [],
    }
