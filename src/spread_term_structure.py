"""
Credit spread term structure analysis.

The credit term structure maps spread levels across maturities:
  Short end (1-3yr): sensitive to near-term default risk / liquidity
  Medium (3-7yr):    reflects medium-term credit views
  Long end (7-10yr): driven by structural / solvency concerns

Key signals:
  - Inverted credit curve (short > long): acute near-term stress
  - Parallel shift up: broad credit deterioration
  - Steepening: long-term credit concern, short-term still OK
  - Flattening: market expects near-term problems to resolve

This module works with whatever Treasury and credit data is available.
Since maturity-bucketed IG OAS is not in the base dataset, it approximates
the credit curve using:
  - Treasury curve: yield_3m, yield_2y, yield_10y (already in df)
  - Credit overlay: ig_spread (~7yr avg duration) at long end
                    bbb_spread at medium end
                    hy_spread (~4yr avg duration) at short-medium end
  - Imputes approximate 5yr and 3yr credit spreads via interpolation

Public API
----------
  TREASURY_MATURITIES = {"3m": "yield_3m", "2y": "yield_2y", "10y": "yield_10y"}
  CREDIT_MATURITIES   = {"4y_HY": "hy_spread", "7y_IG": "ig_spread", "7y_BBB": "bbb_spread"}
  compute_term_structure(df) → pd.DataFrame   adds ts_ prefixed columns
  get_current_term_structure(df) → dict
  run_term_structure_analysis(df) → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TREASURY_MATURITIES: dict[str, str] = {
    "3m": "yield_3m",
    "2y": "yield_2y",
    "10y": "yield_10y",
}

CREDIT_MATURITIES: dict[str, str] = {
    "4y_HY": "hy_spread",
    "7y_IG": "ig_spread",
    "7y_BBB": "bbb_spread",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_Z_WINDOW = 252  # rolling window for z-score calculations


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the column if it exists, otherwise a NaN series of the same length."""
    if col in df.columns:
        return df[col]
    return pd.Series(float("nan"), index=df.index)


def _rolling_zscore(series: pd.Series, window: int = _Z_WINDOW) -> pd.Series:
    """
    Rolling z-score using a trailing window.
    Returns NaN where fewer than 20 observations are available.
    """
    roll_mean = series.rolling(window, min_periods=20).mean()
    roll_std = series.rolling(window, min_periods=20).std(ddof=1)
    z = (series - roll_mean) / roll_std.replace(0, float("nan"))
    return z


def _ts_credit_regime(hy_ig_zscore: pd.Series) -> pd.Series:
    """
    Classify credit curve regime from the HY-IG slope z-score.

    Wide/Steep  z > +1.0   : quality spread elevated; near-term risk concern
    Normal      -1.0 ≤ z ≤ +1.0
    Tight/Flat  z < -1.0   : unusually narrow quality premium; complacency risk
    """
    conditions = [
        hy_ig_zscore > 1.0,
        hy_ig_zscore < -1.0,
    ]
    choices = ["Wide/Steep", "Tight/Flat"]
    return pd.Series(
        np.select(conditions, choices, default="Normal"),
        index=hy_ig_zscore.index,
    )


def _ts_curve_regime(slope_2s10s: pd.Series) -> pd.Series:
    """
    Treasury curve regime from the 2s10s spread.

    Normal    > +0.50%
    Flat      -0.50% to +0.50%
    Inverted  < -0.50%
    """
    conditions = [
        slope_2s10s > 0.5,
        slope_2s10s < -0.5,
    ]
    choices = ["Normal", "Inverted"]
    return pd.Series(
        np.select(conditions, choices, default="Flat"),
        index=slope_2s10s.index,
    )


def _interpret_ts(row: dict) -> str:
    """
    Generate a plain-English description combining treasury and credit curve shapes.
    """
    treasury_regime = row.get("ts_curve_regime", "Unknown")
    credit_regime = row.get("ts_credit_regime", "Unknown")
    slope_2s10s = row.get("ts_2s10s_zscore", float("nan"))
    slope_hy_ig = row.get("ts_hy_ig_zscore", float("nan"))
    carry_diff = row.get("ts_carry_differential", float("nan"))

    parts: list[str] = []

    # Treasury shape
    if treasury_regime == "Inverted":
        parts.append(
            "Treasury curve is inverted (2s10s < -50bp) — a historically reliable "
            "recession signal, typically leading credit deterioration by 6–18 months."
        )
    elif treasury_regime == "Flat":
        parts.append(
            "Treasury curve is flat, suggesting late-cycle conditions or a transition "
            "toward either normalization or inversion."
        )
    else:
        parts.append(
            "Treasury curve is positively sloped (normal), consistent with a "
            "mid-cycle or early-expansion environment."
        )

    # Credit curve shape
    if credit_regime == "Wide/Steep":
        parts.append(
            "Credit quality spread (HY-IG) is above historical norms — the market is "
            "demanding elevated compensation for near-term default risk."
        )
    elif credit_regime == "Tight/Flat":
        parts.append(
            "Credit quality spread (HY-IG) is unusually compressed — potential "
            "complacency; spreads may be vulnerable to a snap-wider."
        )
    else:
        parts.append(
            "Credit quality premium (HY-IG) is within normal historical range."
        )

    # Carry
    if not pd.isna(carry_diff):
        if carry_diff > 1.0:
            parts.append(
                f"Short-end all-in yield exceeds long-end by ~{carry_diff:.1f}%, "
                "making short-duration credit relatively attractive on a carry basis."
            )
        elif carry_diff < -1.0:
            parts.append(
                f"Long-end all-in yield exceeds short-end by ~{abs(carry_diff):.1f}%, "
                "rewarding duration extension — typical of a steep, supportive environment."
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add term-structure columns (all prefixed ``ts_``) to a copy of ``df``.

    Missing source columns are handled gracefully — the corresponding output
    column(s) will be NaN rather than raising.

    New columns
    -----------
    ts_curve_slope_2s10s    : yield_10y − yield_2y
    ts_curve_slope_3m10y    : yield_10y − yield_3m
    ts_credit_slope_hy_ig   : hy_spread − ig_spread  (quality slope)
    ts_credit_slope_bbb_ig  : bbb_spread − ig_spread
    ts_all_in_short         : yield_3m + hy_spread × 0.5
    ts_all_in_long          : yield_10y + ig_spread
    ts_carry_differential   : ts_all_in_short − ts_all_in_long
    ts_curve_regime         : "Normal" / "Flat" / "Inverted"
    ts_credit_regime        : "Wide/Steep" / "Normal" / "Tight/Flat"
    ts_2s10s_zscore         : rolling 252d z-score of ts_curve_slope_2s10s
    ts_3m10y_zscore         : rolling 252d z-score of ts_curve_slope_3m10y
    ts_hy_ig_zscore         : rolling 252d z-score of ts_credit_slope_hy_ig
    """
    df = df.copy()

    y3m = _safe_col(df, "yield_3m")
    y2y = _safe_col(df, "yield_2y")
    y10y = _safe_col(df, "yield_10y")
    hy = _safe_col(df, "hy_spread")
    ig = _safe_col(df, "ig_spread")
    bbb = _safe_col(df, "bbb_spread")

    # ---- Treasury slopes ----
    df["ts_curve_slope_2s10s"] = y10y - y2y
    df["ts_curve_slope_3m10y"] = y10y - y3m

    # ---- Credit quality slopes ----
    df["ts_credit_slope_hy_ig"] = hy - ig
    df["ts_credit_slope_bbb_ig"] = bbb - ig

    # ---- All-in yield approximations ----
    df["ts_all_in_short"] = y3m + hy * 0.5
    df["ts_all_in_long"] = y10y + ig
    df["ts_carry_differential"] = df["ts_all_in_short"] - df["ts_all_in_long"]

    # ---- Regime labels ----
    df["ts_curve_regime"] = _ts_curve_regime(df["ts_curve_slope_2s10s"])
    df["ts_2s10s_zscore"] = _rolling_zscore(df["ts_curve_slope_2s10s"])
    df["ts_3m10y_zscore"] = _rolling_zscore(df["ts_curve_slope_3m10y"])
    df["ts_hy_ig_zscore"] = _rolling_zscore(df["ts_credit_slope_hy_ig"])
    df["ts_credit_regime"] = _ts_credit_regime(df["ts_hy_ig_zscore"])

    return df


def get_current_term_structure(df: pd.DataFrame) -> dict:
    """
    Return the latest row of all ``ts_`` columns plus a plain-English interpretation.

    If the DataFrame does not yet have ``ts_`` columns, ``compute_term_structure``
    is called internally.
    """
    ts_cols = [c for c in df.columns if c.startswith("ts_")]
    if not ts_cols:
        df = compute_term_structure(df)
        ts_cols = [c for c in df.columns if c.startswith("ts_")]

    latest = df[ts_cols].iloc[-1].to_dict()

    # Scalar-safe conversion (numpy → Python builtins)
    for k, v in latest.items():
        if isinstance(v, (np.floating, np.integer)):
            latest[k] = float(v)

    latest["interpretation"] = _interpret_ts(latest)
    return latest


def run_term_structure_analysis(df: pd.DataFrame) -> dict:
    """
    Full term-structure analysis.

    Returns
    -------
    dict with keys:
        current             : get_current_term_structure() result
        df                  : compute_term_structure() result
        available           : True if at least yield_3m and yield_10y are in df
        history_percentiles : for each ts_ slope column, the current value's
                              historical percentile (0–100) over the full dataset
    """
    available = ("yield_3m" in df.columns) and ("yield_10y" in df.columns)

    enriched = compute_term_structure(df)
    current = get_current_term_structure(enriched)

    # ---- Historical percentiles ----
    slope_cols = [
        "ts_curve_slope_2s10s",
        "ts_curve_slope_3m10y",
        "ts_credit_slope_hy_ig",
        "ts_credit_slope_bbb_ig",
        "ts_carry_differential",
        "ts_2s10s_zscore",
        "ts_3m10y_zscore",
        "ts_hy_ig_zscore",
    ]
    percentiles: dict[str, float] = {}
    for col in slope_cols:
        if col not in enriched.columns:
            percentiles[col] = float("nan")
            continue
        series = enriched[col].dropna()
        if series.empty:
            percentiles[col] = float("nan")
            continue
        current_val = current.get(col, float("nan"))
        if pd.isna(current_val):
            percentiles[col] = float("nan")
            continue
        pct = float((series < current_val).mean() * 100.0)
        percentiles[col] = pct

    return {
        "current": current,
        "df": enriched,
        "available": available,
        "history_percentiles": percentiles,
    }
