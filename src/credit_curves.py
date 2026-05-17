"""
Credit quality curve analysis.

Visualises the credit 'quality curve': the spread premium investors demand
as credit quality decreases from IG → BBB → HY. A steep quality curve means
the market demands high compensation for taking credit risk (risk-off signal).
A flat or inverted quality curve often precedes tightening (risk-on entry point).

Key concept taught: credit spread = default risk premium + liquidity premium.
The quality curve shape reveals which premium is being priced.

When the quality curve steepens sharply (HY–IG premium widens), it typically
signals either:
  1. Rising default fears concentrated in lower-quality issuers (HY specific)
  2. Generalised risk-off: investors fleeing to higher-quality credit
  3. Liquidity stress in HY markets (HY is structurally less liquid than IG)

When the quality curve flattens:
  1. Investors are reaching for yield (risk appetite improving)
  2. Default outlook is benign — no differentiation by quality needed
  3. Potential entry point for long credit / HY overweight

Public API
----------
  compute_quality_curve(df) → pd.DataFrame
  run_quality_curve_analysis(df) → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column name resolution helpers
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


_IG_CANDIDATES  = ["ig_spread", "bamlc0a0cm", "ig_oas"]
_BBB_CANDIDATES = ["bbb_spread", "bamlc0a4cbbb", "bbb_oas"]
_HY_CANDIDATES  = ["hy_spread"]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_quality_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the credit quality curve metrics for every row in *df*.

    The 'quality curve' is the term structure of credit spreads across rating
    buckets. Analogous to the Treasury yield curve but for credit quality rather
    than maturity: it shows the spread compensation investors demand as they move
    down the credit quality ladder.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum a ``hy_spread`` column (in percentage points,
        e.g. 4.5 = 450 bps). IG and BBB columns are optional; premiums that
        require a missing column will be filled with NaN.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the following columns appended:

        bbb_ig_premium : float
            BBB OAS minus IG OAS — the premium for stepping from investment-
            grade to the lowest IG tier (BBB). Historically ~30–80 bps.

        hy_ig_premium : float
            HY OAS minus IG OAS — the full quality curve steepness. A reading
            above its historical norm signals risk-off pricing or stress.

        hy_bbb_premium : float
            HY OAS minus BBB OAS — the pure non-investment-grade premium,
            i.e. what the market charges for crossing the IG/HY boundary.

        curve_slope_zscore : float
            Rolling 252-day z-score of ``hy_ig_premium``. Positive values mean
            the quality curve is historically steep (risk-off); negative values
            mean it is historically flat (risk-on / reach-for-yield regime).
    """
    out = df.copy()

    ig_col  = _resolve_column(df, _IG_CANDIDATES)
    bbb_col = _resolve_column(df, _BBB_CANDIDATES)
    hy_col  = _resolve_column(df, _HY_CANDIDATES)

    ig  = out[ig_col].astype(float)  if ig_col  else pd.Series(np.nan, index=out.index)
    bbb = out[bbb_col].astype(float) if bbb_col else pd.Series(np.nan, index=out.index)
    hy  = out[hy_col].astype(float)  if hy_col  else pd.Series(np.nan, index=out.index)

    out["bbb_ig_premium"] = bbb - ig
    out["hy_ig_premium"]  = hy  - ig
    out["hy_bbb_premium"] = hy  - bbb

    # Rolling 252-day z-score of hy_ig_premium (quality curve steepness)
    # Z = (x - rolling_mean) / rolling_std
    # Window=252 captures approximately one calendar year of daily data.
    rolling_mean = out["hy_ig_premium"].rolling(252, min_periods=60).mean()
    rolling_std  = out["hy_ig_premium"].rolling(252, min_periods=60).std(ddof=1)
    out["curve_slope_zscore"] = (out["hy_ig_premium"] - rolling_mean) / rolling_std.replace(0, np.nan)

    return out


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def run_quality_curve_analysis(df: pd.DataFrame) -> dict:
    """
    Run a full credit quality curve analysis and return a structured result.

    Computes quality curve premiums, identifies the current regime, and
    provides historical percentile context for spread levels and premiums.

    Parameters
    ----------
    df : pd.DataFrame
        Historical panel with at minimum a ``hy_spread`` column. The more
        history provided, the more meaningful the percentile context.

    Returns
    -------
    dict with keys:

        df : pd.DataFrame
            Input dataframe enriched with quality curve columns.

        current : dict
            Latest observation summary including spread levels, premiums,
            z-score, and a plain-English interpretation.

        available : bool
            False if the minimum required column (``hy_spread``) is absent.

        percentiles : dict
            Historical percentile of the current level for ig_spread,
            hy_spread, and hy_ig_premium (0–100 scale).
    """
    # Guard: need at least hy_spread for any meaningful analysis
    hy_col = _resolve_column(df, _HY_CANDIDATES)
    if hy_col is None or df.empty:
        return {
            "df": df.copy(),
            "current": {},
            "available": False,
            "percentiles": {},
        }

    enriched = compute_quality_curve(df)

    # Pull the most recent non-null row
    last = enriched.dropna(subset=[hy_col]).iloc[-1] if not enriched.dropna(subset=[hy_col]).empty else None
    if last is None:
        return {
            "df": enriched,
            "current": {},
            "available": False,
            "percentiles": {},
        }

    def _safe(series_or_val):
        """Return float or NaN — never raises."""
        try:
            v = float(series_or_val)
            return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    ig_spread_val  = _safe(last.get("ig_spread",  last.get("bamlc0a0cm",  last.get("ig_oas",  np.nan))))
    bbb_spread_val = _safe(last.get("bbb_spread", last.get("bamlc0a4cbbb", last.get("bbb_oas", np.nan))))
    hy_spread_val  = _safe(last[hy_col])
    bbb_ig_val     = _safe(last.get("bbb_ig_premium", np.nan))
    hy_ig_val      = _safe(last.get("hy_ig_premium",  np.nan))
    hy_bbb_val     = _safe(last.get("hy_bbb_premium", np.nan))
    zscore_val     = _safe(last.get("curve_slope_zscore", np.nan))

    # Interpretation based on z-score thresholds
    if np.isnan(zscore_val):
        interpretation = "Insufficient history for z-score"
    elif zscore_val > 1.5:
        interpretation = "Steep — risk-off pricing (quality curve historically wide)"
    elif zscore_val > 0.5:
        interpretation = "Moderately steep — mild risk-off / caution warranted"
    elif zscore_val > -0.5:
        interpretation = "Normal — quality curve near historical average"
    elif zscore_val > -1.5:
        interpretation = "Moderately flat — mild reach-for-yield / risk-on lean"
    else:
        interpretation = "Flat — risk-on pricing (quality curve historically compressed)"

    # Historical percentiles (rank of current value vs full history)
    def _percentile(col: str, current_val: float) -> float:
        if col not in enriched.columns or np.isnan(current_val):
            return np.nan
        series = enriched[col].dropna()
        if series.empty:
            return np.nan
        return float((series < current_val).mean() * 100)

    ig_col_used  = _resolve_column(df, _IG_CANDIDATES) or "ig_spread"
    bbb_col_used = _resolve_column(df, _BBB_CANDIDATES) or "bbb_spread"

    percentiles = {
        "ig_spread":     _percentile(ig_col_used,       ig_spread_val),
        "hy_spread":     _percentile(hy_col,             hy_spread_val),
        "hy_ig_premium": _percentile("hy_ig_premium",   hy_ig_val),
    }

    current = {
        "ig_spread":          ig_spread_val,
        "bbb_spread":         bbb_spread_val,
        "hy_spread":          hy_spread_val,
        "bbb_ig_premium":     bbb_ig_val,
        "hy_ig_premium":      hy_ig_val,
        "hy_bbb_premium":     hy_bbb_val,
        "curve_slope_zscore": zscore_val,
        "interpretation":     interpretation,
    }

    return {
        "df":          enriched,
        "current":     current,
        "available":   True,
        "percentiles": percentiles,
    }
