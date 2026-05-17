"""
Recession probability model using the yield curve.

The Estrella-Mishkin (1998) probit model maps the 10Y-3M Treasury spread
to 12-month-ahead recession probability:
  P(recession) = Φ(α + β × spread)
  α = -0.6045, β = -0.7374  (Estrella-Mishkin Fed NY estimates)

Also computes the simpler 10Y-2Y spread for context.

Interpretive thresholds (from NY Fed research):
  spread < -0.5%  → P(recession) > 70% within 12 months
  spread < 0%     → P(recession) > 50%
  spread > 1%     → P(recession) < 25%

Public API
----------
  ESTRELLA_ALPHA, ESTRELLA_BETA   — model coefficients
  compute_recession_prob(df) -> pd.DataFrame
  run_recession_analysis(df) -> dict
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model coefficients  (Estrella & Mishkin 1998, NY Fed)
# ---------------------------------------------------------------------------

ESTRELLA_ALPHA: float = -0.6045
ESTRELLA_BETA: float = -0.7374

# Recession-probability signal thresholds
_THRESH_HIGH = 0.50
_THRESH_ELEVATED = 0.30
_THRESH_MODERATE = 0.15

# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

_YIELD_10Y_CANDIDATES = ("yield_10y", "dgs10", "value_10y")
_YIELD_3M_CANDIDATES = ("yield_3m", "dgs3mo", "tbill_3m")
_YIELD_2Y_CANDIDATES = ("yield_2y", "dgs2", "value_2y")


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column present in *df*, or None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


# ---------------------------------------------------------------------------
# Normal CDF (no scipy)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF evaluated at *x*.

    Uses the identity  Φ(x) = (1 + erf(x / √2)) / 2  from the math stdlib.
    Handles NaN inputs by returning NaN.
    """
    if math.isnan(x):
        return float("nan")
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_cdf_series(s: pd.Series) -> pd.Series:
    """Vectorised normal CDF applied element-wise to a pandas Series."""
    return s.apply(_norm_cdf)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_recession_prob(df: pd.DataFrame) -> pd.DataFrame:
    """Add yield-curve and recession-probability columns to *df*.

    Added columns
    -------------
    spread_10y3m        : float — 10Y minus 3M Treasury yield (%)
    spread_10y2y        : float — 10Y minus 2Y Treasury yield (%; contextual)
    recession_prob_12m  : float — Estrella-Mishkin 12-month-ahead probability (0-1)
    recession_signal    : str   — categorical label

    If the required columns (10Y and 3M yields) are absent the function
    returns *df* unchanged.  The 2Y yield is optional; `spread_10y2y` will
    contain NaN when it is unavailable.

    Parameters
    ----------
    df : pd.DataFrame
        Dashboard-scored DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with added columns.
    """
    df = df.copy()

    col_10y = _resolve_column(df, _YIELD_10Y_CANDIDATES)
    col_3m = _resolve_column(df, _YIELD_3M_CANDIDATES)
    col_2y = _resolve_column(df, _YIELD_2Y_CANDIDATES)

    if col_10y is None or col_3m is None:
        # Primary inputs missing; return unchanged so callers can detect
        return df

    y10 = df[col_10y]
    y3m = df[col_3m]

    # Main spread for the probit model
    spread_10y3m = y10 - y3m
    df["spread_10y3m"] = spread_10y3m

    # Contextual spread (optional)
    if col_2y is not None:
        df["spread_10y2y"] = y10 - df[col_2y]
    else:
        df["spread_10y2y"] = np.nan

    # Estrella-Mishkin probit: P = Φ(α + β * spread)
    probit_index = ESTRELLA_ALPHA + ESTRELLA_BETA * spread_10y3m
    df["recession_prob_12m"] = _norm_cdf_series(probit_index)

    df["recession_signal"] = df["recession_prob_12m"].apply(_classify_signal)

    return df


def _classify_signal(prob: float) -> str:
    """Map a recession probability (0-1) to a qualitative signal label.

    Parameters
    ----------
    prob : float
        Recession probability between 0 and 1.

    Returns
    -------
    str
        One of: "High Risk", "Elevated", "Moderate", "Low".
    """
    if math.isnan(prob):
        return "Unknown"
    if prob > _THRESH_HIGH:
        return "High Risk"
    if prob > _THRESH_ELEVATED:
        return "Elevated"
    if prob > _THRESH_MODERATE:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Inversion episode detector
# ---------------------------------------------------------------------------

def _find_inversion_episodes(series: pd.Series) -> pd.DataFrame:
    """Identify contiguous runs where the yield spread is negative.

    Parameters
    ----------
    series : pd.Series
        Spread series indexed by date (or any ordered index).

    Returns
    -------
    pd.DataFrame
        Columns: start, end, duration_days, min_spread.
        Empty DataFrame if no inversions are found.
    """
    inverted = (series < 0).astype(int)

    # Detect transitions: +1 = inversion start, -1 = inversion end
    transitions = inverted.diff().fillna(inverted)

    starts = series.index[transitions == 1].tolist()
    ends = series.index[transitions == -1].tolist()

    # If series ends while still inverted, close the final episode at last date
    if len(starts) > len(ends):
        ends.append(series.index[-1])

    records = []
    for start, end in zip(starts, ends):
        segment = series.loc[start:end]
        # duration in calendar days requires a DatetimeIndex; fall back to row count
        if isinstance(series.index, pd.DatetimeIndex):
            duration = int((end - start).days)  # type: ignore[operator]
        else:
            duration = int(len(segment))

        records.append(
            {
                "start": start,
                "end": end,
                "duration_days": duration,
                "min_spread": float(segment.min()),
            }
        )

    if not records:
        return pd.DataFrame(columns=["start", "end", "duration_days", "min_spread"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# High-level analysis entry point
# ---------------------------------------------------------------------------

def run_recession_analysis(df: pd.DataFrame) -> dict:
    """Run the full recession probability analysis and return a structured dict.

    Parameters
    ----------
    df : pd.DataFrame
        Dashboard-scored DataFrame.

    Returns
    -------
    dict
        Keys:
          "df"                    — DataFrame with spread and prob columns
          "current"               — dict with latest observation values
          "historical_inversions" — DataFrame of inversion episodes
          "available"             — bool; False when required columns absent

    Example
    -------
    >>> result = run_recession_analysis(scored_df)
    >>> if result["available"]:
    ...     print(result["current"]["recession_prob_12m"])
    """
    col_10y = _resolve_column(df, _YIELD_10Y_CANDIDATES)
    col_3m = _resolve_column(df, _YIELD_3M_CANDIDATES)
    col_2y = _resolve_column(df, _YIELD_2Y_CANDIDATES)

    if col_10y is None or col_3m is None:
        return {
            "df": df,
            "current": {},
            "historical_inversions": pd.DataFrame(
                columns=["start", "end", "duration_days", "min_spread"]
            ),
            "available": False,
            "missing_columns": [
                label
                for label, col in (("yield_10y", col_10y), ("yield_3m", col_3m))
                if col is None
            ],
        }

    enriched = compute_recession_prob(df)

    # Drop rows with NaN in the key spread column before deriving stats
    valid = enriched.dropna(subset=["spread_10y3m", "recession_prob_12m"])

    if valid.empty:
        return {
            "df": enriched,
            "current": {},
            "historical_inversions": pd.DataFrame(
                columns=["start", "end", "duration_days", "min_spread"]
            ),
            "available": False,
            "missing_columns": [],
        }

    # --- "current" snapshot (last valid row) --------------------------------
    last = valid.iloc[-1]

    spread_10y2y_val: float | None = (
        float(last["spread_10y2y"])
        if "spread_10y2y" in last.index and not math.isnan(float(last["spread_10y2y"]))
        else None
    )

    # --- Most recent inversion date -----------------------------------------
    inversions_df = _find_inversion_episodes(valid["spread_10y3m"])

    last_inversion_date: str | None = None
    days_since_inversion: int | None = None

    if not inversions_df.empty:
        last_inv_end = inversions_df.iloc[-1]["end"]
        last_inversion_date = str(last_inv_end)

        # Days since the most recent inversion ended
        last_date = valid.index[-1]
        if isinstance(valid.index, pd.DatetimeIndex):
            days_since_inversion = int((last_date - last_inv_end).days)  # type: ignore[operator]
        else:
            # Fall back to row-count distance
            try:
                idx_end = valid.index.get_loc(last_inv_end)
                idx_last = len(valid) - 1
                days_since_inversion = int(idx_last - idx_end)
            except KeyError:
                days_since_inversion = None

    current: dict = {
        "spread_10y3m": float(last["spread_10y3m"]),
        "spread_10y2y": spread_10y2y_val,
        "recession_prob_12m": float(last["recession_prob_12m"]),
        "signal": str(last["recession_signal"]),
        "last_inversion_date": last_inversion_date,
        "days_since_inversion": days_since_inversion,
    }

    return {
        "df": enriched,
        "current": current,
        "historical_inversions": inversions_df,
        "available": True,
        "missing_columns": [],
    }
