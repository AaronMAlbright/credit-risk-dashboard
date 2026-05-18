"""
Credit spread seasonality — historical calendar-month patterns in HY and IG spreads.

HY spreads exhibit consistent seasonal patterns driven by investor positioning
rhythms: typically tight in January (new-year risk-on), wider in late Q3/Q4
(year-end risk reduction, thin holiday markets). Knowing the seasonal bias
helps calibrate whether a move is idiosyncratic or merely seasonal noise.

Methodology
-----------
1. Resample hy_spread / ig_spread to month-end (last observation per calendar month).
2. Compute month-over-month diff (in basis points).
3. Group by calendar month (1–12) and average across all years.
4. Exclude the current (partial) calendar month from the average.
5. Require at least 3 years of complete monthly observations before reporting results.

Hit rate: fraction of historical years in which the spread TIGHTENED (diff < 0)
          for each calendar month.

Public API
----------
  MONTH_NAMES : list[str]  — ["Jan", ..., "Dec"]
  compute_seasonality(df) -> dict
  get_current_seasonal_context(df) -> dict
  run_seasonality_analysis(df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MONTH_NAMES: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_MIN_YEARS: int = 3          # minimum calendar years of data required
_FAVORABLE_THRESH: float = -5.0   # avg bps change: tighter than this → "Favorable"
_UNFAVORABLE_THRESH: float = 5.0  # avg bps change: wider than this → "Unfavorable"
_TOP_N: int = 3                   # number of best/worst months to report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _month_end_changes(series: pd.Series) -> pd.Series:
    """Resample *series* to month-end and return diff (bps change per month).

    Returns a Series of monthly spread changes (bps) indexed by the
    month-end date. The first element is NaN (no prior month-end to diff from).
    Only complete calendar months are kept: the final incomplete month is dropped
    by comparing the last date in the original series to the last resampled date.
    """
    if series.empty or series.dropna().empty:
        return pd.Series(dtype=float)

    # Resample to month-end (last non-NaN observation per month)
    monthly = series.resample("ME").last()

    # Drop the final entry if the original series does not reach (or closely
    # approach) that month-end — i.e. exclude a still-in-progress month.
    last_original_date = series.dropna().index[-1]
    last_monthly_date = monthly.index[-1]

    # If the last observed date is more than 5 calendar days before the
    # month-end bucket, the month is incomplete — exclude it.
    days_gap = (last_monthly_date - last_original_date).days
    if days_gap > 5:
        monthly = monthly.iloc[:-1]

    # Month-over-month diff (bps)
    changes = monthly.diff()
    return changes.dropna()


def _group_by_month(changes: pd.Series) -> dict[int, list[float]]:
    """Group monthly changes by calendar month (1-indexed).

    Returns a dict mapping month_number (1–12) → list of observed changes.
    """
    groups: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for date, val in changes.items():
        if pd.notna(val):
            groups[date.month].append(float(val))
    return groups


def _avg_and_hit_rate(
    groups: dict[int, list[float]]
) -> tuple[list[float], list[float]]:
    """Return (avg_change_per_month, hit_rate_per_month) lists of length 12.

    avg_change : mean bps change for that calendar month (NaN if no data)
    hit_rate   : fraction of years where the change was negative (tightening)
    """
    avg_changes: list[float] = []
    hit_rates: list[float] = []
    for m in range(1, 13):
        vals = groups[m]
        if not vals:
            avg_changes.append(float("nan"))
            hit_rates.append(float("nan"))
        else:
            avg_changes.append(float(np.mean(vals)))
            hit_rates.append(float(np.sum(np.array(vals) < 0) / len(vals)))
    return avg_changes, hit_rates


def _count_years(changes: pd.Series) -> int:
    """Count the number of distinct calendar years represented in *changes*."""
    if changes.empty:
        return 0
    return int(changes.index.year.nunique())


# ---------------------------------------------------------------------------
# compute_seasonality
# ---------------------------------------------------------------------------

def compute_seasonality(df: pd.DataFrame) -> dict:
    """Compute historical credit spread seasonality by calendar month.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex. The date is the index, not a column.
        Uses 'hy_spread' and 'ig_spread' columns when present.

    Returns
    -------
    dict with keys:
        hy_monthly_avg_change  : list[float] length 12 — avg bps change (Jan–Dec)
        ig_monthly_avg_change  : list[float] length 12
        hy_monthly_hit_rate    : list[float] length 12 — fraction of years HY tightened
        ig_monthly_hit_rate    : list[float] length 12
        best_months_hy         : list[str] — top 3 by avg tightening (most negative)
        worst_months_hy        : list[str] — top 3 by avg widening (most positive)
        n_years                : int — number of calendar years used
        available              : bool
    """
    _empty = {
        "hy_monthly_avg_change": [float("nan")] * 12,
        "ig_monthly_avg_change": [float("nan")] * 12,
        "hy_monthly_hit_rate":   [float("nan")] * 12,
        "ig_monthly_hit_rate":   [float("nan")] * 12,
        "best_months_hy":        [],
        "worst_months_hy":       [],
        "n_years":               0,
        "available":             False,
    }

    if df is None or df.empty:
        return _empty

    has_hy = "hy_spread" in df.columns and df["hy_spread"].notna().any()
    has_ig = "ig_spread" in df.columns and df["ig_spread"].notna().any()

    if not has_hy:
        return _empty

    # ---- HY -----------------------------------------------------------------
    hy_changes = _month_end_changes(df["hy_spread"])
    n_years = _count_years(hy_changes)

    if n_years < _MIN_YEARS:
        return _empty

    hy_groups = _group_by_month(hy_changes)
    hy_avg, hy_hit = _avg_and_hit_rate(hy_groups)

    # ---- IG -----------------------------------------------------------------
    if has_ig:
        ig_changes = _month_end_changes(df["ig_spread"])
        ig_groups = _group_by_month(ig_changes)
        ig_avg, ig_hit = _avg_and_hit_rate(ig_groups)
    else:
        ig_avg = [float("nan")] * 12
        ig_hit = [float("nan")] * 12

    # ---- Best / worst months (HY) ------------------------------------------
    # best = most negative avg change (greatest tightening)
    # worst = most positive avg change (greatest widening)
    indexed = [(i, v) for i, v in enumerate(hy_avg) if not np.isnan(v)]

    best_months_hy: list[str] = []
    worst_months_hy: list[str] = []

    if indexed:
        sorted_asc = sorted(indexed, key=lambda x: x[1])   # ascending → most negative first
        sorted_desc = sorted(indexed, key=lambda x: x[1], reverse=True)

        best_months_hy = [MONTH_NAMES[i] for i, _ in sorted_asc[:_TOP_N]]
        worst_months_hy = [MONTH_NAMES[i] for i, _ in sorted_desc[:_TOP_N]]

    return {
        "hy_monthly_avg_change": hy_avg,
        "ig_monthly_avg_change": ig_avg,
        "hy_monthly_hit_rate":   hy_hit,
        "ig_monthly_hit_rate":   ig_hit,
        "best_months_hy":        best_months_hy,
        "worst_months_hy":       worst_months_hy,
        "n_years":               n_years,
        "available":             True,
    }


# ---------------------------------------------------------------------------
# get_current_seasonal_context
# ---------------------------------------------------------------------------

def get_current_seasonal_context(df: pd.DataFrame) -> dict:
    """Return seasonal context for the current calendar month.

    The current month is determined from the last date in *df*'s index.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex. The date is the index, not a column.

    Returns
    -------
    dict with keys:
        current_month       : str (e.g. "May")
        current_month_idx   : int (0–11)
        hy_seasonal_avg_bps : float — avg historical HY change this month
        ig_seasonal_avg_bps : float — avg historical IG change this month
        hy_hit_rate         : float — fraction of years HY tightened this month
        seasonal_bias       : "Favorable" / "Neutral" / "Unfavorable"
        available           : bool
    """
    _empty_ctx = {
        "current_month":       "",
        "current_month_idx":   -1,
        "hy_seasonal_avg_bps": float("nan"),
        "ig_seasonal_avg_bps": float("nan"),
        "hy_hit_rate":         float("nan"),
        "seasonal_bias":       "Neutral",
        "available":           False,
    }

    if df is None or df.empty:
        return _empty_ctx

    seasonality = compute_seasonality(df)
    if not seasonality["available"]:
        return _empty_ctx

    # Determine current month from the last date in the index
    last_date = df.index[-1]
    # pandas Timestamp or datetime — .month is 1-indexed
    month_1idx: int = int(last_date.month)   # 1–12
    month_0idx: int = month_1idx - 1          # 0–11

    hy_avg_bps = seasonality["hy_monthly_avg_change"][month_0idx]
    ig_avg_bps = seasonality["ig_monthly_avg_change"][month_0idx]
    hy_hit = seasonality["hy_monthly_hit_rate"][month_0idx]

    if np.isnan(hy_avg_bps):
        seasonal_bias = "Neutral"
    elif hy_avg_bps < _FAVORABLE_THRESH:
        seasonal_bias = "Favorable"
    elif hy_avg_bps > _UNFAVORABLE_THRESH:
        seasonal_bias = "Unfavorable"
    else:
        seasonal_bias = "Neutral"

    return {
        "current_month":       MONTH_NAMES[month_0idx],
        "current_month_idx":   month_0idx,
        "hy_seasonal_avg_bps": float(hy_avg_bps),
        "ig_seasonal_avg_bps": float(ig_avg_bps) if not np.isnan(ig_avg_bps) else float("nan"),
        "hy_hit_rate":         float(hy_hit) if not np.isnan(hy_hit) else float("nan"),
        "seasonal_bias":       seasonal_bias,
        "available":           True,
    }


# ---------------------------------------------------------------------------
# run_seasonality_analysis
# ---------------------------------------------------------------------------

def run_seasonality_analysis(df: pd.DataFrame) -> dict:
    """Full seasonality analysis packaged for dashboard consumption.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex. The date is the index, not a column.

    Returns
    -------
    dict with keys:
        seasonality     : dict from compute_seasonality()
        current_context : dict from get_current_seasonal_context()
        available       : bool — True if hy_spread present and >= 3 years of data
        interpretation  : one-sentence string combining current month bias and
                          best/worst months
    """
    if df is None or df.empty:
        return {
            "seasonality":     compute_seasonality(pd.DataFrame()),
            "current_context": get_current_seasonal_context(pd.DataFrame()),
            "available":       False,
            "interpretation":  "No data available.",
        }

    seasonality = compute_seasonality(df)
    current_ctx = get_current_seasonal_context(df)
    available = seasonality["available"]

    if not available:
        has_hy = "hy_spread" in df.columns and df["hy_spread"].notna().any()
        if has_hy:
            n_actual = _count_years(_month_end_changes(df["hy_spread"]))
            interp = (
                f"Insufficient history for seasonality analysis: "
                f"{n_actual} year(s) of data available, {_MIN_YEARS} required."
            )
        else:
            interp = "hy_spread not found in data — seasonality analysis unavailable."
        return {
            "seasonality":     seasonality,
            "current_context": current_ctx,
            "available":       False,
            "interpretation":  interp,
        }

    # Build interpretation sentence
    month = current_ctx.get("current_month", "")
    bias = current_ctx.get("seasonal_bias", "Neutral")
    hy_bps = current_ctx.get("hy_seasonal_avg_bps", float("nan"))
    best = seasonality.get("best_months_hy", [])
    worst = seasonality.get("worst_months_hy", [])
    n_years = seasonality.get("n_years", 0)

    bps_str = (
        f"{hy_bps:+.1f}bps avg HY change historically"
        if not np.isnan(hy_bps)
        else "insufficient data"
    )
    best_str = "/".join(best) if best else "n/a"
    worst_str = "/".join(worst) if worst else "n/a"

    interpretation = (
        f"{month} has a '{bias}' seasonal bias ({bps_str}, based on {n_years} years); "
        f"historically best months for HY are {best_str} and worst are {worst_str}."
    )

    return {
        "seasonality":     seasonality,
        "current_context": current_ctx,
        "available":       available,
        "interpretation":  interpretation,
    }
