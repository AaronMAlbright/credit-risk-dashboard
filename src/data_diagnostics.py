"""
Data quality diagnostics for the macro-credit pipeline.

Checks run after the full dataset is assembled but before any scoring.
All checks are non-destructive: they return structured dicts rather than
raising exceptions, so the pipeline can log warnings and continue.

Public API
----------
  check_date_continuity(df)   →  dict
  check_duplicates(df)        →  dict
  check_missing_values(df)    →  dict
  check_lookahead_leakage(df) →  dict
  run_diagnostics(df)         →  dict  (all four checks combined)
  print_diagnostics(report)   →  None  (human-readable console output)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# Maximum business-day gap that is acceptable without flagging (e.g. holidays)
_MAX_ACCEPTABLE_GAP_BDAYS = 5

# Columns that should have NaN tails (these are forward-return labels)
_FORWARD_RETURN_COLS = [
    "sp500_forward_30d_return",
    "sp500_forward_60d_return",
    "hy_forward_30d_change",
    "hy_forward_60d_change",
    "sp500_future_drawdown_30d",
    "sp500_future_drawdown_60d",
]

# Required columns for a meaningful lookahead check
_LOOKAHEAD_HORIZON: dict[str, int] = {
    "sp500_forward_30d_return":  30,
    "sp500_forward_60d_return":  60,
    "sp500_future_drawdown_30d": 30,
    "sp500_future_drawdown_60d": 60,
}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_date_continuity(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect gaps larger than _MAX_ACCEPTABLE_GAP_BDAYS business days.

    Returns
    -------
    {
      passed      : bool
      max_gap_days: int    (largest observed gap)
      n_gaps      : int    (number of gaps exceeding threshold)
      gaps        : list of {'from', 'to', 'bdays'} dicts
    }
    """
    dates = pd.DatetimeIndex(
        df.index if df.index.name == "date" or df.index.dtype == "datetime64[ns]"
        else df["date"]
    ).sort_values()

    if len(dates) < 2:
        return {"passed": True, "max_gap_days": 0, "n_gaps": 0, "gaps": []}

    gaps = []
    for i in range(1, len(dates)):
        gap = len(pd.bdate_range(dates[i - 1] + pd.Timedelta(days=1), dates[i]))
        if gap > _MAX_ACCEPTABLE_GAP_BDAYS:
            gaps.append({
                "from": str(dates[i - 1].date()),
                "to":   str(dates[i].date()),
                "bdays": gap,
            })

    max_gap = max((g["bdays"] for g in gaps), default=0)

    return {
        "passed":       len(gaps) == 0,
        "max_gap_days": max_gap,
        "n_gaps":       len(gaps),
        "gaps":         gaps,
    }


def check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect duplicate timestamps in the index.

    Returns
    -------
    {
      passed       : bool
      n_duplicates : int
      duplicate_dates: list of ISO date strings
    }
    """
    idx = df.index if hasattr(df.index, "duplicated") else pd.DatetimeIndex(df["date"])
    dup_mask = idx.duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    dup_dates = [str(pd.Timestamp(d).date()) for d in idx[dup_mask].unique()]

    return {
        "passed":          n_dup == 0,
        "n_duplicates":    n_dup,
        "duplicate_dates": dup_dates,
    }


def check_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Per-column missing value report.

    Returns
    -------
    {
      passed       : bool   (True if no required column is >10% missing)
      summary      : dict[col → {n_missing, pct_missing, first_valid, last_valid}]
      high_missing : list of column names with >10% missing (excluding fwd labels)
    }
    """
    n = len(df)
    summary: dict[str, Any] = {}
    high_missing = []

    for col in df.columns:
        if col in ("date",):
            continue
        miss = int(df[col].isna().sum())
        pct  = miss / n if n > 0 else 0.0
        non_null = df[col].dropna()
        summary[col] = {
            "n_missing":   miss,
            "pct_missing": round(pct * 100, 1),
            "first_valid": str(non_null.index[0].date()) if len(non_null) > 0 else None,
            "last_valid":  str(non_null.index[-1].date()) if len(non_null) > 0 else None,
        }
        if col not in _FORWARD_RETURN_COLS and pct > 0.10:
            high_missing.append(col)

    return {
        "passed":       len(high_missing) == 0,
        "summary":      summary,
        "high_missing": high_missing,
    }


def check_lookahead_leakage(df: pd.DataFrame) -> dict[str, Any]:
    """
    Verify that forward-return columns have NaN at the tail as expected.

    A forward-return column computed as sp500.shift(-N) / sp500 - 1 should have
    NaN in the last N rows. If it does not, the data may contain lookahead.

    Returns
    -------
    {
      passed   : bool
      warnings : list of {column, horizon, tail_nan_count, expected_min_nan}
    }
    """
    warnings = []

    for col, horizon in _LOOKAHEAD_HORIZON.items():
        if col not in df.columns:
            continue
        tail = df[col].tail(horizon * 2)  # check last 2× horizon rows
        nan_count = int(tail.isna().sum())
        expected  = horizon  # at minimum, the last `horizon` rows should be NaN

        if nan_count < expected:
            warnings.append({
                "column":           col,
                "horizon":          horizon,
                "tail_nan_count":   nan_count,
                "expected_min_nan": expected,
            })

    return {
        "passed":   len(warnings) == 0,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Combined diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run all four checks and return a combined report.

    Returns
    -------
    {
      overall_passed    : bool
      date_continuity   : dict
      duplicates        : dict
      missing_values    : dict
      lookahead_leakage : dict
      n_rows            : int
      date_range        : {'start': str, 'end': str}
    }
    """
    dates = pd.DatetimeIndex(
        df.index if df.index.name in ("date", None) else df["date"]
    )

    continuity = check_date_continuity(df)
    duplicates = check_duplicates(df)
    missing    = check_missing_values(df)
    lookahead  = check_lookahead_leakage(df)

    overall = (
        continuity["passed"]
        and duplicates["passed"]
        and missing["passed"]
        and lookahead["passed"]
    )

    return {
        "overall_passed":    overall,
        "date_continuity":   continuity,
        "duplicates":        duplicates,
        "missing_values":    missing,
        "lookahead_leakage": lookahead,
        "n_rows":            len(df),
        "date_range": {
            "start": str(dates.min().date()) if len(dates) > 0 else None,
            "end":   str(dates.max().date()) if len(dates) > 0 else None,
        },
    }


def print_diagnostics(report: dict[str, Any]) -> None:
    """Print a human-readable summary of run_diagnostics() output."""
    ok = report["overall_passed"]
    print(f"\n=== DATA DIAGNOSTICS {'✓ PASSED' if ok else '✗ WARNINGS'} ===")
    print(f"Rows: {report['n_rows']}  |  "
          f"Range: {report['date_range']['start']} → {report['date_range']['end']}")

    cont = report["date_continuity"]
    tag  = "✓" if cont["passed"] else "✗"
    print(f"\n{tag} Date Continuity: max gap = {cont['max_gap_days']} bdays, "
          f"n large gaps = {cont['n_gaps']}")
    for g in cont.get("gaps", []):
        print(f"    Gap: {g['from']} → {g['to']} ({g['bdays']} bdays)")

    dup = report["duplicates"]
    tag = "✓" if dup["passed"] else "✗"
    print(f"{tag} Duplicates: {dup['n_duplicates']} duplicate timestamps")

    miss = report["missing_values"]
    tag  = "✓" if miss["passed"] else "✗"
    print(f"{tag} Missing Values: {len(miss['high_missing'])} columns >10% missing")
    for col in miss.get("high_missing", []):
        info = miss["summary"].get(col, {})
        print(f"    {col}: {info.get('pct_missing', '?')}% missing "
              f"(first valid: {info.get('first_valid', '?')})")

    la  = report["lookahead_leakage"]
    tag = "✓" if la["passed"] else "✗"
    print(f"{tag} Lookahead Check: {len(la['warnings'])} potential leakage warnings")
    for w in la.get("warnings", []):
        print(f"    {w['column']}: tail NaN = {w['tail_nan_count']}, "
              f"expected ≥ {w['expected_min_nan']}")
