"""
Signal validation module.

Validates cross-asset scores against historical returns and named stress
episodes. All functions are read-only and return structured dicts/DataFrames.
No lookahead: IS/OOS splits use a hard date cutoff.

Public API
----------
  validate_signals_vs_returns(df, oos_cutoff)  →  dict
  compute_stress_episode_stats(df)             →  pd.DataFrame
  print_validation_summary(report)             →  None
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Scores to validate against forward returns
_SIGNAL_COLS = [
    "composite_risk_score_smooth",
    "rates_stress_score_smooth",
    "enhanced_funding_stress_score_smooth",
    "fx_commodity_score_smooth",
    "banking_stress_score_smooth",
    "macro_risk_score_smooth",
    "credit_market_risk_score_smooth",
    "liquidity_regime_score_smooth",
    "treasury_stress_score_smooth",
    "complacency_score_smooth",
]

# Forward return labels to validate against
_RETURN_COLS = [
    "sp500_forward_30d_return",
    "sp500_forward_60d_return",
]

# Named stress episodes (inclusive date ranges) for episode-level validation
_STRESS_EPISODES: dict[str, tuple[str, str]] = {
    "GFC 2008-09":        ("2008-09-01", "2009-03-31"),
    "Euro Crisis 2011":   ("2011-07-01", "2011-11-30"),
    "China/Oil 2015-16":  ("2015-08-01", "2016-02-29"),
    "Q4 2018 Selloff":    ("2018-10-01", "2018-12-31"),
    "COVID 2020":         ("2020-02-01", "2020-04-30"),
    "2022 Rates Shock":   ("2022-01-01", "2022-10-31"),
    "SVB / Bank Stress":  ("2023-03-01", "2023-05-31"),
}


# ---------------------------------------------------------------------------
# Forward return labels
# ---------------------------------------------------------------------------

def add_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sp500_forward_Nd_return columns if not already present.
    Uses sp500 column; rows near the tail will be NaN (no lookahead).
    """
    df = df.copy()
    if "sp500" not in df.columns:
        return df
    sp = df["sp500"]
    for n in (30, 60):
        col = f"sp500_forward_{n}d_return"
        if col not in df.columns:
            df[col] = sp.shift(-n) / sp - 1
    return df


# ---------------------------------------------------------------------------
# IS / OOS signal validation
# ---------------------------------------------------------------------------

def validate_signals_vs_returns(
    df: pd.DataFrame,
    oos_cutoff: str = "2022-01-01",
) -> dict[str, Any]:
    """
    For each signal in _SIGNAL_COLS, compute Spearman correlation and
    hit rate against forward SP500 returns, split IS / OOS.

    The hit rate is directional: high score (>50) → subsequent return < 0.
    This tests whether elevated risk scores precede negative returns.

    Parameters
    ----------
    df         : DataFrame produced by app.py (must include signal and return cols)
    oos_cutoff : ISO date string — everything on or after this date is OOS

    Returns
    -------
    {
      oos_cutoff: str
      results: {
        signal_col: {
          return_col: {
            is:  {n, spearman_r, hit_rate}
            oos: {n, spearman_r, hit_rate}
          }
        }
      }
    }
    """
    df = add_forward_returns(df)
    cutoff = pd.Timestamp(oos_cutoff)
    is_mask  = df.index < cutoff
    oos_mask = df.index >= cutoff

    results: dict[str, Any] = {}

    for sig in _SIGNAL_COLS:
        if sig not in df.columns:
            continue
        results[sig] = {}
        for ret in _RETURN_COLS:
            if ret not in df.columns:
                continue
            row: dict[str, Any] = {}
            for label, mask in (("is", is_mask), ("oos", oos_mask)):
                sub = df.loc[mask, [sig, ret]].dropna()
                n = len(sub)
                if n < 20:
                    row[label] = {"n": n, "spearman_r": None, "hit_rate": None}
                    continue
                # Spearman = Pearson correlation of ranks (no scipy needed)
                r = float(sub[sig].rank().corr(sub[ret].rank()))
                # High score → negative return = correct call
                high_risk = sub[sig] > 50
                hit = float((sub.loc[high_risk, ret] < 0).mean()) if high_risk.any() else None
                row[label] = {
                    "n":          n,
                    "spearman_r": round(r, 3),
                    "hit_rate":   round(hit, 3) if hit is not None else None,
                }
            results[sig][ret] = row

    return {"oos_cutoff": oos_cutoff, "results": results}


# ---------------------------------------------------------------------------
# Stress episode table
# ---------------------------------------------------------------------------

def compute_stress_episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each named stress episode, compute mean and peak level of all
    available signal scores.

    Returns a MultiIndex DataFrame:
      rows    = episodes
      columns = (signal, stat) where stat ∈ {mean, peak}
    """
    df = add_forward_returns(df)
    avail_sigs = [s for s in _SIGNAL_COLS if s in df.columns]
    if not avail_sigs:
        return pd.DataFrame()

    records: list[dict] = []
    for ep_name, (start, end) in _STRESS_EPISODES.items():
        mask = (df.index >= start) & (df.index <= end)
        sub = df.loc[mask, avail_sigs]
        if len(sub) == 0:
            continue
        row: dict[str, Any] = {"episode": ep_name, "start": start, "end": end}
        for sig in avail_sigs:
            vals = sub[sig].dropna()
            if len(vals) == 0:
                row[f"{sig}__mean"] = None
                row[f"{sig}__peak"] = None
            else:
                row[f"{sig}__mean"] = round(float(vals.mean()), 1)
                row[f"{sig}__peak"] = round(float(vals.max()), 1)
        records.append(row)

    if not records:
        return pd.DataFrame()

    flat = pd.DataFrame(records).set_index("episode")
    # Build MultiIndex columns: (signal_short_name, stat)
    col_tuples = []
    for col in flat.columns:
        if col in ("start", "end"):
            col_tuples.append(("meta", col))
        else:
            sig, stat = col.rsplit("__", 1)
            short = sig.replace("_score_smooth", "").replace("_smooth", "")
            col_tuples.append((short, stat))
    flat.columns = pd.MultiIndex.from_tuples(col_tuples)
    return flat


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_validation_summary(report: dict[str, Any]) -> None:
    """Print IS/OOS validation results to stdout."""
    print(f"\n=== SIGNAL VALIDATION (OOS cutoff: {report['oos_cutoff']}) ===")
    for sig, ret_dict in report["results"].items():
        short_sig = sig.replace("_score_smooth", "").replace("_smooth", "")
        print(f"\n  {short_sig}")
        for ret_col, splits in ret_dict.items():
            short_ret = ret_col.replace("sp500_forward_", "").replace("_return", "")
            is_  = splits.get("is",  {})
            oos_ = splits.get("oos", {})
            def _fmt(v):
                return f"{v:+.3f}" if isinstance(v, float) else "  ?"
            def _fmth(v):
                return f"{v:.0%}" if isinstance(v, float) else "  ?"
            print(
                f"    {short_ret:8s}  "
                f"IS  r={_fmt(is_.get('spearman_r')):>7}  hit={_fmth(is_.get('hit_rate')):>5}  n={is_.get('n','?')}  |  "
                f"OOS r={_fmt(oos_.get('spearman_r')):>7}  hit={_fmth(oos_.get('hit_rate')):>5}  n={oos_.get('n','?')}"
            )
