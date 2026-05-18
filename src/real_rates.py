"""
Real rates analysis — decompose nominal yields into real rates and inflation breakevens.

Real rate = nominal yield − TIPS-derived breakeven inflation.
Positive and rising real rates tighten financial conditions; deeply negative
real rates signal financial repression and tend to support risk assets even
as they distort credit pricing.

Key relationships for credit:
  Real rate < -1%  : Financial repression — credit spreads tend to be
                     compressed, cheap funding inflates risk.
  Real rate 0-1%   : Neutral zone — historical average range.
  Real rate > 2%   : Restrictive — corporate borrowing costs elevated,
                     default risk rises for leveraged borrowers.

Public API
----------
  compute_real_rates(df)        → pd.DataFrame   adds real_rate_* columns
  run_real_rates_analysis(df)   → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_PERIODS = 5   # minimum observations for rolling ops


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the column if it exists, otherwise a NaN series aligned to df.index."""
    if col in df.columns:
        return df[col].copy()
    return pd.Series(float("nan"), index=df.index)


def _real_rate_regime(rate: float) -> str:
    """Map a real rate level (%) to a plain-English regime label."""
    if pd.isna(rate):
        return "Unknown"
    if rate > 2.0:
        return "Very Restrictive"
    if rate > 1.0:
        return "Restrictive"
    if rate > 0.0:
        return "Low Positive"
    if rate > -1.0:
        return "Negative"
    return "Deeply Negative"


def _real_rate_credit_signal(rate: float, change_1m: float) -> float:
    """
    Map a real rate level to a 0-100 credit stress score.

    Base score by level
    -------------------
    < -1%   → 20  (financial repression; loose conditions suppress spreads)
    -1 to 0 → 35
    0 to 1  → 50
    1 to 2  → 70
    > 2%    → 85

    Modifier
    --------
    +10 if change_1m > 0.5 (rising fast → extra tightening stress)
    """
    if pd.isna(rate):
        return float("nan")
    if rate < -1.0:
        base = 20.0
    elif rate < 0.0:
        base = 35.0
    elif rate < 1.0:
        base = 50.0
    elif rate < 2.0:
        base = 70.0
    else:
        base = 85.0

    modifier = 10.0 if (not pd.isna(change_1m) and change_1m > 0.5) else 0.0
    return min(base + modifier, 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_real_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add real rate columns to a copy of ``df``.

    New columns
    -----------
    real_rate_10y           : yield_10y − breakeven_10y (both in %)
    real_rate_2y            : yield_2y − breakeven_2y (if both available)
    real_rate_change_1m     : 21-trading-day change in real_rate_10y
    real_rate_regime        : str label for current real rate regime
    real_rate_credit_signal : 0–100 credit stress score

    Parameters
    ----------
    df : pd.DataFrame
        Expected to have a DatetimeIndex.  Relevant columns (used when present):
        yield_10y, breakeven_10y, yield_2y, breakeven_2y.

    Returns
    -------
    pd.DataFrame — copy of df with real_rate_* columns appended.
    """
    out = df.copy()

    yield_10y = _safe_col(out, "yield_10y")
    breakeven_10y = _safe_col(out, "breakeven_10y")
    yield_2y = _safe_col(out, "yield_2y")
    breakeven_2y = _safe_col(out, "breakeven_2y")

    # ---- 10y real rate -------------------------------------------------------
    if "yield_10y" in out.columns and "breakeven_10y" in out.columns:
        out["real_rate_10y"] = yield_10y - breakeven_10y
    elif "yield_10y" in out.columns:
        # breakeven missing; cannot compute true real rate — leave as NaN
        out["real_rate_10y"] = pd.Series(float("nan"), index=out.index)
    else:
        out["real_rate_10y"] = pd.Series(float("nan"), index=out.index)

    # ---- 2y real rate -------------------------------------------------------
    if "yield_2y" in out.columns and "breakeven_2y" in out.columns:
        out["real_rate_2y"] = yield_2y - breakeven_2y
    else:
        out["real_rate_2y"] = pd.Series(float("nan"), index=out.index)

    # ---- 21-day change in real_rate_10y -------------------------------------
    out["real_rate_change_1m"] = out["real_rate_10y"].diff(21)

    # ---- Regime label (row-by-row) ------------------------------------------
    out["real_rate_regime"] = out["real_rate_10y"].apply(_real_rate_regime)

    # ---- Credit signal (row-by-row) -----------------------------------------
    out["real_rate_credit_signal"] = [
        _real_rate_credit_signal(r, c)
        for r, c in zip(out["real_rate_10y"], out["real_rate_change_1m"])
    ]

    return out


def run_real_rates_analysis(df: pd.DataFrame) -> dict:
    """
    Full real rates analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Main market data frame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        df               : DataFrame enriched by compute_real_rates()
        current          : dict of latest values — real_rate_10y, real_rate_2y,
                           breakeven_10y, yield_10y, real_rate_regime,
                           real_rate_credit_signal, real_rate_change_1m
        historical       : last 504 rows of enriched df restricted to
                           real_rate_10y, breakeven_10y, yield_10y columns
        interpretation   : one-sentence plain-English string
        rising_flag      : bool — True if real_rate_change_1m > 0.5
        available        : bool — True if yield_10y AND breakeven_10y both exist
        missing_columns  : list[str] of expected columns absent from df
    """
    _expected = ["yield_10y", "breakeven_10y", "yield_2y", "breakeven_2y"]
    missing_columns = [c for c in _expected if c not in df.columns]

    available = (
        "yield_10y" in df.columns
        and "breakeven_10y" in df.columns
        and df["yield_10y"].notna().any()
        and df["breakeven_10y"].notna().any()
    )

    enriched = compute_real_rates(df)

    # ---- Current snapshot ---------------------------------------------------
    def _latest_scalar(series: pd.Series):
        """Return the last non-NaN value as a Python float, or None."""
        s = series.dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])

    current: dict = {
        "real_rate_10y": _latest_scalar(enriched["real_rate_10y"]),
        "real_rate_2y": _latest_scalar(enriched["real_rate_2y"]),
        "breakeven_10y": _latest_scalar(_safe_col(enriched, "breakeven_10y")),
        "yield_10y": _latest_scalar(_safe_col(enriched, "yield_10y")),
        "real_rate_regime": enriched["real_rate_regime"].iloc[-1]
            if not enriched.empty else "Unknown",
        "real_rate_credit_signal": _latest_scalar(enriched["real_rate_credit_signal"]),
        "real_rate_change_1m": _latest_scalar(enriched["real_rate_change_1m"]),
    }

    # ---- Rising flag --------------------------------------------------------
    chg = current["real_rate_change_1m"]
    rising_flag: bool = bool(chg is not None and chg > 0.5)

    # ---- Historical slice ---------------------------------------------------
    hist_cols = [c for c in ["real_rate_10y", "breakeven_10y", "yield_10y"]
                 if c in enriched.columns]
    historical = enriched[hist_cols].tail(504).copy()

    # ---- Plain-English interpretation ---------------------------------------
    rr = current["real_rate_10y"]
    regime = current["real_rate_regime"]
    signal = current["real_rate_credit_signal"]

    if rr is not None and signal is not None:
        chg_str = ""
        if chg is not None:
            direction = "rising" if chg > 0 else "falling"
            chg_str = f", {direction} {abs(chg):.2f}pp over the past month"
        interpretation = (
            f"The 10y real rate stands at {rr:.2f}% ({regime}{chg_str}), "
            f"corresponding to a credit stress signal of {signal:.0f}/100 — "
            f"{'financial repression supports spread compression' if rr < -1.0 else 'rising real rates increase refinancing pressure on leveraged borrowers' if rr > 1.5 else 'real rates in a broadly neutral zone for credit conditions'}."
        )
    else:
        interpretation = (
            "Real rate data is unavailable; cannot assess financial conditions "
            "from a TIPS-breakeven decomposition."
        )

    return {
        "df": enriched,
        "current": current,
        "historical": historical,
        "interpretation": interpretation,
        "rising_flag": rising_flag,
        "available": available,
        "missing_columns": missing_columns,
    }
