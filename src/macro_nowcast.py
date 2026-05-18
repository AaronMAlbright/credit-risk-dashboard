"""
Macro Nowcast — real-time GDP growth signal from weekly/monthly indicators.

Uses a simple factor model (weighted average of standardised indicator
z-scores) as a nowcast of US GDP growth.  Inspired by the NY Fed / Atlanta
Fed nowcast approach but implemented as a pure signal composite requiring
only numpy and pandas.

Each available indicator is:
  1. Rolling 252-day z-scored (standardised over a trailing year).
  2. Sign-adjusted so that higher z-score always means better growth.
  3. Averaged (equal weight) into a single nowcast_score [0, 100].

Public API
----------
  NOWCAST_INDICATORS      — list of (col_name, label, sign) triples
  compute_nowcast(df)     -> pd.DataFrame
  run_macro_nowcast(df)   -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicator registry
# ---------------------------------------------------------------------------

# Each entry: (df_column_name, human_label, sign)
#   sign = +1 → higher value = better growth (z-score used as-is)
#   sign = -1 → higher value = worse growth  (z-score negated)
#
# Only ONE of (unemployment, unrate) is used — whichever appears first in df.
# The registry is ordered so that earlier entries take precedence for
# deduplicated groups.

NOWCAST_INDICATORS: list[tuple[str, str, int]] = [
    ("unemployment", "Unemployment",    -1),
    ("unrate",       "Unemployment",    -1),   # alias; skipped if unemployment used
    ("initial_claims", "Initial Claims", -1),
    ("yield_10y",    "10y Yield",        1),
    ("sp500",        "SP500 Mom",        1),   # uses 63-day momentum before z-scoring
    ("ism_pmi",      "ISM PMI",          1),
    ("pmi",          "PMI",              1),   # alias; skipped if ism_pmi used
    ("retail_sales", "Retail Sales",     1),
]

# Groups of mutually exclusive columns (use the first one present)
_EXCLUSIVE_GROUPS: list[list[str]] = [
    ["unemployment", "unrate"],
    ["ism_pmi", "pmi"],
]

_ROLLING_WINDOW = 252    # ~1 trading year for z-score
_MIN_PERIODS = 63        # minimum history before z-score is valid
_SP500_MOM_WINDOW = 63   # 3-month momentum for SP500

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Return rolling z-score: (x - rolling_mean) / rolling_std."""
    roll = series.rolling(window, min_periods=min_periods)
    return (series - roll.mean()) / roll.std()


def _clamp_and_scale(series: pd.Series, lo: float = -3.0, hi: float = 3.0) -> pd.Series:
    """
    Clamp a z-score series to [lo, hi] and linearly map to [0, 100].

    lo maps to 0 (deep contraction), hi maps to 100 (strong expansion).
    """
    clamped = series.clip(lo, hi)
    return (clamped - lo) / (hi - lo) * 100.0


def _nowcast_regime(score: float) -> str:
    """Map a 0-100 nowcast score to a plain-English regime label."""
    if pd.isna(score):
        return "Unknown"
    if score > 70:
        return "Strong Growth"
    if score > 55:
        return "Expansion"
    if score > 45:
        return "Neutral"
    if score > 35:
        return "Slowdown"
    return "Contraction"


def _resolve_active_indicators(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """
    Return the subset of NOWCAST_INDICATORS whose columns are present in df,
    respecting mutual-exclusion groups (first present column wins).

    Returns
    -------
    list of (col_name, label, sign) for active indicators.
    """
    # Build a set of excluded columns (losers within exclusive groups)
    excluded: set[str] = set()
    for group in _EXCLUSIVE_GROUPS:
        found = False
        for col in group:
            if col in df.columns and col not in excluded:
                if found:
                    # A winner was already selected — exclude remaining group members
                    excluded.add(col)
                else:
                    found = True
                    # winner; exclude later duplicates only
            elif col in df.columns and found:
                excluded.add(col)

    active: list[tuple[str, str, int]] = []
    for col, label, sign in NOWCAST_INDICATORS:
        if col in df.columns and col not in excluded:
            active.append((col, label, sign))

    return active


# ---------------------------------------------------------------------------
# Public API — DataFrame enrichment
# ---------------------------------------------------------------------------

def compute_nowcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the macro nowcast score and add columns to a copy of df.

    For each available indicator:
      - SP500: compute 63-day momentum (pct_change(63)) first, then z-score.
      - Unemployment/Initial Claims: negate z-score (higher = worse growth).
      - Others: z-score used with sign from NOWCAST_INDICATORS.

    New columns
    -----------
    nowcast_score           : float 0-100 (100 = strong growth, 0 = deep recession)
    nowcast_regime          : str — "Contraction"/"Slowdown"/"Neutral"/"Expansion"
                              /"Strong Growth"
    nowcast_change_1m       : 21-day change in nowcast_score
    nowcast_recession_prob  : float 0-1 — max(0, (50 - nowcast_score) / 50)

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.  Columns defined in NOWCAST_INDICATORS are
        used when present; all others are silently ignored.

    Returns
    -------
    pd.DataFrame — copy of df with nowcast columns appended.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    active = _resolve_active_indicators(out)

    if not active:
        out["nowcast_score"] = float("nan")
        out["nowcast_regime"] = "Unknown"
        out["nowcast_change_1m"] = float("nan")
        out["nowcast_recession_prob"] = float("nan")
        return out

    z_scores: list[pd.Series] = []

    for col, _label, sign in active:
        raw = out[col].copy()

        # SP500: use momentum (63d pct change) as the raw input before z-scoring
        if col == "sp500":
            raw = raw.pct_change(_SP500_MOM_WINDOW) * 100.0

        z = _rolling_zscore(raw, window=_ROLLING_WINDOW, min_periods=_MIN_PERIODS)

        # Apply directional sign
        z = z * sign

        z_scores.append(z)

    # Equal-weight composite
    z_matrix = pd.concat(z_scores, axis=1)
    z_matrix.columns = [col for col, _, _ in active]
    composite_z = z_matrix.mean(axis=1)   # NaN rows where all indicators NaN

    # Map composite z to 0-100
    out["nowcast_score"] = _clamp_and_scale(composite_z)

    out["nowcast_regime"] = out["nowcast_score"].apply(_nowcast_regime)

    out["nowcast_change_1m"] = out["nowcast_score"].diff(21)

    out["nowcast_recession_prob"] = out["nowcast_score"].apply(
        lambda s: float("nan") if pd.isna(s) else max(0.0, (50.0 - s) / 50.0)
    )

    return out


# ---------------------------------------------------------------------------
# Public API — full analysis
# ---------------------------------------------------------------------------

def run_macro_nowcast(df: pd.DataFrame) -> dict:
    """
    Full macro nowcast analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Main market data frame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        df                  : enriched DataFrame
        current             : dict of latest values
        historical          : last 504 rows (index + nowcast columns)
        indicator_weights   : dict {indicator_name: weight}
        momentum            : str — "Improving"/"Deteriorating"/"Stable"
        available           : bool — True if at least 2 indicators found
        interpretation      : one-sentence plain-English string
    """
    active = _resolve_active_indicators(df)
    n_indicators = len(active)
    available = n_indicators >= 2

    enriched = compute_nowcast(df)

    # ---- Current snapshot ---------------------------------------------------
    def _last_valid(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return float("nan")
        return float(s.iloc[-1])

    nowcast_score = _last_valid(enriched["nowcast_score"])
    nowcast_regime_val = (
        enriched["nowcast_regime"].dropna().iloc[-1]
        if enriched["nowcast_regime"].notna().any()
        else "Unknown"
    )
    nowcast_change_1m = _last_valid(enriched["nowcast_change_1m"])
    nowcast_recession_prob = _last_valid(enriched["nowcast_recession_prob"])

    indicators_used = [label for _col, label, _sign in active]

    current: dict = {
        "nowcast_score": nowcast_score,
        "nowcast_regime": nowcast_regime_val,
        "nowcast_change_1m": nowcast_change_1m,
        "nowcast_recession_prob": nowcast_recession_prob,
        "indicators_used": indicators_used,
    }

    # ---- Historical slice ---------------------------------------------------
    hist_cols = [
        c for c in ("nowcast_score", "nowcast_regime", "nowcast_recession_prob")
        if c in enriched.columns
    ]
    historical = enriched[hist_cols].tail(504).copy()

    # ---- Indicator weights (equal weight among available) -------------------
    weight = 1.0 / n_indicators if n_indicators > 0 else float("nan")
    indicator_weights: dict[str, float] = {
        label: weight for _col, label, _sign in active
    }

    # ---- Momentum -----------------------------------------------------------
    if pd.isna(nowcast_change_1m):
        momentum = "Stable"
    elif nowcast_change_1m > 3.0:
        momentum = "Improving"
    elif nowcast_change_1m < -3.0:
        momentum = "Deteriorating"
    else:
        momentum = "Stable"

    # ---- Interpretation -----------------------------------------------------
    if not pd.isna(nowcast_score):
        rec_prob_pct = nowcast_recession_prob * 100 if not pd.isna(nowcast_recession_prob) else float("nan")
        ind_str = ", ".join(indicators_used) if indicators_used else "no indicators"
        chg_str = ""
        if not pd.isna(nowcast_change_1m):
            chg_str = (
                f", {momentum.lower()} {abs(nowcast_change_1m):.1f} points over the "
                f"past month"
            )
        rec_str = (
            f" (recession probability {rec_prob_pct:.0f}%)"
            if not pd.isna(rec_prob_pct) else ""
        )
        interpretation = (
            f"The macro nowcast stands at {nowcast_score:.0f}/100 "
            f"({nowcast_regime_val}{chg_str}){rec_str}, "
            f"based on {n_indicators} indicator{'s' if n_indicators != 1 else ''}: "
            f"{ind_str}."
        )
    else:
        interpretation = (
            "Insufficient indicator data to compute a macro nowcast score; "
            "at least two of the following columns are required: "
            + ", ".join(col for col, _, _ in NOWCAST_INDICATORS)
            + "."
        )

    return {
        "df": enriched,
        "current": current,
        "historical": historical,
        "indicator_weights": indicator_weights,
        "momentum": momentum,
        "available": available,
        "interpretation": interpretation,
    }
