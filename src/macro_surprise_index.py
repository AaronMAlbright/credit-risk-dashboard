"""
Macro Surprise Index — consensus beat/miss proxy without actual consensus data.

For each available macro indicator, compute how much the current value deviates
from its rolling 63-day baseline.  Positive z-score = beating recent trend =
positive economic surprise.  Composite = equal-weight average of all available
indicator z-scores, smoothed with a 10-day EMA.

Positive composite surprise  →  economy beating recent trend  →  credit positive
                                (tighter spread bias)
Negative composite surprise  →  economy missing recent trend  →  credit negative
                                (wider spread bias)

Surprise regimes
----------------
  "Strongly Positive"   composite > 1.0
  "Positive"            composite > 0.3
  "Neutral"             -0.3 <= composite <= 0.3
  "Negative"            composite < -0.3
  "Strongly Negative"   composite < -1.0

Credit signals
--------------
  "Spread Tightening Bias"   Strongly Positive
  "Mild Tightening Bias"     Positive
  "No Directional Bias"      Neutral
  "Mild Widening Bias"       Negative
  "Spread Widening Bias"     Strongly Negative

Public API
----------
  MACRO_INDICATORS    — list of (col_name, col_alias, invert) triples

  _compute_indicator_surprise(series, invert, window) -> pd.Series
  compute_surprise_index(df)                          -> pd.DataFrame
  run_macro_surprise_index(df)                        -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Indicator registry
# ---------------------------------------------------------------------------

# Each entry: (col_name, col_alias, invert)
#   invert=True  → higher value is *bad* for credit (negate z-score before averaging)
#   invert=False → higher value is *good* for credit
#
# Only the first matching column per alias pair is used — the list is ordered so
# the canonical name takes precedence over the alias.

MACRO_INDICATORS: list[tuple[str, str, bool]] = [
    ("unemployment",    "unemployment",     True),
    ("unrate",          "unrate",           True),   # alias; skipped if unemployment used
    ("ism_pmi",         "ism_pmi",          False),
    ("pmi",             "pmi",              False),  # alias; skipped if ism_pmi used
    ("retail_sales",    "retail_sales",     False),
    ("industrial_prod", "industrial_prod",  False),
    ("housing_starts",  "housing_starts",   False),
    ("nonfarm_payroll", "nonfarm_payroll",  False),
    ("cpi_yoy",         "cpi_yoy",          True),
    ("core_cpi",        "core_cpi",         True),
    ("gdp_growth",      "gdp_growth",       False),
    ("consumer_conf",   "consumer_conf",    False),
]

# Mutually exclusive pairs — only the first found column is used
_EXCLUSIVE_GROUPS: list[list[str]] = [
    ["unemployment", "unrate"],
    ["ism_pmi", "pmi"],
]

_SURPRISE_WINDOW: int = 63      # rolling baseline window (≈ 3 months)
_EMA_SPAN: int        = 10      # smoothing span for composite EMA
_CLIP_BOUND: float    = 3.0     # clip z-scores to [-3, 3]
_HIST_ROWS: int       = 504     # rows returned in historical / signal_series
_MIN_ROWS: int        = 126     # minimum history required for available=True
_MIN_INDICATORS: int  = 2       # minimum indicator count required


# ---------------------------------------------------------------------------
# Regime / signal mappings
# ---------------------------------------------------------------------------

def _surprise_regime(composite: float) -> str:
    """Map composite surprise value to a regime label."""
    if np.isnan(composite):
        return "Neutral"
    if composite > 1.0:
        return "Strongly Positive"
    if composite > 0.3:
        return "Positive"
    if composite < -1.0:
        return "Strongly Negative"
    if composite < -0.3:
        return "Negative"
    return "Neutral"


_REGIME_TO_CREDIT_SIGNAL: dict[str, str] = {
    "Strongly Positive": "Spread Tightening Bias",
    "Positive":          "Mild Tightening Bias",
    "Neutral":           "No Directional Bias",
    "Negative":          "Mild Widening Bias",
    "Strongly Negative": "Spread Widening Bias",
}


def _credit_signal(regime: str) -> str:
    return _REGIME_TO_CREDIT_SIGNAL.get(regime, "No Directional Bias")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_active_indicators(df: pd.DataFrame) -> list[tuple[str, bool]]:
    """
    Return list of (col_name, invert) for indicators available in df,
    honouring mutual-exclusion groups (first present column wins).

    Returns
    -------
    list of (col_name, invert) tuples, preserving registry order.
    """
    excluded: set[str] = set()
    for group in _EXCLUSIVE_GROUPS:
        winner_found = False
        for col in group:
            if col in df.columns and df[col].notna().any():
                if winner_found:
                    excluded.add(col)
                else:
                    winner_found = True
            elif col in df.columns:
                # Column present but all-NaN — exclude from dedup competition
                excluded.add(col)

    active: list[tuple[str, bool]] = []
    seen_cols: set[str] = set()
    for col_name, _alias, invert in MACRO_INDICATORS:
        if col_name in seen_cols:
            continue
        if col_name in excluded:
            continue
        if col_name in df.columns and df[col_name].notna().any():
            active.append((col_name, invert))
            seen_cols.add(col_name)

    return active


def _safe_float(val) -> float | None:
    """Convert scalar to Python float; return None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# _compute_indicator_surprise
# ---------------------------------------------------------------------------

def _compute_indicator_surprise(
    series: pd.Series,
    invert: bool,
    window: int = _SURPRISE_WINDOW,
) -> pd.Series:
    """Compute z-score surprise for a single indicator series.

    Algorithm
    ---------
    z = (x - rolling_mean(window)) / rolling_std(window)
    If invert: z = -z
    Clip to [-3, 3].

    Parameters
    ----------
    series : pd.Series
        Raw indicator values (daily or lower-frequency forward-filled).
    invert : bool
        True when higher values are *bad* for credit (e.g. unemployment, CPI).
    window : int
        Rolling baseline window in rows (default 63).

    Returns
    -------
    pd.Series of clipped z-scores, same index as *series*.
    """
    min_p = max(10, window // 3)
    roll  = series.rolling(window, min_periods=min_p)
    mean  = roll.mean()
    std   = roll.std(ddof=1).replace(0, np.nan)
    z     = (series - mean) / std
    if invert:
        z = -z
    return z.clip(-_CLIP_BOUND, _CLIP_BOUND)


# ---------------------------------------------------------------------------
# compute_surprise_index
# ---------------------------------------------------------------------------

def compute_surprise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with macro surprise columns appended.

    For each indicator found in df, adds:
      surprise_{col}              — clipped z-score surprise series

    Also adds aggregate columns:
      surprise_composite          — equal-weight mean of per-indicator z-scores
      surprise_composite_smooth   — 10-day EMA of surprise_composite
      surprise_regime             — regime label (str)
      surprise_credit_signal      — credit signal string
      n_indicators_used           — integer count of indicators used (broadcast)

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame — copy of df with surprise columns appended.
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    active = _resolve_active_indicators(out)

    if not active:
        out["surprise_composite"]        = np.nan
        out["surprise_composite_smooth"] = np.nan
        out["surprise_regime"]           = "Neutral"
        out["surprise_credit_signal"]    = "No Directional Bias"
        out["n_indicators_used"]         = 0
        return out

    # ---- per-indicator z-score columns ----------------------------------------
    z_series_list: list[pd.Series] = []
    for col_name, invert in active:
        z = _compute_indicator_surprise(out[col_name], invert=invert)
        out_col = f"surprise_{col_name}"
        out[out_col] = z
        z_series_list.append(z.rename(out_col))

    # ---- composite (equal-weight row-wise mean) --------------------------------
    z_matrix   = pd.concat(z_series_list, axis=1)
    composite  = z_matrix.mean(axis=1, skipna=True)
    # If a row has *no* valid z-scores, mean returns NaN — that is correct.
    out["surprise_composite"] = composite

    # ---- 10-day EMA smoothing -------------------------------------------------
    out["surprise_composite_smooth"] = (
        composite.ewm(span=_EMA_SPAN, min_periods=1, adjust=False).mean()
    )

    # ---- regime and credit signal ---------------------------------------------
    out["surprise_regime"] = out["surprise_composite"].apply(
        lambda v: _surprise_regime(v) if pd.notna(v) else "Neutral"
    )
    out["surprise_credit_signal"] = out["surprise_regime"].apply(_credit_signal)

    # ---- n_indicators_used (scalar broadcast) ---------------------------------
    # Use the count of non-NaN columns per row in the z_matrix
    out["n_indicators_used"] = z_matrix.notna().sum(axis=1).astype(int)

    return out


# ---------------------------------------------------------------------------
# run_macro_surprise_index
# ---------------------------------------------------------------------------

def run_macro_surprise_index(df: pd.DataFrame) -> dict:
    """Top-level entry point for the macro surprise index.

    Minimum requirements
    --------------------
    - At least 2 macro indicator columns must be present and non-empty.
    - At least 126 rows of data.

    Returns
    -------
    dict with keys:
        available                : bool
        current                  : dict — see below
        historical               : pd.DataFrame — last 504 rows with surprise cols
        signal_series            : pd.Series    — surprise_composite_smooth, last 504 rows
        momentum                 : float        — change in composite_smooth over 21 days
        interpretation           : str

    current dict keys
    -----------------
        surprise_composite       : float
        surprise_composite_smooth: float
        surprise_regime          : str
        surprise_credit_signal   : str
        n_indicators             : int
        indicator_scores         : dict {col: z_score}  — per-indicator z at latest row
        indicators_used          : list[str]
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        if len(df) < _MIN_ROWS:
            return {"available": False}

        active = _resolve_active_indicators(df)
        if len(active) < _MIN_INDICATORS:
            return {"available": False}

        # ---- enrich -----------------------------------------------------------
        enriched = compute_surprise_index(df)
        last     = enriched.iloc[-1]

        # ---- per-indicator z-scores at latest row ----------------------------
        indicator_scores: dict[str, float | None] = {}
        indicators_used: list[str] = []
        for col_name, _invert in active:
            z_col = f"surprise_{col_name}"
            if z_col in enriched.columns:
                val = _safe_float(last[z_col])
                indicator_scores[col_name] = val
                indicators_used.append(col_name)

        # ---- current scalars --------------------------------------------------
        composite_now = _safe_float(last.get("surprise_composite"))
        smooth_now    = _safe_float(last.get("surprise_composite_smooth"))

        regime_now = "Neutral"
        if "surprise_regime" in enriched.columns and pd.notna(last["surprise_regime"]):
            regime_now = str(last["surprise_regime"])

        signal_now = _credit_signal(regime_now)

        n_ind_now: int = 0
        if "n_indicators_used" in enriched.columns and pd.notna(last["n_indicators_used"]):
            n_ind_now = int(last["n_indicators_used"])

        current: dict = {
            "surprise_composite":        composite_now,
            "surprise_composite_smooth": smooth_now,
            "surprise_regime":           regime_now,
            "surprise_credit_signal":    signal_now,
            "n_indicators":              n_ind_now,
            "indicator_scores":          indicator_scores,
            "indicators_used":           indicators_used,
        }

        # ---- historical slice (last 504 rows) --------------------------------
        surprise_cols = (
            [f"surprise_{col}" for col, _ in active]
            + [
                "surprise_composite",
                "surprise_composite_smooth",
                "surprise_regime",
                "surprise_credit_signal",
                "n_indicators_used",
            ]
        )
        hist_cols = [c for c in surprise_cols if c in enriched.columns]
        historical = enriched[hist_cols].tail(_HIST_ROWS).copy()

        # ---- signal series (smoothed composite, last 504 rows) ---------------
        if "surprise_composite_smooth" in enriched.columns:
            signal_series = enriched["surprise_composite_smooth"].tail(_HIST_ROWS).copy()
            signal_series.name = "surprise_composite_smooth"
        else:
            signal_series = pd.Series(dtype=float, name="surprise_composite_smooth")

        # ---- momentum (21-day change in smoothed composite) ------------------
        momentum: float = float("nan")
        if "surprise_composite_smooth" in enriched.columns:
            smooth_series = enriched["surprise_composite_smooth"].dropna()
            if len(smooth_series) >= 22:
                momentum = float(smooth_series.iloc[-1] - smooth_series.iloc[-22])

        # ---- interpretation --------------------------------------------------
        n_active = len(active)
        ind_names = ", ".join(indicators_used) if indicators_used else "no indicators"

        parts: list[str] = []

        if composite_now is not None:
            direction_word = (
                "above" if composite_now > 0 else "below"
            )
            parts.append(
                f"The macro surprise index stands at {composite_now:+.2f} "
                f"(smoothed: {smooth_now:+.2f}), "
                f"{direction_word} the recent baseline — '{regime_now}' regime."
            )
        else:
            parts.append(f"Macro surprise regime: {regime_now}.")

        parts.append(f"Credit bias: {signal_now}.")

        if not np.isnan(momentum):
            mom_dir = "improving" if momentum > 0 else "deteriorating"
            parts.append(
                f"Over the past 21 days, the surprise index has been "
                f"{mom_dir} ({momentum:+.2f})."
            )

        parts.append(
            f"Based on {n_active} indicator{'s' if n_active != 1 else ''}: {ind_names}."
        )

        interpretation = " ".join(parts)

        return {
            "available": True,
            "current": current,
            "historical": historical,
            "signal_series": signal_series,
            "momentum": float("nan") if np.isnan(momentum) else momentum,
            "interpretation": interpretation,
        }

    except Exception:
        return {"available": False}
