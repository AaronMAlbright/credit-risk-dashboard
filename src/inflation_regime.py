"""
Inflation Regime Decomposition.

Decomposes the 10y nominal yield into real rate + breakeven inflation, maps
today onto a four-quadrant inflation framework, and shows historical credit
returns by inflation regime.

Four-quadrant framework (63d changes in real rate and breakeven):
  Rising real  + Rising breakeven  → "Stagflation"    (worst for credit)
  Rising real  + Falling breakeven → "Tightening"     (restrictive, moderately bad)
  Falling real + Rising breakeven  → "Reflation"      (goldilocks, best for credit)
  Falling real + Falling breakeven → "Deflation Risk" (mixed, depth-dependent)

Column resolution (priority order):
  Real rate 10y:   "real_rate_10y" → "yield_10y" - "breakeven_10y" → yield_10y - 2.0
  Breakeven 10y:   "breakeven_10y" → "tips_breakeven" → 2.0 (constant)
  Nominal yield:   "yield_10y"

Public API
----------
  compute_inflation_regime(df)                 -> pd.DataFrame
  get_current_inflation_state(df, enriched)    -> dict
  compute_historical_credit_by_regime(df, enriched) -> dict
  run_inflation_regime(df)                     -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REGIME_WINDOW   = 63    # trading days for trend classification
_MIN_ROWS        = 126   # minimum rows required
_HIST_ROWS       = 504   # ~2 trading years for historical slices
_REGIME_HIST_ROWS = 252  # ~1 trading year for regime_history

_LONG_RUN_BREAKEVEN = 2.0   # fallback constant when breakeven unavailable
_LONG_RUN_REAL_RATE_SPREAD = 2.0  # subtracted from yield_10y when breakeven missing

# Credit outlook by quadrant
_CREDIT_OUTLOOK: dict[str, str] = {
    "Stagflation":    "Bearish",
    "Tightening":     "Cautious",
    "Reflation":      "Bullish",
    "Deflation Risk": "Mixed",
}

# Plain-English HY impact narrative by quadrant (for interpretation)
_HY_IMPACT: dict[str, str] = {
    "Stagflation":    "HY spreads historically widen 30–80bps over 6 months",
    "Tightening":     "HY spreads historically widen 10–30bps over 6 months",
    "Reflation":      "HY spreads historically tighten 20–60bps over 6 months",
    "Deflation Risk": "HY spreads historically widen 0–50bps over 6 months (high variance)",
}

_ALL_QUADRANTS = ("Stagflation", "Tightening", "Reflation", "Deflation Risk")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else NaN Series aligned to df.index."""
    if col in df.columns:
        return df[col].copy()
    return pd.Series(float("nan"), index=df.index, name=col)


def _last_valid(series: pd.Series):
    """Return last non-NaN value as float, or None."""
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def _resolve_real_rate(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Resolve real rate series from available columns.

    Priority:
      1. "real_rate_10y"                    (already computed)
      2. "yield_10y" - "breakeven_10y"
      3. "yield_10y" - 2.0 (long-run breakeven proxy)

    Returns (series, source_label).
    """
    if "real_rate_10y" in df.columns:
        return df["real_rate_10y"].copy(), "real_rate_10y"

    if "yield_10y" in df.columns and "breakeven_10y" in df.columns:
        return (df["yield_10y"] - df["breakeven_10y"]), "yield_10y - breakeven_10y"

    if "yield_10y" in df.columns:
        return (df["yield_10y"] - _LONG_RUN_REAL_RATE_SPREAD), "yield_10y - 2.0 (proxy)"

    return pd.Series(float("nan"), index=df.index), "unavailable"


def _resolve_breakeven(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Resolve inflation breakeven series from available columns.

    Priority:
      1. "breakeven_10y"
      2. "tips_breakeven"
      3. 2.0 constant

    Returns (series, source_label).
    """
    if "breakeven_10y" in df.columns:
        return df["breakeven_10y"].copy(), "breakeven_10y"

    if "tips_breakeven" in df.columns:
        return df["tips_breakeven"].copy(), "tips_breakeven"

    return (
        pd.Series(_LONG_RUN_BREAKEVEN, index=df.index),
        f"{_LONG_RUN_BREAKEVEN} (constant fallback)",
    )


# ---------------------------------------------------------------------------
# Quadrant classification
# ---------------------------------------------------------------------------

def _classify_quadrant(d_real: float, d_breakeven: float) -> str:
    """Map (d_real, d_breakeven) signs to inflation quadrant label."""
    if np.isnan(d_real) or np.isnan(d_breakeven):
        return "Unknown"
    real_rising = d_real >= 0.0
    be_rising   = d_breakeven >= 0.0
    if real_rising and be_rising:
        return "Stagflation"
    if real_rising and not be_rising:
        return "Tightening"
    if not real_rising and be_rising:
        return "Reflation"
    return "Deflation Risk"


# ---------------------------------------------------------------------------
# Main enrichment
# ---------------------------------------------------------------------------

def compute_inflation_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with columns appended:
      infl_real_rate        — resolved real rate series
      infl_breakeven        — resolved breakeven series
      infl_nominal          — nominal 10y yield
      infl_d_real_63d       — 63d change in real rate
      infl_d_breakeven_63d  — 63d change in breakeven
      infl_quadrant         — "Stagflation"/"Tightening"/"Reflation"/"Deflation Risk"
      infl_credit_outlook   — "Bearish"/"Cautious"/"Bullish"/"Mixed"
    """
    out = df.copy()

    real_rate, _  = _resolve_real_rate(df)
    breakeven, _  = _resolve_breakeven(df)

    out["infl_real_rate"]   = real_rate
    out["infl_breakeven"]   = breakeven
    out["infl_nominal"]     = _safe_col(df, "yield_10y")

    out["infl_d_real_63d"]      = out["infl_real_rate"] - out["infl_real_rate"].shift(_REGIME_WINDOW)
    out["infl_d_breakeven_63d"] = out["infl_breakeven"]  - out["infl_breakeven"].shift(_REGIME_WINDOW)

    quadrants = [
        _classify_quadrant(r, b)
        for r, b in zip(out["infl_d_real_63d"], out["infl_d_breakeven_63d"])
    ]
    out["infl_quadrant"]      = quadrants
    out["infl_credit_outlook"] = [
        _CREDIT_OUTLOOK.get(q, "Unknown") for q in quadrants
    ]
    return out


# ---------------------------------------------------------------------------
# Current state
# ---------------------------------------------------------------------------

def get_current_inflation_state(
    df: pd.DataFrame,
    enriched: pd.DataFrame | None = None,
) -> dict:
    """
    Return current state dict.

    {
        available: bool,
        real_rate: float,
        breakeven: float,
        nominal_yield: float,
        d_real_63d: float,
        d_breakeven_63d: float,
        quadrant: str,
        credit_outlook: str,
        real_rate_source: str,
        breakeven_source: str,
        interpretation: str,
    }
    """
    try:
        if "yield_10y" not in df.columns or df["yield_10y"].notna().sum() == 0:
            return {"available": False}

        if enriched is None:
            enriched = compute_inflation_regime(df)

        _, rr_source = _resolve_real_rate(df)
        _, be_source = _resolve_breakeven(df)

        real_rate     = _last_valid(enriched["infl_real_rate"])
        breakeven     = _last_valid(enriched["infl_breakeven"])
        nominal_yield = _last_valid(enriched["infl_nominal"])
        d_real        = _last_valid(enriched["infl_d_real_63d"])
        d_be          = _last_valid(enriched["infl_d_breakeven_63d"])

        quadrant_s = enriched["infl_quadrant"].dropna()
        quadrant   = str(quadrant_s.iloc[-1]) if not quadrant_s.empty else "Unknown"
        outlook    = _CREDIT_OUTLOOK.get(quadrant, "Unknown")

        # Interpretation
        if quadrant in _ALL_QUADRANTS and real_rate is not None and breakeven is not None:
            direction_real = "rising" if (d_real or 0) >= 0 else "falling"
            direction_be   = "rising" if (d_be or 0) >= 0 else "falling"
            hy_note = _HY_IMPACT.get(quadrant, "")
            interp = (
                f"Inflation regime: {quadrant} — real rates {direction_real} "
                f"({d_real:+.2f}pp over 63d), breakeven {direction_be} "
                f"({d_be:+.2f}pp over 63d). "
                f"Real rate {real_rate:.2f}%, breakeven {breakeven:.2f}%, "
                f"nominal yield {nominal_yield:.2f}%. "
                f"Credit outlook: {outlook}. {hy_note}."
            )
        else:
            interp = "Inflation regime classification unavailable — insufficient rate data."

        return {
            "available":        True,
            "real_rate":        real_rate,
            "breakeven":        breakeven,
            "nominal_yield":    nominal_yield,
            "d_real_63d":       d_real,
            "d_breakeven_63d":  d_be,
            "quadrant":         quadrant,
            "credit_outlook":   outlook,
            "real_rate_source": rr_source,
            "breakeven_source": be_source,
            "interpretation":   interp,
        }

    except Exception:
        return {"available": False}


# ---------------------------------------------------------------------------
# Historical credit performance by regime
# ---------------------------------------------------------------------------

def compute_historical_credit_by_regime(
    df: pd.DataFrame,
    enriched: pd.DataFrame,
) -> dict:
    """
    For each inflation quadrant, compute mean HY spread change over the next
    63 trading days, along with observation count and hit rate (fraction of
    periods where spreads widened).

    Returns: {quadrant: {mean_hy_change: float, n_obs: int, hit_rate: float}}
    Returns empty dict if hy_spread is unavailable.
    """
    try:
        if "hy_spread" not in df.columns:
            return {}

        hy = df["hy_spread"].copy()
        forward_change = hy.shift(-63) - hy   # positive = widening

        result: dict[str, dict] = {}
        for q in _ALL_QUADRANTS:
            mask = enriched["infl_quadrant"] == q
            changes = forward_change[mask].dropna()
            if changes.empty:
                result[q] = {"mean_hy_change": float("nan"), "n_obs": 0, "hit_rate": float("nan")}
                continue
            mean_chg = float(changes.mean())
            hit_rate = float((changes > 0).mean())   # fraction where spreads widened
            result[q] = {
                "mean_hy_change": round(mean_chg, 2),
                "n_obs":          int(len(changes)),
                "hit_rate":       round(hit_rate, 3),
            }
        return result

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_inflation_regime(df: pd.DataFrame) -> dict:
    """
    Top-level entry. Requires yield_10y and >= 126 rows.

    Returns
    -------
    {
        available: bool,
        current: dict,          from get_current_inflation_state
        historical: pd.DataFrame,  last 504 rows with infl_ columns
        regime_history: pd.Series, last 252 rows of infl_quadrant
        credit_by_regime: dict,
        quadrant_counts: dict,  {quadrant: days_count} full history
        interpretation: str,
    }
    """
    try:
        if "yield_10y" not in df.columns or len(df) < _MIN_ROWS:
            return {"available": False}

        if df["yield_10y"].notna().sum() < _MIN_ROWS:
            return {"available": False}

        # ---- Enrich --------------------------------------------------------
        enriched = compute_inflation_regime(df)

        # ---- Current state -------------------------------------------------
        current = get_current_inflation_state(df, enriched)
        if not current.get("available", False):
            return {"available": False}

        # ---- Historical slices ---------------------------------------------
        infl_cols = [
            "infl_real_rate", "infl_breakeven", "infl_nominal",
            "infl_d_real_63d", "infl_d_breakeven_63d",
            "infl_quadrant", "infl_credit_outlook",
        ]
        present_cols = [c for c in infl_cols if c in enriched.columns]
        historical     = enriched[present_cols].tail(_HIST_ROWS).copy()
        regime_history = enriched["infl_quadrant"].tail(_REGIME_HIST_ROWS).copy()

        # ---- Credit by regime ----------------------------------------------
        credit_by_regime = compute_historical_credit_by_regime(df, enriched)

        # ---- Quadrant counts (full history) --------------------------------
        quadrant_counts: dict[str, int] = {}
        for q in _ALL_QUADRANTS:
            quadrant_counts[q] = int((enriched["infl_quadrant"] == q).sum())

        # ---- Interpretation ------------------------------------------------
        interp = current.get("interpretation", "")

        if credit_by_regime:
            # Augment with a note about the current quadrant's empirical record
            cq = current.get("quadrant", "")
            stats = credit_by_regime.get(cq)
            if stats and stats.get("n_obs", 0) >= 10:
                chg  = stats["mean_hy_change"]
                nobs = stats["n_obs"]
                hr   = stats["hit_rate"]
                direction = "widen" if chg >= 0 else "tighten"
                interp += (
                    f" Historically in {cq} regimes (n={nobs}), "
                    f"HY spreads {direction} by a mean {abs(chg):.1f}bps over 63d "
                    f"(spread-widening frequency: {hr:.0%})."
                )

        return {
            "available":       True,
            "current":         current,
            "historical":      historical,
            "regime_history":  regime_history,
            "credit_by_regime": credit_by_regime,
            "quadrant_counts": quadrant_counts,
            "interpretation":  interp,
        }

    except Exception:
        return {"available": False}
