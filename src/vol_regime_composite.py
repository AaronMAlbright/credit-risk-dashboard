"""
Volatility Regime Composite — synthesize VIX, VIX term structure, VRP, SKEW,
and MOVE into a single vol regime score (0-100).

Answers the question: "Is vol cheap or expensive?" as a unified signal.

The four primary inputs:
  1. VIX level        — z-score vs 252d history → 0-100 (high VIX = high score)
  2. VIX term slope   — VIX3M/VIX slope; inverted = high stress → 0-100
  3. VRP              — VIX minus realized vol; negative VRP = expensive → 0-100
  4. SKEW             — CBOE SKEW index tail risk → 0-100

Optional fifth input:
  5. MOVE             — Bond vol z-score → 0-100

Composite = equal-weight average of available components.

Vol regime labels:
  "Complacent" (score < 25)   : vol cheap, markets not pricing risk
  "Normal"     (25-50)        : vol appropriately priced
  "Elevated"   (50-75)        : vol elevated, markets cautious
  "Stressed"   (score >= 75)  : vol very expensive, fear dominant

Public API
----------
  compute_vol_composite(df)       -> pd.DataFrame
  run_vol_regime_composite(df)    -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZSCORE_WINDOW   = 252
_HIST_ROWS       = 504
_REGIME_ROWS     = 252

# VIX z-score clipping bounds (maps to 0-100)
_VIX_ZSCORE_LOW  = -2.0   # z <= -2 → score 0
_VIX_ZSCORE_HIGH =  3.0   # z >= 3  → score 100

# Term structure slope thresholds (ratio = VIX3M/VIX - 1)
_TS_CONTANGO_FLOOR   =  0.05  # slope >= 0.05  → score 0  (normal contango)
_TS_INVERTED_CEILING = -0.05  # slope <= -0.05 → score 100 (inverted)

# VRP thresholds (VIX - realized_vol)
_VRP_CHEAP_FLOOR     =  5.0   # VRP > 5  → score 0   (vol cheap)
_VRP_EXPENSIVE_CEIL  = -5.0   # VRP < -5 → score 100 (vol expensive)

# SKEW thresholds
_SKEW_LOW  = 110.0   # SKEW <= 110 → score 0
_SKEW_HIGH = 150.0   # SKEW >= 150 → score 100

# Vol regime breakpoints
_COMPLACENT_MAX = 25.0
_NORMAL_MAX     = 50.0
_ELEVATED_MAX   = 75.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score with min_periods = window // 2."""
    roll = series.rolling(window, min_periods=window // 2)
    mean = roll.mean()
    std  = roll.std(ddof=1).replace(0.0, np.nan)
    return (series - mean) / std


def _zscore_to_0_100(
    zscore: pd.Series,
    low: float = _VIX_ZSCORE_LOW,
    high: float = _VIX_ZSCORE_HIGH,
) -> pd.Series:
    """Linear map from [low, high] z-score range to [0, 100], clipped."""
    scaled = (zscore - low) / (high - low) * 100.0
    return scaled.clip(0.0, 100.0)


def _classify_vol_regime(score: float) -> str:
    """Map composite vol score (0-100) to a regime label."""
    if np.isnan(score):
        return "Unknown"
    if score < _COMPLACENT_MAX:
        return "Complacent"
    if score < _NORMAL_MAX:
        return "Normal"
    if score < _ELEVATED_MAX:
        return "Elevated"
    return "Stressed"


def _safe_last(series: pd.Series) -> float:
    """Return last non-NaN value as float, or NaN."""
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[-1])


def _resolve_vix3m_col(df: pd.DataFrame) -> str | None:
    """Return the first matching VIX3M column name, or None."""
    for candidate in ("vix_3m", "vix3m"):
        if candidate in df.columns:
            return candidate
    return None


def _resolve_skew_col(df: pd.DataFrame) -> str | None:
    """Return the first matching SKEW column name, or None."""
    for candidate in ("skew", "cboe_skew"):
        if candidate in df.columns:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Component score functions
# ---------------------------------------------------------------------------

def _compute_realized_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Compute annualized realized vol from sp500 log returns.

    Returns a Series of realized vol in % terms (matching VIX scale).
    Returns an all-NaN Series if sp500 column is absent.
    """
    if "sp500" not in df.columns or df["sp500"].dropna().empty:
        return pd.Series(np.nan, index=df.index, name="realized_vol")

    log_ret = np.log(df["sp500"] / df["sp500"].shift(1))
    rv = (
        log_ret
        .rolling(window, min_periods=max(5, window // 2))
        .std(ddof=1)
    ) * np.sqrt(252) * 100.0

    return rv.rename("realized_vol")


def _vix_score(df: pd.DataFrame) -> pd.Series:
    """VIX z-score vs 252-day history, clipped and scaled to [0, 100].

    High VIX relative to history → high score (vol expensive/stressed).
    Returns all-NaN Series if VIX column absent.
    """
    if "vix" not in df.columns or df["vix"].dropna().empty:
        return pd.Series(np.nan, index=df.index, name="vol_vix_score")

    vix = pd.to_numeric(df["vix"], errors="coerce")
    zs  = _rolling_zscore(vix, _ZSCORE_WINDOW)
    return _zscore_to_0_100(zs, _VIX_ZSCORE_LOW, _VIX_ZSCORE_HIGH).rename("vol_vix_score")


def _term_structure_score(df: pd.DataFrame) -> pd.Series:
    """VIX3M/VIX slope converted to 0-100 stress score.

    slope = VIX3M / VIX - 1
      slope >  0.05 (contango)  → 0   (calm)
      slope = 0.0               → 50  (flat)
      slope < -0.05 (inverted)  → 100 (stressed)

    Linear interpolation between breakpoints.
    Returns all-NaN Series if VIX or VIX3M column is absent.
    """
    vix3m_col = _resolve_vix3m_col(df)
    if "vix" not in df.columns or vix3m_col is None:
        return pd.Series(np.nan, index=df.index, name="vol_term_score")

    vix   = pd.to_numeric(df["vix"],      errors="coerce")
    vix3m = pd.to_numeric(df[vix3m_col],  errors="coerce")

    # Avoid division by zero
    vix_safe = vix.replace(0.0, np.nan)
    slope = vix3m / vix_safe - 1.0

    # Piecewise linear: contango (slope >= 0.05) → 0; flat (0) → 50; inverted (<= -0.05) → 100
    score = pd.Series(np.nan, index=df.index, name="vol_term_score")

    # slope >= 0.05: score = linear from 0 at 0.05+ (we cap at 0)
    # For slope in [0.05, +inf): score = 0
    # For slope in [-0.05, 0.05]: linear from 100 to 0 mapped through midpoint
    #   at slope=0 → 50; at slope=0.05 → 0; at slope=-0.05 → 100
    # For slope <= -0.05: score = 100
    #
    # Unified formula: score = clamp((−slope / 0.05) * 50 + 50, 0, 100)
    # Check: slope=0.05  → (-0.05/0.05)*50+50 = -50+50 = 0   ✓
    #        slope=0.0   → 0+50 = 50   ✓
    #        slope=-0.05 → (0.05/0.05)*50+50 = 50+50 = 100   ✓
    raw = (-slope / 0.05) * 50.0 + 50.0
    score = raw.clip(0.0, 100.0).rename("vol_term_score")

    # Where either input is NaN, result must be NaN
    score = score.where(vix.notna() & vix3m.notna(), other=np.nan)
    return score


def _vrp_score(df: pd.DataFrame) -> pd.Series:
    """VRP = VIX - realized_vol_21d. Negative VRP → high score (vol expensive).

    VRP > 5  → score = 0   (vol cheap, fear premium intact)
    VRP < -5 → score = 100 (vol expensive, fear premium inverted)
    Linear between.

    Returns all-NaN Series if VIX or sp500 absent.
    """
    if "vix" not in df.columns or df["vix"].dropna().empty:
        return pd.Series(np.nan, index=df.index, name="vol_vrp_score")

    rv  = _compute_realized_vol(df, window=21)
    vix = pd.to_numeric(df["vix"], errors="coerce")
    vrp = vix - rv

    # Linear map: score = clamp((5 - VRP) / 10 * 100, 0, 100)
    # Check: VRP=5  → (5-5)/10*100 = 0    ✓
    #        VRP=0  → (5-0)/10*100 = 50   ✓
    #        VRP=-5 → (5+5)/10*100 = 100  ✓
    raw   = (_VRP_CHEAP_FLOOR - vrp) / (_VRP_CHEAP_FLOOR - _VRP_EXPENSIVE_CEIL) * 100.0
    score = raw.clip(0.0, 100.0).rename("vol_vrp_score")
    score = score.where(vix.notna() & rv.notna(), other=np.nan)
    return score


def _skew_score(df: pd.DataFrame) -> pd.Series:
    """CBOE SKEW index normalized to 0-100.

    SKEW <= 110 → 0
    SKEW >= 150 → 100
    Linear between.

    Returns all-NaN Series if skew column absent.
    """
    skew_col = _resolve_skew_col(df)
    if skew_col is None:
        return pd.Series(np.nan, index=df.index, name="vol_skew_score")

    skew = pd.to_numeric(df[skew_col], errors="coerce")
    raw  = (skew - _SKEW_LOW) / (_SKEW_HIGH - _SKEW_LOW) * 100.0
    return raw.clip(0.0, 100.0).rename("vol_skew_score")


def _move_score(df: pd.DataFrame) -> pd.Series:
    """MOVE index z-score vs 252-day history, clipped and scaled to [0, 100].

    Uses the same z-score → [0, 100] mapping as VIX.
    Returns all-NaN Series if 'move' column absent.
    """
    if "move" not in df.columns or df["move"].dropna().empty:
        return pd.Series(np.nan, index=df.index, name="vol_move_score")

    move = pd.to_numeric(df["move"], errors="coerce")
    zs   = _rolling_zscore(move, _ZSCORE_WINDOW)
    return _zscore_to_0_100(zs, _VIX_ZSCORE_LOW, _VIX_ZSCORE_HIGH).rename("vol_move_score")


# ---------------------------------------------------------------------------
# Composite computation
# ---------------------------------------------------------------------------

def compute_vol_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all component vol scores and the composite.

    Adds columns to a copy of df:
        vol_vix_score, vol_term_score, vol_vrp_score, vol_skew_score,
        vol_move_score, vol_composite_score, vol_regime

    The composite is the equal-weight mean of whatever components have valid
    (non-NaN) values at each row.

    Returns
    -------
    pd.DataFrame — copy of df with vol_ columns appended.
    """
    out = df.copy()

    out["vol_vix_score"]  = _vix_score(df).values
    out["vol_term_score"] = _term_structure_score(df).values
    out["vol_vrp_score"]  = _vrp_score(df).values
    out["vol_skew_score"] = _skew_score(df).values
    out["vol_move_score"] = _move_score(df).values

    component_cols = [
        "vol_vix_score",
        "vol_term_score",
        "vol_vrp_score",
        "vol_skew_score",
        "vol_move_score",
    ]

    # Equal-weight average ignoring NaN at each row
    out["vol_composite_score"] = out[component_cols].mean(axis=1, skipna=True)

    # Where every component is NaN, composite should also be NaN
    all_nan_mask = out[component_cols].isna().all(axis=1)
    out.loc[all_nan_mask, "vol_composite_score"] = np.nan

    out["vol_regime"] = out["vol_composite_score"].apply(
        lambda v: _classify_vol_regime(v) if pd.notna(v) else "Unknown"
    )

    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_vol_regime_composite(df: pd.DataFrame) -> dict:
    """Compute volatility regime composite analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with a DatetimeIndex. Columns used when present:
        vix, vix_3m / vix3m, sp500, skew / cboe_skew, move.

    Returns
    -------
    dict with keys:
        available            : bool
        current              : dict (snapshot of latest values)
        historical           : pd.DataFrame (last 504 rows with vol_ columns)
        percentile_current   : float (0-100, where current score sits vs full history)
        regime_history       : pd.Series (last 252 rows of vol_regime labels)
        interpretation       : str
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        # Minimum requirement: VIX column and 252 rows
        if "vix" not in df.columns or df["vix"].dropna().empty:
            return {"available": False}

        if len(df) < _ZSCORE_WINDOW:
            return {"available": False}

        # Check at least one other component is present beyond VIX
        vix3m_col  = _resolve_vix3m_col(df)
        skew_col   = _resolve_skew_col(df)
        has_sp500  = "sp500" in df.columns and df["sp500"].dropna().any()
        has_move   = "move"  in df.columns and df["move"].dropna().any()
        has_other  = any([
            vix3m_col is not None,
            has_sp500,   # needed for VRP
            skew_col is not None,
            has_move,
        ])
        if not has_other:
            return {"available": False}

        enriched = compute_vol_composite(df)

        # ---- Current snapshot ------------------------------------------------
        last = enriched.iloc[-1]

        def _fval(col: str) -> float:
            v = last.get(col, np.nan)
            try:
                f = float(v)
                return f if not np.isnan(f) else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        composite_score = _fval("vol_composite_score")
        vol_regime      = str(last.get("vol_regime", "Unknown"))

        components: dict[str, float] = {
            "vix":            _fval("vol_vix_score"),
            "term_structure": _fval("vol_term_score"),
            "vrp":            _fval("vol_vrp_score"),
            "skew":           _fval("vol_skew_score"),
            "move":           _fval("vol_move_score"),
        }

        components_available = [
            k for k, v in components.items()
            if not (isinstance(v, float) and np.isnan(v))
        ]

        current = {
            "vol_composite_score":  round(composite_score, 2) if not np.isnan(composite_score) else float("nan"),
            "vol_regime":           vol_regime,
            "components":           {
                k: (round(v, 2) if not (isinstance(v, float) and np.isnan(v)) else float("nan"))
                for k, v in components.items()
            },
            "components_available": components_available,
            "n_components":         len(components_available),
        }

        # ---- Historical slice -------------------------------------------------
        vol_cols = [
            "vol_vix_score", "vol_term_score", "vol_vrp_score",
            "vol_skew_score", "vol_move_score", "vol_composite_score", "vol_regime",
        ]
        historical = enriched[vol_cols].tail(_HIST_ROWS).copy()

        # ---- Percentile of current score vs full history ----------------------
        percentile_current = float("nan")
        try:
            hist_scores = enriched["vol_composite_score"].dropna()
            if not hist_scores.empty and not np.isnan(composite_score):
                pct = float((hist_scores < composite_score).mean() * 100.0)
                percentile_current = round(pct, 1)
        except Exception:
            pass

        # ---- Regime history ---------------------------------------------------
        regime_history = enriched["vol_regime"].tail(_REGIME_ROWS).copy()

        # ---- Interpretation ---------------------------------------------------
        n = current["n_components"]
        score_str = f"{composite_score:.1f}" if not np.isnan(composite_score) else "N/A"

        if vol_regime == "Complacent":
            regime_desc = (
                "Volatility is cheap and markets are not pricing risk adequately. "
                "This can be a contrarian warning — complacency often precedes spikes."
            )
        elif vol_regime == "Normal":
            regime_desc = (
                "Volatility is appropriately priced. "
                "Fear premium is intact and markets appear to be discounting risk normally."
            )
        elif vol_regime == "Elevated":
            regime_desc = (
                "Volatility is elevated. Markets are cautious and pricing in additional risk. "
                "Credit spreads may be under pressure."
            )
        else:  # Stressed or Unknown
            regime_desc = (
                "Volatility is very expensive and fear is dominant. "
                "Stress signals are high across multiple inputs. "
                "This environment is typically associated with spread widening and de-risking."
            )

        pct_str = f"{percentile_current:.0f}th" if not np.isnan(percentile_current) else "N/A"
        interp = (
            f"Vol regime composite score: {score_str}/100 ({pct_str} percentile of full history), "
            f"regime: {vol_regime}. Based on {n} available component(s): "
            f"{', '.join(components_available) if components_available else 'none'}. "
            + regime_desc
        )

        return {
            "available":          True,
            "current":            current,
            "historical":         historical,
            "percentile_current": percentile_current,
            "regime_history":     regime_history,
            "interpretation":     interp,
        }

    except Exception:
        return {"available": False}
