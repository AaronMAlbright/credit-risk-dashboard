"""
Cross-Asset Momentum — systematic momentum scores across 5 asset classes.

Computes 1M/3M/6M/12M momentum scores for Equities, HY Credit, IG Credit,
Rates, and USD, then aggregates into a cross-asset alignment signal.

When all non-rate asset classes align (all positive or all negative 3M momentum),
the composite signal is most reliable and has the strongest historical predictive
power for regime transitions.

Public API
----------
  compute_momentum_score(series, window, invert) -> float
  compute_cross_asset_momentum(df)               -> pd.DataFrame
  get_alignment_score(momentum_df)               -> dict
  run_cross_asset_momentum(df)                   -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

ASSET_CLASSES = [
    # (name, column, invert)
    # invert=True  → higher values = positive momentum (e.g. equities)
    # invert=False → falling values = positive for risk (spreads/vol/DXY)
    # invert=None  → neutral directional signal (rates)
    ("Equities",  "sp500",     True),
    ("HY Credit", "hy_spread", False),
    ("IG Credit", "ig_spread", False),
    ("Rates",     "yield_10y", None),
    ("USD",       "dxy",       False),
]

MOMENTUM_WINDOWS = {
    "1M":  21,
    "3M":  63,
    "6M":  126,
    "12M": 252,
}

# Thresholds for momentum signal classification (applies to invert != None)
_STRONG_POS_THRESH  =  5.0
_POS_THRESH         =  1.0
_NEG_THRESH         = -1.0
_STRONG_NEG_THRESH  = -5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_signal(score: float) -> str:
    """Map a momentum score to a signal label."""
    if score > _STRONG_POS_THRESH:
        return "Strong Positive"
    if score > _POS_THRESH:
        return "Positive"
    if score >= _NEG_THRESH:
        return "Neutral"
    if score >= _STRONG_NEG_THRESH:
        return "Negative"
    return "Strong Negative"


def _is_positive_momentum(score) -> bool:
    """Return True if score exceeds the positive threshold."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return False
    return score > _POS_THRESH


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_momentum_score(series: pd.Series, window: int, invert) -> float:
    """Compute momentum score for a single series over the given window.

    For invert=True  : score = (current - past) / past * 100  (% change)
    For invert=False : score = -(current - past) / past * 100 (tightening = positive)
    For invert=None  : score = current - past                  (raw change, not directional)

    Returns float or NaN if insufficient data or past value is zero/NaN.
    """
    try:
        series = series.dropna()
        if len(series) <= window:
            return float("nan")

        current = float(series.iloc[-1])
        past    = float(series.iloc[-(window + 1)])

        if np.isnan(current) or np.isnan(past):
            return float("nan")

        if invert is None:
            # Raw change for rates — no directional scaling
            return current - past

        if past == 0.0:
            return float("nan")

        pct_change = (current - past) / abs(past) * 100.0
        return -pct_change if (invert is False) else pct_change

    except Exception:
        return float("nan")


def compute_cross_asset_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum scores for all asset classes and windows.

    Returns a DataFrame with one row per asset class and columns:
        asset, window_1m, window_3m, window_6m, window_12m, signal_3m, available
    """
    rows = []
    for name, col, invert in ASSET_CLASSES:
        row: dict = {
            "asset":      name,
            "window_1m":  float("nan"),
            "window_3m":  float("nan"),
            "window_6m":  float("nan"),
            "window_12m": float("nan"),
            "signal_3m":  "N/A",
            "available":  False,
        }

        if col not in df.columns:
            rows.append(row)
            continue

        series = df[col].dropna()
        if series.empty:
            rows.append(row)
            continue

        row["available"] = True
        scores = {}
        for label, window in MOMENTUM_WINDOWS.items():
            sc = compute_momentum_score(series, window, invert)
            scores[label] = sc

        row["window_1m"]  = scores.get("1M",  float("nan"))
        row["window_3m"]  = scores.get("3M",  float("nan"))
        row["window_6m"]  = scores.get("6M",  float("nan"))
        row["window_12m"] = scores.get("12M", float("nan"))

        # Signal classification only for directional assets
        sc_3m = scores.get("3M", float("nan"))
        if invert is not None and not (isinstance(sc_3m, float) and np.isnan(sc_3m)):
            row["signal_3m"] = _classify_signal(sc_3m)
        else:
            row["signal_3m"] = "N/A (Rates)"

        rows.append(row)

    return pd.DataFrame(rows, columns=[
        "asset", "window_1m", "window_3m", "window_6m", "window_12m",
        "signal_3m", "available",
    ])


def get_alignment_score(momentum_df: pd.DataFrame) -> dict:
    """Compute cross-asset alignment from the momentum table.

    Alignment = count of non-rate assets (invert != None) with positive 3M momentum.
    Scale: 0-4 (raw count), 0-100 (percentage).

    Returns
    -------
    dict with keys:
        alignment_score   : int   (0-4)
        alignment_pct     : float (0-100)
        aligned_positive  : list[str]
        aligned_negative  : list[str]
        interpretation    : str
    """
    # Build a lookup of invert by name
    _invert_map = {name: invert for name, _, invert in ASSET_CLASSES}

    aligned_positive: list[str] = []
    aligned_negative: list[str] = []

    for _, row in momentum_df.iterrows():
        asset = row["asset"]
        invert = _invert_map.get(asset)
        if invert is None:
            # Rates — skip alignment count
            continue
        if not row.get("available", False):
            continue

        sc = row.get("window_3m", float("nan"))
        if isinstance(sc, float) and np.isnan(sc):
            continue

        if _is_positive_momentum(sc):
            aligned_positive.append(asset)
        else:
            aligned_negative.append(asset)

    score = len(aligned_positive)  # 0-4
    pct   = score / 4.0 * 100.0

    if score == 4:
        interp = (
            "All non-rate asset classes show positive 3M momentum. "
            "Risk-on signal is strong and well-aligned across markets."
        )
    elif score == 3:
        interp = (
            "Three of four non-rate asset classes show positive momentum. "
            "Risk-on bias with one dissenting signal."
        )
    elif score == 2:
        interp = (
            "Mixed cross-asset momentum: markets are split. "
            "No clear directional bias — uncertainty elevated."
        )
    elif score == 1:
        interp = (
            "Only one non-rate asset class shows positive momentum. "
            "Risk-off bias with one exception."
        )
    else:
        interp = (
            "No non-rate asset classes show positive 3M momentum. "
            "Risk-off signal is strong and well-aligned across markets."
        )

    return {
        "alignment_score":  score,
        "alignment_pct":    round(pct, 1),
        "aligned_positive": aligned_positive,
        "aligned_negative": aligned_negative,
        "interpretation":   interp,
    }


# ---------------------------------------------------------------------------
# Historical alignment helper
# ---------------------------------------------------------------------------

def _compute_historical_alignment(df: pd.DataFrame, lookback: int = 504) -> pd.Series:
    """Compute rolling 3M alignment score (0-4) over the last *lookback* rows.

    Uses every 5th row (weekly resampling) to avoid excessive computation.
    Returns a pd.Series indexed by the DataFrame's DatetimeIndex slice.
    """
    _invert_map = {name: invert for name, _, invert in ASSET_CLASSES}
    _directional = [(name, col) for name, col, invert in ASSET_CLASSES if invert is not None]

    window_3m = MOMENTUM_WINDOWS["3M"]  # 63
    sub = df.tail(lookback)

    # Step every 5 rows (weekly)
    indices = sub.index[::5]
    scores: list[float] = []
    idx_out: list = []

    for ts in indices:
        try:
            pos_count = 0
            valid_count = 0
            for name, col in _directional:
                if col not in df.columns:
                    continue
                invert_val = _invert_map[name]
                # Build a slice up to and including this timestamp
                loc = df.index.get_loc(ts)
                if isinstance(loc, (slice, np.ndarray)):
                    # If there are duplicate timestamps, take the last
                    loc = int(np.where(df.index == ts)[0][-1])
                slice_end = int(loc) + 1
                if slice_end < window_3m + 2:
                    continue
                series = df[col].iloc[:slice_end].dropna()
                sc = compute_momentum_score(series, window_3m, invert_val)
                if not (isinstance(sc, float) and np.isnan(sc)):
                    valid_count += 1
                    if _is_positive_momentum(sc):
                        pos_count += 1

            scores.append(float(pos_count))
            idx_out.append(ts)
        except Exception:
            continue

    if not idx_out:
        return pd.Series(dtype=float)

    return pd.Series(scores, index=idx_out, name="alignment_score")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run_cross_asset_momentum(df: pd.DataFrame) -> dict:
    """Compute cross-asset momentum analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with a DatetimeIndex. Columns used (when present):
        sp500, hy_spread, ig_spread, yield_10y, dxy, final_decision.

    Returns
    -------
    dict with keys:
        available           : bool
        momentum_table      : pd.DataFrame   (from compute_cross_asset_momentum)
        alignment           : dict           (from get_alignment_score)
        historical_alignment: pd.Series      (rolling 3M alignment score)
        current_regime      : str or None
        signal_strength     : str            ("Strong" / "Moderate" / "Weak")
        interpretation      : str
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        # Check minimum row requirement
        if len(df) < MOMENTUM_WINDOWS["12M"]:  # 252
            return {"available": False}

        # Check that at least 2 asset class columns are present
        available_cols = [col for _, col, _ in ASSET_CLASSES if col in df.columns]
        if len(available_cols) < 2:
            return {"available": False}

        # Compute momentum table
        momentum_df = compute_cross_asset_momentum(df)

        # Alignment
        alignment = get_alignment_score(momentum_df)

        # Historical alignment series
        try:
            hist_alignment = _compute_historical_alignment(df, lookback=504)
        except Exception:
            hist_alignment = pd.Series(dtype=float)

        # Current regime
        current_regime: str | None = None
        try:
            if "final_decision" in df.columns:
                val = df["final_decision"].dropna()
                if not val.empty:
                    current_regime = str(val.iloc[-1])
        except Exception:
            pass

        # Signal strength based on alignment
        al_score = alignment["alignment_score"]
        if al_score == 4 or al_score == 0:
            signal_strength = "Strong"
        elif al_score == 3 or al_score == 1:
            signal_strength = "Moderate"
        else:
            signal_strength = "Weak"

        # Build interpretation
        direction = "Risk-On" if al_score >= 3 else ("Risk-Off" if al_score <= 1 else "Mixed")
        interp = (
            f"Cross-asset momentum is {direction} with {al_score}/4 non-rate asset classes "
            f"showing positive 3M momentum (alignment {alignment['alignment_pct']:.0f}%). "
            f"Signal strength: {signal_strength}. "
            + alignment["interpretation"]
        )

        return {
            "available":            True,
            "momentum_table":       momentum_df,
            "alignment":            alignment,
            "historical_alignment": hist_alignment,
            "current_regime":       current_regime,
            "signal_strength":      signal_strength,
            "interpretation":       interp,
        }

    except Exception:
        return {"available": False}
