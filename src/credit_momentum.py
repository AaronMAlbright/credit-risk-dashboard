"""
Credit Momentum — rate-of-change momentum in HY and IG credit spreads.

Concept
-------
Spread tightening (negative change) = positive credit momentum.
Cross-sectional momentum (HY vs IG relative change) signals which segment
the market favours.

Windows
-------
  1M  (21 trading days)
  3M  (63 trading days)
  6M  (126 trading days)

Key derived measures
--------------------
  Momentum z-scores (rolling 252d) for HY and IG at the 63d horizon.
  HY/IG divergence: HY widening faster than IG signals subordinate stress.
  Composite 0–100 stress signal via z-score normalisation.

Columns used
------------
  hy_spread : HY credit spread in bps
  ig_spread : IG credit spread in bps

Both are handled gracefully — if only one is present, available=True and
the missing spread's columns are NaN-filled.

Public API
----------
  MOMENTUM_WINDOWS = [21, 63, 126]

  compute_credit_momentum(df)          -> pd.DataFrame
  run_credit_momentum_analysis(df)     -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOMENTUM_WINDOWS: list[int] = [21, 63, 126]   # 1M, 3M, 6M

_ZSCORE_WINDOW = 252      # rolling window for z-score context
_HIST_ROWS = 504          # rows returned in the "historical" key

# HY 63d regime thresholds (bps)
_STRONG_TIGHT = -50.0
_TIGHT        = -15.0
_FLAT_LOW     =  15.0
_WIDE         =  50.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score with a minimum of window // 2 observations."""
    roll = series.rolling(window, min_periods=window // 2)
    mean = roll.mean()
    std = roll.std(ddof=1).replace(0, np.nan)
    return (series - mean) / std


def _hy_regime(mom_63d: float) -> str:
    """Map hy_mom_63d (bps) to a regime label."""
    if np.isnan(mom_63d):
        return "Unknown"
    if mom_63d < _STRONG_TIGHT:
        return "Strong Tightening"
    if mom_63d < _TIGHT:
        return "Tightening"
    if mom_63d <= _FLAT_LOW:
        return "Flat"
    if mom_63d <= _WIDE:
        return "Widening"
    return "Strong Widening"


def _momentum_signal_from_zscores(hy_z: float, ig_z: float) -> float:
    """Combine HY and IG 63d momentum z-scores into a 0–100 stress score.

    Algorithm
    ---------
    1. Clamp each z-score to [-3, 3].
    2. Map to [0, 100] via (z + 3) / 6 * 100.
    3. Average the two (where available).

    A higher score means spreads are widening faster than normal (more stress).
    """
    scores = []
    for z in (hy_z, ig_z):
        if not np.isnan(z):
            z_clamped = max(-3.0, min(3.0, z))
            scores.append((z_clamped + 3.0) / 6.0 * 100.0)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _safe_float(val) -> float | None:
    """Convert a scalar to Python float; return None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# compute_credit_momentum
# ---------------------------------------------------------------------------

def compute_credit_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with credit-momentum columns appended.

    Required source columns (at least one must be present)
    -------------------------------------------------------
    hy_spread : HY credit spread in bps
    ig_spread : IG credit spread in bps

    Columns added
    -------------
    hy_mom_{21,63,126}d          : hy_spread.diff(n) in bps
    ig_mom_{21,63,126}d          : ig_spread.diff(n) in bps
    hy_mom_zscore_63d            : rolling 252d z-score of hy_mom_63d
    ig_mom_zscore_63d            : rolling 252d z-score of ig_mom_63d
    hy_mom_regime                : regime label based on hy_mom_63d
    hy_ig_momentum_divergence    : hy_mom_63d − ig_mom_63d
    credit_momentum_signal       : 0–100 composite stress score
    """
    out = df.copy()

    has_hy = "hy_spread" in out.columns and out["hy_spread"].notna().any()
    has_ig = "ig_spread" in out.columns and out["ig_spread"].notna().any()

    # ---- momentum (diff) over each window -----------------------------------
    for w in MOMENTUM_WINDOWS:
        out[f"hy_mom_{w}d"] = (
            out["hy_spread"].diff(w) if has_hy else pd.Series(np.nan, index=out.index)
        )
        out[f"ig_mom_{w}d"] = (
            out["ig_spread"].diff(w) if has_ig else pd.Series(np.nan, index=out.index)
        )

    # ---- rolling z-scores at 63d horizon ------------------------------------
    if has_hy and out["hy_mom_63d"].notna().any():
        out["hy_mom_zscore_63d"] = _rolling_zscore(out["hy_mom_63d"], _ZSCORE_WINDOW)
    else:
        out["hy_mom_zscore_63d"] = np.nan

    if has_ig and out["ig_mom_63d"].notna().any():
        out["ig_mom_zscore_63d"] = _rolling_zscore(out["ig_mom_63d"], _ZSCORE_WINDOW)
    else:
        out["ig_mom_zscore_63d"] = np.nan

    # ---- HY regime label ----------------------------------------------------
    if has_hy and out["hy_mom_63d"].notna().any():
        out["hy_mom_regime"] = out["hy_mom_63d"].apply(
            lambda v: _hy_regime(v) if pd.notna(v) else "Unknown"
        )
    else:
        out["hy_mom_regime"] = "Unknown"

    # ---- HY/IG divergence ---------------------------------------------------
    if has_hy and has_ig:
        out["hy_ig_momentum_divergence"] = out["hy_mom_63d"] - out["ig_mom_63d"]
    else:
        out["hy_ig_momentum_divergence"] = np.nan

    # ---- composite stress signal (0-100) ------------------------------------
    hy_z_series = out["hy_mom_zscore_63d"]
    ig_z_series = out["ig_mom_zscore_63d"]

    def _row_signal(idx):
        hz = hy_z_series.iloc[idx]
        iz = ig_z_series.iloc[idx]
        hz = hz if pd.notna(hz) else float("nan")
        iz = iz if pd.notna(iz) else float("nan")
        return _momentum_signal_from_zscores(hz, iz)

    signals = np.array([_row_signal(i) for i in range(len(out))], dtype=float)
    out["credit_momentum_signal"] = signals

    return out


# ---------------------------------------------------------------------------
# run_credit_momentum_analysis
# ---------------------------------------------------------------------------

def run_credit_momentum_analysis(df: pd.DataFrame) -> dict:
    """Top-level entry point for credit momentum analysis.

    Returns
    -------
    dict with keys:
        df                    : enriched DataFrame (full history)
        current               : dict of latest values (see below)
        historical            : last 504 rows (index + hy_mom_21d, hy_mom_63d,
                                ig_mom_63d, credit_momentum_signal)
        trend                 : "Improving" / "Deteriorating" / "Flat"
                                based on sign of hy_mom_21d (most recent)
        momentum_percentile   : float 0–100 — where current hy_mom_63d sits in history
        available             : bool — True if hy_spread OR ig_spread present
        interpretation        : one-sentence plain-English summary

    current keys
    ------------
        hy_spread, ig_spread,
        hy_mom_21d, hy_mom_63d, hy_mom_126d,
        ig_mom_21d, ig_mom_63d, ig_mom_126d,
        hy_mom_regime, hy_ig_momentum_divergence,
        credit_momentum_signal
    """
    if df is None or df.empty:
        return {"available": False}

    has_hy = "hy_spread" in df.columns and df["hy_spread"].notna().any()
    has_ig = "ig_spread" in df.columns and df["ig_spread"].notna().any()
    available = has_hy or has_ig

    if not available:
        return {"available": False}

    enriched = compute_credit_momentum(df)
    last = enriched.iloc[-1]

    # ---- current snapshot ---------------------------------------------------
    current: dict = {
        "hy_spread": _safe_float(last.get("hy_spread")),
        "ig_spread": _safe_float(last.get("ig_spread")),
        "hy_mom_21d": _safe_float(last.get("hy_mom_21d")),
        "hy_mom_63d": _safe_float(last.get("hy_mom_63d")),
        "hy_mom_126d": _safe_float(last.get("hy_mom_126d")),
        "ig_mom_21d": _safe_float(last.get("ig_mom_21d")),
        "ig_mom_63d": _safe_float(last.get("ig_mom_63d")),
        "ig_mom_126d": _safe_float(last.get("ig_mom_126d")),
        "hy_mom_regime": last.get("hy_mom_regime", "Unknown"),
        "hy_ig_momentum_divergence": _safe_float(last.get("hy_ig_momentum_divergence")),
        "credit_momentum_signal": _safe_float(last.get("credit_momentum_signal")),
    }

    # ---- historical slice ---------------------------------------------------
    hist_cols = ["hy_mom_21d", "hy_mom_63d", "ig_mom_63d", "credit_momentum_signal"]
    hist_available = [c for c in hist_cols if c in enriched.columns]
    historical = enriched[hist_available].tail(_HIST_ROWS).copy()

    # ---- trend (based on sign of hy_mom_21d) --------------------------------
    hy_mom_21 = current["hy_mom_21d"]
    if hy_mom_21 is None:
        trend = "Flat"
    elif hy_mom_21 < -5.0:
        trend = "Improving"
    elif hy_mom_21 > 5.0:
        trend = "Deteriorating"
    else:
        trend = "Flat"

    # ---- momentum percentile (current hy_mom_63d vs full history) -----------
    momentum_percentile: float | None = None
    if has_hy and "hy_mom_63d" in enriched.columns:
        mom_series = enriched["hy_mom_63d"].dropna()
        hy_mom_63 = current["hy_mom_63d"]
        if hy_mom_63 is not None and len(mom_series) >= 2:
            momentum_percentile = float((mom_series < hy_mom_63).mean() * 100.0)

    # ---- interpretation -----------------------------------------------------
    regime = current["hy_mom_regime"]
    hy_63 = current["hy_mom_63d"]
    signal = current["credit_momentum_signal"]
    divergence = current["hy_ig_momentum_divergence"]

    if hy_63 is not None:
        direction = "tightened" if hy_63 < 0 else "widened"
        bps_abs = abs(hy_63)
        div_str = ""
        if divergence is not None:
            div_str = (
                f"; HY is widening {abs(divergence):.0f} bps faster than IG"
                if divergence > 5
                else (
                    f"; HY is tightening {abs(divergence):.0f} bps faster than IG"
                    if divergence < -5
                    else ""
                )
            )
        interp = (
            f"HY spreads have {direction} by {bps_abs:.0f} bps over 3M "
            f"('{regime}' regime{div_str}), "
            + (
                f"implying a stress signal of {signal:.0f}/100 — "
                if signal is not None
                else ""
            )
            + {
                "Strong Tightening": "credit momentum is strongly positive with broad spread compression.",
                "Tightening": "credit conditions are improving across the market.",
                "Flat": "credit momentum is neutral with no clear directional trend.",
                "Widening": "credit conditions are deteriorating; monitor for further spread pressure.",
                "Strong Widening": "rapid spread widening signals significant credit stress.",
                "Unknown": "regime undetermined due to insufficient data.",
            }.get(regime, "regime undetermined.")
        )
    else:
        interp = "Insufficient spread data to assess credit momentum."

    return {
        "available": available,
        "df": enriched,
        "current": current,
        "historical": historical,
        "trend": trend,
        "momentum_percentile": momentum_percentile,
        "interpretation": interp,
    }
