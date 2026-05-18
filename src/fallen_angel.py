"""
Fallen angel risk analysis — BBB-to-junk downgrade cliff monitoring.

The BBB tranche is the largest rating bucket in investment-grade credit.
When macro conditions deteriorate, a wave of BBB downgrades into high-yield
("fallen angels") can overwhelm HY index capacity, forcing mechanical selling
and amplifying spread widening.

Key signals
-----------
  HY/IG ratio             : the single most reliable cliff-edge indicator.
                            Historically averages ~3–4x; stress episodes push
                            above 5–6x as HY widens faster than IG.
  BBB–BB differential     : direct measure of the market premium charged for
                            the BBB→BB transition risk. Spikes ahead of
                            downgrades as investors de-risk at the boundary.
  Fallen angel signal     : 0–100 composite stress score based on ratio z-score.
  Cliff risk flag         : True when ratio exceeds 1.5× its historical median.

Available source columns (checked before use)
---------------------------------------------
  hy_spread    ICE BofA HY OAS (bps)
  ig_spread    ICE BofA IG OAS (bps)
  bbb_spread   ICE BofA BBB OAS (bps)   — optional
  bb_spread    ICE BofA BB OAS  (bps)   — optional

Public API
----------
  FALLEN_ANGEL_THRESHOLD = 1.5

  compute_fallen_angel(df) -> pd.DataFrame
  run_fallen_angel_analysis(df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FALLEN_ANGEL_THRESHOLD: float = 1.5   # HY/IG ratio × median triggers cliff flag

_ZSCORE_WINDOW = 252     # trading days for rolling z-score
_HIST_ROWS = 504         # rows returned in "historical" key


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return the column if it exists, otherwise a NaN-filled series."""
    if col in df.columns and df[col].notna().any():
        return df[col]
    return pd.Series(np.nan, index=df.index)


def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score with a trailing window; NaN where fewer than 20 obs."""
    roll_mean = series.rolling(window, min_periods=20).mean()
    roll_std = series.rolling(window, min_periods=20).std(ddof=1).replace(0, np.nan)
    return (series - roll_mean) / roll_std


def _fallen_angel_regime(ratio: float) -> str:
    """Qualitative regime label from the HY/IG ratio level."""
    if np.isnan(ratio):
        return "Unknown"
    if ratio < 3.5:
        return "Compressed (low stress)"
    if ratio < 4.5:
        return "Normal"
    if ratio < 6.0:
        return "Elevated (watch)"
    return "Crisis (fallen angel wave likely)"


def _fallen_angel_signal(zscore: float) -> int:
    """0–100 stress score derived from the HY/IG ratio z-score."""
    if np.isnan(zscore):
        return 20          # no data — default to low-moderate
    if zscore < 0:
        return 20
    if zscore < 1:
        return 40
    if zscore < 2:
        return 65
    return 85


def _build_interpretation(current: dict) -> str:
    """One-sentence plain-English summary of fallen-angel risk."""
    ratio = current.get("hy_ig_ratio")
    regime = current.get("fallen_angel_regime", "Unknown")
    signal = current.get("fallen_angel_signal")
    bbb_bb = current.get("bbb_bb_differential")
    cliff = current.get("cliff_risk_flag")

    if ratio is None or np.isnan(float(ratio if ratio is not None else np.nan)):
        return "Insufficient spread data to assess fallen angel risk."

    parts = [
        f"HY/IG spread ratio is {ratio:.2f}x ({regime})"
        + (f", with a fallen angel stress score of {signal}/100" if signal is not None else "")
        + "."
    ]
    if bbb_bb is not None and not np.isnan(float(bbb_bb)):
        parts.append(
            f"The BBB–BB transition premium is {bbb_bb:.0f}bps, "
            + ("suggesting the market is actively pricing BBB cliff risk."
               if bbb_bb > 150 else
               "within a historically normal range for this boundary.")
        )
    if cliff:
        parts.append(
            "The cliff risk flag is active — the ratio exceeds 1.5× its historical median, "
            "consistent with prior periods of accelerated fallen angel activity."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# compute_fallen_angel
# ---------------------------------------------------------------------------

def compute_fallen_angel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with fallen-angel risk columns appended.

    Columns added
    -------------
    hy_ig_ratio             hy_spread / ig_spread
    hy_ig_ratio_zscore      rolling 252d z-score of hy_ig_ratio
    bbb_bb_differential     bb_spread − bbb_spread  (NaN if either is missing)
    fallen_angel_regime     str label based on hy_ig_ratio level
    fallen_angel_signal     0–100 stress score based on z-score
    cliff_risk_flag         bool — True when ratio > FALLEN_ANGEL_THRESHOLD × median

    Missing source columns are handled gracefully.
    """
    out = df.copy()

    hy = _safe_col(out, "hy_spread")
    ig = _safe_col(out, "ig_spread")

    # ---- HY/IG ratio --------------------------------------------------------
    # Avoid division by zero; NaN when ig is 0 or missing
    ig_safe = ig.replace(0, np.nan)
    out["hy_ig_ratio"] = hy / ig_safe

    # ---- rolling z-score of ratio -------------------------------------------
    out["hy_ig_ratio_zscore"] = _rolling_zscore(out["hy_ig_ratio"])

    # ---- BBB–BB differential (fallen angel premium) -------------------------
    has_bbb = "bbb_spread" in out.columns and out["bbb_spread"].notna().any()
    has_bb = "bb_spread" in out.columns and out["bb_spread"].notna().any()
    if has_bb and has_bbb:
        out["bbb_bb_differential"] = out["bb_spread"] - out["bbb_spread"]
    else:
        out["bbb_bb_differential"] = np.nan

    # ---- regime label -------------------------------------------------------
    out["fallen_angel_regime"] = out["hy_ig_ratio"].apply(
        lambda v: _fallen_angel_regime(v) if pd.notna(v) else "Unknown"
    )

    # ---- stress signal (0–100) ----------------------------------------------
    out["fallen_angel_signal"] = out["hy_ig_ratio_zscore"].apply(
        lambda v: _fallen_angel_signal(v) if pd.notna(v) else 20
    )

    # ---- cliff risk flag ----------------------------------------------------
    ratio_series = out["hy_ig_ratio"].dropna()
    if len(ratio_series) >= 20:
        ratio_median = float(ratio_series.median())
        out["cliff_risk_flag"] = out["hy_ig_ratio"] > (FALLEN_ANGEL_THRESHOLD * ratio_median)
    else:
        out["cliff_risk_flag"] = False

    # Ensure bool dtype (not object) ----------------------------------------
    out["cliff_risk_flag"] = out["cliff_risk_flag"].fillna(False).astype(bool)

    return out


# ---------------------------------------------------------------------------
# run_fallen_angel_analysis
# ---------------------------------------------------------------------------

def run_fallen_angel_analysis(df: pd.DataFrame) -> dict:
    """Top-level entry point for fallen-angel risk analysis.

    Returns
    -------
    dict with keys:
        df                  : enriched DataFrame (full history)
        current             : dict of latest scalar values
        historical          : last 504 rows (index + hy_ig_ratio + zscore + signal)
        percentile_current  : float — where current ratio sits in full history (0–100)
        historical_extremes : dict — max/min ratio with dates
        interpretation      : one-sentence plain-English string
        available           : bool — True when both hy_spread and ig_spread are present
        warning             : str — non-empty when cliff_risk_flag is True
    """
    if df is None or df.empty:
        return {"available": False}

    has_hy = "hy_spread" in df.columns and df["hy_spread"].notna().any()
    has_ig = "ig_spread" in df.columns and df["ig_spread"].notna().any()
    available = has_hy and has_ig

    if not available:
        return {
            "available": False,
            "df": df.copy(),
        }

    enriched = compute_fallen_angel(df)
    last = enriched.iloc[-1]

    # ---- current dict -------------------------------------------------------
    def _scalar(col: str):
        if col not in enriched.columns:
            return None
        val = last[col]
        if isinstance(val, (np.floating, np.integer)):
            return float(val)
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        return val if pd.notna(val) else None

    current = {
        "hy_spread": _scalar("hy_spread"),
        "ig_spread": _scalar("ig_spread"),
        "hy_ig_ratio": _scalar("hy_ig_ratio"),
        "hy_ig_ratio_zscore": _scalar("hy_ig_ratio_zscore"),
        "bbb_bb_differential": _scalar("bbb_bb_differential"),
        "fallen_angel_regime": _scalar("fallen_angel_regime"),
        "fallen_angel_signal": _scalar("fallen_angel_signal"),
        "cliff_risk_flag": _scalar("cliff_risk_flag"),
    }

    # ---- historical slice ---------------------------------------------------
    hist_keep = ["hy_ig_ratio", "hy_ig_ratio_zscore", "fallen_angel_signal"]
    hist_cols_avail = [c for c in hist_keep if c in enriched.columns]
    historical = enriched[hist_cols_avail].tail(_HIST_ROWS).copy()

    # ---- current percentile (0–100) ----------------------------------------
    ratio_series = enriched["hy_ig_ratio"].dropna()
    current_ratio = current.get("hy_ig_ratio")
    if current_ratio is not None and len(ratio_series) > 0:
        percentile_current = float((ratio_series < current_ratio).mean() * 100)
    else:
        percentile_current = float("nan")

    # ---- historical extremes ------------------------------------------------
    if len(ratio_series) > 0:
        max_ratio = float(ratio_series.max())
        min_ratio = float(ratio_series.min())
        # Convert DatetimeIndex positions back to dates safely
        max_date = ratio_series.index[ratio_series.argmax()]
        min_date = ratio_series.index[ratio_series.argmin()]
        # Normalise to date string regardless of index type
        try:
            max_ratio_date = pd.Timestamp(max_date).date().isoformat()
        except Exception:
            max_ratio_date = str(max_date)
        try:
            min_ratio_date = pd.Timestamp(min_date).date().isoformat()
        except Exception:
            min_ratio_date = str(min_date)
        historical_extremes = {
            "max_ratio": max_ratio,
            "max_ratio_date": max_ratio_date,
            "min_ratio": min_ratio,
            "min_ratio_date": min_ratio_date,
        }
    else:
        historical_extremes = {
            "max_ratio": None,
            "max_ratio_date": None,
            "min_ratio": None,
            "min_ratio_date": None,
        }

    # ---- warning -----------------------------------------------------------
    cliff_flag = current.get("cliff_risk_flag")
    if cliff_flag:
        ratio_val = current.get("hy_ig_ratio")
        warning = (
            f"Cliff risk flag is active: HY/IG ratio ({ratio_val:.2f}x) exceeds "
            f"{FALLEN_ANGEL_THRESHOLD}× the historical median — consistent with "
            "elevated fallen angel downgrade pressure."
        )
    else:
        warning = ""

    # ---- interpretation ----------------------------------------------------
    interpretation = _build_interpretation(current)

    return {
        "available": available,
        "df": enriched,
        "current": current,
        "historical": historical,
        "percentile_current": percentile_current,
        "historical_extremes": historical_extremes,
        "interpretation": interpretation,
        "warning": warning,
    }
