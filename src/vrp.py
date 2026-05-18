"""
Volatility Risk Premium (VRP) — implied vol minus realised vol.

VRP = VIX (implied, annualised %) − 21-day realised SP500 vol (annualised %).

Interpretation
--------------
  Positive VRP (normal) : market pays for insurance; fear premium intact.
  Compressed (0–2)      : fear premium thin but positive.
  Inverted (< 0)        : realised vol exceeds implied — fear premium compressed.
                          Historically precedes vol spikes and credit spread
                          widening by ~2 weeks.
  Deeply Inverted (<-2) : severe inversion; strong stress signal.

Columns used
------------
  vix     : VIX level, annualised % (e.g. 18.5)
  sp500   : S&P 500 price level

Public API
----------
  REALIZED_VOL_WINDOW = 21
  VRP_SIGNAL_WINDOWS  = [21, 63]

  compute_vrp(df)             -> pd.DataFrame
  run_vrp_analysis(df)        -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REALIZED_VOL_WINDOW: int = 21          # primary realised-vol window (trading days)
VRP_SIGNAL_WINDOWS: list[int] = [21, 63]

_ANN = np.sqrt(252)                    # annualisation factor for daily returns
_ZSCORE_WINDOW = 252                   # rolling window for z-score context
_HIST_ROWS = 504                       # rows returned in the "historical" key

_VRP_REGIME_THRESHOLDS = (2.0, 0.0, -2.0)  # Normal / Compressed / Inverted / Deeply Inverted


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vrp_regime(vrp: float) -> str:
    """Map a VRP value to a regime label."""
    if np.isnan(vrp):
        return "Unknown"
    if vrp > 2.0:
        return "Normal (positive)"
    if vrp > 0.0:
        return "Compressed"
    if vrp > -2.0:
        return "Inverted (negative)"
    return "Deeply Inverted"


def _vrp_to_signal(vrp: float) -> float:
    """Piecewise mapping from VRP (vrp_21d) to a 0–100 credit stress score.

    Higher score = more stress / less fear premium.

    Breakpoints
    -----------
      vrp > 4     →  15  (abundant fear premium, benign)
      2 < vrp ≤ 4 →  30
      0 < vrp ≤ 2 →  50
     -2 < vrp ≤ 0 →  70
      vrp ≤ -2    →  88  (deeply inverted)
    """
    if np.isnan(vrp):
        return float("nan")
    if vrp > 4.0:
        return 15.0
    if vrp > 2.0:
        return 30.0
    if vrp > 0.0:
        return 50.0
    if vrp > -2.0:
        return 70.0
    return 88.0


def _rolling_zscore(series: pd.Series, window: int = _ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score with a minimum of window // 2 observations."""
    roll = series.rolling(window, min_periods=window // 2)
    mean = roll.mean()
    std = roll.std(ddof=1).replace(0, np.nan)
    return (series - mean) / std


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
# compute_vrp
# ---------------------------------------------------------------------------

def compute_vrp(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with VRP columns appended.

    Required source columns
    -----------------------
    vix    : VIX index level (annualised implied vol, %)
    sp500  : S&P 500 price level

    Columns added
    -------------
    sp500_realized_vol_21d   : annualised 21-day rolling std of SP500 daily log returns (%)
    sp500_realized_vol_63d   : annualised 63-day rolling std of SP500 daily log returns (%)
    vrp_21d                  : vix − sp500_realized_vol_21d (both in % terms)
    vrp_63d                  : vix − sp500_realized_vol_63d
    vrp_regime               : regime label based on vrp_21d
    vrp_zscore_1y            : rolling 252d z-score of vrp_21d
    vrp_signal               : 0–100 credit stress score based on vrp_21d

    Gracefully handles missing columns — affected output columns are NaN-filled.
    """
    out = df.copy()

    has_vix = "vix" in out.columns and out["vix"].notna().any()
    has_sp500 = "sp500" in out.columns and out["sp500"].notna().any()

    # ---- realised vol from SP500 daily pct_change ---------------------------
    for w in VRP_SIGNAL_WINDOWS:
        col = f"sp500_realized_vol_{w}d"
        if has_sp500:
            daily_ret = out["sp500"].pct_change()
            out[col] = (
                daily_ret
                .rolling(w, min_periods=max(5, w // 2))
                .std(ddof=1)
            ) * _ANN * 100.0   # convert to % terms matching VIX
        else:
            out[col] = np.nan

    # ---- VRP columns --------------------------------------------------------
    if has_vix and has_sp500:
        out["vrp_21d"] = out["vix"] - out["sp500_realized_vol_21d"]
        out["vrp_63d"] = out["vix"] - out["sp500_realized_vol_63d"]
    else:
        out["vrp_21d"] = np.nan
        out["vrp_63d"] = np.nan

    # ---- regime label -------------------------------------------------------
    if out["vrp_21d"].notna().any():
        out["vrp_regime"] = out["vrp_21d"].apply(
            lambda v: _vrp_regime(v) if pd.notna(v) else "Unknown"
        )
    else:
        out["vrp_regime"] = "Unknown"

    # ---- rolling z-score (1 year) -------------------------------------------
    if out["vrp_21d"].notna().any():
        out["vrp_zscore_1y"] = _rolling_zscore(out["vrp_21d"], _ZSCORE_WINDOW)
    else:
        out["vrp_zscore_1y"] = np.nan

    # ---- stress signal (0-100) ----------------------------------------------
    if out["vrp_21d"].notna().any():
        out["vrp_signal"] = out["vrp_21d"].apply(
            lambda v: _vrp_to_signal(v) if pd.notna(v) else float("nan")
        )
    else:
        out["vrp_signal"] = np.nan

    return out


# ---------------------------------------------------------------------------
# run_vrp_analysis
# ---------------------------------------------------------------------------

def run_vrp_analysis(df: pd.DataFrame) -> dict:
    """Top-level entry point for VRP analysis.

    Returns
    -------
    dict with keys:
        df               : enriched DataFrame (full history)
        current          : dict of latest values (see below)
        historical       : last 504 rows (index + vrp_21d, vrp_63d,
                           sp500_realized_vol_21d)
        warning          : non-empty str if vrp_21d < 0 (inverted VRP)
        available        : bool — True if vix AND sp500 are present
        interpretation   : one-sentence plain-English summary

    current keys
    ------------
        vix, realized_vol_21d, vrp_21d, vrp_63d, vrp_regime,
        vrp_zscore_1y, vrp_signal
    """
    if df is None or df.empty:
        return {"available": False}

    has_vix = "vix" in df.columns and df["vix"].notna().any()
    has_sp500 = "sp500" in df.columns and df["sp500"].notna().any()
    available = has_vix and has_sp500

    if not available:
        return {"available": False}

    enriched = compute_vrp(df)
    last = enriched.iloc[-1]

    # ---- current snapshot ---------------------------------------------------
    current: dict = {
        "vix": _safe_float(last.get("vix")),
        "realized_vol_21d": _safe_float(last.get("sp500_realized_vol_21d")),
        "vrp_21d": _safe_float(last.get("vrp_21d")),
        "vrp_63d": _safe_float(last.get("vrp_63d")),
        "vrp_regime": last.get("vrp_regime", "Unknown"),
        "vrp_zscore_1y": _safe_float(last.get("vrp_zscore_1y")),
        "vrp_signal": _safe_float(last.get("vrp_signal")),
    }

    # ---- historical slice ---------------------------------------------------
    hist_cols = ["vrp_21d", "vrp_63d", "sp500_realized_vol_21d"]
    hist_available = [c for c in hist_cols if c in enriched.columns]
    historical = enriched[hist_available].tail(_HIST_ROWS).copy()

    # ---- warning ------------------------------------------------------------
    vrp_val = current["vrp_21d"]
    warning = ""
    if vrp_val is not None and vrp_val < 0.0:
        if vrp_val < -2.0:
            warning = (
                f"VRP is deeply inverted ({vrp_val:.2f}%): realised SP500 vol "
                "significantly exceeds implied vol. This has historically preceded "
                "vol spikes and credit spread widening by ~2 weeks. High stress signal."
            )
        else:
            warning = (
                f"VRP is inverted ({vrp_val:.2f}%): realised vol exceeds implied vol. "
                "The fear premium is compressed — monitor for credit spread widening."
            )

    # ---- interpretation -----------------------------------------------------
    regime = current["vrp_regime"]
    vix_val = current["vix"]
    rv21 = current["realized_vol_21d"]

    if vrp_val is not None and vix_val is not None and rv21 is not None:
        interp = (
            f"VIX ({vix_val:.1f}%) is "
            + ("above" if vrp_val >= 0 else "below")
            + f" 21-day realised vol ({rv21:.1f}%), giving a VRP of {vrp_val:+.2f}% "
            f"({regime} regime) — "
            + {
                "Normal (positive)": "the fear premium is healthy and credit markets are broadly benign.",
                "Compressed": "the fear premium is thin but intact; watch for further compression.",
                "Inverted (negative)": "the fear premium has inverted, a potential leading indicator of credit stress.",
                "Deeply Inverted": "severe VRP inversion signals elevated credit risk; historically precedes spread widening.",
                "Unknown": "regime undetermined due to insufficient data.",
            }.get(regime, "regime undetermined.")
        )
    else:
        interp = "Insufficient VIX or SP500 data to compute the Volatility Risk Premium."

    return {
        "available": available,
        "df": enriched,
        "current": current,
        "historical": historical,
        "warning": warning,
        "interpretation": interp,
    }
