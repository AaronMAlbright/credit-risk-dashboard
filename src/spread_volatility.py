"""
Spread volatility analysis — rolling and GARCH(1,1) vol of HY and IG spread changes.

Rising spread vol while spread levels are flat is an early warning signal:
markets are uncertain about direction even when the aggregate level has not moved.
This divergence often precedes a directional break.

Key measures
------------
  Rolling vol  : std of daily spread changes, annualised by sqrt(252), over 21/63/90d windows
  GARCH(1,1)   : conditional vol estimate using fixed params (omega=5% of var, alpha=0.10,
                 beta=0.85). Pure numpy — no arch/statsmodels dependency.
  Vol regime   : Low / Normal / Elevated / Stress, based on 252d percentile of hy_spread_vol_21d
  Vol z-score  : (hy_spread_vol_21d − 252d mean) / 252d std

Public API
----------
  WINDOWS = [21, 63, 90]

  compute_spread_volatility(df) -> pd.DataFrame
  run_spread_volatility_analysis(df) -> dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOWS: list[int] = [21, 63, 90]

_ANN = np.sqrt(252)          # annualisation factor for daily data
_REGIME_WINDOW = 252         # window used for percentile / z-score context
_HIST_ROWS = 504             # rows returned in the "historical" key


# ---------------------------------------------------------------------------
# GARCH(1,1) helper — pure numpy
# ---------------------------------------------------------------------------

def _garch11(returns: np.ndarray, omega_frac: float = 0.05,
             alpha: float = 0.10, beta: float = 0.85) -> np.ndarray:
    """Single-pass GARCH(1,1) conditional volatility (annualised).

    Parameters
    ----------
    returns     : 1-D array of daily spread changes (may contain NaNs at start)
    omega_frac  : omega = omega_frac * unconditional variance
    alpha       : ARCH coefficient
    beta        : GARCH coefficient

    Returns
    -------
    1-D array of annualised conditional volatilities, same length as returns.
    Leading NaNs where returns itself is NaN are preserved.
    """
    n = len(returns)
    h = np.full(n, np.nan)
    var0 = np.nanvar(returns)
    if var0 == 0 or np.isnan(var0):
        return h
    omega = omega_frac * var0
    h[0] = var0
    for t in range(1, n):
        e2 = returns[t - 1] ** 2 if not np.isnan(returns[t - 1]) else var0
        h[t] = omega + alpha * e2 + beta * h[t - 1]
    return np.sqrt(h) * _ANN


# ---------------------------------------------------------------------------
# Vol-regime label helper
# ---------------------------------------------------------------------------

def _vol_regime(pct_rank: float) -> str:
    """Map a 0–100 percentile rank to a vol regime label."""
    if np.isnan(pct_rank):
        return "Unknown"
    if pct_rank < 40:
        return "Low"
    if pct_rank < 70:
        return "Normal"
    if pct_rank < 90:
        return "Elevated"
    return "Stress"


# ---------------------------------------------------------------------------
# compute_spread_volatility
# ---------------------------------------------------------------------------

def compute_spread_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with spread-volatility columns appended.

    Columns added
    -------------
    hy_spread_vol_{w}d        for w in WINDOWS  (HY rolling vol, annualised)
    ig_spread_vol_{w}d        for w in WINDOWS  (IG rolling vol, annualised)
    hy_spread_vol_garch       GARCH(1,1) conditional vol for HY spread changes
    ig_spread_vol_garch       GARCH(1,1) conditional vol for IG spread changes
    hy_vol_regime             "Low" / "Normal" / "Elevated" / "Stress"
    hy_vol_zscore_1y          rolling 252d z-score of hy_spread_vol_21d

    Missing source columns are handled gracefully — the corresponding output
    columns will be NaN-filled rather than raising.
    """
    out = df.copy()

    has_hy = "hy_spread" in out.columns and out["hy_spread"].notna().any()
    has_ig = "ig_spread" in out.columns and out["ig_spread"].notna().any()

    # ---- daily spread changes -----------------------------------------------
    hy_chg = out["hy_spread"].diff() if has_hy else pd.Series(np.nan, index=out.index)
    ig_chg = out["ig_spread"].diff() if has_ig else pd.Series(np.nan, index=out.index)

    # ---- rolling vol (annualised) -------------------------------------------
    for w in WINDOWS:
        out[f"hy_spread_vol_{w}d"] = (
            hy_chg.rolling(w, min_periods=max(5, w // 2)).std(ddof=1) * _ANN
        )
        out[f"ig_spread_vol_{w}d"] = (
            ig_chg.rolling(w, min_periods=max(5, w // 2)).std(ddof=1) * _ANN
        )

    # ---- GARCH(1,1) vol ------------------------------------------------------
    if has_hy:
        hy_arr = hy_chg.to_numpy(dtype=float)
        out["hy_spread_vol_garch"] = _garch11(hy_arr)
    else:
        out["hy_spread_vol_garch"] = np.nan

    if has_ig:
        ig_arr = ig_chg.to_numpy(dtype=float)
        out["ig_spread_vol_garch"] = _garch11(ig_arr)
    else:
        out["ig_spread_vol_garch"] = np.nan

    # ---- vol regime (percentile-based) --------------------------------------
    if f"hy_spread_vol_21d" in out.columns and out["hy_spread_vol_21d"].notna().any():
        vol_21 = out["hy_spread_vol_21d"]

        # rolling 252d rank (0–100 percentile within the trailing window)
        def _rolling_pct_rank(series: pd.Series, window: int = _REGIME_WINDOW) -> pd.Series:
            def _rank(x):
                val = x[-1]
                if np.isnan(val):
                    return np.nan
                return float((x[:-1] < val).mean() * 100)
            return series.rolling(window, min_periods=window // 2).apply(_rank, raw=True)

        pct_rank = _rolling_pct_rank(vol_21)
        out["hy_vol_regime"] = pct_rank.apply(
            lambda v: _vol_regime(v) if pd.notna(v) else "Unknown"
        )

        # rolling 252d z-score
        roll = vol_21.rolling(_REGIME_WINDOW, min_periods=_REGIME_WINDOW // 2)
        roll_mean = roll.mean()
        roll_std = roll.std(ddof=1).replace(0, np.nan)
        out["hy_vol_zscore_1y"] = (vol_21 - roll_mean) / roll_std
    else:
        out["hy_vol_regime"] = "Unknown"
        out["hy_vol_zscore_1y"] = np.nan

    return out


# ---------------------------------------------------------------------------
# run_spread_volatility_analysis
# ---------------------------------------------------------------------------

def run_spread_volatility_analysis(df: pd.DataFrame) -> dict:
    """Top-level entry point for spread-volatility analysis.

    Returns
    -------
    dict with keys:
        df               : enriched DataFrame (full history)
        current          : dict of latest values for all spread-vol columns
        historical       : last 504 rows with key vol columns
        vol_of_vol       : std of hy_spread_vol_21d over the last 63 days
        warning          : non-empty str if regime is Stress or z-score > 2.0
        available        : bool — True when hy_spread or ig_spread is present
        interpretation   : one-sentence plain-English summary
    """
    if df is None or df.empty:
        return {"available": False}

    has_hy = "hy_spread" in df.columns and df["hy_spread"].notna().any()
    has_ig = "ig_spread" in df.columns and df["ig_spread"].notna().any()
    available = has_hy or has_ig

    if not available:
        return {"available": False}

    enriched = compute_spread_volatility(df)

    # ---- current (latest row) -----------------------------------------------
    vol_cols = (
        [f"hy_spread_vol_{w}d" for w in WINDOWS]
        + [f"ig_spread_vol_{w}d" for w in WINDOWS]
        + ["hy_spread_vol_garch", "ig_spread_vol_garch",
           "hy_vol_regime", "hy_vol_zscore_1y"]
    )
    last_row = enriched.iloc[-1]
    current: dict = {}
    for col in vol_cols:
        if col in enriched.columns:
            val = last_row[col]
            if isinstance(val, (np.floating, np.integer)):
                val = float(val)
            current[col] = val if pd.notna(val) else None
        else:
            current[col] = None

    # ---- historical slice ---------------------------------------------------
    hist_keep = ["hy_spread_vol_21d", "hy_spread_vol_63d",
                 "ig_spread_vol_21d", "hy_spread_vol_garch"]
    hist_cols_available = [c for c in hist_keep if c in enriched.columns]
    historical = enriched[hist_cols_available].tail(_HIST_ROWS).copy()

    # ---- vol of vol (std of 21d vol over last 63 days) ----------------------
    if "hy_spread_vol_21d" in enriched.columns:
        vov_series = enriched["hy_spread_vol_21d"].tail(63).dropna()
        vol_of_vol: float | None = float(vov_series.std(ddof=1)) if len(vov_series) >= 5 else None
    else:
        vol_of_vol = None

    # ---- warning -----------------------------------------------------------
    regime = current.get("hy_vol_regime") or "Unknown"
    zscore = current.get("hy_vol_zscore_1y")
    zscore_val = zscore if (zscore is not None and not np.isnan(float(zscore if zscore is not None else np.nan))) else None

    warning = ""
    if regime == "Stress":
        warning = (
            "HY spread volatility is in Stress regime (>90th percentile of trailing 252d distribution). "
            "Elevated uncertainty about credit direction — monitor for spread breakout."
        )
    elif zscore_val is not None and zscore_val > 2.0:
        warning = (
            f"HY spread volatility z-score is {zscore_val:.2f} (>2.0 σ above 1-year mean). "
            "Volatility is unusually high relative to recent history."
        )

    # ---- interpretation ----------------------------------------------------
    hy_vol_21 = current.get("hy_spread_vol_21d")
    ig_vol_21 = current.get("ig_spread_vol_21d")

    if regime in ("Low", "Normal", "Elevated", "Stress") and hy_vol_21 is not None:
        interp = (
            f"HY spread volatility is in the '{regime}' regime "
            f"(21d annualised vol: {hy_vol_21:.1f} bps/yr"
            + (f", IG: {ig_vol_21:.1f} bps/yr" if ig_vol_21 is not None else "")
            + "), suggesting "
            + {
                "Low": "benign credit conditions with limited directional uncertainty.",
                "Normal": "normal credit market volatility with no immediate stress signals.",
                "Elevated": "rising spread uncertainty — a potential precursor to a directional move.",
                "Stress": "significant spread instability; markets are pricing high directional risk.",
            }[regime]
        )
    else:
        interp = "Insufficient spread data to characterise current volatility regime."

    return {
        "available": available,
        "df": enriched,
        "current": current,
        "historical": historical,
        "vol_of_vol": vol_of_vol,
        "warning": warning,
        "interpretation": interp,
    }
