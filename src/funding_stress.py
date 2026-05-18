"""
Interbank / Funding Market Stress Monitor.

The TED spread (3-month T-bill vs 3-month LIBOR/SOFR proxy) and OIS-based
measures capture "plumbing" stress in credit markets. Elevated funding stress
precedes broad HY spread widening by roughly 3-5 weeks — it is the canary in
the coal mine before credit markets seize up.

Column resolution (checked in order):
  Fed Funds : "dff", "fed_funds_rate", "fed_funds"
  3m yield  : "yield_3m"
  LIBOR     : "libor_3m" (optional; improves TED spread accuracy when present)

Public API
----------
  compute_funding_stress(df)        -> pd.DataFrame
  run_funding_stress_analysis(df)   -> dict
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FF_CANDIDATES: tuple[str, ...] = ("dff", "fed_funds_rate", "fed_funds")
_3M_YIELD_COL: str = "yield_3m"
_LIBOR_COL: str = "libor_3m"

_ROLLING_WINDOW: int = 252           # 1 trading year for z-score
_MIN_PERIODS: int = 63               # minimum ~3 months of data
_HISTORY_ROWS: int = 504             # ~2 trading years for historical slice

# TED-spread regime thresholds (basis points)
_REGIME_THRESHOLDS = (10.0, 25.0, 50.0, 100.0)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate present in df.columns, or None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if it exists, else a NaN Series aligned to df.index."""
    if col in df.columns:
        return df[col].copy()
    return pd.Series(float("nan"), index=df.index, name=col)


def _rolling_zscore(series: pd.Series, window: int = _ROLLING_WINDOW) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std over *window* periods."""
    roll = series.rolling(window, min_periods=_MIN_PERIODS)
    return (series - roll.mean()) / roll.std()


def _classify_funding_regime(ted_bps: float) -> str:
    """
    Map TED spread proxy in basis points to a qualitative funding regime.

    thresholds (bps):
        < 10              -> "Benign"
        10 – 25           -> "Normal"
        25 – 50           -> "Elevated"
        50 – 100          -> "Stressed"
        > 100             -> "Crisis"
    """
    if pd.isna(ted_bps):
        return "Unknown"
    lo, mid, hi, extreme = _REGIME_THRESHOLDS
    if ted_bps < lo:
        return "Benign"
    if ted_bps < mid:
        return "Normal"
    if ted_bps < hi:
        return "Elevated"
    if ted_bps < extreme:
        return "Stressed"
    return "Crisis"


def _zscore_to_stress_signal(z: float) -> float:
    """
    Map TED spread z-score to a 0-100 funding stress signal.

    z < -1      ->  15
    -1 to  0    ->  30
     0 to  1    ->  50
     1 to  2    ->  70
    > 2         ->  88
    """
    if pd.isna(z):
        return float("nan")
    if z < -1.0:
        return 15.0
    if z < 0.0:
        return 30.0
    if z < 1.0:
        return 50.0
    if z < 2.0:
        return 70.0
    return 88.0


def _last_valid(series: pd.Series) -> float:
    """Return the last non-NaN value as a Python float, or NaN."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.iloc[-1])


# ---------------------------------------------------------------------------
# Public API — DataFrame enrichment
# ---------------------------------------------------------------------------


def compute_funding_stress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich df with interbank / funding stress columns.

    New columns
    -----------
    ted_spread_proxy     : TED spread in basis points.
                           Uses libor_3m - yield_3m when libor_3m is present;
                           otherwise yield_3m - fed_funds_rate (crude proxy).
    ted_spread_zscore    : Rolling 252-day z-score of ted_spread_proxy.
    ois_proxy            : fed_funds_rate - yield_3m (positive = funding pressure).
    funding_stress_signal: 0-100 score derived from ted_spread_zscore.
    funding_regime       : "Benign" / "Normal" / "Elevated" / "Stressed" / "Crisis".

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex. Fed Funds and yield_3m columns must be present
        for any non-NaN output.

    Returns
    -------
    pd.DataFrame — copy of df with new columns appended (NaN where unavailable).
    """
    out = df.copy()
    out.index = pd.to_datetime(out.index)

    _nan = float("nan")

    ff_col = _resolve_column(out, _FF_CANDIDATES)
    yield_3m_present = _3M_YIELD_COL in out.columns
    libor_present = _LIBOR_COL in out.columns

    if ff_col is None or not yield_3m_present:
        # Cannot compute anything — return NaN columns
        for col in (
            "ted_spread_proxy",
            "ted_spread_zscore",
            "ois_proxy",
            "funding_stress_signal",
            "funding_regime",
        ):
            out[col] = _nan if col != "funding_regime" else "Unknown"
        return out

    ff_series = out[ff_col]
    y3m_series = out[_3M_YIELD_COL]

    # TED spread proxy (in basis points)
    if libor_present:
        # Better measure: LIBOR - T-bill
        ted_raw = out[_LIBOR_COL] - y3m_series
    else:
        # Crude proxy: T-bill minus Fed Funds
        # (Fed Funds tracks OIS; T-bill is slightly below in calm markets)
        ted_raw = y3m_series - ff_series

    out["ted_spread_proxy"] = ted_raw * 100.0  # convert % to bps

    # Rolling z-score
    out["ted_spread_zscore"] = _rolling_zscore(out["ted_spread_proxy"])

    # OIS proxy: Fed Funds - T-bill (positive = banks paying premium over risk-free)
    out["ois_proxy"] = (ff_series - y3m_series) * 100.0  # bps

    # Stress signal from z-score
    out["funding_stress_signal"] = out["ted_spread_zscore"].apply(_zscore_to_stress_signal)

    # Funding regime from raw TED spread level
    out["funding_regime"] = out["ted_spread_proxy"].apply(_classify_funding_regime)

    return out


# ---------------------------------------------------------------------------
# Public API — full analysis
# ---------------------------------------------------------------------------


def run_funding_stress_analysis(df: pd.DataFrame) -> dict:
    """
    Full interbank/funding stress analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Main market data frame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        df               : enriched DataFrame (from compute_funding_stress)
        current          : dict of latest scalar values:
                             ted_spread_proxy_bps, ted_spread_zscore, ois_proxy,
                             funding_stress_signal, funding_regime,
                             yield_3m, fed_funds_rate
        historical       : last 504 rows — index + ted_spread_proxy +
                           funding_stress_signal columns
        warning          : non-empty str if funding_regime is "Stressed" or "Crisis"
        available        : True if yield_3m and at least one fed-funds column present
        interpretation   : one-sentence plain-English summary
        lead_weeks       : int = 3 (historical lead time; always returned as context)
    """
    ff_col = _resolve_column(df, _FF_CANDIDATES)
    yield_3m_present = _3M_YIELD_COL in df.columns
    available = (ff_col is not None) and yield_3m_present

    enriched = compute_funding_stress(df)

    # ---- Current scalars ----------------------------------------------------
    ted_bps = _last_valid(enriched["ted_spread_proxy"])
    ted_z = _last_valid(enriched["ted_spread_zscore"])
    ois = _last_valid(enriched["ois_proxy"])
    stress_signal = _last_valid(enriched["funding_stress_signal"])
    regime = (
        enriched["funding_regime"].dropna().iloc[-1]
        if enriched["funding_regime"].notna().any()
        else "Unknown"
    )
    y3m_val = _last_valid(enriched[_3M_YIELD_COL]) if yield_3m_present else float("nan")
    ff_val = _last_valid(enriched[ff_col]) if ff_col is not None else float("nan")

    current: dict = {
        "ted_spread_proxy_bps": ted_bps,
        "ted_spread_zscore": ted_z,
        "ois_proxy": ois,
        "funding_stress_signal": stress_signal,
        "funding_regime": regime,
        "yield_3m": y3m_val,
        "fed_funds_rate": ff_val,
    }

    # ---- Historical slice ---------------------------------------------------
    hist_cols = ["ted_spread_proxy", "funding_stress_signal"]
    hist = enriched[hist_cols].tail(_HISTORY_ROWS)

    # ---- Warning ------------------------------------------------------------
    if regime in ("Stressed", "Crisis"):
        warning = (
            f"Funding stress ALERT: interbank spread regime is '{regime}' "
            f"(TED proxy {ted_bps:.0f} bps, z-score {ted_z:.2f}). "
            "Credit market seizure risk elevated — monitor HY spreads."
        )
    else:
        warning = ""

    # ---- Interpretation -----------------------------------------------------
    if not available:
        interpretation = (
            "Funding stress analysis unavailable: yield_3m or fed_funds_rate "
            "columns not found in the dataset."
        )
    elif pd.isna(ted_bps):
        interpretation = (
            "Funding stress indicators are present but contain insufficient "
            "data to compute a current TED spread proxy."
        )
    else:
        direction = "above" if ted_bps > 25 else "within"
        interpretation = (
            f"The TED spread proxy stands at {ted_bps:.0f} bps "
            f"(z-score {ted_z:+.2f}), {direction} normal ranges, "
            f"placing interbank funding conditions in the '{regime}' regime."
        )

    return {
        "df": enriched,
        "current": current,
        "historical": hist,
        "warning": warning,
        "available": available,
        "interpretation": interpretation,
        "lead_weeks": 3,
    }
