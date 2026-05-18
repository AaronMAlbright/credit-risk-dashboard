"""
SOFR / Treasury Swap Spread Monitor.

The swap spread (fixed swap rate minus same-maturity Treasury yield) is one of
the most reliable signals of bank balance sheet stress and dollar funding demand.

Mechanics:
  Swap spread = fixed swap rate - Treasury yield (same maturity)

  Positive spread (normal): banks charge a counterparty risk premium.
    Demand for fixed-rate exposure and intermediation capacity is healthy.

  Negative spread (stress): banks cannot absorb more fixed-rate exposure on
    their balance sheets. Regulatory capital constraints (SLR, leverage ratio)
    prevent arbitrage — the Treasury market physically disconnects from the
    derivatives market. This is NOT normal arb pricing; it is a regime break.

Historical regimes:
  2015-2019: 30y swap spread turned persistently negative as post-GFC
             regulation compressed dealer balance sheets.
  March 2020: Acute negative spike across all tenors during UST dash-for-cash.
  2022-2023:  Ongoing negative 30y spread as QT reduces reserves and leverage
              constraints tighten.

A 30y spread < -20bps is a "Deeply Negative — Systemic Stress Signal" and
historically precedes or accompanies wider HY spreads and risk-off episodes.

FRED series used:
  DSWP2   — 2y USD swap rate
  DSWP5   — 5y USD swap rate
  DSWP10  — 10y USD swap rate
  DSWP30  — 30y USD swap rate
  DGS2    — 2y Treasury yield
  DGS5    — 5y Treasury yield
  DGS10   — 10y Treasury yield
  DGS30   — 30y Treasury yield
  SOFR    — Secured Overnight Financing Rate

Also checks df columns directly (swap_2y, swap_5y, swap_10y, swap_30y).

Public API
----------
  FRED_SWAP_SERIES    — dict of FRED series IDs
  fetch_swap_data(df) -> dict
  compute_swap_spreads(swap_data) -> dict
  get_current_spreads(spreads) -> dict
  run_swap_spread_monitor(df) -> dict
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRED_SWAP_SERIES: dict[str, str] = {
    "swap_2y":    "DSWP2",
    "swap_5y":    "DSWP5",
    "swap_10y":   "DSWP10",
    "swap_30y":   "DSWP30",
    "treasury_2y":  "DGS2",
    "treasury_5y":  "DGS5",
    "treasury_10y": "DGS10",
    "treasury_30y": "DGS30",
    "sofr":         "SOFR",
}

_TENORS: list[str] = ["2y", "5y", "10y", "30y"]

# Swap spread regime thresholds for a given tenor (bps)
_WIDE_POS    =  20.0   # > +20bps: Wide Positive — Elevated Bank Demand
_NORMAL_POS  =   5.0   # +5 to +20bps: Normal
_COMPRESSED  =  -5.0   # -5 to +5bps: Compressed
_NEGATIVE    = -20.0   # < -5bps: Negative — Balance Sheet Constrained
                        # < -20bps: Deeply Negative — Systemic Stress Signal

# Percentile window and minimum observations
_PERCENTILE_MIN = 30

# Historical rows to include in output
_HISTORY_ROWS = 504

# Fred data start date
_FRED_START = "2000-01-01"

# Rolling window for percentile computation
_ROLLING_ZSCORE_WINDOW = 252
_ROLLING_MIN_PERIODS = 63


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_valid(series: pd.Series | None) -> float:
    """Return the last non-NaN value in a Series as float, or NaN."""
    if series is None:
        return float("nan")
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[-1])


def _historical_percentile(series: pd.Series, current_val: float) -> float:
    """
    What fraction (0-100) of the series history is at or below current_val.
    Returns NaN if data is insufficient.
    """
    clean = series.dropna()
    if len(clean) < _PERCENTILE_MIN or np.isnan(current_val):
        return float("nan")
    return float((clean <= current_val).mean() * 100.0)


def _spread_regime(spread_bps: float) -> str:
    """
    Classify a swap spread level in basis points into a regime label.

    Boundaries:
        > +20bps  : Wide Positive — Elevated Bank Demand
        +5 to +20 : Normal
        -5 to +5  : Compressed
        -5 to -20 : Negative — Balance Sheet Constrained
        < -20bps  : Deeply Negative — Systemic Stress Signal
    """
    if np.isnan(spread_bps):
        return "Unknown"
    if spread_bps > _WIDE_POS:
        return "Wide Positive — Elevated Bank Demand"
    if spread_bps > _NORMAL_POS:
        return "Normal"
    if spread_bps > _COMPRESSED:
        return "Compressed"
    if spread_bps > _NEGATIVE:
        return "Negative — Balance Sheet Constrained"
    return "Deeply Negative — Systemic Stress Signal"


def _curve_shape(spread_2y: float, spread_30y: float) -> str:
    """
    Classify the swap spread curve shape from 2y vs 30y spread.

    Upward sloping: 30y spread > 2y spread (normal term premium).
    Flat: within ±5bps.
    Inverted: 30y spread materially below 2y (balance sheet constraints
              weigh more heavily on longer maturities).
    """
    if np.isnan(spread_2y) or np.isnan(spread_30y):
        return "Unknown"
    diff = spread_30y - spread_2y
    if diff > 5.0:
        return "Upward sloping"
    if diff < -5.0:
        return "Inverted"
    return "Flat"


def _build_interpretation(
    current_spreads: dict,
    stress_flag: bool,
    systemic_flag: bool,
    percentile_10y: float,
) -> str:
    """Construct a plain-English interpretation string."""
    s10 = current_spreads.get("spread_10y", float("nan"))
    s30 = current_spreads.get("spread_30y", float("nan"))
    regime_10y = current_spreads.get("regime_10y", "Unknown")
    regime_30y = current_spreads.get("regime_30y", "Unknown")
    curve = current_spreads.get("curve_shape", "Unknown")

    parts: list[str] = []

    if not np.isnan(s10):
        parts.append(
            f"The 10y swap spread is {s10:+.1f}bps ({regime_10y})."
        )
    if not np.isnan(s30):
        parts.append(
            f"The 30y spread is {s30:+.1f}bps ({regime_30y})."
        )
    if curve != "Unknown":
        parts.append(f"Swap spread curve shape: {curve}.")

    if systemic_flag:
        parts.append(
            "SYSTEMIC STRESS SIGNAL: at least one tenor is deeply negative (<-20bps). "
            "Banks cannot arbitrage the swap-Treasury basis — balance sheet capacity "
            "is exhausted. This precedes acute risk-off events and HY spread widening."
        )
    elif stress_flag:
        parts.append(
            "Balance sheet stress is present: at least one tenor is in negative "
            "swap spread territory. Dealer intermediation capacity is constrained. "
            "Monitor repo markets and UST auction tail risk."
        )

    if not np.isnan(percentile_10y):
        parts.append(
            f"The 10y spread is at the {percentile_10y:.0f}th percentile of recent history."
        )

    if not parts:
        return "Swap spread data unavailable — check FRED API key and connection."

    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Public: fetch_swap_data
# ---------------------------------------------------------------------------

def fetch_swap_data(df: pd.DataFrame) -> dict:
    """
    Fetch swap rate and Treasury yield data.

    Attempt order:
      1. FRED API via fredapi (requires FRED_API_KEY env var)
      2. df columns (swap_2y, swap_5y, swap_10y, swap_30y and treasury_*)
      3. Return available=False if both fail

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex.

    Returns
    -------
    dict with keys:
        available    : bool
        swap_2y      : pd.Series or None
        swap_5y      : pd.Series or None
        swap_10y     : pd.Series or None
        swap_30y     : pd.Series or None
        treasury_2y  : pd.Series or None
        treasury_5y  : pd.Series or None
        treasury_10y : pd.Series or None
        treasury_30y : pd.Series or None
        sofr         : pd.Series or None
        source       : str  — "FRED" / "df_columns" / "unavailable"
    """
    _unavail: dict = {
        "available": False,
        "swap_2y": None, "swap_5y": None, "swap_10y": None, "swap_30y": None,
        "treasury_2y": None, "treasury_5y": None, "treasury_10y": None, "treasury_30y": None,
        "sofr": None,
        "source": "unavailable",
    }

    # ---- 1. FRED API --------------------------------------------------------
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if fred_api_key:
            from fredapi import Fred
            fred = Fred(api_key=fred_api_key)

            result: dict = {
                "available": False,
                "source": "FRED",
                "swap_2y": None, "swap_5y": None, "swap_10y": None, "swap_30y": None,
                "treasury_2y": None, "treasury_5y": None,
                "treasury_10y": None, "treasury_30y": None,
                "sofr": None,
            }

            any_fetched = False
            for field_name, series_id in FRED_SWAP_SERIES.items():
                try:
                    raw = fred.get_series(series_id, observation_start=_FRED_START)
                    if raw is not None and not raw.empty:
                        result[field_name] = raw.dropna()
                        any_fetched = True
                except Exception:
                    result[field_name] = None

            if any_fetched:
                result["available"] = True
                return result
    except Exception:
        pass

    # ---- 2. df columns ------------------------------------------------------
    try:
        df_cols = set(df.columns)
        df_result: dict = {
            "available": False,
            "source": "df_columns",
            "swap_2y": None, "swap_5y": None, "swap_10y": None, "swap_30y": None,
            "treasury_2y": None, "treasury_5y": None,
            "treasury_10y": None, "treasury_30y": None,
            "sofr": None,
        }

        # Map field names to possible column names in df
        _col_candidates: dict[str, list[str]] = {
            "swap_2y":    ["swap_2y", "dswp2"],
            "swap_5y":    ["swap_5y", "dswp5"],
            "swap_10y":   ["swap_10y", "dswp10"],
            "swap_30y":   ["swap_30y", "dswp30"],
            "treasury_2y":  ["treasury_2y", "dgs2", "yield_2y"],
            "treasury_5y":  ["treasury_5y", "dgs5", "yield_5y"],
            "treasury_10y": ["treasury_10y", "dgs10", "yield_10y"],
            "treasury_30y": ["treasury_30y", "dgs30", "yield_30y"],
            "sofr":         ["sofr", "dff"],
        }

        any_found = False
        for field_name, candidates in _col_candidates.items():
            for col in candidates:
                if col in df_cols:
                    s = df[col].dropna()
                    if not s.empty:
                        df_result[field_name] = s
                        any_found = True
                        break

        if any_found:
            df_result["available"] = True
            return df_result
    except Exception:
        pass

    return _unavail


# ---------------------------------------------------------------------------
# Public: compute_swap_spreads
# ---------------------------------------------------------------------------

def compute_swap_spreads(swap_data: dict) -> dict:
    """
    Compute swap spread = swap_rate - treasury_yield for each tenor.

    Spread is expressed in basis points (multiplied by 100).

    Parameters
    ----------
    swap_data : dict
        Output of fetch_swap_data().

    Returns
    -------
    dict with keys:
        spread_2y        : pd.Series or None   (bps)
        spread_5y        : pd.Series or None
        spread_10y       : pd.Series or None
        spread_30y       : pd.Series or None
        available_tenors : list[str]
    """
    result: dict = {
        "spread_2y": None,
        "spread_5y": None,
        "spread_10y": None,
        "spread_30y": None,
        "available_tenors": [],
    }

    if not swap_data.get("available"):
        return result

    try:
        for tenor in _TENORS:
            swap_key = f"swap_{tenor}"
            tsy_key = f"treasury_{tenor}"

            swap_s: pd.Series | None = swap_data.get(swap_key)
            tsy_s: pd.Series | None = swap_data.get(tsy_key)

            if swap_s is None or tsy_s is None:
                continue
            if swap_s.empty or tsy_s.empty:
                continue

            # Align on shared index dates
            aligned = pd.DataFrame({"swap": swap_s, "tsy": tsy_s}).dropna()
            if aligned.empty:
                continue

            spread = (aligned["swap"] - aligned["tsy"]) * 100.0  # bps
            result[f"spread_{tenor}"] = spread
            result["available_tenors"].append(tenor)

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Public: get_current_spreads
# ---------------------------------------------------------------------------

def get_current_spreads(spreads: dict) -> dict:
    """
    Extract the latest spread values and classify regimes for each tenor.

    Parameters
    ----------
    spreads : dict
        Output of compute_swap_spreads().

    Returns
    -------
    dict with keys:
        spread_2y     : float or None
        spread_5y     : float or None
        spread_10y    : float or None
        spread_30y    : float or None
        regime_10y    : str
        regime_30y    : str
        curve_shape   : str   — spread curve shape (2y vs 30y)
        stress_flag   : bool  — True if any tenor < -5bps
        systemic_flag : bool  — True if any tenor < -20bps
        interpretation: str
    """
    result: dict = {
        "spread_2y": None,
        "spread_5y": None,
        "spread_10y": None,
        "spread_30y": None,
        "regime_10y": "Unknown",
        "regime_30y": "Unknown",
        "curve_shape": "Unknown",
        "stress_flag": False,
        "systemic_flag": False,
        "interpretation": "",
    }

    try:
        for tenor in _TENORS:
            key = f"spread_{tenor}"
            s = spreads.get(key)
            val = _last_valid(s) if s is not None else float("nan")
            result[key] = val if not np.isnan(val) else None

        s10 = result["spread_10y"] if result["spread_10y"] is not None else float("nan")
        s30 = result["spread_30y"] if result["spread_30y"] is not None else float("nan")

        result["regime_10y"] = _spread_regime(s10)
        result["regime_30y"] = _spread_regime(s30)

        s2 = result["spread_2y"] if result["spread_2y"] is not None else float("nan")
        result["curve_shape"] = _curve_shape(s2, s30)

        # Stress flags
        all_vals: list[float] = [
            v for v in [
                result["spread_2y"], result["spread_5y"],
                result["spread_10y"], result["spread_30y"],
            ]
            if v is not None and not np.isnan(v)
        ]
        result["stress_flag"] = any(v < -5.0 for v in all_vals)
        result["systemic_flag"] = any(v < -20.0 for v in all_vals)

        result["interpretation"] = _build_interpretation(
            result,
            stress_flag=result["stress_flag"],
            systemic_flag=result["systemic_flag"],
            percentile_10y=float("nan"),  # percentile populated in run_swap_spread_monitor
        )

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Public: run_swap_spread_monitor
# ---------------------------------------------------------------------------

def run_swap_spread_monitor(df: pd.DataFrame) -> dict:
    """
    Top-level swap spread analysis entry point.

    Fetches FRED data (with df column fallback), computes spreads for all four
    tenors, extracts current levels, classifies regimes, and returns historical
    series for charting.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with DatetimeIndex. Date must NOT be in df.columns —
        it lives in df.index.

    Returns
    -------
    dict with keys:
        available      : bool
        current        : dict   — from get_current_spreads(), with percentile_10y
        historical     : dict   — {tenor: pd.Series (last 504 rows), ...}
        percentile_10y : float or None
        warning        : str or None
        interpretation : str
    """
    _unavail: dict = {"available": False}

    try:
        # ---- Fetch raw rates ------------------------------------------------
        swap_data = fetch_swap_data(df)

        if not swap_data.get("available"):
            return _unavail

        # ---- Compute spreads ------------------------------------------------
        spreads = compute_swap_spreads(swap_data)

        if not spreads.get("available_tenors"):
            return _unavail

        # ---- Current spread levels and regimes ------------------------------
        current = get_current_spreads(spreads)

        # ---- Historical series (last 504 rows) ------------------------------
        historical: dict[str, pd.Series] = {}
        for tenor in _TENORS:
            key = f"spread_{tenor}"
            s = spreads.get(key)
            if s is not None and not s.empty:
                historical[tenor] = s.tail(_HISTORY_ROWS).copy()

        # ---- 10y spread percentile vs history -------------------------------
        percentile_10y: float = float("nan")
        s10_series = spreads.get("spread_10y")
        s10_val = current.get("spread_10y")
        if s10_series is not None and not s10_series.empty and s10_val is not None:
            percentile_10y = _historical_percentile(s10_series, float(s10_val))

        # Update interpretation with actual percentile
        current["interpretation"] = _build_interpretation(
            current,
            stress_flag=current.get("stress_flag", False),
            systemic_flag=current.get("systemic_flag", False),
            percentile_10y=percentile_10y,
        )

        interpretation = current.get("interpretation", "")

        # ---- Warning --------------------------------------------------------
        warning: str | None = None

        source = swap_data.get("source", "unavailable")
        if source == "df_columns":
            warning = (
                "Swap rate data sourced from df columns, not live FRED. "
                "Values may be stale. Set FRED_API_KEY for live data."
            )
        elif source == "unavailable":
            warning = "Swap and Treasury rate data unavailable — FRED API key missing or df lacks required columns."

        if current.get("systemic_flag"):
            sys_warn = (
                "SYSTEMIC SWAP SPREAD ALERT: at least one tenor is deeply negative "
                "(<-20bps). Treasury-derivatives dislocation is at acute stress levels. "
                "This is a confirmed balance sheet capacity crisis signal."
            )
            warning = f"{warning}  {sys_warn}" if warning else sys_warn
        elif current.get("stress_flag"):
            stress_warn = (
                "Swap spread stress detected: one or more tenors are negative. "
                "Bank balance sheet constraints are binding — monitor dealer repo "
                "and Treasury auction tail risk."
            )
            warning = f"{warning}  {stress_warn}" if warning else stress_warn

        return {
            "available": True,
            "current": current,
            "historical": historical,
            "percentile_10y": percentile_10y if not np.isnan(percentile_10y) else None,
            "warning": warning,
            "interpretation": interpretation,
        }

    except Exception:
        return _unavail
