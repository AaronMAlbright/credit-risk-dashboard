"""
Default cycle positioning — where are we in the historical credit cycle?

Default rates move in multi-year cycles driven by economic expansions and
recessions. Using the Jarrow-Turnbull implied default rate (λ = spread / LGD)
as a proxy for market-implied default expectations, this module:

1. Computes the implied default rate time series
2. Identifies named historical default cycles
3. Shows where the current level sits relative to cycle peaks and troughs
4. Estimates "distance from peak" (how much further spreads could widen)

Historical HY default rate peaks (Moody's actual default rates):
  1991 recession:    peak ~11%  (savings & loan crisis)
  2001–02 dotcom:    peak ~11%  (telecom/energy defaults)
  2008–09 GFC:       peak ~14%  (financial crisis)
  2015–16 energy:    peak ~5%   (oil price collapse)
  2020 COVID:        peak ~8%   (pandemic shock, brief)
  2023–24 post-hike: ~3–4%      (rate normalization cycle)

Public API
----------
  HISTORICAL_PEAKS = {cycle_name: {"year": int, "actual_peak_pct": float, "spread_at_peak": float}}
  compute_default_cycle(df) -> pd.DataFrame   adds default_cycle_ columns
  get_cycle_position(df) -> dict
  run_default_cycle_analysis(df) -> dict
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECOVERY_RATE = 0.40
LGD = 0.60  # Loss Given Default = 1 - Recovery Rate

# Historical peak data (Moody's actual HY default rates + approximate OAS at peak)
HISTORICAL_PEAKS = {
    "1991 S&L Crisis": {
        "year": 1991,
        "actual_peak_pct": 11.0,
        "spread_at_peak": 10.5,   # OAS in % (1050 bps)
        "recovery_months": 24,
    },
    "2001-02 Dotcom": {
        "year": 2002,
        "actual_peak_pct": 11.0,
        "spread_at_peak": 9.8,    # OAS in % (980 bps)
        "recovery_months": 30,
    },
    "2008-09 GFC": {
        "year": 2009,
        "actual_peak_pct": 14.0,
        "spread_at_peak": 19.0,   # OAS in % (1900 bps) — peak March 2009
        "recovery_months": 36,
    },
    "2015-16 Energy": {
        "year": 2016,
        "actual_peak_pct": 5.0,
        "spread_at_peak": 8.5,    # OAS in % (850 bps)
        "recovery_months": 18,
    },
    "2020 COVID": {
        "year": 2020,
        "actual_peak_pct": 8.0,
        "spread_at_peak": 11.0,   # OAS in % (1100 bps) — peak March 2020
        "recovery_months": 12,
    },
}

# Phase thresholds (implied default rate in %)
_PHASE_THRESHOLDS = [
    (2.0, "Trough (<2%)"),
    (4.0, "Rising (2-4%)"),
    (7.0, "Elevated (4-7%)"),
    (float("inf"), "Peak/Crisis (>7%)"),
]


def _implied_default_rate(hy_spread_series: pd.Series) -> pd.Series:
    """
    Jarrow-Turnbull implied default rate: λ = spread / LGD.

    If hy_spread is expressed in percentage points (e.g., 5.5 for 550bps),
    divide by 100 to get decimal spread first, then divide by LGD.
    Returns implied default rate as a decimal (e.g., 0.055 / 0.60 ≈ 0.092).
    """
    return (hy_spread_series / 100.0) / LGD


def _phase_label(implied_pct: float) -> str:
    """Map implied default rate (%) to a phase label."""
    if np.isnan(implied_pct):
        return "Unknown"
    for threshold, label in _PHASE_THRESHOLDS:
        if implied_pct < threshold:
            return label
    return "Peak/Crisis (>7%)"


def compute_default_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the DataFrame with default-cycle positioning columns.

    Added columns
    -------------
    implied_default_rate        : decimal (e.g., 0.092)
    default_cycle_pct           : implied default rate as % (e.g., 9.2)
    default_vs_historical_mean  : z-score vs 252-day rolling mean/std
    default_cycle_phase         : string label (Trough / Rising / Elevated / Peak)
    default_cycle_percentile    : rolling 252-day percentile rank (0–100)
    default_cycle_momentum_30d  : 30-day change in implied_default_rate (decimal)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'hy_spread'. Returns unchanged copy if absent.

    Returns
    -------
    pd.DataFrame (copy of input with new columns appended)
    """
    out = df.copy()

    if "hy_spread" not in out.columns:
        # Graceful degradation — return copy without cycle columns
        return out

    idr = _implied_default_rate(out["hy_spread"])
    out["implied_default_rate"] = idr
    out["default_cycle_pct"] = idr * 100.0

    # Z-score vs rolling 252-day window
    roll_mean = idr.rolling(252, min_periods=60).mean()
    roll_std = idr.rolling(252, min_periods=60).std()
    out["default_vs_historical_mean"] = (idr - roll_mean) / roll_std.replace(0, np.nan)

    # Phase label (vectorised via map)
    out["default_cycle_phase"] = out["default_cycle_pct"].map(_phase_label)

    # Rolling percentile rank: what fraction of last 252 days is the current value above?
    def _rolling_pctrank(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
        """Compute rolling percentile rank (0–100) without scipy."""
        ranks = []
        arr = series.values
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                ranks.append(np.nan)
                continue
            start = max(0, i - window + 1)
            window_vals = arr[start : i + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < min_periods:
                ranks.append(np.nan)
            else:
                pct = (valid < arr[i]).sum() / len(valid) * 100.0
                ranks.append(pct)
        return pd.Series(ranks, index=series.index)

    out["default_cycle_percentile"] = _rolling_pctrank(idr)

    # 30-day momentum
    out["default_cycle_momentum_30d"] = idr.diff(30)

    return out


def _closest_historical_analog(current_spread_pct: float, momentum: float) -> str:
    """
    Identify the historical cycle whose spread level is closest to current.

    Parameters
    ----------
    current_spread_pct : float  — current HY spread in % (e.g., 5.5 for 550bps)
    momentum           : float  — recent rate of change (positive = widening)

    Returns
    -------
    str — name of closest cycle
    """
    best_name = "Unknown"
    best_dist = float("inf")

    for name, data in HISTORICAL_PEAKS.items():
        peak_spread = data["spread_at_peak"]
        # Weight: spread distance matters most; momentum direction is a tiebreaker
        dist = abs(current_spread_pct - peak_spread)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


def get_cycle_position(df: pd.DataFrame) -> dict:
    """
    Summarise where the current implied default rate sits relative to history.

    Parameters
    ----------
    df : pd.DataFrame  — raw (unenriched) or already enriched DataFrame

    Returns
    -------
    dict with keys:
        current_implied_pct         : float
        current_spread_pct          : float (HY spread in %)
        current_phase               : str
        pct_of_gfc_peak             : float
        pct_of_covid_peak           : float
        closest_historical_analog   : str
        upside_to_peak              : dict {cycle_name: bps_remaining}
        nearest_peak                : str
        interpretation              : str
        available                   : bool
    """
    if "hy_spread" not in df.columns or df.empty:
        return {
            "available": False,
            "current_implied_pct": np.nan,
            "current_spread_pct": np.nan,
            "current_phase": "Unknown",
            "pct_of_gfc_peak": np.nan,
            "pct_of_covid_peak": np.nan,
            "closest_historical_analog": "Unknown",
            "upside_to_peak": {},
            "nearest_peak": "Unknown",
            "interpretation": "HY spread data not available.",
        }

    enriched = compute_default_cycle(df)
    last = enriched.iloc[-1]

    current_idr = last.get("implied_default_rate", np.nan)
    current_pct = last.get("default_cycle_pct", np.nan)
    current_phase = last.get("default_cycle_phase", "Unknown")
    current_spread_pct = last.get("hy_spread", np.nan) / 100.0  # convert bps-as-% to decimal %

    # Convenience: current HY spread as a plain percentage (e.g., 5.5 = 550bps)
    current_spread_display = last.get("hy_spread", np.nan)  # in % as stored (e.g., 5.5)

    momentum = last.get("default_cycle_momentum_30d", 0.0)
    if np.isnan(momentum):
        momentum = 0.0

    # Pct of historical peaks
    gfc_implied = HISTORICAL_PEAKS["2008-09 GFC"]["actual_peak_pct"]
    covid_implied = HISTORICAL_PEAKS["2020 COVID"]["actual_peak_pct"]

    pct_of_gfc = (current_pct / gfc_implied * 100.0) if not np.isnan(current_pct) else np.nan
    pct_of_covid = (current_pct / covid_implied * 100.0) if not np.isnan(current_pct) else np.nan

    # Upside to each historical peak in bps (spread widening remaining)
    upside_to_peak = {}
    for name, data in HISTORICAL_PEAKS.items():
        peak_spread_pct = data["spread_at_peak"]  # e.g., 19.0 for GFC
        current_sp = current_spread_display        # e.g., 5.5
        if not np.isnan(current_sp) and current_sp < peak_spread_pct:
            # Convert from % to bps: (peak - current) * 100
            upside_bps = (peak_spread_pct - current_sp) * 100.0
        else:
            upside_bps = 0.0
        upside_to_peak[name] = upside_bps

    # Nearest peak by implied default rate proximity
    nearest_peak = "Unknown"
    nearest_dist = float("inf")
    for name, data in HISTORICAL_PEAKS.items():
        # Implied default rate for each peak using same formula
        peak_idr_pct = data["actual_peak_pct"]
        dist = abs(current_pct - peak_idr_pct) if not np.isnan(current_pct) else float("inf")
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_peak = name

    closest_analog = _closest_historical_analog(current_spread_display, momentum)

    # Build plain-English interpretation
    gfc_upside = upside_to_peak.get("2008-09 GFC", 0.0)
    analog_data = HISTORICAL_PEAKS.get(closest_analog, {})
    analog_recovery = analog_data.get("recovery_months", "?")

    direction_str = "widening" if momentum > 0 else "tightening" if momentum < 0 else "flat"
    interpretation_parts = [
        f"Implied defaults at {current_pct:.1f}% — consistent with {closest_analog} conditions.",
        f"Current regime: {current_phase}.",
    ]
    if gfc_upside > 0:
        interpretation_parts.append(
            f"If this evolves like the GFC, spreads could widen a further "
            f"{gfc_upside:.0f}bps from current levels."
        )
    interpretation_parts.append(
        f"Spreads are currently {direction_str}. "
        f"Analog cycle ({closest_analog}) recovered in ~{analog_recovery} months."
    )
    interpretation = " ".join(interpretation_parts)

    return {
        "available": True,
        "current_implied_pct": current_pct,
        "current_spread_pct": current_spread_display,
        "current_phase": current_phase,
        "pct_of_gfc_peak": pct_of_gfc,
        "pct_of_covid_peak": pct_of_covid,
        "closest_historical_analog": closest_analog,
        "upside_to_peak": upside_to_peak,
        "nearest_peak": nearest_peak,
        "interpretation": interpretation,
    }


def run_default_cycle_analysis(df: pd.DataFrame) -> dict:
    """
    Full default cycle analysis packaged for dashboard consumption.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys:
        "df"               : enriched DataFrame (with default_cycle_* columns)
        "current"          : dict from get_cycle_position
        "historical_peaks" : HISTORICAL_PEAKS dict
        "cycle_comparison" : pd.DataFrame comparing current to each peak
        "time_series"      : pd.DataFrame — last 252 rows of default_cycle_pct
                             and implied_default_rate
        "available"        : bool
    """
    if "hy_spread" not in df.columns or df.empty:
        return {
            "df": df,
            "current": get_cycle_position(df),
            "historical_peaks": HISTORICAL_PEAKS,
            "cycle_comparison": pd.DataFrame(),
            "time_series": pd.DataFrame(),
            "available": False,
        }

    enriched = compute_default_cycle(df)
    current = get_cycle_position(df)

    current_spread = current["current_spread_pct"]   # e.g., 5.5 (in %)
    current_idr_pct = current["current_implied_pct"]  # e.g., 9.2 (in %)

    # Build cycle comparison DataFrame
    comparison_rows = []
    for cycle_name, data in HISTORICAL_PEAKS.items():
        peak_spread = data["spread_at_peak"]      # in %
        actual_peak = data["actual_peak_pct"]     # in %
        upside_bps = current["upside_to_peak"].get(cycle_name, 0.0)

        # Pct of that cycle's peak we are currently at (spread basis)
        pct_of_peak = (
            (current_spread / peak_spread * 100.0)
            if (not np.isnan(current_spread) and peak_spread > 0)
            else np.nan
        )

        comparison_rows.append({
            "cycle_name": cycle_name,
            "actual_peak_pct": actual_peak,
            "spread_at_peak_pct": peak_spread,
            "spread_at_peak_bps": peak_spread * 100.0,
            "current_spread_pct": current_spread,
            "current_implied_pct": current_idr_pct,
            "pct_of_peak": pct_of_peak,
            "upside_bps": upside_bps,
            "recovery_months": data["recovery_months"],
        })

    cycle_comparison = pd.DataFrame(comparison_rows).set_index("cycle_name")

    # Time series slice (last 252 trading days)
    ts_cols = [c for c in ["default_cycle_pct", "implied_default_rate",
                            "default_cycle_phase", "default_cycle_percentile",
                            "default_cycle_momentum_30d"]
               if c in enriched.columns]
    if "date" in enriched.columns:
        ts_cols = ["date"] + ts_cols
    time_series = enriched[ts_cols].tail(252).copy()

    return {
        "df": enriched,
        "current": current,
        "historical_peaks": HISTORICAL_PEAKS,
        "cycle_comparison": cycle_comparison,
        "time_series": time_series,
        "available": True,
    }
