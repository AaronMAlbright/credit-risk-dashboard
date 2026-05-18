"""
Signal Traffic Light — per-signal color classification and direction tracking.

Each signal in TRAFFIC_SIGNALS is classified Green/Yellow/Orange/Red based on
its current value's historical percentile rank, with a 1-month direction arrow.

Public API
----------
  TRAFFIC_SIGNALS         — module-level registry of (col, label, invert)
  classify_signal(series, current_val, invert) -> dict
  compute_traffic_light(df) -> pd.DataFrame
  run_traffic_light_analysis(df) -> dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRAFFIC_SIGNALS: list[tuple[str, str, bool]] = [
    # (column_name, label, invert)
    # invert=True means HIGHER values are better (e.g., price going up = good)
    ("composite_risk_score_smooth",          "Composite Risk",   False),
    ("treasury_stress_score_smooth",         "Treasury Stress",  False),
    ("credit_market_risk_score_smooth",      "Credit Risk",      False),
    ("macro_risk_score_smooth",              "Macro Risk",       False),
    ("enhanced_funding_stress_score_smooth", "Funding Stress",   False),
    ("rates_stress_score_smooth",            "Rates Stress",     False),
    ("fx_commodity_score_smooth",            "FX/Commodity",     False),
    ("banking_stress_score_smooth",          "Banking Stress",   False),
    ("complacency_score_smooth",             "Complacency",      True),
    ("liquidity_regime_score_smooth",        "Liquidity Regime", False),
    ("hy_spread",                            "HY Spread",        False),
    ("ig_spread",                            "IG Spread",        False),
    ("vix",                                  "VIX",              False),
    ("yield_10y",                            "10y Yield",        False),
    ("yield_3m",                             "3m Yield",         False),
    ("sp500",                                "S&P 500",          True),
]

# Color ordering for modal / override logic
_COLOR_ORDER = ["Green", "Yellow", "Orange", "Red"]

# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_signal(series: pd.Series, current_val: float, invert: bool) -> dict:
    """
    Classify a single signal into a color based on historical percentile rank.

    Parameters
    ----------
    series      : full historical series (used to compute percentile ranks)
    current_val : the value to classify (typically the last observation)
    invert      : if True, higher values are better (Green at high percentile)

    Returns
    -------
    dict with keys: color, percentile, direction
    """
    try:
        clean = series.dropna()
        if len(clean) < 2:
            return {"color": "Yellow", "percentile": 50.0, "direction": "→"}

        # Percentile rank of current_val within the full history (0–100)
        pct = float((clean < current_val).sum()) / len(clean) * 100.0

        # Color thresholds
        if invert:
            # Higher is better
            if pct >= 70:
                color = "Green"
            elif pct >= 40:
                color = "Yellow"
            elif pct >= 20:
                color = "Orange"
            else:
                color = "Red"
        else:
            # Higher is worse (risk signal)
            if pct < 30:
                color = "Green"
            elif pct < 60:
                color = "Yellow"
            elif pct < 80:
                color = "Orange"
            else:
                color = "Red"

        # Direction: 21-day change vs 1% of historical std
        hist_std = float(clean.std(ddof=1))
        threshold = 0.01 * hist_std if hist_std > 0 else 0.0

        if len(clean) >= 22:
            older_val = float(clean.iloc[-22])
            change = current_val - older_val
        elif len(clean) >= 2:
            older_val = float(clean.iloc[0])
            change = current_val - older_val
        else:
            change = 0.0

        if change > threshold:
            direction = "↑"
        elif change < -threshold:
            direction = "↓"
        else:
            direction = "→"

        return {"color": color, "percentile": round(pct, 1), "direction": direction}

    except Exception:
        return {"color": "Yellow", "percentile": 50.0, "direction": "→"}


# ---------------------------------------------------------------------------
# compute_traffic_light
# ---------------------------------------------------------------------------

def compute_traffic_light(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify every signal in TRAFFIC_SIGNALS against its full history.

    Returns a DataFrame with one row per signal:
      name, label, color, percentile, current_val, direction, available
    """
    rows = []
    for col, label, invert in TRAFFIC_SIGNALS:
        row_base = {
            "name":        col,
            "label":       label,
            "color":       "Yellow",
            "percentile":  50.0,
            "current_val": np.nan,
            "direction":   "→",
            "available":   False,
        }
        try:
            if col not in df.columns:
                rows.append(row_base)
                continue

            series = df[col].dropna()
            if len(series) < 2:
                rows.append(row_base)
                continue

            current_val = float(series.iloc[-1])
            classification = classify_signal(series, current_val, invert)

            rows.append({
                "name":        col,
                "label":       label,
                "color":       classification["color"],
                "percentile":  classification["percentile"],
                "current_val": round(current_val, 4),
                "direction":   classification["direction"],
                "available":   True,
            })
        except Exception:
            rows.append(row_base)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _modal_color(colors: pd.Series) -> str:
    """Return the most common color, breaking ties toward higher severity."""
    if colors.empty:
        return "Yellow"
    counts = colors.value_counts()
    # Among tied counts, prefer higher severity
    max_count = counts.max()
    tied = [c for c in _COLOR_ORDER if counts.get(c, 0) == max_count]
    # Return the highest severity among tied
    return tied[-1] if tied else "Yellow"


def _composite_color(signals_df: pd.DataFrame) -> str:
    """
    Overall color: use composite_risk_score_smooth if available,
    else fall back to modal color of all available signals.
    """
    composite_rows = signals_df[
        (signals_df["name"] == "composite_risk_score_smooth") &
        (signals_df["available"])
    ]
    if not composite_rows.empty:
        return str(composite_rows.iloc[0]["color"])

    available = signals_df[signals_df["available"]]
    if available.empty:
        return "Yellow"
    return _modal_color(available["color"])


# ---------------------------------------------------------------------------
# run_traffic_light_analysis
# ---------------------------------------------------------------------------

def run_traffic_light_analysis(df: pd.DataFrame) -> dict:
    """
    Top-level entry point for the signal traffic light module.

    Returns
    -------
    {
        available    : bool,
        signals      : pd.DataFrame,
        summary      : {green, yellow, orange, red, total},
        overall_color: str,
        last_updated : str,
    }
    """
    try:
        signals_df = compute_traffic_light(df)

        available_df = signals_df[signals_df["available"]]

        summary = {
            "green":  int((available_df["color"] == "Green").sum()),
            "yellow": int((available_df["color"] == "Yellow").sum()),
            "orange": int((available_df["color"] == "Orange").sum()),
            "red":    int((available_df["color"] == "Red").sum()),
            "total":  len(available_df),
        }

        overall_color = _composite_color(signals_df)

        # Last updated: ISO date of the last row in the DataFrame index
        try:
            last_updated = str(df.index[-1].date()) if len(df) > 0 else ""
        except Exception:
            last_updated = ""

        return {
            "available":    summary["total"] > 0,
            "signals":      signals_df,
            "summary":      summary,
            "overall_color": overall_color,
            "last_updated": last_updated,
        }

    except Exception:
        return {"available": False}
