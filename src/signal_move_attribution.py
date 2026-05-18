"""
Signal move attribution engine.

Answers: "What drove the composite score change over the past month?"

Uses a Brinson-style decomposition — identifies which sub-signal moved the
most, what market variable caused it, and how today's composite compares to
1 month / 3 months / 6 months ago.

Public API
----------
  SUB_SIGNALS          — sub-signal specs with market driver columns
  COMPOSITE            — composite score column name
  LOOKBACK_WINDOWS     — {"1M": 21, "3M": 63, "6M": 126}
  attribute_signal_move  — single-window attribution dict
  run_signal_move_attribution — top-level orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUB_SIGNALS: list[tuple[str, str, list[str]]] = [
    ("treasury_stress_score_smooth",         "Treasury Stress",   ["yield_10y", "yield_3m"]),
    ("credit_market_risk_score_smooth",      "Credit Risk",       ["hy_spread", "ig_spread"]),
    ("macro_risk_score_smooth",              "Macro Risk",        ["sp500", "vix"]),
    ("enhanced_funding_stress_score_smooth", "Funding Stress",    ["yield_3m", "fed_funds_rate"]),
    ("rates_stress_score_smooth",            "Rates Stress",      ["yield_10y", "yield_3m"]),
    ("fx_commodity_score_smooth",            "FX/Commodity",      ["dxy"]),
    ("banking_stress_score_smooth",          "Banking Stress",    ["vix", "hy_spread"]),
]

COMPOSITE = "composite_risk_score_smooth"

LOOKBACK_WINDOWS: dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
}

_MIN_ROWS   = 126
_MIN_SIGS   = 2
_NORM_WINDOW = 252   # window for historical std normalization of market drivers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> float | None:
    """Return float or None for NaN/infinite values."""
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _detect_regime(composite_val: float) -> str:
    """Map composite score to regime label (matches composite_engine thresholds)."""
    if composite_val >= 70:
        return "Risk-Off"
    elif composite_val >= 50:
        return "Caution"
    elif composite_val >= 25:
        return "Neutral"
    return "Risk-On"


def _normalize_driver_delta(
    col: str,
    delta: float,
    df: pd.DataFrame,
    window: int = _NORM_WINDOW,
) -> float:
    """
    Normalize a driver delta by the column's rolling historical std.
    Returns abs(delta / std) — a z-score-like magnitude for comparison.
    Falls back to abs(delta) if std is near zero or column is missing.
    """
    try:
        series = df[col].dropna()
        if len(series) < 10:
            return abs(delta)
        daily_changes = series.diff().dropna()
        std = float(daily_changes.rolling(window, min_periods=20).std().iloc[-1])
        if std < 1e-9:
            return abs(delta)
        return abs(delta / std)
    except Exception:
        return abs(delta)


# ---------------------------------------------------------------------------
# Core attribution function
# ---------------------------------------------------------------------------

def attribute_signal_move(df: pd.DataFrame, window: int) -> dict:
    """
    Attribute composite score change over `window` trading days.

    Parameters
    ----------
    df     : Master DataFrame with DatetimeIndex.
    window : Look-back in trading days.

    Returns
    -------
    {
        window_days: int,
        composite_start: float,
        composite_end: float,
        composite_delta: float,
        direction: str,          # "Deteriorated" / "Improved" / "Stable"
        sub_signal_contributions: list[dict],  # sorted by abs(contribution_pct) desc
        dominant_sub_signal: str,
        dominant_market_driver: str,
        available: bool,
    }
    """
    _fail = {
        "available": False,
        "window_days": window,
        "composite_start": None,
        "composite_end": None,
        "composite_delta": None,
        "direction": None,
        "sub_signal_contributions": [],
        "dominant_sub_signal": None,
        "dominant_market_driver": None,
    }

    try:
        if COMPOSITE not in df.columns:
            return _fail

        n = len(df)
        if n <= window:
            return _fail

        now_idx  = n - 1
        past_idx = now_idx - window

        comp_end   = _safe_float(df[COMPOSITE].iloc[now_idx])
        comp_start = _safe_float(df[COMPOSITE].iloc[past_idx])

        if comp_end is None or comp_start is None:
            return _fail

        comp_delta = comp_end - comp_start

        if abs(comp_delta) < 0.05:
            direction = "Stable"
        elif comp_delta > 0:
            direction = "Deteriorated"
        else:
            direction = "Improved"

        # Identify sub-signals present in df
        available_subs = [
            (col, label, drivers)
            for col, label, drivers in SUB_SIGNALS
            if col in df.columns
        ]

        if len(available_subs) < _MIN_SIGS:
            return _fail

        # Compute sub-signal deltas
        sub_deltas: list[tuple[str, str, list[str], float, float, float]] = []
        for col, label, drivers in available_subs:
            end_val   = _safe_float(df[col].iloc[now_idx])
            start_val = _safe_float(df[col].iloc[past_idx])
            if end_val is None or start_val is None:
                continue
            delta = end_val - start_val
            sub_deltas.append((col, label, drivers, start_val, end_val, delta))

        if not sub_deltas:
            return _fail

        # Attribution weights: share of absolute total movement
        total_abs = sum(abs(s[5]) for s in sub_deltas)

        contributions: list[dict] = []
        driver_normalized_moves: dict[str, float] = {}

        for col, label, drivers, start_val, end_val, delta in sub_deltas:
            if total_abs > 1e-9:
                contribution_pct = (delta / total_abs) * 100.0
            else:
                contribution_pct = 0.0

            # Evaluate market drivers for this sub-signal
            driver_candidates: list[tuple[str, float, float]] = []
            for drv_col in drivers:
                if drv_col not in df.columns:
                    continue
                drv_end   = _safe_float(df[drv_col].iloc[now_idx])
                drv_start = _safe_float(df[drv_col].iloc[past_idx])
                if drv_end is None or drv_start is None:
                    continue
                drv_delta     = drv_end - drv_start
                drv_norm      = _normalize_driver_delta(drv_col, drv_delta, df)
                driver_candidates.append((drv_col, drv_delta, drv_norm))
                # Track globally for dominant driver determination
                if drv_col not in driver_normalized_moves or drv_norm > driver_normalized_moves[drv_col]:
                    driver_normalized_moves[drv_col] = drv_norm

            # Top driver for this sub-signal (largest normalized abs move)
            if driver_candidates:
                top_drv = max(driver_candidates, key=lambda x: x[2])
                top_driver_name  = top_drv[0]
                top_driver_delta = round(top_drv[1], 4)
            else:
                top_driver_name  = None
                top_driver_delta = None

            contributions.append({
                "name":              col,
                "label":             label,
                "start":             round(start_val, 2),
                "end":               round(end_val, 2),
                "delta":             round(delta, 2),
                "contribution_pct":  round(contribution_pct, 1),
                "top_driver":        top_driver_name,
                "top_driver_delta":  top_driver_delta,
            })

        # Sort by abs(contribution_pct) descending
        contributions.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)

        dominant_sub  = contributions[0]["label"] if contributions else None
        dominant_drv  = (
            max(driver_normalized_moves, key=driver_normalized_moves.get)
            if driver_normalized_moves
            else None
        )

        return {
            "available":               True,
            "window_days":             window,
            "composite_start":         round(comp_start, 2),
            "composite_end":           round(comp_end, 2),
            "composite_delta":         round(comp_delta, 2),
            "direction":               direction,
            "sub_signal_contributions": contributions,
            "dominant_sub_signal":     dominant_sub,
            "dominant_market_driver":  dominant_drv,
        }

    except Exception:
        return _fail


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_signal_move_attribution(df: pd.DataFrame) -> dict:
    """
    Top-level entry point.  Requires the composite column and >= 2 sub-signals
    present in df, and at least 126 rows of history.

    Returns
    -------
    {
        available: bool,
        attributions: {"1M": dict, "3M": dict, "6M": dict},
        current_composite: float,
        current_regime: str,
        summary_table: pd.DataFrame,
        interpretation: str,
    }
    """
    _fail = {
        "available": False,
        "attributions": {},
        "current_composite": None,
        "current_regime": None,
        "summary_table": pd.DataFrame(),
        "interpretation": "Insufficient data for signal move attribution.",
    }

    try:
        if df is None or df.empty:
            return _fail

        if COMPOSITE not in df.columns:
            return _fail

        if len(df) < _MIN_ROWS:
            return _fail

        # Count how many sub-signals are present
        present_subs = [col for col, _, _ in SUB_SIGNALS if col in df.columns]
        if len(present_subs) < _MIN_SIGS:
            return _fail

        # Current composite and regime
        current_comp = _safe_float(df[COMPOSITE].iloc[-1])
        if current_comp is None:
            return _fail

        current_regime = _detect_regime(current_comp)

        # Run attribution for each window
        attributions: dict[str, dict] = {}
        for label, days in LOOKBACK_WINDOWS.items():
            attributions[label] = attribute_signal_move(df, days)

        # Build summary table: sub-signal deltas across all windows (wide format)
        sub_labels = [label for _, label, _ in SUB_SIGNALS if _ or True]  # all labels
        label_map  = {col: lbl for col, lbl, _ in SUB_SIGNALS}
        rows: list[dict] = []

        for col, sig_label, _ in SUB_SIGNALS:
            if col not in df.columns:
                continue
            row: dict = {"Sub-Signal": sig_label}
            for win_label, days in LOOKBACK_WINDOWS.items():
                attr = attributions.get(win_label, {})
                if not attr.get("available"):
                    row[f"Delta ({win_label})"]       = None
                    row[f"Contribution % ({win_label})"] = None
                    continue
                match = next(
                    (s for s in attr["sub_signal_contributions"] if s["name"] == col),
                    None,
                )
                if match:
                    row[f"Delta ({win_label})"]          = match["delta"]
                    row[f"Contribution % ({win_label})"] = match["contribution_pct"]
                else:
                    row[f"Delta ({win_label})"]          = None
                    row[f"Contribution % ({win_label})"] = None
            rows.append(row)

        summary_table = pd.DataFrame(rows).set_index("Sub-Signal") if rows else pd.DataFrame()

        # Narrative interpretation based on 1M window
        interpretation = _build_interpretation(
            attributions.get("1M", {}),
            current_comp,
            current_regime,
        )

        return {
            "available":        True,
            "attributions":     attributions,
            "current_composite": round(current_comp, 2),
            "current_regime":   current_regime,
            "summary_table":    summary_table,
            "interpretation":   interpretation,
        }

    except Exception:
        return _fail


def _build_interpretation(attr_1m: dict, current_comp: float, regime: str) -> str:
    """Build a plain-English narrative for the 1M attribution."""
    try:
        if not attr_1m.get("available"):
            return "Insufficient data for 1-month attribution narrative."

        delta     = attr_1m["composite_delta"]
        direction = attr_1m["direction"]
        dominant  = attr_1m["dominant_sub_signal"]
        driver    = attr_1m["dominant_market_driver"]
        comp_end  = attr_1m["composite_end"]

        direction_phrase = {
            "Deteriorated": "deteriorated",
            "Improved":     "improved",
            "Stable":       "remained broadly stable",
        }.get(direction, "changed")

        parts = [
            f"Over the past month, the composite risk score {direction_phrase} "
            f"by {abs(delta):.1f} points to {comp_end:.1f} (regime: {regime})."
        ]

        if dominant:
            parts.append(f"The largest contributor was {dominant}.")

        if driver:
            parts.append(f"The primary market driver was {driver}.")

        # Add sub-signal detail for top 2 contributors
        contribs = attr_1m.get("sub_signal_contributions", [])
        top2 = [c for c in contribs[:2] if abs(c.get("contribution_pct", 0)) >= 5]
        if top2:
            detail_parts = []
            for c in top2:
                sign = "+" if c["delta"] >= 0 else ""
                detail_parts.append(
                    f"{c['label']} ({sign}{c['delta']:.1f} pts, "
                    f"{c['contribution_pct']:+.0f}% of move)"
                )
            parts.append("Key movers: " + "; ".join(detail_parts) + ".")

        return " ".join(parts)

    except Exception:
        return "Attribution narrative unavailable."
