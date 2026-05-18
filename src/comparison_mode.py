"""
Signal comparison mode — side-by-side diff of dashboard state at two dates.

Allows users to compare the full signal state at any two dates: "what changed
since last month's Fed meeting?" or "how does today compare to the 2022 peak?"

Returns a structured comparison showing:
  - Signal scores at date A vs date B
  - Delta (B - A) for each signal
  - Regime change (if any)
  - Spread level changes
  - Which signals deteriorated / improved most

Public API
----------
  COMPARISON_SIGNALS  — ordered list of signals to compare
  COMPARISON_MARKET   — market/macro columns to compare
  get_available_dates(df) → list[str]   (unique dates as YYYY-MM-DD strings)
  compare_dates(df, date_a, date_b) → dict
  format_comparison_table(comparison) → pd.DataFrame   (display-ready)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level signal definitions
# ---------------------------------------------------------------------------

COMPARISON_SIGNALS = [
    ("composite_risk_score_smooth",             "Composite Risk Score"),
    ("treasury_stress_score_smooth",            "Treasury Stress"),
    ("credit_market_risk_score_smooth",         "Credit Risk"),
    ("macro_risk_score_smooth",                 "Macro Risk"),
    ("enhanced_funding_stress_score_smooth",    "Funding Stress"),
    ("rates_stress_score_smooth",               "Rates Stress"),
    ("fx_commodity_score_smooth",               "FX/Commodity"),
    ("banking_stress_score_smooth",             "Banking Stress"),
    ("complacency_score_smooth",                "Complacency"),
    ("liquidity_regime_score_smooth",           "Liquidity Regime"),
]

COMPARISON_MARKET = [
    ("hy_spread",      "HY Spread (pp)"),
    ("ig_spread",      "IG Spread (pp)"),
    ("vix",            "VIX"),
    ("yield_10y",      "10y Treasury Yield"),
    ("yield_3m",       "3m Treasury Yield"),
    ("fed_funds_rate", "Fed Funds Rate"),
    ("sp500",          "S&P 500"),
]


# ---------------------------------------------------------------------------
# get_available_dates
# ---------------------------------------------------------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with 'date' as a plain column (from column or DatetimeIndex)."""
    df = df.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df.index, errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_available_dates(df: pd.DataFrame) -> list:
    """Return sorted list of unique date strings (YYYY-MM-DD) with valid data.

    Only dates where ``composite_risk_score_smooth`` is non-NaN are included,
    which filters out non-business days / weekends that have no data.
    """
    if df is None or df.empty:
        return []

    dates = _normalize(df)

    if "composite_risk_score_smooth" in dates.columns:
        dates = dates[dates["composite_risk_score_smooth"].notna()]

    dates = dates.dropna(subset=["date"])
    unique_dates = sorted(dates["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    return unique_dates


# ---------------------------------------------------------------------------
# compare_dates
# ---------------------------------------------------------------------------

def _find_closest_row(df: pd.DataFrame, date_str: str) -> pd.Series:
    """Return the row whose date is closest to ``date_str``. Requires 'date' column."""
    target = pd.Timestamp(date_str)
    idx = (df["date"] - target).abs().argmin()
    return df.iloc[idx]


def _signal_direction(delta: float) -> str:
    """For risk scores: lower is better. Classify the direction of change."""
    if pd.isna(delta):
        return "Stable"
    if delta < -2:
        return "Improved"
    if delta > 2:
        return "Deteriorated"
    return "Stable"


def compare_dates(df: pd.DataFrame, date_a: str, date_b: str) -> dict:
    """Compare dashboard signal state between two dates.

    Parameters
    ----------
    df : pd.DataFrame
        Full history DataFrame.
    date_a : str
        ISO date string "YYYY-MM-DD" for the baseline date.
    date_b : str
        ISO date string "YYYY-MM-DD" for the comparison date.

    Returns
    -------
    dict with keys: date_a, date_b, actual_date_a, actual_date_b,
        signals, market, regime_a, regime_b, regime_changed,
        biggest_movers, risk_summary, composite_delta, available.
    """
    if df is None or df.empty:
        return {"available": False}

    work = _normalize(df).dropna(subset=["date"]).reset_index(drop=True)

    if work.empty:
        return {"available": False}

    try:
        row_a = _find_closest_row(work, date_a)
        row_b = _find_closest_row(work, date_b)
    except Exception:
        return {"available": False}

    actual_date_a = row_a["date"].strftime("%Y-%m-%d")
    actual_date_b = row_b["date"].strftime("%Y-%m-%d")

    # --- signals ------------------------------------------------------------
    signal_records = []
    for col, label in COMPARISON_SIGNALS:
        if col in work.columns:
            va = row_a[col] if pd.notna(row_a[col]) else None
            vb = row_b[col] if pd.notna(row_b[col]) else None
            if va is not None and vb is not None:
                delta = float(vb) - float(va)
                direction = _signal_direction(delta)
            else:
                delta = None
                direction = "Stable"
        else:
            va, vb, delta, direction = None, None, None, "Stable"

        signal_records.append({
            "name": col,
            "label": label,
            "val_a": round(float(va), 2) if va is not None else None,
            "val_b": round(float(vb), 2) if vb is not None else None,
            "delta": round(delta, 2) if delta is not None else None,
            "direction": direction,
        })

    # --- market columns -----------------------------------------------------
    market_records = []
    for col, label in COMPARISON_MARKET:
        if col in work.columns:
            va = row_a[col] if pd.notna(row_a[col]) else None
            vb = row_b[col] if pd.notna(row_b[col]) else None
            if va is not None and vb is not None:
                delta = float(vb) - float(va)
                pct_change = (delta / float(va) * 100) if float(va) != 0 else None
            else:
                delta, pct_change = None, None
        else:
            va, vb, delta, pct_change = None, None, None, None

        market_records.append({
            "name": col,
            "label": label,
            "val_a": round(float(va), 2) if va is not None else None,
            "val_b": round(float(vb), 2) if vb is not None else None,
            "delta": round(delta, 2) if delta is not None else None,
            "pct_change": round(pct_change, 2) if pct_change is not None else None,
        })

    # --- regime comparison --------------------------------------------------
    regime_col = "final_decision"
    if regime_col in work.columns:
        regime_a = str(row_a[regime_col]) if pd.notna(row_a[regime_col]) else "Unknown"
        regime_b = str(row_b[regime_col]) if pd.notna(row_b[regime_col]) else "Unknown"
    else:
        regime_a = "Unknown"
        regime_b = "Unknown"
    regime_changed = regime_a != regime_b

    # --- biggest movers (top 3 by abs(delta)) --------------------------------
    scored = [s for s in signal_records if s["delta"] is not None]
    scored_sorted = sorted(scored, key=lambda x: abs(x["delta"]), reverse=True)
    biggest_movers = [
        {"label": s["label"], "delta": s["delta"], "direction": s["direction"]}
        for s in scored_sorted[:3]
    ]

    # --- risk summary -------------------------------------------------------
    composite_delta = None
    composite_rec = next(
        (s for s in signal_records if s["name"] == "composite_risk_score_smooth"), None
    )
    if composite_rec and composite_rec["delta"] is not None:
        composite_delta = composite_rec["delta"]
        if composite_delta < -2:
            risk_summary = "Risk decreased"
        elif composite_delta > 2:
            risk_summary = "Risk increased"
        else:
            risk_summary = "Mixed"
    else:
        # Fall back to counting signal directions
        n_improved = sum(1 for s in signal_records if s["direction"] == "Improved")
        n_deteriorated = sum(1 for s in signal_records if s["direction"] == "Deteriorated")
        if n_improved > n_deteriorated:
            risk_summary = "Risk decreased"
        elif n_deteriorated > n_improved:
            risk_summary = "Risk increased"
        else:
            risk_summary = "Mixed"

    return {
        "date_a": date_a,
        "date_b": date_b,
        "actual_date_a": actual_date_a,
        "actual_date_b": actual_date_b,
        "signals": signal_records,
        "market": market_records,
        "regime_a": regime_a,
        "regime_b": regime_b,
        "regime_changed": regime_changed,
        "biggest_movers": biggest_movers,
        "risk_summary": risk_summary,
        "composite_delta": composite_delta,
        "available": True,
    }


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------

def format_comparison_table(comparison: dict) -> pd.DataFrame:
    """Convert a comparison dict into a clean DataFrame for st.dataframe().

    Columns: Signal | Date A Value | Date B Value | Change | Direction

    Signals section appears first, followed by a separator row, then market.
    All values are rounded to 1 decimal place and expressed as strings.
    """
    if not comparison.get("available", False):
        return pd.DataFrame(
            columns=["Signal", "Date A Value", "Date B Value", "Change", "Direction"]
        )

    date_a_label = comparison.get("actual_date_a", "Date A")
    date_b_label = comparison.get("actual_date_b", "Date B")

    columns = ["Signal", date_a_label, date_b_label, "Change", "Direction"]

    def _fmt(val) -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val):.1f}"
        except (TypeError, ValueError):
            return str(val)

    def _fmt_delta(val) -> str:
        if val is None:
            return "—"
        try:
            f = float(val)
            return f"+{f:.1f}" if f > 0 else f"{f:.1f}"
        except (TypeError, ValueError):
            return str(val)

    rows = []

    # --- signal rows --------------------------------------------------------
    for s in comparison.get("signals", []):
        rows.append([
            s["label"],
            _fmt(s["val_a"]),
            _fmt(s["val_b"]),
            _fmt_delta(s["delta"]),
            s["direction"],
        ])

    # --- separator ----------------------------------------------------------
    rows.append(["— Market & Macro —", "", "", "", ""])

    # --- market rows --------------------------------------------------------
    for m in comparison.get("market", []):
        rows.append([
            m["label"],
            _fmt(m["val_a"]),
            _fmt(m["val_b"]),
            _fmt_delta(m["delta"]),
            "",  # no Improved/Deteriorated for market cols
        ])

    return pd.DataFrame(rows, columns=columns)
