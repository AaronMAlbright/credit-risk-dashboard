"""
Signal health monitor — meta-view of all signal modules.

Checks which modules are available, which data sources loaded successfully,
how stale each source is, and which are running on synthetic/fallback data.

Public API
----------
  MODULE_REGISTRY  — ordered list of (module_id, display_name, required_cols, category)
  check_signal_health(df) -> dict
  get_health_summary(df) -> pd.DataFrame   (display-ready)
"""

from __future__ import annotations

from datetime import timezone
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODULE_REGISTRY: list[tuple[str, str, list[str], str]] = [
    # id, display_name, required_columns, category
    ("hy_spread",        "HY Spread",             ["hy_spread"],                        "Credit"),
    ("ig_spread",        "IG Spread",              ["ig_spread"],                        "Credit"),
    ("vix",              "VIX",                    ["vix"],                              "Volatility"),
    ("move",             "MOVE Index",             [],                                   "Volatility"),  # live fetch
    ("yield_curve",      "Yield Curve",            ["yield_10y", "yield_3m"],            "Rates"),
    ("real_rates",       "Real Rates",             ["yield_10y", "breakeven_10y"],       "Rates"),
    ("taylor_rule",      "Taylor Rule",            ["dff", "unemployment"],              "Macro"),
    ("recession_prob",   "Recession Probability",  ["yield_10y", "yield_3m"],            "Macro"),
    ("macro_nowcast",    "Macro Nowcast",           ["unemployment", "sp500"],            "Macro"),
    ("funding_stress",   "Funding Stress",          ["yield_3m", "dff"],                 "Credit"),
    ("fed_sentiment",    "Fed Sentiment",           [],                                  "Macro"),     # live fetch
    ("carry_breakeven",  "Carry Breakeven",         ["hy_yield"],                        "Credit"),
    ("fallen_angel",     "Fallen Angel Risk",       ["hy_spread", "ig_spread"],          "Credit"),
    ("credit_momentum",  "Credit Momentum",         ["hy_spread"],                       "Credit"),
    ("spread_vol",       "Spread Volatility",       ["hy_spread"],                       "Volatility"),
    ("vrp",              "Vol Risk Premium",        ["vix", "sp500"],                    "Volatility"),
    ("vix_term",         "VIX Term Structure",      [],                                  "Volatility"), # live fetch
    ("options_skew",     "Options Skew",            [],                                  "Volatility"), # live fetch
    ("em_credit",        "EM Credit",               [],                                  "Credit"),     # live fetch
    ("global_credit",    "Global Credit",           [],                                  "Credit"),     # live fetch
    ("corp_leverage",    "Corp Leverage",           ["hy_spread"],                       "Macro"),
    ("seasonality",      "Seasonality",             ["hy_spread"],                       "Credit"),
    ("eq_credit_corr",   "EQ-Credit Correlation",   ["sp500", "hy_spread"],              "Credit"),
    ("merton_dd",        "Merton Dist-to-Default",  ["sp500", "vix", "hy_spread"],       "Credit"),
    ("corr_heatmap",     "Correlation Heatmap",     ["hy_spread", "vix", "yield_10y"],   "Risk"),
    ("cvar",             "CVaR by Regime",          ["sp500", "final_decision"],          "Risk"),
    ("default_cycle",    "Default Cycle",           ["hy_spread"],                       "Credit"),
    ("backtester",       "Strategy Backtest",       ["sp500", "final_decision"],          "Backtest"),
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STATUS_ORDER = {"Unavailable": 0, "Degraded": 1, "Live": 2, "OK": 3}
_STALE_THRESHOLD_DAYS = 3


def _staleness_days(df: pd.DataFrame, col: str) -> int | None:
    """Return calendar days since *col* last had a non-NaN value, or None if col absent."""
    if col not in df.columns:
        return None
    last_valid = df[col].last_valid_index()
    if last_valid is None:
        return None
    now = pd.Timestamp.now()
    delta = now - pd.Timestamp(last_valid)
    return int(delta.days)


def _date_range(df: pd.DataFrame) -> str:
    """Return 'YYYY-MM-DD → YYYY-MM-DD' for a DataFrame with a DatetimeIndex."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return "N/A"
    return f"{df.index[0].date()} → {df.index[-1].date()}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_signal_health(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect every module in MODULE_REGISTRY against *df* and return a health dict.

    Parameters
    ----------
    df:
        The full model output DataFrame (one row per date, DatetimeIndex).

    Returns
    -------
    dict with keys: modules, summary, data_rows, date_range, checked_at, overall_health
    """
    modules: list[dict[str, Any]] = []

    for mod_id, display_name, required_cols, category in MODULE_REGISTRY:
        is_live_fetch = len(required_cols) == 0

        cols_present = [c for c in required_cols if c in df.columns] if not df.empty else []
        cols_missing = [c for c in required_cols if c not in df.columns]

        available = is_live_fetch or (len(cols_missing) == 0)

        # Staleness: worst-case (max) staleness across present columns
        staleness_map: dict[str, int | None] = {
            c: _staleness_days(df, c) for c in cols_present
        }
        valid_staleness = [v for v in staleness_map.values() if v is not None]
        max_staleness = max(valid_staleness) if valid_staleness else None

        # Determine status
        if is_live_fetch:
            status = "Live"
        elif not available:
            status = "Unavailable"
        elif max_staleness is not None and max_staleness >= _STALE_THRESHOLD_DAYS:
            status = "Degraded"
        else:
            status = "OK"

        # Last-data string: date of last valid value for first present column
        last_data = "N/A"
        if cols_present and not df.empty:
            first_col = cols_present[0]
            lvi = df[first_col].last_valid_index()
            if lvi is not None:
                last_data = str(pd.Timestamp(lvi).date())

        modules.append(
            {
                "id": mod_id,
                "name": display_name,
                "category": category,
                "required_cols": required_cols,
                "cols_present": cols_present,
                "cols_missing": cols_missing,
                "available": available,
                "is_live_fetch": is_live_fetch,
                "staleness_days": max_staleness,
                "status": status,
                "last_data": last_data,
                "data_rows": len(df),
                "date_range": _date_range(df),
            }
        )

    # Summary counts
    status_counts = {s: sum(1 for m in modules if m["status"] == s) for s in _STATUS_ORDER}
    total = len(modules)
    ok_count = status_counts.get("OK", 0)
    degraded_count = status_counts.get("Degraded", 0)
    unavailable_count = status_counts.get("Unavailable", 0)
    live_count = status_counts.get("Live", 0)

    healthy_frac = (ok_count + live_count) / total if total else 0.0
    acceptable_frac = (ok_count + live_count + degraded_count) / total if total else 0.0

    if healthy_frac >= 0.80:
        overall_health = "Healthy"
    elif acceptable_frac >= 0.60:
        overall_health = "Degraded"
    else:
        overall_health = "Critical"

    return {
        "modules": modules,
        "summary": {
            "total": total,
            "ok": ok_count,
            "degraded": degraded_count,
            "unavailable": unavailable_count,
            "live_fetch": live_count,
        },
        "data_rows": len(df),
        "date_range": _date_range(df),
        "checked_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "overall_health": overall_health,
    }


def get_health_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready DataFrame summarising every module's health.

    Columns: Module | Category | Status | Required Cols | Missing | Last Data

    Rows are sorted: Unavailable first, then Degraded, then Live, then OK.
    """
    health = check_signal_health(df)

    rows = []
    for m in health["modules"]:
        rows.append(
            {
                "Module": m["name"],
                "Category": m["category"],
                "Status": m["status"],
                "Required Cols": ", ".join(m["required_cols"]) if m["required_cols"] else "(live fetch)",
                "Missing": ", ".join(m["cols_missing"]) if m["cols_missing"] else "",
                "Last Data": m["last_data"],
            }
        )

    result = pd.DataFrame(rows)
    result["_sort_key"] = result["Status"].map(_STATUS_ORDER)
    result = result.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    return result
