"""
Credit market tear sheet.

Builds a compact table for interview/demo use: current level, percentile,
recent changes, valuation, and action for key credit metrics.
"""

from __future__ import annotations

import pandas as pd

from src.credit_relative_value import compute_credit_relative_value
from src.spread_decomposition import decompose_spreads


def _change(series: pd.Series, days: int):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= days:
        return None
    return float(s.iloc[-1] - s.iloc[-days - 1])


def _latest(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _action(metric: str, valuation: str | None, change_1m) -> str:
    if metric == "HY OAS":
        if valuation in {"Crisis / Very Cheap", "Cheap"} and change_1m is not None and change_1m <= 0:
            return "Consider adding HY beta selectively."
        if valuation in {"Rich", "Very Rich"}:
            return "Avoid chasing HY beta."
        if change_1m is not None and change_1m > 25:
            return "Monitor widening; reduce lower-quality exposure."
    if metric == "Excess Spread" and valuation in {"Rich", "Very Rich"}:
        return "Spread compensation is thin after expected loss."
    return "Maintain discipline; use regime and technical confirmation."


def credit_market_tearsheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rv = compute_credit_relative_value(df)
    decomp = decompose_spreads(df)
    rows = []

    def add(metric, level_col, pct_col=None, valuation_col=None, source=None):
        src = source if source is not None else rv
        if src.empty or level_col not in src.columns:
            return
        level = _latest(src[level_col])
        pct = _latest(src[pct_col]) if pct_col and pct_col in src.columns else None
        valuation = None
        if valuation_col and valuation_col in src.columns:
            vals = src[valuation_col].dropna()
            valuation = vals.iloc[-1] if not vals.empty else None
        chg_1m = _change(src[level_col], 21)
        chg_3m = _change(src[level_col], 63)
        rows.append(
            {
                "metric": metric,
                "level": level,
                "percentile": pct,
                "change_1m": chg_1m,
                "change_3m": chg_3m,
                "valuation": valuation,
                "action": _action(metric, valuation, chg_1m),
            }
        )

    add("HY OAS", "hy_spread_bps", "hy_spread_percentile", "hy_spread_valuation")
    add("IG OAS", "ig_spread_bps", "ig_spread_percentile", "ig_spread_valuation")
    add("HY / IG Ratio", "hy_ig_spread_ratio", "hy_ig_ratio_percentile")
    add("BBB / IG Ratio", "bbb_ig_ratio", "bbb_ig_ratio_percentile")
    if not decomp.empty:
        excess_src = decomp.copy()
        if "excess_spread_percentile" in rv.columns:
            excess_src["excess_spread_percentile"] = rv["excess_spread_percentile"]
        if "excess_spread_valuation" in rv.columns:
            excess_src["excess_spread_valuation"] = rv["excess_spread_valuation"]
        add("Excess Spread", "excess_spread_bps", "excess_spread_percentile", "excess_spread_valuation", excess_src)

    return pd.DataFrame(rows)


def credit_tearsheet_markdown(df: pd.DataFrame) -> str:
    table = credit_market_tearsheet(df)
    if table.empty:
        return "Credit Market Tear Sheet\n\nNo credit tear sheet data available."

    lines = ["# Credit Market Tear Sheet", ""]
    for row in table.itertuples():
        level = "n/a" if pd.isna(row.level) else f"{row.level:.1f}"
        pct = "n/a" if pd.isna(row.percentile) else f"{row.percentile:.0f}"
        chg_1m = "n/a" if pd.isna(row.change_1m) else f"{row.change_1m:+.1f}"
        lines.append(f"- **{row.metric}:** {level}; pctile {pct}; 1M change {chg_1m}; {row.action}")
    return "\n".join(lines)
