"""
Presentation helpers for the institutional credit view.

These functions turn the analytical outputs into concise, readable artifacts
for Streamlit, reports, and interview demos.
"""

from __future__ import annotations

import pandas as pd

from src.channel_attribution import top_channel_drivers
from src.credit_positioning import current_positioning
from src.credit_regime_performance import latest_regime_performance_note
from src.credit_relative_value import latest_relative_value_snapshot
from src.spread_decomposition import latest_spread_snapshot


def _fmt_pct(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.1%}"


def _fmt_bps(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.0f} bps"


def _fmt_num(x, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def build_credit_brief(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"available": False, "headline": "No credit data available."}

    latest = df.iloc[-1]
    regime = latest.get("final_decision", "Unknown")
    score = latest.get("composite_risk_score_smooth", latest.get("composite_risk_score"))
    positioning = current_positioning(df)
    spread = latest_spread_snapshot(df)
    rv = latest_relative_value_snapshot(df)
    perf = latest_regime_performance_note(df)
    drivers = top_channel_drivers(df)

    credit_beta = positioning.get("credit_beta", "Market weight")
    quality_bias = positioning.get("quality_bias", "Balanced quality exposure")
    valuation = rv.get("hy_spread_valuation") or spread.get("spread_valuation", "Unavailable")

    headline = f"{regime}: {credit_beta} credit beta, {quality_bias.lower()}."

    evidence = []
    if spread.get("available"):
        evidence.append(
            f"HY OAS {_fmt_bps(spread.get('spread_oas_bps'))}; "
            f"expected loss {_fmt_bps(spread.get('expected_loss_bps'))}; "
            f"excess spread {_fmt_bps(spread.get('excess_spread_bps'))}."
        )
    if rv.get("available"):
        hy_pct = rv.get("hy_spread_percentile")
        if hy_pct is not None:
            evidence.append(f"HY spread percentile is {_fmt_num(hy_pct, 0)} with valuation bucket '{valuation}'.")
        if rv.get("quality_tilt"):
            evidence.append(rv["quality_tilt"])
    if perf.get("available"):
        evidence.append(
            f"Current-regime validation: n={perf['n_obs']} observations, "
            f"{perf['confidence']} confidence over {perf['horizon_days']} trading days."
        )

    watch_items = [
        "HY widening with low VIX would indicate credit-specific stress.",
        "Funding/liquidity deterioration should lower tolerance for HY beta.",
        "Improving spreads plus falling volatility would support adding credit risk.",
    ]

    return {
        "available": True,
        "date": latest.get("date", df.index[-1]),
        "headline": headline,
        "regime": regime,
        "score": None if pd.isna(score) else float(score),
        "credit_beta": credit_beta,
        "hy_tilt": positioning.get("hy_tilt", "Neutral"),
        "ig_tilt": positioning.get("ig_tilt", "Neutral"),
        "duration_stance": positioning.get("duration_stance", "Neutral"),
        "hedge_posture": positioning.get("hedge_posture", "Hedge only idiosyncratic risks"),
        "valuation": valuation,
        "evidence": evidence,
        "drivers": drivers,
        "watch_items": watch_items,
    }


def credit_brief_markdown(df: pd.DataFrame) -> str:
    brief = build_credit_brief(df)
    if not brief.get("available"):
        return brief["headline"]

    lines = [
        "# Credit Brief",
        "",
        f"**Date:** {brief['date']}",
        f"**Headline:** {brief['headline']}",
        "",
        "## Stance",
        f"- Regime: {brief['regime']}",
        f"- Composite risk score: {_fmt_num(brief['score'])}",
        f"- Credit beta: {brief['credit_beta']}",
        f"- HY tilt: {brief['hy_tilt']}",
        f"- IG tilt: {brief['ig_tilt']}",
        f"- Duration: {brief['duration_stance']}",
        f"- Hedge posture: {brief['hedge_posture']}",
        "",
        "## Evidence",
    ]
    lines.extend([f"- {item}" for item in brief["evidence"]] or ["- Evidence unavailable."])
    lines.extend(["", "## Main Drivers"])
    lines.extend([f"- {item}" for item in brief["drivers"]] or ["- Channel drivers unavailable."])
    lines.extend(["", "## Watch Items"])
    lines.extend([f"- {item}" for item in brief["watch_items"]])
    return "\n".join(lines)


def framework_assumptions_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("HY recovery rate", "40%", "Simplified long-run recovery assumption used in expected-loss math."),
            ("Risk-On default PD", "2.5%", "Benign credit-cycle default assumption."),
            ("Neutral default PD", "4.0%", "Mid-cycle credit default assumption."),
            ("Caution default PD", "7.5%", "Late-cycle or deteriorating default assumption."),
            ("Risk-Off default PD", "12.0%", "Stress default assumption."),
            ("Percentile lookback", "Expanding", "Uses available history after minimum observation threshold."),
            ("Minimum percentile observations", "60", "Avoids over-interpreting early sample percentiles."),
            ("Validation horizons", "21/63/126 trading days", "Roughly 1M/3M/6M forward windows."),
        ],
        columns=["Assumption", "Value", "Use"],
    )


def credit_glossary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("OAS", "Option-adjusted spread; compensation over Treasuries after option effects."),
            ("Expected loss", "Default probability multiplied by loss given default."),
            ("Excess spread", "OAS minus expected-loss spread; simplified risk/liquidity premium proxy."),
            ("HY/IG ratio", "High-yield spread divided by investment-grade spread."),
            ("BBB/IG ratio", "BBB spread divided by broad IG spread; proxy for quality pressure."),
            ("Credit beta", "Directional exposure to spread tightening or widening."),
            ("Quality tilt", "Preference for lower- or higher-quality credit buckets."),
            ("Channel contribution", "Channel score multiplied by its institutional framework weight."),
        ],
        columns=["Term", "Definition"],
    )

