"""
Generate a concise credit strategy memo from the scored dashboard dataset.
"""

from __future__ import annotations

import pandas as pd

from src.credit_taxonomy import latest_channel_snapshot
from src.spread_decomposition import latest_spread_snapshot
from src.credit_relative_value import latest_relative_value_snapshot
from src.channel_attribution import top_channel_drivers


_ACTIONS = {
    "Risk-On": "Overweight credit beta, keep HY exposure, and avoid overpaying for quality.",
    "Neutral": "Keep balanced IG/HY exposure and let valuation drive incremental risk.",
    "Caution": "Reduce lower-quality HY and CCC beta, prefer IG quality, and raise liquidity.",
    "Risk-Off": "Protect capital, minimize HY beta, prefer cash/Treasuries/high-quality IG, and wait for stabilization.",
}


def _fmt(value, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def generate_credit_strategy_memo(df: pd.DataFrame) -> str:
    if df.empty:
        return "Credit Strategy Memo\n\nNo data available."

    latest = df.iloc[-1]
    date = latest.get("date", df.index[-1])
    regime = latest.get("final_decision", "Unknown")
    composite = latest.get("composite_risk_score_smooth", latest.get("composite_risk_score"))

    spread = latest_spread_snapshot(df)
    relative_value = latest_relative_value_snapshot(df)
    channels = latest_channel_snapshot(df)
    drivers = top_channel_drivers(df)
    top_channels = []
    if channels["available"]:
        ranked = sorted(
            [c for c in channels["channels"] if c["score"] is not None],
            key=lambda c: c["score"],
            reverse=True,
        )
        top_channels = ranked[:3]

    lines = [
        "Credit Strategy Memo",
        "",
        f"Date: {date}",
        f"Regime: {regime}",
        f"Composite Risk Score: {_fmt(composite)}",
        "",
        "Market View:",
        _ACTIONS.get(regime, "Maintain balanced risk until signals become clearer."),
        "",
        "Spread Compensation:",
    ]

    if spread["available"]:
        lines.extend(
            [
                f"- HY OAS: {spread['spread_oas_bps']:.1f} bps",
                f"- Expected loss: {spread['expected_loss_bps']:.1f} bps "
                f"(PD {spread['default_probability']:.1%}, recovery {spread['recovery_rate']:.0%})",
                f"- Excess spread: {spread['excess_spread_bps']:.1f} bps "
                f"({spread['spread_valuation']})",
            ]
        )
    else:
        lines.append(f"- Unavailable: {spread.get('reason', 'missing data')}")

    lines.extend(["", "Dominant Risk Channels:"])
    if drivers:
        for driver in drivers:
            lines.append(f"- {driver}")
    elif top_channels:
        for channel in top_channels:
            lines.append(
                f"- {channel['name']}: {channel['score']:.1f} "
                f"(coverage {channel['coverage']:.0%})"
            )
    else:
        lines.append("- Channel scores unavailable; add mapped channel columns for attribution.")

    lines.extend(["", "Relative Value / Quality:"])
    if relative_value.get("available"):
        hy_pct = relative_value.get("hy_spread_percentile")
        ig_pct = relative_value.get("ig_spread_percentile")
        lines.append(
            f"- HY valuation: {relative_value.get('hy_spread_valuation', 'Unavailable')} "
            f"({hy_pct:.0f} pctile)" if hy_pct is not None else "- HY valuation unavailable"
        )
        if ig_pct is not None:
            lines.append(
                f"- IG valuation: {relative_value.get('ig_spread_valuation', 'Unavailable')} "
                f"({ig_pct:.0f} pctile)"
            )
        lines.append(f"- Quality tilt: {relative_value.get('quality_tilt', 'Balanced quality stance.')}")
    else:
        lines.append("- Relative-value data unavailable.")

    lines.extend(
        [
            "",
            "Risk Triggers:",
            "- Further HY widening with contained VIX would indicate credit-specific deterioration.",
            "- A jump in liquidity/funding stress should reduce tolerance for lower-quality credit beta.",
            "- Stabilizing spreads after a high-risk regime would improve forward return asymmetry.",
        ]
    )

    return "\n".join(lines)
