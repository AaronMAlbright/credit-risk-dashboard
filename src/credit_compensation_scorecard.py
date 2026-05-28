"""
Credit compensation scorecard.

This module answers the PM-facing question: are investors being paid enough
to own credit risk right now? It combines spread compensation, carry
breakeven, default-cycle fundamentals, lending standards, and the existing
portfolio regime into a transparent recommendation.
"""
from __future__ import annotations

import math

import pandas as pd

from src.carry_breakeven import get_current_breakeven
from src.spread_decomposition import latest_spread_snapshot


RECOMMENDATIONS = {"Add", "Hold", "Upgrade Quality", "Hedge", "De-risk"}


def _safe_float(value) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(val) else val


def _latest_value(df: pd.DataFrame, column: str) -> float | str | None:
    if column not in df.columns:
        return None
    series = df[column].dropna()
    if series.empty:
        return None
    value = series.iloc[-1]
    if isinstance(value, str):
        return value
    return _safe_float(value)


def _spread_to_bps(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if abs(value) < 50 else value


def _fmt_bps(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} bps"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}%"


def _fmt_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}x"


def _fmt_ordinal(value: float | None) -> str:
    if value is None:
        return "-"
    n = int(round(value))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _classify_sloos(change_90d: float | None) -> str:
    if change_90d is None:
        return "Unavailable"
    if change_90d > 10:
        return "Tightening materially"
    if change_90d > 3:
        return "Tightening"
    if change_90d < -10:
        return "Easing materially"
    if change_90d < -3:
        return "Easing"
    return "Stable"


def _classify_fundamentals(
    chargeoff_change_90d: float | None,
    delinquency_change_90d: float | None,
    implied_vs_actual_gap: float | None,
) -> str:
    worsening = sum(
        1 for value in [chargeoff_change_90d, delinquency_change_90d]
        if value is not None and value > 0.05
    )
    improving = sum(
        1 for value in [chargeoff_change_90d, delinquency_change_90d]
        if value is not None and value < -0.05
    )
    if worsening >= 2:
        return "Realized credit losses worsening"
    if worsening == 1:
        return "Some realized credit deterioration"
    if improving >= 1 and (implied_vs_actual_gap is None or implied_vs_actual_gap > 0):
        return "Realized losses stable/improving"
    return "Stable"


def _allocation_guidance(
    *,
    recommendation: str,
    hy_spread_bps: float | None,
    expected_loss_bps: float | None,
    excess_spread_bps: float | None,
    hy_percentile: float | None,
    ig_percentile: float | None,
    hy_ig_ratio: float | None,
    bbb_ig_ratio: float | None,
    fundamental_status: str,
    sloos_status: str,
) -> dict:
    if recommendation == "Add":
        credit_beta = "Overweight credit beta selectively"
        quality_bias = "Favor BB/B carry; avoid weakest CCC names unless fundamentals confirm"
        hedge_posture = "Keep hedges light; use spread widening stops"
    elif recommendation == "Upgrade Quality":
        credit_beta = "Neutral to underweight HY beta"
        quality_bias = "Rotate toward IG, BB, and shorter spread duration"
        hedge_posture = "Keep cheap downside hedges on lower-quality exposure"
    elif recommendation == "Hedge":
        credit_beta = "Keep core exposure but reduce net downside"
        quality_bias = "Prefer IG and higher-quality HY"
        hedge_posture = "Add index hedges or payer protection against spread widening"
    elif recommendation == "De-risk":
        credit_beta = "Underweight credit beta"
        quality_bias = "Raise IG/cash quality and cut weakest spread beta"
        hedge_posture = "Maintain explicit HY spread hedges until compensation improves"
    else:
        credit_beta = "Neutral credit beta"
        quality_bias = "Balanced IG/HY quality stance"
        hedge_posture = "Use tactical hedges only around event risk"

    if hy_percentile is not None and ig_percentile is not None:
        if hy_percentile - ig_percentile >= 20:
            hy_ig_tilt = "HY screens cheaper than IG"
        elif ig_percentile - hy_percentile >= 20:
            hy_ig_tilt = "IG screens better than HY"
        else:
            hy_ig_tilt = "No major HY/IG valuation gap"
    elif hy_ig_ratio is not None:
        if hy_ig_ratio >= 4.5:
            hy_ig_tilt = "HY offers elevated spread pickup versus IG"
        elif hy_ig_ratio <= 3.0:
            hy_ig_tilt = "HY pickup versus IG is thin"
        else:
            hy_ig_tilt = "HY/IG pickup is near normal"
    else:
        hy_ig_tilt = "HY/IG valuation unavailable"

    if bbb_ig_ratio is not None and bbb_ig_ratio > 1.35:
        quality_bias = "Upgrade quality; BBB pressure is elevated versus broad IG"

    fair_spread_bps = None
    if expected_loss_bps is not None:
        fair_spread_bps = expected_loss_bps + 175.0

    if fair_spread_bps is None:
        watch_level = "Watch SLOOS, charge-offs, and delinquency trend for confirmation"
    elif excess_spread_bps is not None and excess_spread_bps < 175:
        watch_level = f"Reassess risk add above roughly {_fmt_bps(fair_spread_bps)} HY OAS"
    else:
        watch_level = f"Defend gains if HY OAS tightens below roughly {_fmt_bps(fair_spread_bps)}"

    if "worsening" in fundamental_status.lower() or "tightening" in sloos_status.lower():
        watch_level = f"{watch_level}; fundamentals/loan standards are the key veto"

    return {
        "credit_beta": credit_beta,
        "quality_bias": quality_bias,
        "hy_ig_tilt": hy_ig_tilt,
        "hedge_posture": hedge_posture,
        "watch_level": watch_level,
    }


def _trigger_levels(
    *,
    expected_loss_bps: float | None,
    excess_spread_bps: float | None,
    compensation_ratio: float | None,
    hy_spread_bps: float | None,
    sloos_change_90d: float | None,
    chargeoff_change_90d: float | None,
    delinquency_change_90d: float | None,
) -> dict:
    fair_oas = expected_loss_bps + 175.0 if expected_loss_bps is not None else None
    cheap_oas = expected_loss_bps + 300.0 if expected_loss_bps is not None else None
    rich_oas = expected_loss_bps + 75.0 if expected_loss_bps is not None else None
    hedge_oas = hy_spread_bps + 50.0 if hy_spread_bps is not None else None

    add_trigger = (
        f"Add risk if HY OAS is above {_fmt_bps(cheap_oas)} and SLOOS is stable/easing"
        if cheap_oas is not None
        else "Add risk only when excess spread is cheap and lending standards are stable/easing"
    )
    reduce_trigger = (
        f"Reduce risk if HY OAS tightens below {_fmt_bps(rich_oas)} or compensation drops below 1.1x"
        if rich_oas is not None
        else "Reduce risk if compensation ratio falls below 1.1x"
    )
    hedge_trigger = (
        f"Hedge if HY OAS widens through {_fmt_bps(hedge_oas)} with worse loan standards"
        if hedge_oas is not None
        else "Hedge if spreads widen while loan standards or realized losses worsen"
    )

    if excess_spread_bps is not None and excess_spread_bps < 175:
        add_trigger = (
            f"Wait for HY OAS near {_fmt_bps(fair_oas)} before adding risk"
            if fair_oas is not None
            else "Wait for excess spread above 175 bps before adding risk"
        )
    if compensation_ratio is not None and compensation_ratio < 1.1:
        reduce_trigger = "Reduce risk now; spread is not covering expected loss with enough cushion"
    if sloos_change_90d is not None and sloos_change_90d > 3:
        hedge_trigger = "Keep hedges until SLOOS stops tightening"
    if any(
        value is not None and value > 0.05
        for value in [chargeoff_change_90d, delinquency_change_90d]
    ):
        hedge_trigger = f"{hedge_trigger}; realized credit losses are deteriorating"

    return {
        "add_risk": add_trigger,
        "reduce_risk": reduce_trigger,
        "hedge_or_exit": hedge_trigger,
        "fair_oas_bps": fair_oas,
        "cheap_oas_bps": cheap_oas,
        "rich_oas_bps": rich_oas,
    }


def _trade_memo(
    *,
    recommendation: str,
    drivers: list[str],
    allocation: dict,
    triggers: dict,
    hy_spread_bps: float | None,
    excess_spread_bps: float | None,
    compensation_ratio: float | None,
    hy_percentile: float | None,
    fundamental_status: str,
    sloos_status: str,
) -> dict:
    current_view = (
        f"{recommendation} credit: HY OAS {_fmt_bps(hy_spread_bps)}, "
        f"excess spread {_fmt_bps(excess_spread_bps)}, "
        f"compensation {_fmt_ratio(compensation_ratio)}."
    )
    if hy_percentile is not None:
        current_view = f"{current_view} HY screens at the {_fmt_ordinal(hy_percentile)} spread percentile."

    what_changed = "; ".join(drivers) if drivers else "No dominant valuation or fundamental break."
    why_it_matters = (
        f"{allocation['credit_beta']} because spread compensation must clear expected loss, "
        f"fundamentals are {fundamental_status.lower()}, and lending standards are {sloos_status.lower()}."
    )
    trade_expression = (
        f"{allocation['quality_bias']}. {allocation['hy_ig_tilt']}. {allocation['hedge_posture']}."
    )
    invalidation = triggers["add_risk"] if recommendation in {"Upgrade Quality", "Hedge", "De-risk"} else triggers["reduce_risk"]

    return {
        "Current View": current_view,
        "What Changed": what_changed,
        "Why It Matters": why_it_matters,
        "Trade Expression": trade_expression,
        "Invalidation Level": invalidation,
    }


def _recommendation(
    *,
    compensation_ratio: float | None,
    excess_spread_bps: float | None,
    hy_percentile: float | None,
    composite_score: float | None,
    sloos_change_90d: float | None,
    chargeoff_change_90d: float | None,
    delinquency_change_90d: float | None,
) -> tuple[str, list[str]]:
    drivers: list[str] = []
    ratio = compensation_ratio
    excess = excess_spread_bps
    percentile = hy_percentile
    score = composite_score
    technical_tightening = sloos_change_90d is not None and sloos_change_90d > 3
    fundamental_worsening = any(
        value is not None and value > 0.05
        for value in [chargeoff_change_90d, delinquency_change_90d]
    )
    very_rich = (percentile is not None and percentile < 20) or (excess is not None and excess < 75)
    cheap = (percentile is not None and percentile > 65) or (excess is not None and excess >= 175)

    if ratio is not None:
        drivers.append(f"Compensation ratio {ratio:.2f}x")
    if percentile is not None:
        drivers.append(f"HY spread at {_fmt_ordinal(percentile)} percentile")
    if technical_tightening:
        drivers.append("SLOOS tightening")
    if fundamental_worsening:
        drivers.append("Realized credit fundamentals worsening")

    if score is not None and score >= 70:
        drivers.append("Composite risk score is defensive")
        return "De-risk", drivers
    if very_rich and (technical_tightening or fundamental_worsening):
        return "De-risk", drivers
    if very_rich:
        return "Upgrade Quality", drivers
    if ratio is not None and ratio < 1.1 and (technical_tightening or fundamental_worsening):
        return "Hedge", drivers
    if cheap and ratio is not None and ratio >= 1.5 and not technical_tightening and not fundamental_worsening:
        if score is None or score < 50:
            return "Add", drivers
    if technical_tightening or fundamental_worsening:
        return "Hedge", drivers
    return "Hold", drivers


def build_credit_compensation_scorecard(df: pd.DataFrame) -> dict:
    """Return the current credit compensation scorecard."""
    if df.empty:
        return {"available": False, "reason": "No data available"}

    spread = latest_spread_snapshot(df)
    if not spread.get("available"):
        return {"available": False, "reason": spread.get("reason", "Spread data unavailable")}

    breakeven = get_current_breakeven(df)
    hy_spread_bps = _spread_to_bps(
        _safe_float(_latest_value(df, "hy_spread_bps"))
        or _safe_float(spread.get("spread_oas_bps"))
    )
    ig_spread_bps = _spread_to_bps(_safe_float(_latest_value(df, "ig_spread_bps")))
    hy_percentile = _safe_float(_latest_value(df, "hy_spread_percentile"))
    ig_percentile = _safe_float(_latest_value(df, "ig_spread_percentile"))
    hy_ig_ratio = (
        _safe_float(_latest_value(df, "hy_ig_spread_ratio"))
        or _safe_float(_latest_value(df, "hy_ig_ratio"))
    )
    bbb_ig_ratio = _safe_float(_latest_value(df, "bbb_ig_ratio"))
    implied_default_pct = _safe_float(_latest_value(df, "default_cycle_pct"))
    if implied_default_pct is None and hy_spread_bps is not None:
        # Jarrow-Turnbull approximation with 40% recovery / 60% LGD.
        implied_default_pct = hy_spread_bps / 60.0
    chargeoff_pct = (
        _safe_float(_latest_value(df, "actual_chargeoff_pct"))
        or _safe_float(_latest_value(df, "business_chargeoff_rate"))
    )
    delinquency_pct = (
        _safe_float(_latest_value(df, "actual_delinq_pct"))
        or _safe_float(_latest_value(df, "ci_loan_delinquency"))
    )
    implied_vs_actual_gap = _safe_float(_latest_value(df, "implied_vs_actual_gap"))
    sloos_change_90d = _safe_float(_latest_value(df, "sloos_change_90d"))
    chargeoff_change_90d = _safe_float(_latest_value(df, "chargeoff_change_90d"))
    delinquency_change_90d = _safe_float(_latest_value(df, "delinquency_change_90d"))
    composite_score = _safe_float(_latest_value(df, "composite_risk_score_smooth"))
    final_decision = _latest_value(df, "final_decision") or "Unavailable"

    compensation_ratio = _safe_float(spread.get("spread_compensation_ratio"))
    excess_spread_bps = _safe_float(spread.get("excess_spread_bps"))
    expected_loss_bps = _safe_float(spread.get("expected_loss_bps"))
    hy_breakeven_bps = _safe_float(breakeven.get("breakeven_hy_bps")) if breakeven.get("available") else None

    rec, drivers = _recommendation(
        compensation_ratio=compensation_ratio,
        excess_spread_bps=excess_spread_bps,
        hy_percentile=hy_percentile,
        composite_score=composite_score,
        sloos_change_90d=sloos_change_90d,
        chargeoff_change_90d=chargeoff_change_90d,
        delinquency_change_90d=delinquency_change_90d,
    )

    fundamental_status = _classify_fundamentals(
        chargeoff_change_90d, delinquency_change_90d, implied_vs_actual_gap
    )
    sloos_status = _classify_sloos(sloos_change_90d)
    allocation = _allocation_guidance(
        recommendation=rec,
        hy_spread_bps=hy_spread_bps,
        expected_loss_bps=expected_loss_bps,
        excess_spread_bps=excess_spread_bps,
        hy_percentile=hy_percentile,
        ig_percentile=ig_percentile,
        hy_ig_ratio=hy_ig_ratio,
        bbb_ig_ratio=bbb_ig_ratio,
        fundamental_status=fundamental_status,
        sloos_status=sloos_status,
    )
    triggers = _trigger_levels(
        expected_loss_bps=expected_loss_bps,
        excess_spread_bps=excess_spread_bps,
        compensation_ratio=compensation_ratio,
        hy_spread_bps=hy_spread_bps,
        sloos_change_90d=sloos_change_90d,
        chargeoff_change_90d=chargeoff_change_90d,
        delinquency_change_90d=delinquency_change_90d,
    )
    trade_memo = _trade_memo(
        recommendation=rec,
        drivers=drivers,
        allocation=allocation,
        triggers=triggers,
        hy_spread_bps=hy_spread_bps,
        excess_spread_bps=excess_spread_bps,
        compensation_ratio=compensation_ratio,
        hy_percentile=hy_percentile,
        fundamental_status=fundamental_status,
        sloos_status=sloos_status,
    )

    current = {
        "hy_oas_bps": hy_spread_bps,
        "ig_oas_bps": ig_spread_bps,
        "hy_spread_percentile": hy_percentile,
        "ig_spread_percentile": ig_percentile,
        "hy_ig_spread_ratio": hy_ig_ratio,
        "bbb_ig_ratio": bbb_ig_ratio,
        "expected_loss_bps": expected_loss_bps,
        "excess_spread_bps": excess_spread_bps,
        "spread_compensation_ratio": compensation_ratio,
        "carry_breakeven_hy_bps": hy_breakeven_bps,
        "implied_default_pct": implied_default_pct,
        "actual_chargeoff_pct": chargeoff_pct,
        "business_loan_delinquency_pct": delinquency_pct,
        "sloos_change_90d": sloos_change_90d,
        "fundamental_status": fundamental_status,
        "sloos_status": sloos_status,
        "final_decision": final_decision,
        "recommendation": rec,
        "drivers": drivers,
        "allocation": allocation,
        "triggers": triggers,
    }

    rows = pd.DataFrame([
        {"metric": "HY OAS", "value": _fmt_bps(hy_spread_bps), "interpretation": "Current HY spread compensation"},
        {"metric": "IG OAS", "value": _fmt_bps(ig_spread_bps), "interpretation": "Current IG spread anchor"},
        {"metric": "HY Percentile", "value": _fmt_ordinal(hy_percentile), "interpretation": "Higher means spreads are wider vs history"},
        {"metric": "Expected Loss", "value": _fmt_bps(expected_loss_bps), "interpretation": "Regime-based PD times loss-given-default"},
        {"metric": "Excess Spread", "value": _fmt_bps(excess_spread_bps), "interpretation": "Residual spread after expected loss"},
        {"metric": "Compensation Ratio", "value": _fmt_ratio(compensation_ratio), "interpretation": "Spread divided by expected loss"},
        {"metric": "HY Carry Breakeven", "value": _fmt_bps(hy_breakeven_bps), "interpretation": "Annual widening cushion before carry is erased"},
        {"metric": "Implied Default Rate", "value": _fmt_pct(implied_default_pct), "interpretation": "Market-implied default rate from spreads"},
        {"metric": "Charge-Off Rate", "value": _fmt_pct(chargeoff_pct), "interpretation": "Observed business-loan credit loss rate"},
        {"metric": "Business Loan Delinquency", "value": _fmt_pct(delinquency_pct), "interpretation": "Observed delinquency pressure"},
        {"metric": "SLOOS 90D Trend", "value": _fmt_pct(sloos_change_90d), "interpretation": current["sloos_status"]},
        {"metric": "Recommendation", "value": rec, "interpretation": "; ".join(drivers) if drivers else "No dominant risk imbalance"},
    ])
    action_rows = pd.DataFrame([
        {"decision": "Credit beta", "guidance": allocation["credit_beta"], "why it matters": "Translates valuation into portfolio risk budget"},
        {"decision": "Quality bias", "guidance": allocation["quality_bias"], "why it matters": "Separates spread pickup from downgrade/default exposure"},
        {"decision": "HY vs IG", "guidance": allocation["hy_ig_tilt"], "why it matters": "Compares risky-credit compensation against higher-quality spread"},
        {"decision": "Hedge posture", "guidance": allocation["hedge_posture"], "why it matters": "Defines whether to own, insure, or reduce the spread risk"},
        {"decision": "Watch level", "guidance": allocation["watch_level"], "why it matters": "Turns the view into a monitorable trigger"},
    ])
    memo_rows = pd.DataFrame(
        [{"section": section, "view": view} for section, view in trade_memo.items()]
    )
    trigger_rows = pd.DataFrame([
        {"trigger": "Add risk", "condition": triggers["add_risk"], "portfolio action": "Increase HY beta or move down quality selectively"},
        {"trigger": "Reduce risk", "condition": triggers["reduce_risk"], "portfolio action": "Upgrade quality, shorten spread duration, or raise cash"},
        {"trigger": "Hedge / exit", "condition": triggers["hedge_or_exit"], "portfolio action": "Add CDX/HY protection or cut weakest spread beta"},
    ])

    return {
        "available": True,
        "current": current,
        "table": rows,
        "action_table": action_rows,
        "memo": trade_memo,
        "memo_table": memo_rows,
        "trigger_table": trigger_rows,
        "triggers": triggers,
        "allocation": allocation,
        "recommendation": rec,
        "drivers": drivers,
        "summary": (
            f"{rec}: HY OAS {_fmt_bps(hy_spread_bps)}, excess spread {_fmt_bps(excess_spread_bps)}, "
            f"compensation ratio {_fmt_ratio(compensation_ratio)}."
        ),
    }
