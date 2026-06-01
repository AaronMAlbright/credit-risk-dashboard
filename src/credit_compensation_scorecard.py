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
from src.spread_decomposition import decompose_spreads, latest_spread_snapshot


RECOMMENDATIONS = {"Add", "Hold", "Upgrade Quality", "Hedge", "De-risk"}
RATING_BUCKETS = ["IG", "BBB", "BB", "B", "CCC", "Cash", "Hedge"]
BUCKET_ANALYTICS = {
    "IG": {"spread_factor": 1.00, "spread_beta": 0.25, "source": "ig", "loss_factor": 0.20, "duration": 6.5, "stress_widen_bps": 75.0},
    "BBB": {"spread_factor": 1.25, "spread_beta": 0.40, "source": "ig", "loss_factor": 0.35, "duration": 6.0, "stress_widen_bps": 125.0},
    "BB": {"spread_factor": 0.75, "spread_beta": 0.75, "source": "hy", "loss_factor": 0.65, "duration": 4.5, "stress_widen_bps": 250.0},
    "B": {"spread_factor": 1.10, "spread_beta": 1.10, "source": "hy", "loss_factor": 1.15, "duration": 3.75, "stress_widen_bps": 400.0},
    "CCC": {"spread_factor": 1.85, "spread_beta": 1.80, "source": "hy", "loss_factor": 2.25, "duration": 2.75, "stress_widen_bps": 700.0},
    "Cash": {"spread_factor": 0.00, "spread_beta": 0.00, "source": "cash", "loss_factor": 0.00, "duration": 0.0, "stress_widen_bps": 0.0},
    "Hedge": {"spread_factor": 0.00, "spread_beta": -1.00, "source": "hedge", "loss_factor": 0.00, "duration": 4.0, "stress_widen_bps": -300.0},
}
FORWARD_HORIZONS = [
    ("1M", "hy_spread_forward_21d_change", "ig_spread_forward_21d_change"),
    ("3M", "hy_spread_forward_63d_change", "ig_spread_forward_63d_change"),
    ("6M", "hy_spread_forward_126d_change", "ig_spread_forward_126d_change"),
]


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


def _series_to_bps(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median_abs = s.dropna().abs().median()
    if pd.notna(median_abs) and median_abs < 10:
        return s * 100.0
    return s


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


def _rating_bucket_allocation(
    *,
    recommendation: str,
    hy_percentile: float | None,
    ig_percentile: float | None,
    bbb_ig_ratio: float | None,
    sloos_change_90d: float | None,
    chargeoff_change_90d: float | None,
    delinquency_change_90d: float | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    weights_by_rec = {
        "Add": {"IG": 20.0, "BBB": 15.0, "BB": 30.0, "B": 20.0, "CCC": 5.0, "Cash": 5.0, "Hedge": 5.0},
        "Hold": {"IG": 30.0, "BBB": 20.0, "BB": 25.0, "B": 10.0, "CCC": 0.0, "Cash": 10.0, "Hedge": 5.0},
        "Upgrade Quality": {"IG": 40.0, "BBB": 20.0, "BB": 20.0, "B": 5.0, "CCC": 0.0, "Cash": 10.0, "Hedge": 5.0},
        "Hedge": {"IG": 35.0, "BBB": 20.0, "BB": 15.0, "B": 5.0, "CCC": 0.0, "Cash": 15.0, "Hedge": 10.0},
        "De-risk": {"IG": 30.0, "BBB": 15.0, "BB": 5.0, "B": 0.0, "CCC": 0.0, "Cash": 35.0, "Hedge": 15.0},
    }
    weights = dict(weights_by_rec.get(recommendation, weights_by_rec["Hold"]))

    quality_pressure = bbb_ig_ratio is not None and bbb_ig_ratio > 1.35
    lending_tightening = sloos_change_90d is not None and sloos_change_90d > 3
    fundamental_worsening = any(
        value is not None and value > 0.05
        for value in [chargeoff_change_90d, delinquency_change_90d]
    )
    hy_cheap_vs_ig = hy_percentile is not None and ig_percentile is not None and hy_percentile - ig_percentile >= 20
    hy_rich = hy_percentile is not None and hy_percentile < 25

    if quality_pressure:
        weights["BBB"] = max(0.0, weights["BBB"] - 5.0)
        weights["IG"] += 5.0
    if lending_tightening or fundamental_worsening:
        weights["CCC"] = 0.0
        weights["B"] = max(0.0, weights["B"] - 5.0)
        weights["Cash"] += 3.0
        weights["Hedge"] += 2.0
    elif hy_cheap_vs_ig and recommendation in {"Add", "Hold"}:
        weights["IG"] = max(0.0, weights["IG"] - 5.0)
        weights["BB"] += 3.0
        weights["B"] += 2.0
    if hy_rich and recommendation in {"Upgrade Quality", "Hedge", "De-risk"}:
        weights["B"] = max(0.0, weights["B"] - 5.0)
        weights["IG"] += 3.0
        weights["Cash"] += 2.0

    total = sum(weights.values())
    if total:
        weights = {bucket: round(weight * 100.0 / total, 1) for bucket, weight in weights.items()}
        rounding_gap = round(100.0 - sum(weights.values()), 1)
        weights["Cash"] = round(weights["Cash"] + rounding_gap, 1)

    baseline = {"IG": 30.0, "BBB": 20.0, "BB": 20.0, "B": 10.0, "CCC": 2.5, "Cash": 12.5, "Hedge": 5.0}
    rationales = {
        "IG": "Core quality ballast and liquidity source",
        "BBB": "Spread pickup inside IG, reduced when downgrade pressure rises",
        "BB": "Preferred HY risk bucket when compensation is adequate",
        "B": "Selective carry bucket; sensitive to lending standards and defaults",
        "CCC": "Deep credit beta; only funded when compensation is broad and fundamentals stable",
        "Cash": "Dry powder and volatility buffer",
        "Hedge": "Explicit spread protection against widening/default beta",
    }
    rows = []
    for bucket in RATING_BUCKETS:
        weight = weights.get(bucket, 0.0)
        diff = weight - baseline[bucket]
        if diff >= 5:
            tilt = "Overweight"
        elif diff <= -5:
            tilt = "Underweight"
        else:
            tilt = "Neutral"
        rows.append(
            {
                "bucket": bucket,
                "target_weight": weight,
                "tilt": tilt,
                "rationale": rationales[bucket],
            }
        )

    return weights, pd.DataFrame(rows)


def _expected_spread_change_bps(
    *,
    hy_percentile: float | None,
    sloos_change_90d: float | None,
    chargeoff_change_90d: float | None,
    delinquency_change_90d: float | None,
) -> float:
    change = 0.0
    if hy_percentile is not None:
        if hy_percentile >= 80:
            change -= 45.0
        elif hy_percentile >= 65:
            change -= 25.0
        elif hy_percentile <= 20:
            change += 45.0
        elif hy_percentile <= 35:
            change += 20.0

    if sloos_change_90d is not None:
        if sloos_change_90d > 10:
            change += 35.0
        elif sloos_change_90d > 3:
            change += 20.0
        elif sloos_change_90d < -10:
            change -= 20.0
        elif sloos_change_90d < -3:
            change -= 10.0

    worsening_count = sum(
        1 for value in [chargeoff_change_90d, delinquency_change_90d]
        if value is not None and value > 0.05
    )
    change += 20.0 * worsening_count
    return change


def _blended_expected_spread_change(
    *,
    rule_change_bps: float,
    forward_outcomes: dict | None,
) -> dict:
    if not forward_outcomes or not forward_outcomes.get("available"):
        return {
            "expected_hy_spread_change_bps": rule_change_bps,
            "source": "Rules",
            "rule_change_bps": rule_change_bps,
            "historical_change_bps": None,
        }

    table = forward_outcomes.get("table")
    if table is None or table.empty or forward_outcomes.get("sample_count", 0) < 8:
        return {
            "expected_hy_spread_change_bps": rule_change_bps,
            "source": "Rules",
            "rule_change_bps": rule_change_bps,
            "historical_change_bps": None,
        }

    analog_rows = table[table["horizon"].isin(["3M", "1M"])] if "horizon" in table.columns else pd.DataFrame()
    if analog_rows.empty:
        analog_rows = table
    if "3M" in set(analog_rows["horizon"]):
        analog_row = analog_rows.loc[analog_rows["horizon"] == "3M"].iloc[0]
    else:
        analog_row = analog_rows.iloc[0]

    historical_change = _safe_float(analog_row.get("hy_median_change_bps"))
    if historical_change is None:
        return {
            "expected_hy_spread_change_bps": rule_change_bps,
            "source": "Rules",
            "rule_change_bps": rule_change_bps,
            "historical_change_bps": None,
        }

    blended = (rule_change_bps * 0.50) + (historical_change * 0.50)
    return {
        "expected_hy_spread_change_bps": blended,
        "source": "Blended historical analogs + rules",
        "rule_change_bps": rule_change_bps,
        "historical_change_bps": historical_change,
    }


def _bucket_spread_bps(
    bucket: str,
    *,
    hy_spread_bps: float | None,
    ig_spread_bps: float | None,
    bbb_ig_ratio: float | None,
) -> float | None:
    assumptions = BUCKET_ANALYTICS[bucket]
    source = assumptions["source"]
    if source == "cash":
        return 0.0
    if source == "hedge":
        return -75.0
    if source == "ig":
        if ig_spread_bps is None:
            return None
        factor = bbb_ig_ratio if bucket == "BBB" and bbb_ig_ratio is not None else assumptions["spread_factor"]
        return ig_spread_bps * factor
    if hy_spread_bps is None:
        return None
    return hy_spread_bps * assumptions["spread_factor"]


def _bucket_return_and_stress(
    *,
    rating_weights: dict[str, float],
    hy_spread_bps: float | None,
    ig_spread_bps: float | None,
    expected_loss_bps: float | None,
    hy_percentile: float | None,
    bbb_ig_ratio: float | None,
    sloos_change_90d: float | None,
    chargeoff_change_90d: float | None,
    delinquency_change_90d: float | None,
    forward_outcomes: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    rule_hy_change = _expected_spread_change_bps(
        hy_percentile=hy_percentile,
        sloos_change_90d=sloos_change_90d,
        chargeoff_change_90d=chargeoff_change_90d,
        delinquency_change_90d=delinquency_change_90d,
    )
    spread_move = _blended_expected_spread_change(
        rule_change_bps=rule_hy_change,
        forward_outcomes=forward_outcomes,
    )
    expected_hy_change = spread_move["expected_hy_spread_change_bps"]
    hy_loss = expected_loss_bps if expected_loss_bps is not None else 240.0
    rows = []

    for bucket in RATING_BUCKETS:
        assumptions = BUCKET_ANALYTICS[bucket]
        weight = rating_weights.get(bucket, 0.0)
        spread = _bucket_spread_bps(
            bucket,
            hy_spread_bps=hy_spread_bps,
            ig_spread_bps=ig_spread_bps,
            bbb_ig_ratio=bbb_ig_ratio,
        )
        duration = assumptions["duration"]
        loss_bps = hy_loss * assumptions["loss_factor"]
        stress_loss_bps = (duration * assumptions["stress_widen_bps"]) + (loss_bps * 1.5)

        if bucket == "Hedge":
            expected_mtm_bps = duration * expected_hy_change
            expected_return_bps = (spread or 0.0) + expected_mtm_bps
            stress_loss_bps = stress_loss_bps
        elif spread is None:
            expected_mtm_bps = None
            expected_return_bps = None
        else:
            bucket_change = expected_hy_change * assumptions["spread_beta"]
            expected_mtm_bps = -duration * bucket_change
            expected_return_bps = spread - loss_bps + expected_mtm_bps

        weighted_expected = None if expected_return_bps is None else expected_return_bps * weight / 100.0
        weighted_stress = stress_loss_bps * weight / 100.0
        rows.append(
            {
                "bucket": bucket,
                "target_weight": round(weight, 1),
                "spread_carry_bps": None if spread is None else round(float(spread), 1),
                "expected_default_drag_bps": round(float(loss_bps), 1),
                "expected_spread_mtm_bps": None if expected_mtm_bps is None else round(float(expected_mtm_bps), 1),
                "expected_excess_return_bps": None if expected_return_bps is None else round(float(expected_return_bps), 1),
                "recession_stress_loss_bps": round(float(stress_loss_bps), 1),
                "weighted_expected_return_bps": None if weighted_expected is None else round(float(weighted_expected), 1),
                "weighted_stress_loss_bps": round(float(weighted_stress), 1),
            }
        )

    table = pd.DataFrame(rows)
    weighted_expected = pd.to_numeric(table["weighted_expected_return_bps"], errors="coerce").sum()
    weighted_stress = pd.to_numeric(table["weighted_stress_loss_bps"], errors="coerce").sum()
    summary = {
        "expected_hy_spread_change_bps": round(expected_hy_change, 1),
        "expected_hy_spread_change_source": spread_move["source"],
        "rule_hy_spread_change_bps": round(rule_hy_change, 1),
        "historical_hy_spread_change_bps": (
            None if spread_move["historical_change_bps"] is None else round(spread_move["historical_change_bps"], 1)
        ),
        "portfolio_expected_excess_return_bps": round(float(weighted_expected), 1),
        "portfolio_recession_stress_loss_bps": round(float(weighted_stress), 1),
        "interpretation": (
            f"Bucket model estimates {weighted_expected:.0f} bps expected excess return "
            f"against {weighted_stress:.0f} bps recession stress loss "
            f"using {spread_move['source'].lower()} for expected spread move."
        ),
    }
    return table, summary


def _bucket_assumptions_table() -> pd.DataFrame:
    rows = []
    for bucket in RATING_BUCKETS:
        assumptions = BUCKET_ANALYTICS[bucket]
        rows.append(
            {
                "bucket": bucket,
                "spread_source": assumptions["source"],
                "spread_carry_factor": assumptions["spread_factor"],
                "spread_beta": assumptions["spread_beta"],
                "loss_factor": assumptions["loss_factor"],
                "spread_duration": assumptions["duration"],
                "recession_widening_bps": assumptions["stress_widen_bps"],
            }
        )
    return pd.DataFrame(rows)


def _risk_reward_metrics(
    *,
    bucket_return_table: pd.DataFrame,
    rating_weights: dict[str, float],
) -> tuple[pd.DataFrame, dict]:
    expected_return = pd.to_numeric(
        bucket_return_table["weighted_expected_return_bps"], errors="coerce"
    ).sum()
    stress_loss = pd.to_numeric(
        bucket_return_table["weighted_stress_loss_bps"], errors="coerce"
    ).sum()
    carry = (
        pd.to_numeric(bucket_return_table["spread_carry_bps"], errors="coerce")
        * pd.to_numeric(bucket_return_table["target_weight"], errors="coerce")
        / 100.0
    ).sum()
    hedge_benefit = -pd.to_numeric(
        bucket_return_table.loc[bucket_return_table["bucket"] == "Hedge", "weighted_stress_loss_bps"],
        errors="coerce",
    ).sum()
    tail_weight = sum(rating_weights.get(bucket, 0.0) for bucket in ["B", "CCC"])
    tail_stress = pd.to_numeric(
        bucket_return_table.loc[
            bucket_return_table["bucket"].isin(["B", "CCC"]),
            "weighted_stress_loss_bps",
        ],
        errors="coerce",
    ).sum()
    risk_reward = expected_return / stress_loss if stress_loss > 0 else None
    carry_to_stress = carry / stress_loss if stress_loss > 0 else None
    tail_stress_share = tail_stress / stress_loss if stress_loss > 0 else None
    hedge_coverage = hedge_benefit / stress_loss if stress_loss > 0 else None

    rows = pd.DataFrame([
        {
            "metric": "Expected return / stress loss",
            "value": "-" if risk_reward is None else f"{risk_reward:.2f}x",
            "interpretation": "Higher means more expected compensation per unit of recession loss",
        },
        {
            "metric": "Carry / stress loss",
            "value": "-" if carry_to_stress is None else f"{carry_to_stress:.2f}x",
            "interpretation": "Annual spread carry relative to modeled recession drawdown",
        },
        {
            "metric": "B/CCC tail weight",
            "value": f"{tail_weight:.1f}%",
            "interpretation": "Lower-quality exposure that can dominate default and liquidity beta",
        },
        {
            "metric": "B/CCC stress share",
            "value": "-" if tail_stress_share is None else f"{tail_stress_share * 100.0:.1f}%",
            "interpretation": "Share of recession loss coming from lower-quality HY buckets",
        },
        {
            "metric": "Hedge stress offset",
            "value": "-" if hedge_coverage is None else f"{hedge_coverage * 100.0:.1f}%",
            "interpretation": "Modeled recession stress absorbed by explicit hedge exposure",
        },
    ])
    summary = {
        "risk_reward_ratio": None if risk_reward is None else round(float(risk_reward), 2),
        "carry_to_stress_ratio": None if carry_to_stress is None else round(float(carry_to_stress), 2),
        "tail_weight_pct": round(float(tail_weight), 1),
        "tail_stress_share_pct": None if tail_stress_share is None else round(float(tail_stress_share * 100.0), 1),
        "hedge_stress_offset_pct": None if hedge_coverage is None else round(float(hedge_coverage * 100.0), 1),
    }
    return rows, summary


def _marginal_allocation_advice(
    *,
    bucket_return_table: pd.DataFrame,
    rating_weights: dict[str, float],
    risk_reward_summary: dict,
) -> pd.DataFrame:
    table = bucket_return_table.copy()
    table["expected_excess_return_bps"] = pd.to_numeric(table["expected_excess_return_bps"], errors="coerce")
    table["recession_stress_loss_bps"] = pd.to_numeric(table["recession_stress_loss_bps"], errors="coerce")
    table["stress_adjusted_score"] = table["expected_excess_return_bps"] / table["recession_stress_loss_bps"].abs().replace(0, pd.NA)

    credit = table[table["bucket"].isin(["IG", "BBB", "BB", "B", "CCC"])].copy()
    tail_stress_share = risk_reward_summary.get("tail_stress_share_pct") or 0.0
    eligible_adds = credit[
        (credit["expected_excess_return_bps"] > 0)
        & (credit["bucket"].ne("CCC") | (tail_stress_share < 30.0))
    ]
    if eligible_adds.empty:
        add_row = table.loc[table["bucket"].isin(["Cash", "Hedge"])].sort_values(
            ["expected_excess_return_bps"], ascending=False
        ).iloc[0]
    else:
        add_row = eligible_adds.sort_values(
            ["stress_adjusted_score", "expected_excess_return_bps"], ascending=False
        ).iloc[0]

    trim_candidates = table[
        (table["target_weight"] > 0)
        & table["bucket"].ne(add_row["bucket"])
        & table["bucket"].ne("Hedge")
    ].copy()
    if add_row["bucket"] in {"IG", "BBB", "BB", "B", "CCC"}:
        trim_candidates["funding_rank"] = trim_candidates.apply(
            lambda row: (
                0 if row["bucket"] == "Cash" else
                1 if row["expected_excess_return_bps"] < 0 else
                2
            ),
            axis=1,
        )
        fund_row = trim_candidates.sort_values(
            ["funding_rank", "stress_adjusted_score", "expected_excess_return_bps"],
            ascending=[True, True, True],
        ).iloc[0]
    else:
        fund_row = trim_candidates.sort_values(
            ["stress_adjusted_score", "expected_excess_return_bps"],
            ascending=[True, True],
        ).iloc[0]

    def _impact(add, fund) -> tuple[float, float]:
        expected = (add["expected_excess_return_bps"] - fund["expected_excess_return_bps"]) * 0.05
        stress = (add["recession_stress_loss_bps"] - fund["recession_stress_loss_bps"]) * 0.05
        return round(float(expected), 1), round(float(stress), 1)

    expected_impact, stress_impact = _impact(add_row, fund_row)
    rows = [
        {
            "action": "Add",
            "bucket": add_row["bucket"],
            "suggested_shift": "+5%",
            "funding_source": fund_row["bucket"],
            "reason": "Best marginal stress-adjusted compensation among eligible buckets",
            "expected_return_impact_bps": expected_impact,
            "stress_impact_bps": stress_impact,
        }
    ]

    if tail_stress_share >= 35.0:
        tail = table[
            table["bucket"].isin(["B", "CCC"])
            & (table["target_weight"] > 0)
        ].sort_values(["stress_adjusted_score", "recession_stress_loss_bps"], ascending=[True, False])
        if not tail.empty:
            hedge = table.loc[table["bucket"] == "Hedge"].iloc[0]
            trim = tail.iloc[0]
            expected_impact, stress_impact = _impact(hedge, trim)
            rows.append(
                {
                    "action": "Trim",
                    "bucket": trim["bucket"],
                    "suggested_shift": "-5%",
                    "funding_source": "Hedge",
                    "reason": "B/CCC stress share is elevated; use hedge to reduce tail beta",
                    "expected_return_impact_bps": expected_impact,
                    "stress_impact_bps": stress_impact,
                }
            )

    weak_credit = credit[
        (credit["target_weight"] > 0)
        & (credit["expected_excess_return_bps"] < 0)
    ].sort_values("expected_excess_return_bps")
    if not weak_credit.empty:
        weak = weak_credit.iloc[0]
        cash = table.loc[table["bucket"] == "Cash"].iloc[0]
        expected_impact, stress_impact = _impact(cash, weak)
        rows.append(
            {
                "action": "Trim",
                "bucket": weak["bucket"],
                "suggested_shift": "-5%",
                "funding_source": "Cash",
                "reason": "Negative expected excess return does not justify incremental credit beta",
                "expected_return_impact_bps": expected_impact,
                "stress_impact_bps": stress_impact,
            }
        )

    return pd.DataFrame(rows)


def _spread_shock_sensitivity(rating_weights: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    shocks = [25, 50, 100]
    rows = []
    totals = {shock: 0.0 for shock in shocks}
    hedge_offsets = {shock: 0.0 for shock in shocks}

    for bucket in RATING_BUCKETS:
        assumptions = BUCKET_ANALYTICS[bucket]
        weight = rating_weights.get(bucket, 0.0)
        row = {"bucket": bucket, "target_weight": round(weight, 1)}
        for shock in shocks:
            loss = assumptions["duration"] * assumptions["spread_beta"] * shock
            weighted_loss = loss * weight / 100.0
            row[f"+{shock}bps_loss_bps"] = round(float(loss), 1)
            row[f"weighted_+{shock}bps_bps"] = round(float(weighted_loss), 1)
            totals[shock] += weighted_loss
            if bucket == "Hedge":
                hedge_offsets[shock] += -weighted_loss
        rows.append(row)

    total_row = {"bucket": "Portfolio", "target_weight": round(sum(rating_weights.values()), 1)}
    for shock in shocks:
        total_row[f"+{shock}bps_loss_bps"] = None
        total_row[f"weighted_+{shock}bps_bps"] = round(float(totals[shock]), 1)
    rows.append(total_row)

    summary = {
        f"portfolio_+{shock}bps_loss_bps": round(float(totals[shock]), 1)
        for shock in shocks
    }
    summary.update({
        f"hedge_+{shock}bps_offset_bps": round(float(hedge_offsets[shock]), 1)
        for shock in shocks
    })
    summary["interpretation"] = (
        f"A parallel +100 bps spread shock implies about {totals[100]:.0f} bps portfolio loss, "
        f"with hedges offsetting about {hedge_offsets[100]:.0f} bps."
    )
    return pd.DataFrame(rows), summary


def _net_spread_beta(rating_weights: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    rows = []
    gross_long_beta = 0.0
    hedge_beta = 0.0
    net_beta = 0.0
    net_duration_beta = 0.0

    for bucket in RATING_BUCKETS:
        assumptions = BUCKET_ANALYTICS[bucket]
        weight = rating_weights.get(bucket, 0.0)
        spread_beta = assumptions["spread_beta"]
        duration = assumptions["duration"]
        weighted_beta = spread_beta * weight / 100.0
        weighted_duration_beta = duration * spread_beta * weight / 100.0
        net_beta += weighted_beta
        net_duration_beta += weighted_duration_beta
        if spread_beta > 0:
            gross_long_beta += weighted_beta
        elif spread_beta < 0:
            hedge_beta += weighted_beta
        rows.append(
            {
                "bucket": bucket,
                "target_weight": round(weight, 1),
                "spread_beta": round(float(spread_beta), 2),
                "spread_duration": round(float(duration), 2),
                "weighted_spread_beta": round(float(weighted_beta), 3),
                "weighted_duration_beta": round(float(weighted_duration_beta), 3),
            }
        )

    rows.append(
        {
            "bucket": "Portfolio",
            "target_weight": round(sum(rating_weights.values()), 1),
            "spread_beta": None,
            "spread_duration": None,
            "weighted_spread_beta": round(float(net_beta), 3),
            "weighted_duration_beta": round(float(net_duration_beta), 3),
        }
    )
    hedge_offset_pct = abs(hedge_beta) / gross_long_beta * 100.0 if gross_long_beta else 0.0
    summary = {
        "gross_long_spread_beta": round(float(gross_long_beta), 3),
        "hedge_spread_beta": round(float(hedge_beta), 3),
        "net_spread_beta": round(float(net_beta), 3),
        "net_duration_beta": round(float(net_duration_beta), 3),
        "hedge_offset_pct": round(float(hedge_offset_pct), 1),
        "hy_equivalent_exposure_pct": round(float(net_beta * 100.0), 1),
        "parallel_100bps_loss_bps": round(float(net_duration_beta * 100.0), 1),
        "interpretation": (
            f"Net spread beta is {net_beta:.2f}x HY-equivalent after hedge beta of {hedge_beta:.2f}x; "
            f"a +100 bps spread shock maps to about {net_duration_beta * 100.0:.0f} bps price loss."
        ),
    }
    return pd.DataFrame(rows), summary


def _cdx_hedge_sizing(
    *,
    rating_weights: dict[str, float],
    recommendation: str,
    net_spread_beta_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    target_by_recommendation = {
        "Add": 0.75,
        "Hold": 0.45,
        "Upgrade Quality": 0.30,
        "Hedge": 0.20,
        "De-risk": 0.05,
    }
    cdx_duration = 4.0
    cdx_running_spread_bps = 75.0
    current_net_beta = float(net_spread_beta_summary.get("net_spread_beta", 0.0))
    gross_long_beta = float(net_spread_beta_summary.get("gross_long_spread_beta", 0.0))
    current_hedge_pct = rating_weights.get("Hedge", 0.0)
    target_net_beta = target_by_recommendation.get(recommendation, target_by_recommendation["Hold"])
    incremental_protection_pct = max(0.0, (current_net_beta - target_net_beta) * 100.0)
    post_trade_net_beta = current_net_beta - incremental_protection_pct / 100.0
    loss_reduction_bps = incremental_protection_pct / 100.0 * cdx_duration * 100.0
    carry_cost_bps = incremental_protection_pct / 100.0 * cdx_running_spread_bps
    target_hedge_pct = current_hedge_pct + incremental_protection_pct
    action = (
        "Add CDX HY protection"
        if incremental_protection_pct >= 1.0
        else "No additional CDX hedge required"
    )

    rows = [
        {"metric": "Current hedge bucket", "value": round(float(current_hedge_pct), 1), "unit": "% NAV"},
        {"metric": "Gross long spread beta", "value": round(float(gross_long_beta), 3), "unit": "x HY"},
        {"metric": "Current net spread beta", "value": round(float(current_net_beta), 3), "unit": "x HY"},
        {"metric": "Target net spread beta", "value": round(float(target_net_beta), 3), "unit": "x HY"},
        {"metric": "Incremental CDX HY protection", "value": round(float(incremental_protection_pct), 1), "unit": "% NAV"},
        {"metric": "Target hedge bucket", "value": round(float(target_hedge_pct), 1), "unit": "% NAV"},
        {"metric": "Post-trade net spread beta", "value": round(float(post_trade_net_beta), 3), "unit": "x HY"},
        {"metric": "+100 bps loss reduction", "value": round(float(loss_reduction_bps), 1), "unit": "bps NAV"},
        {"metric": "Estimated annual carry cost", "value": round(float(carry_cost_bps), 1), "unit": "bps NAV"},
    ]
    summary = {
        "action": action,
        "current_hedge_pct": round(float(current_hedge_pct), 1),
        "target_hedge_pct": round(float(target_hedge_pct), 1),
        "incremental_cdx_hy_protection_pct": round(float(incremental_protection_pct), 1),
        "target_net_spread_beta": round(float(target_net_beta), 3),
        "post_trade_net_spread_beta": round(float(post_trade_net_beta), 3),
        "estimated_annual_carry_cost_bps": round(float(carry_cost_bps), 1),
        "loss_reduction_100bps_bps": round(float(loss_reduction_bps), 1),
        "interpretation": (
            f"{action}: target net beta {target_net_beta:.2f}x versus current {current_net_beta:.2f}x; "
            f"incremental CDX HY protection {incremental_protection_pct:.1f}% NAV."
        ),
    }
    return pd.DataFrame(rows), summary


def _scenario_preset_stress(
    *,
    rating_weights: dict[str, float],
    expected_loss_bps: float | None,
) -> tuple[pd.DataFrame, dict]:
    hy_loss = expected_loss_bps if expected_loss_bps is not None else 240.0
    scenarios = [
        {
            "scenario": "Soft landing",
            "hy_shock_bps": -50.0,
            "default_loss_multiplier": 0.75,
            "primary_driver": "Carry plus modest spread tightening",
            "portfolio_action": "Add BB/B selectively if compensation remains positive",
        },
        {
            "scenario": "Recession widening",
            "hy_shock_bps": 250.0,
            "default_loss_multiplier": 1.50,
            "primary_driver": "Broad spread beta and rising expected losses",
            "portfolio_action": "Upgrade quality, raise cash, keep HY hedge",
        },
        {
            "scenario": "Liquidity shock",
            "hy_shock_bps": 350.0,
            "default_loss_multiplier": 1.10,
            "primary_driver": "Liquidity premium and ETF/index pressure",
            "portfolio_action": "Use hedges first; avoid forced selling of cash bonds",
        },
        {
            "scenario": "Fallen-angel wave",
            "hy_shock_bps": 175.0,
            "default_loss_multiplier": 1.25,
            "primary_driver": "BBB downgrade pressure and BB supply",
            "portfolio_action": "Cut BBB/BB downgrade beta and keep dry powder",
        },
        {
            "scenario": "Default-cycle shock",
            "hy_shock_bps": 450.0,
            "default_loss_multiplier": 2.00,
            "primary_driver": "Default drag plus lower-quality spread beta",
            "portfolio_action": "De-risk B/CCC, maximize hedge and cash buffers",
        },
    ]
    rows = []
    for scenario in scenarios:
        gross_loss = 0.0
        hedge_offset = 0.0
        default_drag = 0.0
        for bucket in RATING_BUCKETS:
            assumptions = BUCKET_ANALYTICS[bucket]
            weight = rating_weights.get(bucket, 0.0) / 100.0
            spread_loss = assumptions["duration"] * assumptions["spread_beta"] * scenario["hy_shock_bps"]
            bucket_default_drag = hy_loss * assumptions["loss_factor"] * (scenario["default_loss_multiplier"] - 1.0)
            weighted = (spread_loss + bucket_default_drag) * weight
            gross_loss += weighted
            default_drag += bucket_default_drag * weight
            if bucket == "Hedge":
                hedge_offset += -weighted

        rows.append(
            {
                "scenario": scenario["scenario"],
                "hy_shock_bps": scenario["hy_shock_bps"],
                "default_loss_multiplier": scenario["default_loss_multiplier"],
                "modeled_portfolio_loss_bps": round(float(gross_loss), 1),
                "hedge_offset_bps": round(float(hedge_offset), 1),
                "incremental_default_drag_bps": round(float(default_drag), 1),
                "primary_driver": scenario["primary_driver"],
                "portfolio_action": scenario["portfolio_action"],
            }
        )

    table = pd.DataFrame(rows)
    worst = table.sort_values("modeled_portfolio_loss_bps", ascending=False).iloc[0]
    summary = {
        "worst_scenario": worst["scenario"],
        "worst_scenario_loss_bps": float(worst["modeled_portfolio_loss_bps"]),
        "interpretation": (
            f"Worst preset is {worst['scenario']} at roughly "
            f"{worst['modeled_portfolio_loss_bps']:.0f} bps modeled portfolio loss."
        ),
    }
    return table, summary


def _portfolio_constraint_table(
    *,
    rating_weights: dict[str, float],
    recommendation: str,
    hy_percentile: float | None,
    sloos_change_90d: float | None,
) -> tuple[pd.DataFrame, dict]:
    b_ccc = rating_weights.get("B", 0.0) + rating_weights.get("CCC", 0.0)
    hy_beta = rating_weights.get("BB", 0.0) + b_ccc
    min_cash = 20.0 if recommendation == "De-risk" else 15.0 if recommendation == "Hedge" else 5.0
    min_hedge = 10.0 if sloos_change_90d is not None and sloos_change_90d > 3 else 5.0
    max_hy_beta = 25.0 if hy_percentile is not None and hy_percentile < 25 else 55.0
    constraints = [
        {
            "constraint": "Max CCC",
            "current": rating_weights.get("CCC", 0.0),
            "limit": 5.0,
            "direction": "max",
            "portfolio_action": "Trim CCC into BB/cash/hedge",
        },
        {
            "constraint": "Max B/CCC tail",
            "current": b_ccc,
            "limit": 25.0,
            "direction": "max",
            "portfolio_action": "Reduce lower-quality HY tail beta",
        },
        {
            "constraint": "Min cash",
            "current": rating_weights.get("Cash", 0.0),
            "limit": min_cash,
            "direction": "min",
            "portfolio_action": "Raise cash before adding credit beta",
        },
        {
            "constraint": "Min hedge",
            "current": rating_weights.get("Hedge", 0.0),
            "limit": min_hedge,
            "direction": "min",
            "portfolio_action": "Add HY/CDX protection",
        },
        {
            "constraint": "Max HY beta",
            "current": hy_beta,
            "limit": max_hy_beta,
            "direction": "max",
            "portfolio_action": "Upgrade quality or fund risk from HY beta",
        },
    ]
    rows = []
    for item in constraints:
        if item["direction"] == "max":
            breach = item["current"] > item["limit"]
            gap = item["current"] - item["limit"]
        else:
            breach = item["current"] < item["limit"]
            gap = item["limit"] - item["current"]
        rows.append(
            {
                "constraint": item["constraint"],
                "current_pct": round(float(item["current"]), 1),
                "limit_pct": round(float(item["limit"]), 1),
                "status": "Breach" if breach else "OK",
                "gap_pct": round(float(gap if breach else 0.0), 1),
                "portfolio_action": item["portfolio_action"] if breach else "No action required",
            }
        )
    table = pd.DataFrame(rows)
    breaches = int((table["status"] == "Breach").sum())
    summary = {
        "breach_count": breaches,
        "interpretation": (
            "All portfolio constraints are satisfied."
            if breaches == 0 else f"{breaches} portfolio constraint(s) breached; review sizing before implementation."
        ),
    }
    return table, summary


def _pm_final_verdict(
    *,
    recommendation: str,
    marginal_allocation_table: pd.DataFrame,
    bucket_return_summary: dict,
    shock_summary: dict,
    triggers: dict,
) -> str:
    primary_trade = "hold current risk budget"
    if marginal_allocation_table is not None and not marginal_allocation_table.empty:
        row = marginal_allocation_table.iloc[0]
        primary_trade = (
            f"{str(row['action']).lower()} {row['suggested_shift']} {row['bucket']} "
            f"funded from {row['funding_source']}"
        )

    expected = bucket_return_summary.get("portfolio_expected_excess_return_bps")
    shock = shock_summary.get("portfolio_+100bps_loss_bps")
    trigger = triggers.get("hedge_or_exit") or triggers.get("reduce_risk") or "Monitor credit triggers"
    return (
        f"{recommendation}: {primary_trade}; expected excess return {expected:.0f} bps, "
        f"+100 bps spread shock {shock:.0f} bps, key risk trigger: {trigger}."
    )


def _confidence_from_sample(sample_count: int | None) -> str:
    if sample_count is None:
        return "Rules-only"
    if sample_count >= 50:
        return "Robust"
    if sample_count >= 20:
        return "Indicative"
    if sample_count >= 8:
        return "Thin sample"
    return "Rules-only"


def _scorecard_confidence_flags(
    *,
    forward_outcomes: dict,
    bucket_return_summary: dict,
) -> tuple[pd.DataFrame, dict]:
    analog_sample = forward_outcomes.get("sample_count") if forward_outcomes.get("available") else None
    analog_conf = _confidence_from_sample(analog_sample)
    expected_source = bucket_return_summary.get("expected_hy_spread_change_source", "Rules")
    expected_conf = analog_conf if "Blended" in expected_source else "Rules-only"

    rows = pd.DataFrame([
        {
            "output": "Recommendation",
            "confidence": "Rules-only",
            "basis": "Transparent scorecard thresholds; validate with recommendation backtest",
        },
        {
            "output": "Historical analogs",
            "confidence": analog_conf,
            "basis": (
                f"{analog_sample} similar observations"
                if analog_sample is not None else forward_outcomes.get("reason", "No analog sample")
            ),
        },
        {
            "output": "Expected return model",
            "confidence": expected_conf,
            "basis": f"Expected spread move source: {expected_source}",
        },
        {
            "output": "Marginal allocation advice",
            "confidence": "Rules-only",
            "basis": "Stress-adjusted ranking from bucket assumptions",
        },
        {
            "output": "Spread shock sensitivity",
            "confidence": "Rules-only",
            "basis": "Deterministic duration x spread beta scenario",
        },
    ])
    summary = {
        "overall_confidence": (
            "Rules-only" if "Rules-only" in set(rows["confidence"]) else rows["confidence"].iloc[0]
        ),
        "analog_confidence": analog_conf,
        "expected_return_confidence": expected_conf,
    }
    return rows, summary


def _historical_forward_outcomes(
    df: pd.DataFrame,
    *,
    recommendation: str,
    hy_percentile: float | None,
    compensation_ratio: float | None,
    excess_spread_bps: float | None,
) -> dict:
    available_horizons = [
        (label, hy_col, ig_col)
        for label, hy_col, ig_col in FORWARD_HORIZONS
        if hy_col in df.columns or ig_col in df.columns
    ]
    if not available_horizons:
        return {"available": False, "reason": "Forward spread columns unavailable"}

    hist = df.iloc[:-1].copy()
    if hist.empty:
        return {"available": False, "reason": "No prior observations available"}

    decomp = decompose_spreads(hist)
    for col in ["spread_compensation_ratio", "excess_spread_bps"]:
        if col not in hist.columns and col in decomp.columns:
            hist[col] = decomp[col]

    filters = pd.Series(True, index=hist.index)
    if hy_percentile is not None and "hy_spread_percentile" in hist.columns:
        pct = pd.to_numeric(hist["hy_spread_percentile"], errors="coerce")
        filters &= pct.sub(hy_percentile).abs() <= 20
    if compensation_ratio is not None and "spread_compensation_ratio" in hist.columns:
        ratio = pd.to_numeric(hist["spread_compensation_ratio"], errors="coerce")
        filters &= ratio.sub(compensation_ratio).abs() <= 0.50
    if excess_spread_bps is not None and "excess_spread_bps" in hist.columns:
        excess = pd.to_numeric(hist["excess_spread_bps"], errors="coerce")
        filters &= excess.sub(excess_spread_bps).abs() <= 125

    similar = hist[filters].copy()
    if len(similar) < 8:
        filters = pd.Series(True, index=hist.index)
        if hy_percentile is not None and "hy_spread_percentile" in hist.columns:
            pct = pd.to_numeric(hist["hy_spread_percentile"], errors="coerce")
            filters &= pct.sub(hy_percentile).abs() <= 35
        if compensation_ratio is not None and "spread_compensation_ratio" in hist.columns:
            ratio = pd.to_numeric(hist["spread_compensation_ratio"], errors="coerce")
            filters &= ratio.sub(compensation_ratio).abs() <= 0.85
        similar = hist[filters].copy()

    if similar.empty:
        return {"available": False, "reason": "No similar historical states found"}

    rows = []
    widening_flags = []
    for label, hy_col, ig_col in available_horizons:
        hy = _series_to_bps(similar[hy_col]).dropna() if hy_col in similar.columns else pd.Series(dtype=float)
        ig = _series_to_bps(similar[ig_col]).dropna() if ig_col in similar.columns else pd.Series(dtype=float)
        if hy.empty and ig.empty:
            continue

        hy_tightened = float((hy < 0).mean() * 100.0) if not hy.empty else None
        hy_material_widen = float((hy > 50).mean() * 100.0) if not hy.empty else None
        if hy_material_widen is not None:
            widening_flags.append(hy_material_widen)

        rows.append(
            {
                "horizon": label,
                "sample_count": int(max(len(hy), len(ig))),
                "hy_median_change_bps": None if hy.empty else round(float(hy.median()), 1),
                "ig_median_change_bps": None if ig.empty else round(float(ig.median()), 1),
                "hy_tightened_pct": None if hy_tightened is None else round(hy_tightened, 0),
                "hy_material_widen_pct": None if hy_material_widen is None else round(hy_material_widen, 0),
                "hy_worst_widening_bps": None if hy.empty else round(float(hy.max()), 1),
            }
        )

    outcome_table = pd.DataFrame(rows)
    if outcome_table.empty:
        return {"available": False, "reason": "Similar states lack forward spread outcomes"}

    first = outcome_table.iloc[0]
    hy_1m = first.get("hy_median_change_bps")
    widen_risk = max(widening_flags) if widening_flags else None
    if hy_1m is not None and hy_1m < 0 and recommendation in {"Add", "Hold"}:
        interpretation = "Similar states historically favored owning carry; HY spreads typically tightened."
    elif hy_1m is not None and hy_1m > 0 and recommendation in {"Upgrade Quality", "Hedge", "De-risk"}:
        interpretation = "Historical analogs support caution; similar states skewed toward HY spread widening."
    elif widen_risk is not None and widen_risk >= 35:
        interpretation = "Forward outcomes show meaningful widening risk; keep quality and hedges prominent."
    else:
        interpretation = "Historical analogs are mixed; use the scorecard triggers rather than adding beta mechanically."

    return {
        "available": True,
        "sample_count": int(outcome_table["sample_count"].max()),
        "table": outcome_table,
        "interpretation": interpretation,
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
    rating_weights, rating_table = _rating_bucket_allocation(
        recommendation=rec,
        hy_percentile=hy_percentile,
        ig_percentile=ig_percentile,
        bbb_ig_ratio=bbb_ig_ratio,
        sloos_change_90d=sloos_change_90d,
        chargeoff_change_90d=chargeoff_change_90d,
        delinquency_change_90d=delinquency_change_90d,
    )
    forward_outcomes = _historical_forward_outcomes(
        df,
        recommendation=rec,
        hy_percentile=hy_percentile,
        compensation_ratio=compensation_ratio,
        excess_spread_bps=excess_spread_bps,
    )
    bucket_return_table, bucket_return_summary = _bucket_return_and_stress(
        rating_weights=rating_weights,
        hy_spread_bps=hy_spread_bps,
        ig_spread_bps=ig_spread_bps,
        expected_loss_bps=expected_loss_bps,
        hy_percentile=hy_percentile,
        bbb_ig_ratio=bbb_ig_ratio,
        sloos_change_90d=sloos_change_90d,
        chargeoff_change_90d=chargeoff_change_90d,
        delinquency_change_90d=delinquency_change_90d,
        forward_outcomes=forward_outcomes,
    )
    risk_reward_table, risk_reward_summary = _risk_reward_metrics(
        bucket_return_table=bucket_return_table,
        rating_weights=rating_weights,
    )
    marginal_allocation_table = _marginal_allocation_advice(
        bucket_return_table=bucket_return_table,
        rating_weights=rating_weights,
        risk_reward_summary=risk_reward_summary,
    )
    net_spread_beta_table, net_spread_beta_summary = _net_spread_beta(rating_weights)
    cdx_hedge_table, cdx_hedge_summary = _cdx_hedge_sizing(
        rating_weights=rating_weights,
        recommendation=rec,
        net_spread_beta_summary=net_spread_beta_summary,
    )
    spread_shock_table, spread_shock_summary = _spread_shock_sensitivity(rating_weights)
    scenario_preset_table, scenario_preset_summary = _scenario_preset_stress(
        rating_weights=rating_weights,
        expected_loss_bps=expected_loss_bps,
    )
    constraint_table, constraint_summary = _portfolio_constraint_table(
        rating_weights=rating_weights,
        recommendation=rec,
        hy_percentile=hy_percentile,
        sloos_change_90d=sloos_change_90d,
    )
    pm_final_verdict = _pm_final_verdict(
        recommendation=rec,
        marginal_allocation_table=marginal_allocation_table,
        bucket_return_summary=bucket_return_summary,
        shock_summary=spread_shock_summary,
        triggers=triggers,
    )
    confidence_table, confidence_summary = _scorecard_confidence_flags(
        forward_outcomes=forward_outcomes,
        bucket_return_summary=bucket_return_summary,
    )
    assumptions_table = _bucket_assumptions_table()
    largest_rating_overweights = [
        f"{row.bucket} {row.target_weight:.1f}%"
        for row in rating_table.itertuples()
        if row.tilt == "Overweight"
    ][:3]
    if largest_rating_overweights:
        trade_memo["Trade Expression"] = (
            f"{trade_memo['Trade Expression']} Rating allocation: "
            f"{', '.join(largest_rating_overweights)}."
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
        "rating_weights": rating_weights,
        "forward_outcomes": forward_outcomes,
        "bucket_return_summary": bucket_return_summary,
        "risk_reward_summary": risk_reward_summary,
        "marginal_allocation_table": marginal_allocation_table,
        "net_spread_beta_summary": net_spread_beta_summary,
        "cdx_hedge_summary": cdx_hedge_summary,
        "spread_shock_summary": spread_shock_summary,
        "scenario_preset_summary": scenario_preset_summary,
        "constraint_summary": constraint_summary,
        "pm_final_verdict": pm_final_verdict,
        "confidence_summary": confidence_summary,
    }

    rows = pd.DataFrame([
        {"metric": "HY OAS", "value": _fmt_bps(hy_spread_bps), "interpretation": "Current HY spread compensation"},
        {"metric": "IG OAS", "value": _fmt_bps(ig_spread_bps), "interpretation": "Current IG spread anchor"},
        {"metric": "HY Percentile", "value": _fmt_ordinal(hy_percentile), "interpretation": "Higher means spreads are wider vs history"},
        {"metric": "Expected Loss", "value": _fmt_bps(expected_loss_bps), "interpretation": "Regime-based PD times loss-given-default"},
        {"metric": "Excess Spread", "value": _fmt_bps(excess_spread_bps), "interpretation": "Residual spread after expected loss"},
        {"metric": "Compensation Ratio", "value": _fmt_ratio(compensation_ratio), "interpretation": "Spread divided by expected loss"},
        {"metric": "HY Carry Breakeven", "value": _fmt_bps(hy_breakeven_bps), "interpretation": "Annual widening cushion before carry is erased"},
        {"metric": "Net Spread Beta", "value": f"{net_spread_beta_summary['net_spread_beta']:.2f}x", "interpretation": "HY-equivalent spread beta after cash and hedge buckets"},
        {"metric": "CDX Hedge Sizing", "value": f"{cdx_hedge_summary['incremental_cdx_hy_protection_pct']:.1f}% NAV", "interpretation": cdx_hedge_summary["action"]},
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
        "rating_weights": rating_weights,
        "rating_bucket_table": rating_table,
        "bucket_return_table": bucket_return_table,
        "bucket_return_summary": bucket_return_summary,
        "bucket_return_summary_text": bucket_return_summary["interpretation"],
        "risk_reward_table": risk_reward_table,
        "risk_reward_summary": risk_reward_summary,
        "marginal_allocation_table": marginal_allocation_table,
        "net_spread_beta_table": net_spread_beta_table,
        "net_spread_beta_summary": net_spread_beta_summary,
        "net_spread_beta_summary_text": net_spread_beta_summary["interpretation"],
        "cdx_hedge_table": cdx_hedge_table,
        "cdx_hedge_summary": cdx_hedge_summary,
        "cdx_hedge_summary_text": cdx_hedge_summary["interpretation"],
        "spread_shock_table": spread_shock_table,
        "spread_shock_summary": spread_shock_summary,
        "spread_shock_summary_text": spread_shock_summary["interpretation"],
        "scenario_preset_table": scenario_preset_table,
        "scenario_preset_summary": scenario_preset_summary,
        "scenario_preset_summary_text": scenario_preset_summary["interpretation"],
        "constraint_table": constraint_table,
        "constraint_summary": constraint_summary,
        "constraint_summary_text": constraint_summary["interpretation"],
        "pm_final_verdict": pm_final_verdict,
        "confidence_table": confidence_table,
        "confidence_summary": confidence_summary,
        "bucket_assumptions_table": assumptions_table,
        "forward_outcomes": forward_outcomes,
        "forward_outcomes_table": forward_outcomes.get("table", pd.DataFrame()),
        "forward_outcomes_summary": forward_outcomes.get("interpretation", forward_outcomes.get("reason", "")),
        "allocation": allocation,
        "recommendation": rec,
        "drivers": drivers,
        "summary": (
            f"{rec}: HY OAS {_fmt_bps(hy_spread_bps)}, excess spread {_fmt_bps(excess_spread_bps)}, "
            f"compensation ratio {_fmt_ratio(compensation_ratio)}."
        ),
    }
