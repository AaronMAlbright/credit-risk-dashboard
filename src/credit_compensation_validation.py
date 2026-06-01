"""
Historical validation for the credit compensation scorecard.

The goal is to test whether the scorecard's Add/Hold/Upgrade/Hedge/De-risk
recommendations line up with subsequent HY and IG spread outcomes.
"""
from __future__ import annotations

import pandas as pd

from src.credit_compensation_scorecard import _latest_value, _recommendation, _safe_float
from src.credit_regime_performance import HORIZONS, add_forward_market_moves, confidence_flag
from src.spread_decomposition import decompose_spreads


DEFENSIVE_RECOMMENDATIONS = {"Upgrade Quality", "Hedge", "De-risk"}
OFFENSIVE_RECOMMENDATIONS = {"Add"}
NEUTRAL_RECOMMENDATIONS = {"Hold"}
STRESS_EPISODES = [
    ("2018 Q4 growth scare", "2018-10-03", "2018-12-24"),
    ("COVID liquidity shock", "2020-02-19", "2020-03-23"),
    ("2022 hiking shock", "2022-01-03", "2022-10-14"),
    ("Regional bank shock", "2023-03-08", "2023-03-24"),
]


def _as_bps(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median_abs = s.dropna().abs().median()
    if pd.notna(median_abs) and median_abs < 10:
        return s * 100.0
    return s


def _row_float(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    return _safe_float(row.get(column))


def _scorecard_recommendation_series(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object, index=df.index)

    decomp = decompose_spreads(df)
    out = []
    for idx, row in df.iterrows():
        dec = decomp.loc[idx] if idx in decomp.index and not decomp.empty else pd.Series(dtype=float)
        rec, _drivers = _recommendation(
            compensation_ratio=_safe_float(dec.get("spread_compensation_ratio")),
            excess_spread_bps=_safe_float(dec.get("excess_spread_bps")),
            hy_percentile=_row_float(row, "hy_spread_percentile"),
            composite_score=_row_float(row, "composite_risk_score_smooth"),
            sloos_change_90d=_row_float(row, "sloos_change_90d"),
            chargeoff_change_90d=_row_float(row, "chargeoff_change_90d"),
            delinquency_change_90d=_row_float(row, "delinquency_change_90d"),
        )
        out.append(rec)
    return pd.Series(out, index=df.index, name="scorecard_recommendation")


def add_scorecard_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with historical scorecard recommendations."""
    out = df.copy()
    out["scorecard_recommendation"] = _scorecard_recommendation_series(out)
    return out


def _ensure_forward_spread_changes(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    work = df.copy()
    if "ig_spread" not in work.columns and "ig_spread_bps" in work.columns:
        work["ig_spread"] = work["ig_spread_bps"]
    work = add_forward_market_moves(work, horizons)
    for h in horizons:
        for col in [f"hy_spread_forward_{h}d_change", f"ig_spread_forward_{h}d_change"]:
            if col in work.columns:
                work[col] = _as_bps(work[col])
    return work


def _favorable_hit_rate(recommendation: str, hy_changes: pd.Series) -> float | None:
    s = pd.to_numeric(hy_changes, errors="coerce").dropna()
    if s.empty:
        return None
    if recommendation in OFFENSIVE_RECOMMENDATIONS:
        return float((s < 0).mean() * 100.0)
    if recommendation in DEFENSIVE_RECOMMENDATIONS:
        return float((s > 0).mean() * 100.0)
    return float((s.abs() <= 50.0).mean() * 100.0)


def _excess_return_proxy(recommendation: str, hy_changes: pd.Series) -> float | None:
    s = pd.to_numeric(hy_changes, errors="coerce").dropna()
    if s.empty:
        return None
    if recommendation in DEFENSIVE_RECOMMENDATIONS:
        return float(s.mean())
    return float((-s).mean())


def _scorecard_outcome_label(
    recommendation: str,
    hy_change_bps: float,
    materiality_bps: float,
) -> str:
    if recommendation in OFFENSIVE_RECOMMENDATIONS:
        if hy_change_bps <= -materiality_bps:
            return "true_positive"
        if hy_change_bps >= materiality_bps:
            return "false_positive"
        return "noise"
    if recommendation in DEFENSIVE_RECOMMENDATIONS:
        if hy_change_bps >= materiality_bps:
            return "true_positive"
        if hy_change_bps <= -materiality_bps:
            return "false_positive"
        return "noise"
    if recommendation in NEUTRAL_RECOMMENDATIONS:
        if hy_change_bps <= -materiality_bps:
            return "false_negative_missed_rally"
        if hy_change_bps >= materiality_bps:
            return "false_negative_missed_widening"
        return "true_negative"
    return "unclassified"


def _scorecard_error_reason(
    recommendation: str,
    hy_change_bps: float,
    materiality_bps: float,
) -> str:
    if recommendation in OFFENSIVE_RECOMMENDATIONS and hy_change_bps >= materiality_bps:
        return "Add call was followed by material HY widening"
    if recommendation in DEFENSIVE_RECOMMENDATIONS and hy_change_bps <= -materiality_bps:
        return f"{recommendation} call was followed by material HY tightening"
    if recommendation in NEUTRAL_RECOMMENDATIONS and hy_change_bps <= -materiality_bps:
        return "Hold missed a material HY tightening rally"
    if recommendation in NEUTRAL_RECOMMENDATIONS and hy_change_bps >= materiality_bps:
        return "Hold missed material HY widening risk"
    return "Outcome inside materiality band"


def _nearest_index(index: pd.Index, date: pd.Timestamp) -> pd.Timestamp | None:
    if len(index) == 0:
        return None
    ordered = pd.DatetimeIndex(index).sort_values()
    loc = ordered.searchsorted(date)
    candidates = []
    if loc < len(ordered):
        candidates.append(ordered[loc])
    if loc > 0:
        candidates.append(ordered[loc - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item - date))


def validate_scorecard_recommendations(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    """
    Summarize forward spread outcomes by scorecard recommendation.

    Spread changes are positive when spreads widen. For Add, a favorable hit is
    HY tightening. For Upgrade Quality/Hedge/De-risk, a favorable hit is HY
    widening, because the recommendation avoids or hedges credit beta.
    """
    if df.empty or "hy_spread" not in df.columns:
        return {"available": False, "reason": "hy_spread unavailable"}

    work = add_scorecard_recommendations(df)
    work = _ensure_forward_spread_changes(work, horizons)
    if "scorecard_recommendation" not in work.columns:
        return {"available": False, "reason": "scorecard recommendation unavailable"}

    rows = []
    for recommendation, group in work.groupby("scorecard_recommendation", dropna=True):
        for h in horizons:
            hy_col = f"hy_spread_forward_{h}d_change"
            ig_col = f"ig_spread_forward_{h}d_change"
            if hy_col not in group.columns:
                continue

            hy = pd.to_numeric(group[hy_col], errors="coerce").dropna()
            ig = pd.to_numeric(group[ig_col], errors="coerce").dropna() if ig_col in group.columns else pd.Series(dtype=float)
            if hy.empty:
                continue

            n_obs = int(len(hy))
            rows.append(
                {
                    "recommendation": recommendation,
                    "horizon_days": h,
                    "n_obs": n_obs,
                    "confidence": confidence_flag(n_obs),
                    "hy_median_change_bps": round(float(hy.median()), 1),
                    "ig_median_change_bps": None if ig.empty else round(float(ig.median()), 1),
                    "hy_tightened_pct": round(float((hy < 0).mean() * 100.0), 0),
                    "hy_widened_pct": round(float((hy > 0).mean() * 100.0), 0),
                    "favorable_hit_rate_pct": round(_favorable_hit_rate(recommendation, hy), 0),
                    "worst_hy_widening_bps": round(float(hy.max()), 1),
                    "avg_excess_return_proxy_bps": round(_excess_return_proxy(recommendation, hy), 1),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return {"available": False, "reason": "forward spread outcomes unavailable"}

    current_rec = _latest_value(work, "scorecard_recommendation")
    current_rows = table[table["recommendation"] == current_rec] if current_rec is not None else pd.DataFrame()
    summary = "Scorecard validation available across historical recommendations."
    if not current_rows.empty:
        pref = current_rows[current_rows["horizon_days"] == 63]
        row = pref.iloc[0] if not pref.empty else current_rows.iloc[0]
        summary = (
            f"Current recommendation {current_rec}: {int(row.n_obs)} historical observations "
            f"at {int(row.horizon_days)}d, {row.favorable_hit_rate_pct:.0f}% favorable hit rate, "
            f"median HY change {row.hy_median_change_bps:.0f} bps ({row.confidence})."
        )

    return {
        "available": True,
        "table": table,
        "summary": summary,
        "current_recommendation": current_rec,
    }


def analyze_scorecard_prediction_errors(
    df: pd.DataFrame,
    horizon_days: int = 63,
    materiality_bps: float = 25.0,
) -> dict:
    """
    Identify historical scorecard false positives and false negatives.

    False positives are active calls that were followed by an unfavorable
    material move: Add before widening, or defensive recommendations before
    tightening. False negatives are Hold calls that missed a material HY move.
    """
    if df.empty or "hy_spread" not in df.columns:
        return {"available": False, "reason": "hy_spread unavailable"}

    work = add_scorecard_recommendations(df)
    work = _ensure_forward_spread_changes(work, (horizon_days,))
    hy_col = f"hy_spread_forward_{horizon_days}d_change"
    if hy_col not in work.columns:
        return {"available": False, "reason": "forward spread outcomes unavailable"}

    rows = []
    for idx, row in work.dropna(subset=["scorecard_recommendation", hy_col]).iterrows():
        recommendation = str(row["scorecard_recommendation"])
        hy_change = _safe_float(row[hy_col])
        if hy_change is None:
            continue

        classification = _scorecard_outcome_label(recommendation, hy_change, materiality_bps)
        rows.append(
            {
                "date": idx,
                "recommendation": recommendation,
                f"hy_forward_{horizon_days}d_bps": round(float(hy_change), 1),
                "classification": classification,
                "reason": _scorecard_error_reason(recommendation, hy_change, materiality_bps),
            }
        )

    classified = pd.DataFrame(rows)
    if classified.empty:
        return {"available": False, "reason": "no classified outcomes"}

    error_mask = classified["classification"].isin(
        ["false_positive", "false_negative_missed_rally", "false_negative_missed_widening"]
    )
    error_table = classified[error_mask].copy()
    false_positive_table = classified[classified["classification"] == "false_positive"].copy()
    false_negative_table = classified[
        classified["classification"].isin(["false_negative_missed_rally", "false_negative_missed_widening"])
    ].copy()

    summary_table = (
        classified.groupby(["recommendation", "classification"], dropna=True)
        .size()
        .reset_index(name="n_obs")
    )
    summary_by_recommendation = (
        classified.assign(is_error=error_mask)
        .groupby("recommendation", dropna=True)
        .agg(n_obs=("classification", "size"), error_count=("is_error", "sum"))
        .reset_index()
    )
    summary_by_recommendation["error_rate_pct"] = (
        summary_by_recommendation["error_count"] / summary_by_recommendation["n_obs"] * 100.0
    ).round(1)
    summary_by_recommendation["confidence"] = summary_by_recommendation["n_obs"].apply(confidence_flag)

    fp_count = int(len(false_positive_table))
    fn_count = int(len(false_negative_table))
    n_obs = int(len(classified))
    error_rate = (fp_count + fn_count) / n_obs * 100.0 if n_obs else 0.0
    worst_error = None
    if not error_table.empty:
        move_col = f"hy_forward_{horizon_days}d_bps"
        worst = error_table.iloc[error_table[move_col].abs().argmax()]
        worst_error = {
            "date": worst["date"],
            "recommendation": worst["recommendation"],
            "classification": worst["classification"],
            "hy_forward_change_bps": float(worst[move_col]),
            "reason": worst["reason"],
        }

    summary = {
        "n_obs": n_obs,
        "horizon_days": horizon_days,
        "materiality_bps": materiality_bps,
        "false_positive_count": fp_count,
        "false_negative_count": fn_count,
        "error_rate_pct": round(float(error_rate), 1),
        "worst_error": worst_error,
        "interpretation": (
            f"{fp_count} active-call false positives and {fn_count} Hold false negatives "
            f"over {n_obs} scored observations at a {horizon_days}d horizon."
        ),
    }

    return {
        "available": True,
        "classified_table": classified,
        "false_positive_table": false_positive_table,
        "false_negative_table": false_negative_table,
        "error_table": error_table,
        "summary_table": summary_table,
        "summary_by_recommendation": summary_by_recommendation,
        "summary": summary,
        "summary_text": summary["interpretation"],
    }


def build_scorecard_validation_report(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict:
    """Build downloadable markdown and CSV validation artifacts."""
    result = validate_scorecard_recommendations(df, horizons=horizons)
    if not result.get("available"):
        return result

    table = result["table"].copy()
    markdown_table = _markdown_table(table)
    current = result.get("current_recommendation", "Unavailable")
    lines = [
        "# Credit Compensation Scorecard Validation",
        "",
        f"Current scorecard recommendation: **{current}**",
        "",
        result.get("summary", ""),
        "",
        "## Validation Table",
        "",
        markdown_table,
        "",
        "## Interpretation",
        "",
        "- Positive HY spread changes indicate widening, which is unfavorable for long credit beta.",
        "- For Add, favorable hit rate measures subsequent HY tightening.",
        "- For Upgrade Quality, Hedge, and De-risk, favorable hit rate measures subsequent HY widening avoided or hedged.",
        "- Confidence is based on historical sample count and should govern how strongly results are used.",
    ]
    return {
        **result,
        "markdown": "\n".join(lines),
        "csv": table.to_csv(index=False),
    }


def replay_scorecard_stress_episodes(
    df: pd.DataFrame,
    episodes: list[tuple[str, str, str]] | None = None,
) -> dict:
    """Replay named stress windows against historical scorecard recommendations."""
    if df.empty or "hy_spread" not in df.columns:
        return {"available": False, "reason": "hy_spread unavailable"}

    work = add_scorecard_recommendations(df)
    if "ig_spread" not in work.columns and "ig_spread_bps" in work.columns:
        work["ig_spread"] = work["ig_spread_bps"]
    if not isinstance(work.index, pd.DatetimeIndex):
        return {"available": False, "reason": "datetime index required"}

    work = work.sort_index()
    episode_defs = episodes or STRESS_EPISODES
    rows = []
    for name, start_raw, end_raw in episode_defs:
        start_target = pd.Timestamp(start_raw)
        end_target = pd.Timestamp(end_raw)
        if end_target < work.index.min() or start_target > work.index.max():
            continue
        start_idx = _nearest_index(work.index, start_target)
        end_idx = _nearest_index(work.index, end_target)
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            continue

        window = work.loc[start_idx:end_idx]
        if window.empty:
            continue
        start_row = work.loc[start_idx]
        end_row = work.loc[end_idx]
        rec = str(start_row.get("scorecard_recommendation"))
        start_hy = _safe_float(start_row.get("hy_spread"))
        end_hy = _safe_float(end_row.get("hy_spread"))
        start_ig = _safe_float(start_row.get("ig_spread")) if "ig_spread" in work.columns else None
        end_ig = _safe_float(end_row.get("ig_spread")) if "ig_spread" in work.columns else None
        if start_hy is None or end_hy is None:
            continue

        hy_window = _as_bps(window["hy_spread"])
        hy_change = _as_bps(pd.Series([end_hy - start_hy])).iloc[0]
        max_hy_widening = hy_window.max() - _as_bps(pd.Series([start_hy])).iloc[0]
        ig_change = None
        if start_ig is not None and end_ig is not None:
            ig_change = _as_bps(pd.Series([end_ig - start_ig])).iloc[0]

        if rec in DEFENSIVE_RECOMMENDATIONS:
            assessment = "Protected before stress"
        elif rec in OFFENSIVE_RECOMMENDATIONS:
            assessment = "Risk-on into stress"
        else:
            assessment = "Neutral into stress"

        rows.append(
            {
                "episode": name,
                "start": start_idx,
                "end": end_idx,
                "start_recommendation": rec,
                "hy_change_bps": round(float(hy_change), 1),
                "max_hy_widening_bps": round(float(max_hy_widening), 1),
                "ig_change_bps": None if ig_change is None else round(float(ig_change), 1),
                "assessment": assessment,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return {"available": False, "reason": "no configured stress episodes overlap available history"}

    protected = int(table["assessment"].eq("Protected before stress").sum())
    risk_on = int(table["assessment"].eq("Risk-on into stress").sum())
    summary = {
        "episode_count": int(len(table)),
        "protected_count": protected,
        "risk_on_count": risk_on,
        "worst_episode": table.sort_values("max_hy_widening_bps", ascending=False).iloc[0]["episode"],
        "interpretation": (
            f"Replay covers {len(table)} stress episode(s): {protected} began defensive, "
            f"{risk_on} began risk-on."
        ),
    }
    return {
        "available": True,
        "table": table,
        "summary": summary,
        "summary_text": summary["interpretation"],
    }


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return ""
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in table.itertuples(index=False):
        values = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze_scorecard_transitions(
    df: pd.DataFrame,
    horizon_days: int = 21,
) -> dict:
    """Measure scorecard recommendation stability and outcomes after transitions."""
    if df.empty or "hy_spread" not in df.columns:
        return {"available": False, "reason": "hy_spread unavailable"}

    work = add_scorecard_recommendations(df)
    work = _ensure_forward_spread_changes(work, (horizon_days,))
    rec = work["scorecard_recommendation"].dropna()
    if len(rec) < 2:
        return {"available": False, "reason": "not enough recommendations"}

    transitions = pd.DataFrame({
        "from_recommendation": rec.shift(1),
        "to_recommendation": rec,
    }).dropna()
    transitions = transitions[transitions["from_recommendation"] != transitions["to_recommendation"]]

    matrix = pd.crosstab(
        transitions["from_recommendation"],
        transitions["to_recommendation"],
        dropna=False,
    )
    matrix_table = matrix.reset_index().rename(columns={"from_recommendation": "from"})

    episodes = []
    start_idx = rec.index[0]
    current = rec.iloc[0]
    length = 1
    for idx, value in rec.iloc[1:].items():
        if value == current:
            length += 1
        else:
            episodes.append({"recommendation": current, "start": start_idx, "end": rec.index[rec.index.get_loc(idx) - 1], "duration_days": length})
            start_idx = idx
            current = value
            length = 1
    episodes.append({"recommendation": current, "start": start_idx, "end": rec.index[-1], "duration_days": length})
    episodes_df = pd.DataFrame(episodes)

    duration_table = (
        episodes_df.groupby("recommendation", dropna=True)["duration_days"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={
            "count": "episode_count",
            "mean": "avg_duration_days",
            "median": "median_duration_days",
            "min": "shortest_duration_days",
            "max": "longest_duration_days",
        })
    )
    for col in ["avg_duration_days", "median_duration_days"]:
        duration_table[col] = duration_table[col].round(1)

    hy_col = f"hy_spread_forward_{horizon_days}d_change"
    transition_outcomes = transitions.copy()
    transition_outcomes["transition"] = (
        transition_outcomes["from_recommendation"].astype(str)
        + " -> "
        + transition_outcomes["to_recommendation"].astype(str)
    )
    if hy_col in work.columns:
        transition_outcomes["hy_forward_change_bps"] = _as_bps(work.loc[transition_outcomes.index, hy_col])
    outcome_table = (
        transition_outcomes.groupby("transition", dropna=True)["hy_forward_change_bps"]
        .agg(["count", "median", "max"])
        .reset_index()
        .rename(columns={
            "count": "n_obs",
            "median": f"median_hy_forward_{horizon_days}d_bps",
            "max": f"worst_hy_widening_{horizon_days}d_bps",
        })
    ) if "hy_forward_change_bps" in transition_outcomes.columns else pd.DataFrame()
    if not outcome_table.empty:
        for col in [f"median_hy_forward_{horizon_days}d_bps", f"worst_hy_widening_{horizon_days}d_bps"]:
            outcome_table[col] = outcome_table[col].round(1)
        outcome_table["confidence"] = outcome_table["n_obs"].apply(confidence_flag)

    transition_count = int(len(transitions))
    one_day_episodes = int((episodes_df["duration_days"] <= 1).sum())
    whipsaw_rate = one_day_episodes / len(episodes_df) * 100.0 if len(episodes_df) else 0.0
    most_common = None
    if not transitions.empty:
        most_common = (
            transitions["from_recommendation"].astype(str)
            + " -> "
            + transitions["to_recommendation"].astype(str)
        ).value_counts().idxmax()

    summary = {
        "transition_count": transition_count,
        "episode_count": int(len(episodes_df)),
        "one_day_episode_count": one_day_episodes,
        "whipsaw_rate_pct": round(float(whipsaw_rate), 1),
        "most_common_transition": most_common,
        "interpretation": (
            f"{transition_count} transitions across {len(episodes_df)} episodes; "
            f"whipsaw rate {whipsaw_rate:.1f}%."
        ),
    }

    return {
        "available": True,
        "matrix_table": matrix_table,
        "duration_table": duration_table,
        "transition_outcome_table": outcome_table,
        "episode_table": episodes_df,
        "summary": summary,
        "summary_text": summary["interpretation"],
    }
