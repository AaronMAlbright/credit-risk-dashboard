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
