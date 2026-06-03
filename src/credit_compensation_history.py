"""Persistence helpers for credit compensation scorecard history."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.credit_compensation_scorecard import build_credit_compensation_scorecard


HISTORY_PATH = Path("history") / "credit_compensation_scorecard_history.csv"

HISTORY_COLUMNS = [
    "as_of",
    "recorded_at",
    "recommendation",
    "hy_oas_bps",
    "ig_oas_bps",
    "expected_loss_bps",
    "excess_spread_bps",
    "spread_compensation_ratio",
    "hy_spread_percentile",
    "ig_spread_percentile",
    "composite_risk_score",
    "sloos_change_90d",
    "chargeoff_change_90d",
    "delinquency_change_90d",
    "ig_weight_pct",
    "bbb_weight_pct",
    "bb_weight_pct",
    "b_weight_pct",
    "ccc_weight_pct",
    "cash_weight_pct",
    "hedge_weight_pct",
    "net_spread_beta",
    "gross_long_spread_beta",
    "hedge_offset_pct",
    "target_net_spread_beta",
    "incremental_cdx_hy_protection_pct",
    "post_trade_net_spread_beta",
    "constraint_breach_count",
    "breached_constraints",
    "pm_final_verdict",
]

SPREAD_TREND_COLUMNS = ["hy_oas_bps", "expected_loss_bps", "excess_spread_bps"]
BETA_TREND_COLUMNS = ["net_spread_beta", "target_net_spread_beta"]
HEDGE_TREND_COLUMNS = ["incremental_cdx_hy_protection_pct", "constraint_breach_count"]
WEIGHT_TREND_COLUMNS = [
    "ig_weight_pct",
    "bbb_weight_pct",
    "bb_weight_pct",
    "b_weight_pct",
    "ccc_weight_pct",
    "cash_weight_pct",
    "hedge_weight_pct",
]


def _as_date_string(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).date())


def _round_or_none(value, digits: int = 2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def build_scorecard_history_row(
    df: pd.DataFrame,
    *,
    recorded_at: datetime | None = None,
) -> dict:
    """Build one persistable scorecard snapshot from a scored dashboard frame."""
    scorecard = build_credit_compensation_scorecard(df)
    if not scorecard.get("available"):
        return {
            "available": False,
            "reason": scorecard.get("reason", "Scorecard unavailable"),
        }

    current = scorecard.get("current", {})
    weights = scorecard.get("rating_weights", {})
    net_beta = scorecard.get("net_spread_beta_summary", {})
    cdx_hedge = scorecard.get("cdx_hedge_summary", {})
    constraints = scorecard.get("constraint_table", pd.DataFrame())
    breached = []
    if constraints is not None and not constraints.empty and "status" in constraints.columns:
        breached = list(constraints.loc[constraints["status"] == "Breach", "constraint"])
    as_of = scorecard.get("audit_summary", {}).get("as_of")
    now = recorded_at or datetime.now()

    row = {
        "as_of": _as_date_string(as_of),
        "recorded_at": now.isoformat(timespec="seconds"),
        "recommendation": scorecard.get("recommendation"),
        "hy_oas_bps": _round_or_none(current.get("hy_oas_bps"), 1),
        "ig_oas_bps": _round_or_none(current.get("ig_oas_bps"), 1),
        "expected_loss_bps": _round_or_none(current.get("expected_loss_bps"), 1),
        "excess_spread_bps": _round_or_none(current.get("excess_spread_bps"), 1),
        "spread_compensation_ratio": _round_or_none(current.get("spread_compensation_ratio"), 3),
        "hy_spread_percentile": _round_or_none(current.get("hy_spread_percentile"), 1),
        "ig_spread_percentile": _round_or_none(current.get("ig_spread_percentile"), 1),
        "composite_risk_score": _round_or_none(current.get("composite_risk_score"), 1),
        "sloos_change_90d": _round_or_none(current.get("sloos_change_90d"), 2),
        "chargeoff_change_90d": _round_or_none(current.get("chargeoff_change_90d"), 2),
        "delinquency_change_90d": _round_or_none(current.get("delinquency_change_90d"), 2),
        "ig_weight_pct": _round_or_none(weights.get("IG"), 1),
        "bbb_weight_pct": _round_or_none(weights.get("BBB"), 1),
        "bb_weight_pct": _round_or_none(weights.get("BB"), 1),
        "b_weight_pct": _round_or_none(weights.get("B"), 1),
        "ccc_weight_pct": _round_or_none(weights.get("CCC"), 1),
        "cash_weight_pct": _round_or_none(weights.get("Cash"), 1),
        "hedge_weight_pct": _round_or_none(weights.get("Hedge"), 1),
        "net_spread_beta": _round_or_none(net_beta.get("net_spread_beta"), 3),
        "gross_long_spread_beta": _round_or_none(net_beta.get("gross_long_spread_beta"), 3),
        "hedge_offset_pct": _round_or_none(net_beta.get("hedge_offset_pct"), 1),
        "target_net_spread_beta": _round_or_none(cdx_hedge.get("target_net_spread_beta"), 3),
        "incremental_cdx_hy_protection_pct": _round_or_none(
            cdx_hedge.get("incremental_cdx_hy_protection_pct"), 1
        ),
        "post_trade_net_spread_beta": _round_or_none(cdx_hedge.get("post_trade_net_spread_beta"), 3),
        "constraint_breach_count": int(len(breached)),
        "breached_constraints": "; ".join(str(item) for item in breached),
        "pm_final_verdict": scorecard.get("pm_final_verdict", ""),
    }
    return {"available": True, "row": {col: row.get(col) for col in HISTORY_COLUMNS}}


def load_scorecard_history(history_path: Path | str = HISTORY_PATH) -> pd.DataFrame:
    """Load persisted scorecard history, returning an empty typed frame if absent."""
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        history = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    for col in HISTORY_COLUMNS:
        if col not in history.columns:
            history[col] = pd.NA
    return history[HISTORY_COLUMNS]


def build_scorecard_trend_views(history: pd.DataFrame) -> dict:
    """Return chart-ready scorecard history views for the dashboard."""
    if history is None or history.empty:
        empty = pd.DataFrame()
        return {
            "available": False,
            "reason": "Scorecard history unavailable",
            "history": empty,
            "recommendation_timeline": empty,
            "spread_trends": empty,
            "compensation_trend": empty,
            "beta_trends": empty,
            "hedge_trends": empty,
            "weight_trends": empty,
            "has_charts": False,
        }

    work = history.copy()
    work["as_of"] = pd.to_datetime(work["as_of"], errors="coerce")
    work = work.dropna(subset=["as_of"]).sort_values("as_of")
    for col in set(SPREAD_TREND_COLUMNS + BETA_TREND_COLUMNS + HEDGE_TREND_COLUMNS + WEIGHT_TREND_COLUMNS + ["spread_compensation_ratio"]):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    recommendation_timeline = work[["as_of", "recommendation"]].copy()
    spread_trends = work[["as_of"] + [col for col in SPREAD_TREND_COLUMNS if col in work.columns]].set_index("as_of")
    compensation_trend = work[["as_of", "spread_compensation_ratio"]].set_index("as_of") if "spread_compensation_ratio" in work.columns else pd.DataFrame()
    beta_trends = work[["as_of"] + [col for col in BETA_TREND_COLUMNS if col in work.columns]].set_index("as_of")
    hedge_trends = work[["as_of"] + [col for col in HEDGE_TREND_COLUMNS if col in work.columns]].set_index("as_of")
    weight_trends = work[["as_of"] + [col for col in WEIGHT_TREND_COLUMNS if col in work.columns]].set_index("as_of")

    return {
        "available": True,
        "reason": "",
        "history": work,
        "recommendation_timeline": recommendation_timeline,
        "spread_trends": spread_trends,
        "compensation_trend": compensation_trend,
        "beta_trends": beta_trends,
        "hedge_trends": hedge_trends,
        "weight_trends": weight_trends,
        "has_charts": len(work) >= 2,
    }


def append_scorecard_history(
    df: pd.DataFrame,
    *,
    history_path: Path | str = HISTORY_PATH,
    recorded_at: datetime | None = None,
) -> dict:
    """Upsert the latest scorecard snapshot into the persisted daily history CSV."""
    built = build_scorecard_history_row(df, recorded_at=recorded_at)
    if not built.get("available"):
        return built

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = built["row"]
    history = load_scorecard_history(path)
    if not history.empty and row.get("as_of") is not None:
        history = history[history["as_of"].astype(str) != str(row["as_of"])]
    output = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    output["as_of_sort"] = pd.to_datetime(output["as_of"], errors="coerce")
    output = output.sort_values(["as_of_sort", "recorded_at"]).drop(columns=["as_of_sort"])
    output.to_csv(path, index=False)
    return {
        "available": True,
        "path": str(path),
        "as_of": row.get("as_of"),
        "row_count": int(len(output)),
        "row": row,
    }
