"""
Legacy versus institutional composite comparison.
"""

from __future__ import annotations

import pandas as pd

from src.credit_taxonomy import compute_channel_scores


def compare_composites(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "composite_risk_score_smooth" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    if "institutional_credit_score" not in work.columns:
        scores = compute_channel_scores(work)
        if "institutional_credit_score" in scores.columns:
            work = work.join(scores[["institutional_credit_score"]])

    if "institutional_credit_score" not in work.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=work.index)
    if "date" in work.columns:
        out["date"] = work["date"]
    out["legacy_composite"] = pd.to_numeric(work["composite_risk_score_smooth"], errors="coerce")
    out["institutional_composite"] = pd.to_numeric(work["institutional_credit_score"], errors="coerce")
    out["composite_gap"] = out["institutional_composite"] - out["legacy_composite"]
    out["abs_gap"] = out["composite_gap"].abs()
    return out


def composite_comparison_summary(df: pd.DataFrame) -> dict:
    comp = compare_composites(df).dropna(subset=["legacy_composite", "institutional_composite"])
    if comp.empty:
        return {"available": False, "reason": "Composite comparison unavailable"}

    corr = comp["legacy_composite"].corr(comp["institutional_composite"])
    latest = comp.iloc[-1]
    largest = comp.sort_values("abs_gap", ascending=False).head(5)
    return {
        "available": True,
        "correlation": None if pd.isna(corr) else float(corr),
        "latest_legacy": float(latest["legacy_composite"]),
        "latest_institutional": float(latest["institutional_composite"]),
        "latest_gap": float(latest["composite_gap"]),
        "mean_abs_gap": float(comp["abs_gap"].mean()),
        "large_gap_count": int((comp["abs_gap"] >= 15).sum()),
        "largest_gaps": largest.reset_index(drop=True),
    }


def composite_governance_note(df: pd.DataFrame) -> str:
    summary = composite_comparison_summary(df)
    if not summary.get("available"):
        return summary.get("reason", "Composite comparison unavailable.")
    return (
        f"Legacy vs institutional composite correlation is {summary['correlation']:.2f}; "
        f"latest gap is {summary['latest_gap']:+.1f} points; "
        f"{summary['large_gap_count']} observations have gaps >=15 points. "
        "Do not replace the production composite until disagreement periods are reviewed."
    )

