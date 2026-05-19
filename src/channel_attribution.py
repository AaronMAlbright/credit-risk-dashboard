"""
Institutional channel contribution attribution.
"""

from __future__ import annotations

import pandas as pd

from src.credit_taxonomy import CHANNELS, compute_channel_scores


def channel_contribution_table(df: pd.DataFrame, lookback_days: int = 21) -> pd.DataFrame:
    scores = compute_channel_scores(df)
    if scores.empty:
        return pd.DataFrame()

    rows = []
    latest = scores.iloc[-1]
    prior = scores.iloc[-lookback_days - 1] if len(scores) > lookback_days else None

    for channel in CHANNELS:
        score_col = f"{channel.key}_channel_score"
        coverage_col = f"{channel.key}_channel_coverage"
        score = latest.get(score_col)
        if pd.isna(score):
            continue

        contribution = float(score) * channel.weight
        change = None
        contribution_change = None
        if prior is not None and score_col in scores.columns and pd.notna(prior.get(score_col)):
            change = float(score) - float(prior.get(score_col))
            contribution_change = change * channel.weight

        rows.append(
            {
                "channel": channel.name,
                "weight": channel.weight,
                "score": float(score),
                "contribution": contribution,
                "score_change_1m": change,
                "contribution_change_1m": contribution_change,
                "coverage": float(latest.get(coverage_col, 0.0)),
                "description": channel.description,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)
    total_contribution = out["contribution"].sum()
    if total_contribution:
        out["contribution_share"] = out["contribution"] / total_contribution
    else:
        out["contribution_share"] = 0.0
    return out


def top_channel_drivers(df: pd.DataFrame, n: int = 3) -> list[str]:
    table = channel_contribution_table(df)
    if table.empty:
        return []
    return [
        f"{row.channel}: {row.score:.1f} score, {row.contribution_share:.0%} of channel contribution"
        for row in table.head(n).itertuples()
    ]

