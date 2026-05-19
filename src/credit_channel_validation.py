"""
Forward validation for the institutional credit channel framework.
"""

from __future__ import annotations

import pandas as pd

from src.credit_regime_performance import HORIZONS, confidence_flag, add_forward_market_moves
from src.credit_taxonomy import CHANNELS, compute_channel_scores


def _safe_pct(x):
    if x is None or pd.isna(x):
        return None
    return float(x)


def channel_validation_table(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    score_quantile: float = 0.75,
) -> pd.DataFrame:
    """
    Validate high-score channel states against forward market moves.

    High-score days are defined as rows where the channel score is at or above
    the channel's historical score_quantile threshold.
    """
    if df.empty:
        return pd.DataFrame()

    scores = compute_channel_scores(df)
    if scores.empty:
        return pd.DataFrame()

    work = add_forward_market_moves(df, horizons=horizons)
    score_cols = [col for col in scores.columns if col not in work.columns]
    if score_cols:
        work = work.join(scores[score_cols], how="left")

    rows: list[dict] = []
    for channel in CHANNELS:
        score_col = f"{channel.key}_channel_score"
        if score_col not in work.columns or work[score_col].notna().sum() == 0:
            continue

        threshold = work[score_col].quantile(score_quantile)
        high_mask = work[score_col] >= threshold

        for h in horizons:
            sp_col = f"sp500_forward_{h}d_return"
            hy_col = f"hy_spread_forward_{h}d_change"
            ig_col = f"ig_spread_forward_{h}d_change"

            cols = [score_col]
            for col in (sp_col, hy_col, ig_col):
                if col in work.columns:
                    cols.append(col)
            forward_cols = [col for col in (sp_col, hy_col, ig_col) if col in work.columns]
            sub = work.loc[high_mask, cols].dropna(subset=forward_cols, how="all")

            n_obs = int(len(sub))
            if n_obs == 0:
                continue

            row = {
                "channel": channel.name,
                "horizon_days": h,
                "score_threshold": float(threshold),
                "observations": n_obs,
                "confidence": confidence_flag(n_obs),
                "avg_sp500_return": _safe_pct(sub[sp_col].mean()) if sp_col in sub.columns else None,
                "worst_5pct_sp500_return": _safe_pct(sub[sp_col].quantile(0.05)) if sp_col in sub.columns else None,
                "avg_hy_spread_change": _safe_pct(sub[hy_col].mean()) if hy_col in sub.columns else None,
                "avg_ig_spread_change": _safe_pct(sub[ig_col].mean()) if ig_col in sub.columns else None,
                "hit_rate_sp500_down": _safe_pct((sub[sp_col] < 0).mean()) if sp_col in sub.columns else None,
                "hit_rate_hy_widening": _safe_pct((sub[hy_col] > 0).mean()) if hy_col in sub.columns else None,
                "hit_rate_ig_widening": _safe_pct((sub[ig_col] > 0).mean()) if ig_col in sub.columns else None,
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["horizon_days", "channel"]).reset_index(drop=True)


def latest_channel_validation_snapshot(df: pd.DataFrame, score_quantile: float = 0.75) -> dict:
    table = channel_validation_table(df, score_quantile=score_quantile)
    if table.empty:
        return {"available": False, "reason": "channel validation unavailable", "table": table}

    summary = table.groupby("channel", as_index=False).agg(
        observations=("observations", "sum"),
        avg_hy_spread_change=("avg_hy_spread_change", "mean"),
        avg_ig_spread_change=("avg_ig_spread_change", "mean"),
        avg_sp500_return=("avg_sp500_return", "mean"),
    )
    return {"available": True, "table": table, "summary": summary}
