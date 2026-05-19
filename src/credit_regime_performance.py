"""
Regime-conditioned credit performance.

This module answers the institutional question: what happened after each model
regime historically? It keeps the statistics intentionally transparent and
uses only columns already present in the scored dashboard dataset.
"""

from __future__ import annotations

import pandas as pd


HORIZONS = (21, 63, 126)
MIN_OBS_EXPLORATORY = 20
MIN_OBS_INDICATIVE = 50


def confidence_flag(n_obs: int) -> str:
    if n_obs >= MIN_OBS_INDICATIVE:
        return "Reliable"
    if n_obs >= MIN_OBS_EXPLORATORY:
        return "Indicative"
    return "Exploratory"


def add_forward_market_moves(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """
    Add forward S&P returns and spread changes for each horizon.

    Spread changes are positive when spreads widen, which is bad for credit.
    """
    out = df.copy()

    if "sp500" in out.columns:
        for h in horizons:
            out[f"sp500_forward_{h}d_return"] = out["sp500"].shift(-h) / out["sp500"] - 1

    if "hy_spread" in out.columns:
        for h in horizons:
            out[f"hy_spread_forward_{h}d_change"] = out["hy_spread"].shift(-h) - out["hy_spread"]

    if "ig_spread" in out.columns:
        for h in horizons:
            out[f"ig_spread_forward_{h}d_change"] = out["ig_spread"].shift(-h) - out["ig_spread"]

    return out


def summarize_by_regime(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """
    Build a compact regime x horizon table.

    Output includes sample size, mean equity return, worst-decile equity return,
    HY/IG average spread change, and widening probabilities where available.
    """
    if df.empty or regime_col not in df.columns:
        return pd.DataFrame()

    work = add_forward_market_moves(df, horizons)
    rows: list[dict] = []

    for regime, group in work.groupby(regime_col, dropna=True):
        for h in horizons:
            row = {"regime": regime, "horizon_days": h}

            sp_col = f"sp500_forward_{h}d_return"
            hy_col = f"hy_spread_forward_{h}d_change"
            ig_col = f"ig_spread_forward_{h}d_change"

            valid_cols = [c for c in (sp_col, hy_col, ig_col) if c in group.columns]
            if valid_cols:
                valid = group[valid_cols].dropna(how="all")
            else:
                valid = pd.DataFrame(index=group.index)

            n_obs = len(valid)
            row["n_obs"] = n_obs
            row["confidence"] = confidence_flag(n_obs)

            if sp_col in group.columns:
                s = group[sp_col].dropna()
                row["avg_sp500_return"] = s.mean()
                row["hit_rate_sp500_positive"] = (s > 0).mean() if len(s) else pd.NA
                row["worst_decile_sp500_return"] = s.quantile(0.10) if len(s) else pd.NA

            if hy_col in group.columns:
                s = group[hy_col].dropna()
                row["avg_hy_spread_change"] = s.mean()
                row["prob_hy_widening"] = (s > 0).mean() if len(s) else pd.NA

            if ig_col in group.columns:
                s = group[ig_col].dropna()
                row["avg_ig_spread_change"] = s.mean()
                row["prob_ig_widening"] = (s > 0).mean() if len(s) else pd.NA

            rows.append(row)

    return pd.DataFrame(rows)


def latest_regime_performance_note(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    horizon_days: int = 63,
) -> dict:
    if df.empty or regime_col not in df.columns:
        return {"available": False, "reason": "regime column unavailable"}

    current_regime = df[regime_col].dropna().iloc[-1] if df[regime_col].notna().any() else None
    if current_regime is None:
        return {"available": False, "reason": "current regime unavailable"}

    table = summarize_by_regime(df, regime_col=regime_col, horizons=(horizon_days,))
    if table.empty:
        return {"available": False, "reason": "performance table unavailable"}

    row = table[table["regime"] == current_regime]
    if row.empty:
        return {"available": False, "reason": "current regime has no historical observations"}

    r = row.iloc[0].to_dict()
    return {
        "available": True,
        "regime": current_regime,
        "horizon_days": horizon_days,
        "n_obs": int(r.get("n_obs", 0)),
        "confidence": r.get("confidence", "Exploratory"),
        "avg_sp500_return": r.get("avg_sp500_return"),
        "prob_hy_widening": r.get("prob_hy_widening"),
        "avg_hy_spread_change": r.get("avg_hy_spread_change"),
    }

