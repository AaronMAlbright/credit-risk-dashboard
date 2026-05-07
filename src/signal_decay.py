"""
Signal decay analysis.

Measures how long regime-based forward-return edges persist across
multiple holding horizons (7, 14, 30, 60, 90 trading days).

Public API
----------
  compute_forward_returns  — augment df with sp500_fwd_{n}d columns
  compute_decay_by_regime  — mean return + hit rate + bootstrap CI per (regime, horizon)
  compute_score_correlations — signal vs forward-return correlation at each horizon
  run_signal_decay         — orchestrator returning all results in a dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS: list[int] = [7, 14, 30, 60, 90]

_DEFAULT_N_BOOT = 500
_DEFAULT_CI = 0.95
_MIN_OBS_BOOT = 5   # below this, CI is NaN rather than unreliable bootstrap

# Existing precomputed columns we can reuse
_EXISTING_FWD: dict[int, str] = {
    30: "sp500_forward_30d_return",
    60: "sp500_forward_60d_return",
}

_SCORE_COLS: dict[str, str] = {
    "Macro Risk":     "macro_risk_score_smooth",
    "Credit Risk":    "credit_market_risk_score_smooth",
    "Complacency":    "complacency_score_smooth",
    "Liquidity":      "liquidity_regime_score_smooth",
    "Treasury":       "treasury_stress_score_smooth",
    "Mean Reversion": "mean_reversion_score_smooth",
    "Composite":      "composite_risk_score_smooth",
}


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def compute_forward_returns(df: pd.DataFrame, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """
    Add sp500_fwd_{n}d columns to df for each requested horizon.

    Reuses existing precomputed columns (30d, 60d) when available.
    New horizons are computed from the sp500 price series.
    Does not modify the original DataFrame.
    """
    out = df.copy()

    for n in horizons:
        col = f"sp500_fwd_{n}d"
        if col in out.columns:
            continue
        if n in _EXISTING_FWD and _EXISTING_FWD[n] in out.columns:
            out[col] = out[_EXISTING_FWD[n]]
        elif "sp500" in out.columns:
            price = out["sp500"].astype(float)
            out[col] = (price.shift(-n) / price - 1).values
        # If sp500 not present, column simply isn't added

    return out


# ---------------------------------------------------------------------------
# Regime decay
# ---------------------------------------------------------------------------

def compute_decay_by_regime(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    horizons: list[int] = HORIZONS,
    n_boot: int = _DEFAULT_N_BOOT,
    ci: float = _DEFAULT_CI,
) -> pd.DataFrame:
    """
    Per-regime forward-return statistics at each horizon with bootstrap CIs.

    Returns DataFrame indexed by (regime, horizon) with columns:
      mean_return, hit_rate, n_obs, ci_lower, ci_upper
    """
    if regime_col not in df.columns or df.empty:
        return pd.DataFrame()

    df_aug = compute_forward_returns(df, horizons)
    alpha = (1 - ci) / 2
    lo_pct, hi_pct = alpha * 100, (1 - alpha) * 100

    rng_state = np.random.get_state()
    np.random.seed(42)
    try:
        records = []
        for regime, grp in df_aug.groupby(regime_col):
            for n in horizons:
                col = f"sp500_fwd_{n}d"
                if col not in grp.columns:
                    continue
                vals = grp[col].dropna().values
                n_obs = len(vals)
                if n_obs == 0:
                    continue

                mean_ret = float(vals.mean())
                hit_rate = float((vals > 0).mean())

                if n_obs >= _MIN_OBS_BOOT:
                    boot = np.array([
                        vals[np.random.choice(n_obs, size=n_obs, replace=True)].mean()
                        for _ in range(n_boot)
                    ])
                    ci_lower = float(np.percentile(boot, lo_pct))
                    ci_upper = float(np.percentile(boot, hi_pct))
                else:
                    ci_lower = ci_upper = np.nan

                records.append({
                    "regime":      regime,
                    "horizon":     n,
                    "mean_return": round(mean_ret, 4),
                    "hit_rate":    round(hit_rate, 3),
                    "n_obs":       n_obs,
                    "ci_lower":    round(ci_lower, 4) if not np.isnan(ci_lower) else np.nan,
                    "ci_upper":    round(ci_upper, 4) if not np.isnan(ci_upper) else np.nan,
                })
    finally:
        np.random.set_state(rng_state)

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .set_index(["regime", "horizon"])
        .sort_index()
    )


# ---------------------------------------------------------------------------
# Score correlation decay
# ---------------------------------------------------------------------------

def compute_score_correlations(
    df: pd.DataFrame,
    horizons: list[int] = HORIZONS,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Pearson correlation of each smoothed score vs forward SP500 return at each horizon.

    Returns DataFrame indexed by signal name, columns = horizon (int).
    Values are NaN when fewer than min_obs clean pairs are available.
    """
    df_aug = compute_forward_returns(df, horizons)

    rows: dict[str, dict[int, float]] = {}
    for name, score_col in _SCORE_COLS.items():
        if score_col not in df_aug.columns:
            continue
        corrs: dict[int, float] = {}
        for n in horizons:
            fwd_col = f"sp500_fwd_{n}d"
            if fwd_col not in df_aug.columns:
                corrs[n] = np.nan
                continue
            clean = df_aug[[score_col, fwd_col]].dropna()
            if len(clean) < min_obs:
                corrs[n] = np.nan
            else:
                corrs[n] = round(float(clean[score_col].corr(clean[fwd_col])), 3)
        rows[name] = corrs

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).T  # index=signal, columns=horizons


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_signal_decay(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    horizons: list[int] = HORIZONS,
    n_boot: int = _DEFAULT_N_BOOT,
) -> dict:
    """
    Full signal decay analysis.

    Returns dict with:
      decay_by_regime    — (regime, horizon) indexed DataFrame
      score_correlations — signal × horizon correlation table
      horizons           — list of horizons used
    """
    return {
        "decay_by_regime":    compute_decay_by_regime(df, regime_col, horizons, n_boot),
        "score_correlations": compute_score_correlations(df, horizons),
        "horizons":           horizons,
    }
