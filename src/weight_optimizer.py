"""
Composite weight sensitivity and optimization.

Monte Carlo samples weight vectors from the 5-simplex (Dirichlet uniform),
computes a simplified strategy return for each, and ranks by in-sample
Sharpe ratio and predictive correlation.

Also provides a tornado chart: how much does Sharpe change when each weight
is perturbed ±5 percentage points (with the rest renormalized)?

Public API
----------
  CURRENT_WEIGHTS          — dict of current production weights
  compute_composite        — custom-weight composite score (vectorized)
  compute_strategy_sharpe  — Sharpe for one weight vector
  run_weight_grid_search   — Monte Carlo search → results DataFrame
  compute_tornado          — ±delta perturbation table
  find_optimal_weights     — top row from grid-search results
  run_weight_optimization  — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys must match composite_engine.py column usage
SCORE_COL_MAP: dict[str, str] = {
    "macro_risk":     "macro_risk_score_smooth",
    "credit_risk":    "credit_market_risk_score_smooth",
    "complacency":    "complacency_score_smooth",
    "liquidity":      "liquidity_regime_score_smooth",
    "treasury":       "treasury_stress_score_smooth",
    "mean_reversion": "mean_reversion_score_smooth",
}

# mean_reversion is inverted in the composite engine (100 - score)
INVERTED: set[str] = {"mean_reversion"}

# Current production weights (must sum to 1.0)
CURRENT_WEIGHTS: dict[str, float] = {
    "macro_risk":     0.25,
    "credit_risk":    0.25,
    "complacency":    0.20,
    "liquidity":      0.15,
    "treasury":       0.10,
    "mean_reversion": 0.05,
}

DEFAULT_N_SAMPLES = 2000
DEFAULT_TORNADO_DELTA = 0.05
_ANNUALIZE = 252
_MIN_STRAT_OBS = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _available(df: pd.DataFrame) -> dict[str, str]:
    return {k: v for k, v in SCORE_COL_MAP.items() if v in df.columns}


def _build_score_matrix(df: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (score_matrix, valid_mask) where score_matrix is (n_rows, n_scores)
    with mean_reversion inverted, and valid_mask marks rows with no NaN.
    """
    cols = [SCORE_COL_MAP[k] for k in keys]
    score_raw = df[cols].astype(float).values.copy()  # (n_rows, n_scores) — writable copy
    for i, k in enumerate(keys):
        if k in INVERTED:
            score_raw[:, i] = 100.0 - score_raw[:, i]
    valid = ~np.isnan(score_raw).any(axis=1)
    return score_raw, valid


def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe from daily returns array. NaN if insufficient or zero-std."""
    clean = returns[~np.isnan(returns)]
    if len(clean) < _MIN_STRAT_OBS:
        return np.nan
    std = float(clean.std(ddof=1))
    if std < 1e-10:
        return np.nan
    return float(clean.mean() / std * np.sqrt(_ANNUALIZE))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Renormalize so weights sum to 1.0."""
    total = sum(weights.values())
    if total < 1e-9:
        raise ValueError("Weight vector sums to zero.")
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_composite(
    df: pd.DataFrame,
    weights: dict[str, float],
    available: dict[str, str] | None = None,
) -> pd.Series:
    """
    Custom-weight composite score (0–100) for a given weight dict.
    mean_reversion is inverted (risk component = 100 − score).
    Weights are renormalised internally.
    """
    if available is None:
        available = _available(df)
    norm_w = _normalize_weights({k: weights.get(k, 0.0) for k in available})
    composite = pd.Series(0.0, index=df.index)
    for key in available:
        score = df[SCORE_COL_MAP[key]].astype(float)
        if key in INVERTED:
            score = 100.0 - score
        composite += norm_w[key] * score
    return composite.clip(0, 100)


def compute_strategy_sharpe(
    df: pd.DataFrame,
    weights: dict[str, float],
    available: dict[str, str] | None = None,
) -> float:
    """
    Simplified in-sample Sharpe for a weight vector.

    Equity weight = 1 − composite/100 (linear, clipped [0,1]).
    Strategy return_t = equity_weight_{t−1} × sp500_daily_return_t.
    """
    if "sp500_daily_return" not in df.columns:
        return np.nan
    if available is None:
        available = _available(df)
    comp = compute_composite(df, weights, available)
    eq_w = (1.0 - comp / 100.0).clip(0.0, 1.0)
    eq_w_lag = eq_w.shift(1)
    strat_ret = eq_w_lag * df["sp500_daily_return"]
    return _sharpe(strat_ret.values)


def run_weight_grid_search(
    df: pd.DataFrame,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo weight optimisation via Dirichlet sampling.

    Returns DataFrame sorted by Sharpe (desc) with columns:
      {score_key} × 6, sharpe, corr_30d
    corr_30d: Pearson correlation of custom composite vs sp500_forward_30d_return
              (negative = risk-off signal working — composite rises when market falls)
    """
    av = _available(df)
    if len(av) < 3 or "sp500_daily_return" not in df.columns:
        return pd.DataFrame()

    keys = list(av.keys())
    n_scores = len(keys)

    score_matrix, valid = _build_score_matrix(df, keys)
    daily_ret = df["sp500_daily_return"].values
    fwd_ret_30 = df["sp500_forward_30d_return"].values if "sp500_forward_30d_return" in df.columns \
        else np.full(len(df), np.nan)

    # Work on clean rows only
    s_clean = score_matrix[valid]          # (n_clean, n_scores)
    d_clean = daily_ret[valid]             # (n_clean,)
    f_clean = fwd_ret_30[valid]            # (n_clean,)

    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        w_samples = np.random.dirichlet(np.ones(n_scores), size=n_samples)  # (n_samples, n_scores)
    finally:
        np.random.set_state(rng_state)

    # Vectorised composite: (n_samples, n_clean)
    composites = w_samples @ s_clean.T
    # Equity weights: 1 - composite/100, shifted 1 day
    eq_weights = np.clip(1.0 - composites / 100.0, 0.0, 1.0)
    # Strategy returns: lag eq_weight by 1 obs
    strat_ret = eq_weights[:, :-1] * d_clean[1:]   # (n_samples, n_clean-1)

    # Sharpe (vectorised)
    means = strat_ret.mean(axis=1)
    stds = strat_ret.std(axis=1, ddof=1)
    sharpes = np.where(stds > 1e-10, means / stds * np.sqrt(_ANNUALIZE), np.nan)

    # Predictive correlation vs 30d forward return (vectorised)
    fwd_valid = ~np.isnan(f_clean)
    if fwd_valid.sum() >= 20:
        c_fwd = composites[:, fwd_valid]         # (n_samples, n_fwd)
        f_z = f_clean[fwd_valid]
        f_mean, f_std = f_z.mean(), f_z.std(ddof=1)
        c_mean = c_fwd.mean(axis=1, keepdims=True)
        c_std = c_fwd.std(axis=1, ddof=1, keepdims=True)
        c_std = np.where(c_std < 1e-10, 1.0, c_std)
        c_z = (c_fwd - c_mean) / c_std
        f_std = max(f_std, 1e-10)
        corrs = c_z @ (f_z - f_mean) / (fwd_valid.sum() * f_std)
    else:
        corrs = np.full(n_samples, np.nan)

    records = []
    for i in range(n_samples):
        row = {k: round(float(w_samples[i, j]), 4) for j, k in enumerate(keys)}
        row["sharpe"] = round(float(sharpes[i]), 4) if not np.isnan(sharpes[i]) else np.nan
        row["corr_30d"] = round(float(corrs[i]), 4) if not np.isnan(corrs[i]) else np.nan
        records.append(row)

    result = pd.DataFrame(records)
    return result.sort_values("sharpe", ascending=False).reset_index(drop=True)


def compute_tornado(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    delta: float = DEFAULT_TORNADO_DELTA,
) -> pd.DataFrame:
    """
    Sharpe sensitivity to ±delta weight perturbation per score.

    For each score i:
      - Increase w_i by delta, decrease all others proportionally
      - Decrease w_i by delta (floor at 0), increase others proportionally
      - Compute Sharpe delta vs baseline

    Returns DataFrame indexed by score name with columns:
      baseline_weight, sharpe_up, sharpe_down, delta_up, delta_down, swing
    """
    av = _available(df)
    if len(av) < 3:
        return pd.DataFrame()

    if weights is None:
        weights = CURRENT_WEIGHTS
    base_w = _normalize_weights({k: weights.get(k, 0.0) for k in av})
    baseline_sharpe = compute_strategy_sharpe(df, base_w, av)

    rows = []
    for key in av:
        # --- perturb up ---
        w_up = dict(base_w)
        w_up[key] = w_up[key] + delta
        others = {k: v for k, v in base_w.items() if k != key}
        others_sum = sum(others.values())
        if others_sum > 1e-9:
            scale = (1.0 - w_up[key]) / others_sum
            for k in others:
                w_up[k] = others[k] * scale
        s_up = compute_strategy_sharpe(df, w_up, av)

        # --- perturb down ---
        new_base = max(0.0, base_w[key] - delta)
        w_dn = dict(base_w)
        w_dn[key] = new_base
        others_dn = {k: v for k, v in base_w.items() if k != key}
        others_sum_dn = sum(others_dn.values())
        if others_sum_dn > 1e-9:
            scale_dn = (1.0 - new_base) / others_sum_dn
            for k in others_dn:
                w_dn[k] = others_dn[k] * scale_dn
        s_dn = compute_strategy_sharpe(df, w_dn, av)

        rows.append({
            "score":            key,
            "baseline_weight":  round(base_w[key], 4),
            "sharpe_up":        round(s_up, 4) if not np.isnan(s_up) else np.nan,
            "sharpe_down":      round(s_dn, 4) if not np.isnan(s_dn) else np.nan,
            "delta_up":         round(s_up - baseline_sharpe, 4) if not np.isnan(s_up) else np.nan,
            "delta_down":       round(s_dn - baseline_sharpe, 4) if not np.isnan(s_dn) else np.nan,
        })

    result = pd.DataFrame(rows).set_index("score")
    result["swing"] = (result["delta_up"] - result["delta_down"]).abs().round(4)
    return result.sort_values("swing", ascending=False)


def find_optimal_weights(results_df: pd.DataFrame) -> dict[str, float]:
    """Return the weight dict from the top-Sharpe row of a grid-search result."""
    if results_df.empty:
        return {}
    score_keys = [k for k in SCORE_COL_MAP if k in results_df.columns]
    best = results_df.iloc[0]
    return {k: float(best[k]) for k in score_keys}


def run_weight_optimization(
    df: pd.DataFrame,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> dict:
    """
    Full weight optimisation pipeline.

    Returns dict with:
      grid_results    — sorted DataFrame of all sampled weight combinations
      tornado         — sensitivity to ±5% weight perturbation
      current_sharpe  — Sharpe at production weights
      optimal_weights — weights from top-Sharpe sample
      optimal_sharpe  — Sharpe at optimal weights
    """
    grid = run_weight_grid_search(df, n_samples=n_samples)
    tornado = compute_tornado(df)
    current_sharpe = compute_strategy_sharpe(df, CURRENT_WEIGHTS)
    optimal_w = find_optimal_weights(grid)
    optimal_sharpe = compute_strategy_sharpe(df, optimal_w) if optimal_w else np.nan

    return {
        "grid_results":    grid,
        "tornado":         tornado,
        "current_sharpe":  round(current_sharpe, 4) if not np.isnan(current_sharpe) else np.nan,
        "optimal_weights": optimal_w,
        "optimal_sharpe":  round(optimal_sharpe, 4) if not np.isnan(optimal_sharpe) else np.nan,
    }
