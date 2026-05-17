"""
Regime-conditioned forward simulation — fan chart of portfolio paths.

Rather than a single expected return, this module simulates N forward paths
starting from today, drawing daily returns from the empirical distribution
of each regime, weighted by transition probabilities.

Concept:
  1. Estimate the empirical return distribution (mean, vol, skew) for each regime
     from strategy_daily_return grouped by final_decision.
  2. Use the regime transition matrix to estimate the probability of being in
     each regime on day t, starting from today's regime.
  3. For each simulation path: draw a regime sequence (Markov chain), then
     draw a daily return from that regime's empirical distribution.
  4. Compute portfolio value paths and report percentile fan chart.

Why this matters: Point estimates (avg return) hide the fat left tail.
The fan chart shows that Risk-Off regimes produce a wide spread of outcomes,
while Risk-On regimes cluster tightly upward.

Public API
----------
  N_SIMULATIONS   = 1000
  HORIZONS_DAYS   = [10, 21, 42, 63]
  PERCENTILES     = [5, 10, 25, 50, 75, 90, 95]
  run_forward_simulation(df, n_sim, seed) → dict
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SIMULATIONS: int = 1_000
HORIZONS_DAYS: list[int] = [10, 21, 42, 63]
PERCENTILES: list[int] = [5, 10, 25, 50, 75, 90, 95]

_REGIME_ORDER: list[str] = ["Risk-On", "Neutral", "Caution", "Risk-Off"]
_RETURN_COL = "strategy_daily_return"
_LABEL_COL = "final_decision"
_MIN_OBS = 20
_MIN_ROWS = 100
_LAPLACE = 0.01


# ---------------------------------------------------------------------------
# Step 1: Regime return profiles
# ---------------------------------------------------------------------------

def _hand_skew(arr: np.ndarray) -> float:
    """Pearson's moment coefficient of skewness, no scipy."""
    n = len(arr)
    if n < 3:
        return 0.0
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(((arr - mu) ** 3).mean() / (sigma ** 3))


def _build_regime_profiles(
    df: pd.DataFrame,
    global_mean: float,
    global_std: float,
) -> dict[str, dict]:
    """
    Compute per-regime (mean, std, skewness, n_obs) from strategy_daily_return.

    Falls back to global mean/std for regimes with fewer than _MIN_OBS observations.
    """
    profiles: dict[str, dict] = {}
    for regime in _REGIME_ORDER:
        mask = df[_LABEL_COL] == regime
        rets = df.loc[mask, _RETURN_COL].dropna().values
        n = len(rets)
        if n >= _MIN_OBS:
            mean_d = float(rets.mean())
            std_d = float(rets.std(ddof=1)) if n > 1 else global_std
            skew = _hand_skew(rets)
        else:
            mean_d = global_mean
            std_d = global_std
            skew = 0.0
            n = 0  # flag that fallback was used
        profiles[regime] = {
            "mean_daily": mean_d,
            "std_daily": max(std_d, 1e-6),
            "skewness": skew,
            "n_obs": n,
        }
    return profiles


# ---------------------------------------------------------------------------
# Step 2: Transition matrix
# ---------------------------------------------------------------------------

def _build_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 4×4 row-stochastic transition matrix from consecutive regime pairs.

    Laplace smoothing of _LAPLACE is added to every cell before normalisation.
    """
    regimes = _REGIME_ORDER
    n = len(regimes)
    idx = {r: i for i, r in enumerate(regimes)}

    counts = np.full((n, n), _LAPLACE)

    labels = df[_LABEL_COL].values
    for t in range(len(labels) - 1):
        cur = labels[t]
        nxt = labels[t + 1]
        if cur in idx and nxt in idx:
            counts[idx[cur], idx[nxt]] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    # Guard against zero rows (shouldn't happen with Laplace, but be safe)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    P = counts / row_sums

    return pd.DataFrame(P, index=regimes, columns=regimes)


# ---------------------------------------------------------------------------
# Step 3: Monte Carlo simulation
# ---------------------------------------------------------------------------

def _simulate_paths(
    profiles: dict[str, dict],
    trans_df: pd.DataFrame,
    current_regime: str,
    n_sim: int,
    max_horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate n_sim paths of length max_horizon.

    Returns array of shape (n_sim, max_horizon) representing cumulative
    portfolio value (starting at 1.0).

    Regime draw: multinomial from transition row.
    Return draw: Cornish-Fisher skew adjustment:
        r = mean + std * (z + skewness/6 * (z^2 - 1))
    """
    regimes = _REGIME_ORDER
    n_regimes = len(regimes)
    regime_idx = {r: i for i, r in enumerate(regimes)}

    # Build arrays for vectorised regime-conditioned draws
    means = np.array([profiles[r]["mean_daily"] for r in regimes])
    stds = np.array([profiles[r]["std_daily"] for r in regimes])
    skews = np.array([profiles[r]["skewness"] for r in regimes])

    # Pre-draw all standard normals: (n_sim, max_horizon)
    Z = rng.standard_normal((n_sim, max_horizon))

    # Simulate regime sequences: (n_sim, max_horizon)
    regime_seqs = np.empty((n_sim, max_horizon), dtype=np.int32)
    trans_matrix = trans_df.values  # (4, 4)

    start_idx = regime_idx.get(current_regime, 0)
    current_indices = np.full(n_sim, start_idx, dtype=np.int32)

    for t in range(max_horizon):
        regime_seqs[:, t] = current_indices
        # Vectorised multinomial draw per path
        # Each path has a row in trans_matrix indexed by current_indices
        probs_batch = trans_matrix[current_indices]  # (n_sim, 4)
        # Use cumsum + uniform for multinomial draw (vectorised)
        u = rng.random(n_sim)
        cumprobs = probs_batch.cumsum(axis=1)
        # next regime = first index where cumprob > u
        next_indices = (cumprobs < u[:, np.newaxis]).sum(axis=1)
        next_indices = np.clip(next_indices, 0, n_regimes - 1)
        current_indices = next_indices

    # Compute daily returns from regime sequences
    r_means = means[regime_seqs]    # (n_sim, max_horizon)
    r_stds = stds[regime_seqs]
    r_skews = skews[regime_seqs]

    # Cornish-Fisher skew adjustment
    daily_returns = r_means + r_stds * (Z + r_skews / 6.0 * (Z ** 2 - 1.0))

    # Cumulative product: path[t] = prod(1 + r_s, s=1..t)
    cum_paths = np.cumprod(1.0 + daily_returns, axis=1)  # (n_sim, max_horizon)
    return cum_paths


# ---------------------------------------------------------------------------
# Step 4: Percentile bands
# ---------------------------------------------------------------------------

def _compute_percentile_bands(paths: np.ndarray) -> pd.DataFrame:
    """
    Compute PERCENTILES across n_sim paths for each day.

    Returns DataFrame with index = day (1-based), columns = percentile values.
    """
    n_sim, max_horizon = paths.shape
    days = np.arange(1, max_horizon + 1)
    data = np.percentile(paths, PERCENTILES, axis=0).T  # (max_horizon, n_pct)
    return pd.DataFrame(data, index=days, columns=PERCENTILES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_forward_simulation(
    df: pd.DataFrame,
    n_sim: int = N_SIMULATIONS,
    seed: int = 42,
) -> dict:
    """
    Run regime-conditioned Monte Carlo forward simulation of portfolio returns.

    Parameters
    ----------
    df : pd.DataFrame
        Master DataFrame with strategy_daily_return and final_decision columns.
    n_sim : int
        Number of simulation paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        paths_df          — percentile fan chart (index=day, cols=percentiles)
        regime_profiles   — per-regime mean/std/skew/n_obs
        transition_matrix — 4×4 DataFrame
        current_regime    — str
        horizon_summary   — {horizon: {5: pct5, 50: median, 95: pct95}}
        available         — bool
        n_sim             — int
        seed              — int
    """
    _unavailable = {
        "paths_df": pd.DataFrame(),
        "regime_profiles": {},
        "transition_matrix": pd.DataFrame(),
        "current_regime": "Unknown",
        "horizon_summary": {},
        "available": False,
        "n_sim": n_sim,
        "seed": seed,
    }

    # Guard: need return column and sufficient history
    if _RETURN_COL not in df.columns or _LABEL_COL not in df.columns:
        return _unavailable
    if len(df) < _MIN_ROWS:
        return _unavailable

    df = df.dropna(subset=[_RETURN_COL, _LABEL_COL]).copy()
    if len(df) < _MIN_ROWS:
        return _unavailable

    # --- Step 1: Regime profiles ---
    all_rets = df[_RETURN_COL].values
    global_mean = float(all_rets.mean())
    global_std = float(all_rets.std(ddof=1)) if len(all_rets) > 1 else 0.001
    regime_profiles = _build_regime_profiles(df, global_mean, global_std)

    # --- Step 2: Transition matrix ---
    trans_df = _build_transition_matrix(df)

    # --- Step 3: Monte Carlo ---
    current_regime = str(df.iloc[-1][_LABEL_COL])
    max_horizon = max(HORIZONS_DAYS)

    rng = np.random.default_rng(seed)
    paths = _simulate_paths(
        regime_profiles, trans_df, current_regime, n_sim, max_horizon, rng
    )

    # --- Step 4: Percentile bands ---
    paths_df = _compute_percentile_bands(paths)

    # Horizon summary: convert cumulative value to total return (value - 1)
    horizon_summary: dict[int, dict[int, float]] = {}
    for h in HORIZONS_DAYS:
        if h <= max_horizon:
            h_vals = paths[:, h - 1] - 1.0  # total return at day h
            horizon_summary[h] = {
                pct: float(np.percentile(h_vals, pct))
                for pct in [5, 50, 95]
            }

    return {
        "paths_df": paths_df,
        "regime_profiles": regime_profiles,
        "transition_matrix": trans_df,
        "current_regime": current_regime,
        "horizon_summary": horizon_summary,
        "available": True,
        "n_sim": n_sim,
        "seed": seed,
    }
