"""
Monte Carlo forward simulation of composite risk score paths.

Combines two already-computed components:
  1. Regime transition matrix (Markov chain) from src.regime_transition
  2. Current regime probability distribution from src.regime_probability

At each simulated step the regime transitions according to the empirical
day-over-day Markov matrix; the composite score for that step is drawn
from the regime's historical Gaussian distribution (clipped to [0, 100]).
Paths are fully vectorised (n_paths × H) via numpy broadcasting.

Transition matrix is Laplace-smoothed (_LAPLACE_EPS = 0.5) so that no
transition has exactly zero probability — prevents the simulation from
becoming trapped in degenerate regimes with sparse data.

Public API
----------
  build_transition_matrix    — row-stochastic matrix + regime list from df
  simulate_paths             — vectorised path simulation
  run_monte_carlo            — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_HORIZONS:    list[int] = [21, 63, 126]      # ~1 month, ~3 months, ~6 months
_N_PATHS:     int       = 1_000
_SEED:        int       = 42
_LAPLACE_EPS: float     = 0.5               # pseudo-count for smoothing
_SCORE_COL    = "composite_risk_score_smooth"
_LABEL_COL    = "final_decision"
_DATE_COL     = "date"
_SCORE_FLOOR  = 2.0   # minimum std for per-regime Gaussian
_PERCENTILES  = [5, 25, 50, 75, 95]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_transition_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Build a Laplace-smoothed row-stochastic transition matrix.

    Counts day-over-day label transitions in df[_LABEL_COL], applies
    Laplace smoothing (_LAPLACE_EPS per cell), then normalises rows.

    Returns (regimes, P) where P[i, j] = P(next=j | current=i).
    Returns ([], empty array) if data is insufficient.
    """
    if _LABEL_COL not in df.columns or len(df) < 2:
        return [], np.empty((0, 0))

    labels = df[_LABEL_COL].values
    regimes = sorted(df[_LABEL_COL].unique())
    idx = {r: i for i, r in enumerate(regimes)}
    n = len(regimes)

    counts = np.zeros((n, n), dtype=float)
    for t in range(len(labels) - 1):
        i = idx.get(labels[t])
        j = idx.get(labels[t + 1])
        if i is not None and j is not None:
            counts[i, j] += 1.0

    # Laplace smoothing — prevents zero-probability transitions
    counts += _LAPLACE_EPS

    row_sums = counts.sum(axis=1, keepdims=True)
    P = counts / row_sums
    return regimes, P


def _score_params(df: pd.DataFrame, regimes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-regime empirical mean and std of composite_risk_score_smooth.

    Returns (means, stds) each of shape (len(regimes),).
    Regimes absent from df get grand-mean / grand-std.
    """
    grand_mean = float(df[_SCORE_COL].mean()) if _SCORE_COL in df.columns else 50.0
    grand_std  = max(_SCORE_FLOOR, float(df[_SCORE_COL].std(ddof=1))) if _SCORE_COL in df.columns else 10.0

    means = np.full(len(regimes), grand_mean)
    stds  = np.full(len(regimes), grand_std)

    if _SCORE_COL not in df.columns or _LABEL_COL not in df.columns:
        return means, stds

    for i, r in enumerate(regimes):
        sub = df.loc[df[_LABEL_COL] == r, _SCORE_COL].dropna()
        if len(sub) >= 5:
            means[i] = float(sub.mean())
            stds[i]  = max(_SCORE_FLOOR, float(sub.std(ddof=1)))

    return means, stds


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_paths(
    P: np.ndarray,
    regimes: list[str],
    score_means: np.ndarray,
    score_stds: np.ndarray,
    start_probs: dict[str, float],
    horizons: list[int] = _HORIZONS,
    n_paths: int = _N_PATHS,
    seed: int = _SEED,
) -> dict:
    """
    Simulate forward composite score paths via a Markov regime model.

    Parameters
    ----------
    P            : (n_reg, n_reg) row-stochastic transition matrix
    regimes      : ordered regime names (index into P)
    score_means  : (n_reg,) per-regime composite score mean
    score_stds   : (n_reg,) per-regime composite score std
    start_probs  : current regime probability dict {regime: prob}
    horizons     : list of forward horizons (trading days)
    n_paths      : number of Monte Carlo paths
    seed         : random seed for reproducibility

    Returns dict:
      paths            — {H: ndarray (n_paths, H), composite scores}
      percentiles      — {H: DataFrame with columns p5/p25/p50/p75/p95}
      regime_paths     — {H: ndarray (n_paths, H), regime indices}
      regime_probs_at  — {H: {regime: float}} regime distribution at horizon H
      summary          — {H: {median, p5, p95, prob_above_50, prob_above_70}}
    """
    n_reg   = len(regimes)
    max_H   = max(horizons)
    reg_idx = {r: i for i, r in enumerate(regimes)}
    rng     = np.random.default_rng(seed)

    # Starting regime distribution — align with P's regime order
    start_vec = np.array([start_probs.get(r, 0.0) for r in regimes], dtype=float)
    if start_vec.sum() < 1e-9:
        start_vec = np.ones(n_reg) / n_reg
    else:
        start_vec /= start_vec.sum()

    # Sample starting regimes for all paths
    cum_start = start_vec.cumsum()
    u0 = rng.random(n_paths)
    current = (cum_start[np.newaxis, :] < u0[:, np.newaxis]).sum(axis=1)
    current = np.clip(current, 0, n_reg - 1)             # (n_paths,)

    # Pre-compute cumulative transition matrix for vectorised sampling
    cumP = P.cumsum(axis=1)                              # (n_reg, n_reg)

    # Simulate max_H steps for all paths
    all_regimes = np.empty((n_paths, max_H), dtype=np.intp)
    all_scores  = np.empty((n_paths, max_H), dtype=float)

    for t in range(max_H):
        # Vectorised categorical draw: gather CDF row, compare with uniform
        cdf_rows = cumP[current]                         # (n_paths, n_reg)
        u = rng.random((n_paths, 1))
        next_reg = (cdf_rows < u).sum(axis=1)           # (n_paths,)
        next_reg = np.clip(next_reg, 0, n_reg - 1)

        # Draw composite score from regime's Gaussian, clipped to [0, 100]
        means_t = score_means[next_reg]                  # (n_paths,)
        stds_t  = score_stds[next_reg]
        scores_t = rng.normal(means_t, stds_t)
        scores_t = np.clip(scores_t, 0.0, 100.0)

        all_regimes[:, t] = next_reg
        all_scores[:, t]  = scores_t
        current           = next_reg

    # Build output by horizon
    paths:           dict = {}
    percentiles:     dict = {}
    regime_paths:    dict = {}
    regime_probs_at: dict = {}
    summary:         dict = {}

    for H in horizons:
        sc = all_scores[:, :H]       # (n_paths, H)
        rg = all_regimes[:, :H]

        paths[H]        = sc
        regime_paths[H] = rg

        # Percentile bands over paths, per step
        pct_matrix = np.percentile(sc, _PERCENTILES, axis=0)  # (5, H)
        pct_df = pd.DataFrame(
            pct_matrix.T,
            columns=[f"p{p}" for p in _PERCENTILES],
        )
        percentiles[H] = pct_df

        # Regime distribution at the final horizon step
        reg_at_H = rg[:, -1]
        rp = {}
        for i, r in enumerate(regimes):
            rp[r] = float((reg_at_H == i).mean())
        regime_probs_at[H] = rp

        # Summary statistics at horizon H (final composite score of each path)
        final_scores = sc[:, -1]
        summary[H] = {
            "median":         float(np.median(final_scores)),
            "p5":             float(np.percentile(final_scores, 5)),
            "p95":            float(np.percentile(final_scores, 95)),
            "prob_above_50":  float((final_scores > 50).mean()),
            "prob_above_70":  float((final_scores > 70).mean()),
        }

    return {
        "paths":            paths,
        "percentiles":      percentiles,
        "regime_paths":     regime_paths,
        "regime_probs_at":  regime_probs_at,
        "summary":          summary,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_monte_carlo(
    df: pd.DataFrame,
    current_probs: dict[str, float] | None = None,
    n_paths: int = _N_PATHS,
    horizons: list[int] | None = None,
    seed: int = _SEED,
) -> dict:
    """
    Full Monte Carlo forward simulation.

    Parameters
    ----------
    df            : scored DataFrame (must contain _LABEL_COL and _SCORE_COL)
    current_probs : regime probability dict from regime_probability module;
                    if None, uses the empirical regime frequency as prior
    n_paths       : number of simulation paths
    horizons      : forward horizons in trading days (default [21, 63, 126])
    seed          : random seed

    Returns dict:
      regimes            — list of regime names in matrix order
      transition_matrix  — (n_reg, n_reg) smoothed row-stochastic matrix
      score_means        — per-regime composite score means
      score_stds         — per-regime composite score stds
      simulation         — output of simulate_paths()
      horizons           — horizons used
      last_actual        — most recent actual composite score
      last_date          — most recent date in df
    """
    if horizons is None:
        horizons = _HORIZONS

    if _LABEL_COL not in df.columns or df.empty:
        return {}

    regimes, P = build_transition_matrix(df)
    if not regimes:
        return {}

    score_means, score_stds = _score_params(df, regimes)

    # Default start probs: empirical regime frequency
    if current_probs is None:
        vc = df[_LABEL_COL].value_counts(normalize=True)
        current_probs = vc.to_dict()

    sim = simulate_paths(
        P, regimes, score_means, score_stds,
        start_probs=current_probs,
        horizons=horizons,
        n_paths=n_paths,
        seed=seed,
    )

    last_actual = float(df[_SCORE_COL].iloc[-1]) if _SCORE_COL in df.columns else None
    last_date   = df[_DATE_COL].iloc[-1] if _DATE_COL in df.columns else None

    return {
        "regimes":           regimes,
        "transition_matrix": P,
        "score_means":       score_means,
        "score_stds":        score_stds,
        "simulation":        sim,
        "horizons":          horizons,
        "last_actual":       last_actual,
        "last_date":         last_date,
    }
