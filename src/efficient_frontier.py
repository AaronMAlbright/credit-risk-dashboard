"""
Mean-variance efficient frontier for the credit portfolio.

Modern Portfolio Theory (Markowitz 1952): given a set of assets, the
efficient frontier traces the portfolios with maximum expected return for
each level of risk (volatility). Every rational investor should hold a
portfolio ON the frontier.

Assets considered: HY credit, IG credit, Cash (risk-free proxy ~3% annual)

Computation:
  1. Estimate mean and covariance of daily returns from historical data
  2. Simulate N random portfolios (Monte Carlo) — weight combos that sum to 1
  3. For each portfolio compute: expected annual return, annual vol, Sharpe
  4. Trace the frontier as the upper-left envelope of the cloud
  5. Mark: current regime allocation, minimum-variance portfolio, max-Sharpe portfolio

Public API
----------
  RISK_FREE_RATE     — 0.03 (3% annual cash proxy)
  N_SIMULATIONS      — 5000
  compute_efficient_frontier(df) → dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_FREE_RATE: float = 0.03   # 3 % annual cash proxy
N_SIMULATIONS: int = 5_000     # number of Monte Carlo portfolios
TRADING_DAYS: int = 252        # annualisation factor

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _portfolio_stats(
    weights: np.ndarray,
    mean_annual: np.ndarray,
    cov_annual: np.ndarray,
) -> tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe) for a weight vector.

    Parameters
    ----------
    weights:     shape (n_assets,)
    mean_annual: annualised mean return vector, shape (n_assets,)
    cov_annual:  annualised covariance matrix, shape (n_assets, n_assets)
    """
    ret = float(weights @ mean_annual)
    var = float(weights @ cov_annual @ weights)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-12 else 0.0
    return ret, vol, sharpe


def _simulate_portfolios(
    mean_annual: np.ndarray,
    cov_annual: np.ndarray,
    n_assets: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Monte-Carlo sample N_SIMULATIONS random long-only portfolios.

    Weights are drawn from a uniform Dirichlet distribution so that they
    sum to exactly 1 with all components >= 0.

    Returns a DataFrame with columns:
        hy_w, ig_w, cash_w, annual_return, annual_vol, sharpe
    """
    # Dirichlet(1,1,...) is the uniform distribution over the simplex
    raw = rng.exponential(scale=1.0, size=(N_SIMULATIONS, n_assets))
    weights_matrix = raw / raw.sum(axis=1, keepdims=True)

    records = []
    for w in weights_matrix:
        ret, vol, sharpe = _portfolio_stats(w, mean_annual, cov_annual)
        records.append(
            {
                "hy_w": w[0],
                "ig_w": w[1],
                "cash_w": w[2],
                "annual_return": ret,
                "annual_vol": vol,
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(records)


def _asset_stats(mean_annual: np.ndarray, cov_annual: np.ndarray) -> dict:
    """Return per-asset statistics (mean return, vol, Sharpe)."""
    asset_names = ["HY", "IG", "Cash"]
    result = {}
    for i, name in enumerate(asset_names):
        ret = float(mean_annual[i])
        vol = float(np.sqrt(max(cov_annual[i, i], 0.0)))
        sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-12 else 0.0
        result[name] = {"mean_return": ret, "vol": vol, "sharpe": sharpe}
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_efficient_frontier(df: pd.DataFrame) -> dict:
    """Compute the mean-variance efficient frontier for HY / IG / Cash.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at a minimum the columns ``hy_total_return_daily`` and
        ``ig_total_return_daily``.  Optionally ``hy_weight``, ``ig_weight``
        for the current allocation overlay.

    Returns
    -------
    dict with keys:
        ``simulated``          — pd.DataFrame of all simulated portfolios
        ``min_var``            — dict: weights + stats of minimum-variance portfolio
        ``max_sharpe``         — dict: weights + stats of maximum-Sharpe portfolio
        ``current_allocation`` — dict: current regime weights + their stats
        ``asset_stats``        — dict: per-asset mean return, vol, Sharpe
        ``available``          — bool: False if insufficient data
        ``n_obs``              — int: number of observations used
    """
    # ------------------------------------------------------------------
    # 1. Build return matrix
    # ------------------------------------------------------------------
    required_cols = {"hy_total_return_daily", "ig_total_return_daily"}
    missing = required_cols - set(df.columns)
    if missing:
        return {
            "simulated": pd.DataFrame(),
            "min_var": {},
            "max_sharpe": {},
            "current_allocation": {},
            "asset_stats": {},
            "available": False,
            "n_obs": 0,
        }

    returns_df = df[["hy_total_return_daily", "ig_total_return_daily"]].dropna(
        how="all"
    )

    if len(returns_df) < 60:
        return {
            "simulated": pd.DataFrame(),
            "min_var": {},
            "max_sharpe": {},
            "current_allocation": {},
            "asset_stats": {},
            "available": False,
            "n_obs": len(returns_df),
        }

    # Drop rows where *both* HY and IG are NaN; fill remaining NaN with 0
    returns_df = returns_df.dropna(how="all").fillna(0.0)

    # Cash column: constant daily return equivalent to RISK_FREE_RATE annual
    cash_daily = RISK_FREE_RATE / TRADING_DAYS
    returns_df = returns_df.copy()
    returns_df["cash_daily"] = cash_daily

    n_obs = len(returns_df)

    # ------------------------------------------------------------------
    # 2. Annualise moments
    # ------------------------------------------------------------------
    mean_daily = returns_df.values.mean(axis=0)          # shape (3,)
    cov_daily = np.cov(returns_df.values, rowvar=False)  # shape (3,3)

    mean_annual = mean_daily * TRADING_DAYS
    cov_annual = cov_daily * TRADING_DAYS

    # Cash variance is effectively 0; keep matrix positive-semi-definite
    # by zeroing the cash row/col covariances with risky assets
    # (cash is deterministic, not a random variable)
    cov_annual[2, :] = 0.0
    cov_annual[:, 2] = 0.0
    # Tiny nugget for numerical stability
    cov_annual[2, 2] = 1e-12

    # ------------------------------------------------------------------
    # 3. Simulate portfolios
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    simulated = _simulate_portfolios(mean_annual, cov_annual, n_assets=3, rng=rng)

    # ------------------------------------------------------------------
    # 4. Find min-variance and max-Sharpe portfolios
    # ------------------------------------------------------------------
    min_var_idx = simulated["annual_vol"].idxmin()
    max_sharpe_idx = simulated["sharpe"].idxmax()

    def _row_to_dict(row: pd.Series) -> dict:
        return {
            "hy_weight": float(row["hy_w"]),
            "ig_weight": float(row["ig_w"]),
            "cash_weight": float(row["cash_w"]),
            "annual_return": float(row["annual_return"]),
            "annual_vol": float(row["annual_vol"]),
            "sharpe": float(row["sharpe"]),
        }

    min_var_portfolio = _row_to_dict(simulated.loc[min_var_idx])
    max_sharpe_portfolio = _row_to_dict(simulated.loc[max_sharpe_idx])

    # ------------------------------------------------------------------
    # 5. Current allocation
    # ------------------------------------------------------------------
    current_allocation: dict = {}
    last_row = df.iloc[-1]
    hy_w = last_row.get("hy_weight", None)
    ig_w = last_row.get("ig_weight", None)

    if hy_w is not None and ig_w is not None and not (
        pd.isna(hy_w) or pd.isna(ig_w)
    ):
        hy_w = float(hy_w)
        ig_w = float(ig_w)
        # Clamp to [0,1] and derive cash weight
        hy_w = max(0.0, min(1.0, hy_w))
        ig_w = max(0.0, min(1.0 - hy_w, ig_w))
        cash_w = max(0.0, 1.0 - hy_w - ig_w)
        w_vec = np.array([hy_w, ig_w, cash_w])
        cur_ret, cur_vol, cur_sharpe = _portfolio_stats(w_vec, mean_annual, cov_annual)
        current_allocation = {
            "hy_weight": hy_w,
            "ig_weight": ig_w,
            "cash_weight": cash_w,
            "expected_return": cur_ret,
            "annual_vol": cur_vol,
            "sharpe": cur_sharpe,
        }

    # ------------------------------------------------------------------
    # Per-asset statistics
    # ------------------------------------------------------------------
    asset_statistics = _asset_stats(mean_annual, cov_annual)

    return {
        "simulated": simulated,
        "min_var": min_var_portfolio,
        "max_sharpe": max_sharpe_portfolio,
        "current_allocation": current_allocation,
        "asset_stats": asset_statistics,
        "available": True,
        "n_obs": n_obs,
    }
