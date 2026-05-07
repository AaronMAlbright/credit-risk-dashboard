"""
Tail risk analysis.

Regime-conditional CVaR (Conditional Value at Risk / Expected Shortfall),
rolling CVaR over time, forward drawdown profiles by regime, and a
strategy-vs-benchmark tail comparison.

Convention: returns are signed (positive = gain). CVaR and VaR are reported
as negative numbers when losses dominate, consistent with financial practice.

Public API
----------
  compute_var_cvar             — point VaR and CVaR from a return array
  compute_regime_tail_stats    — per-regime tail statistics with bootstrap CIs
  compute_rolling_cvar         — rolling 60-day CVaR time series
  compute_drawdown_by_regime   — mean/worst future drawdown at regime entry
  compute_strategy_vs_sp500    — side-by-side tail comparison
  run_tail_risk_analysis       — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ALPHA = 0.05          # 95% confidence level (5th-percentile tail)
DEFAULT_ROLLING_WINDOW = 60   # trading days ≈ 3 months
DEFAULT_N_BOOT = 500
DEFAULT_CI = 0.95
_MIN_OBS_CVAR = 10            # below this, CVaR is not computed
_MIN_OBS_BOOT = 20            # below this, CI is NaN


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------

def compute_var_cvar(
    returns: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """
    Point estimates for VaR and CVaR at confidence level (1-alpha).

    Parameters
    ----------
    returns : array of daily (or period) returns
    alpha   : tail probability (0.05 → 95% confidence)

    Returns
    -------
    (var, cvar) both as floats. NaN when fewer than _MIN_OBS_CVAR observations.
    VaR  = alpha-quantile of returns
    CVaR = mean of returns ≤ VaR  (always ≤ VaR)
    """
    clean = returns[~np.isnan(returns)]
    if len(clean) < _MIN_OBS_CVAR:
        return np.nan, np.nan

    var = float(np.percentile(clean, alpha * 100))
    tail = clean[clean <= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def _bootstrap_cvar_ci(
    returns: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
) -> tuple[float, float]:
    """Bootstrap percentile CI on CVaR. Returns (ci_lower, ci_upper)."""
    clean = returns[~np.isnan(returns)]
    n = len(clean)
    if n < _MIN_OBS_BOOT:
        return np.nan, np.nan

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 - (1 - ci) / 2) * 100

    boot_cvars = []
    for _ in range(n_boot):
        sample = clean[np.random.choice(n, size=n, replace=True)]
        var_b = np.percentile(sample, alpha * 100)
        tail_b = sample[sample <= var_b]
        if len(tail_b) > 0:
            boot_cvars.append(float(tail_b.mean()))

    if not boot_cvars:
        return np.nan, np.nan

    boot_arr = np.array(boot_cvars)
    return float(np.percentile(boot_arr, lo_pct)), float(np.percentile(boot_arr, hi_pct))


# ---------------------------------------------------------------------------
# Per-regime tail statistics
# ---------------------------------------------------------------------------

def compute_regime_tail_stats(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    return_col: str = "sp500_daily_return",
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """
    Per-regime VaR, CVaR, and bootstrap CIs.

    Returns DataFrame indexed by regime with columns:
      n_obs, mean_return, std_return, var_95, cvar_95,
      ci_lower, ci_upper, max_loss
    Sorted by cvar_95 ascending (worst tail first).
    """
    if regime_col not in df.columns or return_col not in df.columns or df.empty:
        return pd.DataFrame()

    rng_state = np.random.get_state()
    np.random.seed(42)
    try:
        rows = []
        for regime, grp in df.groupby(regime_col):
            vals = grp[return_col].dropna().values
            n = len(vals)
            var, cvar = compute_var_cvar(vals, alpha)
            ci_lo, ci_hi = _bootstrap_cvar_ci(vals, alpha, n_boot, ci) if n >= _MIN_OBS_BOOT \
                else (np.nan, np.nan)
            rows.append({
                "regime":      regime,
                "n_obs":       n,
                "mean_return": round(float(vals.mean()), 5) if n > 0 else np.nan,
                "std_return":  round(float(vals.std(ddof=1)), 5) if n > 1 else np.nan,
                "var_95":      round(var, 5) if not np.isnan(var) else np.nan,
                "cvar_95":     round(cvar, 5) if not np.isnan(cvar) else np.nan,
                "ci_lower":    round(ci_lo, 5) if not np.isnan(ci_lo) else np.nan,
                "ci_upper":    round(ci_hi, 5) if not np.isnan(ci_hi) else np.nan,
                "max_loss":    round(float(vals.min()), 5) if n > 0 else np.nan,
            })
    finally:
        np.random.set_state(rng_state)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index("regime")
        .sort_values("cvar_95", ascending=True)  # worst first
    )


# ---------------------------------------------------------------------------
# Rolling CVaR
# ---------------------------------------------------------------------------

def compute_rolling_cvar(
    df: pd.DataFrame,
    return_col: str = "sp500_daily_return",
    window: int = DEFAULT_ROLLING_WINDOW,
    alpha: float = DEFAULT_ALPHA,
) -> pd.Series:
    """
    Rolling CVaR time series.

    Returns pd.Series indexed by date (if present) or integer index.
    Values are NaN until the window is full.
    """
    if return_col not in df.columns or df.empty:
        return pd.Series(dtype=float)

    returns = df[return_col].astype(float)

    def _rolling_cvar(arr):
        clean = arr[~np.isnan(arr)]
        if len(clean) < _MIN_OBS_CVAR:
            return np.nan
        var = np.percentile(clean, alpha * 100)
        tail = clean[clean <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    result = returns.rolling(window).apply(_rolling_cvar, raw=True)

    if "date" in df.columns:
        result.index = pd.to_datetime(df["date"].values)

    return result.round(5)


# ---------------------------------------------------------------------------
# Forward drawdown by regime
# ---------------------------------------------------------------------------

def compute_drawdown_by_regime(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
) -> pd.DataFrame:
    """
    Mean and worst (most negative) future drawdown at entry into each regime.

    Uses sp500_future_drawdown_30d and sp500_future_drawdown_60d where available.

    Returns DataFrame indexed by regime with columns:
      n_obs, mean_dd_30d, worst_dd_30d, mean_dd_60d, worst_dd_60d
    """
    if regime_col not in df.columns or df.empty:
        return pd.DataFrame()

    dd_cols = {
        "30d": "sp500_future_drawdown_30d",
        "60d": "sp500_future_drawdown_60d",
    }
    available_dd = {k: v for k, v in dd_cols.items() if v in df.columns}

    if not available_dd:
        return pd.DataFrame()

    rows = []
    for regime, grp in df.groupby(regime_col):
        row = {"regime": regime, "n_obs": len(grp)}
        for suffix, col in available_dd.items():
            vals = grp[col].dropna().values
            row[f"mean_dd_{suffix}"] = round(float(vals.mean()), 5) if len(vals) > 0 else np.nan
            row[f"worst_dd_{suffix}"] = round(float(vals.min()), 5) if len(vals) > 0 else np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index("regime")
        .sort_values("mean_dd_30d" if "mean_dd_30d" in pd.DataFrame(rows).columns else "n_obs",
                     ascending=True)
    )


# ---------------------------------------------------------------------------
# Strategy vs SP500 tail comparison
# ---------------------------------------------------------------------------

def compute_strategy_vs_sp500(
    df: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    """
    Side-by-side tail risk: strategy vs SP500 buy-and-hold.

    Returns dict with keys: strategy, sp500
    Each value is a dict: {var, cvar, ci_lower, ci_upper, max_loss, ann_vol, n_obs}
    """
    result = {}
    col_map = {
        "strategy": "strategy_daily_return",
        "sp500":    "sp500_daily_return",
    }

    rng_state = np.random.get_state()
    np.random.seed(42)
    try:
        for label, col in col_map.items():
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            n = len(vals)
            var, cvar = compute_var_cvar(vals, alpha)
            ci_lo, ci_hi = _bootstrap_cvar_ci(vals, alpha, n_boot) if n >= _MIN_OBS_BOOT \
                else (np.nan, np.nan)
            ann_vol = float(vals.std(ddof=1) * np.sqrt(252)) if n > 1 else np.nan
            result[label] = {
                "n_obs":     n,
                "var":       round(var, 5) if not np.isnan(var) else np.nan,
                "cvar":      round(cvar, 5) if not np.isnan(cvar) else np.nan,
                "ci_lower":  round(ci_lo, 5) if not np.isnan(ci_lo) else np.nan,
                "ci_upper":  round(ci_hi, 5) if not np.isnan(ci_hi) else np.nan,
                "max_loss":  round(float(vals.min()), 5) if n > 0 else np.nan,
                "ann_vol":   round(ann_vol, 5) if not np.isnan(ann_vol) else np.nan,
            }
    finally:
        np.random.set_state(rng_state)

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_tail_risk_analysis(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict:
    """
    Full tail risk analysis.

    Returns dict with:
      regime_tail_stats   — per-regime CVaR/VaR table (SP500 daily returns)
      strategy_tail_stats — per-regime CVaR/VaR table (strategy daily returns)
      rolling_cvar_sp500  — rolling CVaR series for SP500
      rolling_cvar_strat  — rolling CVaR series for strategy
      drawdown_by_regime  — future drawdown profile per regime
      vs_benchmark        — strategy vs SP500 full-sample tail comparison
      alpha               — tail probability used
    """
    return {
        "regime_tail_stats":   compute_regime_tail_stats(df, regime_col, "sp500_daily_return",
                                                          alpha, n_boot),
        "strategy_tail_stats": compute_regime_tail_stats(df, regime_col, "strategy_daily_return",
                                                          alpha, n_boot)
                               if "strategy_daily_return" in df.columns else pd.DataFrame(),
        "rolling_cvar_sp500":  compute_rolling_cvar(df, "sp500_daily_return", rolling_window, alpha),
        "rolling_cvar_strat":  compute_rolling_cvar(df, "strategy_daily_return", rolling_window, alpha)
                               if "strategy_daily_return" in df.columns
                               else pd.Series(dtype=float),
        "drawdown_by_regime":  compute_drawdown_by_regime(df, regime_col),
        "vs_benchmark":        compute_strategy_vs_sp500(df, alpha, n_boot),
        "alpha":               alpha,
    }
