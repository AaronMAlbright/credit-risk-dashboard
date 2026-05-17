"""
Conditional Value-at-Risk (CVaR) by market regime.

VaR (Value-at-Risk): the loss not exceeded with probability (1-α).
  VaR_95 = 5th percentile of return distribution (loss threshold)

CVaR (Conditional VaR / Expected Shortfall): the expected loss GIVEN
that the loss exceeds VaR. CVaR is always worse than VaR and is the
preferred risk measure under Basel III because it captures tail shape.

Key insight this teaches: regimes have dramatically different tail risk profiles.
Risk-Off regime CVaR is often 3-5× worse than Risk-On regime CVaR.

Historical simulation is used (no distributional assumption).

Background
----------
Traditional VaR answers: "How bad can things get 95% of the time?"
CVaR answers: "When things are bad — in that worst 5% — how bad on average?"

VaR has a well-known flaw: it tells you nothing about the severity of losses
in the tail. Two strategies can have identical VaR_95 while one has occasional
-2% days in the tail and the other has occasional -20% days. CVaR distinguishes
them. This is why Basel III shifted from VaR to Expected Shortfall (CVaR).

Regime conditioning is the key addition here. A pooled CVaR across all regimes
masks the fact that tail risk is highly regime-dependent. A Risk-Off regime
often exhibits CVaR_95 of -3% to -5% per day, while Risk-On CVaR_95 might be
only -1% to -1.5%. Knowing your current regime therefore dramatically changes
your tail-risk estimate.

Historical simulation (non-parametric) is used throughout — no normality
assumption is imposed. This matters because return distributions exhibit
negative skew and excess kurtosis (fat tails), especially in stress regimes.

Public API
----------
  CONFIDENCE_LEVELS    — [0.90, 0.95, 0.99]
  compute_regime_var(df, return_col, confidence) → pd.DataFrame
  run_cvar_analysis(df) → dict
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

CONFIDENCE_LEVELS: list[float] = [0.90, 0.95, 0.99]

_REGIME_ORDER = ["Risk-On", "Neutral", "Caution", "Risk-Off"]


def _safe_percentile(series: pd.Series, q: float) -> float:
    """Return the q-th percentile of series, or NaN if too few observations."""
    arr = series.dropna().values
    if len(arr) < 5:
        return float("nan")
    return float(np.percentile(arr, q * 100))


def _skewness(series: pd.Series) -> float:
    """
    Third standardised central moment.

    Negative skewness means the left tail is longer — more extreme losses
    than a symmetric distribution would predict. Typical for equity/credit
    returns especially in stress regimes.
    """
    arr = series.dropna().values
    n = len(arr)
    if n < 4:
        return float("nan")
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma == 0:
        return float("nan")
    return float(np.mean(((arr - mu) / sigma) ** 3))


def _kurtosis(series: pd.Series) -> float:
    """
    Excess kurtosis = fourth standardised moment - 3.

    Normal distribution has kurtosis = 3, so excess kurtosis = 0.
    Positive excess kurtosis (leptokurtosis) means fatter tails than normal
    — more frequent extreme events. Financial returns routinely show excess
    kurtosis of 2–10, which is why parametric VaR (assuming normality)
    systematically under-estimates tail risk.
    """
    arr = series.dropna().values
    n = len(arr)
    if n < 5:
        return float("nan")
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    if sigma == 0:
        return float("nan")
    return float(np.mean(((arr - mu) / sigma) ** 4) - 3.0)


def _compute_stats_for_series(returns: pd.Series) -> dict:
    """
    Compute the full suite of risk statistics for a single return series.

    All statistics use the historical (non-parametric) approach: no
    distributional assumption is made. Annual figures assume 252 trading days.

    Parameters
    ----------
    returns : pd.Series
        Daily return series (decimal, e.g. 0.01 = 1%).

    Returns
    -------
    dict with keys:
        n_obs, mean_return, volatility, var_95, cvar_95, var_99, cvar_99,
        max_loss, skewness, kurtosis
    """
    clean = returns.dropna()
    n = len(clean)

    if n < 5:
        return {
            "n_obs": n,
            "mean_return": float("nan"),
            "volatility": float("nan"),
            "var_95": float("nan"),
            "cvar_95": float("nan"),
            "var_99": float("nan"),
            "cvar_99": float("nan"),
            "max_loss": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
        }

    # Annualised mean and vol
    mean_ann = float(clean.mean() * 252)
    vol_ann = float(clean.std(ddof=1) * math.sqrt(252))

    # VaR at 95% and 99% confidence (5th and 1st percentile)
    var_95 = _safe_percentile(clean, 0.05)
    var_99 = _safe_percentile(clean, 0.01)

    # CVaR: expected return given we are in the loss tail
    # This is the average of all returns worse than VaR — the true tail mean.
    tail_95 = clean[clean <= var_95]
    cvar_95 = float(tail_95.mean()) if len(tail_95) >= 2 else var_95

    tail_99 = clean[clean <= var_99]
    cvar_99 = float(tail_99.mean()) if len(tail_99) >= 1 else var_99

    max_loss = float(clean.min())

    return {
        "n_obs": n,
        "mean_return": mean_ann,
        "volatility": vol_ann,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
        "max_loss": max_loss,
        "skewness": _skewness(clean),
        "kurtosis": _kurtosis(clean),
    }


def compute_regime_var(
    df: pd.DataFrame,
    return_col: str = "strategy_daily_return",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compute CVaR and related tail-risk statistics grouped by market regime.

    This is the core educational output of the module. By segmenting returns
    into the four macro regimes (Risk-On / Neutral / Caution / Risk-Off), we
    reveal how tail risk varies with the macro environment.

    Key findings typically observed:
    - Risk-Off CVaR_95 is 3–5× worse than Risk-On CVaR_95
    - Risk-Off returns show the most negative skewness (fat left tail)
    - Kurtosis is highest in Risk-Off and lowest in Risk-On
    - Mean annualised return is negative in Risk-Off regimes

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `final_decision` and the specified `return_col`.
    return_col : str
        Column name for daily returns. Default: "strategy_daily_return".
    confidence : float
        Confidence level (unused in output columns but preserved for API
        compatibility). Output always includes 95% and 99% statistics.

    Returns
    -------
    pd.DataFrame
        One row per regime, indexed by regime name, with columns:
        n_obs, mean_return, volatility, var_95, cvar_95, var_99, cvar_99,
        max_loss, skewness, kurtosis.
        Ordered Risk-On → Neutral → Caution → Risk-Off where present.
    """
    if return_col not in df.columns or "final_decision" not in df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for regime, group in df.groupby("final_decision"):
        stats = _compute_stats_for_series(group[return_col])
        stats["regime"] = regime
        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("regime")

    # Enforce canonical regime ordering where possible
    ordered = [r for r in _REGIME_ORDER if r in result.index]
    others = [r for r in result.index if r not in _REGIME_ORDER]
    result = result.loc[ordered + others]

    return result


def _rolling_cvar(returns: pd.Series, window: int = 252, q: float = 0.05) -> pd.Series:
    """
    Compute a rolling CVaR_95 series using a trailing window.

    Rolling CVaR answers: "What is the expected shortfall based on the past
    `window` trading days?" This is more informative than static CVaR because
    it shows how tail risk evolved through time — spiking during crises and
    compressing during calm periods.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    window : int
        Rolling window in trading days. 252 = 1 year.
    q : float
        Tail quantile. 0.05 corresponds to CVaR_95.

    Returns
    -------
    pd.Series
        Rolling CVaR with the same index as `returns`. NaN for the first
        (window - 1) observations.
    """
    def _cvar_func(arr: np.ndarray) -> float:
        cutoff = np.percentile(arr, q * 100)
        tail = arr[arr <= cutoff]
        return float(tail.mean()) if len(tail) >= 1 else float(cutoff)

    clean = returns.dropna()
    if len(clean) < window:
        return pd.Series(dtype=float, index=returns.index)

    rolling_vals = (
        clean.rolling(window=window, min_periods=max(20, window // 4))
        .apply(_cvar_func, raw=True)
    )
    return rolling_vals.reindex(returns.index)


def run_cvar_analysis(df: pd.DataFrame) -> dict:
    """
    Run the complete CVaR analysis pipeline and return a results bundle.

    Attempts to use `strategy_daily_return` first; falls back to
    `sp500_daily_return` if the strategy column is absent. Sets
    `available=False` if neither column is present.

    The result dictionary is designed for direct consumption by dashboard
    components — all heavy computation is done here so display code remains
    thin.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including regime labels, returns, and a date column.

    Returns
    -------
    dict with keys:
        regime_stats : pd.DataFrame
            Per-regime CVaR statistics from compute_regime_var().
        full_period : dict
            CVaR statistics for the entire return series.
        oos_period : dict
            CVaR statistics for observations after 2016-01-01 (out-of-sample
            period used for regime model validation).
        current_regime : str
            Regime label from the most recent row.
        current_cvar_95 : float
            CVaR_95 for the current regime (daily, decimal).
        current_cvar_99 : float
            CVaR_99 for the current regime (daily, decimal).
        rolling_cvar : pd.Series
            252-day rolling CVaR_95, indexed by date.
        available : bool
            False if no return column could be located.
    """
    # --- Column resolution ---
    return_col: str | None = None
    available = True

    if "strategy_daily_return" in df.columns:
        return_col = "strategy_daily_return"
    elif "sp500_daily_return" in df.columns:
        return_col = "sp500_daily_return"
        available = False  # fallback: strategy series absent
    else:
        return {
            "regime_stats": pd.DataFrame(),
            "full_period": {},
            "oos_period": {},
            "current_regime": "Unknown",
            "current_cvar_95": float("nan"),
            "current_cvar_99": float("nan"),
            "rolling_cvar": pd.Series(dtype=float),
            "available": False,
        }

    # --- Regime stats ---
    regime_stats = compute_regime_var(df, return_col=return_col)

    # --- Full-period stats ---
    full_period = _compute_stats_for_series(df[return_col])

    # --- OOS period (post-2016) ---
    oos_period: dict = {}
    if "date" in df.columns:
        df_dated = df.copy()
        df_dated["date"] = pd.to_datetime(df_dated["date"], errors="coerce")
        oos_mask = df_dated["date"] >= pd.Timestamp("2016-01-01")
        if oos_mask.sum() >= 20:
            oos_period = _compute_stats_for_series(df_dated.loc[oos_mask, return_col])
    if not oos_period:
        # Fallback: use last 50% of data as proxy OOS
        half = len(df) // 2
        if half >= 20:
            oos_period = _compute_stats_for_series(df[return_col].iloc[half:])

    # --- Current regime ---
    current_regime = "Unknown"
    if "final_decision" in df.columns:
        last_valid = df["final_decision"].dropna()
        if len(last_valid) > 0:
            current_regime = str(last_valid.iloc[-1])

    # --- CVaR for current regime ---
    current_cvar_95 = float("nan")
    current_cvar_99 = float("nan")
    if current_regime in regime_stats.index:
        current_cvar_95 = float(regime_stats.loc[current_regime, "cvar_95"])
        current_cvar_99 = float(regime_stats.loc[current_regime, "cvar_99"])

    # --- Rolling CVaR ---
    rolling_index = df.get("date", pd.RangeIndex(len(df)))
    returns_indexed = df[return_col].copy()
    if "date" in df.columns:
        returns_indexed.index = pd.to_datetime(df["date"], errors="coerce")
    rolling_cvar_series = _rolling_cvar(returns_indexed, window=252, q=0.05)

    return {
        "regime_stats": regime_stats,
        "full_period": full_period,
        "oos_period": oos_period,
        "current_regime": current_regime,
        "current_cvar_95": current_cvar_95,
        "current_cvar_99": current_cvar_99,
        "rolling_cvar": rolling_cvar_series,
        "available": available,
    }
