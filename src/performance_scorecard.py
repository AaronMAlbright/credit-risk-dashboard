"""
Strategy performance scorecard.

Full-period and rolling risk-adjusted metrics for the strategy vs. SP500
buy-and-hold benchmark, plus regime-conditional performance breakdown and
a monthly-return calendar.

Convention: returns are signed (positive = gain). Drawdown is reported as
a negative number. Risk-free rate assumed zero throughout (short history).

Public API
----------
  compute_full_period_stats   — annualised Sharpe, Calmar, max DD, hit rate, etc.
  compute_rolling_metrics     — rolling Sharpe + rolling max drawdown time series
  compute_regime_performance  — per-regime daily-return statistics
  compute_monthly_calendar    — monthly return pivot (strategy and SP500)
  run_performance_scorecard   — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZE = 252
_MIN_OBS   = 20          # below this, rolling/regime stats are NaN
_WINDOWS   = [63, 126]   # ~3 months, ~6 months

_STRATEGY_COL = "strategy_daily_return"
_SP500_COL    = "sp500_daily_return"
_DECISION_COL = "final_decision"
_DATE_COL     = "date"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ann_return(returns: np.ndarray) -> float:
    clean = returns[~np.isnan(returns)]
    if len(clean) < _MIN_OBS:
        return np.nan
    total = float(np.prod(1.0 + clean)) - 1.0
    years = len(clean) / _ANNUALIZE
    return float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else np.nan


def _ann_vol(returns: np.ndarray) -> float:
    clean = returns[~np.isnan(returns)]
    if len(clean) < _MIN_OBS:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(_ANNUALIZE))


def _sharpe(returns: np.ndarray) -> float:
    r = _ann_return(returns)
    v = _ann_vol(returns)
    if np.isnan(r) or np.isnan(v) or v < 1e-10:
        return np.nan
    return float(r / v)


def _max_drawdown(returns: np.ndarray) -> float:
    clean = returns[~np.isnan(returns)]
    if len(clean) < 2:
        return np.nan
    curve = np.cumprod(1.0 + clean)
    peak  = np.maximum.accumulate(curve)
    dd    = (curve - peak) / peak
    return float(dd.min())


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    def _calc(arr):
        clean = arr[~np.isnan(arr)]
        if len(clean) < _MIN_OBS:
            return np.nan
        std = float(clean.std(ddof=1))
        if std < 1e-10:
            return np.nan
        return float(clean.mean() / std * np.sqrt(_ANNUALIZE))
    return returns.rolling(window).apply(_calc, raw=True)


def _rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    def _calc(arr):
        clean = arr[~np.isnan(arr)]
        if len(clean) < 2:
            return np.nan
        curve = np.cumprod(1.0 + clean)
        peak  = np.maximum.accumulate(curve)
        return float(((curve - peak) / peak).min())
    return returns.rolling(window).apply(_calc, raw=True)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_full_period_stats(df: pd.DataFrame) -> dict[str, dict]:
    """
    Full-period risk-adjusted metrics for strategy and SP500.

    Returns dict: {label: {ann_return, ann_vol, sharpe, max_drawdown,
                            calmar, hit_rate, total_return, best_day,
                            worst_day, n_obs}}
    Only includes keys where the return column exists in df.
    """
    col_map = {"strategy": _STRATEGY_COL, "sp500": _SP500_COL}
    result: dict[str, dict] = {}

    for label, col in col_map.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna().values
        n = len(vals)
        ar  = _ann_return(vals)
        av  = _ann_vol(vals)
        sh  = _sharpe(vals)
        mdd = _max_drawdown(vals)
        calmar = float(ar / abs(mdd)) if (not np.isnan(ar) and not np.isnan(mdd)
                                          and abs(mdd) > 1e-10) else np.nan
        total  = float(np.prod(1.0 + vals)) - 1.0 if n > 0 else np.nan
        hr     = float((vals > 0).mean()) if n > 0 else np.nan
        best   = float(vals.max()) if n > 0 else np.nan
        worst  = float(vals.min()) if n > 0 else np.nan

        result[label] = {
            "n_obs":        n,
            "total_return": round(total, 5) if not np.isnan(total) else np.nan,
            "ann_return":   round(ar,    5) if not np.isnan(ar)    else np.nan,
            "ann_vol":      round(av,    5) if not np.isnan(av)    else np.nan,
            "sharpe":       round(sh,    3) if not np.isnan(sh)    else np.nan,
            "max_drawdown": round(mdd,   5) if not np.isnan(mdd)   else np.nan,
            "calmar":       round(calmar,3) if not np.isnan(calmar) else np.nan,
            "hit_rate":     round(hr,    4) if not np.isnan(hr)    else np.nan,
            "best_day":     round(best,  5) if not np.isnan(best)  else np.nan,
            "worst_day":    round(worst, 5) if not np.isnan(worst) else np.nan,
        }

    return result


def compute_rolling_metrics(
    df: pd.DataFrame,
    windows: list[int] = _WINDOWS,
) -> dict[int, pd.DataFrame]:
    """
    Rolling Sharpe and max drawdown for each window size.

    Returns dict: {window: DataFrame} with columns:
      date (if present), strategy_sharpe, sp500_sharpe,
      strategy_max_dd, sp500_max_dd
    Values are NaN until window is full.
    """
    if df.empty:
        return {w: pd.DataFrame() for w in windows}

    result = {}
    for w in windows:
        cols: dict[str, pd.Series] = {}

        if _STRATEGY_COL in df.columns:
            s = df[_STRATEGY_COL].astype(float)
            cols["strategy_sharpe"] = _rolling_sharpe(s, w).round(3)
            cols["strategy_max_dd"] = _rolling_max_drawdown(s, w).round(5)

        if _SP500_COL in df.columns:
            b = df[_SP500_COL].astype(float)
            cols["sp500_sharpe"] = _rolling_sharpe(b, w).round(3)
            cols["sp500_max_dd"] = _rolling_max_drawdown(b, w).round(5)

        if not cols:
            result[w] = pd.DataFrame()
            continue

        frame = pd.DataFrame(cols)
        if _DATE_COL in df.columns:
            frame.index = pd.to_datetime(df[_DATE_COL].values)

        result[w] = frame

    return result


def compute_regime_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-regime daily-return statistics for strategy and SP500.

    Returns DataFrame indexed by regime with columns:
      n_obs,
      strat_mean, strat_hit_rate, strat_ann_return, strat_ann_vol, strat_sharpe,
      sp500_mean, sp500_hit_rate, sp500_ann_return, sp500_ann_vol, sp500_sharpe
    Sorted by strat_sharpe descending (best first). Metrics NaN when n < _MIN_OBS.
    """
    if df.empty or _DECISION_COL not in df.columns:
        return pd.DataFrame()

    rows = []
    for regime, grp in df.groupby(_DECISION_COL):
        row: dict = {"regime": regime, "n_obs": len(grp)}

        for prefix, col in [("strat", _STRATEGY_COL), ("sp500", _SP500_COL)]:
            if col not in grp.columns:
                for k in ["mean", "hit_rate", "ann_return", "ann_vol", "sharpe"]:
                    row[f"{prefix}_{k}"] = np.nan
                continue
            vals = grp[col].dropna().values
            n = len(vals)
            row[f"{prefix}_mean"]       = round(float(vals.mean()), 6) if n > 0 else np.nan
            row[f"{prefix}_hit_rate"]   = round(float((vals > 0).mean()), 4) if n > 0 else np.nan
            row[f"{prefix}_ann_return"] = round(_ann_return(vals), 5) if n >= _MIN_OBS else np.nan
            row[f"{prefix}_ann_vol"]    = round(_ann_vol(vals), 5)    if n >= _MIN_OBS else np.nan
            row[f"{prefix}_sharpe"]     = round(_sharpe(vals), 3)     if n >= _MIN_OBS else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("regime")
    sort_col = "strat_sharpe" if "strat_sharpe" in out.columns else "n_obs"
    return out.sort_values(sort_col, ascending=False, na_position="last")


def compute_monthly_calendar(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Monthly return calendar for strategy and SP500.

    Returns dict: {label: pivot DataFrame}
      rows = year (int), columns = month (1–12)
      values = compound monthly return
    Only includes labels where the return column exists.
    """
    if df.empty or _DATE_COL not in df.columns:
        return {}

    df = df.copy()
    df[_DATE_COL] = pd.to_datetime(df[_DATE_COL])
    df["_year"]  = df[_DATE_COL].dt.year
    df["_month"] = df[_DATE_COL].dt.month

    result = {}
    for label, col in [("strategy", _STRATEGY_COL), ("sp500", _SP500_COL)]:
        if col not in df.columns:
            continue
        monthly = (
            df.groupby(["_year", "_month"])[col]
            .apply(lambda r: float(np.prod(1.0 + r.dropna())) - 1.0)
            .reset_index()
        )
        monthly.columns = ["year", "month", "return"]
        pivot = monthly.pivot(index="year", columns="month", values="return")
        pivot.columns.name = None
        result[label] = pivot.round(5)

    return result


def run_performance_scorecard(df: pd.DataFrame) -> dict:
    """
    Full performance scorecard.

    Returns dict with:
      full_period     — full-period stats per series
      rolling         — {window: DataFrame} rolling metrics
      regime_perf     — per-regime performance breakdown
      monthly_cal     — {label: monthly return pivot}
      windows         — list of window sizes used
    """
    return {
        "full_period":  compute_full_period_stats(df),
        "rolling":      compute_rolling_metrics(df),
        "regime_perf":  compute_regime_performance(df),
        "monthly_cal":  compute_monthly_calendar(df),
        "windows":      _WINDOWS,
    }
