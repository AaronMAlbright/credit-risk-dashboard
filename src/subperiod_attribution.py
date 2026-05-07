"""
Rolling sub-period attribution.

Splits the scored history into calendar-year sub-periods (plus a
full-period row) and computes performance, factor, and regime statistics
for each slice. Also produces rolling 63-day Sharpe and beta time series
to show whether the model's edge is consistent or concentrated in one window.

Public API
----------
  define_periods           — list of sub-period dicts from df
  compute_period_stats     — performance + factor stats for one sub-df
  compute_subperiod_table  — full stats DataFrame (one row per period)
  compute_rolling_stats    — rolling Sharpe + beta time series
  compute_regime_frequency — period × regime frequency pivot
  run_subperiod_attribution — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STRAT_COL   = "strategy_daily_return"
_SP500_COL   = "sp500_daily_return"
_LABEL_COL   = "final_decision"
_SCORE_COL   = "composite_risk_score_smooth"
_DATE_COL    = "date"
_ANNUALIZE   = 252
_MIN_OBS     = 20        # minimum rows for a period to be included
_ROLL_WINDOW = 63        # trading days (~3 months)


# ---------------------------------------------------------------------------
# Period definitions
# ---------------------------------------------------------------------------

def define_periods(df: pd.DataFrame) -> list[dict]:
    """
    Build sub-period definitions from df.

    Returns a list of dicts (ordered chronologically, full-period last):
      {label, start, end, n_obs, is_full}

    Each calendar year in df becomes one period. Years where the data
    covers only a partial year are labelled accordingly.
    A full-period entry is always appended last.
    """
    if _DATE_COL not in df.columns or df.empty:
        return []

    periods: list[dict] = []
    years = sorted(df[_DATE_COL].dt.year.unique())

    for y in years:
        sub = df[df[_DATE_COL].dt.year == y]
        if len(sub) < _MIN_OBS:
            continue
        first_month = sub[_DATE_COL].min().month
        last_month  = sub[_DATE_COL].max().month
        partial = (first_month != 1) or (last_month != 12)
        label   = f"{y} (partial)" if partial else str(y)
        periods.append({
            "label":   label,
            "start":   sub[_DATE_COL].min(),
            "end":     sub[_DATE_COL].max(),
            "n_obs":   len(sub),
            "is_full": False,
        })

    # Full period (always last)
    periods.append({
        "label":   "Full Period",
        "start":   df[_DATE_COL].min(),
        "end":     df[_DATE_COL].max(),
        "n_obs":   len(df),
        "is_full": True,
    })
    return periods


# ---------------------------------------------------------------------------
# Per-period statistics
# ---------------------------------------------------------------------------

def _drawdown_series(ret: np.ndarray) -> np.ndarray:
    """Compute running max-drawdown series from a return array."""
    cum   = np.cumprod(1.0 + ret)
    peak  = np.maximum.accumulate(cum)
    dd    = (cum - peak) / np.where(peak > 0, peak, 1.0)
    return dd


def _ols_simple(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """OLS beta, ann_alpha, R². NaN when insufficient data."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    if len(xc) < _MIN_OBS:
        return np.nan, np.nan, np.nan
    xm, ym = xc.mean(), yc.mean()
    var_x  = float(((xc - xm) ** 2).sum())
    if var_x < 1e-14:
        return np.nan, np.nan, np.nan
    beta  = float(((xc - xm) * (yc - ym)).sum() / var_x)
    alpha = float(ym - beta * xm)
    resid = yc - alpha - beta * xc
    ss_tot = float(((yc - ym) ** 2).sum())
    r2    = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 1e-14 else np.nan
    return beta, float(alpha * _ANNUALIZE), r2


def compute_period_stats(sub: pd.DataFrame) -> dict:
    """
    Compute performance + factor statistics for a sub-period DataFrame.

    Returns a flat dict with keys:
      n_obs, start, end
      strat_total_ret, strat_ann_ret, strat_ann_vol, strat_sharpe,
      strat_max_dd, strat_calmar, strat_hit_rate
      sp500_total_ret, sp500_ann_ret, sp500_ann_vol, sp500_sharpe, sp500_max_dd
      excess_ann_ret (strat − sp500 ann ret)
      beta, ann_alpha, r2
      mean_composite
    """
    out: dict = {}

    if sub.empty or len(sub) < _MIN_OBS:
        return out

    out["n_obs"] = int(len(sub))
    out["start"] = sub[_DATE_COL].min() if _DATE_COL in sub.columns else None
    out["end"]   = sub[_DATE_COL].max() if _DATE_COL in sub.columns else None

    def _perf(col: str) -> dict:
        if col not in sub.columns:
            return {}
        r  = sub[col].dropna().values.astype(float)
        if len(r) < _MIN_OBS:
            return {}
        n  = len(r)
        total_ret = float(np.prod(1.0 + r) - 1.0)
        ann_ret   = float((1.0 + total_ret) ** (_ANNUALIZE / n) - 1.0)
        ann_vol   = float(r.std(ddof=1) * np.sqrt(_ANNUALIZE))
        sharpe    = float(ann_ret / ann_vol) if ann_vol > 1e-10 else np.nan
        dd        = _drawdown_series(r)
        max_dd    = float(dd.min())
        calmar    = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-10 else np.nan
        hit_rate  = float((r > 0).mean())
        return {
            "total_ret": round(total_ret, 5),
            "ann_ret":   round(ann_ret,   5),
            "ann_vol":   round(ann_vol,   5),
            "sharpe":    round(sharpe,    3) if not np.isnan(sharpe) else np.nan,
            "max_dd":    round(max_dd,    5),
            "calmar":    round(calmar,    3) if not np.isnan(calmar) else np.nan,
            "hit_rate":  round(hit_rate,  4),
        }

    sp = _perf(_STRAT_COL)
    mp = _perf(_SP500_COL)
    for k, v in sp.items():
        out[f"strat_{k}"] = v
    for k, v in mp.items():
        out[f"sp500_{k}"] = v

    if "strat_ann_ret" in out and "sp500_ann_ret" in out:
        out["excess_ann_ret"] = round(out["strat_ann_ret"] - out["sp500_ann_ret"], 5)

    # Factor stats
    if _STRAT_COL in sub.columns and _SP500_COL in sub.columns:
        s = sub[_STRAT_COL].values.astype(float)
        b = sub[_SP500_COL].values.astype(float)
        beta, ann_alpha, r2 = _ols_simple(s, b)
        out["beta"]      = round(beta,      4) if not np.isnan(beta)      else np.nan
        out["ann_alpha"] = round(ann_alpha,  5) if not np.isnan(ann_alpha) else np.nan
        out["r2"]        = round(r2,         4) if not np.isnan(r2)        else np.nan

    if _SCORE_COL in sub.columns:
        out["mean_composite"] = round(float(sub[_SCORE_COL].mean()), 2)

    return out


# ---------------------------------------------------------------------------
# Full sub-period table
# ---------------------------------------------------------------------------

def compute_subperiod_table(
    df: pd.DataFrame,
    periods: list[dict] | None = None,
) -> pd.DataFrame:
    """
    One row per sub-period with all key metrics.

    If periods is None, calls define_periods(df).
    Returns DataFrame indexed by period label.
    """
    if periods is None:
        periods = define_periods(df)
    if not periods:
        return pd.DataFrame()

    rows = []
    for p in periods:
        sub = df[(df[_DATE_COL] >= p["start"]) & (df[_DATE_COL] <= p["end"])]
        stats = compute_period_stats(sub)
        stats["period"] = p["label"]
        rows.append(stats)

    return pd.DataFrame(rows).set_index("period")


# ---------------------------------------------------------------------------
# Rolling time series
# ---------------------------------------------------------------------------

def compute_rolling_stats(
    df: pd.DataFrame,
    window: int = _ROLL_WINDOW,
) -> pd.DataFrame:
    """
    Rolling Sharpe ratio and beta (strategy vs SP500).

    Returns DataFrame indexed by date with columns:
      strat_sharpe, sp500_sharpe, beta, ann_alpha
    NaN for the first (window-1) rows.
    """
    if _STRAT_COL not in df.columns or _SP500_COL not in df.columns or df.empty:
        return pd.DataFrame()

    s_ret = df[_STRAT_COL].values.astype(float)
    m_ret = df[_SP500_COL].values.astype(float)
    n     = len(df)

    strat_sh  = np.full(n, np.nan)
    sp500_sh  = np.full(n, np.nan)
    betas     = np.full(n, np.nan)
    alphas    = np.full(n, np.nan)

    for end in range(window - 1, n):
        start = end - window + 1
        sr = s_ret[start: end + 1]
        mr = m_ret[start: end + 1]

        mask = ~(np.isnan(sr) | np.isnan(mr))
        if mask.sum() < _MIN_OBS:
            continue

        for arr, out_arr in [(sr, strat_sh), (mr, sp500_sh)]:
            a  = arr[mask]
            vol = a.std(ddof=1) * np.sqrt(_ANNUALIZE)
            ann_r = (1 + np.prod(1 + a) ** (_ANNUALIZE / len(a)) - 1)
            if vol > 1e-10:
                out_arr[end] = ann_r / vol

        beta, ann_alpha, _ = _ols_simple(sr, mr)
        betas[end]  = beta
        alphas[end] = ann_alpha

    out = pd.DataFrame({
        "strat_sharpe": strat_sh,
        "sp500_sharpe": sp500_sh,
        "beta":         betas,
        "ann_alpha":    alphas,
    })
    if _DATE_COL in df.columns:
        out.index = pd.to_datetime(df[_DATE_COL].values)
    return out.round(4)


# ---------------------------------------------------------------------------
# Regime frequency pivot
# ---------------------------------------------------------------------------

def compute_regime_frequency(
    df: pd.DataFrame,
    periods: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Period × regime frequency pivot (values = fraction of days in each regime).

    Returns DataFrame: rows = period labels, columns = regime names.
    """
    if _LABEL_COL not in df.columns or _DATE_COL not in df.columns or df.empty:
        return pd.DataFrame()

    if periods is None:
        periods = define_periods(df)

    rows = {}
    for p in periods:
        sub = df[(df[_DATE_COL] >= p["start"]) & (df[_DATE_COL] <= p["end"])]
        if len(sub) < _MIN_OBS:
            continue
        vc = sub[_LABEL_COL].value_counts(normalize=True)
        rows[p["label"]] = vc

    if not rows:
        return pd.DataFrame()

    pivot = pd.DataFrame(rows).T.fillna(0.0)
    # Sort columns by mean frequency descending
    pivot = pivot[pivot.mean().sort_values(ascending=False).index]
    return pivot.round(4)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_subperiod_attribution(df: pd.DataFrame) -> dict:
    """
    Full sub-period attribution analysis.

    Returns dict:
      periods         — list of period dicts
      table           — DataFrame of per-period stats (indexed by label)
      rolling         — rolling Sharpe + beta DataFrame
      regime_freq     — period × regime frequency pivot
      roll_window     — rolling window size used
    """
    if df.empty or _DATE_COL not in df.columns:
        return {
            "periods":      [],
            "table":        pd.DataFrame(),
            "rolling":      pd.DataFrame(),
            "regime_freq":  pd.DataFrame(),
            "roll_window":  _ROLL_WINDOW,
        }

    periods     = define_periods(df)
    table       = compute_subperiod_table(df, periods)
    rolling     = compute_rolling_stats(df)
    regime_freq = compute_regime_frequency(df, periods)

    return {
        "periods":     periods,
        "table":       table,
        "rolling":     rolling,
        "regime_freq": regime_freq,
        "roll_window": _ROLL_WINDOW,
    }
