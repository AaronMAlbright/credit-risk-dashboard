"""
Factor exposure and beta decomposition.

Fits a single-factor OLS model (strategy ~ alpha + beta * SP500) over the
full period and on rolling windows. Decomposes cumulative strategy returns
into market-beta contribution and residual alpha. Reports regime-conditional
beta to validate that the signal correctly modulates market exposure.

Public API
----------
  compute_factor_regression      — full-period OLS stats
  compute_rolling_factor_exposure — rolling beta / alpha / R² time series
  compute_regime_beta            — per-regime beta and alpha breakdown
  compute_return_decomposition   — daily and cumulative alpha/beta contributions
  run_factor_analysis            — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZE    = 252
_MIN_OBS      = 20        # below this, stats are NaN
_WINDOWS      = [63, 126] # ~3 months, ~6 months

_STRATEGY_COL = "strategy_daily_return"
_SP500_COL    = "sp500_daily_return"
_DECISION_COL = "final_decision"
_DATE_COL     = "date"


# ---------------------------------------------------------------------------
# Core OLS helper
# ---------------------------------------------------------------------------

def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """
    Single-factor OLS: y = alpha + beta * x + residual.

    Returns (alpha_daily, beta, r2, residual_std_daily).
    All NaN when fewer than _MIN_OBS clean observations or zero variance in x.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    n = len(xc)
    if n < _MIN_OBS:
        return np.nan, np.nan, np.nan, np.nan

    xm, ym  = xc.mean(), yc.mean()
    var_x   = float(((xc - xm) ** 2).sum())
    if var_x < 1e-14:
        return np.nan, np.nan, np.nan, np.nan

    beta    = float(((xc - xm) * (yc - ym)).sum() / var_x)
    alpha   = float(ym - beta * xm)
    resid   = yc - alpha - beta * xc
    ss_tot  = float(((yc - ym) ** 2).sum())
    r2      = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 1e-14 else np.nan
    dof     = max(n - 2, 1)
    resid_std = float(np.sqrt((resid ** 2).sum() / dof))

    return alpha, beta, r2, resid_std


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_factor_regression(df: pd.DataFrame) -> dict:
    """
    Full-period single-factor OLS: strategy ~ alpha + beta * SP500.

    Returns dict with:
      beta             — market exposure (0=cash, 1=fully invested)
      alpha_daily      — daily alpha intercept
      ann_alpha        — annualised alpha  (alpha_daily * 252)
      r2               — fraction of strategy variance explained by SP500
      residual_vol     — annualised residual (idiosyncratic) volatility
      info_ratio       — ann_alpha / residual_vol  (risk-adjusted alpha)
      tracking_error   — ann vol of (strategy − SP500) return differences
      n_obs            — clean observations used
    All values NaN when return columns are missing or insufficient data.
    """
    if _STRATEGY_COL not in df.columns or _SP500_COL not in df.columns or df.empty:
        return {}

    s = df[_STRATEGY_COL].values.astype(float)
    b = df[_SP500_COL].values.astype(float)

    alpha_d, beta, r2, resid_std_d = _ols(s, b)

    ann_alpha    = float(alpha_d * _ANNUALIZE)      if not np.isnan(alpha_d) else np.nan
    resid_vol    = float(resid_std_d * np.sqrt(_ANNUALIZE)) if not np.isnan(resid_std_d) else np.nan
    info_ratio   = float(ann_alpha / resid_vol) if (not np.isnan(ann_alpha)
                                                    and not np.isnan(resid_vol)
                                                    and resid_vol > 1e-10) else np.nan

    mask = ~(np.isnan(s) | np.isnan(b))
    te   = float((s[mask] - b[mask]).std(ddof=1) * np.sqrt(_ANNUALIZE)) if mask.sum() > 1 else np.nan

    return {
        "beta":           round(beta,       4) if not np.isnan(beta)       else np.nan,
        "alpha_daily":    round(alpha_d,    6) if not np.isnan(alpha_d)    else np.nan,
        "ann_alpha":      round(ann_alpha,  5) if not np.isnan(ann_alpha)  else np.nan,
        "r2":             round(r2,         4) if not np.isnan(r2)         else np.nan,
        "residual_vol":   round(resid_vol,  5) if not np.isnan(resid_vol)  else np.nan,
        "info_ratio":     round(info_ratio, 3) if not np.isnan(info_ratio) else np.nan,
        "tracking_error": round(te,         5) if not np.isnan(te)         else np.nan,
        "n_obs":          int(mask.sum()),
    }


def compute_rolling_factor_exposure(
    df: pd.DataFrame,
    windows: list[int] = _WINDOWS,
) -> dict[int, pd.DataFrame]:
    """
    Rolling OLS beta, annualised alpha, and R² for each window.

    Returns dict: {window: DataFrame} with columns:
      beta, ann_alpha, r2
    Index is datetime (from 'date' col) or integer. NaN until window is full.
    """
    if _STRATEGY_COL not in df.columns or _SP500_COL not in df.columns or df.empty:
        return {w: pd.DataFrame() for w in windows}

    s_series = df[_STRATEGY_COL].astype(float)
    b_series = df[_SP500_COL].astype(float)
    result   = {}

    for w in windows:
        betas, alphas, r2s = [], [], []

        for end in range(len(df)):
            start = end - w + 1
            if start < 0:
                betas.append(np.nan)
                alphas.append(np.nan)
                r2s.append(np.nan)
                continue
            yi = s_series.iloc[start : end + 1].values
            xi = b_series.iloc[start : end + 1].values
            a, be, r2, _ = _ols(yi, xi)
            betas.append(be)
            alphas.append(float(a * _ANNUALIZE) if not np.isnan(a) else np.nan)
            r2s.append(r2)

        frame = pd.DataFrame({"beta": betas, "ann_alpha": alphas, "r2": r2s})
        if _DATE_COL in df.columns:
            frame.index = pd.to_datetime(df[_DATE_COL].values)

        result[w] = frame.round(4)

    return result


def compute_regime_beta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-regime beta, annualised alpha, R², and n_obs.

    Returns DataFrame indexed by regime, sorted by beta ascending
    (lowest market exposure first). Rows with n < _MIN_OBS get NaN stats.
    """
    if (_STRATEGY_COL not in df.columns or _SP500_COL not in df.columns
            or _DECISION_COL not in df.columns or df.empty):
        return pd.DataFrame()

    rows = []
    for regime, grp in df.groupby(_DECISION_COL):
        s = grp[_STRATEGY_COL].values.astype(float)
        b = grp[_SP500_COL].values.astype(float)
        n = int((~(np.isnan(s) | np.isnan(b))).sum())
        a, be, r2, resid_std = _ols(s, b)

        rows.append({
            "regime":       regime,
            "n_obs":        n,
            "beta":         round(be,    4) if not np.isnan(be)    else np.nan,
            "ann_alpha":    round(a * _ANNUALIZE, 4) if not np.isnan(a) else np.nan,
            "r2":           round(r2,    4) if not np.isnan(r2)    else np.nan,
            "residual_vol": round(float(resid_std * np.sqrt(_ANNUALIZE)), 4)
                            if not np.isnan(resid_std) else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index("regime")
        .sort_values("beta", ascending=True)
    )


def compute_return_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose strategy returns into beta contribution and residual alpha.

    Uses the full-period beta to split each day's strategy return:
      beta_contribution_t  = beta * sp500_return_t
      alpha_contribution_t = strategy_return_t − beta_contribution_t

    Returns DataFrame with columns:
      date (if present), strategy_return, sp500_return,
      beta_contribution, alpha_contribution,
      cum_strategy, cum_sp500, cum_beta, cum_alpha
    All cumulative columns start at 1.0.
    """
    if _STRATEGY_COL not in df.columns or _SP500_COL not in df.columns or df.empty:
        return pd.DataFrame()

    reg = compute_factor_regression(df)
    beta = reg.get("beta", np.nan)
    if np.isnan(beta):
        return pd.DataFrame()

    out = df[[_DATE_COL, _STRATEGY_COL, _SP500_COL]].copy() if _DATE_COL in df.columns \
        else df[[_STRATEGY_COL, _SP500_COL]].copy()

    out = out.rename(columns={_STRATEGY_COL: "strategy_return",
                               _SP500_COL:    "sp500_return"})

    out["beta_contribution"]  = beta * out["sp500_return"]
    out["alpha_contribution"] = out["strategy_return"] - out["beta_contribution"]

    for col, src in [("cum_strategy", "strategy_return"),
                     ("cum_sp500",    "sp500_return"),
                     ("cum_beta",     "beta_contribution"),
                     ("cum_alpha",    "alpha_contribution")]:
        out[col] = (1.0 + out[src].fillna(0)).cumprod()

    return out.reset_index(drop=True)


def compute_performance_attribution(df: pd.DataFrame) -> dict:
    """
    Decompose total strategy performance into distinct economic contributions.

    Components (all annualised):
      beta_reduction   — return from having lower-than-1 market beta
                         (constant_beta_port − full_beta_port)
      vol_timing       — improvement from time-varying beta vs constant beta
      crash_avoidance  — P&L attribution during SP500 drawdowns > 10%
      regime_timing    — P&L from regime transitions (entry/exit returns)
      residual_alpha   — total − explained

    Uses the full-period OLS beta as the constant-beta reference.

    Returns dict with per-component annualised return, cumulative series,
    and a summary table.  All NaN when data is insufficient.
    """
    if (_STRATEGY_COL not in df.columns or _SP500_COL not in df.columns
            or df.empty):
        return {}

    n   = len(df)
    s   = df[_STRATEGY_COL].values.astype(float)
    b   = df[_SP500_COL].values.astype(float)
    mask = ~(np.isnan(s) | np.isnan(b))

    reg  = compute_factor_regression(df)
    beta = reg.get("beta", np.nan)
    if np.isnan(beta):
        return {}

    # --- 1. Beta reduction: how much "being at beta<1" helped vs full SP500 ---
    # Full-beta port: r_b = b (buy-and-hold SP500)
    # Constant-beta port: r_c = beta * b
    # Beta reduction = r_c - r_b  (negative beta means you owned less than market)
    beta_reduction_daily = (beta - 1.0) * b   # = r_c - r_b per day

    # --- 2. Vol timing: actual strategy vs constant-beta reference ---
    # vol_timing = strategy - constant_beta_port
    # This captures whether varying beta (timing) added value over constant beta
    vol_timing_daily = s - beta * b            # = alpha residual at full-period beta

    # --- 3. Crash avoidance: SP500 drawdowns > 10% ---
    cum_sp500   = (1 + pd.Series(b).fillna(0)).cumprod().values
    peak_sp500  = np.maximum.accumulate(cum_sp500)
    dd_sp500    = cum_sp500 / peak_sp500 - 1
    crash_mask  = dd_sp500 < -0.10

    crash_strat = np.where(crash_mask, s, 0.0)
    crash_sp500 = np.where(crash_mask, b, 0.0)
    crash_avoid_daily = crash_strat - beta * crash_sp500   # outperformance in crashes

    # --- 4. Regime timing: P&L around regime transitions (± 5 days) ---
    regime_timing_daily = np.zeros(n)
    if _DECISION_COL in df.columns:
        labels  = df[_DECISION_COL].values
        changes = np.where(np.array(labels[1:]) != np.array(labels[:-1]))[0] + 1
        window  = 5
        for ch in changes:
            start = max(0, ch - window)
            end   = min(n, ch + window + 1)
            # Contribution vs constant-beta reference around transition
            regime_timing_daily[start:end] += (
                s[start:end] - beta * b[start:end]
            ) / max(1, len(changes))  # normalise to avoid double-counting

    # --- 5. Residual alpha (anything not explained above) ---
    # Total strategy excess = s - b
    # = beta_reduction + vol_timing
    # But we want pure alpha not explained by any timing
    _, _, _, _ = _ols(s, b)
    alpha_d = reg.get("alpha_daily", 0.0)
    residual_daily = np.full(n, float(alpha_d) if not np.isnan(alpha_d) else 0.0)

    # --- Annualised summaries ---
    def _ann(arr):
        v = arr[mask]
        return float(v.mean() * _ANNUALIZE) if len(v) > _MIN_OBS else np.nan

    def _cumulative(arr):
        return (1 + pd.Series(arr).fillna(0)).cumprod().values

    summary = [
        {"component": "Beta Reduction",
         "description": "Lower-than-market beta (defensive positioning)",
         "ann_return":  _ann(beta_reduction_daily)},
        {"component": "Volatility / Regime Timing",
         "description": "Time-varying beta vs constant beta (when defensive paid off)",
         "ann_return":  _ann(vol_timing_daily)},
        {"component": "Crash Avoidance",
         "description": "Outperformance during SP500 drawdowns >10%",
         "ann_return":  _ann(crash_avoid_daily)},
        {"component": "Regime Transition P&L",
         "description": "Returns ±5 days around regime changes",
         "ann_return":  _ann(regime_timing_daily)},
        {"component": "Residual Alpha",
         "description": "Unexplained excess return (OLS intercept)",
         "ann_return":  _ann(residual_daily)},
    ]

    out_series = pd.DataFrame()
    if _DATE_COL in df.columns:
        out_series["date"] = df[_DATE_COL].values
    out_series["beta_reduction_daily"]   = beta_reduction_daily
    out_series["vol_timing_daily"]       = vol_timing_daily
    out_series["crash_avoidance_daily"]  = crash_avoid_daily
    out_series["regime_timing_daily"]    = regime_timing_daily
    out_series["residual_alpha_daily"]   = residual_daily
    out_series["strategy_daily"]         = s
    out_series["sp500_daily"]            = b
    for col in [c for c in out_series.columns if c.endswith("_daily")]:
        cum_col = "cum_" + col.replace("_daily", "")
        out_series[cum_col] = _cumulative(out_series[col].values)

    # Key headline: what fraction of total strategy return is "just beta timing"?
    strat_ann  = _ann(s)
    sp500_ann  = _ann(b)
    beta_ann   = _ann(beta_reduction_daily)
    timing_ann = _ann(vol_timing_daily)
    if not np.isnan(strat_ann) and abs(strat_ann) > 1e-6:
        beta_pct   = (beta_ann   / strat_ann) * 100 if not np.isnan(beta_ann)   else np.nan
        timing_pct = (timing_ann / strat_ann) * 100 if not np.isnan(timing_ann) else np.nan
    else:
        beta_pct = timing_pct = np.nan

    return {
        "summary":           pd.DataFrame(summary),
        "series":            out_series,
        "ann_strategy":      round(strat_ann,  4) if not np.isnan(strat_ann)  else None,
        "ann_sp500":         round(sp500_ann,  4) if not np.isnan(sp500_ann)  else None,
        "full_period_beta":  round(beta,       4),
        "beta_share_pct":    round(beta_pct,   1) if not np.isnan(beta_pct)   else None,
        "timing_share_pct":  round(timing_pct, 1) if not np.isnan(timing_pct) else None,
    }


def run_factor_analysis(df: pd.DataFrame) -> dict:
    """
    Full factor exposure analysis.

    Returns dict with:
      regression       — full-period OLS stats dict
      rolling          — {window: DataFrame} rolling beta/alpha/R²
      regime_beta      — per-regime beta DataFrame
      decomposition    — return decomposition DataFrame
      attribution      — performance attribution breakdown
      multi_factor     — multi-factor OLS result dict
      windows          — window sizes used
    """
    return {
        "regression":    compute_factor_regression(df),
        "rolling":       compute_rolling_factor_exposure(df),
        "regime_beta":   compute_regime_beta(df),
        "decomposition": compute_return_decomposition(df),
        "attribution":   compute_performance_attribution(df),
        "multi_factor":  compute_multi_factor_regression(df),
        "windows":       _WINDOWS,
    }


# ---------------------------------------------------------------------------
# Multi-factor OLS (numpy lstsq — no scipy)
# ---------------------------------------------------------------------------

_MULTI_FACTORS: list[tuple[str, str]] = [
    ("sp500_daily_return",   "SP500"),
    ("_vix_daily_chg",       "VIX Δ"),
    ("_hy_daily_chg",        "HY Δ"),
    ("_rates_daily_chg",     "Rates Δ"),
    ("_momentum_proxy",      "Momentum"),
]


def _build_factor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build multi-factor matrix from scored DataFrame.
    Derives VIX/HY/rates daily changes and 12m momentum proxy.
    """
    out = df.copy()

    if "vix" in out.columns:
        out["_vix_daily_chg"] = out["vix"].diff()
    if "hy_spread" in out.columns:
        out["_hy_daily_chg"] = out["hy_spread"].diff()
    if "yield_10y" in out.columns:
        out["_rates_daily_chg"] = out["yield_10y"].diff()
    if "sp500_return_30d" in out.columns:
        out["_momentum_proxy"] = out["sp500_return_30d"]
    elif "sp500_daily_return" in out.columns:
        out["_momentum_proxy"] = out["sp500_daily_return"].rolling(252).sum()

    return out


def compute_multi_factor_regression(df: pd.DataFrame) -> dict:
    """
    Multi-factor OLS: strategy ~ alpha + Σ(beta_k × factor_k) + ε

    Factors: SP500, VIX daily change, HY OAS daily change,
             10y yield daily change, 12-month momentum proxy.

    Returns dict with per-factor betas, full-period alpha, R², n_obs,
    and a factor_contribution table showing each factor's R² contribution.
    All NaN when insufficient data.
    """
    if _STRATEGY_COL not in df.columns or df.empty:
        return {}

    df_aug = _build_factor_matrix(df)

    # Collect available factors
    avail = [(col, name) for col, name in _MULTI_FACTORS if col in df_aug.columns]
    if not avail:
        return {}

    factor_cols = [c for c, _ in avail]
    factor_names = [n for _, n in avail]

    sub = df_aug[[_STRATEGY_COL] + factor_cols].dropna()
    n = len(sub)
    if n < _MIN_OBS + len(factor_cols):
        return {"error": f"Insufficient observations ({n}) for {len(factor_cols)}-factor model"}

    y = sub[_STRATEGY_COL].values.astype(float)
    X = sub[factor_cols].values.astype(float)
    X = np.column_stack([np.ones(n), X])   # add intercept

    betas, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_d = float(betas[0])
    factor_betas = betas[1:].tolist()

    y_hat = X @ betas
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else np.nan
    resid_std_d = float(np.sqrt(ss_res / max(n - len(betas), 1)))

    factor_table = []
    for name, beta in zip(factor_names, factor_betas):
        factor_table.append({
            "factor": name,
            "beta":   round(float(beta), 5),
        })

    return {
        "alpha_daily":     round(alpha_d, 6),
        "ann_alpha":       round(alpha_d * _ANNUALIZE, 5),
        "r2":              round(r2, 4),
        "residual_vol":    round(resid_std_d * np.sqrt(_ANNUALIZE), 5),
        "n_obs":           n,
        "n_factors":       len(factor_names),
        "factor_betas":    factor_table,
        "factor_names":    factor_names,
    }
