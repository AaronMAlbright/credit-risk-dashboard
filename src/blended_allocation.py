"""
Probability-blended portfolio allocation.

Replaces the hard step-function regime→weight mapping with a smooth,
probability-weighted allocation that uses GNB posterior probabilities.

Pipeline per row:
  1. raw_weight  = Σ P(regime_i) × equity_i     (probability-weighted blend)
  2. smoothed    = EMA(raw_weight, halflife=SMOOTH_HALFLIFE days)
  3. vol_adj     = min(TARGET_VOL / rolling_21d_ann_vol, 1.0)
  4. final       = clamp(smoothed * vol_adj, MIN_EQUITY, MAX_EQUITY)

Turnover and transaction-cost-adjusted performance are tracked.

Public API
----------
  PER_REGIME_EQUITY        — default equity fraction per regime (overridable)
  SMOOTH_HALFLIFE          — EMA half-life in trading days
  TARGET_VOL               — annualised vol target for vol overlay
  MIN_EQUITY / MAX_EQUITY  — allocation floor/ceiling
  ONE_WAY_COST             — one-way transaction cost assumption
  blend_allocation         — compute raw blended weight from prob dict
  compute_blended_series   — full history DataFrame
  run_blended_allocation   — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PER_REGIME_EQUITY: dict[str, float] = {
    "Risk-On":  0.85,
    "Neutral":  0.60,
    "Caution":  0.35,
    "Risk-Off": 0.10,
}

SMOOTH_HALFLIFE: int   = 5      # trading days (~1 week)
TARGET_VOL:      float = 0.10   # 10% annualised
MIN_EQUITY:      float = 0.05
MAX_EQUITY:      float = 0.95
ONE_WAY_COST:    float = 0.001  # 10 bps

_ANNUALIZE   = 252
_VOL_WINDOW  = 21    # days for rolling vol
_MIN_OBS     = 10
_DATE_COL    = "date"
_STRAT_COL   = "strategy_daily_return"
_SP500_COL   = "sp500_daily_return"
_LABEL_COL   = "final_decision"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def blend_allocation(
    probs: dict[str, float],
    per_regime_equity: dict[str, float] | None = None,
) -> float:
    """
    Compute a probability-weighted raw equity allocation.

    Parameters
    ----------
    probs : {regime: probability} — must sum to ~1.0
    per_regime_equity : override PER_REGIME_EQUITY

    Returns float in [0, 1].  Returns 0.5 on empty/invalid input.
    """
    rw = per_regime_equity or PER_REGIME_EQUITY
    total_p = sum(probs.values())
    if total_p < 1e-9 or not probs:
        return 0.5

    w = sum(probs.get(r, 0.0) * rw.get(r, 0.5) for r in probs)
    return float(np.clip(w / total_p, 0.0, 1.0))


def _ema_weights(n: int, halflife: int) -> np.ndarray:
    """Unnormalised EMA weights (most-recent last)."""
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    idx   = np.arange(n)
    w     = (1 - alpha) ** (n - 1 - idx)
    return w / w.sum()


def _apply_ema(series: np.ndarray, halflife: int) -> np.ndarray:
    """Apply exponential moving average with the given half-life."""
    alpha  = 1.0 - np.exp(-np.log(2) / halflife)
    result = np.empty_like(series, dtype=float)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_ann_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling annualised volatility. NaN for the first (window-1) rows."""
    n  = len(returns)
    rv = np.full(n, np.nan)
    for end in range(window - 1, n):
        sl = returns[end - window + 1: end + 1]
        sl = sl[~np.isnan(sl)]
        if len(sl) >= _MIN_OBS:
            rv[end] = float(sl.std(ddof=1) * np.sqrt(_ANNUALIZE))
    return rv


# ---------------------------------------------------------------------------
# Full-history blended series
# ---------------------------------------------------------------------------

def compute_blended_series(
    df: pd.DataFrame,
    prob_history: pd.DataFrame,
    per_regime_equity: dict[str, float] | None = None,
    smooth_halflife: int = SMOOTH_HALFLIFE,
    target_vol: float = TARGET_VOL,
    min_equity: float = MIN_EQUITY,
    max_equity: float = MAX_EQUITY,
    apply_vol_overlay: bool = True,
) -> pd.DataFrame:
    """
    Compute the full blended allocation history.

    Parameters
    ----------
    df            : scored DataFrame with date + strategy_daily_return
    prob_history  : DataFrame of regime probabilities (output of compute_prob_history)
    per_regime_equity : override PER_REGIME_EQUITY
    smooth_halflife   : EMA half-life in trading days
    target_vol    : annualised vol target for vol-targeting overlay
    min_equity / max_equity : allocation floor/ceiling
    apply_vol_overlay : whether to apply vol-target scaling

    Returns DataFrame with columns:
      date, label_weight (hard step-function),
      raw_weight, smoothed_weight, vol_scalar, final_weight,
      turnover (absolute daily change in final_weight)
    """
    rw  = per_regime_equity or PER_REGIME_EQUITY

    # Align prob_history to df
    regime_cols = [c for c in prob_history.columns
                   if c not in ("top_regime", "entropy")]

    n = len(df)
    raw_weights = np.full(n, np.nan)

    # Reindex prob_history to match df index
    prob_arr = prob_history[regime_cols].values  # (m, k)
    n_prob   = len(prob_arr)
    use_n    = min(n, n_prob)

    for i in range(use_n):
        row_probs = {regime_cols[j]: float(prob_arr[i, j])
                     for j in range(len(regime_cols))
                     if not np.isnan(prob_arr[i, j])}
        raw_weights[i] = blend_allocation(row_probs, rw)

    # For rows without probability data, fall back to hard label weight
    if _LABEL_COL in df.columns:
        for i in range(n):
            if np.isnan(raw_weights[i]):
                label = df[_LABEL_COL].iloc[i]
                raw_weights[i] = rw.get(label, 0.5)

    # Hard step-function weight (for comparison)
    label_weights = np.full(n, np.nan)
    if _LABEL_COL in df.columns:
        for i in range(n):
            label = df[_LABEL_COL].iloc[i]
            label_weights[i] = rw.get(label, 0.5)

    # EMA smoothing
    valid_start = int(np.argmax(~np.isnan(raw_weights)))
    smoothed = raw_weights.copy()
    smoothed[valid_start:] = _apply_ema(raw_weights[valid_start:], smooth_halflife)

    # Vol-targeting scalar
    vol_scalar = np.ones(n)
    if apply_vol_overlay and _STRAT_COL in df.columns:
        r = df[_STRAT_COL].values.astype(float)
        rv = _rolling_ann_vol(r, _VOL_WINDOW)
        for i in range(n):
            if not np.isnan(rv[i]) and rv[i] > 1e-6:
                vol_scalar[i] = min(1.0, target_vol / rv[i])

    # Final weight
    final = np.clip(smoothed * vol_scalar, min_equity, max_equity)

    out = pd.DataFrame()
    if _DATE_COL in df.columns:
        out["date"] = df[_DATE_COL].values
    out["label_weight"]    = label_weights
    out["raw_weight"]      = raw_weights
    out["smoothed_weight"] = smoothed
    out["vol_scalar"]      = vol_scalar
    out["final_weight"]    = final
    out["turnover"]        = np.abs(np.concatenate([[0.0],
                                     np.diff(final)]))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Backtest with blended weights
# ---------------------------------------------------------------------------

def compute_blended_backtest(
    df: pd.DataFrame,
    blended: pd.DataFrame,
    one_way_cost: float = ONE_WAY_COST,
) -> pd.DataFrame:
    """
    Apply blended allocation weights to strategy returns (lagged 1 day).

    Returns DataFrame with:
      date, strategy_return, sp500_return,
      label_return       (hard step-function weight × sp500),
      blended_return     (blended weight × sp500),
      net_return         (blended − transaction costs),
      cum_sp500, cum_label, cum_blended, cum_net,
      ann_turnover       (single value: annualised mean daily turnover)
    """
    if _SP500_COL not in df.columns or blended.empty:
        return pd.DataFrame()

    strat  = df[_STRAT_COL].values.astype(float) if _STRAT_COL in df.columns else None
    sp500  = df[_SP500_COL].values.astype(float)
    n      = len(sp500)

    label_w   = blended["label_weight"].values
    final_w   = blended["final_weight"].values
    turnover  = blended["turnover"].values

    # Lag all weights by 1 day
    label_lag = np.concatenate([[label_w[0]], label_w[:-1]])
    final_lag = np.concatenate([[final_w[0]], final_w[:-1]])
    turn_lag  = np.concatenate([[0.0], turnover[:-1]])

    label_ret   = label_lag * sp500
    blended_ret = final_lag * sp500
    cost        = turn_lag  * one_way_cost
    net_ret     = blended_ret - cost

    out = pd.DataFrame()
    if _DATE_COL in df.columns:
        out["date"] = df[_DATE_COL].values
    if strat is not None:
        out["strategy_return"] = strat
    out["sp500_return"]   = sp500
    out["label_return"]   = label_ret
    out["blended_return"] = blended_ret
    out["net_return"]     = net_ret

    for col, src in [("cum_sp500",    sp500),
                     ("cum_label",    label_ret),
                     ("cum_blended",  blended_ret),
                     ("cum_net",      net_ret)]:
        out[col] = (1 + pd.Series(src).fillna(0)).cumprod().values

    ann_turnover = float(np.nanmean(turnover) * _ANNUALIZE)
    out["ann_turnover"] = ann_turnover

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Current allocation summary
# ---------------------------------------------------------------------------

def get_current_blended(
    df: pd.DataFrame,
    blended: pd.DataFrame,
) -> dict:
    """
    Latest blended allocation values and key diagnostics.

    Returns dict:
      raw_weight, smoothed_weight, vol_scalar, final_weight,
      label_weight, ann_turnover, alloc_stability (std of final_weight 63d)
    """
    if blended.empty:
        return {}

    last = blended.iloc[-1]
    recent_63 = blended["final_weight"].iloc[-63:].dropna()

    return {
        "raw_weight":      round(float(last.get("raw_weight",      0.5)), 4),
        "smoothed_weight": round(float(last.get("smoothed_weight", 0.5)), 4),
        "vol_scalar":      round(float(last.get("vol_scalar",      1.0)), 4),
        "final_weight":    round(float(last.get("final_weight",    0.5)), 4),
        "label_weight":    round(float(last.get("label_weight",    0.5)), 4),
        "ann_turnover":    round(float(blended["turnover"].mean() * _ANNUALIZE), 2),
        "alloc_stability": round(float(recent_63.std()), 4) if len(recent_63) > 5 else None,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_blended_allocation(
    df: pd.DataFrame,
    prob_history: pd.DataFrame | None = None,
    per_regime_equity: dict[str, float] | None = None,
    smooth_halflife: int = SMOOTH_HALFLIFE,
    target_vol: float = TARGET_VOL,
    apply_vol_overlay: bool = True,
    one_way_cost: float = ONE_WAY_COST,
) -> dict:
    """
    Full probability-blended allocation analysis.

    Parameters
    ----------
    df            : scored DataFrame
    prob_history  : pre-computed regime probability history (optional)
    per_regime_equity : equity fractions per regime (default PER_REGIME_EQUITY)
    smooth_halflife   : EMA half-life for smoothing
    target_vol    : annualised vol target
    apply_vol_overlay : apply vol-target scaling
    one_way_cost  : one-way transaction cost

    Returns dict:
      blended      — full allocation history DataFrame
      backtest     — blended backtest DataFrame
      current      — current allocation summary dict
      per_regime_equity — regime weights used
      smooth_halflife   — halflife used
      target_vol        — vol target used
    """
    if df.empty:
        return {}

    if prob_history is None or prob_history.empty:
        try:
            from src.regime_probability import compute_prob_history
            prob_history = compute_prob_history(df)
        except Exception:
            prob_history = pd.DataFrame()

    blended  = compute_blended_series(
        df, prob_history,
        per_regime_equity=per_regime_equity,
        smooth_halflife=smooth_halflife,
        target_vol=target_vol,
        apply_vol_overlay=apply_vol_overlay,
    )
    backtest = compute_blended_backtest(df, blended, one_way_cost=one_way_cost)
    current  = get_current_blended(df, blended)

    return {
        "blended":          blended,
        "backtest":         backtest,
        "current":          current,
        "per_regime_equity": per_regime_equity or PER_REGIME_EQUITY,
        "smooth_halflife":  smooth_halflife,
        "target_vol":       target_vol,
        "one_way_cost":     one_way_cost,
    }
