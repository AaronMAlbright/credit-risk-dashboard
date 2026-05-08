"""
Position sizing module.

Converts composite score, regime probabilities, rolling performance, and
realised volatility into four complementary sizing recommendations:

  1. Score-based    — piecewise-linear map from composite score to weight.
                      Thresholds are calibrated to the observed score range
                      (21–43) so that sizing varies meaningfully within the data.

  2. Regime-prob    — probability-weighted average of per-regime target weights
                      (Σ P(regime_i) × w_i), using the Gaussian Naive Bayes
                      posterior from src.regime_probability.

  3. Half-Kelly     — rolling half-Kelly fraction: 0.5 × (μ / σ²) where μ and σ
                      are annualised strategy stats over a 63-day window.
                      Naturally reduces allocation after drawdown periods.

  4. Vol-target     — weight = target_vol / rolling_21d_vol (clipped [0,1]).
                      Targets a fixed annualised portfolio volatility (default 10%).

A fifth column 'blend' averages all four methods.

All weights are clipped to [_MIN_WEIGHT, _MAX_WEIGHT].

The module also backtests each sizing method applied to the strategy returns
(lagged one day to avoid look-ahead) and compares cumulative returns vs.
full allocation and SP500.

Public API
----------
  REGIME_WEIGHTS           — default per-regime target weights (overridable)
  SCORE_BREAKPOINTS        — default score→weight breakpoints  (overridable)
  compute_score_sizing     — score-based sizing series
  compute_regime_prob_sizing — probability-weighted regime sizing
  compute_kelly_sizing     — rolling half-Kelly sizing
  compute_vol_target_sizing — volatility-targeting sizing
  compute_sized_backtest   — apply sizing to returns, return backtest DataFrame
  get_current_sizing       — current weights from all methods
  run_position_sizing      — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants — all can be overridden by callers
# ---------------------------------------------------------------------------

_STRAT_COL   = "strategy_daily_return"
_SP500_COL   = "sp500_daily_return"
_SCORE_COL   = "composite_risk_score_smooth"
_LABEL_COL   = "final_decision"
_DATE_COL    = "date"

_MIN_WEIGHT   = 0.00
_MAX_WEIGHT   = 1.00
_KELLY_WINDOW = 63       # trading days (~3 months)
_VOL_WINDOW   = 21       # trading days (~1 month)
_TARGET_VOL   = 0.10     # annualised; used by vol-target method
_ANNUALIZE    = 252
_MIN_OBS      = 20       # minimum obs for rolling estimates

# Piecewise-linear score → weight map.
# Calibrated to the empirically observed composite score range (21–43).
SCORE_BREAKPOINTS: list[tuple[float, float]] = [
    (0.0,   1.00),   # theoretical floor
    (28.0,  1.00),   # ~p20: full allocation
    (33.0,  0.75),   # ~p50: moderate reduction
    (37.0,  0.45),   # ~p80: significant reduction
    (41.0,  0.20),   # ~p95: substantial reduction
    (100.0, 0.05),   # theoretical ceiling
]

# Default per-regime target equity weights (4-regime framework).
# Risk-On = fully deploy; Risk-Off = maximum defensive.
# Override to reflect your own risk preferences.
REGIME_WEIGHTS: dict[str, float] = {
    "Risk-On":  1.00,
    "Neutral":  0.75,
    "Caution":  0.40,
    "Risk-Off": 0.10,
}


# ---------------------------------------------------------------------------
# Score-based sizing
# ---------------------------------------------------------------------------

def _interp_weight(score: float,
                   breakpoints: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation from score to weight."""
    xs = [p[0] for p in breakpoints]
    ys = [p[1] for p in breakpoints]
    return float(np.clip(np.interp(score, xs, ys), _MIN_WEIGHT, _MAX_WEIGHT))


def compute_score_sizing(
    df: pd.DataFrame,
    breakpoints: list[tuple[float, float]] | None = None,
) -> pd.Series:
    """
    Piecewise-linear map from composite_risk_score_smooth to position weight.

    Returns a Series indexed like df (date if available), values in [0, 1].
    """
    if _SCORE_COL not in df.columns or df.empty:
        return pd.Series(dtype=float)

    bp = breakpoints or SCORE_BREAKPOINTS
    weights = df[_SCORE_COL].apply(lambda s: _interp_weight(float(s), bp))

    if _DATE_COL in df.columns:
        weights.index = pd.to_datetime(df[_DATE_COL].values)

    return weights.rename("score_sizing")


# ---------------------------------------------------------------------------
# Regime-probability sizing
# ---------------------------------------------------------------------------

def compute_regime_prob_sizing(
    df: pd.DataFrame,
    regime_weights: dict[str, float] | None = None,
    prob_history: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Probability-weighted regime sizing: Σ P(regime_i | scores_t) × w_i.

    If prob_history is None, fits the Gaussian Naive Bayes model and computes
    the full history internally (adds ~0.3s for 500-row datasets).

    Returns a Series indexed by date (if available), values in [0, 1].
    """
    rw = regime_weights or REGIME_WEIGHTS

    if prob_history is None:
        try:
            from src.regime_probability import compute_prob_history
            prob_history = compute_prob_history(df)
        except Exception:
            return pd.Series(dtype=float)

    if prob_history.empty:
        return pd.Series(dtype=float)

    regime_cols = [c for c in prob_history.columns
                   if c not in ("top_regime", "entropy")]

    weights = pd.Series(0.0, index=prob_history.index)
    for col in regime_cols:
        w = rw.get(col, 0.50)          # default 0.5 for unrecognised regimes
        weights += prob_history[col] * w

    return weights.clip(_MIN_WEIGHT, _MAX_WEIGHT).rename("regime_prob_sizing")


# ---------------------------------------------------------------------------
# Half-Kelly sizing
# ---------------------------------------------------------------------------

def compute_kelly_sizing(
    df: pd.DataFrame,
    window: int = _KELLY_WINDOW,
) -> pd.Series:
    """
    Rolling half-Kelly fraction: f = 0.5 × (ann_μ / ann_σ²).

    Uses strategy daily returns. Clipped to [0, 1].
    NaN for the first (window-1) rows and whenever ann_σ² is near zero.
    """
    if _STRAT_COL not in df.columns or df.empty:
        return pd.Series(dtype=float)

    r = df[_STRAT_COL].values.astype(float)
    n = len(r)
    kelly = np.full(n, np.nan)

    for end in range(window - 1, n):
        start = end - window + 1
        sl = r[start: end + 1]
        sl = sl[~np.isnan(sl)]
        if len(sl) < _MIN_OBS:
            continue
        ann_mu  = float(sl.mean() * _ANNUALIZE)
        ann_var = float((sl.std(ddof=1) ** 2) * _ANNUALIZE)
        if ann_var < 1e-8:
            continue
        kelly[end] = np.clip(0.5 * ann_mu / ann_var, _MIN_WEIGHT, _MAX_WEIGHT)

    out = pd.Series(kelly, name="kelly_sizing")
    if _DATE_COL in df.columns:
        out.index = pd.to_datetime(df[_DATE_COL].values)
    return out


# ---------------------------------------------------------------------------
# Volatility-target sizing
# ---------------------------------------------------------------------------

def compute_vol_target_sizing(
    df: pd.DataFrame,
    target_vol: float = _TARGET_VOL,
    window: int = _VOL_WINDOW,
) -> pd.Series:
    """
    Volatility-targeting: weight = target_vol / rolling_ann_vol.
    Uses strategy daily returns. Clipped to [0, 1].
    NaN for the first (window-1) rows.
    """
    if _STRAT_COL not in df.columns or df.empty:
        return pd.Series(dtype=float)

    r = df[_STRAT_COL].values.astype(float)
    n = len(r)
    vt = np.full(n, np.nan)

    for end in range(window - 1, n):
        start = end - window + 1
        sl = r[start: end + 1]
        sl = sl[~np.isnan(sl)]
        if len(sl) < 5:
            continue
        ann_vol = float(sl.std(ddof=1) * np.sqrt(_ANNUALIZE))
        if ann_vol < 1e-6:
            continue
        vt[end] = np.clip(target_vol / ann_vol, _MIN_WEIGHT, _MAX_WEIGHT)

    out = pd.Series(vt, name="vol_target_sizing")
    if _DATE_COL in df.columns:
        out.index = pd.to_datetime(df[_DATE_COL].values)
    return out


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def compute_sized_backtest(
    df: pd.DataFrame,
    sizes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply each sizing column to strategy returns (lagged 1 day).

    sized_return_t = size_{t-1} × strategy_return_t

    Returns DataFrame with columns:
      date, strategy_return, sp500_return,
      <method>_return  for each sizing column,
      cum_full, cum_sp500, cum_<method>
    All cumulative columns rebased to 1.0 at start.
    NaN sizing rows use full allocation (1.0) as fallback.
    """
    if _STRAT_COL not in df.columns or sizes.empty:
        return pd.DataFrame()

    strat = df[_STRAT_COL].values.astype(float)
    sp500 = df[_SP500_COL].values.astype(float) if _SP500_COL in df.columns else None

    out = pd.DataFrame()
    if _DATE_COL in df.columns:
        out["date"] = df[_DATE_COL].values
    out["strategy_return"] = strat
    if sp500 is not None:
        out["sp500_return"] = sp500

    sizing_cols = [c for c in sizes.columns if c != _DATE_COL]
    for col in sizing_cols:
        # Align sizing series to df index; fill NaN with 1.0 (full allocation)
        s = sizes[col].reindex(df.index).fillna(1.0).values.astype(float)
        lagged = np.concatenate([[1.0], s[:-1]])   # lag by 1
        out[f"{col}_return"] = lagged * strat

    # Cumulative returns
    out["cum_full"] = (1 + pd.Series(strat).fillna(0)).cumprod().values
    if sp500 is not None:
        out["cum_sp500"] = (1 + pd.Series(sp500).fillna(0)).cumprod().values
    for col in sizing_cols:
        r_col = f"{col}_return"
        if r_col in out.columns:
            out[f"cum_{col}"] = (1 + out[r_col].fillna(0)).cumprod().values

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Current sizing summary
# ---------------------------------------------------------------------------

def get_current_sizing(
    df: pd.DataFrame,
    sizes: pd.DataFrame | None = None,
) -> dict:
    """
    Current position weights from all sizing methods.

    Returns dict:
      score_sizing, regime_prob_sizing, kelly_sizing, vol_target_sizing,
      blend  — mean of non-NaN methods
      current_composite — latest composite score
      current_regime    — latest regime label
    """
    if sizes is None or sizes.empty:
        return {}

    # Use last non-NaN value for each method
    sizing_cols = [c for c in sizes.columns if c not in (_DATE_COL,)]
    current: dict = {}
    vals = []
    for col in sizing_cols:
        series = sizes[col].dropna()
        if len(series) > 0:
            v = float(series.iloc[-1])
            current[col] = round(v, 4)
            vals.append(v)

    if vals:
        current["blend"] = round(float(np.mean(vals)), 4)

    if _SCORE_COL in df.columns:
        current["current_composite"] = float(df[_SCORE_COL].iloc[-1])
    if _LABEL_COL in df.columns:
        current["current_regime"] = str(df[_LABEL_COL].iloc[-1])

    return current


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_position_sizing(
    df: pd.DataFrame,
    prob_history: pd.DataFrame | None = None,
    breakpoints: list[tuple[float, float]] | None = None,
    regime_weights: dict[str, float] | None = None,
    target_vol: float = _TARGET_VOL,
    kelly_window: int = _KELLY_WINDOW,
    vol_window: int = _VOL_WINDOW,
) -> dict:
    """
    Full position sizing analysis.

    Parameters
    ----------
    df             : scored DataFrame
    prob_history   : pre-computed regime probability history (optional;
                     if None, computes internally from regime_probability)
    breakpoints    : score→weight breakpoints (default SCORE_BREAKPOINTS)
    regime_weights : per-regime target weights (default REGIME_WEIGHTS)
    target_vol     : annualised vol target for vol-targeting method
    kelly_window   : rolling window for Kelly estimation (trading days)
    vol_window     : rolling window for vol estimation (trading days)

    Returns dict:
      sizes          — DataFrame(date index, columns = 4 methods + blend)
      current        — dict of current weights per method
      backtest       — sized backtest DataFrame
      breakpoints    — breakpoints used
      regime_weights — regime weights used
      target_vol     — vol target used
    """
    if df.empty or _DATE_COL not in df.columns:
        return {}

    bp = breakpoints or SCORE_BREAKPOINTS
    rw = regime_weights or REGIME_WEIGHTS

    score_s  = compute_score_sizing(df, bp)
    regprob_s = compute_regime_prob_sizing(df, rw, prob_history)
    kelly_s  = compute_kelly_sizing(df, kelly_window)
    volt_s   = compute_vol_target_sizing(df, target_vol, vol_window)

    # Align all series to a common date index
    date_idx = pd.to_datetime(df[_DATE_COL].values)
    sizes = pd.DataFrame(index=date_idx)
    for s, name in [
        (score_s,   "score_sizing"),
        (regprob_s, "regime_prob_sizing"),
        (kelly_s,   "kelly_sizing"),
        (volt_s,    "vol_target_sizing"),
    ]:
        if s is not None and not s.empty:
            sizes[name] = s.values if len(s) == len(df) else s.reindex(date_idx)

    # Blend: mean of non-NaN methods per row
    if not sizes.empty:
        sizes["blend"] = sizes.mean(axis=1, skipna=True)

    current = get_current_sizing(df, sizes)

    # Re-index df to match for backtest
    bt_df = df.reset_index(drop=True)
    sizes_reset = sizes.reset_index(drop=True)
    sizes_reset.columns = sizes.columns
    backtest = compute_sized_backtest(bt_df, sizes_reset)

    return {
        "sizes":          sizes,
        "current":        current,
        "backtest":       backtest,
        "breakpoints":    bp,
        "regime_weights": rw,
        "target_vol":     target_vol,
    }
