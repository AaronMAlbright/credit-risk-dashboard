"""
Custom composite score builder.

Allows the user to specify custom weights for the 7 sub-signals and compute
what the composite score would have been historically under those weights,
along with a backtest Sharpe ratio.

Sub-signals
-----------
  treasury_stress_score_smooth          Treasury Stress
  credit_market_risk_score_smooth       Credit Risk
  macro_risk_score_smooth               Macro Risk
  enhanced_funding_stress_score_smooth  Funding Stress
  rates_stress_score_smooth             Rates Stress
  fx_commodity_score_smooth             FX/Commodity
  banking_stress_score_smooth           Banking Stress

Strategy (backtest)
-------------------
  If composite > 50 → go short/underweight credit (expect widening).
  If composite ≤ 50 → go long/overweight credit (expect tightening).
  Uses HY spread change as proxy.  Spread tightening (negative change) is
  the desired outcome when long credit; widening is desired when short.

Public API
----------
  SUB_SIGNALS               — ordered list of (col, label, default_weight)
  DEFAULT_WEIGHTS           — dict col → 1/7
  compute_custom_composite  — weighted composite Series
  backtest_custom_composite — Sharpe / hit-rate backtest
  compare_to_default        — custom vs equal-weight comparison
  run_custom_composite_analysis — orchestrator
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

SUB_SIGNALS: list[tuple[str, str, float]] = [
    ("treasury_stress_score_smooth",          "Treasury Stress",  1 / 7),
    ("credit_market_risk_score_smooth",       "Credit Risk",      1 / 7),
    ("macro_risk_score_smooth",               "Macro Risk",       1 / 7),
    ("enhanced_funding_stress_score_smooth",  "Funding Stress",   1 / 7),
    ("rates_stress_score_smooth",             "Rates Stress",     1 / 7),
    ("fx_commodity_score_smooth",             "FX/Commodity",     1 / 7),
    ("banking_stress_score_smooth",           "Banking Stress",   1 / 7),
]

DEFAULT_WEIGHTS: dict[str, float] = {col: w for col, _, w in SUB_SIGNALS}

_MIN_ROWS: int = 252            # ~1 trading year
_MIN_SIGNALS: int = 3           # at least 3 of 7 sub-signals must be present
_SIGNAL_THRESHOLD: float = 50.0 # composite > 50 → elevated risk
_ANNUALIZE: int = 252
_DISPLAY_ROWS: int = 504        # ~2 trading years for UI display


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def _score_to_regime(score: float) -> str:
    """Map a 0-100 composite score to a regime label."""
    if score < 30:
        return "Risk-On"
    elif score < 50:
        return "Neutral"
    elif score < 65:
        return "Caution"
    return "Risk-Off"


# ---------------------------------------------------------------------------
# Core: weighted composite
# ---------------------------------------------------------------------------

def compute_custom_composite(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted composite score using given weights.

    Parameters
    ----------
    df      : master DataFrame with DatetimeIndex
    weights : dict mapping column_name -> weight (need not sum to 1; normalized)

    Behaviour
    ---------
    - Only columns that exist in df AND have a positive weight are used.
    - Weights are re-normalized so they sum to 1 over the available columns.
    - Rows where ALL contributing sub-signals are NaN remain NaN.
    - Result is clipped to [0, 100].

    Returns
    -------
    pd.Series aligned with df.index, name='custom_composite'
    """
    # Filter to columns present in df
    available = {col: w for col, w in weights.items() if col in df.columns and w > 0}

    if not available:
        return pd.Series(np.nan, index=df.index, name="custom_composite")

    total_w = sum(available.values())
    if total_w <= 0:
        return pd.Series(np.nan, index=df.index, name="custom_composite")

    # Normalized weights
    norm_w = {col: w / total_w for col, w in available.items()}

    # Weighted sum — NaN-safe: use where data is available per row
    cols = list(norm_w.keys())
    wts = np.array([norm_w[c] for c in cols], dtype=float)

    data = df[cols].astype(float).values  # shape (n, k)
    finite = np.isfinite(data)            # True where data exists

    # Rows where ALL signals are NaN → result NaN
    all_nan_rows = ~finite.any(axis=1)

    # For partial NaN rows: re-normalize weights on available signals
    weighted_sum = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        row_finite = finite[i]
        if not row_finite.any():
            continue
        row_wts = wts * row_finite.astype(float)
        row_total = row_wts.sum()
        if row_total <= 0:
            continue
        weighted_sum[i] = np.dot(data[i] * row_finite, row_wts) / row_total

    result = pd.Series(weighted_sum, index=df.index, name="custom_composite")
    result = result.clip(0, 100)
    result[all_nan_rows] = np.nan
    return result


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest_custom_composite(
    df: pd.DataFrame,
    custom_composite: pd.Series,
    target_col: str = "hy_spread",
) -> dict:
    """Compute backtest metrics for a custom composite as a spread predictor.

    Strategy
    --------
    Signal at day t:
      composite > SIGNAL_THRESHOLD  → short credit (-1): expect HY widening
      composite ≤ SIGNAL_THRESHOLD  → long credit  (+1): expect HY tightening

    Signal return for period [t, t+FORWARD_DAYS]:
      long  (+1): positive return when HY tightens (spread falls)
      short (-1): positive return when HY widens   (spread rises)
    Return = signal × (−Δspread_30d)  scaled to bps perspective

    We measure at non-overlapping 30-day windows to avoid autocorrelation.

    Returns
    -------
    dict with:
      sharpe, hit_rate, avg_signal_return, n_periods,
      total_return
    """
    _nan_result = {
        "sharpe":             float("nan"),
        "hit_rate":           float("nan"),
        "avg_signal_return":  float("nan"),
        "n_periods":          0,
        "total_return":       float("nan"),
    }

    try:
        if target_col not in df.columns:
            return _nan_result

        spread = df[target_col].astype(float)
        composite = custom_composite.reindex(df.index)

        STEP = 30  # non-overlapping windows

        signals: list[float] = []
        returns: list[float] = []

        n = len(df)
        for i in range(0, n - STEP, STEP):
            comp_val = composite.iloc[i]
            spr_start = spread.iloc[i]
            spr_end = spread.iloc[i + STEP]

            if not (np.isfinite(comp_val) and np.isfinite(spr_start) and np.isfinite(spr_end)):
                continue

            # Signal: +1 (long credit) if composite ≤ threshold, -1 (short) otherwise
            signal = -1.0 if comp_val > _SIGNAL_THRESHOLD else 1.0
            d_spread = spr_end - spr_start            # positive = widening

            # Return: long credit gains when spread tightens (d_spread < 0)
            period_return = signal * (-d_spread)

            signals.append(signal)
            returns.append(period_return)

        n_periods = len(returns)
        if n_periods < 2:
            return _nan_result

        ret_arr = np.array(returns, dtype=float)
        sig_arr = np.array(signals, dtype=float)

        mean_ret = float(ret_arr.mean())
        std_ret = float(ret_arr.std(ddof=1))

        # Annualize: each period is 30 days; ~252/30 periods per year
        periods_per_year = _ANNUALIZE / STEP
        sharpe = (mean_ret * periods_per_year) / (std_ret * np.sqrt(periods_per_year)) if std_ret > 1e-9 else float("nan")

        # Hit rate: % of periods where return > 0
        hit_rate = float((ret_arr > 0).mean())

        # Avg signal return when composite > threshold (short credit positions)
        short_mask = sig_arr < 0
        avg_signal_return = float(ret_arr[short_mask].mean()) if short_mask.any() else float("nan")

        # Total return: cumulative (treat each period return as fractional — divide by
        # a normalizing constant in bps to keep in sensible range)
        # Raw: sum of spread-change-based returns in bps
        total_return = float(ret_arr.sum())

        return {
            "sharpe":            round(sharpe, 3) if np.isfinite(sharpe) else float("nan"),
            "hit_rate":          round(hit_rate, 4),
            "avg_signal_return": round(avg_signal_return, 3) if np.isfinite(avg_signal_return) else float("nan"),
            "n_periods":         n_periods,
            "total_return":      round(total_return, 2),
        }

    except Exception:
        return _nan_result


# ---------------------------------------------------------------------------
# Compare custom vs default
# ---------------------------------------------------------------------------

def compare_to_default(
    df: pd.DataFrame,
    weights: dict,
) -> dict:
    """Compare custom-weights composite vs default equal-weight composite.

    Parameters
    ----------
    df      : master DataFrame with DatetimeIndex
    weights : custom weights dict (col → weight)

    Returns
    -------
    dict with:
      custom_composite  : pd.Series (last DISPLAY_ROWS rows)
      default_composite : pd.Series (last DISPLAY_ROWS rows)
      custom_backtest   : dict from backtest_custom_composite
      default_backtest  : dict from backtest_custom_composite
      custom_current    : float (latest score)
      default_current   : float (latest score)
      custom_regime     : str
      default_regime    : str
      correlation       : float (full-history Pearson correlation)
      sharpe_improvement: float (custom sharpe - default sharpe)
    """
    try:
        custom_full = compute_custom_composite(df, weights)
        default_full = compute_custom_composite(df, DEFAULT_WEIGHTS)

        # Backtest on full history
        custom_bt = backtest_custom_composite(df, custom_full)
        default_bt = backtest_custom_composite(df, default_full)

        # Latest values
        custom_current = float(custom_full.dropna().iloc[-1]) if custom_full.notna().any() else float("nan")
        default_current = float(default_full.dropna().iloc[-1]) if default_full.notna().any() else float("nan")

        # Regime labels
        custom_regime = _score_to_regime(custom_current) if np.isfinite(custom_current) else "Unknown"
        default_regime = _score_to_regime(default_current) if np.isfinite(default_current) else "Unknown"

        # Full-history correlation
        paired = pd.DataFrame({"c": custom_full, "d": default_full}).dropna()
        correlation = float(paired["c"].corr(paired["d"])) if len(paired) >= 2 else float("nan")

        # Sharpe improvement
        c_sh = custom_bt.get("sharpe", float("nan"))
        d_sh = default_bt.get("sharpe", float("nan"))
        sharpe_improvement = float(c_sh - d_sh) if (np.isfinite(c_sh) and np.isfinite(d_sh)) else float("nan")

        # Trim to display window
        display_custom = custom_full.iloc[-_DISPLAY_ROWS:] if len(custom_full) > _DISPLAY_ROWS else custom_full
        display_default = default_full.iloc[-_DISPLAY_ROWS:] if len(default_full) > _DISPLAY_ROWS else default_full

        return {
            "custom_composite":   display_custom,
            "default_composite":  display_default,
            "custom_backtest":    custom_bt,
            "default_backtest":   default_bt,
            "custom_current":     round(custom_current, 2) if np.isfinite(custom_current) else float("nan"),
            "default_current":    round(default_current, 2) if np.isfinite(default_current) else float("nan"),
            "custom_regime":      custom_regime,
            "default_regime":     default_regime,
            "correlation":        round(correlation, 4) if np.isfinite(correlation) else float("nan"),
            "sharpe_improvement": round(sharpe_improvement, 3) if np.isfinite(sharpe_improvement) else float("nan"),
        }

    except Exception:
        return {
            "custom_composite":   pd.Series(dtype=float),
            "default_composite":  pd.Series(dtype=float),
            "custom_backtest":    {},
            "default_backtest":   {},
            "custom_current":     float("nan"),
            "default_current":    float("nan"),
            "custom_regime":      "Unknown",
            "default_regime":     "Unknown",
            "correlation":        float("nan"),
            "sharpe_improvement": float("nan"),
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_custom_composite_analysis(
    df: pd.DataFrame,
    weights: dict | None = None,
) -> dict:
    """Full custom composite analysis.

    Parameters
    ----------
    df      : master DataFrame with DatetimeIndex
    weights : custom weights dict (col → weight).  None → use DEFAULT_WEIGHTS.

    Minimum data requirements
    -------------------------
    - At least _MIN_SIGNALS (3) of the 7 sub-signals must be present
    - At least _MIN_ROWS (252) rows

    Returns
    -------
    dict with:
      available     : bool
      comparison    : dict from compare_to_default
      signal_labels : list[dict]  (col, label, weight, default_weight)
      weights_used  : dict
    """
    try:
        if df is None or df.empty:
            return {"available": False}

        if len(df) < _MIN_ROWS:
            return {"available": False}

        # Count how many sub-signals are present
        present_signals = [col for col, _, _ in SUB_SIGNALS if col in df.columns]
        if len(present_signals) < _MIN_SIGNALS:
            return {"available": False}

        # Resolve weights
        if weights is None:
            weights_used = dict(DEFAULT_WEIGHTS)
        else:
            # Accept only known sub-signal columns; fall back to zero for unknowns
            weights_used = {col: float(weights.get(col, 0.0)) for col, _, _ in SUB_SIGNALS}

        # Run comparison
        comparison = compare_to_default(df, weights_used)

        # Build signal_labels for UI display
        total_custom = sum(weights_used.get(col, 0.0) for col in weights_used)
        signal_labels = []
        for col, label, default_w in SUB_SIGNALS:
            raw_w = weights_used.get(col, 0.0)
            # Normalize for display (same logic as compute_custom_composite)
            available_w = {
                c: w for c, _, _ in SUB_SIGNALS
                if c in df.columns and weights_used.get(c, 0.0) > 0
            }
            avail_total = sum(available_w.values()) if available_w else 1.0
            effective_w = raw_w / avail_total if avail_total > 0 else 0.0

            signal_labels.append({
                "col":           col,
                "label":         label,
                "weight":        round(effective_w, 4),
                "default_weight": round(default_w, 4),
                "present":       col in df.columns,
            })

        return {
            "available":     True,
            "comparison":    comparison,
            "signal_labels": signal_labels,
            "weights_used":  weights_used,
        }

    except Exception:
        return {"available": False}
