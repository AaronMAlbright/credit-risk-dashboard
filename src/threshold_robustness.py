"""
Threshold robustness analysis for the 4-regime composite-score boundaries.

Stress-tests the three regime boundaries:
  risk_on  = 20  (below → Risk-On)
  neutral  = 32  (below → Neutral)
  caution  = 45  (below → Caution, above → Risk-Off)

For each boundary shifted by ±1, ±2, ±3, ±5 points:
  - Reclassifies all historical rows
  - Computes strategy performance metrics (IS + OOS)
  - Measures regime distribution, persistence, turnover
  - Reports separability (eta-squared vs SP500 forward returns)

No scipy needed — all statistics use pure numpy.

Public API
----------
  SHIFT_GRID                 — default shifts to test
  DEFAULT_THRESHOLDS         — current production thresholds
  reclassify_with_thresholds — apply custom thresholds to df
  compute_regime_metrics     — performance/regime stats for a classified df
  run_threshold_sweep        — sweep one boundary across shift grid
  run_threshold_robustness   — orchestrator (all three boundaries)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, float] = {
    "risk_on": 20.0,
    "neutral": 32.0,
    "caution": 45.0,
}

SHIFT_GRID: list[int] = [-5, -3, -2, -1, 0, 1, 2, 3, 5]

_SCORE_COL    = "composite_risk_score_smooth"
_DATE_COL     = "date"
_STRAT_COL    = "strategy_daily_return"
_SP500_COL    = "sp500_daily_return"
_FWD_COL      = "sp500_forward_30d_return"
_ANNUALIZE    = 252
_MIN_OBS      = 20

# Frozen IS/OOS split dates (mirrors walk_forward.FROZEN_SPLITS)
_OOS_START    = "2016-01-01"   # primary OOS window


# ---------------------------------------------------------------------------
# Reclassification
# ---------------------------------------------------------------------------

def reclassify_with_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    shock_flag_col: str = "shock_flag",
    max_stress_cols: tuple[str, ...] = (
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
    ),
) -> pd.Series:
    """
    Classify every row using custom thresholds.

    Mirrors classify_regime_band from risk_engine.py but operates
    vectorised on the full DataFrame.

    Returns Series of regime labels indexed like df.
    """
    if _SCORE_COL not in df.columns:
        return pd.Series(["Neutral"] * len(df), index=df.index)

    score  = df[_SCORE_COL].values
    t_ro   = thresholds.get("risk_on",  DEFAULT_THRESHOLDS["risk_on"])
    t_neu  = thresholds.get("neutral",  DEFAULT_THRESHOLDS["neutral"])
    t_cau  = thresholds.get("caution",  DEFAULT_THRESHOLDS["caution"])

    labels = np.where(score >= t_cau, "Risk-Off",
             np.where(score >= t_neu, "Caution",
             np.where(score >= t_ro,  "Neutral", "Risk-On")))

    # Shock override
    if shock_flag_col in df.columns:
        max_stress = np.zeros(len(df))
        for col in max_stress_cols:
            if col in df.columns:
                max_stress = np.maximum(max_stress, df[col].fillna(0).values)
        shock_mask = (
            (df[shock_flag_col].fillna("No Shock") != "No Shock")
            & (max_stress >= 40)
        ).values
        labels[shock_mask] = "Risk-Off"

    return pd.Series(labels, index=df.index)


# ---------------------------------------------------------------------------
# Metrics for a classified df
# ---------------------------------------------------------------------------

def compute_regime_metrics(
    df: pd.DataFrame,
    labels: pd.Series,
    oos_start: str = _OOS_START,
) -> dict:
    """
    Compute key metrics for a given classification.

    Returns dict:
      n_total, regime_counts, persistence, turnover_rate,
      is_sharpe, is_maxdd, is_hit_rate,
      oos_sharpe, oos_maxdd, oos_hit_rate,
      separability  (eta-squared × 100)
    """
    n = len(df)
    if n < _MIN_OBS or _STRAT_COL not in df.columns:
        return {}

    strat = df[_STRAT_COL].values.astype(float)

    # Regime distribution
    regime_counts = pd.Series(labels.values).value_counts().to_dict()

    # Persistence: mean run length
    runs, current_len = [], 1
    for i in range(1, n):
        if labels.iloc[i] == labels.iloc[i - 1]:
            current_len += 1
        else:
            runs.append(current_len)
            current_len = 1
    runs.append(current_len)
    persistence = float(np.mean(runs)) if runs else np.nan

    # Turnover: fraction of days with regime change
    changes     = (labels.values[1:] != labels.values[:-1]).sum()
    turnover_rate = float(changes / max(n - 1, 1))

    # IS/OOS split
    if _DATE_COL in df.columns:
        dates   = pd.to_datetime(df[_DATE_COL].values)
        oos_mask = dates >= pd.Timestamp(oos_start)
        is_mask  = ~oos_mask
    else:
        is_mask  = np.ones(n, dtype=bool)
        oos_mask = np.zeros(n, dtype=bool)

    def _perf(mask):
        r = strat[mask]
        r = r[~np.isnan(r)]
        if len(r) < _MIN_OBS:
            return {"sharpe": np.nan, "maxdd": np.nan, "hit_rate": np.nan}
        ann_ret = float(r.mean() * _ANNUALIZE)
        ann_vol = float(r.std(ddof=1) * np.sqrt(_ANNUALIZE))
        sharpe  = ann_ret / ann_vol if ann_vol > 1e-9 else np.nan
        cum     = (1 + r).cumprod()
        peak    = np.maximum.accumulate(cum)
        maxdd   = float((cum / peak - 1).min())
        hit_rt  = float((r > 0).mean())
        return {"sharpe": sharpe, "maxdd": maxdd, "hit_rate": hit_rt}

    is_p  = _perf(is_mask)
    oos_p = _perf(oos_mask)

    # Separability: eta-squared on forward 30d returns (if available)
    separability = np.nan
    if _FWD_COL in df.columns:
        fwd     = df[_FWD_COL].values.astype(float)
        lbl_arr = labels.values
        valid   = ~np.isnan(fwd)
        fwd_v   = fwd[valid]
        lbl_v   = lbl_arr[valid]
        if len(fwd_v) >= _MIN_OBS:
            grand_mean = fwd_v.mean()
            ss_between = sum(
                ((fwd_v[lbl_v == r]).mean() - grand_mean) ** 2 * (lbl_v == r).sum()
                for r in np.unique(lbl_v)
                if (lbl_v == r).sum() >= 5
            )
            ss_total = ((fwd_v - grand_mean) ** 2).sum()
            if ss_total > 1e-14:
                separability = round(float(ss_between / ss_total * 100), 3)

    return {
        "n_total":       n,
        "regime_counts": regime_counts,
        "persistence":   round(persistence, 2) if not np.isnan(persistence) else None,
        "turnover_rate": round(turnover_rate, 4),
        "is_sharpe":     round(is_p["sharpe"],   4) if not np.isnan(is_p["sharpe"])   else None,
        "is_maxdd":      round(is_p["maxdd"],     4) if not np.isnan(is_p["maxdd"])    else None,
        "is_hit_rate":   round(is_p["hit_rate"],  4) if not np.isnan(is_p["hit_rate"]) else None,
        "oos_sharpe":    round(oos_p["sharpe"],   4) if not np.isnan(oos_p["sharpe"])  else None,
        "oos_maxdd":     round(oos_p["maxdd"],    4) if not np.isnan(oos_p["maxdd"])   else None,
        "oos_hit_rate":  round(oos_p["hit_rate"], 4) if not np.isnan(oos_p["hit_rate"]) else None,
        "separability":  separability,
    }


# ---------------------------------------------------------------------------
# Sweep one boundary
# ---------------------------------------------------------------------------

def run_threshold_sweep(
    df: pd.DataFrame,
    boundary: str,
    shift_grid: list[int] = SHIFT_GRID,
    base_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Sweep a single boundary across shift_grid and collect metrics.

    Parameters
    ----------
    df         : scored DataFrame
    boundary   : one of 'risk_on', 'neutral', 'caution'
    shift_grid : list of integer shifts to apply
    base_thresholds : override DEFAULT_THRESHOLDS

    Returns DataFrame with one row per shift value.
    """
    base = {**DEFAULT_THRESHOLDS, **(base_thresholds or {})}
    rows = []

    for shift in shift_grid:
        thresh = {**base, boundary: base[boundary] + shift}
        # Guard: thresholds must remain monotonically ordered
        if thresh["risk_on"] >= thresh["neutral"] or thresh["neutral"] >= thresh["caution"]:
            continue

        labels  = reclassify_with_thresholds(df, thresh)
        metrics = compute_regime_metrics(df, labels)
        if not metrics:
            continue

        row = {
            "boundary":   boundary,
            "base_value": base[boundary],
            "shift":      shift,
            "threshold":  thresh[boundary],
            **{k: v for k, v in metrics.items() if k not in ("regime_counts",)},
        }
        # Flatten regime counts
        for regime in ("Risk-On", "Neutral", "Caution", "Risk-Off"):
            row[f"n_{regime.lower().replace('-', '_')}"] = \
                metrics.get("regime_counts", {}).get(regime, 0)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stability summary
# ---------------------------------------------------------------------------

def _robustness_score(sweep_df: pd.DataFrame, metric: str = "oos_sharpe") -> dict:
    """
    Summarise stability of a metric across shifts.

    Returns:
      mean, std, cv (coefficient of variation), min, max,
      stable (bool: cv < 0.25 and no sign flip)
    """
    vals = sweep_df[metric].dropna().values.astype(float)
    if len(vals) < 3:
        return {"mean": None, "std": None, "cv": None, "stable": False}
    mean = float(vals.mean())
    std  = float(vals.std(ddof=1))
    cv   = abs(std / mean) if abs(mean) > 1e-9 else None
    sign_flip = bool((vals > 0).any() and (vals < 0).any())
    stable    = (cv is not None and cv < 0.25 and not sign_flip)
    return {
        "mean":       round(mean, 4),
        "std":        round(std,  4),
        "cv":         round(cv,   4) if cv is not None else None,
        "min":        round(float(vals.min()), 4),
        "max":        round(float(vals.max()), 4),
        "sign_flip":  sign_flip,
        "stable":     stable,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_threshold_robustness(
    df: pd.DataFrame,
    boundaries: tuple[str, ...] = ("risk_on", "neutral", "caution"),
    shift_grid: list[int] = SHIFT_GRID,
) -> dict:
    """
    Full threshold robustness analysis.

    Returns dict:
      sweeps       — {boundary: DataFrame of metrics per shift}
      stability    — {boundary: stability summary dict}
      summary_df   — combined DataFrame of all sweeps (for heatmaps)
      baseline     — metrics at production thresholds
      warnings     — list of instability warnings
    """
    if df.empty or _SCORE_COL not in df.columns:
        return {}

    sweeps     = {}
    stability  = {}
    all_rows   = []

    for boundary in boundaries:
        sw = run_threshold_sweep(df, boundary, shift_grid)
        if sw.empty:
            continue
        sweeps[boundary]   = sw
        stability[boundary] = {
            "oos_sharpe":   _robustness_score(sw, "oos_sharpe"),
            "oos_maxdd":    _robustness_score(sw, "oos_maxdd"),
            "turnover_rate": _robustness_score(sw, "turnover_rate"),
            "separability": _robustness_score(sw, "separability"),
        }
        all_rows.append(sw)

    # Baseline (shift=0) metrics
    base_labels  = reclassify_with_thresholds(df, DEFAULT_THRESHOLDS)
    baseline     = compute_regime_metrics(df, base_labels)

    # Warnings
    warnings: list[str] = []
    for boundary, stab in stability.items():
        sh_stab = stab.get("oos_sharpe", {})
        if not sh_stab.get("stable", True):
            cv_str = f"CV={sh_stab['cv']:.2f}" if sh_stab.get("cv") else ""
            warnings.append(
                f"⚠ '{boundary}' threshold: OOS Sharpe is sensitive to shifts "
                f"({cv_str}, sign_flip={sh_stab.get('sign_flip')})"
            )
        if stab.get("turnover_rate", {}).get("cv", 0) or 0 > 0.40:
            warnings.append(
                f"⚠ '{boundary}' threshold: turnover is highly variable across shifts"
            )

    return {
        "sweeps":     sweeps,
        "stability":  stability,
        "summary_df": pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(),
        "baseline":   baseline,
        "warnings":   warnings,
    }
