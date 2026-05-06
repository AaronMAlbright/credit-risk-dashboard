"""
Regime attribution engine.

Explains which sub-scores drove composite risk levels and regime transitions,
without modifying any production scoring logic.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

# Matches composite_engine.py exactly
COMPOSITE_WEIGHTS = OrderedDict([
    ("macro_risk",     0.25),
    ("credit_risk",    0.25),
    ("complacency",    0.20),
    ("liquidity",      0.15),
    ("treasury",       0.10),
    ("mean_reversion", 0.05),   # contribution = weight × (100 − score)
])

# Additional tracked score (not in composite, but informative for attribution)
SUPPLEMENTAL_SCORES = OrderedDict([
    ("cross_asset", "cross_asset_divergence_score_smooth"),
])

SCORE_COLS = OrderedDict([
    ("macro_risk",     "macro_risk_score_smooth"),
    ("credit_risk",    "credit_market_risk_score_smooth"),
    ("complacency",    "complacency_score_smooth"),
    ("liquidity",      "liquidity_regime_score_smooth"),
    ("treasury",       "treasury_stress_score_smooth"),
    ("mean_reversion", "mean_reversion_score_smooth"),
    ("cross_asset",    "cross_asset_divergence_score_smooth"),
])

DISPLAY_NAMES = {
    "macro_risk":     "Macro Risk",
    "credit_risk":    "Credit Risk",
    "complacency":    "Complacency",
    "liquidity":      "Liquidity",
    "treasury":       "Treasury Stress",
    "mean_reversion": "Mean Reversion",
    "cross_asset":    "Cross-Asset Divergence",
}

DELTA_LOOKBACK = 21      # trading days ≈ 1 month
ELEVATION_PERCENTILE = 75


def _present(df):
    """Return SCORE_COLS entries whose columns exist in df."""
    return OrderedDict((k, v) for k, v in SCORE_COLS.items() if v in df.columns)


def compute_rolling_contributions(df):
    """
    Decompose composite risk into each sub-score's weighted contribution
    using the smoothed scores.

    contribution_k = weight_k × score_k  (or weight_k × (100 − score_k) for mean_reversion)

    The six composite scores sum to approximately composite_risk_score (unsmoothed).
    cross_asset is tracked but excluded from the weighted stack.

    Returns DataFrame with same index as df and columns:
      {key}_contribution  for each score in COMPOSITE_WEIGHTS
    """
    available = _present(df)
    out = {}
    for key, col in available.items():
        if key not in COMPOSITE_WEIGHTS:
            continue
        w = COMPOSITE_WEIGHTS[key]
        if key == "mean_reversion":
            out[f"{key}_contribution"] = (w * (100 - df[col])).clip(lower=0)
        else:
            out[f"{key}_contribution"] = (w * df[col]).clip(lower=0)
    return pd.DataFrame(out, index=df.index)


def compute_shift_attribution(df, regime_col="final_decision", lookback=DELTA_LOOKBACK):
    """
    For each regime transition, identify which scores moved most over the
    preceding `lookback` trading days and classify them as the trigger.

    Score deltas are signed so that positive = more risk added:
      - all scores except mean_reversion: delta = current − prior
      - mean_reversion: delta = prior − current (higher MR score = lower risk)

    Returns DataFrame with columns:
      date, from_regime, to_regime,
      {key}_delta  (one per score in SCORE_COLS),
      primary_driver, secondary_driver,
      composite_delta, direction
    """
    available = _present(df)
    if regime_col not in df.columns or df.empty:
        return pd.DataFrame()

    series = df[regime_col]
    is_transition = (series != series.shift(1)) & series.notna() & series.shift(1).notna()
    is_transition.iloc[0] = False

    records = []
    for loc, (idx, _) in enumerate(df[is_transition].iterrows()):
        abs_loc = df.index.get_loc(idx)
        prior_loc = max(0, abs_loc - lookback)
        prior_idx = df.index[prior_loc]
        prev_idx = df.index[abs_loc - 1]

        row = {
            "date": idx,
            "from_regime": df.at[prev_idx, regime_col],
            "to_regime": df.at[idx, regime_col],
        }

        deltas = {}
        for key, col in available.items():
            current_val = df.at[idx, col]
            prior_val = df.at[prior_idx, col]
            if key == "mean_reversion":
                delta = prior_val - current_val
            else:
                delta = current_val - prior_val
            row[f"{key}_delta"] = round(delta, 2)
            deltas[key] = delta

        ranked = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
        row["primary_driver"] = DISPLAY_NAMES.get(ranked[0][0], ranked[0][0]) if ranked else None
        row["secondary_driver"] = DISPLAY_NAMES.get(ranked[1][0], ranked[1][0]) if len(ranked) > 1 else None

        if "composite_risk_score_smooth" in df.columns:
            c_delta = df.at[idx, "composite_risk_score_smooth"] - df.at[prior_idx, "composite_risk_score_smooth"]
            row["composite_delta"] = round(c_delta, 2)
            row["direction"] = "Risk-Off" if c_delta > 0 else "Risk-On"
        else:
            row["composite_delta"] = None
            row["direction"] = None

        records.append(row)

    return pd.DataFrame(records)


def compute_trigger_elevation(df, regime_col="final_decision"):
    """
    For each decision type, compare mean score levels to global 75th-percentile
    thresholds to identify which scores are systematically elevated.

    Returns DataFrame indexed by decision with columns:
      n_obs,
      {key}_mean  (mean score level while in this regime),
      {key}_elevated  (bool: mean > global 75th pct)
    """
    available = _present(df)
    if regime_col not in df.columns or df.empty:
        return pd.DataFrame()

    thresholds = {k: df[v].quantile(ELEVATION_PERCENTILE / 100) for k, v in available.items()}

    records = []
    for decision, grp in df.groupby(regime_col):
        row = {"decision": decision, "n_obs": len(grp)}
        for key, col in available.items():
            mean_val = grp[col].mean()
            row[f"{key}_mean"] = round(mean_val, 1)
            row[f"{key}_elevated"] = bool(mean_val > thresholds[key])
        records.append(row)

    return (
        pd.DataFrame(records)
        .set_index("decision")
        .sort_values("n_obs", ascending=False)
    )


def compute_top_drivers(row, df):
    """
    Rank sub-scores for a single row by how far above their 75th-percentile
    threshold they sit.

    Returns list of dicts sorted by excess descending:
      {name, key, level, threshold_75, excess, elevated}
    """
    available = _present(df)
    thresholds = {k: df[v].quantile(ELEVATION_PERCENTILE / 100) for k, v in available.items()}

    drivers = []
    for key, col in available.items():
        if col not in row.index:
            continue
        level = float(row[col])
        t75 = thresholds[key]
        excess = level - t75
        drivers.append({
            "name": DISPLAY_NAMES.get(key, key),
            "key": key,
            "level": round(level, 1),
            "threshold_75": round(t75, 1),
            "excess": round(excess, 1),
            "elevated": bool(excess > 0),
        })

    return sorted(drivers, key=lambda x: x["excess"], reverse=True)


def run_regime_attribution(df):
    """
    Full attribution analysis over the scored DataFrame.

    Returns dict with:
      rolling_contributions  — DataFrame of weighted per-score contributions over time
      shift_attribution      — DataFrame of per-transition primary/secondary drivers
      trigger_elevation      — DataFrame of score elevation by decision type
      top_drivers_current    — ranked driver list for the latest row
    """
    return {
        "rolling_contributions": compute_rolling_contributions(df),
        "shift_attribution": compute_shift_attribution(df),
        "trigger_elevation": compute_trigger_elevation(df),
        "top_drivers_current": compute_top_drivers(df.iloc[-1], df),
    }
