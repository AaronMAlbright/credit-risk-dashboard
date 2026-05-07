"""
Scenario analysis.

Applies user-defined market shocks to the latest observation and propagates
them through the full risk scoring chain:

  observable inputs → derived features → component scores (raw) →
  composite score (delta-on-smooth) → regime probabilities → position sizing

Shock propagation rule:
  - Level variables (vix, hy_spread, spread, nfci, unemployment) shift directly.
  - Change/momentum variables (hy_change_30d, vix_change_30d, etc.) absorb the
    same delta — interpreted as "this shock built over the lookback window."
  - Classification features (credit_equity_divergence, vol_credit_mismatch,
    shock_flag, market_internals_score, cross_asset_divergence_score) are
    recomputed from the shocked inputs.
  - Smoothed scores use a delta-on-smooth approximation:
      shocked_smooth_i = baseline_smooth_i + (shocked_raw_i - baseline_raw_i)
    This preserves the existing trend and adds the instantaneous shock effect.

Public API
----------
  DEFAULT_SHOCKS      — zero-shock dict (all keys = 0)
  SCENARIO_PRESETS    — named built-in shock dicts
  TORNADO_SHOCKS      — single-variable shocks for sensitivity analysis
  build_shocked_row   — apply shocks to a pd.Series, return updated row
  score_shocked_row   — compute all scores + decision for a shocked row
  run_scenario        — single scenario → comparison dict
  run_scenario_grid   — multiple scenarios → comparison DataFrame
  build_tornado_data  — per-variable sensitivity DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_engine import (
    classify_yield_curve_regime,
    classify_labor_warning,
    classify_credit_equity_divergence,
    classify_vol_credit_mismatch,
    detect_shock,
    compute_macro_risk_score,
    compute_credit_market_risk_score,
    compute_liquidity_score,
    compute_risk_appetite_score,
    compute_complacency_score,
    compute_mean_reversion_score,
    generate_final_decision,
)
from src.liquidity_engine import compute_liquidity_regime_score
from src.market_internals_engine import (
    compute_cross_asset_divergence_score,
    compute_market_internals_score,
)
from src.treasury_engine import compute_treasury_stress_score
from src.position_sizing import (
    REGIME_WEIGHTS,
    SCORE_BREAKPOINTS,
    _interp_weight,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPOSITE_WEIGHTS = {
    "macro_risk_score_smooth":         0.25,
    "credit_market_risk_score_smooth": 0.25,
    "liquidity_regime_score_smooth":   0.15,
    "treasury_stress_score_smooth":    0.10,
    "complacency_score_smooth":        0.20,
}
_MEAN_REV_WEIGHT = 0.05   # applied to (100 - mean_reversion_score_smooth)

# Columns carried from the latest row that aren't recomputed
_PASSTHROUGH_COLS = [
    "macro_risk_momentum_10d",
    "credit_risk_momentum_10d",
    "treasury_stress_score_smooth",
    "cross_asset_divergence_score_smooth",
    "transition_regime",
    "treasury_stress_score",
    "real_yield_z",
    "real_yield_change_90d",
    "curve_steepening_velocity_90d",
]

DEFAULT_SHOCKS: dict[str, float] = {
    "vix_shock":          0.0,
    "hy_spread_shock":    0.0,
    "sp500_shock":        0.0,   # relative, e.g. -0.15 = -15%
    "spread_shock":       0.0,   # 2/10 Treasury spread change in pp
    "nfci_shock":         0.0,
    "unemployment_shock": 0.0,
}

SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    "Baseline": dict(DEFAULT_SHOCKS),
    "Mild Risk-Off": {
        "vix_shock":          5.0,
        "hy_spread_shock":    0.50,
        "sp500_shock":       -0.05,
        "spread_shock":      -0.25,
        "nfci_shock":         0.10,
        "unemployment_shock": 0.0,
    },
    "Moderate Stress": {
        "vix_shock":          15.0,
        "hy_spread_shock":    1.50,
        "sp500_shock":       -0.15,
        "spread_shock":      -0.50,
        "nfci_shock":         0.25,
        "unemployment_shock": 0.30,
    },
    "Severe Stress": {
        "vix_shock":          30.0,
        "hy_spread_shock":    3.50,
        "sp500_shock":       -0.30,
        "spread_shock":      -0.75,
        "nfci_shock":         1.00,
        "unemployment_shock": 1.50,
    },
    "Buy Stress Setup": {
        "vix_shock":          25.0,
        "hy_spread_shock":    2.00,
        "sp500_shock":       -0.20,
        "spread_shock":       0.0,
        "nfci_shock":         0.30,
        "unemployment_shock": 0.0,
    },
    "Risk-On Rally": {
        "vix_shock":         -5.0,
        "hy_spread_shock":   -0.50,
        "sp500_shock":        0.10,
        "spread_shock":       0.25,
        "nfci_shock":        -0.20,
        "unemployment_shock": 0.0,
    },
    "Yield Curve Inversion": {
        "vix_shock":          2.0,
        "hy_spread_shock":    0.25,
        "sp500_shock":       -0.03,
        "spread_shock":      -0.75,
        "nfci_shock":         0.10,
        "unemployment_shock": 0.0,
    },
}

# Single-variable shocks for tornado chart (representative magnitude)
TORNADO_SHOCKS: dict[str, dict[str, float]] = {
    "HY Spread +1.5pp":    {"hy_spread_shock":    1.5},
    "VIX +15":             {"vix_shock":          15.0},
    "SP500 -15%":          {"sp500_shock":        -0.15},
    "NFCI +0.3":           {"nfci_shock":          0.3},
    "Spread -0.5pp":       {"spread_shock":       -0.5},
    "Unemployment +0.5":   {"unemployment_shock":  0.5},
}

_COMPONENT_SCORE_PAIRS = [
    ("macro_risk_score",          "macro_risk_score_smooth"),
    ("credit_market_risk_score",  "credit_market_risk_score_smooth"),
    ("liquidity_regime_score",    "liquidity_regime_score_smooth"),
    ("treasury_stress_score",     "treasury_stress_score_smooth"),
    ("complacency_score",         "complacency_score_smooth"),
    ("mean_reversion_score",      "mean_reversion_score_smooth"),
]


# ---------------------------------------------------------------------------
# Shock propagation
# ---------------------------------------------------------------------------

def build_shocked_row(latest: pd.Series, shocks: dict) -> pd.Series:
    """
    Apply market shocks to the latest observation.

    Returns a copy of `latest` with all affected raw-feature and
    classification-feature columns updated.  Smoothed score columns
    are NOT touched here; use score_shocked_row which applies delta-on-smooth.
    """
    row = latest.copy()

    vix_shock          = float(shocks.get("vix_shock",          0.0))
    hy_spread_shock    = float(shocks.get("hy_spread_shock",    0.0))
    sp500_shock        = float(shocks.get("sp500_shock",        0.0))
    spread_shock       = float(shocks.get("spread_shock",       0.0))
    nfci_shock         = float(shocks.get("nfci_shock",         0.0))
    unemployment_shock = float(shocks.get("unemployment_shock", 0.0))

    # ── VIX ──
    row["vix"]           = max(5.0, float(row.get("vix", 20)) + vix_shock)
    row["vix_change_30d"] = float(row.get("vix_change_30d", 0)) + vix_shock
    row["vix_change_5d"]  = float(row.get("vix_change_5d",  0)) + vix_shock

    # ── HY spread ──
    row["hy_spread"]       = max(0.5, float(row.get("hy_spread", 3.0))     + hy_spread_shock)
    row["hy_change_30d"]   = float(row.get("hy_change_30d",  0)) + hy_spread_shock
    row["hy_change_90d"]   = float(row.get("hy_change_90d",  0)) + hy_spread_shock
    row["hy_change_5d"]    = float(row.get("hy_change_5d",   0)) + hy_spread_shock
    row["credit_impulse"]  = float(row.get("credit_impulse", 0)) + hy_spread_shock

    # ── SP500 (multiplicative shock) ──
    if sp500_shock != 0.0:
        row["sp500_return_30d"] = float(row.get("sp500_return_30d", 0)) + sp500_shock
        row["sp500_return_5d"]  = float(row.get("sp500_return_5d",  0)) + sp500_shock
        orig_dd = float(row.get("sp500_drawdown", 0))
        row["sp500_drawdown"] = float((orig_dd + 1.0) * (1.0 + sp500_shock) - 1.0)

    # ── Treasury spread ──
    row["spread"]            = float(row.get("spread",           0.5))  + spread_shock
    row["spread_change_90d"] = float(row.get("spread_change_90d", 0))   + spread_shock
    row["spread_change_5d"]  = float(row.get("spread_change_5d",  0))   + spread_shock
    row["yield_curve_regime"] = classify_yield_curve_regime(float(row["spread"]))

    # ── NFCI ──
    row["nfci"]            = float(row.get("nfci",           0)) + nfci_shock
    row["nfci_90d_avg"]    = float(row.get("nfci_90d_avg",   0)) + nfci_shock
    row["nfci_change_90d"] = float(row.get("nfci_change_90d", 0)) + nfci_shock

    # ── Unemployment ──
    row["unemployment"]            = max(0.0, float(row.get("unemployment",           4.0)) + unemployment_shock)
    row["unemployment_change_90d"] = float(row.get("unemployment_change_90d", 0))            + unemployment_shock
    row["sahm_like"]               = max(0.0, float(row.get("sahm_like", 0))                 + unemployment_shock)
    row["labor_warning"]           = classify_labor_warning(float(row["sahm_like"]),
                                                            float(row["unemployment_change_90d"]))

    # ── Recompute classification features ──
    row["credit_equity_divergence"] = classify_credit_equity_divergence(
        float(row.get("sp500_return_30d", 0)), float(row["hy_change_30d"])
    )
    row["vol_credit_mismatch"] = classify_vol_credit_mismatch(
        float(row["vix"]), float(row["vix_change_30d"]), float(row["hy_change_30d"])
    )
    row["market_internals_score"] = compute_market_internals_score(
        float(row.get("sp500_return_30d", 0)), float(row.get("sp500_return_5d", 0)),
        float(row.get("sp500_drawdown",   0)), float(row["vix"]), float(row["vix_change_30d"])
    )
    row["cross_asset_divergence_score"] = compute_cross_asset_divergence_score(
        float(row.get("sp500_return_30d", 0)), float(row.get("sp500_drawdown", 0)),
        float(row["hy_change_30d"]), float(row["vix"]), float(row["vix_change_30d"])
    )
    row["shock_flag"] = detect_shock(
        float(row["vix_change_5d"]), float(row["hy_change_5d"]),
        float(row.get("sp500_return_5d", 0)), float(row["spread_change_5d"])
    )

    return row


# ---------------------------------------------------------------------------
# Score recomputation (delta-on-smooth)
# ---------------------------------------------------------------------------

def score_shocked_row(shocked: pd.Series, baseline: pd.Series) -> dict:
    """
    Recompute all component and composite scores for a shocked row.

    Delta-on-smooth formula:
      shocked_smooth_i = baseline_smooth_i + (shocked_raw_i - baseline_raw_i)
    clipped to [0, 100].

    Returns a flat dict with:
      shocked market levels, component smooth scores, composite, decision.
    """
    # ── First-order raw scores ──
    macro_raw = compute_macro_risk_score(
        float(shocked.get("spread",            0)),
        float(shocked.get("unemployment",      4)),
        float(shocked.get("nfci_90d_avg",      0)),
        float(shocked.get("unemployment_change_90d", 0)),
        float(shocked.get("spread_change_90d", 0)),
        float(shocked.get("nfci_change_90d",   0)),
        float(shocked.get("sahm_like",         0)),
    )
    credit_raw = compute_credit_market_risk_score(
        float(shocked.get("hy_spread",             3)),
        float(shocked.get("hy_change_30d",         0)),
        float(shocked.get("hy_change_90d",         0)),
        float(shocked.get("credit_impulse",        0)),
        float(shocked.get("vix",                  20)),
        float(shocked.get("vix_change_30d",        0)),
        str(shocked.get("credit_equity_divergence", "Neutral")),
        str(shocked.get("vol_credit_mismatch",      "Neutral")),
    )
    liquidity_raw = compute_liquidity_score(
        float(shocked.get("nfci_90d_avg",   0)),
        float(shocked.get("nfci_change_90d", 0)),
        float(shocked.get("hy_change_30d",   0)),
        float(shocked.get("vix_change_30d",  0)),
    )
    treasury_raw = float(baseline.get("treasury_stress_score", 0))
    liq_regime_raw = compute_liquidity_regime_score(
        float(shocked.get("nfci_90d_avg",   0)),
        float(shocked.get("nfci_change_90d", 0)),
        float(shocked.get("hy_change_30d",   0)),
        float(shocked.get("vix_change_30d",  0)),
        treasury_raw,
    )

    # ── Delta-on-smooth for first-order scores ──
    def _delta_smooth(b_smooth_key, b_raw_key, shocked_raw):
        base_smooth = float(baseline.get(b_smooth_key, 0))
        base_raw    = float(baseline.get(b_raw_key,    0))
        return float(np.clip(base_smooth + (shocked_raw - base_raw), 0, 100))

    macro_smooth    = _delta_smooth("macro_risk_score_smooth",           "macro_risk_score",          macro_raw)
    credit_smooth   = _delta_smooth("credit_market_risk_score_smooth",   "credit_market_risk_score",  credit_raw)
    liq_reg_smooth  = _delta_smooth("liquidity_regime_score_smooth",     "liquidity_regime_score",    liq_regime_raw)
    treasury_smooth = float(baseline.get("treasury_stress_score_smooth", 0))

    # ── Second-order raw scores (risk_appetite → complacency, mean_reversion) ──
    # Momentum terms are 10-day windows of history — unchanged by instantaneous shock
    macro_mom  = float(baseline.get("macro_risk_momentum_10d",  0))
    credit_mom = float(baseline.get("credit_risk_momentum_10d", 0))

    risk_appetite_raw = compute_risk_appetite_score(
        float(shocked.get("hy_change_30d",  0)),
        float(shocked.get("vix_change_30d", 0)),
        float(shocked.get("sp500_drawdown", 0)),
        macro_mom, credit_mom,
    )
    complacency_raw = compute_complacency_score(
        float(shocked.get("vix",            20)),
        float(shocked.get("hy_spread",       3)),
        float(shocked.get("sp500_drawdown",  0)),
        macro_mom,
        float(shocked.get("nfci_90d_avg",   0)),
        float(shocked.get("hy_change_30d",  0)),
        risk_appetite_raw,
        float(shocked.get("market_internals_score", 50)),
    )
    mean_rev_raw = compute_mean_reversion_score(
        macro_smooth, credit_smooth,
        float(shocked.get("hy_spread",      3)),
        float(shocked.get("hy_change_30d",  0)),
        float(shocked.get("vix",           20)),
        float(shocked.get("vix_change_30d", 0)),
        float(shocked.get("sp500_drawdown", 0)),
    )

    complacency_smooth = _delta_smooth("complacency_score_smooth",    "complacency_score",    complacency_raw)
    mean_rev_smooth    = _delta_smooth("mean_reversion_score_smooth", "mean_reversion_score", mean_rev_raw)

    # ── Cross-asset divergence (for final_decision) ──
    cross_asset_raw    = float(shocked.get("cross_asset_divergence_score", 0))
    cross_asset_smooth = _delta_smooth(
        "cross_asset_divergence_score_smooth", "cross_asset_divergence_score", cross_asset_raw
    )

    # ── Composite ──
    composite = float(np.clip(
        0.25 * macro_smooth
        + 0.25 * credit_smooth
        + 0.15 * liq_reg_smooth
        + 0.10 * treasury_smooth
        + 0.20 * complacency_smooth
        + 0.05 * (100.0 - mean_rev_smooth),
        0, 100,
    ))

    # ── Final decision ──
    transition_regime = str(baseline.get("transition_regime", "Complacency / Late Cycle"))
    decision_obj = generate_final_decision(
        macro_smooth, credit_smooth, liq_reg_smooth, cross_asset_smooth,
        complacency_smooth, mean_rev_smooth, transition_regime,
        str(shocked.get("shock_flag", "No Shock")),
        str(shocked.get("credit_equity_divergence", "Neutral")),
        str(shocked.get("vol_credit_mismatch",      "Neutral")),
        composite,
    )

    return {
        # Shocked market levels
        "vix":            float(shocked.get("vix",            0)),
        "hy_spread":      float(shocked.get("hy_spread",      0)),
        "sp500_drawdown": float(shocked.get("sp500_drawdown", 0)),
        "spread":         float(shocked.get("spread",         0)),
        "nfci":           float(shocked.get("nfci",           0)),
        "unemployment":   float(shocked.get("unemployment",   0)),
        # Component smoothed scores
        "macro_risk_score_smooth":         macro_smooth,
        "credit_market_risk_score_smooth": credit_smooth,
        "liquidity_regime_score_smooth":   liq_reg_smooth,
        "treasury_stress_score_smooth":    treasury_smooth,
        "complacency_score_smooth":        complacency_smooth,
        "mean_reversion_score_smooth":     mean_rev_smooth,
        # Composite
        "composite_risk_score_smooth":     composite,
        # Flags
        "shock_flag":               str(shocked.get("shock_flag",               "No Shock")),
        "credit_equity_divergence": str(shocked.get("credit_equity_divergence", "Neutral")),
        "vol_credit_mismatch":      str(shocked.get("vol_credit_mismatch",      "Neutral")),
        # Decision
        "final_decision":    decision_obj["final_decision"],
        "final_environment": decision_obj.get("environment", ""),
        "final_action":      decision_obj.get("action",      ""),
    }


# ---------------------------------------------------------------------------
# Baseline extraction
# ---------------------------------------------------------------------------

def _extract_baseline(df: pd.DataFrame) -> dict:
    """Extract current scores from the latest row of a scored DataFrame."""
    if df.empty:
        return {}
    row = df.iloc[-1]
    keys = [
        "vix", "hy_spread", "sp500_drawdown", "spread", "nfci", "unemployment",
        "macro_risk_score_smooth", "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth", "treasury_stress_score_smooth",
        "complacency_score_smooth", "mean_reversion_score_smooth",
        "composite_risk_score_smooth", "final_decision",
        "shock_flag", "credit_equity_divergence", "vol_credit_mismatch",
    ]
    return {k: row[k] for k in keys if k in row.index}


# ---------------------------------------------------------------------------
# Regime probability and sizing helpers
# ---------------------------------------------------------------------------

def _regime_probs_for_scores(scores: dict, model: dict) -> dict:
    """Compute regime probabilities for a shocked scores dict using fitted GNB model."""
    try:
        from src.regime_probability import score_to_regime_probs
        return score_to_regime_probs(scores, model)
    except Exception:
        return {}


def _sizing_for_scenario(scores: dict, regime_probs: dict) -> dict:
    """Score-based and regime-prob sizing for a scenario."""
    composite = float(scores.get("composite_risk_score_smooth", 50))
    score_sz  = _interp_weight(composite, SCORE_BREAKPOINTS)

    regime_sz = 0.0
    if regime_probs:
        regime_sz = sum(
            float(p) * float(REGIME_WEIGHTS.get(r, 0.50))
            for r, p in regime_probs.items()
        )
        regime_sz = float(np.clip(regime_sz, 0, 1))

    blend_vals = [v for v in [score_sz, regime_sz] if not np.isnan(v)]
    blend = float(np.mean(blend_vals)) if blend_vals else np.nan

    return {
        "score_sizing":       round(score_sz,  4),
        "regime_prob_sizing": round(regime_sz, 4),
        "blend":              round(blend,      4),
    }


# ---------------------------------------------------------------------------
# Single scenario
# ---------------------------------------------------------------------------

def run_scenario(
    df: pd.DataFrame,
    shocks: dict,
    model: dict | None = None,
) -> dict:
    """
    Full scenario comparison: baseline vs. shocked.

    Parameters
    ----------
    df     : scored DataFrame (latest row used as baseline)
    shocks : dict of shock magnitudes (see DEFAULT_SHOCKS for keys)
    model  : pre-fitted GNB regime model (optional; fitted internally if None)

    Returns dict:
      baseline     — baseline scores/levels dict (latest row)
      scenario     — shocked scores/levels dict
      delta        — component score deltas (scenario - baseline)
      shocks       — the shocks applied
      regime_probs — shocked regime probabilities {regime: float}
      sizing       — {score_sizing, regime_prob_sizing, blend} under scenario
      baseline_sizing — same keys for the baseline
    """
    if df.empty:
        return {}

    # Fit model if not provided
    if model is None:
        try:
            from src.regime_probability import fit_naive_bayes
            model = fit_naive_bayes(df)
        except Exception:
            model = {}

    latest   = df.iloc[-1]
    baseline = _extract_baseline(df)

    full_shocks = {**DEFAULT_SHOCKS, **shocks}
    shocked_row = build_shocked_row(latest, full_shocks)
    scenario    = score_shocked_row(shocked_row, latest)

    # Zero-shock reference for delta computation (removes smoothing-vs-formula gap)
    _zero_row      = build_shocked_row(latest, DEFAULT_SHOCKS)
    _zero_scenario = score_shocked_row(_zero_row, latest)

    _COMP_KEYS = [
        "macro_risk_score_smooth", "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth", "treasury_stress_score_smooth",
        "complacency_score_smooth", "mean_reversion_score_smooth",
        "composite_risk_score_smooth",
    ]
    delta = {k: round(float(scenario.get(k, 0)) - float(_zero_scenario.get(k, 0)), 2)
             for k in _COMP_KEYS}

    regime_probs  = _regime_probs_for_scores(scenario, model) if model else {}
    sizing        = _sizing_for_scenario(scenario, regime_probs)
    baseline_probs = _regime_probs_for_scores(baseline, model) if model else {}
    baseline_sizing = _sizing_for_scenario(baseline, baseline_probs)

    return {
        "baseline":        baseline,
        "scenario":        scenario,
        "delta":           delta,
        "shocks":          full_shocks,
        "regime_probs":    regime_probs,
        "baseline_probs":  baseline_probs,
        "sizing":          sizing,
        "baseline_sizing": baseline_sizing,
    }


# ---------------------------------------------------------------------------
# Scenario grid
# ---------------------------------------------------------------------------

def run_scenario_grid(
    df: pd.DataFrame,
    scenarios: dict[str, dict],
    model: dict | None = None,
) -> pd.DataFrame:
    """
    Run multiple named scenarios and return a comparison DataFrame.

    Parameters
    ----------
    df        : scored DataFrame
    scenarios : dict mapping scenario name → shock dict
    model     : pre-fitted GNB model (fitted once internally if None)

    Returns DataFrame indexed by scenario name with columns:
      composite_risk_score_smooth, delta_composite,
      final_decision, score_sizing, regime_prob_sizing, blend,
      macro_risk_score_smooth, credit_market_risk_score_smooth, ...
    """
    if df.empty or not scenarios:
        return pd.DataFrame()

    if model is None:
        try:
            from src.regime_probability import fit_naive_bayes
            model = fit_naive_bayes(df)
        except Exception:
            model = {}

    # Reference composite: zero-shock scenario (removes smoothing-vs-formula gap)
    _base_result = run_scenario(df, DEFAULT_SHOCKS, model)
    baseline_composite = float(
        _base_result.get("scenario", {}).get("composite_risk_score_smooth",
        float(df.iloc[-1].get("composite_risk_score_smooth", np.nan)))
    )

    rows = []
    for name, shocks in scenarios.items():
        result = run_scenario(df, shocks, model)
        sc  = result.get("scenario", {})
        sz  = result.get("sizing",   {})
        rows.append({
            "scenario":                        name,
            "composite_risk_score_smooth":     sc.get("composite_risk_score_smooth"),
            "delta_composite":                 round(sc.get("composite_risk_score_smooth", baseline_composite) - baseline_composite, 2),
            "final_decision":                  sc.get("final_decision", ""),
            "score_sizing":                    sz.get("score_sizing"),
            "regime_prob_sizing":              sz.get("regime_prob_sizing"),
            "blend_sizing":                    sz.get("blend"),
            "macro_risk_score_smooth":         sc.get("macro_risk_score_smooth"),
            "credit_market_risk_score_smooth": sc.get("credit_market_risk_score_smooth"),
            "liquidity_regime_score_smooth":   sc.get("liquidity_regime_score_smooth"),
            "complacency_score_smooth":        sc.get("complacency_score_smooth"),
            "mean_reversion_score_smooth":     sc.get("mean_reversion_score_smooth"),
            "shock_flag":                      sc.get("shock_flag", "No Shock"),
            "vix":                             sc.get("vix"),
            "hy_spread":                       sc.get("hy_spread"),
            "sp500_drawdown":                  sc.get("sp500_drawdown"),
        })

    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# Sensitivity / tornado
# ---------------------------------------------------------------------------

def build_tornado_data(
    df: pd.DataFrame,
    model: dict | None = None,
    tornado_shocks: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Run each variable's shock independently and measure impact on composite score.

    Returns DataFrame with columns:
      variable, delta_composite, direction ('up' or 'down')
    sorted by abs(delta_composite) descending.
    """
    if df.empty:
        return pd.DataFrame()

    if model is None:
        try:
            from src.regime_probability import fit_naive_bayes
            model = fit_naive_bayes(df)
        except Exception:
            model = {}

    shocks = tornado_shocks or TORNADO_SHOCKS
    _base = run_scenario(df, DEFAULT_SHOCKS, model)
    baseline_composite = float(
        _base.get("scenario", {}).get("composite_risk_score_smooth",
        float(df.iloc[-1].get("composite_risk_score_smooth", np.nan)))
    )
    rows = []
    for name, shock_dict in shocks.items():
        result = run_scenario(df, shock_dict, model)
        sc = result.get("scenario", {})
        delta = round(sc.get("composite_risk_score_smooth", baseline_composite) - baseline_composite, 2)
        rows.append({
            "variable":        name,
            "delta_composite": delta,
            "direction":       "up" if delta >= 0 else "down",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["delta_composite"].abs().sort_values(ascending=False).index)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_scenario_analysis(
    df: pd.DataFrame,
    selected_shocks: dict | None = None,
) -> dict:
    """
    Full scenario analysis pipeline.

    Parameters
    ----------
    df             : scored DataFrame
    selected_shocks: shock dict to run (defaults to SCENARIO_PRESETS["Moderate Stress"])

    Returns dict:
      model          — fitted GNB model
      baseline       — baseline scores dict
      preset_grid    — DataFrame from run_scenario_grid over SCENARIO_PRESETS
      tornado        — DataFrame from build_tornado_data
      selected       — run_scenario result for selected_shocks
    """
    if df.empty:
        return {
            "model":       {},
            "baseline":    {},
            "preset_grid": pd.DataFrame(),
            "tornado":     pd.DataFrame(),
            "selected":    {},
        }

    try:
        from src.regime_probability import fit_naive_bayes
        model = fit_naive_bayes(df)
    except Exception:
        model = {}

    if selected_shocks is None:
        selected_shocks = SCENARIO_PRESETS["Moderate Stress"]

    baseline    = _extract_baseline(df)
    preset_grid = run_scenario_grid(df, SCENARIO_PRESETS, model)
    tornado     = build_tornado_data(df, model)
    selected    = run_scenario(df, selected_shocks, model)

    return {
        "model":       model,
        "baseline":    baseline,
        "preset_grid": preset_grid,
        "tornado":     tornado,
        "selected":    selected,
    }
