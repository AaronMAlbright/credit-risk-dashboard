"""
Validation hardening layer.

Classifies statistical confidence for every analytical output in the framework
and generates structured warnings for low-evidence findings.

Confidence tiers
----------------
  Robust      — sufficient observations to draw conclusions
  Indicative  — directionally informative, interpret with caution
  Exploratory — too few observations; findings are illustrative only

Thresholds (conservative, appropriate for daily financial time-series)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

# Regime-level statistics (number of daily observations while in that regime)
MIN_OBS_REGIME: dict[str, int] = {"robust": 50, "indicative": 20}

# Transition-probability estimates (number of observed outgoing transitions)
MIN_TRANSITIONS: dict[str, int] = {"robust": 20, "indicative": 5}

# Walk-forward validation windows
MIN_WINDOWS: dict[str, int] = {"robust": 12, "indicative": 6}

CONFIDENCE_ROBUST = "Robust"
CONFIDENCE_INDICATIVE = "Indicative"
CONFIDENCE_EXPLORATORY = "Exploratory"

# Minimum correlation absolute value to be considered informative
MIN_INFORMATIVE_CORR = 0.10

# Display labels with sigil
CONFIDENCE_SIGILS = {
    CONFIDENCE_ROBUST:      "✓",
    CONFIDENCE_INDICATIVE:  "~",
    CONFIDENCE_EXPLORATORY: "⚠",
}


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_confidence(n: int, kind: str = "regime") -> str:
    """
    Classify the statistical confidence of an estimate given n observations.

    Parameters
    ----------
    n    : number of observations (days, transitions, or windows)
    kind : "regime", "transition", or "window"

    Returns one of: CONFIDENCE_ROBUST, CONFIDENCE_INDICATIVE, CONFIDENCE_EXPLORATORY
    """
    thresholds = {
        "regime":     MIN_OBS_REGIME,
        "transition": MIN_TRANSITIONS,
        "window":     MIN_WINDOWS,
    }.get(kind, MIN_OBS_REGIME)

    if n >= thresholds["robust"]:
        return CONFIDENCE_ROBUST
    if n >= thresholds["indicative"]:
        return CONFIDENCE_INDICATIVE
    return CONFIDENCE_EXPLORATORY


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------

def audit_regime_stats(
    df: pd.DataFrame,
    regime_col: str = "final_decision",
    return_col: str = "sp500_forward_30d_return",
) -> pd.DataFrame:
    """
    Per-regime statistics with confidence classification.

    Returns DataFrame indexed by regime with columns:
      n_obs, mean_return, hit_rate, worst_5pct,
      confidence, suppressed (bool — exploratory findings)
    """
    if regime_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for regime, grp in df.groupby(regime_col):
        ret = grp[return_col].dropna() if return_col in grp.columns else pd.Series(dtype=float)
        n = len(ret)
        conf = classify_confidence(n, "regime")
        rows.append({
            "regime":      regime,
            "n_obs":       n,
            "mean_return": round(ret.mean(), 4) if n > 0 else np.nan,
            "hit_rate":    round((ret > 0).mean(), 3) if n > 0 else np.nan,
            "worst_5pct":  round(ret.quantile(0.05), 4) if n >= 5 else np.nan,
            "confidence":  conf,
            "suppressed":  conf == CONFIDENCE_EXPLORATORY,
        })

    return (
        pd.DataFrame(rows)
        .set_index("regime")
        .sort_values("n_obs", ascending=False)
    )


def audit_transition_matrix(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify confidence for each row of a transition counts matrix.

    counts_df: DataFrame where rows = from_regime, cols = to_regime,
               values = integer counts (output of compute_transition_matrix).

    Returns DataFrame with columns:
      from_regime, n_outgoing, confidence, suppressed
    """
    if counts_df is None or counts_df.empty:
        return pd.DataFrame()

    rows = []
    for from_regime in counts_df.index:
        n_out = int(counts_df.loc[from_regime].sum())
        conf = classify_confidence(n_out, "transition")
        rows.append({
            "from_regime": from_regime,
            "n_outgoing":  n_out,
            "confidence":  conf,
            "suppressed":  conf == CONFIDENCE_EXPLORATORY,
        })

    return pd.DataFrame(rows).set_index("from_regime").sort_values("n_outgoing", ascending=False)


def audit_walk_forward(windows_df: pd.DataFrame | None) -> dict:
    """
    Assess overall walk-forward validation confidence.

    Returns dict with keys:
      n_windows, confidence, warnings (list of str),
      sharpe_cv (coefficient of variation on strategy Sharpe)
    """
    if windows_df is None or windows_df.empty:
        return {
            "n_windows":  0,
            "confidence": CONFIDENCE_EXPLORATORY,
            "warnings":   ["No walk-forward results available."],
            "sharpe_cv":  np.nan,
        }

    n = len(windows_df)
    conf = classify_confidence(n, "window")
    warnings: list[str] = []

    if conf != CONFIDENCE_ROBUST:
        warnings.append(
            f"Only {n} walk-forward window{'s' if n != 1 else ''} available "
            f"(robust threshold: {MIN_WINDOWS['robust']}). "
            "Sharpe and drawdown estimates carry wide uncertainty."
        )

    sharpe_cv = np.nan
    if "strategy_sharpe" in windows_df.columns:
        sharpe_vals = windows_df["strategy_sharpe"].dropna()
        if len(sharpe_vals) >= 2 and abs(sharpe_vals.mean()) > 1e-6:
            sharpe_cv = round(sharpe_vals.std() / abs(sharpe_vals.mean()), 3)
            if sharpe_cv > 0.5:
                warnings.append(
                    f"High Sharpe ratio variability across windows (CV = {sharpe_cv:.2f}). "
                    "Performance is inconsistent — results may not generalise."
                )

    pct_beat = np.nan
    if "strategy_sharpe" in windows_df.columns and "sp500_sharpe" in windows_df.columns:
        beat = (windows_df["strategy_sharpe"] > windows_df["sp500_sharpe"]).sum()
        pct_beat = beat / n
        if pct_beat < 0.5:
            warnings.append(
                f"Strategy beat SP500 Sharpe in only {beat}/{n} windows ({pct_beat:.0%}). "
                "Edge may not be consistent."
            )

    return {
        "n_windows":  n,
        "confidence": conf,
        "warnings":   warnings,
        "sharpe_cv":  sharpe_cv,
    }


def audit_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score–return correlation audit with minimum informative threshold.

    Returns DataFrame with columns:
      signal, return_col, correlation, informative (bool), confidence
    """
    score_cols = {
        "Macro Risk":       "macro_risk_score_smooth",
        "Credit Risk":      "credit_market_risk_score_smooth",
        "Complacency":      "complacency_score_smooth",
        "Liquidity":        "liquidity_regime_score_smooth",
        "Treasury Stress":  "treasury_stress_score_smooth",
        "Mean Reversion":   "mean_reversion_score_smooth",
        "Composite":        "composite_risk_score_smooth",
    }
    return_col = "sp500_forward_30d_return"
    rows = []
    for name, col in score_cols.items():
        if col not in df.columns or return_col not in df.columns:
            continue
        clean = df[[col, return_col]].dropna()
        n = len(clean)
        if n < 20:
            rows.append({
                "signal": name, "correlation": np.nan,
                "n_obs": n, "informative": False,
                "confidence": CONFIDENCE_EXPLORATORY,
            })
            continue
        corr = round(clean[col].corr(clean[return_col]), 3)
        informative = abs(corr) >= MIN_INFORMATIVE_CORR
        rows.append({
            "signal":      name,
            "correlation": corr,
            "n_obs":       n,
            "informative": informative,
            "confidence":  classify_confidence(n, "regime"),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("signal")


def generate_warnings(audit: dict) -> list[str]:
    """
    Flatten all warnings across audit sub-results into a single list.

    audit: dict output of run_validation_audit.
    """
    warnings: list[str] = []

    wf = audit.get("walk_forward", {})
    warnings.extend(wf.get("warnings", []))

    regime_df = audit.get("regime_stats")
    if regime_df is not None and not regime_df.empty:
        exp = regime_df[regime_df["confidence"] == CONFIDENCE_EXPLORATORY]
        if not exp.empty:
            names = ", ".join(exp.index.tolist())
            warnings.append(
                f"Exploratory regime(s) with < {MIN_OBS_REGIME['indicative']} observations: "
                f"{names}. Forward return estimates are not reliable."
            )
        ind = regime_df[regime_df["confidence"] == CONFIDENCE_INDICATIVE]
        if not ind.empty:
            names = ", ".join(ind.index.tolist())
            warnings.append(
                f"Indicative regime(s) with {MIN_OBS_REGIME['indicative']}–"
                f"{MIN_OBS_REGIME['robust'] - 1} observations: {names}. "
                "Interpret directionally."
            )

    trans_df = audit.get("transition_matrix")
    if trans_df is not None and not trans_df.empty:
        exp = trans_df[trans_df["confidence"] == CONFIDENCE_EXPLORATORY]
        if not exp.empty:
            names = ", ".join(exp.index.tolist())
            warnings.append(
                f"Transition probabilities from [{names}] have < "
                f"{MIN_TRANSITIONS['indicative']} observed transitions. "
                "Probability estimates are unreliable."
            )

    return warnings


def run_validation_audit(
    df: pd.DataFrame,
    windows_df: pd.DataFrame | None = None,
    transition_counts: pd.DataFrame | None = None,
    regime_col: str = "final_decision",
) -> dict:
    """
    Full validation audit across all analytical layers.

    Returns dict with:
      regime_stats       — per-regime confidence table
      transition_matrix  — per-from-regime transition confidence
      walk_forward       — walk-forward confidence and warnings
      correlations       — signal–return correlation audit
      warnings           — deduplicated human-readable warning list
      summary            — overall system confidence ("Robust" / "Indicative" / "Exploratory")
    """
    regime_stats  = audit_regime_stats(df, regime_col=regime_col)
    transition_m  = audit_transition_matrix(transition_counts) if transition_counts is not None else pd.DataFrame()
    walk_forward  = audit_walk_forward(windows_df)
    correlations  = audit_correlations(df)

    audit = {
        "regime_stats":      regime_stats,
        "transition_matrix": transition_m,
        "walk_forward":      walk_forward,
        "correlations":      correlations,
    }

    warnings = generate_warnings(audit)
    audit["warnings"] = warnings

    # Overall system confidence = worst tier present
    tiers = [CONFIDENCE_ROBUST, CONFIDENCE_INDICATIVE, CONFIDENCE_EXPLORATORY]

    all_confs: list[str] = [walk_forward["confidence"]]
    if not regime_stats.empty:
        all_confs.extend(regime_stats["confidence"].tolist())
    if not transition_m.empty:
        all_confs.extend(transition_m["confidence"].tolist())

    worst = max(all_confs, key=lambda c: tiers.index(c))
    audit["summary"] = {
        "overall_confidence": worst,
        "n_robust":      all_confs.count(CONFIDENCE_ROBUST),
        "n_indicative":  all_confs.count(CONFIDENCE_INDICATIVE),
        "n_exploratory": all_confs.count(CONFIDENCE_EXPLORATORY),
    }

    return audit
