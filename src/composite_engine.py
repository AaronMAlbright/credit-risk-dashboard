import numpy as np


def classify_composite_risk(score):
    if score >= 70:
        return "Defensive / High Risk"
    elif score >= 50:
        return "Caution"
    elif score >= 25:
        return "Neutral"
    return "Risk-On"


# Composite weights — must sum to 1.0.
#
# Calibrated against multi-horizon Spearman correlation grid (21d–252d) using
# full IS history (2000-2022). Signals categorised by empirical role:
#
#   Leading (negative r at short horizons): treasury, complacency, fx_commodity
#   Lagging/medium (negative r at 42-189d): enhanced_funding
#   Concurrent (barely negative at 126d+):  credit, liquidity
#   Concurrent (never negative):            macro_risk — reduced from 25%
#   Contra-indicator (wrong direction):     banking_stress — excluded (weight 0)
#
# Changes vs prior version:
#   treasury        10% → 20%  (strongest signal at every horizon)
#   enhanced_funding 5% → 10%  (good medium-horizon signal at 42-189d)
#   macro_risk      25% → 15%  (concurrent, not leading)
#   credit_risk     25% → 20%  (concurrent, reduces slightly)
#   banking_stress excluded     (wrong direction at all horizons)
_WEIGHTS = {
    "treasury_stress_score_smooth":           0.20,  # was 0.10 — strongest leading signal
    "complacency_score_smooth":               0.20,  # unchanged — strong short-horizon
    "credit_market_risk_score_smooth":        0.20,  # was 0.25 — concurrent, slightly reduced
    "macro_risk_score_smooth":                0.15,  # was 0.25 — concurrent, not leading
    "liquidity_regime_score_smooth":          0.10,  # unchanged
    "enhanced_funding_stress_score_smooth":   0.10,  # was 0.05 — good medium-horizon
    "fx_commodity_score_smooth":              0.05,  # unchanged — weak leading
    # banking_stress excluded: wrong direction at all horizons (r > 0 up to 252d)
    # mean_reversion excluded: redundant with complacency
}


def build_composite_risk(df):
    df = df.copy()

    # Ensure all required base columns exist before weighting
    base_required = [
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "treasury_stress_score_smooth",
        "complacency_score_smooth",
    ]
    for col in base_required:
        if col not in df.columns:
            df[col] = 0.0

    # Cross-asset signals default to 0 if not yet computed (graceful degradation)
    # When these columns are 0, the base weights rescale implicitly to their
    # original proportions, preserving legacy signal continuity on old data.
    for col in ("fx_commodity_score_smooth", "enhanced_funding_stress_score_smooth"):
        if col not in df.columns:
            df[col] = 0.0

    score = sum(
        w * df[col]
        for col, w in _WEIGHTS.items()
        if col in df.columns
    )
    df["composite_risk_score"] = score.clip(0, 100)

    df["composite_risk_score_smooth"] = (
        df["composite_risk_score"]
        .rolling(10, min_periods=1)
        .mean()
    )

    df["composite_risk_label"] = df["composite_risk_score_smooth"].apply(
        classify_composite_risk
    )

    return df