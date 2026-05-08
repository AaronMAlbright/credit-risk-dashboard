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
# Cross-asset signals (fx_commodity, enhanced_funding) added at conservative
# initial weights after 2-year OOS hold-out. liquidity_regime trimmed 15→10%
# to make room; mean_reversion removed (redundant with complacency overlap).
_WEIGHTS = {
    "macro_risk_score_smooth":            0.25,
    "credit_market_risk_score_smooth":    0.25,
    "liquidity_regime_score_smooth":      0.10,  # was 0.15
    "treasury_stress_score_smooth":       0.10,
    "complacency_score_smooth":           0.20,
    "fx_commodity_score_smooth":          0.05,  # new cross-asset signal
    "enhanced_funding_stress_score_smooth": 0.05, # new cross-asset signal
    # mean_reversion_risk_component dropped (weight→0; reduces overfitting)
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