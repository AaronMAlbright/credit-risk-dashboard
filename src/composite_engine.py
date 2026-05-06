import numpy as np


def classify_composite_risk(score):
    if score >= 70:
        return "Defensive / High Risk"
    elif score >= 50:
        return "Caution"
    elif score >= 25:
        return "Neutral"
    return "Risk-On"


def build_composite_risk(df):
    df = df.copy()

    required_cols = [
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "treasury_stress_score_smooth",
        "complacency_score_smooth",
        "mean_reversion_score_smooth",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Mean-reversion is inverted because high mean-reversion is usually opportunity, not pure danger.
    df["mean_reversion_risk_component"] = 100 - df["mean_reversion_score_smooth"]

    df["composite_risk_score"] = (
        0.25 * df["macro_risk_score_smooth"]
        + 0.25 * df["credit_market_risk_score_smooth"]
        + 0.15 * df["liquidity_regime_score_smooth"]
        + 0.10 * df["treasury_stress_score_smooth"]
        + 0.20 * df["complacency_score_smooth"]
        + 0.05 * df["mean_reversion_risk_component"]
    )

    df["composite_risk_score"] = df["composite_risk_score"].clip(0, 100)

    df["composite_risk_score_smooth"] = (
        df["composite_risk_score"]
        .rolling(10, min_periods=1)
        .mean()
    )

    df["composite_risk_label"] = df["composite_risk_score_smooth"].apply(
        classify_composite_risk
    )

    return df