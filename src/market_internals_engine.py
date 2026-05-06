import pandas as pd


def compute_cross_asset_divergence_score(
    sp500_return_30d,
    sp500_drawdown,
    hy_change_30d,
    vix,
    vix_change_30d,
):
    score = 0

    if pd.notna(sp500_return_30d) and pd.notna(hy_change_30d):
        if sp500_return_30d > 0.03 and hy_change_30d > 0.10:
            score += 35
        elif sp500_return_30d > 0.01 and hy_change_30d > 0:
            score += 20

        if sp500_return_30d < -0.03 and hy_change_30d < 0:
            score += 15

    if pd.notna(vix) and pd.notna(hy_change_30d):
        if vix < 18 and hy_change_30d > 0.15:
            score += 30
        elif vix < 20 and hy_change_30d > 0:
            score += 15

    if pd.notna(vix_change_30d) and pd.notna(hy_change_30d):
        if vix_change_30d < -5 and hy_change_30d > 0:
            score += 25

    if pd.notna(sp500_drawdown) and sp500_drawdown > -0.02 and pd.notna(hy_change_30d) and hy_change_30d > 0:
        score += 20

    return round(max(0, min(score, 100)), 1)


def classify_cross_asset_divergence(score):
    if score >= 70:
        return "Severe Divergence"
    elif score >= 40:
        return "Elevated Divergence"
    elif score >= 20:
        return "Moderate Divergence"
    return "Low Divergence"


def compute_market_internals_score(
    sp500_return_30d,
    sp500_return_5d,
    sp500_drawdown,
    vix,
    vix_change_30d,
):
    score = 0

    if pd.notna(sp500_return_30d) and sp500_return_30d > 0.05:
        score += 25
    elif pd.notna(sp500_return_30d) and sp500_return_30d > 0.02:
        score += 15

    if pd.notna(sp500_return_5d) and sp500_return_5d > 0.015:
        score += 15

    if pd.notna(sp500_drawdown):
        if sp500_drawdown > -0.01:
            score += 25
        elif sp500_drawdown > -0.03:
            score += 15
        elif sp500_drawdown < -0.10:
            score -= 20

    if pd.notna(vix):
        if vix < 15:
            score += 20
        elif vix < 18:
            score += 12
        elif vix > 25:
            score -= 20

    if pd.notna(vix_change_30d) and vix_change_30d < -5:
        score += 15
    elif pd.notna(vix_change_30d) and vix_change_30d > 5:
        score -= 15

    return round(max(0, min(score, 100)), 1)


def classify_market_internals(score):
    if score >= 70:
        return "Strong / Extended"
    elif score >= 40:
        return "Healthy"
    elif score >= 20:
        return "Weakening"
    return "Poor"