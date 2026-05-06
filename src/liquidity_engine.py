import pandas as pd


def compute_liquidity_regime_score(
    nfci,
    nfci_change_90d,
    hy_change_30d,
    vix_change_30d,
    treasury_stress_score,
):
    score = 0

    if pd.notna(nfci):
        if nfci > 0.5:
            score += 35
        elif nfci > 0:
            score += 20
        elif nfci < -0.5:
            score -= 10

    if pd.notna(nfci_change_90d):
        if nfci_change_90d > 0.20:
            score += 30
        elif nfci_change_90d > 0.05:
            score += 15
        elif nfci_change_90d < -0.10:
            score -= 10

    if pd.notna(hy_change_30d):
        if hy_change_30d > 0.50:
            score += 30
        elif hy_change_30d > 0.20:
            score += 15
        elif hy_change_30d < -0.30:
            score -= 10

    if pd.notna(vix_change_30d):
        if vix_change_30d > 7:
            score += 25
        elif vix_change_30d > 3:
            score += 12
        elif vix_change_30d < -5:
            score -= 10

    if pd.notna(treasury_stress_score):
        if treasury_stress_score > 70:
            score += 25
        elif treasury_stress_score > 40:
            score += 12

    return round(max(0, min(score, 100)), 1)


def classify_liquidity_regime(score):
    if score >= 70:
        return "Severe Tightening"
    elif score >= 40:
        return "Tightening"
    elif score >= 20:
        return "Neutral"
    return "Easy / Supportive"


def generate_liquidity_signal(score):
    if score >= 70:
        return "DEFENSIVE"
    elif score >= 40:
        return "REDUCE RISK"
    elif score >= 20:
        return "NEUTRAL"
    return "SUPPORTIVE"