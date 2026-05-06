import numpy as np


CRISIS_ANALOGS = {
    "COVID Crash": {
        "macro_risk": 90,
        "credit_risk": 95,
        "complacency": 5,
        "mean_reversion": 95,
        "treasury_stress": 60,
    },
    "Inflation / Duration Shock": {
        "macro_risk": 65,
        "credit_risk": 55,
        "complacency": 20,
        "mean_reversion": 35,
        "treasury_stress": 90,
    },
    "Late-Cycle Melt-Up": {
        "macro_risk": 15,
        "credit_risk": 10,
        "complacency": 90,
        "mean_reversion": 0,
        "treasury_stress": 20,
    },
    "Credit Tightening": {
        "macro_risk": 45,
        "credit_risk": 75,
        "complacency": 25,
        "mean_reversion": 55,
        "treasury_stress": 45,
    },
    "Benign Risk-On": {
        "macro_risk": 10,
        "credit_risk": 10,
        "complacency": 45,
        "mean_reversion": 0,
        "treasury_stress": 10,
    },
}


def compute_crisis_similarity(row):
    current = np.array([
        row["macro_risk_score_smooth"],
        row["credit_market_risk_score_smooth"],
        row["complacency_score_smooth"],
        row["mean_reversion_score_smooth"],
        row["treasury_stress_score_smooth"],
    ])

    results = []

    for name, values in CRISIS_ANALOGS.items():
        analog = np.array([
            values["macro_risk"],
            values["credit_risk"],
            values["complacency"],
            values["mean_reversion"],
            values["treasury_stress"],
        ])

        distance = np.linalg.norm(current - analog)
        similarity = max(0, 100 - distance)
        results.append((name, round(similarity, 1)))

    return sorted(results, key=lambda x: x[1], reverse=True)