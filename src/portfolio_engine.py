def normalize_weights(weights):
    total = (
        weights["equity_weight"]
        + weights["credit_weight"]
        + weights["cash_weight"]
    )

    if total == 0:
        return {
            "equity_weight": 0.0,
            "credit_weight": 0.0,
            "cash_weight": 1.0,
            "duration_bias": weights.get("duration_bias", "Neutral"),
        }

    return {
        "equity_weight": round(weights["equity_weight"] / total, 4),
        "credit_weight": round(weights["credit_weight"] / total, 4),
        "cash_weight": round(weights["cash_weight"] / total, 4),
        "duration_bias": weights.get("duration_bias", "Neutral"),
    }


def generate_portfolio_weights(row):
    decision = row["final_decision"]

    if decision == "Buy Stress":
        weights = {
            "equity_weight": 0.55,
            "credit_weight": 0.30,
            "cash_weight": 0.15,
            "duration_bias": "Long Duration",
        }

    elif decision == "Watch Entry":
        weights = {
            "equity_weight": 0.35,
            "credit_weight": 0.25,
            "cash_weight": 0.40,
            "duration_bias": "Neutral / Slight Long",
        }

    elif decision == "Stress / Stabilization Watch":
        weights = {
            "equity_weight": 0.25,
            "credit_weight": 0.15,
            "cash_weight": 0.60,
            "duration_bias": "Long Duration",
        }

    elif decision == "Credit Warning":
        weights = {
            "equity_weight": 0.35,
            "credit_weight": 0.15,
            "cash_weight": 0.50,
            "duration_bias": "Neutral",
        }

    elif decision == "Divergence Warning":
        weights = {
            "equity_weight": 0.35,
            "credit_weight": 0.20,
            "cash_weight": 0.45,
            "duration_bias": "Neutral",
        }

    elif decision == "Avoid Chasing Risk":
        weights = {
            "equity_weight": 0.30,
            "credit_weight": 0.20,
            "cash_weight": 0.50,
            "duration_bias": "Neutral / Defensive",
        }

    elif decision == "Hold / Do Not Chase":
        weights = {
            "equity_weight": 0.40,
            "credit_weight": 0.25,
            "cash_weight": 0.35,
            "duration_bias": "Neutral",
        }

    elif decision == "Wait":
        weights = {
            "equity_weight": 0.30,
            "credit_weight": 0.20,
            "cash_weight": 0.50,
            "duration_bias": "Neutral",
        }

    elif decision == "Neutral":
        weights = {
            "equity_weight": 0.45,
            "credit_weight": 0.25,
            "cash_weight": 0.30,
            "duration_bias": "Neutral",
        }

    else:
        weights = {
            "equity_weight": 0.40,
            "credit_weight": 0.25,
            "cash_weight": 0.35,
            "duration_bias": "Neutral",
        }

    return normalize_weights(weights)