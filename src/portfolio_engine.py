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

    if decision == "Risk-On":
        weights = {
            "equity_weight": 0.65,
            "credit_weight": 0.25,
            "cash_weight": 0.10,
            "duration_bias": "Neutral",
        }

    elif decision == "Neutral":
        weights = {
            "equity_weight": 0.50,
            "credit_weight": 0.25,
            "cash_weight": 0.25,
            "duration_bias": "Neutral",
        }

    elif decision == "Caution":
        weights = {
            "equity_weight": 0.30,
            "credit_weight": 0.20,
            "cash_weight": 0.50,
            "duration_bias": "Neutral / Slight Long",
        }

    elif decision == "Risk-Off":
        weights = {
            "equity_weight": 0.15,
            "credit_weight": 0.10,
            "cash_weight": 0.75,
            "duration_bias": "Long Duration",
        }

    else:
        weights = {
            "equity_weight": 0.40,
            "credit_weight": 0.25,
            "cash_weight": 0.35,
            "duration_bias": "Neutral",
        }

    return normalize_weights(weights)