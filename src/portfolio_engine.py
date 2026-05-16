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
            "hy_weight": 0.0,
            "ig_weight": 0.0,
            "duration_target": weights.get("duration_target", 6.0),
        }

    scale = 1.0 / total
    return {
        "equity_weight":  round(weights["equity_weight"]  * scale, 4),
        "credit_weight":  round(weights["credit_weight"]  * scale, 4),
        "cash_weight":    round(weights["cash_weight"]    * scale, 4),
        "duration_bias":  weights.get("duration_bias", "Neutral"),
        # hy/ig weights are absolute portfolio fractions (already fraction of total)
        "hy_weight":      round(weights.get("hy_weight", 0.0) * scale, 4),
        "ig_weight":      round(weights.get("ig_weight", 0.0) * scale, 4),
        "duration_target": weights.get("duration_target", 6.0),
    }


def generate_portfolio_weights(row):
    decision = row["final_decision"]

    if decision == "Risk-On":
        # Full risk appetite: heavy equity, HY-tilted credit, short duration
        weights = {
            "equity_weight":  0.65,
            "credit_weight":  0.25,
            "cash_weight":    0.10,
            "duration_bias":  "Neutral",
            "hy_weight":      0.175,   # 70% of credit in HY
            "ig_weight":      0.075,   # 30% of credit in IG
            "duration_target": 5.0,
        }

    elif decision == "Neutral":
        # Balanced: equal HY/IG split, moderate duration
        weights = {
            "equity_weight":  0.50,
            "credit_weight":  0.25,
            "cash_weight":    0.25,
            "duration_bias":  "Neutral",
            "hy_weight":      0.125,   # 50% HY
            "ig_weight":      0.125,   # 50% IG
            "duration_target": 6.0,
        }

    elif decision == "Caution":
        # Defensive tilt: reduce equity, rotate HY→IG, extend duration slightly
        weights = {
            "equity_weight":  0.30,
            "credit_weight":  0.20,
            "cash_weight":    0.50,
            "duration_bias":  "Neutral / Slight Long",
            "hy_weight":      0.04,    # 20% HY
            "ig_weight":      0.16,    # 80% IG
            "duration_target": 7.5,
        }

    elif decision == "Risk-Off":
        # Flight to quality: minimal equity, zero HY, all-IG credit, long duration
        weights = {
            "equity_weight":  0.15,
            "credit_weight":  0.10,
            "cash_weight":    0.75,
            "duration_bias":  "Long Duration",
            "hy_weight":      0.00,    # no HY exposure
            "ig_weight":      0.10,    # all credit in IG
            "duration_target": 8.5,
        }

    else:
        weights = {
            "equity_weight":  0.40,
            "credit_weight":  0.25,
            "cash_weight":    0.35,
            "duration_bias":  "Neutral",
            "hy_weight":      0.10,
            "ig_weight":      0.15,
            "duration_target": 6.0,
        }

    return normalize_weights(weights)