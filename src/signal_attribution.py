def get_signal_contributions(row):
    contributions = {}

    if row["hy_spread"] < 3:
        contributions["HY spread tightness"] = 18

    if row["hy_change_30d"] < -0.3:
        contributions["HY spread compression"] = 14

    if row["vix"] < 18:
        contributions["Low VIX"] = 15

    if row["vix_change_30d"] < -5:
        contributions["VIX compression"] = 12

    if row["sp500_drawdown"] > -0.02:
        contributions["SP500 near highs"] = 12

    if row["risk_appetite_score_smooth"] > 70:
        contributions["High risk appetite"] = 16

    if row["complacency_score_smooth"] > 70:
        contributions["Elevated complacency"] = 20

    if row["credit_market_risk_score_smooth"] > 40:
        contributions["Credit stress"] = 20

    if row["macro_risk_score_smooth"] > 40:
        contributions["Macro stress"] = 20

    if row["treasury_stress_score_smooth"] > 40:
        contributions["Treasury / real-rate stress"] = 16

    if row["shock_flag"] != "No Shock":
        contributions["Active shock flag"] = 25

    return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))


def format_top_contributions(contributions, limit=7):
    return list(contributions.items())[:limit]