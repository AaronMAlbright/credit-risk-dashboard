import pandas as pd


def check_portfolio_weights(row, tolerance=0.001):
    total = (
        row["equity_weight"]
        + row["credit_weight"]
        + row["cash_weight"]
    )

    if abs(total - 1.0) <= tolerance:
        return "PASS"

    return f"FAIL: weights sum to {total:.2%}"


def check_missing_values(df, required_columns):
    missing_report = {}

    for col in required_columns:
        if col not in df.columns:
            missing_report[col] = "MISSING COLUMN"
        else:
            missing_count = df[col].isna().sum()
            missing_report[col] = missing_count

    return missing_report


def check_sample_sizes(df, regime_col, min_obs=30):
    counts = df[regime_col].value_counts()
    weak = counts[counts < min_obs]

    if weak.empty:
        return "PASS"

    return weak.to_dict()


def check_score_bounds(df, score_columns):
    issues = {}

    for col in score_columns:
        if col not in df.columns:
            issues[col] = "MISSING COLUMN"
            continue

        below_zero = (df[col] < 0).sum()
        above_100 = (df[col] > 100).sum()

        if below_zero > 0 or above_100 > 0:
            issues[col] = {
                "below_zero": int(below_zero),
                "above_100": int(above_100),
            }

    return "PASS" if len(issues) == 0 else issues


def run_model_health_check(df):
    latest = df.iloc[-1]

    required_columns = [
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "cross_asset_divergence_score_smooth",
        "complacency_score_smooth",
        "mean_reversion_score_smooth",
        "final_decision",
        "equity_weight",
        "credit_weight",
        "cash_weight",
    ]

    score_columns = [
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "cross_asset_divergence_score_smooth",
        "market_internals_score_smooth",
        "risk_appetite_score_smooth",
        "complacency_score_smooth",
        "mean_reversion_score_smooth",
        "treasury_stress_score_smooth",
    ]

    return {
        "portfolio_weight_check": check_portfolio_weights(latest),
        "missing_values": check_missing_values(df, required_columns),
        "final_decision_sample_check": check_sample_sizes(df, "final_decision"),
        "score_bounds_check": check_score_bounds(df, score_columns),
    }


def print_model_health_check(health_check):
    print("\n=== MODEL HEALTH CHECK ===")
    print(f"Portfolio Weights: {health_check['portfolio_weight_check']}")

    print("\nMissing Values:")
    for key, value in health_check["missing_values"].items():
        print(f"{key}: {value}")

    print("\nFinal Decision Sample Sizes:")
    print(health_check["final_decision_sample_check"])

    print("\nScore Bounds:")
    print(health_check["score_bounds_check"])