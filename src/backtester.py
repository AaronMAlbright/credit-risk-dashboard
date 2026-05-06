def assign_strategy_return(row):
    if row["sp500_forward_30d_return"] != row["sp500_forward_30d_return"]:
        return None

    return (
        row["equity_weight"] * row["sp500_forward_30d_return"]
        + row["cash_weight"] * 0.002
    )


def compute_backtest_summary(df):
    valid = df.dropna(subset=["strategy_forward_30d_return"])

    return {
        "avg_strategy_30d_return": valid["strategy_forward_30d_return"].mean(),
        "avg_sp500_30d_return": valid["sp500_forward_30d_return"].mean(),
        "strategy_hit_rate": (valid["strategy_forward_30d_return"] > 0).mean(),
        "sp500_hit_rate": (valid["sp500_forward_30d_return"] > 0).mean(),
        "strategy_worst_5pct": valid["strategy_forward_30d_return"].quantile(0.05),
        "sp500_worst_5pct": valid["sp500_forward_30d_return"].quantile(0.05),
    }