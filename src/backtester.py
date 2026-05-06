import numpy as np


def assign_strategy_return(row):
    if row["sp500_forward_30d_return"] != row["sp500_forward_30d_return"]:
        return None

    return (
        row["equity_weight"] * row["sp500_forward_30d_return"]
        + row["cash_weight"] * 0.002
    )


def build_strategy_backtest(df):
    df = df.copy()

    df["sp500_daily_return"] = df["sp500"].pct_change()

    def assign_strategy_weight(row):
        decision = row.get("final_decision", "Neutral")

        if decision in ["Buy Stress", "Watch Entry"]:
            return 1.00
        if decision == "Risk On":
            return 0.90
        if decision == "Neutral":
            return 0.70
        if decision in ["Hold / Do Not Chase", "Avoid Chasing Risk"]:
            return 0.45
        if decision in ["Credit Warning", "Reduce Risk"]:
            return 0.25
        if decision in ["Active Stress", "Wait"]:
            return 0.15

        return row.get("equity_weight", 0.50)

    df["strategy_weight"] = df.apply(assign_strategy_weight, axis=1)
    df["strategy_weight_lagged"] = df["strategy_weight"].shift(1)

    df["strategy_daily_return"] = (
        df["strategy_weight_lagged"] * df["sp500_daily_return"]
    )

    df["strategy_daily_return"] = df["strategy_daily_return"].fillna(0)
    df["sp500_daily_return"] = df["sp500_daily_return"].fillna(0)

    df["strategy_equity_curve"] = (1 + df["strategy_daily_return"]).cumprod()
    df["sp500_equity_curve"] = (1 + df["sp500_daily_return"]).cumprod()

    df["strategy_drawdown"] = (
        df["strategy_equity_curve"] / df["strategy_equity_curve"].cummax()
        - 1
    )

    df["sp500_backtest_drawdown"] = (
        df["sp500_equity_curve"] / df["sp500_equity_curve"].cummax()
        - 1
    )

    return df


def compute_backtest_summary(df):
    valid = df.dropna(subset=["strategy_forward_30d_return"])

    summary = {
        "avg_strategy_30d_return": valid["strategy_forward_30d_return"].mean(),
        "avg_sp500_30d_return": valid["sp500_forward_30d_return"].mean(),
        "strategy_hit_rate": (valid["strategy_forward_30d_return"] > 0).mean(),
        "sp500_hit_rate": (valid["sp500_forward_30d_return"] > 0).mean(),
        "strategy_worst_5pct": valid["strategy_forward_30d_return"].quantile(0.05),
        "sp500_worst_5pct": valid["sp500_forward_30d_return"].quantile(0.05),
    }

    if "strategy_equity_curve" in df.columns:
        summary.update({
            "strategy_total_return": df["strategy_equity_curve"].iloc[-1] - 1,
            "sp500_total_return": df["sp500_equity_curve"].iloc[-1] - 1,
            "strategy_volatility": df["strategy_daily_return"].std() * np.sqrt(252),
            "sp500_volatility": df["sp500_daily_return"].std() * np.sqrt(252),
            "strategy_max_drawdown": df["strategy_drawdown"].min(),
            "sp500_max_drawdown": df["sp500_backtest_drawdown"].min(),
        })

    return summary