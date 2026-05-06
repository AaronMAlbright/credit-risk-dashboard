def safe_corr(df, x_col, y_col):
    clean = df[[x_col, y_col]].dropna()

    if len(clean) < 20:
        return None

    return clean[x_col].corr(clean[y_col])


def validation_table_by_regime(df, regime_col):
    return (
        df.groupby(regime_col)
        .agg(
            observations=("sp500_forward_30d_return", "count"),
            avg_30d_return=("sp500_forward_30d_return", "mean"),
            hit_rate_30d=("sp500_forward_30d_return", lambda x: (x > 0).mean()),
            worst_5pct_30d=("sp500_forward_30d_return", lambda x: x.quantile(0.05)),
            avg_future_drawdown_30d=("sp500_future_drawdown_30d", "mean"),
        )
        .sort_values("observations", ascending=False)
    )


def print_correlation_block(df):
    checks = {
        "Macro Risk vs Forward 30D SP500 Return": ("macro_risk_score_smooth", "sp500_forward_30d_return"),
        "Credit Risk vs Forward 30D SP500 Return": ("credit_market_risk_score_smooth", "sp500_forward_30d_return"),
        "Liquidity Regime vs Forward 30D SP500 Return": ("liquidity_regime_score_smooth", "sp500_forward_30d_return"),
        "Cross-Asset Divergence vs Forward 30D SP500 Return": ("cross_asset_divergence_score_smooth", "sp500_forward_30d_return"),
        "Complacency vs Forward 30D SP500 Return": ("complacency_score_smooth", "sp500_forward_30d_return"),
        "Mean-Reversion vs Forward 30D SP500 Return": ("mean_reversion_score_smooth", "sp500_forward_30d_return"),
    }

    print("\n=== CORRELATION CHECKS ===")
    for label, cols in checks.items():
        corr = safe_corr(df, cols[0], cols[1])
        print(f"{label}: {corr:.2f}" if corr is not None else f"{label}: insufficient data")