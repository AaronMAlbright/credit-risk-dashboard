import matplotlib.pyplot as plt


def plot_macro_credit_complacency(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["macro_risk_score_smooth"], label="Macro Risk")
    plt.plot(df.index, df["credit_market_risk_score_smooth"], label="Credit Risk")
    plt.plot(df.index, df["complacency_score_smooth"], label="Complacency")
    plt.plot(df.index, df["mean_reversion_score_smooth"], label="Mean Reversion")
    plt.axhline(40)
    plt.axhline(70)
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Macro Risk vs Credit Risk vs Complacency vs Mean-Reversion")
    plt.savefig(f"{chart_dir}/macro_credit_complacency.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_hy_credit_spread(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["hy_spread"])
    plt.title("High-Yield Credit Spread")
    plt.savefig(f"{chart_dir}/hy_credit_spread.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_credit_impulse(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["credit_impulse"])
    plt.axhline(0)
    plt.title("Credit Impulse: 30D HY Change Minus Prior 30D Change")
    plt.savefig(f"{chart_dir}/credit_impulse.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_treasury_credit_macro(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["treasury_stress_score_smooth"], label="Treasury Stress")
    plt.plot(df.index, df["credit_market_risk_score_smooth"], label="Credit Risk")
    plt.plot(df.index, df["macro_risk_score_smooth"], label="Macro Risk")
    plt.axhline(40)
    plt.axhline(70)
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Treasury Stress vs Credit Risk vs Macro Risk")
    plt.savefig(f"{chart_dir}/treasury_credit_macro.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_cross_asset_liquidity(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["cross_asset_divergence_score_smooth"], label="Cross-Asset Divergence")
    plt.plot(df.index, df["liquidity_regime_score_smooth"], label="Liquidity Regime")
    plt.axhline(40)
    plt.axhline(70)
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Cross-Asset Divergence vs Liquidity Regime")
    plt.savefig(f"{chart_dir}/cross_asset_liquidity.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_sp500_drawdown(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["sp500_drawdown"] * 100)
    plt.axhline(0)
    plt.title("SP500 Drawdown (%)")
    plt.savefig(f"{chart_dir}/sp500_drawdown.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_risk_appetite_vs_complacency(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["risk_appetite_score_smooth"], label="Risk Appetite")
    plt.plot(df.index, df["complacency_score_smooth"], label="Complacency")
    plt.axhline(70)
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Risk Appetite vs Complacency")
    plt.savefig(f"{chart_dir}/risk_appetite_vs_complacency.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_composite_risk_score(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["composite_risk_score_smooth"], label="Composite Risk")
    plt.axhline(25)
    plt.axhline(50)
    plt.axhline(70)
    plt.ylim(0, 100)
    plt.legend()
    plt.title("Composite Macro Credit Risk Score")
    plt.savefig(f"{chart_dir}/composite_risk_score.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_composite_regime_timeline(df, chart_dir, show=False):
    regime_map = {
        "Risk-On": 1,
        "Neutral": 2,
        "Caution": 3,
        "Defensive / High Risk": 4,
    }
    regime_numeric = df["composite_risk_label"].map(regime_map)

    plt.figure(figsize=(10, 5))
    plt.step(df.index, regime_numeric, where="post", label="Composite Regime")
    plt.yticks([1, 2, 3, 4], ["Risk-On", "Neutral", "Caution", "Defensive"])
    plt.ylim(0.5, 4.5)
    plt.legend()
    plt.title("Composite Risk Regime Timeline")
    plt.savefig(f"{chart_dir}/composite_regime_timeline.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_backtest_equity_curve(df, chart_dir, show=False):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["strategy_equity_curve"], label="Regime Strategy")
    plt.plot(df.index, df["sp500_equity_curve"], label="SP500 Buy & Hold")
    plt.legend()
    plt.title("Backtest: Regime Strategy vs SP500")
    plt.savefig(f"{chart_dir}/backtest_equity_curve.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def generate_all_charts(df, chart_dir, show=False):
    plot_macro_credit_complacency(df, chart_dir, show)
    plot_hy_credit_spread(df, chart_dir, show)
    plot_credit_impulse(df, chart_dir, show)
    plot_treasury_credit_macro(df, chart_dir, show)
    plot_cross_asset_liquidity(df, chart_dir, show)
    plot_sp500_drawdown(df, chart_dir, show)
    plot_risk_appetite_vs_complacency(df, chart_dir, show)
    plot_composite_risk_score(df, chart_dir, show)
    plot_composite_regime_timeline(df, chart_dir, show)
    plot_backtest_equity_curve(df, chart_dir, show)
