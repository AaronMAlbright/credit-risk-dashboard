import pandas as pd

from config import (
    OUTPUT_DATA_DIR,
    OUTPUT_REPORT_DIR,
    OUTPUT_CHART_DIR,
)

from src.fred_loader import get_series
from src.feature_engine import build_features
from src.charts import generate_all_charts

from src.risk_engine import (
    compute_macro_risk_score,
    compute_credit_market_risk_score,
    compute_liquidity_score,
    compute_risk_appetite_score,
    compute_complacency_score,
    compute_mean_reversion_score,
    classify_risk_score,
    classify_score,
    classify_complacency_score,
    classify_mean_reversion_score,
    generate_macro_signal,
    generate_mean_reversion_signal,
    classify_transition_regime,
    generate_transition_signal,
    classify_regime_band,
    generate_final_decision,
    calculate_trigger_distances,
    get_signal_drivers,
    compute_model_confidence,
)

from src.treasury_engine import compute_treasury_stress_score

from src.market_internals_engine import (
    classify_cross_asset_divergence,
    classify_market_internals,
)

from src.liquidity_engine import (
    compute_liquidity_regime_score,
    classify_liquidity_regime,
    generate_liquidity_signal,
)

from src.composite_engine import build_composite_risk
from src.portfolio_engine import generate_portfolio_weights

from src.model_health_check import (
    run_model_health_check,
    print_model_health_check,
)

from src.signal_attribution import (
    get_signal_contributions,
    format_top_contributions,
)

from src.crisis_similarity import compute_crisis_similarity

from src.backtester import (
    assign_strategy_return,
    build_strategy_backtest,
    compute_backtest_summary,
)

from src.scenario_engine import simulate_shock, summarize_scenario

from src.validation_engine import (
    validation_table_by_regime,
    print_correlation_block,
)

from src.run_logger import log_model_run

from src.utils import (
    ensure_dirs,
    latest_date,
    days_in_current_value,
    hit_rate,
    worst_5pct,
)


def fetch_data() -> tuple:
    """
    Pull all required FRED series. Returns (series_dict, freshness).
    series_dict keys match the expected input of feature_engine.build_features().
    """
    ten_year      = get_series("DGS10")
    two_year      = get_series("DGS2")
    unrate        = get_series("UNRATE")
    nfci          = get_series("NFCI")
    hy_spread     = get_series("BAMLH0A0HYM2")
    sp500         = get_series("SP500")
    vix           = get_series("VIXCLS")
    breakeven_10y = get_series("T10YIE")

    series_dict = {
        "DGS10":        ten_year,
        "DGS2":         two_year,
        "UNRATE":       unrate,
        "NFCI":         nfci,
        "BAMLH0A0HYM2": hy_spread,
        "SP500":        sp500,
        "VIXCLS":       vix,
        "T10YIE":       breakeven_10y,
    }

    freshness = {
        "DGS10":         latest_date(ten_year),
        "DGS2":          latest_date(two_year),
        "UNRATE":        latest_date(unrate),
        "NFCI":          latest_date(nfci),
        "HY Spread":     latest_date(hy_spread),
        "SP500":         latest_date(sp500),
        "VIX":           latest_date(vix),
        "10Y Breakeven": latest_date(breakeven_10y),
    }

    return series_dict, freshness


def run_scoring_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all risk scores, smooth them, generate labels and signals,
    classify transition regimes, and produce the final decision for every row.

    Input:  featured DataFrame from feature_engine.build_features()
    Output: same DataFrame with all score, label, signal, and decision columns added
    """
    df = df.copy()

    # -------------------------
    # Core scores
    # -------------------------
    df["macro_risk_score"] = df.apply(
        lambda row: compute_macro_risk_score(
            row["spread"],
            row["unemployment"],
            row["nfci_90d_avg"],
            row["unemployment_change_90d"],
            row["spread_change_90d"],
            row["nfci_change_90d"],
            row["sahm_like"],
        ),
        axis=1,
    )

    df["credit_market_risk_score"] = df.apply(
        lambda row: compute_credit_market_risk_score(
            row["hy_spread"],
            row["hy_change_30d"],
            row["hy_change_90d"],
            row["credit_impulse"],
            row["vix"],
            row["vix_change_30d"],
            row["credit_equity_divergence"],
            row["vol_credit_mismatch"],
        ),
        axis=1,
    )

    df["liquidity_score"] = df.apply(
        lambda row: compute_liquidity_score(
            row["nfci_90d_avg"],
            row["nfci_change_90d"],
            row["hy_change_30d"],
            row["vix_change_30d"],
        ),
        axis=1,
    )

    df["treasury_stress_score"] = df.apply(
        lambda row: compute_treasury_stress_score(
            row["real_yield_z"],
            row["real_yield_change_90d"],
            row["curve_steepening_velocity_90d"],
        ),
        axis=1,
    )

    df["liquidity_regime_score"] = df.apply(
        lambda row: compute_liquidity_regime_score(
            row["nfci_90d_avg"],
            row["nfci_change_90d"],
            row["hy_change_30d"],
            row["vix_change_30d"],
            row["treasury_stress_score"],
        ),
        axis=1,
    )

    # -------------------------
    # Smooth core scores
    # -------------------------
    df["macro_risk_score_smooth"]              = df["macro_risk_score"].rolling(21).mean()
    df["credit_market_risk_score_smooth"]      = df["credit_market_risk_score"].rolling(21).mean()
    df["liquidity_score_smooth"]               = df["liquidity_score"].rolling(21).mean()
    df["liquidity_regime_score_smooth"]        = df["liquidity_regime_score"].rolling(21).mean()
    df["treasury_stress_score_smooth"]         = df["treasury_stress_score"].rolling(21).mean()
    df["cross_asset_divergence_score_smooth"]  = df["cross_asset_divergence_score"].rolling(10).mean()
    df["market_internals_score_smooth"]        = df["market_internals_score"].rolling(10).mean()

    df = df.dropna()

    # -------------------------
    # Momentum
    # -------------------------
    df["macro_risk_momentum_10d"]   = df["macro_risk_score_smooth"].diff(10)
    df["credit_risk_momentum_10d"]  = df["credit_market_risk_score_smooth"].diff(10)

    # -------------------------
    # Second-order scores (depend on smoothed first-order scores)
    # -------------------------
    df["risk_appetite_score"] = df.apply(
        lambda row: compute_risk_appetite_score(
            row["hy_change_30d"],
            row["vix_change_30d"],
            row["sp500_drawdown"],
            row["macro_risk_momentum_10d"],
            row["credit_risk_momentum_10d"],
        ),
        axis=1,
    )

    df["complacency_score"] = df.apply(
        lambda row: compute_complacency_score(
            row["vix"],
            row["hy_spread"],
            row["sp500_drawdown"],
            row["macro_risk_momentum_10d"],
            row["nfci_90d_avg"],
            row["hy_change_30d"],
            row["risk_appetite_score"],
            row["market_internals_score_smooth"],
        ),
        axis=1,
    )

    df["mean_reversion_score"] = df.apply(
        lambda row: compute_mean_reversion_score(
            row["macro_risk_score_smooth"],
            row["credit_market_risk_score_smooth"],
            row["hy_spread"],
            row["hy_change_30d"],
            row["vix"],
            row["vix_change_30d"],
            row["sp500_drawdown"],
        ),
        axis=1,
    )

    df["risk_appetite_score_smooth"]  = df["risk_appetite_score"].rolling(10).mean()
    df["complacency_score_smooth"]    = df["complacency_score"].rolling(10).mean()
    df["mean_reversion_score_smooth"] = df["mean_reversion_score"].rolling(10).mean()

    df = df.dropna()
    df = build_composite_risk(df)

    # -------------------------
    # Labels
    # -------------------------
    df["macro_risk_label"]            = df["macro_risk_score_smooth"].apply(classify_risk_score)
    df["credit_market_risk_label"]    = df["credit_market_risk_score_smooth"].apply(classify_risk_score)
    df["liquidity_label"]             = df["liquidity_score_smooth"].apply(classify_score)
    df["liquidity_regime_label"]      = df["liquidity_regime_score_smooth"].apply(classify_liquidity_regime)
    df["liquidity_signal"]            = df["liquidity_regime_score_smooth"].apply(generate_liquidity_signal)
    df["treasury_stress_label"]       = df["treasury_stress_score_smooth"].apply(classify_score)
    df["cross_asset_divergence_label"]= df["cross_asset_divergence_score_smooth"].apply(classify_cross_asset_divergence)
    df["market_internals_label"]      = df["market_internals_score_smooth"].apply(classify_market_internals)
    df["risk_appetite_label"]         = df["risk_appetite_score_smooth"].apply(classify_score)
    df["complacency_label"]           = df["complacency_score_smooth"].apply(classify_complacency_score)
    df["mean_reversion_label"]        = df["mean_reversion_score_smooth"].apply(classify_mean_reversion_score)

    # -------------------------
    # Signals
    # -------------------------
    df["macro_signal"]          = df["macro_risk_score_smooth"].apply(generate_macro_signal)
    df["mean_reversion_signal"] = df["mean_reversion_score_smooth"].apply(generate_mean_reversion_signal)

    df["combined_stress_momentum_10d"] = (
        df[["macro_risk_momentum_10d", "credit_risk_momentum_10d"]].mean(axis=1)
    )

    df["transition_regime"] = df.apply(
        lambda row: classify_transition_regime(
            row["macro_risk_score_smooth"],
            row["credit_market_risk_score_smooth"],
            row["combined_stress_momentum_10d"],
        ),
        axis=1,
    )

    df["transition_signal"] = df["transition_regime"].apply(generate_transition_signal)

    # -------------------------
    # Final decision
    # -------------------------
    df["final_decision_obj"] = df.apply(
        lambda row: generate_final_decision(
            row["macro_risk_score_smooth"],
            row["credit_market_risk_score_smooth"],
            row["liquidity_regime_score_smooth"],
            row["cross_asset_divergence_score_smooth"],
            row["complacency_score_smooth"],
            row["mean_reversion_score_smooth"],
            row["transition_regime"],
            row["shock_flag"],
            row["credit_equity_divergence"],
            row["vol_credit_mismatch"],
            row["composite_risk_score_smooth"],
        ),
        axis=1,
    )

    # Preserve granular 9-label signal as diagnostic detail
    df["final_decision_detail"] = df["final_decision_obj"].apply(lambda x: x["final_decision"])
    df["final_environment"]     = df["final_decision_obj"].apply(lambda x: x["environment"])
    df["final_action"]          = df["final_decision_obj"].apply(lambda x: x["action"])

    # Primary 4-regime output: composite-score bands (statistically grounded)
    df["final_decision"] = df.apply(
        lambda row: classify_regime_band(
            row["composite_risk_score_smooth"],
            row["shock_flag"],
            max(row["macro_risk_score_smooth"],
                row["credit_market_risk_score_smooth"],
                row["liquidity_regime_score_smooth"]),
        ),
        axis=1,
    )

    return df


def run_analytics(df: pd.DataFrame) -> tuple:
    """
    Compute portfolio weights, signal attribution, crisis analogs,
    forward-return validation labels, backtest, and model confidence.

    Returns (df, health_check, backtest_summary).
    """
    df = df.copy()

    # -------------------------
    # Portfolio weights
    # -------------------------
    portfolio_weights   = df.apply(generate_portfolio_weights, axis=1)
    df["equity_weight"] = portfolio_weights.apply(lambda x: x["equity_weight"])
    df["credit_weight"] = portfolio_weights.apply(lambda x: x["credit_weight"])
    df["cash_weight"]   = portfolio_weights.apply(lambda x: x["cash_weight"])
    df["duration_bias"] = portfolio_weights.apply(lambda x: x["duration_bias"])

    health_check = run_model_health_check(df)

    # -------------------------
    # Attribution and analogs
    # -------------------------
    df["signal_contributions"] = df.apply(get_signal_contributions, axis=1)
    df["crisis_analogs"]       = df.apply(compute_crisis_similarity, axis=1)

    # -------------------------
    # Forward return labels
    # -------------------------
    df["sp500_forward_30d_return"] = df["sp500"].shift(-30) / df["sp500"] - 1
    df["sp500_forward_60d_return"] = df["sp500"].shift(-60) / df["sp500"] - 1

    df["hy_forward_30d_change"] = df["hy_spread"].shift(-30) - df["hy_spread"]
    df["hy_forward_60d_change"] = df["hy_spread"].shift(-60) - df["hy_spread"]

    df["sp500_future_min_30d"] = df["sp500"].shift(-1).rolling(30).min().shift(-29)
    df["sp500_future_min_60d"] = df["sp500"].shift(-1).rolling(60).min().shift(-59)

    df["sp500_future_drawdown_30d"] = df["sp500_future_min_30d"] / df["sp500"] - 1
    df["sp500_future_drawdown_60d"] = df["sp500_future_min_60d"] / df["sp500"] - 1

    # -------------------------
    # Backtest
    # -------------------------
    df["strategy_forward_30d_return"] = df.apply(assign_strategy_return, axis=1)
    df = build_strategy_backtest(df)
    backtest_summary = compute_backtest_summary(df)

    # -------------------------
    # Model confidence
    # -------------------------
    decision_counts = df["final_decision"].value_counts().to_dict()

    df["model_confidence_obj"]     = df.apply(
        lambda row: compute_model_confidence(row, decision_counts),
        axis=1,
    )
    df["model_confidence"]         = df["model_confidence_obj"].apply(lambda x: x[0])
    df["model_confidence_reasons"] = df["model_confidence_obj"].apply(lambda x: x[1])

    return df, health_check, backtest_summary


def run_outputs(
    df: pd.DataFrame,
    freshness: dict,
    health_check: dict,
    backtest_summary: dict,
) -> None:
    """
    Print the full signal report to stdout, write the scored CSV,
    and write the text report file. All snapshot objects are derived
    locally from df so this function has no side effects on the DataFrame.
    """
    latest   = df.iloc[-1]
    decision = latest["final_decision_obj"]

    trigger_distances = calculate_trigger_distances(
        latest["vix"],
        latest["sp500_drawdown"],
        latest["hy_change_5d"],
    )

    days_current_transition  = days_in_current_value(df["transition_regime"])
    days_current_decision    = days_in_current_value(df["final_decision"])
    days_current_complacency = days_in_current_value(df["complacency_label"])

    shock_scenario   = simulate_shock(latest)
    scenario_summary = summarize_scenario(latest, shock_scenario)

    decision_validation = validation_table_by_regime(df, "final_decision")

    macro_threshold_75      = df["macro_risk_score_smooth"].quantile(0.75)
    credit_threshold_75     = df["credit_market_risk_score_smooth"].quantile(0.75)
    mean_rev_threshold_75   = df["mean_reversion_score_smooth"].quantile(0.75)
    complacency_threshold_75= df["complacency_score_smooth"].quantile(0.75)

    high_macro_risk    = df["macro_risk_score_smooth"]         >= macro_threshold_75
    high_credit_risk   = df["credit_market_risk_score_smooth"] >= credit_threshold_75
    high_mean_reversion= df["mean_reversion_score_smooth"]     >= mean_rev_threshold_75
    high_complacency   = df["complacency_score_smooth"]        >= complacency_threshold_75

    # -------------------------
    # Print
    # -------------------------
    print("\n=== DATA FRESHNESS ===")
    for name, dt in freshness.items():
        print(f"{name}: {dt}")

    print("\n=== CURRENT MACRO / CREDIT SNAPSHOT ===")
    print(f"Spread: {latest['spread']:.2f}")
    print(f"Yield Curve Regime: {latest['yield_curve_regime']}")
    print(f"Unemployment: {latest['unemployment']:.1f}")
    print(f"Labor Warning: {latest['labor_warning']}")
    print(f"Sahm-like Labor Stress: {latest['sahm_like']:.2f}")
    print(f"NFCI 90D Avg: {latest['nfci_90d_avg']:.2f}")
    print(f"NFCI 90D Change: {latest['nfci_change_90d']:.2f}")
    print(f"HY Spread: {latest['hy_spread']:.2f}")
    print(f"30D HY Spread Change: {latest['hy_change_30d']:.2f}")
    print(f"90D HY Spread Change: {latest['hy_change_90d']:.2f}")
    print(f"Credit Impulse: {latest['credit_impulse']:.2f}")
    print(f"VIX: {latest['vix']:.2f}")
    print(f"30D VIX Change: {latest['vix_change_30d']:.2f}")
    print(f"SP500: {latest['sp500']:.2f}")
    print(f"SP500 30D Return: {latest['sp500_return_30d']:.2%}")
    print(f"SP500 Drawdown: {latest['sp500_drawdown']:.2%}")
    print(f"Credit / Equity Divergence: {latest['credit_equity_divergence']}")
    print(f"Vol / Credit Mismatch: {latest['vol_credit_mismatch']}")
    print(f"Shock Flag: {latest['shock_flag']}")

    print("\n=== TREASURY / RATES STRESS ===")
    print(f"10Y Breakeven: {latest['breakeven_10y']:.2f}")
    print(f"Real Yield Proxy: {latest['real_yield_proxy']:.2f}")
    print(f"Real Yield Z-Score: {latest['real_yield_z']:.2f}")
    print(f"Real Yield 90D Change: {latest['real_yield_change_90d']:.2f}")
    print(f"Treasury Stress Score: {latest['treasury_stress_score_smooth']:.1f} ({latest['treasury_stress_label']})")

    print("\n=== SIGNAL SNAPSHOT ===")
    print(f"Composite Risk Score: {latest['composite_risk_score_smooth']:.1f} ({latest['composite_risk_label']})")
    print(f"Macro Risk Score: {latest['macro_risk_score_smooth']:.1f} ({latest['macro_risk_label']})")
    print(f"Credit Market Risk Score: {latest['credit_market_risk_score_smooth']:.1f} ({latest['credit_market_risk_label']})")
    print(f"Liquidity Regime Score: {latest['liquidity_regime_score_smooth']:.1f} ({latest['liquidity_regime_label']})")
    print(f"Liquidity Signal: {latest['liquidity_signal']}")
    print(f"Cross-Asset Divergence Score: {latest['cross_asset_divergence_score_smooth']:.1f} ({latest['cross_asset_divergence_label']})")
    print(f"Market Internals Score: {latest['market_internals_score_smooth']:.1f} ({latest['market_internals_label']})")
    print(f"Risk Appetite Score: {latest['risk_appetite_score_smooth']:.1f} ({latest['risk_appetite_label']})")
    print(f"Complacency Score: {latest['complacency_score_smooth']:.1f} ({latest['complacency_label']})")
    print(f"Mean-Reversion Score: {latest['mean_reversion_score_smooth']:.1f} ({latest['mean_reversion_label']})")
    print(f"Macro Risk Momentum 10D: {latest['macro_risk_momentum_10d']:.1f}")
    print(f"Credit Risk Momentum 10D: {latest['credit_risk_momentum_10d']:.1f}")
    print(f"Transition Regime: {latest['transition_regime']}")
    print(f"Transition Signal: {latest['transition_signal']}")

    print("\n=== PORTFOLIO STANCE ===")
    print(f"Equity Weight: {latest['equity_weight']:.0%}")
    print(f"Credit Weight: {latest['credit_weight']:.0%}")
    print(f"Cash Weight: {latest['cash_weight']:.0%}")
    print(f"Duration Bias: {latest['duration_bias']}")

    print_model_health_check(health_check)

    print("\n=== REGIME DURATION ===")
    print(f"Days in Current Transition Regime: {days_current_transition}")
    print(f"Days in Current Final Decision: {days_current_decision}")
    print(f"Days in Current Complacency Label: {days_current_complacency}")

    print("\n=== FINAL DECISION ===")
    print(f"Decision: {decision['final_decision']}")
    print(f"Environment: {decision['environment']}")
    print(f"Action: {decision['action']}")
    print(f"Buy Trigger: {decision['buy_trigger']}")
    print(f"Risk-Off Trigger: {decision['risk_off_trigger']}")

    print("\n=== SIGNAL ATTRIBUTION ===")
    for name, value in format_top_contributions(latest["signal_contributions"]):
        print(f"{name}: +{value}")

    print("\n=== TOP SIGNAL DRIVERS ===")
    for driver in get_signal_drivers(latest):
        print(f"- {driver}")

    print("\n=== HISTORICAL ANALOGS ===")
    for name, similarity in latest["crisis_analogs"][:3]:
        print(f"{name}: {similarity}% similarity")

    print("\n=== MODEL CONFIDENCE ===")
    print(f"Confidence: {latest['model_confidence']}")
    for reason in latest["model_confidence_reasons"]:
        print(f"- {reason}")

    print("\n=== TRIGGER DISTANCES ===")
    print(f"VIX points to 25: {trigger_distances['vix_to_25']:.2f}")
    print(f"SP500 drawdown needed to reach -5%: {trigger_distances['sp500_to_5pct_drawdown']:.2%}")
    print(f"HY 5D widening needed to reach +0.25: {trigger_distances['hy_5d_to_025_widening']:.2f}")

    print("\n=== SCENARIO SHOCK TEST ===")
    print(f"Base VIX: {scenario_summary['base_vix']:.2f} -> Shock VIX: {scenario_summary['shock_vix']:.2f}")
    print(f"Base HY Spread: {scenario_summary['base_hy_spread']:.2f} -> Shock HY Spread: {scenario_summary['shock_hy_spread']:.2f}")
    print(f"Base SP500 Drawdown: {scenario_summary['base_sp500_drawdown']:.2%} -> Shock Drawdown: {scenario_summary['shock_sp500_drawdown']:.2%}")

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Macro Risk 75th Percentile Threshold: {macro_threshold_75:.1f}")
    print(f"Credit Risk 75th Percentile Threshold: {credit_threshold_75:.1f}")
    print(f"Mean-Reversion 75th Percentile Threshold: {mean_rev_threshold_75:.1f}")
    print(f"Complacency 75th Percentile Threshold: {complacency_threshold_75:.1f}")

    print("Avg Forward 30D SP500 Return | High Macro Risk:",    f"{df.loc[high_macro_risk,     'sp500_forward_30d_return'].mean():.2%}")
    print("Avg Forward 30D SP500 Return | High Credit Risk:",   f"{df.loc[high_credit_risk,    'sp500_forward_30d_return'].mean():.2%}")
    print("Avg Forward 30D SP500 Return | High Mean-Reversion:",f"{df.loc[high_mean_reversion, 'sp500_forward_30d_return'].mean():.2%}")
    print("Avg Forward 30D SP500 Return | High Complacency:",   f"{df.loc[high_complacency,    'sp500_forward_30d_return'].mean():.2%}")
    print("Hit Rate 30D SP500 | High Complacency:",  f"{hit_rate(df.loc[high_complacency, 'sp500_forward_30d_return']):.2%}")
    print("Worst 5% 30D SP500 Return | High Complacency:", f"{worst_5pct(df.loc[high_complacency, 'sp500_forward_30d_return']):.2%}")

    print("\n=== SIMPLE STRATEGY BACKTEST ===")
    for key, value in backtest_summary.items():
        if value == value:  # preserves original NaN guard
            if any(term in key for term in ["return", "rate", "worst", "drawdown", "volatility"]):
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")

    print("\n=== FINAL DECISION VALIDATION ===")
    print(decision_validation)

    print_correlation_block(df)

    # -------------------------
    # Export
    # -------------------------
    df.to_csv(f"{OUTPUT_DATA_DIR}/scored_macro_credit_data.csv")

    report = f"""
MACRO CREDIT SIGNAL REPORT

Current Date: {df.index[-1]}

FINAL DECISION
Decision: {decision['final_decision']}
Environment: {decision['environment']}
Action: {decision['action']}
Buy Trigger: {decision['buy_trigger']}
Risk-Off Trigger: {decision['risk_off_trigger']}

PORTFOLIO STANCE
Equity Weight: {latest['equity_weight']:.0%}
Credit Weight: {latest['credit_weight']:.0%}
Cash Weight: {latest['cash_weight']:.0%}
Duration Bias: {latest['duration_bias']}

KEY SCORES
Composite Risk Score: {latest['composite_risk_score_smooth']:.1f}
Composite Regime: {latest['composite_risk_label']}
Macro Risk Score: {latest['macro_risk_score_smooth']:.1f}
Credit Market Risk Score: {latest['credit_market_risk_score_smooth']:.1f}
Liquidity Regime Score: {latest['liquidity_regime_score_smooth']:.1f}
Treasury Stress Score: {latest['treasury_stress_score_smooth']:.1f}
Cross-Asset Divergence Score: {latest['cross_asset_divergence_score_smooth']:.1f}
Market Internals Score: {latest['market_internals_score_smooth']:.1f}
Risk Appetite Score: {latest['risk_appetite_score_smooth']:.1f}
Complacency Score: {latest['complacency_score_smooth']:.1f}
Mean-Reversion Score: {latest['mean_reversion_score_smooth']:.1f}

MARKET STATE
Credit / Equity Divergence: {latest['credit_equity_divergence']}
Vol / Credit Mismatch: {latest['vol_credit_mismatch']}
Transition Regime: {latest['transition_regime']}
Shock Flag: {latest['shock_flag']}
Labor Warning: {latest['labor_warning']}

MODEL HEALTH
Portfolio Weight Check: {health_check['portfolio_weight_check']}

TOP ATTRIBUTIONS
{chr(10).join([f"- {name}: +{value}" for name, value in format_top_contributions(latest["signal_contributions"])])}

HISTORICAL ANALOGS
{chr(10).join([f"- {name}: {similarity}% similarity" for name, similarity in latest["crisis_analogs"][:3]])}

MODEL CONFIDENCE
Confidence: {latest['model_confidence']}
{chr(10).join(["- " + reason for reason in latest["model_confidence_reasons"]])}
"""

    with open(f"{OUTPUT_REPORT_DIR}/latest_signal_report.txt", "w") as f:
        f.write(report)


def run(show_charts: bool = False) -> pd.DataFrame:
    """
    Full pipeline entry point. Equivalent to running the original app.py.
    Stages run in order; each returns an updated df for the next stage.
    """
    ensure_dirs([OUTPUT_DATA_DIR, OUTPUT_REPORT_DIR, OUTPUT_CHART_DIR])

    series_dict, freshness = fetch_data()
    df = build_features(series_dict)
    df = run_scoring_pipeline(df)
    df, health_check, backtest_summary = run_analytics(df)

    log_model_run(df.iloc[-1])

    run_outputs(df, freshness, health_check, backtest_summary)
    generate_all_charts(df, OUTPUT_CHART_DIR, show=show_charts)

    return df
