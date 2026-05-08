from config import (
    OUTPUT_DATA_DIR,
    OUTPUT_REPORT_DIR,
    OUTPUT_CHART_DIR,
)

from src.market_data import load_all_series, get_freshness
from src.cross_asset_engine import (
    compute_rates_stress_score,
    compute_enhanced_funding_stress_score,
    compute_fx_commodity_score,
    compute_banking_stress_score,
    rolling_zscore,
)
from src.data_diagnostics import run_diagnostics, print_diagnostics

from src.risk_engine import (
    classify_yield_curve_regime,
    classify_credit_regime,
    classify_labor_warning,
    compute_macro_risk_score,
    compute_credit_market_risk_score,
    compute_liquidity_score,
    compute_risk_appetite_score,
    compute_complacency_score,
    compute_mean_reversion_score,
    classify_credit_equity_divergence,
    classify_vol_credit_mismatch,
    detect_shock,
    classify_risk_score,
    classify_score,
    classify_complacency_score,
    classify_mean_reversion_score,
    generate_macro_signal,
    generate_mean_reversion_signal,
    classify_transition_regime,
    generate_transition_signal,
    generate_final_decision,
    calculate_trigger_distances,
    get_signal_drivers,
    compute_model_confidence,
)

from src.treasury_engine import (
    compute_treasury_features,
    compute_treasury_stress_score,
)

from src.market_internals_engine import (
    compute_cross_asset_divergence_score,
    classify_cross_asset_divergence,
    compute_market_internals_score,
    classify_market_internals,
)

from src.liquidity_engine import (
    compute_liquidity_regime_score,
    classify_liquidity_regime,
    generate_liquidity_signal,
)

from src.scenario_engine import (
    simulate_shock,
    summarize_scenario,
)

from src.validation_engine import (
    validation_table_by_regime,
    print_correlation_block,
)

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

from src.utils import (
    ensure_dirs,
    latest_date,
    days_in_current_value,
    hit_rate,
    worst_5pct,
)

from src.run_logger import log_model_run
from src.composite_engine import build_composite_risk

import matplotlib.pyplot as plt


# =====================
# 0. Setup
# =====================
ensure_dirs([
    OUTPUT_DATA_DIR,
    OUTPUT_REPORT_DIR,
    OUTPUT_CHART_DIR,
])


# =====================
# 1. Pull Data
# =====================
# load_all_series fetches from FRED + yfinance with local parquet caching
# (12-hour TTL). SP500 sourced from yfinance (^GSPC) for full 1999+ history.
# T10YIE (10Y breakeven) pre-2003 is filled with 2.5% proxy.
_START_DATE = "1999-01-01"
_raw = load_all_series(start=_START_DATE)
print(f"[data] Loaded {len(_raw)} rows from {_raw.index[0].date()} to {_raw.index[-1].date()}")


# =====================
# 2. Data Freshness
# =====================
freshness = {
    "DGS10":        str(_raw["yield_10y"].dropna().index[-1].date()),
    "DGS2":         str(_raw["yield_2y"].dropna().index[-1].date()),
    "UNRATE":       str(_raw["unemployment"].dropna().index[-1].date()),
    "NFCI":         str(_raw["nfci"].dropna().index[-1].date()),
    "HY Spread":    str(_raw["hy_spread"].dropna().index[-1].date()),
    "SP500":        str(_raw["sp500"].dropna().index[-1].date()),
    "VIX":          str(_raw["vix"].dropna().index[-1].date()),
    "10Y Breakeven":str(_raw["breakeven_10y"].dropna().index[-1].date()),
}


# =====================
# 3. Build Dataset
# =====================
# Rename to match the column names the scoring engine expects.
# yield_10y/yield_2y → value_10y/value_2y are internal aliases from the old
# join-based loader; we adopt cleaner names throughout.
df = _raw.copy()
df["spread"] = df["yield_10y"] - df["yield_2y"]
df["yield_curve_regime"] = df["spread"].apply(classify_yield_curve_regime)

# nfci_90d_avg used by scoring engine
df["nfci_90d_avg"] = df["nfci"].rolling(90, min_periods=20).mean()

# Drop rows where all required core series are simultaneously missing
# (e.g. pre-market-open timestamps). Keep partial rows for optional series.
_core = ["yield_10y", "yield_2y", "vix", "hy_spread", "sp500", "unemployment", "nfci"]
df = df.dropna(subset=[c for c in _core if c in df.columns])


# =====================
# 4. Derived Features — Core (existing, unchanged)
# =====================
df["unemployment_change_90d"] = df["unemployment"].diff(90)
df["spread_change_90d"] = df["spread"].diff(90)
df["spread_change_5d"] = df["spread"].diff(5)

df["hy_change_30d"] = df["hy_spread"].diff(30)
df["hy_change_90d"] = df["hy_spread"].diff(90)
df["hy_change_5d"] = df["hy_spread"].diff(5)
df["hy_change_30d_prior"] = df["hy_change_30d"].shift(30)
df["credit_impulse"] = df["hy_change_30d"] - df["hy_change_30d_prior"]

df["nfci_change_90d"] = df["nfci_90d_avg"].diff(90)

df["vix_change_30d"] = df["vix"].diff(30)
df["vix_change_5d"] = df["vix"].diff(5)

df["sp500_return_5d"] = df["sp500"].pct_change(5)
df["sp500_return_30d"] = df["sp500"].pct_change(30)

df["unemployment_12m_low"] = df["unemployment"].rolling(252, min_periods=126).min()
df["sahm_like"] = df["unemployment"] - df["unemployment_12m_low"]

df["sp500_peak"] = df["sp500"].cummax()
df["sp500_drawdown"] = df["sp500"] / df["sp500_peak"] - 1

df = compute_treasury_features(df)


# =====================
# 4b. Derived Features — Cross-Asset (new)
# =====================

# Rates / Fed policy
if "fed_funds_rate" in df.columns:
    df["fed_funds_change_90d"]  = df["fed_funds_rate"].diff(90)
    df["fed_funds_change_360d"] = df["fed_funds_rate"].diff(360)

# 3M-10Y spread (better recession predictor than 10Y-2Y)
# Use precomputed FRED series where available; fallback to computed
if "spread_10y3m" in df.columns:
    df["spread_10y3m_change_30d"] = df["spread_10y3m"].diff(30)
elif "yield_3m" in df.columns:
    df["spread_10y3m"] = df["yield_10y"] - df["yield_3m"]
    df["spread_10y3m_change_30d"] = df["spread_10y3m"].diff(30)

# FX / commodity signals
if "eurusd" in df.columns:
    df["eurusd_change_30d"] = df["eurusd"].pct_change(30)
if "usdjpy" in df.columns:
    df["usdjpy_change_30d"] = df["usdjpy"].diff(30)
if "oil_wti" in df.columns:
    df["oil_change_30d"] = df["oil_wti"].pct_change(30)
    df["oil_change_90d"] = df["oil_wti"].pct_change(90)

# Banking / funding channel
if "bank_deposits" in df.columns:
    df["deposit_growth_90d"] = df["bank_deposits"].pct_change(90)
if "total_loans" in df.columns:
    df["loan_growth_90d"] = df["total_loans"].pct_change(90)
if "fed_balance_sheet" in df.columns:
    df["fed_bs_change_90d"] = df["fed_balance_sheet"].pct_change(90)

# Initial claims z-score (rolling 2Y window = 504 business days)
if "initial_claims" in df.columns:
    df["initial_claims_zscore"] = rolling_zscore(df["initial_claims"], window=504)

# Enhanced liquidity indices
if "anfci" in df.columns:
    df["anfci_change_30d"] = df["anfci"].diff(30)
if "stlfsi" in df.columns:
    df["stlfsi_change_30d"] = df["stlfsi"].diff(30)

# Drop rows missing any of the core SCORED columns needed by risk engine
# (90d diffs require at least 90 rows; 252d rolling requires 126 with min_periods)
_scored_required = [
    "spread", "unemployment", "nfci_90d_avg",
    "unemployment_change_90d", "spread_change_90d", "nfci_change_90d",
    "sahm_like", "hy_spread", "hy_change_30d", "hy_change_90d",
    "credit_impulse", "vix", "vix_change_30d",
    "sp500_return_5d", "sp500_return_30d", "sp500_drawdown",
    "real_yield_z", "real_yield_change_90d", "curve_steepening_velocity_90d",
]
df = df.dropna(subset=[c for c in _scored_required if c in df.columns])


# =====================
# 5. Regime / Divergence Features
# =====================
df["credit_regime"] = df.apply(
    lambda row: classify_credit_regime(row["spread"], row["unemployment"]),
    axis=1,
)

df["labor_warning"] = df.apply(
    lambda row: classify_labor_warning(
        row["sahm_like"],
        row["unemployment_change_90d"],
    ),
    axis=1,
)

df["credit_equity_divergence"] = df.apply(
    lambda row: classify_credit_equity_divergence(
        row["sp500_return_30d"],
        row["hy_change_30d"],
    ),
    axis=1,
)

df["vol_credit_mismatch"] = df.apply(
    lambda row: classify_vol_credit_mismatch(
        row["vix"],
        row["vix_change_30d"],
        row["hy_change_30d"],
    ),
    axis=1,
)

df["cross_asset_divergence_score"] = df.apply(
    lambda row: compute_cross_asset_divergence_score(
        row["sp500_return_30d"],
        row["sp500_drawdown"],
        row["hy_change_30d"],
        row["vix"],
        row["vix_change_30d"],
    ),
    axis=1,
)

df["market_internals_score"] = df.apply(
    lambda row: compute_market_internals_score(
        row["sp500_return_30d"],
        row["sp500_return_5d"],
        row["sp500_drawdown"],
        row["vix"],
        row["vix_change_30d"],
    ),
    axis=1,
)

df["shock_flag"] = df.apply(
    lambda row: detect_shock(
        row["vix_change_5d"],
        row["hy_change_5d"],
        row["sp500_return_5d"],
        row["spread_change_5d"],
    ),
    axis=1,
)


# =====================
# 6. Core Scores
# =====================
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


# =====================
# 7. Smooth Scores
# =====================
df["macro_risk_score_smooth"] = df["macro_risk_score"].rolling(21).mean()
df["credit_market_risk_score_smooth"] = df["credit_market_risk_score"].rolling(21).mean()
df["liquidity_score_smooth"] = df["liquidity_score"].rolling(21).mean()
df["liquidity_regime_score_smooth"] = df["liquidity_regime_score"].rolling(21).mean()
df["treasury_stress_score_smooth"] = df["treasury_stress_score"].rolling(21).mean()
df["cross_asset_divergence_score_smooth"] = df["cross_asset_divergence_score"].rolling(10).mean()
df["market_internals_score_smooth"] = df["market_internals_score"].rolling(10).mean()

# Only require the core smoothed scores — optional cross-asset columns may be NaN
_smooth_required = [
    "macro_risk_score_smooth", "credit_market_risk_score_smooth",
    "liquidity_score_smooth", "liquidity_regime_score_smooth",
    "treasury_stress_score_smooth",
    "cross_asset_divergence_score_smooth", "market_internals_score_smooth",
]
df = df.dropna(subset=_smooth_required)

df["macro_risk_momentum_10d"] = df["macro_risk_score_smooth"].diff(10)
df["credit_risk_momentum_10d"] = df["credit_market_risk_score_smooth"].diff(10)

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

df["risk_appetite_score_smooth"] = df["risk_appetite_score"].rolling(10).mean()
df["complacency_score_smooth"] = df["complacency_score"].rolling(10).mean()
df["mean_reversion_score_smooth"] = df["mean_reversion_score"].rolling(10).mean()

_second_order_required = [
    "risk_appetite_score_smooth", "complacency_score_smooth",
    "mean_reversion_score_smooth", "macro_risk_momentum_10d", "credit_risk_momentum_10d",
]
df = df.dropna(subset=_second_order_required)
df = build_composite_risk(df)


# =====================
# 7b. Cross-Asset Scores (informational — not wired into composite yet)
# =====================
# These are additive context signals. The composite formula is NOT changed here
# to avoid retrofitting weights to history. Weights will be tuned after OOS testing.

if "spread_10y3m" in df.columns and "fed_funds_rate" in df.columns:
    df["rates_stress_score"] = df.apply(
        lambda row: compute_rates_stress_score(
            row.get("spread_10y3m", float("nan")),
            row.get("fed_funds_rate", float("nan")),
            row.get("fed_funds_change_90d", float("nan")),
            row.get("fed_funds_change_360d", float("nan")),
        ),
        axis=1,
    )
    df["rates_stress_score_smooth"] = df["rates_stress_score"].rolling(21, min_periods=5).mean()

if "anfci" in df.columns or "stlfsi" in df.columns:
    df["enhanced_funding_stress_score"] = df.apply(
        lambda row: compute_enhanced_funding_stress_score(
            row.get("nfci_90d_avg", float("nan")),
            row.get("anfci", float("nan")),
            row.get("stlfsi", float("nan")),
            row.get("initial_claims_zscore", float("nan")),
        ),
        axis=1,
    )
    df["enhanced_funding_stress_score_smooth"] = (
        df["enhanced_funding_stress_score"].rolling(21, min_periods=5).mean()
    )

if "eurusd_change_30d" in df.columns or "oil_change_30d" in df.columns:
    df["fx_commodity_score"] = df.apply(
        lambda row: compute_fx_commodity_score(
            row.get("eurusd_change_30d", float("nan")),
            row.get("oil_change_30d", float("nan")),
            row.get("usdjpy_change_30d", float("nan")),
        ),
        axis=1,
    )
    df["fx_commodity_score_smooth"] = df["fx_commodity_score"].rolling(10, min_periods=3).mean()

if "deposit_growth_90d" in df.columns or "loan_growth_90d" in df.columns:
    df["banking_stress_score"] = df.apply(
        lambda row: compute_banking_stress_score(
            row.get("deposit_growth_90d", float("nan")),
            row.get("loan_growth_90d", float("nan")),
            row.get("fed_bs_change_90d", float("nan")),
        ),
        axis=1,
    )
    df["banking_stress_score_smooth"] = df["banking_stress_score"].rolling(21, min_periods=5).mean()


# =====================
# 8. Labels / Signals
# =====================
df["macro_risk_label"] = df["macro_risk_score_smooth"].apply(classify_risk_score)
df["credit_market_risk_label"] = df["credit_market_risk_score_smooth"].apply(classify_risk_score)
df["liquidity_label"] = df["liquidity_score_smooth"].apply(classify_score)
df["liquidity_regime_label"] = df["liquidity_regime_score_smooth"].apply(classify_liquidity_regime)
df["liquidity_signal"] = df["liquidity_regime_score_smooth"].apply(generate_liquidity_signal)
df["treasury_stress_label"] = df["treasury_stress_score_smooth"].apply(classify_score)
df["cross_asset_divergence_label"] = df["cross_asset_divergence_score_smooth"].apply(classify_cross_asset_divergence)
df["market_internals_label"] = df["market_internals_score_smooth"].apply(classify_market_internals)
df["risk_appetite_label"] = df["risk_appetite_score_smooth"].apply(classify_score)
df["complacency_label"] = df["complacency_score_smooth"].apply(classify_complacency_score)
df["mean_reversion_label"] = df["mean_reversion_score_smooth"].apply(classify_mean_reversion_score)

df["macro_signal"] = df["macro_risk_score_smooth"].apply(generate_macro_signal)
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


# =====================
# 9. Final Decision
# =====================
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

df["final_decision"] = df["final_decision_obj"].apply(lambda x: x["final_decision"])
df["final_environment"] = df["final_decision_obj"].apply(lambda x: x["environment"])
df["final_action"] = df["final_decision_obj"].apply(lambda x: x["action"])

# Simplified 5-bucket grouped regime for validation (more observations per bucket)
_GROUPED_REGIMES = {
    "Buy Stress":                   "Recovery / Buy Stress",
    "Watch Entry":                  "Risk-On",
    "Risk On":                      "Risk-On",
    "Neutral":                      "Neutral",
    "Stress / Stabilization Watch": "Caution",
    "Hold / Do Not Chase":          "Neutral",
    "Divergence Warning":           "Caution",
    "Wait":                         "Caution",
    "Credit Warning":               "Stress",
    "Avoid Chasing Risk":           "Stress",
    "Active Stress":                "Stress",
    "Reduce Risk":                  "Stress",
}
df["grouped_regime"] = df["final_decision"].map(_GROUPED_REGIMES).fillna("Neutral")


# =====================
# 10. Portfolio / Attribution / Analogs
# =====================
portfolio_weights = df.apply(generate_portfolio_weights, axis=1)
df["equity_weight"] = portfolio_weights.apply(lambda x: x["equity_weight"])
df["credit_weight"] = portfolio_weights.apply(lambda x: x["credit_weight"])
df["cash_weight"] = portfolio_weights.apply(lambda x: x["cash_weight"])
df["duration_bias"] = portfolio_weights.apply(lambda x: x["duration_bias"])

health_check = run_model_health_check(df)

df["signal_contributions"] = df.apply(get_signal_contributions, axis=1)
df["crisis_analogs"] = df.apply(compute_crisis_similarity, axis=1)


# =====================
# 11. Validation Metrics / Backtest
# =====================
df["sp500_forward_30d_return"] = df["sp500"].shift(-30) / df["sp500"] - 1
df["sp500_forward_60d_return"] = df["sp500"].shift(-60) / df["sp500"] - 1

df["hy_forward_30d_change"] = df["hy_spread"].shift(-30) - df["hy_spread"]
df["hy_forward_60d_change"] = df["hy_spread"].shift(-60) - df["hy_spread"]

df["sp500_future_min_30d"] = df["sp500"].shift(-1).rolling(30).min().shift(-29)
df["sp500_future_min_60d"] = df["sp500"].shift(-1).rolling(60).min().shift(-59)

df["sp500_future_drawdown_30d"] = df["sp500_future_min_30d"] / df["sp500"] - 1
df["sp500_future_drawdown_60d"] = df["sp500_future_min_60d"] / df["sp500"] - 1

df["strategy_forward_30d_return"] = df.apply(assign_strategy_return, axis=1)
df = build_strategy_backtest(df)
backtest_summary = compute_backtest_summary(df)

macro_threshold_75 = df["macro_risk_score_smooth"].quantile(0.75)
credit_threshold_75 = df["credit_market_risk_score_smooth"].quantile(0.75)
mean_rev_threshold_75 = df["mean_reversion_score_smooth"].quantile(0.75)
complacency_threshold_75 = df["complacency_score_smooth"].quantile(0.75)

high_macro_risk = df["macro_risk_score_smooth"] >= macro_threshold_75
high_credit_risk = df["credit_market_risk_score_smooth"] >= credit_threshold_75
high_mean_reversion = df["mean_reversion_score_smooth"] >= mean_rev_threshold_75
high_complacency = df["complacency_score_smooth"] >= complacency_threshold_75

decision_counts = df["final_decision"].value_counts().to_dict()

df["model_confidence_obj"] = df.apply(
    lambda row: compute_model_confidence(row, decision_counts),
    axis=1,
)
df["model_confidence"] = df["model_confidence_obj"].apply(lambda x: x[0])
df["model_confidence_reasons"] = df["model_confidence_obj"].apply(lambda x: x[1])


# =====================
# 12. Scenario Test
# =====================
latest = df.iloc[-1]
log_model_run(latest)
shock_scenario = simulate_shock(latest)
scenario_summary = summarize_scenario(latest, shock_scenario)


# =====================
# 13. Latest Snapshot Objects
# =====================
decision = latest["final_decision_obj"]
trigger_distances = calculate_trigger_distances(
    latest["vix"],
    latest["sp500_drawdown"],
    latest["hy_change_5d"],
)

days_current_transition = days_in_current_value(df["transition_regime"])
days_current_decision = days_in_current_value(df["final_decision"])
days_current_complacency = days_in_current_value(df["complacency_label"])

decision_validation = validation_table_by_regime(df, "final_decision")


# =====================
# 14. Print Output
# =====================
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

print("Avg Forward 30D SP500 Return | High Macro Risk:", f"{df.loc[high_macro_risk, 'sp500_forward_30d_return'].mean():.2%}")
print("Avg Forward 30D SP500 Return | High Credit Risk:", f"{df.loc[high_credit_risk, 'sp500_forward_30d_return'].mean():.2%}")
print("Avg Forward 30D SP500 Return | High Mean-Reversion:", f"{df.loc[high_mean_reversion, 'sp500_forward_30d_return'].mean():.2%}")
print("Avg Forward 30D SP500 Return | High Complacency:", f"{df.loc[high_complacency, 'sp500_forward_30d_return'].mean():.2%}")
print("Hit Rate 30D SP500 | High Complacency:", f"{hit_rate(df.loc[high_complacency, 'sp500_forward_30d_return']):.2%}")
print("Worst 5% 30D SP500 Return | High Complacency:", f"{worst_5pct(df.loc[high_complacency, 'sp500_forward_30d_return']):.2%}")

print("\n=== SIMPLE STRATEGY BACKTEST ===")
for key, value in backtest_summary.items():
    if value == value:
        if any(term in key for term in ["return", "rate", "worst", "drawdown", "volatility"]):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")

print("\n=== FINAL DECISION VALIDATION ===")
print(decision_validation)

print_correlation_block(df)

# Run data quality diagnostics and print summary
_diag = run_diagnostics(df)
print_diagnostics(_diag)

print(f"\n[data] Historical coverage: {_diag['date_range']['start']} → {_diag['date_range']['end']} "
      f"({_diag['n_rows']} rows)")

# Print cross-asset score snapshot if available
_xasset_cols = [c for c in [
    "rates_stress_score_smooth",
    "enhanced_funding_stress_score_smooth",
    "fx_commodity_score_smooth",
    "banking_stress_score_smooth",
] if c in df.columns]
if _xasset_cols:
    _lat = df.iloc[-1]
    print("\n=== CROSS-ASSET SCORES (informational) ===")
    for col in _xasset_cols:
        print(f"{col}: {_lat[col]:.1f}")


# =====================
# 15. Export Outputs
# =====================
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


# =====================
# 16. Plots
# =====================

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
plt.savefig(f"{OUTPUT_CHART_DIR}/macro_credit_complacency.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["hy_spread"])
plt.title("High-Yield Credit Spread")
plt.savefig(f"{OUTPUT_CHART_DIR}/hy_credit_spread.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["credit_impulse"])
plt.axhline(0)
plt.title("Credit Impulse: 30D HY Change Minus Prior 30D Change")
plt.savefig(f"{OUTPUT_CHART_DIR}/credit_impulse.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["treasury_stress_score_smooth"], label="Treasury Stress")
plt.plot(df.index, df["credit_market_risk_score_smooth"], label="Credit Risk")
plt.plot(df.index, df["macro_risk_score_smooth"], label="Macro Risk")
plt.axhline(40)
plt.axhline(70)
plt.ylim(0, 100)
plt.legend()
plt.title("Treasury Stress vs Credit Risk vs Macro Risk")
plt.savefig(f"{OUTPUT_CHART_DIR}/treasury_credit_macro.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["cross_asset_divergence_score_smooth"], label="Cross-Asset Divergence")
plt.plot(df.index, df["liquidity_regime_score_smooth"], label="Liquidity Regime")
plt.axhline(40)
plt.axhline(70)
plt.ylim(0, 100)
plt.legend()
plt.title("Cross-Asset Divergence vs Liquidity Regime")
plt.savefig(f"{OUTPUT_CHART_DIR}/cross_asset_liquidity.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["sp500_drawdown"] * 100)
plt.axhline(0)
plt.title("SP500 Drawdown (%)")
plt.savefig(f"{OUTPUT_CHART_DIR}/sp500_drawdown.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["risk_appetite_score_smooth"], label="Risk Appetite")
plt.plot(df.index, df["complacency_score_smooth"], label="Complacency")
plt.axhline(70)
plt.ylim(0, 100)
plt.legend()
plt.title("Risk Appetite vs Complacency")
plt.savefig(f"{OUTPUT_CHART_DIR}/risk_appetite_vs_complacency.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["composite_risk_score_smooth"], label="Composite Risk")
plt.axhline(25)
plt.axhline(50)
plt.axhline(70)
plt.ylim(0, 100)
plt.legend()
plt.title("Composite Macro Credit Risk Score")
plt.savefig(f"{OUTPUT_CHART_DIR}/composite_risk_score.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

regime_map = {
    "Risk-On": 1,
    "Neutral": 2,
    "Caution": 3,
    "Defensive / High Risk": 4,
}

df["composite_regime_numeric"] = df["composite_risk_label"].map(regime_map)

plt.figure(figsize=(10, 5))
plt.step(df.index, df["composite_regime_numeric"], where="post", label="Composite Regime")
plt.yticks([1, 2, 3, 4], ["Risk-On", "Neutral", "Caution", "Defensive"])
plt.ylim(0.5, 4.5)
plt.legend()
plt.title("Composite Risk Regime Timeline")
plt.savefig(f"{OUTPUT_CHART_DIR}/composite_regime_timeline.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["strategy_equity_curve"], label="Regime Strategy")
plt.plot(df.index, df["sp500_equity_curve"], label="SP500 Buy & Hold")
plt.legend()
plt.title("Backtest: Regime Strategy vs SP500")
plt.savefig(f"{OUTPUT_CHART_DIR}/backtest_equity_curve.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()