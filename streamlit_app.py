import os
import pandas as pd
import streamlit as st
from pathlib import Path

# Inject Streamlit Cloud secrets into os.environ so all downstream
# os.environ.get() calls (fred_loader, alert_engine, etc.) just work.
for _k, _v in st.secrets.items():
    if isinstance(_v, str):
        os.environ.setdefault(_k, _v)

from src.report_generator import generate_html_report, generate_excel_report
from src.data_pipeline import check_api_key, run_pipeline, fetch_source_dates
from src.bootstrap import run_bootstrap_analysis
from src.regime_attribution import COMPOSITE_WEIGHTS, DISPLAY_NAMES, SCORE_COLS, run_regime_attribution
from src.regime_charts import (
    build_decision_timeline,
    build_score_history,
    build_sp500_with_regime_overlay,
)
from src.regime_transition import run_regime_analysis
from src.signal_decay import HORIZONS, run_signal_decay
from src.score_orthogonality import run_orthogonality_analysis, VIF_HIGH, VIF_MODERATE
from src.weight_optimizer import (
    CURRENT_WEIGHTS,
    run_weight_optimization,
)
from src.tail_risk import run_tail_risk_analysis
from src.stress_episodes import STRESS_EPISODES, run_stress_analysis
from src.performance_scorecard import run_performance_scorecard
from src.factor_exposure import run_factor_analysis
from src.regime_probability import run_regime_probability
from src.monte_carlo import run_monte_carlo
from src.subperiod_attribution import run_subperiod_attribution
from src.position_sizing import REGIME_WEIGHTS, SCORE_BREAKPOINTS, run_position_sizing
from src.scenario_analysis import (
    DEFAULT_SHOCKS,
    SCENARIO_PRESETS,
    run_scenario,
    run_scenario_grid,
    build_tornado_data,
)
from src.portfolio_engine import generate_portfolio_weights
from src.crisis_similarity import CRISIS_ANALOGS, compute_crisis_similarity
from src.signal_validation import (
    validate_signals_vs_returns,
    validate_signals_multi_horizon,
    compute_stress_episode_stats,
    SIGNAL_ROLES,
)
from src.model_health_check import (
    check_missing_values,
    check_sample_sizes,
    check_score_bounds,
)
from src.regime_validity import run_regime_validity
from src.confirmation_engine import get_current_confirmation, run_confirmation_series, DOMAIN_LABELS
from src.failure_analysis import run_failure_analysis
from src.economic_ontology import get_ontology_df
from src.macro_chronology import get_events_df, get_fed_cycles_df
from src.walk_forward import run_frozen_splits, FROZEN_SPLITS, _CAVEAT as _OOS_CAVEAT
from src.regime_transition import compute_current_regime_forecast
from src.blended_allocation import (
    PER_REGIME_EQUITY, SMOOTH_HALFLIFE, TARGET_VOL, ONE_WAY_COST,
    run_blended_allocation,
)
from src.threshold_robustness import (
    DEFAULT_THRESHOLDS, SHIFT_GRID,
    run_threshold_robustness,
)
from src.factor_exposure import compute_performance_attribution
from src.validation_guard import (
    CONFIDENCE_EXPLORATORY,
    CONFIDENCE_INDICATIVE,
    CONFIDENCE_ROBUST,
    CONFIDENCE_SIGILS,
    run_validation_audit,
)
from src.alert_engine import (
    AlertConfig,
    config_from_env,
    extract_current_state,
    format_alert_email,
    load_alert_state,
    run_alerts,
)
from src.llm_briefing import generate_morning_briefing
from src.credit_cycle_clock import build_credit_cycle_clock
from src.historical_analogs import find_historical_analogs, get_analog_summary
from src.live_snapshot import get_live_snapshot
from src.stress_contagion import run_contagion_analysis
from src.regime_persistence import run_persistence_analysis
from src.drawdown_attribution import run_drawdown_attribution
from src.taylor_rule import run_taylor_analysis
from src.recession_model import run_recession_analysis
from src.credit_curves import run_quality_curve_analysis
from src.default_forecaster import run_default_analysis
from src.dv01 import run_dv01_analysis
from src.conditional_var import run_cvar_analysis
from src.merton import run_merton_analysis
from src.efficient_frontier import compute_efficient_frontier
from src.kelly import run_kelly_analysis
from src.granger import run_granger_analysis
from src.move_index import get_move_snapshot, run_move_analysis
from src.spread_term_structure import run_term_structure_analysis
from src.rolling_correlation_regime import run_correlation_analysis
from src.forward_simulation import run_forward_simulation
from src.cdx_proxy import run_cdx_analysis
from src.fed_sentiment import run_fed_sentiment
from src.snapshot_pdf import generate_snapshot_bytes
from src.vix_term_structure import run_vix_term_analysis
from src.options_skew import run_skew_analysis
from src.regime_return_table import run_regime_return_analysis
from src.default_cycle import run_default_cycle_analysis
from src.carry_breakeven import run_breakeven_analysis
from src.comparison_mode import get_available_dates, compare_dates, format_comparison_table
from src.real_rates import run_real_rates_analysis
from src.correlation_heatmap import run_correlation_heatmap_analysis
from src.spread_volatility import run_spread_volatility_analysis
from src.fallen_angel import run_fallen_angel_analysis
from src.em_credit import run_em_credit_analysis
from src.macro_nowcast import run_macro_nowcast
from src.vrp import run_vrp_analysis
from src.credit_momentum import run_credit_momentum_analysis
from src.funding_stress import run_funding_stress_analysis
from src.global_credit import run_global_credit_analysis
from src.corporate_leverage import run_corporate_leverage_analysis
from src.seasonality import run_seasonality_analysis
from src.signal_traffic_light import run_traffic_light_analysis
from src.shock_simulator import run_shock_analysis, get_default_shocks
from src.alert_backtest import run_alert_backtest
from src.pca_decomposition import run_pca_analysis
from src.regime_forecast import run_regime_forecast
from src.custom_composite import run_custom_composite_analysis, SUB_SIGNALS as CUSTOM_COMPOSITE_SIGNALS
from src.cross_asset_momentum import run_cross_asset_momentum
from src.vol_regime_composite import run_vol_regime_composite
from src.credit_quality_migration import run_credit_quality_migration
from src.macro_surprise_index import run_macro_surprise_index
from src.loan_market_monitor import run_loan_market_monitor
from src.regime_duration import run_regime_duration
from src.systematic_deleveraging import run_systematic_deleveraging
from src.inflation_regime import run_inflation_regime
from src.sector_divergence import run_sector_divergence
from src.put_call_sentiment import run_put_call_sentiment
from src.credit_basis import run_credit_basis
from src.drawdown_recovery import run_drawdown_recovery
from src.signal_move_attribution import run_signal_move_attribution
from src.risk_parity_allocation import run_risk_parity_allocation
from src.tail_dependency import run_tail_dependency
from src.fed_liquidity import run_fed_liquidity
from src.g4_divergence import run_g4_divergence
from src.portfolio_stress_test import run_portfolio_stress_test, DEFAULT_PORTFOLIO, ASSET_LABELS, SHOCK_SCENARIOS as PST_SCENARIOS
from src.at1_coco_monitor import run_at1_coco_monitor
from src.swap_spread_monitor import run_swap_spread_monitor
from src.cross_currency_basis import run_cross_currency_basis
from src.cre_stress import run_cre_stress
from src.primary_market_issuance import run_primary_market_issuance
from src.distressed_debt import run_distressed_debt_analysis

st.set_page_config(
    page_title="Macro Credit Risk Dashboard",
    page_icon="📉",
    layout="wide",
)


def _sample_flag(n: int) -> str:
    if n < 20:  return "Exploratory"
    if n < 50:  return "Indicative"
    return "Reliable"

def _sample_flag_color(n: int) -> str:
    if n < 20:  return "#e74c3c"
    if n < 50:  return "#e67e22"
    return "#27ae60"

def _sample_badge(n: int) -> str:
    label = _sample_flag(n)
    color = _sample_flag_color(n)
    return (
        f'<span style="font-size:0.68rem;color:{color};border:1px solid {color};'
        f'border-radius:3px;padding:1px 5px;margin-left:6px;vertical-align:middle">'
        f'{label} (n={n})</span>'
    )

st.markdown("""
<style>
/* ── Hide Streamlit chrome ───────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Global typography ───────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }

/* ── Page title ──────────────────────────────────────────────────────── */
h1 { font-size: 1.5rem !important; font-weight: 700 !important;
     letter-spacing: -0.3px; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important;
     color: #c8ccd4 !important; margin-top: 1.5rem !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important;
     color: #9aa0aa !important; text-transform: uppercase;
     letter-spacing: .6px; margin-top: 1.2rem !important; }

/* ── Metric cards ────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #151820;
    border: 1px solid #232838;
    border-radius: 8px;
    padding: 14px 18px !important;
    transition: border-color 0.15s;
}
[data-testid="stMetric"]:hover {
    border-color: #2d3550 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: .6px;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Tabs ────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    gap: 2px;
    background: #0e1117;
    border-bottom: 1px solid #1e2435;
}
[data-baseweb="tab"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    color: #6b7280 !important;
    border-radius: 6px 6px 0 0 !important;
    background: transparent !important;
}
[aria-selected="true"] {
    color: #e2e8f0 !important;
    background: #151820 !important;
    border-bottom: 2px solid #4f8ef7 !important;
}
[data-baseweb="tab-highlight"] { background-color: #4f8ef7 !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0b0e16 !important;
    border-right: 1px solid #1e2435;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #0e1117;
}

/* ── Divider ─────────────────────────────────────────────────────────── */
hr { border-color: #1e2435 !important; }

/* ── Dataframe ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Buttons ─────────────────────────────────────────────────────────── */
[data-testid="baseButton-secondary"] {
    background: #151820 !important;
    border: 1px solid #2d3550 !important;
    color: #c8ccd4 !important;
    border-radius: 6px !important;
}
[data-testid="baseButton-primary"] {
    border-radius: 6px !important;
}

/* ── Download button ─────────────────────────────────────────────────── */
[data-testid="baseButton-secondary"]:hover {
    border-color: #4f8ef7 !important;
    color: #e2e8f0 !important;
}

/* ── Caption / small text ────────────────────────────────────────────── */
.stCaption, small { color: #4b5563 !important; font-size: 0.75rem !important; }

/* ── Expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1e2435 !important;
    border-radius: 8px !important;
    background: #0e1117 !important;
}

/* ── Top padding reduction ───────────────────────────────────────────── */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path("data/scored_macro_credit_data.csv")
HISTORY_PATH = Path("history/model_run_history.csv")
REPORT_PATH = Path("outputs/reports/latest_signal_report.txt")
CHART_DIR = Path("outputs/charts")
WF_WINDOWS_PATH = Path("outputs/validation/walk_forward_windows.csv")
WF_REGIMES_PATH = Path("outputs/validation/walk_forward_regimes.csv")
SENS_RESULTS_PATH = Path("outputs/sensitivity/sensitivity_results.csv")
SENS_REPORT_PATH  = Path("outputs/sensitivity/sensitivity_report.txt")
SENS_HEATMAP_DIR  = Path("outputs/sensitivity/heatmaps")
REGIME_TRANS_DIR  = Path("outputs/regime_transition")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_history():
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


@st.cache_data
def load_walk_forward():
    if WF_WINDOWS_PATH.exists() and WF_REGIMES_PATH.exists():
        windows_df = pd.read_csv(WF_WINDOWS_PATH, parse_dates=["train_end", "test_start", "test_end"])
        regimes_df = pd.read_csv(WF_REGIMES_PATH, parse_dates=["train_end", "test_start", "test_end"])
        return windows_df, regimes_df
    return None, None


@st.cache_data
def load_sensitivity():
    if SENS_RESULTS_PATH.exists():
        return pd.read_csv(SENS_RESULTS_PATH)
    return None


@st.cache_data
def load_regime_transition(_df):
    """Run (or reload) regime transition analysis from the scored dataset."""
    return run_regime_analysis(_df)


@st.cache_data
def load_bootstrap(_df, _windows_df):
    """Run bootstrap CI analysis (cached against df hash)."""
    return run_bootstrap_analysis(_df, windows_df=_windows_df, n_boot=1000)


@st.cache_data
def load_attribution(_df):
    """Run regime attribution analysis (cached against df hash)."""
    return run_regime_attribution(_df)


@st.cache_data
def load_tail_risk(_df):
    """Run tail risk analysis (cached)."""
    return run_tail_risk_analysis(_df)


@st.cache_data
def load_weight_optimization(_df):
    """Run weight optimisation (2000 Monte Carlo samples, cached)."""
    return run_weight_optimization(_df, n_samples=2000)


@st.cache_data
def load_orthogonality(_df):
    """Run score orthogonality analysis (cached)."""
    return run_orthogonality_analysis(_df)


@st.cache_data
def load_signal_decay(_df):
    """Run signal decay analysis (cached)."""
    return run_signal_decay(_df)


@st.cache_data
def load_stress_analysis(_df):
    """Run stress episode analysis (cached)."""
    return run_stress_analysis(_df, STRESS_EPISODES)


@st.cache_data
def load_performance_scorecard(_df):
    """Run performance scorecard (cached)."""
    return run_performance_scorecard(_df)


@st.cache_data
def load_factor_analysis(_df):
    """Run factor exposure analysis (cached)."""
    return run_factor_analysis(_df)


@st.cache_data
def load_regime_probability(_df):
    """Run regime probability nowcast (cached)."""
    return run_regime_probability(_df)


@st.cache_data
def load_subperiod_attribution(_df):
    """Run rolling sub-period attribution (cached)."""
    return run_subperiod_attribution(_df)


@st.cache_data
def load_monte_carlo(_df, _current_probs_key: str):
    """Run Monte Carlo simulation (cached; key encodes the start-probs hash)."""
    rp = run_regime_probability(_df)
    cur_probs = rp.get("current", {}).get("probs") or {}
    return run_monte_carlo(_df, current_probs=cur_probs or None)


@st.cache_data
def load_scenario_grid(_df):
    """Run scenario grid over all presets (cached)."""
    try:
        from src.regime_probability import fit_naive_bayes
        model = fit_naive_bayes(_df)
    except Exception:
        model = {}
    grid    = run_scenario_grid(_df, SCENARIO_PRESETS, model)
    tornado = build_tornado_data(_df, model)
    return {"model": model, "grid": grid, "tornado": tornado}


@st.cache_data
def load_position_sizing(_df):
    """Run position sizing analysis (cached)."""
    rp = run_regime_probability(_df)
    prob_history = rp.get("history")
    return run_position_sizing(_df, prob_history=prob_history)


@st.cache_data
def load_signal_validation(_df, _oos_cutoff="2016-01-01"):
    """Run IS/OOS signal validation against forward returns (cached)."""
    return validate_signals_vs_returns(_df, oos_cutoff=_oos_cutoff)


@st.cache_data
def load_signal_horizon_grid(_df, _oos_cutoff="2016-01-01"):
    """Compute multi-horizon Spearman correlation grid (cached)."""
    return validate_signals_multi_horizon(_df, oos_cutoff=_oos_cutoff)


@st.cache_data
def load_stress_episode_stats(_df):
    """Compute per-signal stats for named stress episodes (cached)."""
    return compute_stress_episode_stats(_df)


@st.cache_data
def load_blended_allocation(_df):
    """Run probability-blended allocation (cached)."""
    from src.regime_probability import compute_prob_history
    try:
        prob_history = compute_prob_history(_df)
    except Exception:
        prob_history = None
    return run_blended_allocation(_df, prob_history=prob_history)


@st.cache_data
def load_threshold_robustness(_df):
    """Run threshold robustness stress-test (cached)."""
    return run_threshold_robustness(_df)


@st.cache_data
def load_performance_attribution(_df):
    """Run performance attribution decomposition (cached)."""
    return compute_performance_attribution(_df)


@st.cache_data
def load_regime_validity(_df):
    """Run regime validity tests (cached)."""
    return run_regime_validity(_df)


@st.cache_data
def load_failure_analysis(_df):
    """Run failure analysis (cached)."""
    return run_failure_analysis(_df)


@st.cache_data
def load_confirmation_series(_df):
    """Run cross-asset confirmation series (cached)."""
    return run_confirmation_series(_df)


@st.cache_data
def load_frozen_splits(_df):
    """Run frozen OOS split evaluation (cached)."""
    return run_frozen_splits(_df)


@st.cache_data
def load_regime_forecast(_df):
    """Compute current regime transition forecast (cached)."""
    return compute_current_regime_forecast(_df)


@st.cache_data
def load_llm_briefing(_df):
    """Generate LLM morning briefing (cached by date via briefing_cache.json)."""
    return generate_morning_briefing(_df)


@st.cache_data
def load_historical_analogs(_df):
    """Find top-5 historical analog periods (cached)."""
    return find_historical_analogs(_df)


@st.cache_data
def load_contagion(_df):
    """Compute stress contagion matrix and index (cached)."""
    return run_contagion_analysis(_df)


@st.cache_data
def load_persistence(_df):
    """Compute regime persistence / dwell-time survival (cached)."""
    return run_persistence_analysis(_df)


@st.cache_data
def load_drawdown_attribution(_df):
    """Run drawdown attribution across sizing components (cached)."""
    try:
        from src.backtester import build_strategy_backtest, compute_benchmark_returns
        _bt = compute_benchmark_returns(build_strategy_backtest(_df))
        return run_drawdown_attribution(_bt)
    except Exception as _e:
        return {"drawdowns": [], "avg_component_drag": {}, "error": str(_e)}


@st.cache_data
def load_taylor(_df):
    """Run Taylor Rule analysis (cached)."""
    return run_taylor_analysis(_df)


@st.cache_data
def load_recession(_df):
    """Run Estrella-Mishkin recession probability model (cached)."""
    return run_recession_analysis(_df)


@st.cache_data
def load_quality_curve(_df):
    """Run credit quality curve analysis (cached)."""
    return run_quality_curve_analysis(_df)


@st.cache_data
def load_default_analysis(_df):
    """Run Jarrow-Turnbull default rate analysis (cached)."""
    return run_default_analysis(_df)


@st.cache_data
def load_dv01(_df):
    """Run DV01 / spread sensitivity analysis (cached)."""
    return run_dv01_analysis(_df)


@st.cache_data
def load_cvar(_df):
    """Run CVaR by regime analysis (cached)."""
    return run_cvar_analysis(_df)


@st.cache_data
def load_merton(_df):
    """Run Merton Distance-to-Default analysis (cached)."""
    return run_merton_analysis(_df)


@st.cache_data
def load_efficient_frontier(_df):
    """Run efficient frontier Monte Carlo (cached)."""
    return compute_efficient_frontier(_df)


@st.cache_data
def load_kelly(_df):
    """Run Kelly criterion sizing analysis (cached)."""
    return run_kelly_analysis(_df)


@st.cache_data
def load_granger(_df):
    """Run Granger causality tests (cached)."""
    return run_granger_analysis(_df)


@st.cache_data
def load_move(_df):
    """Run MOVE index analysis (cached)."""
    return run_move_analysis(_df)


@st.cache_data
def load_term_structure(_df):
    """Run credit/rates term structure analysis (cached)."""
    return run_term_structure_analysis(_df)


@st.cache_data
def load_correlation_regime(_df):
    """Run equity-credit rolling correlation regime analysis (cached)."""
    return run_correlation_analysis(_df)


@st.cache_data
def load_forward_simulation(_df):
    """Run regime-conditioned forward simulation fan chart (cached)."""
    return run_forward_simulation(_df, n_sim=500, seed=42)


@st.cache_data
def load_cdx_proxy(_df):
    """Run synthetic CDX proxy analysis (cached)."""
    return run_cdx_analysis(_df)


@st.cache_data(ttl=3600)
def load_fed_sentiment(_df):
    """Run Fed statement sentiment scoring (cached 1h — fetches live from Fed website)."""
    return run_fed_sentiment(_df)


@st.cache_data(ttl=3600)
def load_vix_term(_df):
    """Run VIX term structure analysis (cached 1h — fetches live VIX3M data)."""
    return run_vix_term_analysis(_df)


@st.cache_data(ttl=3600)
def load_options_skew(_df):
    """Run SKEW/VVIX options skew analysis (cached 1h — fetches live data)."""
    return run_skew_analysis(_df)


@st.cache_data
def load_regime_return_table(_df):
    """Compute regime-conditional return table."""
    return run_regime_return_analysis(_df)


@st.cache_data
def load_default_cycle(_df):
    """Run default cycle positioning analysis."""
    return run_default_cycle_analysis(_df)


@st.cache_data
def load_carry_breakeven(_df):
    """Compute carry breakeven analysis."""
    return run_breakeven_analysis(_df)


@st.cache_data
def load_real_rates(_df):
    """Compute real rate decomposition (nominal - breakeven)."""
    return run_real_rates_analysis(_df)


@st.cache_data
def load_correlation_heatmap(_df):
    """Compute rolling 90d cross-asset correlation matrix."""
    return run_correlation_heatmap_analysis(_df)


@st.cache_data
def load_spread_volatility(_df):
    """Compute rolling HY/IG spread volatility and GARCH estimate."""
    return run_spread_volatility_analysis(_df)


@st.cache_data
def load_fallen_angel(_df):
    """Compute fallen angel risk (HY/IG ratio, BBB-BB differential)."""
    return run_fallen_angel_analysis(_df)


@st.cache_data(ttl=3600)
def load_em_credit(_df):
    """Run EM credit stress analysis (cached 1h — fetches live EMB/HYG data)."""
    return run_em_credit_analysis(_df)


@st.cache_data
def load_macro_nowcast(_df):
    """Compute macro GDP nowcast from weekly/monthly indicators."""
    return run_macro_nowcast(_df)


@st.cache_data
def load_vrp(_df):
    """Compute Volatility Risk Premium (VIX minus realized vol)."""
    return run_vrp_analysis(_df)


@st.cache_data
def load_credit_momentum(_df):
    """Compute HY/IG spread momentum at 1M/3M/6M horizons."""
    return run_credit_momentum_analysis(_df)


@st.cache_data
def load_funding_stress(_df):
    """Compute TED/OIS funding stress proxy."""
    return run_funding_stress_analysis(_df)


@st.cache_data(ttl=3600)
def load_global_credit(_df):
    """Run global credit divergence analysis (cached 1h — fetches live HYG/HYXU data)."""
    return run_global_credit_analysis(_df)


@st.cache_data
def load_corporate_leverage(_df):
    """Compute corporate leverage cycle signal."""
    return run_corporate_leverage_analysis(_df)


@st.cache_data
def load_seasonality(_df):
    """Compute historical credit spread seasonality."""
    return run_seasonality_analysis(_df)


@st.cache_data
def load_traffic_light(_df):
    return run_traffic_light_analysis(_df)


@st.cache_data
def load_shock_analysis(_df):
    return run_shock_analysis(_df)


@st.cache_data
def load_alert_backtest(_df):
    return run_alert_backtest(_df)


@st.cache_data
def load_pca_analysis(_df):
    return run_pca_analysis(_df)


@st.cache_data
def load_regime_forecast(_df):
    return run_regime_forecast(_df)


@st.cache_data
def load_custom_composite(_df):
    return run_custom_composite_analysis(_df)


@st.cache_data
def load_cross_asset_momentum(_df):
    return run_cross_asset_momentum(_df)


@st.cache_data(ttl=3600)
def load_vol_regime_composite(_df):
    return run_vol_regime_composite(_df)


@st.cache_data
def load_credit_quality_migration(_df):
    return run_credit_quality_migration(_df)


@st.cache_data
def load_macro_surprise_index(_df):
    return run_macro_surprise_index(_df)


@st.cache_data(ttl=3600)
def load_loan_market_monitor(_df):
    return run_loan_market_monitor(_df)


@st.cache_data
def load_regime_duration(_df):
    return run_regime_duration(_df)


@st.cache_data
def load_systematic_deleveraging(_df):
    return run_systematic_deleveraging(_df)


@st.cache_data
def load_inflation_regime(_df):
    return run_inflation_regime(_df)


@st.cache_data(ttl=3600)
def load_sector_divergence(_df):
    return run_sector_divergence(_df)


@st.cache_data(ttl=3600)
def load_put_call_sentiment(_df):
    return run_put_call_sentiment(_df)


@st.cache_data(ttl=3600)
def load_credit_basis(_df):
    return run_credit_basis(_df)


@st.cache_data
def load_drawdown_recovery(_df):
    return run_drawdown_recovery(_df)


@st.cache_data
def load_signal_move_attribution(_df):
    return run_signal_move_attribution(_df)


@st.cache_data
def load_risk_parity_allocation(_df):
    return run_risk_parity_allocation(_df)


@st.cache_data
def load_tail_dependency(_df):
    return run_tail_dependency(_df)


@st.cache_data(ttl=3600)
def load_fed_liquidity(_df):
    return run_fed_liquidity(_df)


@st.cache_data(ttl=3600)
def load_g4_divergence(_df):
    return run_g4_divergence(_df)


@st.cache_data
def load_portfolio_stress_test(_df):
    return run_portfolio_stress_test(_df)


@st.cache_data(ttl=3600)
def load_at1_coco_monitor(_df):
    return run_at1_coco_monitor(_df)


@st.cache_data(ttl=3600)
def load_swap_spread_monitor(_df):
    return run_swap_spread_monitor(_df)


@st.cache_data(ttl=3600)
def load_cross_currency_basis(_df):
    return run_cross_currency_basis(_df)


@st.cache_data(ttl=3600)
def load_cre_stress(_df):
    return run_cre_stress(_df)


@st.cache_data(ttl=3600)
def load_primary_market_issuance(_df):
    return run_primary_market_issuance(_df)


@st.cache_data
def load_distressed_debt(_df):
    return run_distressed_debt_analysis(_df)


@st.cache_data
def load_validation_audit(_df, _windows_df, _transition_counts):
    """Run full validation audit (cached)."""
    return run_validation_audit(_df, windows_df=_windows_df, transition_counts=_transition_counts)


@st.cache_data
def load_report_html(_df, _audit, _decay, _tail, _weight_opt, _attr, _stress,
                     _regime_prob=None, _position_sizing=None,
                     _scenario_grid=None, _subperiod_table=None, _monte_carlo=None):
    """Generate the HTML signal report (cached against df hash)."""
    return generate_html_report(
        _df, audit=_audit, decay_results=_decay,
        tail_risk=_tail, weight_opt=_weight_opt, attr=_attr, stress=_stress,
        regime_prob=_regime_prob, position_sizing=_position_sizing,
        scenario_grid=_scenario_grid, subperiod_table=_subperiod_table,
        monte_carlo=_monte_carlo,
    )


df = load_data()
history = load_history()
latest = df.iloc[-1]

# ── Sidebar: data status + live refresh ──────────────────────────────────────
_last_date = pd.to_datetime(df["date"].max()).normalize()
_today = pd.Timestamp.now().normalize()
_bdays_stale = max(0, len(pd.bdate_range(_last_date + pd.Timedelta(days=1), _today)))
_last_date_str = _last_date.strftime("%Y-%m-%d")

st.sidebar.title("Data Status")
if _bdays_stale == 0:
    st.sidebar.success(f"Current — {_last_date_str}")
elif _bdays_stale == 1:
    st.sidebar.warning(f"1 trading day stale — {_last_date_str}")
else:
    st.sidebar.error(f"{_bdays_stale} trading days stale — {_last_date_str}")
st.sidebar.caption(f"Dataset rows: {len(df):,}")

# API key status
_key_info = check_api_key()
if _key_info["available"]:
    st.sidebar.caption(f"FRED key: {_key_info['key_preview']}")
else:
    st.sidebar.caption("⚠ FRED_API_KEY not set — refresh disabled")

st.sidebar.divider()

# Refresh button
_refresh_disabled = not _key_info["available"]
if st.sidebar.button("⟳ Refresh Data", disabled=_refresh_disabled,
                     help="Fetch latest FRED data and re-score the model (takes ~30–60s)"):
    with st.sidebar:
        with st.spinner("Running pipeline…"):
            _result = run_pipeline()
    if _result["success"]:
        st.sidebar.success(
            f"Updated to {_result['csv_last_date']} "
            f"in {_result['elapsed_s']:.0f}s"
        )
        st.cache_data.clear()
        st.rerun()
    else:
        st.sidebar.error(f"Refresh failed: {_result['error']}")
        if _result.get("stderr"):
            with st.sidebar.expander("Error details"):
                st.code(_result["stderr"][-800:])

# Source freshness expander (slow — only loads when expanded)
with st.sidebar.expander("Source freshness"):
    if st.button("Check FRED freshness", key="check_freshness"):
        with st.spinner("Querying FRED…"):
            _src = fetch_source_dates()
        for sid, info in _src.items():
            if not info["available"]:
                st.write(f"❌ {info['label']}: unavailable")
            elif info["days_stale"] == 0:
                st.write(f"✓ {info['label']}: {info['last_date']}")
            elif info["days_stale"] <= 5:
                st.write(f"⚠ {info['label']}: {info['last_date']} ({info['days_stale']}d stale)")
            else:
                st.write(f"⚠⚠ {info['label']}: {info['last_date']} ({info['days_stale']}d stale)")

st.sidebar.divider()

# ── Sidebar: Alert engine ─────────────────────────────────────────────────────
_ALERT_STATE_PATH = Path("history/alert_state.json")
_prev_alert_state = load_alert_state(_ALERT_STATE_PATH)

st.sidebar.subheader("Alerts")

_prev_decision = _prev_alert_state.get("decision", "")
_prev_date     = _prev_alert_state.get("last_date", "")
if _prev_decision:
    st.sidebar.caption(
        f"Last run: {_prev_alert_state.get('last_run','—')}\n\n"
        f"Regime: **{_prev_decision}**\n\n"
        f"Data date: {_prev_date}"
    )
else:
    st.sidebar.caption("No previous alert state found.")

_alert_col1, _alert_col2 = st.sidebar.columns(2)

if _alert_col1.button("Test Alert", key="sidebar_test_alert",
                       help="Run alert checks in dry-run mode (no email sent)"):
    with st.sidebar:
        with st.spinner("Checking…"):
            _sizing_for_alert = load_position_sizing(df)
            _alert_cfg = AlertConfig(dry_run=True, min_level="INFO")
            _alert_result = run_alerts(df, config=_alert_cfg,
                                       sizing=_sizing_for_alert,
                                       state_path=_ALERT_STATE_PATH)
    _fired = _alert_result["alerts"]
    if _fired:
        for _a in _fired:
            _lv = _a["level"]
            _msg = _a["message"]
            if _lv == "ALERT":
                st.sidebar.error(f"**{_lv}** — {_msg}")
            elif _lv == "WARNING":
                st.sidebar.warning(f"**{_lv}** — {_msg}")
            else:
                st.sidebar.info(f"**{_lv}** — {_msg}")
    else:
        st.sidebar.success("No alerts triggered.")

if _alert_col2.button("Send Alert", key="sidebar_send_alert",
                       help="Run alert checks and send email if triggered"):
    with st.sidebar:
        with st.spinner("Sending…"):
            _sizing_for_alert = load_position_sizing(df)
            _alert_cfg = config_from_env()
            _alert_result = run_alerts(df, config=_alert_cfg,
                                       sizing=_sizing_for_alert,
                                       state_path=_ALERT_STATE_PATH)
    _fired = _alert_result["alerts"]
    if not _fired:
        st.sidebar.success("No alerts triggered — nothing sent.")
    elif _alert_result["email_sent"]:
        st.sidebar.success(f"{len(_fired)} alert(s) sent.")
    elif not _alert_cfg.smtp_host:
        st.sidebar.warning("SMTP not configured. Set ALERT_SMTP_* env vars.")
    else:
        st.sidebar.error("Email send failed.")

st.sidebar.divider()

# ── Sidebar: Custom Alert Rules ───────────────────────────────────────────────
_CUSTOM_RULE_SIGNALS = {
    "Composite Risk Score":    "composite_risk_score_smooth",
    "HY Spread":               "hy_spread",
    "VIX":                     "vix",
    "Treasury Stress":         "treasury_stress_score_smooth",
    "Credit Risk Score":       "credit_market_risk_score_smooth",
    "Macro Risk Score":        "macro_risk_score_smooth",
    "Complacency Score":       "complacency_score_smooth",
    "Funding Stress":          "enhanced_funding_stress_score_smooth",
    "FX/Commodity Score":      "fx_commodity_score_smooth",
}
_CUSTOM_RULE_OPS = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
                    "≥": lambda a, b: a >= b, "≤": lambda a, b: a <= b}

with st.sidebar.expander("🔔 Custom Alert Rules"):
    st.caption("Trigger a banner in the Signal tab when any rule fires.")
    if "custom_rules" not in st.session_state:
        st.session_state.custom_rules = [
            {"signal": "Composite Risk Score", "op": ">", "threshold": 70.0},
        ]
    _n_rules = st.number_input("Number of rules", 1, 5,
                               len(st.session_state.custom_rules), step=1,
                               key="n_custom_rules")
    while len(st.session_state.custom_rules) < _n_rules:
        st.session_state.custom_rules.append(
            {"signal": "HY Spread", "op": ">", "threshold": 5.0}
        )
    _updated_rules = []
    for _ri in range(int(_n_rules)):
        _rc = st.session_state.custom_rules[_ri]
        _rc1, _rc2, _rc3 = st.columns([3, 1, 2])
        _sig = _rc1.selectbox("Signal", list(_CUSTOM_RULE_SIGNALS.keys()),
                              index=list(_CUSTOM_RULE_SIGNALS.keys()).index(_rc["signal"])
                              if _rc["signal"] in _CUSTOM_RULE_SIGNALS else 0,
                              key=f"rule_sig_{_ri}", label_visibility="collapsed")
        _op  = _rc2.selectbox("Op", list(_CUSTOM_RULE_OPS.keys()),
                              index=list(_CUSTOM_RULE_OPS.keys()).index(_rc["op"])
                              if _rc["op"] in _CUSTOM_RULE_OPS else 0,
                              key=f"rule_op_{_ri}", label_visibility="collapsed")
        _thr = _rc3.number_input("Threshold", value=float(_rc["threshold"]),
                                 key=f"rule_thr_{_ri}", label_visibility="collapsed",
                                 format="%.1f")
        _updated_rules.append({"signal": _sig, "op": _op, "threshold": _thr})
    st.session_state.custom_rules = _updated_rules

# Evaluate custom rules against latest row (results shown in tab1)
_custom_rule_fires = []
for _rule in st.session_state.custom_rules:
    _col = _CUSTOM_RULE_SIGNALS.get(_rule["signal"])
    _fn  = _CUSTOM_RULE_OPS.get(_rule["op"])
    if _col and _fn and _col in df.columns:
        _val = latest.get(_col)
        if _val is not None and not pd.isna(_val):
            if _fn(float(_val), float(_rule["threshold"])):
                _custom_rule_fires.append(
                    f"**{_rule['signal']}** {_rule['op']} {_rule['threshold']} "
                    f"(current: {float(_val):.2f})"
                )

st.sidebar.divider()

# ── Sidebar: Model Config ──────────────────────────────────────────────────────
with st.sidebar.expander("⚙ Model Config"):
    _cfg_tc_bps = st.slider(
        "Transaction cost (bps)", 0, 50, 10, step=5,
        help="Deducted per day of equity weight change. 10 bps ≈ one-way institutional cost.",
    )
    _cfg_equity_floor = st.slider(
        "Min equity weight (%)", 20, 60, 40, step=5,
        help="Floor below which strategy weight never falls, regardless of signal.",
    )
    _cfg_equity_cap = st.slider(
        "Max equity weight (%)", 80, 100, 100, step=5,
        help="Ceiling above which strategy weight never rises.",
    )
    _cfg_target_vol = st.slider(
        "Vol target (%)", 5, 20, 10, step=1,
        help="Annualised realised vol target for the volatility-targeting component.",
    )
    _cfg_momentum_lookback = st.slider(
        "Momentum MA (days)", 50, 252, 200, step=25,
        help="Lookback for the trend filter (SP500 vs rolling MA). 200 = standard.",
    )
    st.caption(
        "Changes here apply to live computations only and are not persisted. "
        "Edit `src/backtester.py` constants to change defaults."
    )

# ── Executive Overview ───────────────────────────────────────────────────────
st.title("Macro Credit Risk Dashboard")

# ── Research Confidence Banner ────────────────────────────────────────────────
_regime_counts_ov = df["final_decision"].value_counts() if "final_decision" in df.columns else pd.Series(dtype=int)
_min_n_ov    = int(_regime_counts_ov.min()) if not _regime_counts_ov.empty else 0
_oos_rows_ov = int((pd.to_datetime(df["date"]) >= pd.Timestamp("2020-01-01")).sum())
_n_total_ov  = len(df)

if _min_n_ov >= 50 and _n_total_ov >= 1500 and _oos_rows_ov >= 500:
    _conf_level = "Indicative"
    _conf_color = "#e67e22"
    _conf_desc  = "Signal shows regime information. Interpret results directionally — limited walk-forward history."
elif _min_n_ov >= 30 and _n_total_ov >= 800:
    _conf_level = "Indicative"
    _conf_color = "#e67e22"
    _conf_desc  = "Developing signal. Some regimes are thinly observed. Treat as directional guidance only."
else:
    _conf_level = "Exploratory"
    _conf_color = "#e74c3c"
    _conf_desc  = "Insufficient history for robust validation. Results are hypothesis-generating, not confirmatory."

st.markdown(
    f'<div style="border-left:3px solid {_conf_color};'
    f'background:rgba(0,0,0,0.25);padding:9px 14px;'
    f'border-radius:0 6px 6px 0;margin-bottom:8px;display:flex;align-items:center;gap:16px">'
    f'<span style="color:{_conf_color};font-weight:700;font-size:0.78rem;'
    f'text-transform:uppercase;letter-spacing:.6px;white-space:nowrap">'
    f'Research Confidence: {_conf_level}</span>'
    f'<span style="color:#6b7280;font-size:0.78rem">'
    f'Tactical risk overlay · {_conf_desc} · Not a proven alpha model.</span>'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="background:#0e1420;border:1px solid #2d3550;border-radius:7px;'
    'padding:10px 16px;margin-bottom:14px;font-size:0.78rem;color:#6b7280;line-height:1.6">'
    '<span style="color:#9ca3af;font-weight:600">Framework Context</span> · '
    'This is a <strong style="color:#c8ccd4">tactical risk-overlay</strong> framework. '
    'Primary value: <strong style="color:#c8ccd4">risk control and drawdown management</strong>, '
    'not alpha generation. '
    'Factor analysis shows R²≈0.94 vs SP500 — most performance is beta-reduction timing, '
    'not independent alpha. '
    'OOS splits are structurally stable but were designed with awareness of the full dataset '
    '(pseudo-OOS, not truly frozen-parameter). '
    'Regime separability is low (η²≈0.9%) — the 4-regime consolidation reflects this. '
    'All findings, including weaknesses, are displayed without filtering.'
    '</div>',
    unsafe_allow_html=True,
)

decision    = str(latest.get("final_decision",    "N/A"))
environment = str(latest.get("final_environment", "N/A"))
action      = str(latest.get("final_action",      "N/A"))
composite   = float(latest.get("composite_risk_score_smooth", 0))
comp_label  = str(latest.get("composite_risk_label", "N/A"))

_DECISION_COLORS = {
    "Risk-On":  "#27ae60",
    "Neutral":  "#95a5a6",
    "Caution":  "#e67e22",
    "Risk-Off": "#e74c3c",
}
_badge_bg = _DECISION_COLORS.get(decision, "#555555")

if _bdays_stale == 0:
    _stale_txt = "🟢 Current"
elif _bdays_stale == 1:
    _stale_txt = "🟡 1d stale"
else:
    _stale_txt = f"🔴 {_bdays_stale}d stale"

st.markdown(
    f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:4px">'
    f'<span style="background:{_badge_bg};color:#fff;padding:5px 14px;'
    f'border-radius:6px;font-size:1.05em;font-weight:600;letter-spacing:.3px">'
    f'{decision}</span>'
    f'<span style="color:#555;font-size:0.95em">{environment}</span>'
    f'<span style="margin-left:auto;color:#777;font-size:0.9em">'
    f'{_stale_txt}&nbsp;&nbsp;·&nbsp;&nbsp;{_last_date_str}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Key metrics strip
_perf_ov = load_performance_scorecard(df)
_fa_ov   = load_factor_analysis(df)
_strat   = _perf_ov.get("full_period", {}).get("strategy", {})
_sharpe  = _strat.get("sharpe",       float("nan"))
_max_dd  = _strat.get("max_drawdown", float("nan"))
_hit_rt  = _strat.get("hit_rate",     float("nan"))
_beta    = _fa_ov.get("regression",   {}).get("beta", float("nan"))

mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
mc1.metric("Composite Score", f"{composite:.1f} / 100")
mc2.metric("Composite Label", comp_label)
mc3.metric("Strategy Sharpe", f"{_sharpe:.2f}" if not pd.isna(_sharpe) else "—")
mc4.metric("Market Beta",     f"{_beta:.2f}"   if not pd.isna(_beta)   else "—",
           help="β vs. SP500 — full-period OLS")
mc5.metric("Max Drawdown",    f"{_max_dd:.1%}" if not pd.isna(_max_dd) else "—")
mc6.metric("Hit Rate",        f"{_hit_rt:.1%}" if not pd.isna(_hit_rt) else "—",
           help="% of trading days with positive strategy return")

# Regime probability strip
_rp_ov = load_regime_probability(df)
_cur_ov = _rp_ov.get("current", {})
if _cur_ov:
    _top3 = sorted(_cur_ov["probs"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    _BADGE_COLORS = {
        "Risk-On":  "#27ae60",
        "Neutral":  "#95a5a6",
        "Caution":  "#e67e22",
        "Risk-Off": "#e74c3c",
    }
    _ent = _cur_ov.get("entropy", 0.0)
    _max_ent = max(1.0, _ent)
    st.markdown("**Regime probability distribution** (Gaussian Naive Bayes · in-sample)")
    _prob_cols = st.columns(len(_top3) + 1)
    for _ci, (_rname, _rprob) in enumerate(_top3):
        _col_hex = _BADGE_COLORS.get(_rname, "#555")
        _prob_cols[_ci].markdown(
            f'<div style="border-left:3px solid {_col_hex};padding:6px 12px;'
            f'background:#151820;border-radius:0 6px 6px 0;margin-bottom:4px">'
            f'<div style="font-size:0.68rem;color:#4b5563;text-transform:uppercase;'
            f'letter-spacing:.4px">{_rname}</div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:#e2e8f0">{_rprob:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    _prob_cols[-1].markdown(
        f'<div style="padding:6px 12px;background:#151820;border-radius:6px;'
        f'border:1px solid #1e2435;margin-bottom:4px">'
        f'<div style="font-size:0.68rem;color:#4b5563;text-transform:uppercase;'
        f'letter-spacing:.4px">Uncertainty</div>'
        f'<div style="font-size:1.2rem;font-weight:700;color:#9aa0aa">{_ent:.2f} bits</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# Two-column overview body
_ov_l, _ov_r = st.columns([1, 2])

with _ov_l:
    st.subheader("Score Breakdown")
    _scores = [
        ("Treasury Stress", latest.get("treasury_stress_score_smooth")),
        ("Complacency",     latest.get("complacency_score_smooth")),
        ("Credit Risk",     latest.get("credit_market_risk_score_smooth")),
        ("Macro Risk",      latest.get("macro_risk_score_smooth")),
        ("Liquidity",       latest.get("liquidity_regime_score_smooth")),
        ("Funding Stress",  latest.get("enhanced_funding_stress_score_smooth")),
        ("FX / Commodity",  latest.get("fx_commodity_score_smooth")),
        ("Mean Reversion †", latest.get("mean_reversion_score_smooth")),
    ]
    _sdf = pd.DataFrame(_scores, columns=["Component", "Score"]).set_index("Component")
    _sdf["Score"] = pd.to_numeric(_sdf["Score"], errors="coerce").round(1)

    def _ov_color(v):
        if pd.isna(v): return ""
        if v >= 70: return "background-color:rgba(231,76,60,0.2);color:#e74c3c"
        if v >= 50: return "background-color:rgba(230,126,34,0.2);color:#e67e22"
        return "background-color:rgba(39,174,96,0.15);color:#27ae60"

    st.dataframe(
        _sdf.style.map(_ov_color, subset=["Score"]).format({"Score": "{:.1f}"}),
        use_container_width=True,
        height=310,
    )
    st.caption("† Mean Reversion is tracked but excluded from composite (redundant with Complacency).")

with _ov_r:
    import plotly.graph_objects as _ov_go

    _recent90 = df.tail(90).copy()
    _fig_ov = _ov_go.Figure()
    _fig_ov.add_trace(_ov_go.Scatter(
        x=_recent90["date"],
        y=_recent90["composite_risk_score_smooth"],
        mode="lines",
        line=dict(color="#4f8ef7", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(79,142,247,0.12)",
        name="Composite",
    ))
    _fig_ov.add_hrect(y0=70, y1=100, fillcolor="rgba(231,76,60,0.12)",  line_width=0,
                      annotation_text="Elevated", annotation_position="top left",
                      annotation_font_color="rgba(231,76,60,0.8)")
    _fig_ov.add_hrect(y0=50, y1=70,  fillcolor="rgba(230,126,34,0.10)", line_width=0,
                      annotation_text="Caution",  annotation_position="top left",
                      annotation_font_color="rgba(230,126,34,0.8)")
    _fig_ov.update_layout(
        title=dict(text="Composite Score — Last 90 Days", font=dict(size=12, color="#9aa0aa")),
        height=240,
        margin=dict(l=0, r=8, t=32, b=8),
        xaxis=dict(showgrid=False, title=None, color="#6b7280"),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   title=None, color="#6b7280"),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9aa0aa"),
    )
    st.plotly_chart(_fig_ov, use_container_width=True)

    if "strategy_daily_return" in df.columns and "sp500_daily_return" in df.columns:
        _cum_s = (1 + df["strategy_daily_return"].fillna(0)).cumprod()
        _cum_m = (1 + df["sp500_daily_return"].fillna(0)).cumprod()
        _fig_cum = _ov_go.Figure()
        _fig_cum.add_trace(_ov_go.Scatter(x=df["date"], y=_cum_s,
                                          name="Strategy",
                                          line=dict(color="#4f8ef7", width=2)))
        _fig_cum.add_trace(_ov_go.Scatter(x=df["date"], y=_cum_m,
                                          name="SP500",
                                          line=dict(color="#6b7280", width=1.5, dash="dot")))
        _fig_cum.update_layout(
            title=dict(text="Cumulative Return — Full Period", font=dict(size=12, color="#9aa0aa")),
            height=200,
            margin=dict(l=0, r=8, t=32, b=0),
            xaxis=dict(showgrid=False, title=None, color="#6b7280"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title=None, color="#6b7280"),
            legend=dict(orientation="h", y=-0.25, x=0, font=dict(color="#9aa0aa")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
        )
        st.plotly_chart(_fig_cum, use_container_width=True)

# Alerts banner
if _bdays_stale >= 2:
    st.warning(
        f"Data is **{_bdays_stale} trading days stale** (last: {_last_date_str}). "
        f"Use '⟳ Refresh Data' in the sidebar to fetch the latest FRED data."
    )
if composite >= 70:
    st.warning(
        f"Composite risk is **elevated at {composite:.1f}**. "
        f"Current decision: **{decision}**. Review signal carefully before acting."
    )

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Signal", "Charts", "Portfolio", "Backtest", "Analytics", "Models", "History", "Allocation", "⚙ Health"
])

with tab1:
    st.header("Current Signal Snapshot")

    # ── Custom rule banners ───────────────────────────────────────────────────
    if _custom_rule_fires:
        for _fire_msg in _custom_rule_fires:
            st.warning(f"⚠ Custom Alert: {_fire_msg}", icon="🔔")

    # ── Live intraday snapshot ────────────────────────────────────────────────
    try:
        _live = get_live_snapshot()
        _move_snap = get_move_snapshot()
        if _live and "error" not in _live:
            st.subheader("Live Market Snapshot")
            _ls_cols = st.columns(5)
            for _ls_col, (_ls_key, _ls_label) in zip(
                _ls_cols,
                [("vix","VIX"), ("sp500","S&P 500"), ("hyg","HYG (HY ETF)"), ("lqd","LQD (IG ETF)")]
            ):
                _ls_d = _live.get(_ls_key)
                if _ls_d:
                    _ls_chg = _ls_d.get("day_chg_pct", 0) or 0
                    _ls_col.metric(
                        _ls_label,
                        f"{_ls_d['current']:.2f}",
                        delta=f"{_ls_chg:+.2f}%",
                        delta_color="inverse" if _ls_key == "vix" else "normal",
                    )
            if _move_snap and _move_snap.get("available"):
                _ls_cols[4].metric(
                    f"MOVE ({_move_snap.get('regime','—')})",
                    f"{_move_snap['current']:.1f}",
                    delta=f"{_move_snap.get('day_chg_pct', 0):+.2f}%",
                    delta_color="inverse",
                    help="ICE BofA MOVE Index — bond market implied volatility. >80 = elevated, >120 = stress.",
                )
            st.caption(f"Live prices as of {_live.get('as_of','—')} · refreshes on page load · MOVE: rates market volatility (bond VIX)")
            st.divider()
    except Exception:
        pass

    # ── LLM morning briefing ──────────────────────────────────────────────────
    st.subheader("Morning Briefing")
    _briefing_result = load_llm_briefing(df)
    if _briefing_result.get("error"):
        if "ANTHROPIC_API_KEY not set" in str(_briefing_result["error"]):
            st.caption("Set `ANTHROPIC_API_KEY` in Streamlit secrets to enable AI briefings.")
        else:
            st.caption(f"Briefing unavailable: {_briefing_result['error']}")
    elif _briefing_result.get("text"):
        _brief_cached = _briefing_result.get("cached", False)
        st.markdown(
            f'<div style="border-left:3px solid #4f8ef7;background:rgba(79,142,247,0.06);'
            f'padding:14px 18px;border-radius:0 6px 6px 0;margin-bottom:8px;'
            f'font-size:0.9rem;color:#c8ccd4;line-height:1.6">'
            f'{_briefing_result["text"].replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{'Cached' if _brief_cached else 'Generated'} {_briefing_result.get('date','')}"
                   f" · powered by Claude Haiku")

    # ── Fed Communication Sentiment ───────────────────────────────────────────
    st.divider()
    _fed_col1, _fed_col2 = st.columns([2, 1])
    with _fed_col1:
        st.subheader("Fed Communication Sentiment")
        st.caption(
            "Scores the most recent FOMC statement 0–100 for hawkish/dovish tone using Claude AI. "
            "0 = very dovish (cuts imminent), 50 = neutral, 100 = very hawkish (hikes signaled). "
            "Hawkish statements → tighter financial conditions → wider credit spreads."
        )
    with _fed_col2:
        try:
            _pdf_bytes = generate_snapshot_bytes(df)
            if _pdf_bytes:
                import datetime as _dt
                st.download_button(
                    label="Download PDF Snapshot",
                    data=_pdf_bytes,
                    file_name=f"credit_dashboard_{_dt.date.today()}.pdf",
                    mime="application/pdf",
                    help="Download a PDF briefing of the current dashboard state",
                )
        except Exception:
            pass

    try:
        _fed = load_fed_sentiment(df)
        _fed_cur = _fed.get("current", {})
        _fed_score = _fed_cur.get("score")
        if _fed_score is not None:
            _f1, _f2, _f3, _f4 = st.columns(4)
            _fed_color = "#e74c3c" if _fed_score > 65 else "#f39c12" if _fed_score > 50 else "#27ae60" if _fed_score < 35 else "#6b7280"
            _f1.metric("Hawkish/Dovish Score", f"{_fed_score}/100",
                       help="0=Very Dovish · 50=Neutral · 100=Very Hawkish")
            _f2.metric("Label", _fed_cur.get("label", "—"))
            _f3.metric("Statement Date", _fed_cur.get("date", "—"))
            _f4.metric("Trend", _fed.get("trend", "—"),
                       help="Direction of last 3 cached FOMC statements")
            if _fed_cur.get("reasoning"):
                st.markdown(
                    f'<div style="border-left:3px solid {_fed_color};background:rgba(255,255,255,0.03);'
                    f'padding:10px 14px;border-radius:0 6px 6px 0;margin-top:8px">'
                    f'<span style="color:#9aa0aa;font-size:0.85rem">{_fed_cur["reasoning"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            _fed_hist = _fed.get("history", [])
            if len(_fed_hist) > 1:
                with st.expander(f"FOMC sentiment history ({len(_fed_hist)} meetings)"):
                    _fh_df = pd.DataFrame(_fed_hist)
                    st.dataframe(_fh_df, use_container_width=True, hide_index=True)
        elif not _fed.get("api_key_present"):
            st.caption("Set `ANTHROPIC_API_KEY` in Streamlit secrets to enable Fed sentiment scoring.")
        else:
            st.caption("Fetching FOMC statement… (first load may take a moment)")
    except Exception as _fed_e:
        st.caption(f"Fed sentiment unavailable: {_fed_e}")

    # ── NL query ─────────────────────────────────────────────────────────────
    with st.expander("Ask a question about the data"):
        _nl_question = st.text_input(
            "Question", placeholder='e.g. "When was the last Risk-Off regime longer than 30 days?"',
            key="nl_query_input",
        )
        if _nl_question and st.button("Ask", key="nl_query_btn"):
            import os as _os
            _ant_key = _os.environ.get("ANTHROPIC_API_KEY", "")
            if not _ant_key:
                st.warning("Set ANTHROPIC_API_KEY to enable natural language queries.")
            else:
                try:
                    import anthropic as _ant
                    _nl_client = _ant.Anthropic(api_key=_ant_key)
                    _nl_ctx = (
                        f"You are an analyst assistant for a credit risk dashboard. "
                        f"The scored dataset has {len(df)} rows from {df['date'].min()} to {df['date'].max()}. "
                        f"Columns include: date, final_decision (Risk-On/Neutral/Caution/Risk-Off), "
                        f"composite_risk_score_smooth (0-100), hy_spread, vix, sp500, "
                        f"treasury_stress_score_smooth, credit_market_risk_score_smooth, "
                        f"macro_risk_score_smooth, complacency_score_smooth. "
                        f"Current: regime={latest.get('final_decision')}, "
                        f"score={latest.get('composite_risk_score_smooth'):.1f}, "
                        f"hy_spread={latest.get('hy_spread'):.2f}."
                    )
                    # Compute answer using Python on the df for factual questions
                    _nl_data_ctx = ""
                    if "risk-off" in _nl_question.lower() or "regime" in _nl_question.lower():
                        _runs = []
                        _cur_reg, _start, _cnt = None, None, 0
                        for _, _r in df[["date","final_decision"]].iterrows():
                            if _r["final_decision"] != _cur_reg:
                                if _cur_reg is not None:
                                    _runs.append((_cur_reg, _start, _r["date"], _cnt))
                                _cur_reg, _start, _cnt = _r["final_decision"], _r["date"], 1
                            else:
                                _cnt += 1
                        if _cur_reg:
                            _runs.append((_cur_reg, _start, df["date"].iloc[-1], _cnt))
                        _ro_runs = [(s,e,d) for reg,s,e,d in _runs if reg == "Risk-Off"]
                        _nl_data_ctx = (
                            f"Risk-Off episodes: {[(str(s)[:10], str(e)[:10], d) for s,e,d in sorted(_ro_runs, key=lambda x: x[2], reverse=True)[:5]]}"
                        )
                    with st.spinner("Thinking…"):
                        _nl_resp = _nl_client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=300,
                            system=_nl_ctx + (" Data: " + _nl_data_ctx if _nl_data_ctx else ""),
                            messages=[{"role": "user", "content": _nl_question}],
                        )
                    st.markdown(_nl_resp.content[0].text)
                except Exception as _nl_e:
                    st.error(f"Query failed: {_nl_e}")

    def _kv_card(label, value, color="#9aa0aa"):
        return (
            f'<div style="padding:10px 14px;border-radius:7px;background:#151820;'
            f'border:1px solid #1e2435;margin-bottom:6px">'
            f'<span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:.5px;'
            f'color:#4b5563">{label}</span><br>'
            f'<span style="font-size:0.92rem;font-weight:600;color:{color}">{value}</span>'
            f'</div>'
        )

    left, right = st.columns(2)

    with left:
        st.subheader("Market Snapshot")
        _yield_curve = str(latest.get('yield_curve_regime', 'N/A'))
        _yc_color = "#e74c3c" if "Invert" in _yield_curve else \
                    "#e67e22" if "Flat" in _yield_curve else "#27ae60"
        st.markdown(
            _kv_card("HY Spread", f"{latest.get('hy_spread', 0):.2f}%") +
            _kv_card("VIX", f"{latest.get('vix', 0):.1f}") +
            _kv_card("Unemployment", f"{latest.get('unemployment', 0):.1f}%") +
            _kv_card("SP500", f"{latest.get('sp500', 0):,.0f}") +
            _kv_card("Yield Curve", _yield_curve, _yc_color) +
            _kv_card("Shock Flag", str(latest.get('shock_flag', 'No Shock')),
                     "#e74c3c" if latest.get('shock_flag', 'No Shock') != 'No Shock' else "#27ae60"),
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Signal Regime")
        st.markdown(
            _kv_card("Macro Label",       str(latest.get('macro_risk_label', 'N/A'))) +
            _kv_card("Credit Label",      str(latest.get('credit_market_risk_label', 'N/A'))) +
            _kv_card("Liquidity Label",   str(latest.get('liquidity_regime_label', 'N/A'))) +
            _kv_card("Complacency Label", str(latest.get('complacency_label', 'N/A'))) +
            _kv_card("Mean Reversion",    str(latest.get('mean_reversion_label', 'N/A'))) +
            _kv_card("Transition Signal", str(latest.get('transition_signal', 'N/A'))) +
            _kv_card("Composite Regime",  str(latest.get('composite_risk_label', 'N/A'))),
            unsafe_allow_html=True,
        )

    # ── Cross-Asset Scores snapshot ───────────────────────────────────────────
    _ca_score_cols = [
        ("rates_stress_score_smooth",            "Rates Stress",   None),
        ("enhanced_funding_stress_score_smooth", "Funding Stress", None),
        ("fx_commodity_score_smooth",            "FX/Commodity",   None),
        ("banking_stress_score_smooth",          "Banking Stress †", None),
    ]
    _ca_available = [(col, label, c) for col, label, c in _ca_score_cols if col in df.columns]
    if _ca_available:
        st.subheader("Cross-Asset Signals")
        _ca_cols_ui = st.columns(len(_ca_available))
        for _ui_col, (col, label, _) in zip(_ca_cols_ui, _ca_available):
            _val = latest.get(col, float("nan"))
            if isinstance(_val, float) and not pd.isna(_val):
                _color = "#e74c3c" if _val >= 60 else "#e67e22" if _val >= 40 else "#27ae60"
                _ui_col.markdown(
                    _kv_card(label, f"{_val:.1f} / 100", _color),
                    unsafe_allow_html=True,
                )
        st.caption("† Banking Stress is informational only — excluded from composite (wrong-direction forward correlation at all horizons).")
        # MOVE index card
        if "move_index" in df.columns:
            _mv_val = latest.get("move_index", float("nan"))
            if isinstance(_mv_val, float) and not pd.isna(_mv_val):
                _mv_color = "#e74c3c" if _mv_val >= 150 else "#e67e22" if _mv_val >= 100 else "#27ae60"
                st.markdown(
                    _kv_card("MOVE Index (Bond Vol)", f"{_mv_val:.1f}", _mv_color),
                    unsafe_allow_html=True,
                )

    # ── Report export ─────────────────────────────────────────────────────────
    st.subheader("Export Signal Report")
    with st.spinner("Building report…"):
        _regime_results_t1 = load_regime_transition(df)
        _trans_counts_t1 = _regime_results_t1.get("transition_counts") if _regime_results_t1 else None
        _wf_windows_t1, _ = load_walk_forward()
        _audit_t1    = load_validation_audit(df, _wf_windows_t1, _trans_counts_t1)
        _decay_t1    = load_signal_decay(df)
        _tail_t1     = load_tail_risk(df)
        _wopt_t1     = load_weight_optimization(df)
        _attr_t1     = load_attribution(df)
        _stress_t1   = load_stress_analysis(df)
        _rp_t1       = load_regime_probability(df)
        _ps_t1       = load_position_sizing(df)
        _sg_t1       = load_scenario_grid(df)
        _sp_t1       = load_subperiod_attribution(df)
        _mc_t1_key   = str(hash(str(_rp_t1.get("current", {}).get("probs", {}))))
        _mc_t1       = load_monte_carlo(df, _mc_t1_key)

        _html_report = load_report_html(
            df, _audit_t1, _decay_t1, _tail_t1, _wopt_t1, _attr_t1, _stress_t1,
            _regime_prob=_rp_t1,
            _position_sizing=_ps_t1,
            _scenario_grid=_sg_t1.get("grid"),
            _subperiod_table=_sp_t1.get("table"),
            _monte_carlo=_mc_t1,
        )
        _excel_report = generate_excel_report(
            df,
            position_sizing=_ps_t1,
            scenario_grid=_sg_t1.get("grid"),
            subperiod_table=_sp_t1.get("table"),
            regime_prob=_rp_t1,
            monte_carlo=_mc_t1,
        )

    _report_date = _last_date_str
    _dl_col1, _dl_col2 = st.columns(2)
    with _dl_col1:
        st.download_button(
            label="⬇ Download HTML Report",
            data=_html_report.encode("utf-8"),
            file_name=f"macro_credit_signal_report_{_report_date}.html",
            mime="text/html",
            use_container_width=True,
        )
        st.caption("Self-contained HTML — open in any browser.")
    with _dl_col2:
        st.download_button(
            label="⬇ Download Excel Workbook",
            data=_excel_report,
            file_name=f"macro_credit_dashboard_{_report_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.caption("6 sheets: Summary, Scores, Sizing, Scenarios, Attribution, Monte Carlo.")

    # ── Factor attribution delta ──────────────────────────────────────────────
    st.subheader("What Moved the Score Today")
    st.caption("Day-over-day change in each component's weighted contribution to the composite risk score.")
    if len(df) >= 2:
        _delta_today = df.iloc[-1]
        _delta_prev  = df.iloc[-2]
        _delta_rows  = []
        for _dk, _dw in COMPOSITE_WEIGHTS.items():
            _dcol = SCORE_COLS.get(_dk)
            if _dcol and _dcol in df.columns:
                _d_now  = float(_delta_today.get(_dcol, float("nan")))
                _d_prev = float(_delta_prev.get(_dcol, float("nan")))
                if not (pd.isna(_d_now) or pd.isna(_d_prev)):
                    _d_raw_delta = _d_now - _d_prev
                    _d_contrib_delta = _d_raw_delta * _dw
                    _delta_rows.append({
                        "Signal":        DISPLAY_NAMES.get(_dk, _dk),
                        "Weight":        _dw,
                        "Yesterday":     round(_d_prev, 1),
                        "Today":         round(_d_now, 1),
                        "Raw Δ":         round(_d_raw_delta, 1),
                        "Weighted Δ":    round(_d_contrib_delta, 2),
                    })
        if _delta_rows:
            import plotly.graph_objects as _dgo
            _delta_df = pd.DataFrame(_delta_rows).sort_values("Weighted Δ", key=abs, ascending=False)
            _delta_colors = ["#e74c3c" if v > 0 else "#27ae60" for v in _delta_df["Weighted Δ"]]
            _delta_fig = _dgo.Figure(_dgo.Bar(
                x=_delta_df["Signal"],
                y=_delta_df["Weighted Δ"],
                marker_color=_delta_colors,
                text=[f"{v:+.2f}" for v in _delta_df["Weighted Δ"]],
                textposition="outside",
                hovertemplate="%{x}<br>Weighted Δ: %{y:+.2f}<br>Raw Δ: %{customdata:+.1f}<extra></extra>",
                customdata=_delta_df["Raw Δ"].values,
            ))
            _delta_fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
            _delta_fig.update_layout(
                height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                margin=dict(l=8, r=8, t=8, b=80),
                xaxis=dict(showgrid=False, color="#6b7280", tickangle=-25),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Weighted Contribution Δ"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_delta_fig, use_container_width=True)
            _total_delta = _delta_df["Weighted Δ"].sum()
            _composite_now  = float(_delta_today.get("composite_risk_score_smooth", float("nan")))
            _composite_prev = float(_delta_prev.get("composite_risk_score_smooth", float("nan")))
            _c1d, _c2d, _c3d = st.columns(3)
            _c1d.metric("Composite Score (Yesterday)", f"{_composite_prev:.1f}" if not pd.isna(_composite_prev) else "—")
            _c2d.metric("Composite Score (Today)",     f"{_composite_now:.1f}" if not pd.isna(_composite_now) else "—",
                        delta=f"{_composite_now - _composite_prev:+.1f}" if not (pd.isna(_composite_now) or pd.isna(_composite_prev)) else None)
            _c3d.metric("Sum of Weighted Δ", f"{_total_delta:+.2f}",
                        help="Should approximately match the composite score change (rounding/smoothing may cause small diff).")
            with st.expander("Full delta table"):
                st.dataframe(
                    _delta_df.style.format({"Weight": "{:.0%}", "Yesterday": "{:.1f}", "Today": "{:.1f}",
                                           "Raw Δ": "{:+.1f}", "Weighted Δ": "{:+.2f}"}),
                    use_container_width=True,
                )

    # ── Taylor Rule + Monetary Policy Gap ────────────────────────────────────
    st.divider()
    st.subheader("Monetary Policy: Taylor Rule")
    st.caption(
        "The **Taylor Rule** (1993) estimates where the Fed funds rate *should* be based on "
        "inflation and the output gap: r = r* + π + 0.5(π − π*) + 0.5(y − y*). "
        "A **positive policy gap** (actual > Taylor rate) means policy is *restrictively tight* — "
        "historically associated with credit spread widening and financial stress."
    )
    try:
        _tr = load_taylor(df)
        if _tr.get("available"):
            _tr_cur = _tr["current"]
            _tr1, _tr2, _tr3, _tr4 = st.columns(4)
            _tr1.metric("Fed Funds Rate", f"{_tr_cur.get('fed_funds', float('nan')):.2f}%")
            _tr2.metric("Taylor Rule Rate", f"{_tr_cur.get('taylor_rate', float('nan')):.2f}%",
                        help="Estimated neutral rate given inflation and unemployment gap")
            _tr3.metric("Policy Gap", f"{_tr_cur.get('policy_gap', 0):+.2f}pp",
                        delta=f"{_tr_cur.get('policy_gap', 0):+.2f}pp",
                        delta_color="inverse",
                        help="Actual − Taylor rate. Positive = too tight.")
            _tr4.metric("Stance", _tr_cur.get("stance", "—"))

            # Rolling policy gap chart
            if "df" in _tr and "policy_gap" in _tr["df"].columns:
                import plotly.graph_objects as _trgo
                _tr_df = _tr["df"].copy()
                _tr_df["date"] = pd.to_datetime(_tr_df["date"])
                _tr_fig = _trgo.Figure()
                _tr_colors = ["#e74c3c" if v > 0 else "#27ae60" for v in _tr_df["policy_gap"].fillna(0)]
                _tr_fig.add_trace(_trgo.Bar(
                    x=_tr_df["date"], y=_tr_df["policy_gap"],
                    marker_color=_tr_colors, name="Policy Gap",
                    hovertemplate="%{x|%Y-%m-%d}<br>Policy Gap: %{y:+.2f}pp<extra></extra>",
                ))
                _tr_fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
                _tr_fig.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Actual − Taylor Rate (pp)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_tr_fig, use_container_width=True)
                st.caption("Red = policy tighter than Taylor Rule prescribes · Green = looser than prescribed")
        else:
            st.info("Taylor Rule unavailable — requires FEDFUNDS, CPIAUCSL / T10YIE, UNRATE in dataset.")
    except Exception as _tr_e:
        st.caption(f"Taylor Rule unavailable: {_tr_e}")

    # ── Recession Probability ─────────────────────────────────────────────────
    st.subheader("Recession Probability (Estrella-Mishkin)")
    st.caption(
        "The **Estrella-Mishkin (1998)** model uses the yield curve spread (10y − 3m) as "
        "the sole predictor of recessions 12 months ahead via a probit regression. "
        "It was calibrated on post-war US data and has accurately signaled every recession since 1969. "
        "**Threshold**: p > 25% = elevated concern; p > 40% = high probability."
    )
    try:
        _rec = load_recession(df)
        if _rec.get("available"):
            _rc = _rec["current"]
            _r1, _r2, _r3, _r4 = st.columns(4)
            _r1.metric("10y−3m Spread", f"{_rc['spread_10y3m']:.2f}%",
                       help="Negative = inverted yield curve")
            _r2.metric("10y−2y Spread", f"{_rc['spread_10y2y']:.2f}%")
            _r3.metric("Recession Prob (12m)", f"{_rc['recession_prob_12m']:.1%}",
                       delta_color="inverse",
                       delta=f"{_rc['recession_prob_12m']:.1%}")
            _r4.metric("Signal", _rc.get("signal", "—"))

            # Rolling recession probability chart
            if "df" in _rec and "recession_prob_12m" in _rec["df"].columns:
                import plotly.graph_objects as _recgo
                _rec_df = _rec["df"].copy()
                _rec_df["date"] = pd.to_datetime(_rec_df["date"])
                _rec_fig = _recgo.Figure()
                _rec_fig.add_trace(_recgo.Scatter(
                    x=_rec_df["date"], y=_rec_df["recession_prob_12m"] * 100,
                    name="Recession Prob", line=dict(color="#e67e22", width=2),
                    fill="tozeroy", fillcolor="rgba(230,126,34,0.12)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Recession Prob: %{y:.1f}%<extra></extra>",
                ))
                _rec_fig.add_hline(y=25, line=dict(color="#e74c3c", dash="dash", width=1),
                                   annotation_text="25% Elevated", annotation_position="top right",
                                   annotation_font=dict(color="#e74c3c", size=10))
                _rec_fig.add_hline(y=40, line=dict(color="#9b59b6", dash="dot", width=1),
                                   annotation_text="40% High", annotation_position="top right",
                                   annotation_font=dict(color="#9b59b6", size=10))
                _rec_fig.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Probability (%)", range=[0, 100]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_rec_fig, use_container_width=True)

            # Inversion episodes table
            if _rec.get("historical_inversions") is not None and not _rec["historical_inversions"].empty:
                with st.expander("Historical yield curve inversion episodes"):
                    st.dataframe(_rec["historical_inversions"], use_container_width=True, hide_index=True)
        else:
            st.info("Recession model unavailable — requires 10-year and 3-month Treasury yields in dataset.")
    except Exception as _rec_e:
        st.caption(f"Recession model unavailable: {_rec_e}")

    # ── VIX Term Structure ────────────────────────────────────────────────────
    st.subheader("VIX Term Structure")
    st.caption(
        "The VIX term structure measures the slope between near-term and longer-term implied vol. "
        "**Backwardation** (VIX > VIX3M) signals acute fear — short-dated demand for protection exceeds "
        "longer-dated, historically preceding credit spread widening by 2–4 weeks."
    )
    try:
        _vts = load_vix_term(df)
        if _vts.get("available"):
            _snap = _vts.get("snapshot", {})
            _v1, _v2, _v3, _v4 = st.columns(4)
            _v1.metric("VIX", f"{_snap.get('vix', float('nan')):.1f}")
            _v2.metric("VIX3M", f"{_snap.get('vix3m', float('nan')):.1f}")
            _v3.metric("Slope (VIX3M−VIX)", f"{_snap.get('slope', float('nan')):.1f}",
                       help="Negative = backwardation (acute stress)")
            _v4.metric("Structure", _snap.get("structure", "—"))
            if _snap.get("structure", "").startswith("Backwardation") or _snap.get("structure", "") == "Severe Backwardation":
                st.warning(f"VIX Backwardation detected — short-dated vol exceeds longer-dated. Historically precedes HY spread widening.")
            if _vts.get("key_insight"):
                st.caption(_vts["key_insight"])
        else:
            st.info("VIX term structure unavailable — requires live market data (^VIX, ^VIX3M).")
    except Exception as _vts_e:
        st.caption(f"VIX term structure unavailable: {_vts_e}")

    # ── Options Skew (CBOE SKEW / VVIX) ──────────────────────────────────────
    st.subheader("Options Skew & Tail Risk")
    st.caption(
        "The **CBOE SKEW Index** measures the price of left-tail protection relative to ATM vol. "
        "SKEW > 140 signals elevated tail-risk hedging. "
        "**Hidden danger**: SKEW > 140 while VIX < 20 means markets are complacent about near-term vol "
        "but paying up for deep downside protection — often precedes sharp dislocations."
    )
    try:
        _skw = load_options_skew(df)
        if _skw.get("available"):
            _ss = _skw.get("snapshot", {})
            _s1, _s2, _s3, _s4 = st.columns(4)
            _s1.metric("SKEW Index", f"{_ss.get('skew', float('nan')):.1f}")
            _s2.metric("VVIX", f"{_ss.get('vvix', float('nan')):.1f}",
                       help="VIX of VIX — uncertainty about future fear")
            _s3.metric("Regime", _ss.get("skew_regime", "—"))
            _s4.metric("30d Avg SKEW", f"{_ss.get('skew_30d_avg', float('nan')):.1f}")
            if _ss.get("hidden_danger"):
                st.error("HIDDEN DANGER: SKEW > 140 + VIX < 20 — complacency with elevated tail hedging. Review downside positioning.")
            if _skw.get("interpretation"):
                st.caption(_skw["interpretation"])
        else:
            st.info("Options skew unavailable — requires live market data (^SKEW, ^VVIX).")
    except Exception as _skw_e:
        st.caption(f"Options skew unavailable: {_skw_e}")

    # ── Real Rates Decomposition ──────────────────────────────────────────────
    st.subheader("Real Rates: Nominal − Breakeven Inflation")
    st.caption(
        "**Real rates** = 10y nominal yield − 10y TIPS breakeven. Rising real rates tighten "
        "financial conditions and are the primary transmission channel for credit stress. "
        "Real rates above 2% have historically preceded HY spread widening. "
        "Deeply negative real rates (<−1%) reflect financial repression — easy conditions for credit."
    )
    try:
        _rr = load_real_rates(df)
        if _rr.get("available"):
            _rrc = _rr["current"]
            _rr1, _rr2, _rr3, _rr4 = st.columns(4)
            _rr1.metric("Real Rate (10y)", f"{_rrc.get('real_rate_10y', float('nan')):.2f}%",
                        delta=f"{_rrc.get('real_rate_change_1m', 0):+.2f}pp 1M",
                        delta_color="inverse")
            _rr2.metric("Nominal 10y", f"{_rrc.get('yield_10y', float('nan')):.2f}%")
            _rr3.metric("Breakeven Inflation", f"{_rrc.get('breakeven_10y', float('nan')):.2f}%")
            _rr4.metric("Credit Signal", f"{_rrc.get('real_rate_credit_signal', 0):.0f}/100",
                        help="Higher = tighter real rates = worse for credit")
            if _rr.get("rising_flag"):
                st.warning("Real rates rising rapidly — historically precedes HY spread widening by 4–8 weeks.")
            st.caption(f"Regime: **{_rrc.get('real_rate_regime', '—')}** · {_rr.get('interpretation', '')}")
            _rr_hist = _rr.get("historical")
            if _rr_hist is not None and not _rr_hist.empty and "real_rate_10y" in _rr_hist.columns:
                import plotly.graph_objects as _rrgo
                _rr_fig = _rrgo.Figure()
                _rr_fig.add_trace(_rrgo.Scatter(
                    x=pd.to_datetime(_rr_hist.index), y=_rr_hist["real_rate_10y"],
                    name="Real Rate (10y)", line=dict(color="#4f8ef7", width=2),
                    fill="tozeroy", fillcolor="rgba(79,142,247,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Real Rate: %{y:.2f}%<extra></extra>",
                ))
                if "breakeven_10y" in _rr_hist.columns:
                    _rr_fig.add_trace(_rrgo.Scatter(
                        x=pd.to_datetime(_rr_hist.index), y=_rr_hist["breakeven_10y"],
                        name="Breakeven Inflation", line=dict(color="#e67e22", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>Breakeven: %{y:.2f}%<extra></extra>",
                    ))
                _rr_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _rr_fig.add_hline(y=2.0, line_color="rgba(231,76,60,0.4)", line_width=1,
                                  annotation_text="Restrictive threshold",
                                  annotation_font=dict(color="#e74c3c", size=10))
                _rr_fig.update_layout(
                    height=210, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="%"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_rr_fig, use_container_width=True)
        else:
            st.info("Real rates unavailable — requires yield_10y and breakeven_10y in dataset.")
    except Exception as _rr_e:
        st.caption(f"Real rates unavailable: {_rr_e}")

    # ── EM Credit Stress ──────────────────────────────────────────────────────
    st.subheader("EM Credit Stress (EMB/HYG)")
    st.caption(
        "**EM credit** often leads DM HY stress by 4–6 weeks. Dollar strength (DXY rising) tightens "
        "financial conditions for EM dollar-debt borrowers. The EMB/HYG price ratio tracks EM vs US HY "
        "relative performance — a falling ratio signals EM underperformance and global risk-off."
    )
    try:
        _em = load_em_credit(df)
        _em_snap = _em.get("snapshot", {})
        _em_cur = _em.get("current", {})
        _e1, _e2, _e3, _e4 = st.columns(4)
        _e1.metric("EMB/HYG Ratio", f"{_em_cur.get('em_hyg_ratio', 0):.3f}" if _em_cur.get('em_hyg_ratio') else "—")
        _e2.metric("30d Change", f"{(_em_cur.get('em_hyg_ratio_30d_chg') or 0)*100:+.1f}%")
        _e3.metric("EM Signal", f"{_em_cur.get('em_vs_dm_signal', 0):.0f}/100",
                   help="Higher = more EM stress vs DM")
        _e4.metric("EM Regime", _em_snap.get("em_stress_regime", "—"))
        if _em.get("lead_signal"):
            st.warning(_em["lead_signal"])
        if _em.get("interpretation"):
            st.caption(_em["interpretation"])
    except Exception as _em_e:
        st.caption(f"EM credit unavailable: {_em_e}")

    # ── Macro Nowcast ─────────────────────────────────────────────────────────
    st.subheader("Macro Nowcast")
    st.caption(
        "Real-time GDP growth signal from weekly/monthly indicators: unemployment, initial claims, "
        "equity momentum, yield curve slope, and PMI. Score 0–100: above 55 = expansion, below 45 = contraction signal."
    )
    try:
        _nc = load_macro_nowcast(df)
        if _nc.get("available"):
            _ncc = _nc["current"]
            _n1, _n2, _n3, _n4 = st.columns(4)
            _n1.metric("Nowcast Score", f"{_ncc.get('nowcast_score', 0):.1f}/100",
                       delta=f"{_ncc.get('nowcast_change_1m', 0):+.1f} 1M")
            _n2.metric("Regime", _ncc.get("nowcast_regime", "—"))
            _n3.metric("Momentum", _nc.get("momentum", "—"))
            _n4.metric("Recession Prob", f"{_ncc.get('nowcast_recession_prob', 0):.1%}")
            st.caption(f"Indicators: {', '.join(_ncc.get('indicators_used', []))} · {_nc.get('interpretation', '')}")
            _nc_hist = _nc.get("historical")
            if _nc_hist is not None and not _nc_hist.empty and "nowcast_score" in _nc_hist.columns:
                import plotly.graph_objects as _ncgo
                _nc_fig = _ncgo.Figure()
                _nc_fig.add_trace(_ncgo.Scatter(
                    x=pd.to_datetime(_nc_hist.index), y=_nc_hist["nowcast_score"],
                    name="Nowcast Score", line=dict(color="#27ae60", width=2),
                    fill="tozeroy", fillcolor="rgba(39,174,96,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Nowcast: %{y:.1f}<extra></extra>",
                ))
                _nc_fig.add_hline(y=55, line=dict(color="rgba(39,174,96,0.4)", dash="dash", width=1),
                                  annotation_text="Expansion", annotation_font=dict(color="#27ae60", size=10))
                _nc_fig.add_hline(y=45, line=dict(color="rgba(231,76,60,0.4)", dash="dash", width=1),
                                  annotation_text="Contraction signal", annotation_font=dict(color="#e74c3c", size=10))
                _nc_fig.update_layout(
                    height=210, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Score (0-100)", range=[0, 100]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_nc_fig, use_container_width=True)
        else:
            st.info("Macro nowcast unavailable — requires at least 2 indicators (unemployment, claims, SP500, yield curve).")
    except Exception as _nc_e:
        st.caption(f"Macro nowcast unavailable: {_nc_e}")

    # ── Volatility Risk Premium (VRP) ─────────────────────────────────────────
    st.subheader("Volatility Risk Premium (VRP)")
    st.caption(
        "**VRP = VIX − 21d realized SP500 vol.** Positive = fear premium intact (normal). "
        "Negative (inverted) = realized vol exceeds implied → fear premium collapsed → "
        "historically precedes vol spikes and credit spread widening by ~2 weeks."
    )
    try:
        _vrp = load_vrp(df)
        if _vrp.get("available"):
            _vc = _vrp["current"]
            _vp1, _vp2, _vp3, _vp4 = st.columns(4)
            _vp1.metric("VRP (21d)", f"{_vc.get('vrp_21d', 0):.1f}%",
                        help="VIX minus 21d realized vol. Negative = inverted.")
            _vp2.metric("VIX", f"{_vc.get('vix', 0):.1f}")
            _vp3.metric("Realized Vol (21d)", f"{_vc.get('realized_vol_21d', 0):.1f}%")
            _vp4.metric("Regime", _vc.get("vrp_regime", "—"))
            if _vrp.get("warning"):
                st.warning(_vrp["warning"])
            _vrp_hist = _vrp.get("historical")
            if _vrp_hist is not None and not _vrp_hist.empty and "vrp_21d" in _vrp_hist.columns:
                import plotly.graph_objects as _vrpgo
                _vrp_fig = _vrpgo.Figure()
                _vrp_fig.add_trace(_vrpgo.Scatter(
                    x=pd.to_datetime(_vrp_hist.index), y=_vrp_hist["vrp_21d"],
                    name="VRP (21d)", line=dict(color="#9b59b6", width=2),
                    fill="tozeroy", fillcolor="rgba(155,89,182,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>VRP: %{y:.1f}%<extra></extra>",
                ))
                _vrp_fig.add_hline(y=0, line_color="rgba(231,76,60,0.6)", line_width=1.5,
                                   annotation_text="Inversion threshold",
                                   annotation_font=dict(color="#e74c3c", size=10))
                _vrp_fig.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="VRP (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_vrp_fig, use_container_width=True)
        else:
            st.info("VRP unavailable — requires VIX and SP500 data.")
    except Exception as _vrp_e:
        st.caption(f"VRP unavailable: {_vrp_e}")

    # ── Credit Spread Momentum ────────────────────────────────────────────────
    st.subheader("Credit Spread Momentum")
    st.caption(
        "Rate-of-change in HY and IG spreads at 1M, 3M, and 6M horizons. "
        "Spread tightening (negative) = positive credit momentum = risk-on. "
        "Momentum divergence between HY and IG signals which segment the market favors."
    )
    try:
        _cm = load_credit_momentum(df)
        if _cm.get("available"):
            _cmc = _cm["current"]
            _m1, _m2, _m3, _m4 = st.columns(4)
            _m1.metric("HY Mom (1M)", f"{_cmc.get('hy_mom_21d', 0):+.0f} bps",
                       delta_color="inverse")
            _m2.metric("HY Mom (3M)", f"{_cmc.get('hy_mom_63d', 0):+.0f} bps",
                       delta_color="inverse")
            _m3.metric("HY Regime", _cmc.get("hy_mom_regime", "—"))
            _m4.metric("Signal", f"{_cmc.get('credit_momentum_signal', 0):.0f}/100")
            _div = _cmc.get("hy_ig_momentum_divergence", 0)
            st.caption(
                f"Trend: **{_cm['trend']}** · "
                f"HY/IG divergence: {_div:+.0f} bps · "
                f"{_cm.get('interpretation', '')}"
            )
            _cm_hist = _cm.get("historical")
            if _cm_hist is not None and not _cm_hist.empty and "hy_mom_63d" in _cm_hist.columns:
                import plotly.graph_objects as _cmgo
                _cm_fig = _cmgo.Figure()
                _cm_fig.add_trace(_cmgo.Scatter(
                    x=pd.to_datetime(_cm_hist.index), y=_cm_hist["hy_mom_63d"],
                    name="HY Mom (3M)", line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY 3M: %{y:+.0f} bps<extra></extra>",
                ))
                if "ig_mom_63d" in _cm_hist.columns:
                    _cm_fig.add_trace(_cmgo.Scatter(
                        x=pd.to_datetime(_cm_hist.index), y=_cm_hist["ig_mom_63d"],
                        name="IG Mom (3M)", line=dict(color="#3498db", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>IG 3M: %{y:+.0f} bps<extra></extra>",
                    ))
                _cm_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _cm_fig.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="bps"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_cm_fig, use_container_width=True)
        else:
            st.info("Credit momentum unavailable — requires HY or IG spread data.")
    except Exception as _cm_e:
        st.caption(f"Credit momentum unavailable: {_cm_e}")

    # ── Funding Stress (TED / OIS proxy) ─────────────────────────────────────
    st.subheader("Funding Stress (TED / OIS Proxy)")
    st.caption(
        "The **TED spread** (3m T-bill vs interbank rate) measures plumbing stress in credit markets. "
        "Here proxied as the T-bill vs Fed Funds differential. "
        "Stress in funding markets typically precedes broad HY spread widening by **3–5 weeks**."
    )
    try:
        _fs = load_funding_stress(df)
        if _fs.get("available"):
            _fsc = _fs["current"]
            _f1, _f2, _f3, _f4 = st.columns(4)
            _f1.metric("TED Proxy (bps)", f"{_fsc.get('ted_spread_proxy_bps', 0):.1f}")
            _f2.metric("Z-Score (1y)", f"{_fsc.get('ted_spread_zscore', 0):.2f}")
            _f3.metric("Funding Regime", _fsc.get("funding_regime", "—"))
            _f4.metric("Signal", f"{_fsc.get('funding_stress_signal', 0):.0f}/100")
            if _fs.get("warning"):
                st.error(_fs["warning"])
            if _fs.get("interpretation"):
                st.caption(_fs["interpretation"])
            _fs_hist = _fs.get("historical")
            if _fs_hist is not None and not _fs_hist.empty and "ted_spread_proxy" in _fs_hist.columns:
                import plotly.graph_objects as _fsgo
                _fs_fig = _fsgo.Figure()
                _fs_fig.add_trace(_fsgo.Scatter(
                    x=pd.to_datetime(_fs_hist.index), y=_fs_hist["ted_spread_proxy"],
                    name="TED Proxy", line=dict(color="#e67e22", width=2),
                    fill="tozeroy", fillcolor="rgba(230,126,34,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>TED: %{y:.1f} bps<extra></extra>",
                ))
                for _thresh, _col, _lbl in [(100, "#e74c3c", "Crisis"), (50, "#e67e22", "Stressed"), (25, "#f39c12", "Elevated")]:
                    _fs_fig.add_hline(y=_thresh, line=dict(color=_col, dash="dash", width=1),
                                      annotation_text=_lbl, annotation_font=dict(color=_col, size=9))
                _fs_fig.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="bps"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fs_fig, use_container_width=True)
        else:
            st.info("Funding stress unavailable — requires yield_3m and Fed Funds rate (dff) in dataset.")
    except Exception as _fs_e:
        st.caption(f"Funding stress unavailable: {_fs_e}")

with tab2:
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots
    from src.backtester import OOS_CUTOFF as _OOS_CUTOFF

    # ── Shared dark layout helper ─────────────────────────────────────────────
    def _dlayout(**kw):
        base = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="system-ui, sans-serif", color="#9aa0aa", size=11),
            margin=dict(l=8, r=8, t=40, b=8),
            xaxis=dict(showgrid=False, color="#6b7280", linecolor="#1e2435",
                       tickcolor="#1e2435"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                       color="#6b7280", linecolor="#1e2435", tickcolor="#1e2435"),
            legend=dict(orientation="h", y=1.08, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                            font=dict(color="#e2e8f0")),
        )
        base.update(kw)
        return base

    _df2 = df.copy()
    _df2["date"] = pd.to_datetime(_df2["date"])

    # ── Credit Cycle Clock ────────────────────────────────────────────────────
    st.subheader("Credit Cycle Clock")
    st.caption(
        "Circular map of the credit cycle phase derived from the composite risk score. "
        "Dot = today · trail = last 90 days. "
        "Early Expansion → Late Cycle → Contraction → Recovery."
    )
    try:
        _clock_fig = build_credit_cycle_clock(df)
        _clk_col1, _clk_col2 = st.columns([1, 1])
        with _clk_col1:
            st.plotly_chart(_clock_fig, use_container_width=True)
        with _clk_col2:
            st.markdown(
                '<div style="padding:16px">'
                '<p style="color:#9aa0aa;font-size:0.85rem;line-height:1.7">'
                '<strong style="color:#27ae60">🟢 Early Expansion (315°–45°)</strong><br>'
                'Score 0–30 · Spreads recovering, macro improving, low stress.<br><br>'
                '<strong style="color:#f1c40f">🟡 Late Cycle (45°–135°)</strong><br>'
                'Score 30–50 · Complacency rising, spreads near tight, momentum slowing.<br><br>'
                '<strong style="color:#e74c3c">🔴 Contraction (135°–225°)</strong><br>'
                'Score 50–70 · Stress rising, spreads widening, macro deteriorating.<br><br>'
                '<strong style="color:#3498db">🔵 Recovery (225°–315°)</strong><br>'
                'Score 70–100 · Stress near peak, spreads at highs, macro bottoming.'
                '</p></div>',
                unsafe_allow_html=True,
            )
    except Exception as _clk_e:
        st.caption(f"Clock unavailable: {_clk_e}")

    st.divider()

    # ── 1. Composite Risk Score ───────────────────────────────────────────────
    st.subheader("Composite Risk Score")
    _fig1 = _go.Figure()
    _fig1.add_hrect(y0=70, y1=100, fillcolor="rgba(231,76,60,0.10)", line_width=0,
                    annotation_text="Elevated Risk", annotation_position="top left",
                    annotation_font=dict(color="rgba(231,76,60,0.7)", size=10))
    _fig1.add_hrect(y0=50, y1=70, fillcolor="rgba(230,126,34,0.08)", line_width=0,
                    annotation_text="Caution", annotation_position="top left",
                    annotation_font=dict(color="rgba(230,126,34,0.7)", size=10))
    _fig1.add_trace(_go.Scatter(
        x=_df2["date"], y=_df2["composite_risk_score_smooth"],
        mode="lines", name="Composite (smooth)",
        line=dict(color="#4f8ef7", width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
    ))
    _fig1.add_trace(_go.Scatter(
        x=_df2["date"], y=_df2["composite_risk_score"],
        mode="lines", name="Composite (raw)",
        line=dict(color="#4f8ef7", width=1, dash="dot"),
        opacity=0.4,
    ))
    _fig1.update_layout(**_dlayout(height=280,
        yaxis=dict(range=[0, 100], showgrid=True,
                   gridcolor="rgba(255,255,255,0.06)",
                   color="#6b7280", title=None)))
    st.plotly_chart(_fig1, use_container_width=True)

    # ── 2. Component Scores ───────────────────────────────────────────────────
    st.subheader("Component Scores")
    _components = [
        ("macro_risk_score_smooth",            "Macro Risk",      "#e74c3c"),
        ("credit_market_risk_score_smooth",    "Credit Risk",     "#e67e22"),
        ("complacency_score_smooth",           "Complacency",     "#f1c40f"),
        ("risk_appetite_score_smooth",         "Risk Appetite",   "#27ae60"),
        ("mean_reversion_score_smooth",        "Mean Reversion",  "#9b59b6"),
        ("treasury_stress_score_smooth",       "Treasury Stress", "#1abc9c"),
        ("liquidity_regime_score_smooth",      "Liquidity",       "#3498db"),
    ]
    _fig2 = _go.Figure()
    for _col, _name, _color in _components:
        if _col in _df2.columns:
            _fig2.add_trace(_go.Scatter(
                x=_df2["date"], y=_df2[_col],
                mode="lines", name=_name,
                line=dict(color=_color, width=1.8),
            ))
    _fig2.update_layout(**_dlayout(height=300,
        yaxis=dict(range=[0, 100], showgrid=True,
                   gridcolor="rgba(255,255,255,0.06)",
                   color="#6b7280", title=None)))
    st.plotly_chart(_fig2, use_container_width=True)

    # ── 3+4. HY Spread & VIX  |  SP500 & Drawdown ────────────────────────────
    _c3, _c4 = st.columns(2)

    with _c3:
        st.subheader("HY Spread & VIX")
        _fig3 = _make_subplots(specs=[[{"secondary_y": True}]])
        _fig3.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2["hy_spread"],
            name="HY Spread", line=dict(color="#e74c3c", width=2),
        ), secondary_y=False)
        _fig3.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2["vix"],
            name="VIX", line=dict(color="#e67e22", width=1.8, dash="dot"),
        ), secondary_y=True)
        _fig3.update_layout(**_dlayout(height=280,
            legend=dict(orientation="h", y=1.1, x=0,
                        bgcolor="rgba(0,0,0,0)")))
        _fig3.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", linecolor="#1e2435",
                           title_text="HY Spread %", secondary_y=False)
        _fig3.update_yaxes(showgrid=False, color="#6b7280",
                           title_text="VIX", secondary_y=True)
        _fig3.update_xaxes(showgrid=False, color="#6b7280", linecolor="#1e2435")
        st.plotly_chart(_fig3, use_container_width=True)

    with _c4:
        st.subheader("SP500 & Drawdown")
        _fig4 = _make_subplots(specs=[[{"secondary_y": True}]])
        _fig4.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2["sp500"],
            name="SP500", line=dict(color="#27ae60", width=2),
            fill="tozeroy", fillcolor="rgba(39,174,96,0.06)",
        ), secondary_y=False)
        _fig4.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2["sp500_drawdown"] * 100,
            name="Drawdown %", line=dict(color="#e74c3c", width=1.5),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
        ), secondary_y=True)
        _fig4.update_layout(**_dlayout(height=280,
            legend=dict(orientation="h", y=1.1, x=0,
                        bgcolor="rgba(0,0,0,0)")))
        _fig4.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", linecolor="#1e2435",
                           title_text="SP500", secondary_y=False)
        _fig4.update_yaxes(showgrid=False, color="#e74c3c",
                           title_text="Drawdown %", secondary_y=True)
        _fig4.update_xaxes(showgrid=False, color="#6b7280", linecolor="#1e2435")
        st.plotly_chart(_fig4, use_container_width=True)

    # ── 5+6. Credit Impulse  |  Risk Appetite vs Complacency ─────────────────
    _c5, _c6 = st.columns(2)

    with _c5:
        st.subheader("Credit Impulse")
        _ci = _df2["credit_impulse"].fillna(0)
        _fig5 = _go.Figure()
        _fig5.add_trace(_go.Bar(
            x=_df2["date"], y=_ci,
            marker_color=[
                "#27ae60" if v < 0 else "#e74c3c" for v in _ci
            ],
            name="Credit Impulse",
            marker_line_width=0,
        ))
        _fig5.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
        _fig5.update_layout(**_dlayout(height=280, showlegend=False))
        st.plotly_chart(_fig5, use_container_width=True)

    with _c6:
        st.subheader("Risk Appetite vs Complacency")
        _fig6 = _go.Figure()
        _fig6.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2["risk_appetite_score_smooth"],
            name="Risk Appetite", line=dict(color="#27ae60", width=2),
        ))
        _fig6.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2["complacency_score_smooth"],
            name="Complacency", line=dict(color="#f1c40f", width=2),
            fill="tonexty", fillcolor="rgba(241,196,15,0.05)",
        ))
        _fig6.update_layout(**_dlayout(height=280,
            yaxis=dict(range=[0, 100], showgrid=True,
                       gridcolor="rgba(255,255,255,0.06)",
                       color="#6b7280", title=None)))
        st.plotly_chart(_fig6, use_container_width=True)

    # ── 7. Treasury Stress & Macro vs Credit ─────────────────────────────────
    st.subheader("Treasury Stress, Macro Risk & Credit Risk")
    _fig7 = _go.Figure()
    for _col, _name, _color, _dash in [
        ("treasury_stress_score_smooth", "Treasury Stress", "#1abc9c", "solid"),
        ("macro_risk_score_smooth",      "Macro Risk",      "#e74c3c", "dot"),
        ("credit_market_risk_score_smooth", "Credit Risk",  "#e67e22", "dash"),
    ]:
        _fig7.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2[_col],
            name=_name, line=dict(color=_color, width=2, dash=_dash),
        ))
    _fig7.update_layout(**_dlayout(height=260,
        yaxis=dict(range=[0, 100], showgrid=True,
                   gridcolor="rgba(255,255,255,0.06)",
                   color="#6b7280", title=None)))
    st.plotly_chart(_fig7, use_container_width=True)

    # ── 8. Cross-Asset Stress Scores ─────────────────────────────────────────
    _cross_asset_cols = [
        ("rates_stress_score_smooth",              "Rates Stress",     "#1abc9c"),
        ("enhanced_funding_stress_score_smooth",   "Funding Stress",   "#9b59b6"),
        ("fx_commodity_score_smooth",              "FX / Commodity",   "#e67e22"),
        ("banking_stress_score_smooth",            "Banking Stress",   "#e74c3c"),
    ]
    _ca_present = [t for t in _cross_asset_cols if t[0] in _df2.columns]
    if _ca_present:
        st.subheader("Cross-Asset Stress Scores")
        _fig_ca = _go.Figure()
        for _col, _name, _color in _ca_present:
            _fig_ca.add_trace(_go.Scatter(
                x=_df2["date"], y=_df2[_col],
                mode="lines", name=_name,
                line=dict(color=_color, width=2),
            ))
        _fig_ca.add_hline(y=50, line_color="rgba(230,126,34,0.4)",
                          line_dash="dot", line_width=1,
                          annotation_text="Caution", annotation_position="right",
                          annotation_font=dict(color="rgba(230,126,34,0.7)", size=9))
        _fig_ca.update_layout(**_dlayout(height=270,
            yaxis=dict(range=[0, 100], showgrid=True,
                       gridcolor="rgba(255,255,255,0.06)",
                       color="#6b7280", title=None),
            title=dict(text="Each score 0–100 | Higher = more stress",
                       font=dict(size=10, color="#4b5563"), x=0)))
        st.plotly_chart(_fig_ca, use_container_width=True)

    # ── 9. MOVE Index (bond vol) ─────────────────────────────────────────────
    if "move_index" in _df2.columns:
        st.subheader("MOVE Index — Bond Market Implied Vol")
        _fig_mv = _go.Figure()
        _fig_mv.add_hrect(y0=150, y1=_df2["move_index"].max() + 10,
                          fillcolor="rgba(231,76,60,0.08)", line_width=0,
                          annotation_text="Crisis (>150)", annotation_position="top left",
                          annotation_font=dict(color="rgba(231,76,60,0.6)", size=9))
        _fig_mv.add_hrect(y0=100, y1=150,
                          fillcolor="rgba(230,126,34,0.06)", line_width=0,
                          annotation_text="Elevated (>100)", annotation_position="top left",
                          annotation_font=dict(color="rgba(230,126,34,0.6)", size=9))
        _fig_mv.add_trace(_go.Scatter(
            x=_df2["date"], y=_df2["move_index"],
            mode="lines", name="MOVE",
            line=dict(color="#4f8ef7", width=1.8),
            fill="tozeroy", fillcolor="rgba(79,142,247,0.07)",
        ))
        _fig_mv.update_layout(**_dlayout(height=250,
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                       color="#6b7280", title=None)))
        st.plotly_chart(_fig_mv, use_container_width=True)

    # ── 10. Credit Spread Complex (IG / BBB / HY) ────────────────────────────
    _has_ig  = "ig_spread"  in _df2.columns and _df2["ig_spread"].notna().sum() > 50
    _has_bbb = "bbb_spread" in _df2.columns and _df2["bbb_spread"].notna().sum() > 50
    if _has_ig or _has_bbb:
        st.subheader("Credit Spread Complex — IG / BBB / HY")
        st.caption(
            "IG OAS = investment-grade universe; BBB OAS = bottom rung of IG (cliff risk); "
            "HY OAS = high-yield. BBB diverging from IG signals fallen-angel pressure."
        )
        _ca, _cb = st.columns(2)

        with _ca:
            _fig_cs = _go.Figure()
            _fig_cs.add_trace(_go.Scatter(
                x=_df2["date"], y=_df2["hy_spread"],
                name="HY OAS", line=dict(color="#e74c3c", width=2),
            ))
            if _has_bbb:
                _fig_cs.add_trace(_go.Scatter(
                    x=_df2["date"], y=_df2["bbb_spread"],
                    name="BBB OAS", line=dict(color="#e67e22", width=1.8, dash="dash"),
                ))
            if _has_ig:
                _fig_cs.add_trace(_go.Scatter(
                    x=_df2["date"], y=_df2["ig_spread"],
                    name="IG OAS", line=dict(color="#27ae60", width=1.8, dash="dot"),
                ))
            _fig_cs.update_layout(**_dlayout(height=280,
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="OAS %")))
            st.plotly_chart(_fig_cs, use_container_width=True)

        with _cb:
            if _has_ig:
                _fig_ratio = _make_subplots(specs=[[{"secondary_y": True}]])
                _hy_ig = (_df2["hy_spread"] / _df2["ig_spread"].replace(0, float("nan")))
                _fig_ratio.add_trace(_go.Scatter(
                    x=_df2["date"], y=_hy_ig,
                    name="HY/IG Ratio", line=dict(color="#4f8ef7", width=2),
                ), secondary_y=False)
                if _has_bbb:
                    _fig_ratio.add_trace(_go.Scatter(
                        x=_df2["date"], y=_df2["bbb_spread"] - _df2["ig_spread"],
                        name="BBB excess vs IG", line=dict(color="#e67e22", width=1.5, dash="dash"),
                    ), secondary_y=True)
                _fig_ratio.update_layout(**_dlayout(height=280))
                _fig_ratio.update_yaxes(title_text="HY/IG Ratio", showgrid=True,
                                        gridcolor="rgba(255,255,255,0.06)",
                                        color="#6b7280", secondary_y=False)
                _fig_ratio.update_yaxes(title_text="BBB excess (pp)", showgrid=False,
                                        color="#e67e22", secondary_y=True)
                _fig_ratio.update_xaxes(showgrid=False, color="#6b7280")
                st.caption("HY/IG ratio rising = stress is idiosyncratic to lower-quality credit. "
                           "BBB excess widening = fallen-angel risk rising.")
                st.plotly_chart(_fig_ratio, use_container_width=True)
            else:
                st.info("IG OAS data unavailable — run pipeline to fetch BAMLC0A0CM.")

    # ── 11. SLOOS — Bank Lending Standards ───────────────────────────────────
    if "sloos_ci" in _df2.columns and _df2["sloos_ci"].notna().sum() > 10:
        st.subheader("SLOOS — Bank C&I Lending Standards")
        st.caption(
            "Net % of banks tightening credit standards for C&I loans. "
            "Positive = tightening (credit supply contracting). Leads default rates by 2–4 quarters."
        )
        _sloos = _df2["sloos_ci"].dropna()
        _fig_sl = _go.Figure()
        _fig_sl.add_hrect(y0=20, y1=max(100, float(_sloos.max()) + 5),
                          fillcolor="rgba(231,76,60,0.07)", line_width=0,
                          annotation_text="Significant Tightening",
                          annotation_position="top left",
                          annotation_font=dict(color="rgba(231,76,60,0.6)", size=9))
        _fig_sl.add_hline(y=0, line_color="rgba(255,255,255,0.2)",
                          line_width=1, line_dash="dot")
        _fig_sl.add_trace(_go.Bar(
            x=_df2["date"], y=_df2["sloos_ci"].fillna(0),
            marker_color=[
                "#e74c3c" if v > 10 else "#27ae60" if v < -5 else "#6b7280"
                for v in _df2["sloos_ci"].fillna(0)
            ],
            name="SLOOS C&I",
            marker_line_width=0,
        ))
        _fig_sl.update_layout(**_dlayout(height=240, showlegend=False))
        st.plotly_chart(_fig_sl, use_container_width=True)

    # ── 12. Spread Regime Positioning Map ────────────────────────────────────
    if "hy_spread" in _df2.columns and "hy_change_30d" in _df2.columns:
        st.subheader("Spread Regime Positioning Map")
        st.caption(
            "Where are HY spreads today relative to the past 5 years? "
            "X axis = rolling spread percentile (0 = tightest, 100 = widest). "
            "Y axis = 30-day momentum (positive = widening). "
            "Historical points colored by model regime."
        )
        _pos_src = _df2[["date", "hy_spread", "hy_change_30d"]].copy()
        _pos_src["regime"] = _df2["grouped_regime"] if "grouped_regime" in _df2.columns else "Unknown"
        _pos_src = _pos_src.dropna(subset=["hy_spread", "hy_change_30d"])

        if len(_pos_src) >= 100:
            _lookback = min(len(_pos_src), 252 * 5)
            _pos_w = _pos_src.tail(_lookback).copy()
            _pos_w["pct_rank"] = _pos_w["hy_spread"].rank(pct=True) * 100
            _curr_pct = float(_pos_w["pct_rank"].iloc[-1])
            _curr_mom = float(_pos_w["hy_change_30d"].iloc[-1])
            _max_y = max(3.0, float(_pos_w["hy_change_30d"].abs().max()) + 0.3)

            _reg_col = {"Risk-On": "#27ae60", "Neutral": "#f1c40f",
                        "Caution": "#e67e22", "Risk-Off": "#e74c3c", "Unknown": "#6b7280"}
            _fig_pos = _go.Figure()
            _fig_pos.add_hrect(y0=0, y1=_max_y, fillcolor="rgba(231,76,60,0.04)", line_width=0)
            _fig_pos.add_hrect(y0=-_max_y, y1=0, fillcolor="rgba(39,174,96,0.04)", line_width=0)
            _fig_pos.add_vline(x=50, line_color="rgba(255,255,255,0.12)", line_width=1, line_dash="dot")
            _fig_pos.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1, line_dash="dot")

            for _reg, _grp in _pos_w.groupby("regime"):
                _fig_pos.add_trace(_go.Scatter(
                    x=_grp["pct_rank"], y=_grp["hy_change_30d"],
                    mode="markers", name=_reg,
                    marker=dict(color=_reg_col.get(_reg, "#6b7280"), size=3, opacity=0.3),
                    hovertemplate="%{customdata}<br>Pct: %{x:.0f}<br>Δ30d: %{y:+.2f}pp<extra></extra>",
                    customdata=_grp["date"].astype(str).values,
                ))

            _fig_pos.add_trace(_go.Scatter(
                x=[_curr_pct], y=[_curr_mom],
                mode="markers+text", name="Now",
                marker=dict(color="#ffffff", size=14, symbol="star",
                            line=dict(color="#4f8ef7", width=1.5)),
                text=["Now"], textposition="top center",
                textfont=dict(color="#ffffff", size=10),
            ))
            _fig_pos.update_layout(**_dlayout(
                height=340,
                xaxis=dict(title="HY Spread Percentile (5yr rolling)",
                           range=[0, 100], showgrid=False, color="#6b7280"),
                yaxis=dict(title="30-Day Spread Change (pp)",
                           range=[-_max_y, _max_y],
                           showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
            ))
            st.plotly_chart(_fig_pos, use_container_width=True)

            _q1, _q2, _q3, _q4 = st.columns(4)
            _q1.metric("HY Spread", f"{float(_df2['hy_spread'].iloc[-1]):.2f}%")
            _q2.metric("5yr Percentile", f"{_curr_pct:.0f}th")
            _q3.metric("30d Momentum", f"{_curr_mom:+.2f}pp")
            _q4.metric(
                "Quadrant",
                ("Wide + Widening" if _curr_pct >= 50 and _curr_mom > 0 else
                 "Wide + Tightening" if _curr_pct >= 50 else
                 "Tight + Widening" if _curr_mom > 0 else "Tight + Tightening"),
            )

    # ── 13. Credit Cycle Indicators ──────────────────────────────────────────
    _has_def = "hy_default_rate" in _df2.columns and _df2["hy_default_rate"].notna().sum() > 10
    if _has_def or ("sloos_ci" in _df2.columns and _df2["sloos_ci"].notna().sum() > 10):
        st.subheader("Credit Cycle Indicators")
        st.caption(
            "SLOOS leads default rates by 2–4 quarters. "
            "Watch for SLOOS tightening followed by default rate rises as the classic credit cycle turn."
        )
        _cc1, _cc2 = st.columns(2)
        with _cc1:
            if _has_def:
                _dr = _df2[["date", "hy_default_rate"]].dropna()
                _fig_dr = _go.Figure()
                _fig_dr.add_hrect(
                    y0=4, y1=max(15.0, float(_dr["hy_default_rate"].max()) + 1),
                    fillcolor="rgba(231,76,60,0.07)", line_width=0,
                    annotation_text="Stress >4%", annotation_position="top left",
                    annotation_font=dict(color="rgba(231,76,60,0.6)", size=9),
                )
                _fig_dr.add_trace(_go.Scatter(
                    x=_dr["date"], y=_dr["hy_default_rate"],
                    mode="lines", name="HY Default Rate",
                    line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
                ))
                _fig_dr.update_layout(**_dlayout(
                    height=230,
                    yaxis=dict(title="Default Rate %", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                ))
                st.caption("Moody's 12m Trailing HY Default Rate")
                st.plotly_chart(_fig_dr, use_container_width=True)
            else:
                st.info("HY default rate not loaded. Add DHMF1Y to market_data.py to enable.")

        with _cc2:
            if "sloos_ci" in _df2.columns and "hy_spread" in _df2.columns:
                _cyc = _df2[["date", "sloos_ci", "hy_spread"]].dropna(subset=["sloos_ci"])
                _cyc = _cyc.copy()
                _cyc["sloos_norm"] = (_cyc["sloos_ci"] - _cyc["sloos_ci"].mean()) / (_cyc["sloos_ci"].std() + 1e-9)
                _cyc["spread_norm"] = (_cyc["hy_spread"] - _cyc["hy_spread"].mean()) / (_cyc["hy_spread"].std() + 1e-9)
                _fig_cyc = _go.Figure()
                _fig_cyc.add_trace(_go.Scatter(
                    x=_cyc["date"], y=_cyc["sloos_norm"],
                    name="SLOOS (z-score)", line=dict(color="#e67e22", width=1.8),
                ))
                _fig_cyc.add_trace(_go.Scatter(
                    x=_cyc["date"], y=_cyc["spread_norm"],
                    name="HY Spread (z-score)", line=dict(color="#e74c3c", width=1.8, dash="dot"),
                ))
                _fig_cyc.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig_cyc.update_layout(**_dlayout(
                    height=230,
                    yaxis=dict(title="Std Deviations", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                ))
                st.caption("SLOOS vs HY Spread — normalized (SLOOS leads by ~3 quarters)")
                st.plotly_chart(_fig_cyc, use_container_width=True)

    # ── 14. Strategy vs SP500 Equity Curve ───────────────────────────────────
    if "strategy_equity_curve" in _df2.columns:
        st.subheader("Strategy vs SP500 — Cumulative Return (Equity)")
        st.caption(f"In-sample = before {_OOS_CUTOFF}. Credit total return benchmarks are in the Backtest tab.")
        _fig8 = _go.Figure()
        _fig8.add_trace(_go.Scatter(
            x=_df2["date"], y=(_df2["strategy_equity_curve"] - 1) * 100,
            name="Strategy", line=dict(color="#4f8ef7", width=2.5),
            fill="tozeroy", fillcolor="rgba(79,142,247,0.07)",
        ))
        _fig8.add_trace(_go.Scatter(
            x=_df2["date"], y=(_df2["sp500_equity_curve"] - 1) * 100,
            name="SP500", line=dict(color="#6b7280", width=1.8, dash="dot"),
        ))
        _fig8.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        _fig8.update_layout(**_dlayout(height=280,
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                       color="#6b7280", title="Return %")))
        st.plotly_chart(_fig8, use_container_width=True)

    # ── Credit Quality Curve ──────────────────────────────────────────────────
    st.subheader("Credit Quality Curve")
    st.caption(
        "The **credit quality curve** measures the spread premium demanded at each rating tier: "
        "IG → BBB → HY. A **steep curve** (large HY-IG differential) signals risk aversion — "
        "investors demand extra compensation for credit risk. A **flat/inverted curve** signals complacency. "
        "The z-score is relative to the trailing 252-day history."
    )
    try:
        _qc = load_quality_curve(df)
        if _qc.get("available"):
            _qcc = _qc["current"]
            _qc1, _qc2, _qc3, _qc4 = st.columns(4)
            _qc1.metric("HY Spread", f"{_qcc.get('hy_spread', float('nan')):.2f}%")
            _qc2.metric("HY−IG Premium", f"{_qcc.get('hy_ig_premium', float('nan')):.2f}pp")
            _qc3.metric("Curve Z-Score", f"{_qcc.get('curve_slope_zscore', float('nan')):.2f}",
                        help="High z-score = steep curve = elevated risk aversion")
            _qc4.metric("Interpretation", _qcc.get("interpretation", "—"))

            if "df" in _qc and "hy_ig_premium" in _qc["df"].columns:
                _qc_df = _qc["df"].copy()
                _qc_df["date"] = pd.to_datetime(_qc_df["date"])
                _qcfig = _go.Figure()
                if "bbb_ig_premium" in _qc_df.columns:
                    _qcfig.add_trace(_go.Scatter(
                        x=_qc_df["date"], y=_qc_df["bbb_ig_premium"],
                        name="BBB−IG Premium", line=dict(color="#f39c12", width=1.5),
                    ))
                _qcfig.add_trace(_go.Scatter(
                    x=_qc_df["date"], y=_qc_df["hy_ig_premium"],
                    name="HY−IG Premium", line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
                ))
                _qcfig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _qcfig.update_layout(**_dlayout(
                    height=240,
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Spread Premium (pp)"),
                ))
                st.plotly_chart(_qcfig, use_container_width=True)
        else:
            st.info("Credit quality curve unavailable — requires HY spread data in dataset.")
    except Exception as _qc_e:
        st.caption(f"Quality curve unavailable: {_qc_e}")

    # ── Rates & Credit Term Structure ────────────────────────────────────────
    st.subheader("Rates & Credit Term Structure")
    st.caption(
        "The **yield curve** (Treasury 3m/2y/10y) and **credit quality slope** (IG → HY spread differential) "
        "jointly describe where stress sits in the maturity spectrum. "
        "Front-end credit stress = near-term liquidity fear. Back-end = solvency concerns. "
        "Negative 2s10s = inverted curve = classic recession signal."
    )
    try:
        _ts = load_term_structure(df)
        if _ts.get("available"):
            _tsc = _ts.get("current", {})
            _ts1, _ts2, _ts3, _ts4 = st.columns(4)
            _ts1.metric("2s10s Slope", f"{_tsc.get('ts_curve_slope_2s10s', float('nan')):+.2f}pp",
                        help="10y − 2y Treasury spread. Negative = inverted (recession signal)")
            _ts2.metric("3m10y Slope", f"{_tsc.get('ts_curve_slope_3m10y', float('nan')):+.2f}pp",
                        help="10y − 3m Treasury spread. Estrella-Mishkin recession predictor")
            _ts3.metric("HY−IG Credit Slope", f"{_tsc.get('ts_credit_slope_hy_ig', float('nan')):+.2f}pp",
                        help="HY OAS minus IG OAS. High = steep credit quality curve = risk aversion")
            _ts4.metric("Treasury Regime", _tsc.get("ts_curve_regime", "—"))

            if _tsc.get("interpretation"):
                st.caption(_tsc["interpretation"])

            # Percentile ranks
            _ts_pct = _ts.get("history_percentiles", {})
            if _ts_pct:
                _pct_cols = st.columns(len(_ts_pct))
                for _pc, (_pk, _pv) in zip(_pct_cols, _ts_pct.items()):
                    _short_k = _pk.replace("ts_", "").replace("_zscore", " z").replace("_", " ").title()
                    _pc.metric(f"{_short_k} Pctile", f"{_pv:.0f}th" if _pv is not None else "—",
                               help=f"Current value's percentile vs full history")

            # Time series of 2s10s and HY-IG slope
            if "df" in _ts and "ts_curve_slope_2s10s" in _ts["df"].columns:
                import plotly.graph_objects as _tsgo
                _ts_df = _ts["df"].copy()
                _ts_df["date"] = pd.to_datetime(_ts_df["date"])
                _ts_fig = _tsgo.Figure()
                _ts_fig.add_trace(_tsgo.Scatter(
                    x=_ts_df["date"], y=_ts_df["ts_curve_slope_2s10s"],
                    name="2s10s (Treasury)", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>2s10s: %{y:+.2f}pp<extra></extra>",
                ))
                if "ts_credit_slope_hy_ig" in _ts_df.columns:
                    _ts_fig.add_trace(_tsgo.Scatter(
                        x=_ts_df["date"], y=_ts_df["ts_credit_slope_hy_ig"],
                        name="HY−IG (Credit)", line=dict(color="#e74c3c", width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>HY−IG: %{y:+.2f}pp<extra></extra>",
                    ))
                _ts_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _ts_fig.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Spread (pp)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_ts_fig, use_container_width=True)
                st.caption("Below zero = inverted. 2s10s inversion has preceded every US recession since 1970.")
        else:
            st.info("Term structure unavailable — requires Treasury yield columns (yield_3m, yield_10y).")
    except Exception as _ts_e:
        st.caption(f"Term structure unavailable: {_ts_e}")

with tab3:
    st.header("Portfolio Stance")

    _pw = generate_portfolio_weights(latest)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equity Weight",  f"{_pw['equity_weight']:.0%}")
    col2.metric("Credit Weight",  f"{_pw['credit_weight']:.0%}")
    col3.metric("Cash Weight",    f"{_pw['cash_weight']:.0%}")
    col4.metric("Duration Bias",  _pw["duration_bias"])

    # Credit allocation breakdown
    _hy_w = _pw.get("hy_weight", 0.0)
    _ig_w = _pw.get("ig_weight", 0.0)
    _dur  = _pw.get("duration_target", 6.0)
    if _hy_w + _ig_w > 0:
        st.markdown("**Credit Allocation Detail**")
        _dc1, _dc2, _dc3, _dc4 = st.columns(4)
        _dc1.metric("HY Weight",       f"{_hy_w:.1%}",
                    delta=f"{_hy_w / (_hy_w + _ig_w):.0%} of credit" if (_hy_w + _ig_w) > 0 else None)
        _dc2.metric("IG Weight",       f"{_ig_w:.1%}",
                    delta=f"{_ig_w / (_hy_w + _ig_w):.0%} of credit" if (_hy_w + _ig_w) > 0 else None)
        _dc3.metric("Duration Target", f"{_dur:.1f} yrs")
        _dc4.metric("HY/IG Split",
                    f"{_hy_w / (_hy_w + _ig_w):.0%} / {_ig_w / (_hy_w + _ig_w):.0%}"
                    if (_hy_w + _ig_w) > 0 else "N/A")

    st.subheader("Decision Logic")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Environment:** {environment}")
    st.write(f"**Action:** {action}")
    st.write(f"**Buy Trigger:** {latest.get('buy_trigger', 'N/A')}")
    st.write(f"**Risk-Off Trigger:** {latest.get('risk_off_trigger', 'N/A')}")

    # ── Crisis Similarity ─────────────────────────────────────────────────────
    st.subheader("Historical Analog Similarity")
    st.caption(
        "Euclidean distance between today's component scores and "
        "five historical regime archetypes. Higher = more similar."
    )

    _similarity = compute_crisis_similarity(latest)
    _top_analog  = _similarity[0][0]
    _top_score   = _similarity[0][1]

    _sim_cols = st.columns(len(_similarity))
    for _col, (_name, _score) in zip(_sim_cols, _similarity):
        _color = "#27ae60" if _score >= 70 else "#e67e22" if _score >= 40 else "#95a5a6"
        _col.markdown(
            f"""<div style="text-align:center;padding:12px 8px;border-radius:8px;
                            background:#1a1f2e;border:1px solid #2d3550">
                  <div style="font-size:11px;color:#aaa;margin-bottom:6px">{_name}</div>
                  <div style="font-size:24px;font-weight:700;color:{_color}">{_score:.0f}</div>
                  <div style="font-size:10px;color:#888">/ 100</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"Closest analog: **{_top_analog}** (similarity {_top_score:.0f}/100)",
        help="Similarity = max(0, 100 − L2 distance) across macro, credit, "
             "complacency, mean-reversion, and treasury stress scores.",
    )

    # ── Model Health Check ────────────────────────────────────────────────────
    st.subheader("Model Health")

    _score_cols = [
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
    _required_cols = _score_cols + ["final_decision"]

    _missing  = check_missing_values(df, _required_cols)
    _bounds   = check_score_bounds(df, _score_cols)
    _samples  = check_sample_sizes(df, "final_decision")

    _missing_issues  = {k: v for k, v in _missing.items() if v != 0}
    _bounds_ok   = _bounds == "PASS"
    _samples_ok  = _samples == "PASS"
    _missing_ok  = len(_missing_issues) == 0

    _hc1, _hc2, _hc3 = st.columns(3)
    _hc1.metric("Score Bounds",    "✓ PASS" if _bounds_ok  else "✗ FAIL",
                delta=None if _bounds_ok  else "out-of-range values detected")
    _hc2.metric("Missing Values",  "✓ PASS" if _missing_ok else "✗ FAIL",
                delta=None if _missing_ok else f"{len(_missing_issues)} column(s) flagged")
    _hc3.metric("Regime Samples",  "✓ PASS" if _samples_ok else "⚠ WARN",
                delta=None if _samples_ok else "some regimes < 30 obs")

    if not _bounds_ok:
        with st.expander("Score bounds detail"):
            st.json(_bounds)
    if not _missing_ok:
        with st.expander("Missing values detail"):
            st.json(_missing_issues)
    if not _samples_ok:
        with st.expander("Regime sample sizes"):
            st.json(_samples)

with tab5:  # Analytics — 61 sub-tabs
    (_analytics_sub1, _analytics_sub2, _analytics_sub3, _analytics_sub4,
     _analytics_sub5, _analytics_sub6, _analytics_sub7, _analytics_sub8,
     _analytics_sub9, _analytics_sub10, _analytics_sub11,
     _analytics_sub12, _analytics_sub13, _analytics_sub14,
     _analytics_sub15, _analytics_sub16, _analytics_sub17, _analytics_sub18,
     _analytics_sub19, _analytics_sub20, _analytics_sub21, _analytics_sub22,
     _analytics_sub23, _analytics_sub24, _analytics_sub25,
     _analytics_sub26, _analytics_sub27, _analytics_sub28,
     _analytics_sub29, _analytics_sub30, _analytics_sub31,
     _analytics_sub32, _analytics_sub33, _analytics_sub34,
     _analytics_sub35, _analytics_sub36, _analytics_sub37,
     _analytics_sub38, _analytics_sub39, _analytics_sub40,
     _analytics_sub41, _analytics_sub42, _analytics_sub43,
     _analytics_sub44, _analytics_sub45, _analytics_sub46,
     _analytics_sub47, _analytics_sub48, _analytics_sub49,
     _analytics_sub50, _analytics_sub51, _analytics_sub52,
     _analytics_sub53, _analytics_sub54, _analytics_sub55,
     _analytics_sub56, _analytics_sub57, _analytics_sub58,
     _analytics_sub59, _analytics_sub60, _analytics_sub61) = st.tabs([
        "Validation", "Attribution", "Timeline", "Sig Decay",
        "Ortho", "Tail Risk", "Stress", "Performance", "Factors",
        "Regime Validity", "Failure Analysis",
        "Contagion", "Analogs", "Persistence",
        "Merton DD", "Frontier", "Kelly", "Granger", "Defaults",
        "Fwd Sim", "CDX Proxy", "EQ-Credit Corr",
        "Regime Returns", "Default Cycle", "Compare Dates",
        "Corr Heatmap", "Spread Vol", "Fallen Angel",
        "Global Credit", "Corp Leverage", "Seasonality",
        "Traffic Light", "Shock Sim", "Alert BT",
        "PCA", "Regime Fcast", "Composite",
        "X-Asset Mom", "Vol Regime", "Quality Migr",
        "Macro Surprise", "Loan Market", "Regime Age",
        "Deleveraging", "Inflation Regime", "Sector Stress",
        "Put/Call", "Credit Basis", "Drawdown",
        "Signal Move", "Risk Parity", "Tail Dependency",
        "Fed Liquidity", "G4 Divergence", "Port Stress",
        "AT1/CoCo", "Swap Spreads", "XCcy Basis",
        "CRE Stress", "Issuance", "Distressed",
    ])

with tab6:  # Models — 9 sub-tabs
    (_models_sub1, _models_sub2, _models_sub3, _models_sub4,
     _models_sub5, _models_sub6, _models_sub7, _models_sub8,
     _models_sub9) = st.tabs([
        "Sensitivity", "Transitions", "Regimes", "Monte Carlo",
        "Sub-period", "Sizing", "Scenarios", "OOS Splits", "Thresholds",
    ])

with _analytics_sub1:
    st.header("Walk-Forward Validation")

    wf_windows, wf_regimes = load_walk_forward()

    # ── Statistical Confidence Summary ───────────────────────────────────────
    _regime_results = load_regime_transition(df)
    _trans_counts = _regime_results.get("transition_counts") if _regime_results else None
    _audit = load_validation_audit(df, wf_windows, _trans_counts)
    _summary = _audit.get("summary", {})
    _overall = _summary.get("overall_confidence", CONFIDENCE_EXPLORATORY)

    _badge_color = {
        CONFIDENCE_ROBUST:      "green",
        CONFIDENCE_INDICATIVE:  "orange",
        CONFIDENCE_EXPLORATORY: "red",
    }.get(_overall, "gray")

    st.markdown(
        f"**System Confidence:** :{_badge_color}[{CONFIDENCE_SIGILS[_overall]} {_overall}]  "
        f"&nbsp;&nbsp; Robust: **{_summary.get('n_robust', 0)}** · "
        f"Indicative: **{_summary.get('n_indicative', 0)}** · "
        f"Exploratory: **{_summary.get('n_exploratory', 0)}**"
    )

    _warnings = _audit.get("warnings", [])
    if _warnings:
        with st.expander(f"Confidence Warnings ({len(_warnings)})", expanded=(_overall == CONFIDENCE_EXPLORATORY)):
            for w in _warnings:
                st.warning(w)
    else:
        st.success("No confidence warnings. All analytical layers meet robust thresholds.")

    with st.expander("Regime Confidence Detail", expanded=False):
        _regime_stats_audit = _audit.get("regime_stats")
        _trans_audit = _audit.get("transition_matrix")
        _wf_audit = _audit.get("walk_forward", {})
        _corr_audit = _audit.get("correlations")

        # ── Grouped regime stats (5 buckets — more obs per cell) ─────────────
        if "grouped_regime" in df.columns and "sp500_forward_30d_return" in df.columns:
            _grp = (
                df.groupby("grouped_regime")["sp500_forward_30d_return"]
                .agg(n="count", mean_fwd=("mean"), hit_rate=lambda x: (x > 0).mean())
                .sort_values("mean_fwd")
                .rename(columns={"n": "N Obs", "mean_fwd": "Mean Fwd 30d", "hit_rate": "Hit Rate"})
            )
            _grp["Sample Flag"] = _grp["N Obs"].apply(_sample_flag)
            _ORDER = ["Risk-On", "Neutral", "Caution", "Risk-Off"]
            _grp = _grp.reindex([r for r in _ORDER if r in _grp.index])
            st.markdown("**Regime Forward Returns (4 regimes)**")
            st.caption(
                "Sample reliability: Exploratory=n<20 · Indicative=n<50 · Reliable=n≥50"
            )
            st.dataframe(
                _grp.style.format({"Mean Fwd 30d": "{:.2%}", "Hit Rate": "{:.0%}"}),
                use_container_width=True,
            )
        if _regime_stats_audit is not None and not _regime_stats_audit.empty:
            with st.expander("Detailed regime stats"):
                st.caption("Sample reliability: Exploratory = n<20 · Indicative = n<50 · Reliable = n≥50")
                _rs_disp = _regime_stats_audit[["n_obs", "mean_return", "hit_rate", "confidence"]].copy()
                _rs_disp["mean_return"] = _rs_disp["mean_return"].map(
                    lambda x: f"{x:.2%}" if pd.notna(x) else "—"
                )
                _rs_disp["hit_rate"] = _rs_disp["hit_rate"].map(
                    lambda x: f"{x:.0%}" if pd.notna(x) else "—"
                )
                _rs_disp["confidence"] = _rs_disp["confidence"].map(
                    lambda c: f"{CONFIDENCE_SIGILS[c]} {c}"
                )
                _rs_disp["Sample Flag"] = _regime_stats_audit["n_obs"].apply(_sample_flag)
                _rs_disp.columns = ["N Obs", "Mean Return", "Hit Rate", "Confidence", "Sample Flag"]
                st.dataframe(_rs_disp, use_container_width=True)

                if _trans_audit is not None and not _trans_audit.empty:
                    st.markdown("**Transition Confidence by From-Regime**")
                    _td_disp = _trans_audit[["n_outgoing", "confidence"]].copy()
                    _td_disp["confidence"] = _td_disp["confidence"].map(
                        lambda c: f"{CONFIDENCE_SIGILS[c]} {c}"
                    )
                    _td_disp.columns = ["N Outgoing", "Confidence"]
                    st.dataframe(_td_disp, use_container_width=True)

                if _corr_audit is not None and not _corr_audit.empty:
                    st.markdown("**Signal–Return Correlations**")
                    _ca_disp = _corr_audit[["correlation", "n_obs", "informative", "confidence"]].copy()
                    _ca_disp["correlation"] = _ca_disp["correlation"].map(
                        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                    )
                    _ca_disp["informative"] = _ca_disp["informative"].map(
                        lambda x: "Yes" if x else "No"
                    )
                    _ca_disp["confidence"] = _ca_disp["confidence"].map(
                        lambda c: f"{CONFIDENCE_SIGILS[c]} {c}"
                    )
                    _ca_disp.columns = ["Correlation", "N Obs", "Informative", "Confidence"]
                    st.dataframe(_ca_disp, use_container_width=True)

        _wf_conf = _wf_audit.get("confidence", CONFIDENCE_EXPLORATORY)
        st.markdown(
            f"**Walk-Forward:** {CONFIDENCE_SIGILS[_wf_conf]} {_wf_conf} "
            f"({_wf_audit.get('n_windows', 0)} windows)"
        )

    st.divider()

    # ── Regime Persistence ────────────────────────────────────────────────────
    st.subheader("Regime Persistence")
    st.caption(
        "How many days does the model typically stay in each regime before transitioning? "
        "Regimes with very short mean duration (< 3 days) are likely label noise — "
        "treat their forward-return stats with extra caution even when n-obs is adequate."
    )

    import plotly.graph_objects as _pers_go

    _fd_results = _regime_results.get("final_decision", {}) if _regime_results else {}
    _dur = _fd_results.get("durations")
    _fwd = _fd_results.get("forward_returns")

    if _dur is not None and not _dur.empty:
        _dur_sorted = _dur.sort_values("mean_days", ascending=True)
        _pers_colors = [
            "#e74c3c" if v < 3 else "#f39c12" if v < 10 else "#27ae60"
            for v in _dur_sorted["mean_days"]
        ]

        _p_col1, _p_col2 = st.columns([3, 2])

        with _p_col1:
            st.caption("Mean episode duration by regime (trading days) — error bars = ±1 SE")
            _fig_pers = _pers_go.Figure(_pers_go.Bar(
                y=_dur_sorted.index.tolist(),
                x=_dur_sorted["mean_days"].round(1).tolist(),
                orientation="h",
                marker_color=_pers_colors,
                error_x=dict(
                    type="data",
                    array=(_dur_sorted["std_days"] / _dur_sorted["count"].pow(0.5)).round(1).tolist(),
                    visible=True,
                    color="rgba(200,200,200,0.35)",
                ),
                hovertemplate="%{y}<br>Mean: %{x:.1f}d · Episodes: %{customdata}<extra></extra>",
                customdata=_dur_sorted["count"].tolist(),
            ))
            _fig_pers.add_vline(
                x=3, line_color="rgba(231,76,60,0.45)", line_dash="dot", line_width=1.5,
                annotation_text="Noise threshold",
                annotation_font=dict(color="rgba(231,76,60,0.65)", size=9),
                annotation_position="top right",
            )
            _fig_pers.update_layout(
                height=max(240, len(_dur_sorted) * 34 + 60),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                margin=dict(l=8, r=8, t=8, b=8),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Mean Duration (trading days)"),
                yaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                                font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig_pers, use_container_width=True)

        with _p_col2:
            st.caption("Persistence vs. signal reliability")
            _dur_tbl = _dur[["mean_days", "count"]].copy()
            if _fwd is not None and not _fwd.empty:
                _fwd_col = _fwd.get("sp500_forward_30d_return", pd.Series(dtype=float))
                _dur_tbl = _dur_tbl.join(_fwd_col.rename("mean_fwd_30d"), how="left")
            _dur_tbl["reliability"] = _dur_tbl["mean_days"].apply(
                lambda d: "Stable" if d >= 10 else ("Moderate" if d >= 3 else "Noisy")
            )
            _dur_tbl = _dur_tbl.sort_values("mean_days", ascending=False)
            _dur_tbl.index.name = "Regime"
            _fmt = {"mean_days": "{:.1f}", "count": "{:.0f}"}
            if "mean_fwd_30d" in _dur_tbl.columns:
                _fmt["mean_fwd_30d"] = "{:.2%}"
                _dur_tbl.columns = ["Mean Days", "Episodes", "Mean Fwd 30D", "Reliability"]
            else:
                _dur_tbl.columns = ["Mean Days", "Episodes", "Reliability"]
            st.dataframe(_dur_tbl.style.format(_fmt), use_container_width=True)
            st.caption("🟢 Stable ≥10d · 🟡 Moderate 3–10d · 🔴 Noisy <3d")
    else:
        st.info("Run `python app.py` to generate regime transition data.")

    st.divider()

    if wf_windows is None:
        st.warning("No walk-forward results found. Run `python app.py` to generate them.")
    else:
        n_windows = len(wf_windows)
        strategy_beat = (wf_windows["strategy_sharpe"] > wf_windows["sp500_sharpe"]).sum()
        avg_strat_sharpe = wf_windows["strategy_sharpe"].mean()
        avg_sp500_sharpe = wf_windows["sp500_sharpe"].mean()
        avg_strat_return = wf_windows["strategy_total_return"].mean()
        avg_sp500_return = wf_windows["sp500_total_return"].mean()
        avg_strat_dd = wf_windows["strategy_max_drawdown"].mean()
        avg_sp500_dd = wf_windows["sp500_max_drawdown"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Windows", n_windows)
        col2.metric("Windows: Strategy Beat SP500", f"{strategy_beat} / {n_windows}")
        col3.metric("Avg Strategy Sharpe", f"{avg_strat_sharpe:.2f}")
        col4.metric("Avg SP500 Sharpe", f"{avg_sp500_sharpe:.2f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Avg Strategy Return", f"{avg_strat_return:.1%}")
        col6.metric("Avg SP500 Return", f"{avg_sp500_return:.1%}")
        col7.metric("Avg Strategy Max Drawdown", f"{avg_strat_dd:.1%}")
        col8.metric("Avg SP500 Max Drawdown", f"{avg_sp500_dd:.1%}")

        st.subheader("Sharpe Ratio by Window")
        sharpe_chart = wf_windows[["test_start", "strategy_sharpe", "sp500_sharpe"]].copy()
        sharpe_chart["test_start"] = sharpe_chart["test_start"].dt.strftime("%b %Y")
        st.bar_chart(sharpe_chart.set_index("test_start"), color=["#4C9BE8", "#E8834C"])

        st.subheader("Per-Window Performance")
        disp = wf_windows[[
            "window_id", "test_start", "test_end", "dominant_regime",
            "strategy_total_return", "strategy_sharpe", "strategy_max_drawdown",
            "sp500_total_return", "sp500_sharpe", "sp500_max_drawdown",
        ]].copy()
        disp["test_start"] = disp["test_start"].dt.strftime("%Y-%m-%d")
        disp["test_end"] = disp["test_end"].dt.strftime("%Y-%m-%d")
        for c in ["strategy_total_return", "strategy_max_drawdown", "sp500_total_return", "sp500_max_drawdown"]:
            disp[c] = disp[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        for c in ["strategy_sharpe", "sp500_sharpe"]:
            disp[c] = disp[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.subheader("Regime Breakdown")
        rd = wf_regimes[[
            "window_id", "test_start", "regime", "n_obs",
            "avg_forward_30d_return", "hit_rate", "worst_5pct", "avg_equity_weight",
        ]].copy()
        rd["test_start"] = rd["test_start"].dt.strftime("%Y-%m-%d")
        for c in ["avg_forward_30d_return", "worst_5pct"]:
            rd[c] = rd[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        rd["hit_rate"] = rd["hit_rate"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        rd["avg_equity_weight"] = rd["avg_equity_weight"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
        st.dataframe(rd, use_container_width=True, hide_index=True)

    st.subheader("Validation Dataset")

    validation_cols = [
        "date",
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "complacency_score_smooth",
        "mean_reversion_score_smooth",
        "composite_risk_score_smooth",
        "composite_risk_label",
        "final_decision",
        "sp500_forward_30d_return",
        "sp500_forward_60d_return",
        "hy_forward_30d_change",
        "sp500_future_drawdown_30d",
    ]

    existing_cols = [c for c in validation_cols if c in df.columns]
    st.dataframe(df[existing_cols].tail(100), use_container_width=True)

    st.subheader("Correlation Snapshot")

    corr_cols = [
        "macro_risk_score_smooth",
        "credit_market_risk_score_smooth",
        "liquidity_regime_score_smooth",
        "complacency_score_smooth",
        "mean_reversion_score_smooth",
        "composite_risk_score_smooth",
        "sp500_forward_30d_return",
        "sp500_forward_60d_return",
        "hy_forward_30d_change",
        "sp500_future_drawdown_30d",
    ]

    available = [c for c in corr_cols if c in df.columns]
    st.dataframe(df[available].corr(), use_container_width=True)

    # ── Bootstrap Confidence Intervals ───────────────────────────────────────
    with st.expander("Bootstrap Confidence Intervals (95%)", expanded=False):
        st.caption(
            "Percentile bootstrap with 1,000 resamples. "
            "⚠️ = flagged: fewer than 30 obs for regime metrics, "
            "fewer than 5 transitions for probability cells, "
            "fewer than 10 windows for walk-forward metrics."
        )

        boot = load_bootstrap(df, wf_windows)

        def _flag(val):
            return "⚠️" if val else ""

        def _fmt_ci_row(mean, lo, hi, pct=True):
            if pd.isna(lo):
                return f"{mean:.2%}" if pct else f"{mean:.3f}", "—", "—"
            if pct:
                return f"{mean:.2%}", f"{lo:.2%}", f"{hi:.2%}"
            return f"{mean:.3f}", f"{lo:.3f}", f"{hi:.3f}"

        # Walk-forward window CIs
        wm = boot.get("window_metrics", pd.DataFrame())
        if not wm.empty:
            st.markdown("**Walk-Forward Window Metrics**")
            pct_metrics = {"strategy_total_return", "sp500_total_return",
                           "strategy_max_drawdown", "sp500_max_drawdown", "strategy_hit_rate"}
            rows = []
            for metric, row in wm.iterrows():
                pct = metric in pct_metrics
                m, lo, hi = _fmt_ci_row(row["mean"], row.get("ci_lower"), row.get("ci_upper"), pct)
                rows.append({
                    "Metric": metric,
                    "Mean": m,
                    "CI Lower": lo,
                    "CI Upper": hi,
                    "N Windows": int(row["n_windows"]),
                    "": _flag(row["flagged"]),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

        # Regime forward returns CIs
        rr = boot.get("regime_returns", pd.DataFrame())
        if not rr.empty:
            st.markdown("**Mean 30D Forward Return by Regime**")
            rows = []
            for regime, row in rr.iterrows():
                m, lo, hi = _fmt_ci_row(row["mean"], row.get("ci_lower"), row.get("ci_upper"), pct=True)
                rows.append({
                    "Regime": regime, "Mean": m,
                    "CI Lower": lo, "CI Upper": hi,
                    "N Obs": int(row["n_obs"]),
                    "": _flag(row["flagged"]),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Regime"), use_container_width=True)

        # Hit rate CIs
        hr = boot.get("regime_hit_rates", pd.DataFrame())
        if not hr.empty:
            st.markdown("**Hit Rate (% Positive 30D Returns) by Regime**")
            rows = []
            for regime, row in hr.iterrows():
                m, lo, hi = _fmt_ci_row(row["hit_rate"], row.get("ci_lower"), row.get("ci_upper"), pct=True)
                rows.append({
                    "Regime": regime, "Hit Rate": m,
                    "CI Lower": lo, "CI Upper": hi,
                    "N Obs": int(row["n_obs"]),
                    "": _flag(row["flagged"]),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Regime"), use_container_width=True)

        # Transition probability CIs
        tp = boot.get("transition_probs_final_decision", pd.DataFrame())
        if not tp.empty:
            st.markdown("**Transition Probability CIs (Final Decision)**")
            rows = []
            for (frm, to), row in tp.iterrows():
                m, lo, hi = _fmt_ci_row(row["mean_prob"], row.get("ci_lower"), row.get("ci_upper"), pct=True)
                rows.append({
                    "From": frm, "To": to, "Prob": m,
                    "CI Lower": lo, "CI Upper": hi,
                    "N Transitions": int(row["n_obs"]),
                    "": _flag(row["flagged"]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab4:
    from src.backtester import OOS_CUTOFF, build_strategy_backtest, compute_oos_split, compute_benchmark_returns
    import plotly.graph_objects as _bt_go

    st.header("Backtest")

    # ── Interpretation summary ────────────────────────────────────────────────
    st.markdown(
        '<div style="border-left:3px solid #4f8ef7;background:rgba(79,142,247,0.07);'
        'padding:12px 16px;border-radius:0 6px 6px 0;margin-bottom:16px">'
        '<strong style="color:#c8ccd4">What this backtest shows</strong><br>'
        '<span style="color:#9aa0aa;font-size:0.85rem">'
        'The strategy reduces <strong>volatility and max drawdown</strong> vs. buy-and-hold SP500, '
        'but underperforms on <strong>total return</strong> during strong trending markets. '
        'This model is best framed as <strong>defensive / tactical beta management</strong> — '
        'not standalone alpha generation. '
        'The 2020–2026 out-of-sample period was an unusually strong bull run; '
        'a strategy that scales back equity exposure in elevated-risk regimes will naturally lag. '
        'Treat Sharpe and drawdown numbers as the primary signal-quality metrics, not total return.'
        '</span></div>',
        unsafe_allow_html=True,
    )

    # TC slider — defaults to sidebar config value
    _bt_tc = st.slider(
        "Transaction cost (bps)", 0, 50, _cfg_tc_bps, step=5, key="bt_tc_slider",
        help="Applied per day of equity weight change. Adjust to see impact on net return.",
    )

    st.caption(
        f"**In-sample** = signal development period (before {OOS_CUTOFF}).  "
        f"**Out-of-sample** = honest forward test (from {OOS_CUTOFF} onward).  "
        f"Credit spread proxy: Moody's Baa (BAA10Y) — ICE BofA series unavailable pre-2023."
    )

    required_cols = ["strategy_equity_curve", "sp500_equity_curve",
                     "strategy_daily_return", "sp500_daily_return",
                     "strategy_drawdown", "sp500_backtest_drawdown"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.warning(f"Missing backtest columns: {missing_cols}. Run `python app.py` first.")
    else:
        # Re-run backtest live with current config panel settings so sliders
        # are truly interactive (fast — just weight math, no API calls).
        _bt_live = build_strategy_backtest(
            df,
            equity_floor   = _cfg_equity_floor / 100,
            equity_cap     = _cfg_equity_cap / 100,
            target_vol     = _cfg_target_vol / 100,
            ma_window      = _cfg_momentum_lookback,
        )
        _bt_live = compute_benchmark_returns(_bt_live)
        _split = compute_oos_split(_bt_live, cutoff=OOS_CUTOFF, tc_bps=_bt_tc)
        _is  = _split["in_sample"]
        _oos = _split["out_of_sample"]
        _fp  = _split["full_period"]

        def _pct(v, fallback="—"):
            return f"{v:.2%}" if isinstance(v, float) and not pd.isna(v) else fallback

        def _f2(v, fallback="—"):
            return f"{v:.2f}" if isinstance(v, float) and not pd.isna(v) else fallback

        # ════════════════════════════════════════════════════════════════════════
        # 1. SIGNAL QUALITY — does the composite score predict future returns?
        # ════════════════════════════════════════════════════════════════════════
        st.subheader("1. Signal Quality")
        st.caption(
            "Measures whether the composite risk score and regime labels contain "
            "predictive information about future SP500 returns — independent of "
            "any position sizing or allocation decisions."
        )

        _sq_df = df.dropna(subset=["composite_risk_score_smooth", "sp500_forward_30d_return"]).copy()

        # ── Correlation snapshot ─────────────────────────────────────────────
        _corr_score = _sq_df["composite_risk_score_smooth"].corr(_sq_df["sp500_forward_30d_return"])
        _corr_hy    = _sq_df["composite_risk_score_smooth"].corr(_sq_df.get("hy_forward_30d_change", pd.Series(dtype=float))) if "hy_forward_30d_change" in _sq_df.columns else float("nan")
        _hit_all    = (_sq_df["sp500_forward_30d_return"] > 0).mean()

        _sq_hy_df = df.dropna(subset=["composite_risk_score_smooth", "hy_forward_30d_change"]) if "hy_forward_30d_change" in df.columns else pd.DataFrame()
        _corr_hy_disp = (
            f"{float(_sq_hy_df['composite_risk_score_smooth'].corr(_sq_hy_df['hy_forward_30d_change'])):.2f}"
            if len(_sq_hy_df) >= 20 else "—"
        )

        _sq_c1, _sq_c2, _sq_c3, _sq_c4 = st.columns(4)
        _sq_c1.metric(
            "Score↔SP500 30D Corr",
            f"{_corr_score:.2f}",
            help="Negative = high risk score precedes lower equity returns (expected direction).",
        )
        _sq_c2.metric(
            "Score↔HY Spread 30D Corr",
            _corr_hy_disp,
            help="Positive = high risk score precedes HY spread widening (correct credit direction).",
        )
        _sq_c3.metric(
            "Unconditional Hit Rate",
            f"{_hit_all:.1%}",
            help="% of all days where SP500 30-day forward return was positive.",
        )
        _sq_c4.metric(
            "Total Signal Observations",
            f"{len(_sq_df):,}",
            help="Trading days with valid score and forward return data.",
        )

        # ── Grouped regime forward returns bar chart ─────────────────────────
        if "grouped_regime" in _sq_df.columns:
            _grp_order = ["Risk-On", "Neutral", "Caution", "Risk-Off"]
            _grp_stats = (
                _sq_df.groupby("grouped_regime")["sp500_forward_30d_return"]
                .agg(mean="mean", count="count", std="std")
                .reindex([r for r in _grp_order if r in _sq_df["grouped_regime"].unique()])
            )
            _grp_stats["hit_rate"] = (
                _sq_df.groupby("grouped_regime")["sp500_forward_30d_return"]
                .apply(lambda x: (x > 0).mean())
            )
            _grp_stats["se"] = _grp_stats["std"] / _grp_stats["count"].pow(0.5)
            _grp_stats["flag"] = _grp_stats["count"].apply(_sample_flag)

            _sq_col1, _sq_col2 = st.columns(2)

            with _sq_col1:
                st.caption("Mean SP500 30D forward return by grouped regime")
                _bar_colors = [
                    "#27ae60" if v >= 0 else "#e74c3c"
                    for v in _grp_stats["mean"]
                ]
                _fig_sq = _bt_go.Figure(_bt_go.Bar(
                    x=_grp_stats.index.tolist(),
                    y=(_grp_stats["mean"] * 100).round(2).tolist(),
                    marker_color=_bar_colors,
                    error_y=dict(
                        type="data",
                        array=(_grp_stats["se"] * 100).round(2).tolist(),
                        visible=True,
                        color="rgba(200,200,200,0.4)",
                    ),
                    hovertemplate="%{x}<br>Mean: %{y:.2f}%<extra></extra>",
                ))
                _fig_sq.add_hline(
                    y=_hit_all * 100 - _hit_all * 100,  # zero line
                    line_color="rgba(255,255,255,0.15)", line_width=1,
                )
                _fig_sq.update_layout(
                    height=260,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    margin=dict(l=8, r=8, t=8, b=60),
                    xaxis=dict(showgrid=False, color="#6b7280", tickangle=-20),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Mean Fwd Return %"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                                    font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig_sq, use_container_width=True)

            with _sq_col2:
                st.caption("Hit rate (% days with positive 30D return) by grouped regime")
                _fig_hit = _bt_go.Figure(_bt_go.Bar(
                    x=_grp_stats.index.tolist(),
                    y=(_grp_stats["hit_rate"] * 100).round(1).tolist(),
                    marker_color=[
                        "#27ae60" if v >= _hit_all else "#e74c3c"
                        for v in _grp_stats["hit_rate"]
                    ],
                    hovertemplate="%{x}<br>Hit Rate: %{y:.1f}%<extra></extra>",
                ))
                _fig_hit.add_hline(
                    y=_hit_all * 100,
                    line_color="rgba(255,255,255,0.3)", line_width=1.5,
                    line_dash="dash",
                    annotation_text=f"Unconditional {_hit_all:.0%}",
                    annotation_font=dict(color="#9aa0aa", size=10),
                    annotation_position="bottom right",
                )
                _fig_hit.update_layout(
                    height=260,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    margin=dict(l=8, r=8, t=8, b=60),
                    xaxis=dict(showgrid=False, color="#6b7280", tickangle=-20),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Hit Rate %"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                                    font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig_hit, use_container_width=True)

            # Regime stats table
            _grp_display = _grp_stats.copy()
            _grp_display["mean"] = (_grp_display["mean"] * 100).round(2)
            _grp_display["hit_rate"] = (_grp_display["hit_rate"] * 100).round(1)
            _grp_display["std"] = (_grp_display["std"] * 100).round(2)
            _grp_display.columns = ["Mean Fwd Ret %", "Days", "Std %", "Hit Rate %", "Std Err %", "Flag"]
            with st.expander("Regime signal stats table"):
                st.dataframe(_grp_display, use_container_width=True)
                st.caption(
                    "Error bars = ±1 standard error (std / √n). "
                    "Flag = sample reliability: Exploratory (<20) · Indicative (<50) · Reliable (≥50)."
                )

        # ── Regime → HY Spread Forward Change (credit lens) ─────────────────
        if "grouped_regime" in _sq_df.columns and "hy_forward_30d_change" in df.columns:
            _hy_sq_df = df.dropna(subset=["grouped_regime", "hy_forward_30d_change"]).copy()
            if len(_hy_sq_df) >= 20:
                _hy_grp = (
                    _hy_sq_df.groupby("grouped_regime")["hy_forward_30d_change"]
                    .agg(mean="mean", count="count", std="std")
                    .reindex([r for r in _grp_order if r in _hy_sq_df["grouped_regime"].unique()])
                )
                _hy_grp["se"] = _hy_grp["std"] / _hy_grp["count"].pow(0.5)
                _hy_grp["hit_wide"] = (
                    _hy_sq_df.groupby("grouped_regime")["hy_forward_30d_change"]
                    .apply(lambda x: (x > 0).mean())
                )
                _hq1, _hq2 = st.columns(2)
                with _hq1:
                    st.caption("Mean HY spread 30d change by regime (pp) — positive = widening (credit stress confirmed)")
                    _hy_bar_colors = ["#e74c3c" if v >= 0 else "#27ae60" for v in _hy_grp["mean"]]
                    _fig_hy_sq = _bt_go.Figure(_bt_go.Bar(
                        x=_hy_grp.index.tolist(),
                        y=(_hy_grp["mean"] * 100).round(3).tolist(),
                        marker_color=_hy_bar_colors,
                        error_y=dict(type="data", array=(_hy_grp["se"] * 100).round(3).tolist(),
                                     visible=True, color="rgba(200,200,200,0.4)"),
                        hovertemplate="%{x}<br>Mean Δ: %{y:.3f}pp × 100<extra></extra>",
                    ))
                    _fig_hy_sq.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                    _fig_hy_sq.update_layout(
                        height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa"), margin=dict(l=8, r=8, t=8, b=60),
                        xaxis=dict(showgrid=False, color="#6b7280", tickangle=-20),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#6b7280", title="Mean Fwd Spread Change (×100)"),
                        hoverlabel=dict(bgcolor="#1a1f2e", font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig_hy_sq, use_container_width=True)
                with _hq2:
                    _hy_hit_all = float((_hy_sq_df["hy_forward_30d_change"] > 0).mean())
                    st.caption(f"Widening hit rate by regime — dashed line = unconditional {_hy_hit_all:.0%}")
                    _fig_hy_hit = _bt_go.Figure(_bt_go.Bar(
                        x=_hy_grp.index.tolist(),
                        y=(_hy_grp["hit_wide"] * 100).round(1).tolist(),
                        marker_color=["#e74c3c" if v >= _hy_hit_all else "#27ae60" for v in _hy_grp["hit_wide"]],
                        hovertemplate="%{x}<br>Widening: %{y:.1f}%<extra></extra>",
                    ))
                    _fig_hy_hit.add_hline(y=_hy_hit_all * 100, line_color="rgba(255,255,255,0.3)",
                                          line_width=1.5, line_dash="dash")
                    _fig_hy_hit.update_layout(
                        height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa"), margin=dict(l=8, r=8, t=8, b=60),
                        xaxis=dict(showgrid=False, color="#6b7280", tickangle=-20),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#6b7280", title="% Widening"),
                        hoverlabel=dict(bgcolor="#1a1f2e", font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig_hy_hit, use_container_width=True)

        st.divider()

        # ════════════════════════════════════════════════════════════════════════
        # 2. STRATEGY PERFORMANCE — does position sizing monetize the signal?
        # ════════════════════════════════════════════════════════════════════════
        st.subheader("2. Strategy Performance")
        st.caption(
            "Measures whether the 4-method sizing blend (score, regime, vol-target, momentum) "
            "translates signal quality into risk-adjusted returns. "
            "Compare IS vs OOS carefully — OOS is the only honest test."
        )

        # ── IS vs OOS summary table ───────────────────────────────────────────
        # ── Benchmark stats (60/40 and risk parity) ──────────────────────────
        def _bench_stats(col_daily, col_curve, mask):
            sub = _bt_live[mask].copy()
            if sub.empty or col_daily not in sub.columns:
                return {}
            ret_s = sub[col_daily].fillna(0)
            eq    = (1 + ret_s).cumprod()
            ann_ret = (eq.iloc[-1] ** (252 / max(len(eq), 1))) - 1
            ann_vol = ret_s.std() * np.sqrt(252)
            sharpe  = (ret_s.mean() / ret_s.std() * np.sqrt(252)) if ret_s.std() > 0 else float("nan")
            max_dd  = (eq / eq.cummax() - 1).min()
            return {"ret": ann_ret, "sharpe": sharpe, "dd": max_dd, "vol": ann_vol}

        _cutoff_ts2 = pd.Timestamp(OOS_CUTOFF)
        _bt_dates   = pd.to_datetime(_bt_live["date"])
        _is_m  = (_bt_dates < _cutoff_ts2).values
        _oos_m = (_bt_dates >= _cutoff_ts2).values
        _all_m = pd.Series([True] * len(_bt_live)).values

        _bm_6040_is  = _bench_stats("sixty_forty_daily",  "sixty_forty_curve",  _is_m)
        _bm_6040_oos = _bench_stats("sixty_forty_daily",  "sixty_forty_curve",  _oos_m)
        _bm_6040_fp  = _bench_stats("sixty_forty_daily",  "sixty_forty_curve",  _all_m)
        _bm_rp_is    = _bench_stats("risk_parity_daily",  "risk_parity_curve",  _is_m)
        _bm_rp_oos   = _bench_stats("risk_parity_daily",  "risk_parity_curve",  _oos_m)
        _bm_rp_fp    = _bench_stats("risk_parity_daily",  "risk_parity_curve",  _all_m)

        _bt_metrics = {
            "Period":          [f"In-Sample ({_split['is_start']} → {_split['is_end']})",
                                f"Out-of-Sample ({_split['oos_start']} → {_split['oos_end']})",
                                f"Full Period"],
            "Trading Days":    [_split["is_n_days"], _split["oos_n_days"],
                                _split["is_n_days"] + _split["oos_n_days"]],
            "Strategy Return": [_pct(_is.get("strategy_total_return")),
                                _pct(_oos.get("strategy_total_return")),
                                _pct(_fp.get("strategy_total_return"))],
            "SP500 Return":    [_pct(_is.get("sp500_total_return")),
                                _pct(_oos.get("sp500_total_return")),
                                _pct(_fp.get("sp500_total_return"))],
            "60/40 Return":    [_pct(_bm_6040_is.get("ret")),
                                _pct(_bm_6040_oos.get("ret")),
                                _pct(_bm_6040_fp.get("ret"))],
            "Risk Parity Ret": [_pct(_bm_rp_is.get("ret")),
                                _pct(_bm_rp_oos.get("ret")),
                                _pct(_bm_rp_fp.get("ret"))],
            "Strategy Sharpe": [_f2(_is.get("strategy_sharpe")),
                                _f2(_oos.get("strategy_sharpe")),
                                _f2(_fp.get("strategy_sharpe"))],
            "SP500 Sharpe":    [_f2(_is.get("sp500_sharpe")),
                                _f2(_oos.get("sp500_sharpe")),
                                _f2(_fp.get("sp500_sharpe"))],
            "60/40 Sharpe":    [_f2(_bm_6040_is.get("sharpe")),
                                _f2(_bm_6040_oos.get("sharpe")),
                                _f2(_bm_6040_fp.get("sharpe"))],
            "Max Drawdown":    [_pct(_is.get("strategy_max_drawdown")),
                                _pct(_oos.get("strategy_max_drawdown")),
                                _pct(_fp.get("strategy_max_drawdown"))],
            "60/40 Max DD":    [_pct(_bm_6040_is.get("dd")),
                                _pct(_bm_6040_oos.get("dd")),
                                _pct(_bm_6040_fp.get("dd"))],
            "Volatility":      [_pct(_is.get("strategy_volatility")),
                                _pct(_oos.get("strategy_volatility")),
                                _pct(_fp.get("strategy_volatility"))],
            "Hit Rate":        [_pct(_is.get("strategy_hit_rate")),
                                _pct(_oos.get("strategy_hit_rate")),
                                _pct(_fp.get("strategy_hit_rate"))],
        }
        _bt_df = pd.DataFrame(_bt_metrics).set_index("Period")

        def _bt_row_color(row):
            colors = []
            for col in row.index:
                if col in ("Strategy Sharpe", "Strategy Return", "Hit Rate"):
                    try:
                        v = float(str(row[col]).replace("%", ""))
                        if v > 0:
                            colors.append("background-color:rgba(39,174,96,0.15);color:#27ae60")
                        else:
                            colors.append("background-color:rgba(231,76,60,0.15);color:#e74c3c")
                    except Exception:
                        colors.append("")
                else:
                    colors.append("")
            return colors

        st.dataframe(
            _bt_df.style.apply(_bt_row_color, axis=1),
            use_container_width=True,
        )

        st.info(
            "**How to read this:** The out-of-sample period is the only honest test. "
            "The in-sample period is where signal thresholds were developed — "
            "good performance there is expected and not informative.",
            icon="ℹ️",
        )

        # ── Equity curve with IS/OOS shading ─────────────────────────────────
        st.subheader("Equity Curve — Strategy vs SP500")
        _bt_df2 = _bt_live[["date", "strategy_equity_curve", "sp500_equity_curve"]].copy()
        _bt_df2["date"] = pd.to_datetime(_bt_df2["date"])
        _cutoff_ts = pd.Timestamp(OOS_CUTOFF)

        _fig_bt = _bt_go.Figure()

        # IS shading
        _is_dates = _bt_df2[_bt_df2["date"] < _cutoff_ts]["date"]
        if not _is_dates.empty:
            _fig_bt.add_vrect(
                x0=str(_is_dates.iloc[0].date()),
                x1=str(_is_dates.iloc[-1].date()),
                fillcolor="rgba(255,255,255,0.03)",
                line_width=0,
                annotation_text="In-Sample",
                annotation_position="top left",
                annotation_font=dict(color="rgba(150,150,150,0.7)", size=10),
            )

        # OOS shading
        _oos_dates = _bt_df2[_bt_df2["date"] >= _cutoff_ts]["date"]
        if not _oos_dates.empty:
            _fig_bt.add_vrect(
                x0=str(_oos_dates.iloc[0].date()),
                x1=str(_oos_dates.iloc[-1].date()),
                fillcolor="rgba(79,142,247,0.04)",
                line_width=0,
                annotation_text="Out-of-Sample",
                annotation_position="top left",
                annotation_font=dict(color="rgba(79,142,247,0.7)", size=10),
            )

        # Cutoff line
        _fig_bt.add_vline(
            x=OOS_CUTOFF, line_color="rgba(79,142,247,0.4)",
            line_dash="dash", line_width=1.5,
        )

        _fig_bt.add_trace(_bt_go.Scatter(
            x=_bt_df2["date"],
            y=(_bt_df2["strategy_equity_curve"] - 1) * 100,
            name="Strategy", line=dict(color="#4f8ef7", width=2.5),
            fill="tozeroy", fillcolor="rgba(79,142,247,0.07)",
        ))
        _fig_bt.add_trace(_bt_go.Scatter(
            x=_bt_df2["date"],
            y=(_bt_df2["sp500_equity_curve"] - 1) * 100,
            name="SP500", line=dict(color="#6b7280", width=1.8, dash="dot"),
        ))
        if "sixty_forty_curve" in _bt_live.columns:
            _fig_bt.add_trace(_bt_go.Scatter(
                x=_bt_df2["date"],
                y=(_bt_live["sixty_forty_curve"] - 1) * 100,
                name="60/40", line=dict(color="#f39c12", width=1.5, dash="dash"),
            ))
        if "risk_parity_curve" in _bt_live.columns:
            _fig_bt.add_trace(_bt_go.Scatter(
                x=_bt_df2["date"],
                y=(_bt_live["risk_parity_curve"] - 1) * 100,
                name="Risk Parity", line=dict(color="#9b59b6", width=1.5, dash="dashdot"),
            ))
        _fig_bt.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)
        _fig_bt.update_layout(
            height=320,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            margin=dict(l=8, r=8, t=32, b=8),
            xaxis=dict(showgrid=False, color="#6b7280"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                       color="#6b7280", title="Cumulative Return %"),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                            font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(_fig_bt, use_container_width=True)

        # ── Turnover & transaction cost analysis ─────────────────────────────
        st.subheader("Turnover & Transaction Cost Analysis")
        if "strategy_turnover" in _bt_live.columns:
            _to_daily  = _bt_live["strategy_turnover"].mean()
            _to_annual = _to_daily * 252
            _tc_annual = _to_annual * (_bt_tc / 10_000) * 100
            _tc_5yr    = _tc_annual * 5

            _tc1, _tc2, _tc3, _tc4 = st.columns(4)
            _tc1.metric("Avg Daily Turnover",  f"{_to_daily:.3f}",
                        help="Mean absolute daily change in equity weight")
            _tc2.metric("Annual Turnover",     f"{_to_annual:.1%}",
                        help="Daily turnover × 252 trading days")
            _tc3.metric("Annual TC Drag",      f"{_tc_annual:.2f}%",
                        help=f"Annual turnover × {_bt_tc} bps cost")
            _tc4.metric("5-Year TC Drag",      f"{_tc_5yr:.1f}%",
                        help="Cumulative cost over 5 years at this TC assumption")

            # Regime turnover breakdown
            if "final_decision" in _bt_live.columns:
                _regime_to = (
                    _bt_live.groupby("final_decision")["strategy_turnover"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "Avg Daily Turnover", "count": "Days"})
                    .sort_values("Avg Daily Turnover", ascending=False)
                )
                _regime_to["Flag"] = _regime_to["Days"].apply(_sample_flag)
                with st.expander("Turnover by regime"):
                    st.dataframe(
                        _regime_to.style.format({"Avg Daily Turnover": "{:.4f}"}),
                        use_container_width=True,
                    )
                    st.caption(
                        "Flag = sample reliability of each regime. "
                        "Exploratory: n<20 · Indicative: n<50 · Reliable: n≥50"
                    )

        st.subheader("Recent Strategy Weights (4-Method Blend)")
        weight_cols = ["date", "final_decision",
                       "score_weight", "regime_weight", "vol_target_weight",
                       "momentum_weight", "strategy_weight_raw",
                       "strategy_weight", "strategy_turnover"]
        existing_weight_cols = [c for c in weight_cols if c in _bt_live.columns]
        st.dataframe(
            _bt_live[existing_weight_cols].tail(50)
            .style.format({c: "{:.3f}" for c in existing_weight_cols if c not in ("date", "final_decision")}),
            use_container_width=True,
        )
        st.caption(
            "score = piecewise composite score · regime = decision-label map · "
            "vol_target = target_vol/realised_vol · momentum = score×trend_scalar · "
            "raw = mean of all four · strategy = raw clipped to [floor, cap]"
        )

    # ── Drawdown Attribution ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Drawdown Attribution — Which Component Drove Each Drawdown?")
    st.caption(
        "During each strategy drawdown > 5%, shows the counterfactual return "
        "if only one sizing component (score, regime, vol-target, momentum) had been used. "
        "Identifies which component was most/least defensive."
    )
    try:
        _dd_attr = load_drawdown_attribution(df)
        _dd_list = _dd_attr.get("drawdowns", [])
        _dd_drag = _dd_attr.get("avg_component_drag", {})
        if not _dd_list:
            st.success("No drawdowns exceeding 5% found in the backtest history.")
        else:
            # Summary drag metrics
            if _dd_drag:
                _da_cols = st.columns(4)
                for _da_c, (_da_k, _da_v) in zip(_da_cols, _dd_drag.items()):
                    _da_label = {"score":"Score Weight","regime":"Regime Weight",
                                 "vol_target":"Vol-Target","momentum":"Momentum"}.get(_da_k,_da_k)
                    _da_cols[list(_dd_drag.keys()).index(_da_k)].metric(
                        f"Avg {_da_label} Return", f"{_da_v:+.2%}",
                        help=f"Mean return of {_da_label} used in isolation during drawdown periods")

            # Per-drawdown table
            import plotly.graph_objects as _go_da
            _dd_rows = []
            for _dd in _dd_list:
                _dd_rows.append({
                    "Start": str(_dd.get("start",""))[:10],
                    "End":   str(_dd.get("end",""))[:10],
                    "Days":  _dd.get("duration_days","—"),
                    "Strategy": f"{_dd.get('strategy_return',0):+.2%}",
                    "SP500":    f"{_dd.get('sp500_return',0):+.2%}",
                    "Score-only":     f"{_dd.get('component_returns',{}).get('score',0):+.2%}",
                    "Regime-only":    f"{_dd.get('component_returns',{}).get('regime',0):+.2%}",
                    "VolTarget-only": f"{_dd.get('component_returns',{}).get('vol_target',0):+.2%}",
                    "Momentum-only":  f"{_dd.get('component_returns',{}).get('momentum',0):+.2%}",
                })
            st.dataframe(pd.DataFrame(_dd_rows), use_container_width=True)

            # Bar chart of component performance per drawdown
            _dd_names = [f"{r['Start']} → {r['End']}" for r in _dd_rows]
            _da_bar = _go_da.Figure()
            for _da_comp, _da_label, _da_color in [
                ("score","Score","#4f8ef7"), ("regime","Regime","#9b59b6"),
                ("vol_target","Vol-Target","#27ae60"), ("momentum","Momentum","#f39c12"),
            ]:
                _da_bar.add_trace(_go_da.Bar(
                    name=_da_label, x=_dd_names,
                    y=[_dd.get("component_returns",{}).get(_da_comp,0)*100 for _dd in _dd_list],
                    marker_color=_da_color,
                    hovertemplate=f"{_da_label}<br>%{{x}}<br>Return: %{{y:+.2f}}%<extra></extra>",
                ))
            _da_bar.add_trace(_go_da.Scatter(
                name="Strategy", x=_dd_names,
                y=[_dd.get("strategy_return",0)*100 for _dd in _dd_list],
                mode="markers", marker=dict(symbol="diamond", size=10, color="white"),
                hovertemplate="Strategy<br>%{x}<br>%{y:+.2f}%<extra></extra>",
            ))
            _da_bar.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
            _da_bar.update_layout(
                barmode="group", height=320,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"), margin=dict(l=8,r=8,t=8,b=80),
                xaxis=dict(showgrid=False, color="#6b7280", tickangle=-20),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Cumulative Return %"),
                legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
                hoverlabel=dict(bgcolor="#1a1f2e", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_da_bar, use_container_width=True)
            st.caption("Bars = each component used in isolation · diamond = actual blended strategy.")
    except Exception as _da_e:
        st.caption(f"Drawdown attribution unavailable: {_da_e}")

    # ── DV01 / Spread Sensitivity ─────────────────────────────────────────────
    st.subheader("3. DV01 — Interest Rate & Spread Sensitivity")
    st.caption(
        "**DV01** (Dollar Value of 01) measures portfolio P&L impact of a 1 basis-point spread move. "
        "Formula: DV01 = Portfolio Value × Blended Duration × 0.0001. "
        "Duration is the price sensitivity of a bond to yield changes: a 4y HY bond loses ~4% in value "
        "per 100bp rise in yields. This section shows scenario P&L across a range of spread shocks."
    )
    try:
        _dv = load_dv01(df)
        if _dv.get("available"):
            _dvc = _dv["current"]
            _dv1, _dv2, _dv3, _dv4 = st.columns(4)
            _dv1.metric("Blended Duration", f"{_dvc.get('blended_duration', 0):.1f} yrs")
            _dv_pnl = _dvc.get('pnl_scenarios', {})
            _dv2.metric("DV01 (per $1M)", f"${_dvc.get('dv01', 0):,.0f}",
                        help="P&L change per 1bp spread move on a $1M portfolio")
            _dv3.metric("P&L at +100bps", f"${_dv_pnl.get('+100bps', 0):,.0f}",
                        delta_color="inverse",
                        delta=f"${_dv_pnl.get('+100bps', 0):,.0f}")
            _dv4.metric("P&L at −50bps",  f"${_dv_pnl.get('-50bps', 0):,.0f}",
                        delta=f"${_dv_pnl.get('-50bps', 0):,.0f}")

            _scen_tbl = _dv.get("scenario_table", pd.DataFrame())
            if not _scen_tbl.empty:
                with st.expander("Full scenario table (−150 to +300 bps)"):
                    _st_disp = _scen_tbl.copy()
                    if "pnl_pct" in _st_disp.columns:
                        _st_disp["pnl_pct"] = _st_disp["pnl_pct"].map("{:.2%}".format)
                    if "pnl" in _st_disp.columns:
                        _st_disp["pnl"] = _st_disp["pnl"].map("${:,.0f}".format)
                    st.dataframe(_st_disp, use_container_width=True, hide_index=True)
        else:
            st.info("DV01 unavailable — requires HY spread data.")
    except Exception as _dv_e:
        st.caption(f"DV01 unavailable: {_dv_e}")

    # ── Carry Breakeven ───────────────────────────────────────────────────────
    st.subheader("4. Carry Breakeven — How Much Spread Widening Can Carry Absorb?")
    st.caption(
        "**Carry Breakeven** answers: *how far do spreads have to widen before my carry income is wiped out?* "
        "Formula: Breakeven = All-In Yield (%) / Duration × 100 bps/yr. "
        "Example: HY yield 8%, duration 4yr → breakeven = 200bps/yr. "
        "If spreads widen more than that over the next year, total return turns negative."
    )
    try:
        _cb = load_carry_breakeven(df)
        if _cb.get("available"):
            _cbc = _cb.get("current", {})
            _cb1, _cb2, _cb3, _cb4 = st.columns(4)
            _cb1.metric("HY Yield", f"{(_cbc.get('hy_yield_bps') or 0)/100:.1f}%",
                        help="All-in yield used for carry calculation")
            _cb2.metric("HY Breakeven", f"{_cbc.get('breakeven_hy_bps', 0):.0f} bps/yr",
                        help="Max spread widening before 1yr total return = 0")
            _cb3.metric("IG Breakeven", f"{_cbc.get('breakeven_ig_bps', 0):.0f} bps/yr")
            _cb4.metric("HY Carry Regime", _cbc.get("carry_hy_regime", "—"))

            # Scenario table
            _cb_st = _cb.get("scenario_table")
            if _cb_st is not None and hasattr(_cb_st, "shape") and not _cb_st.empty:
                with st.expander("Spread shock scenario analysis"):
                    _cb_disp = _cb_st.copy()
                    _cb_disp.columns = ["Shock (bps)", "HY Months", "HY P&L (bps)", "HY Action",
                                        "IG Months", "IG P&L (bps)", "IG Action"]
                    st.dataframe(_cb_disp, use_container_width=True, hide_index=True)

            if _cbc.get("interpretation"):
                st.caption(_cbc["interpretation"])

            # Historical breakeven chart
            _cb_hist = _cb.get("historical_breakeven")
            if _cb_hist is not None and hasattr(_cb_hist, "shape") and not _cb_hist.empty:
                import plotly.graph_objects as _cbgo
                _cb_fig = _cbgo.Figure()
                if "breakeven_hy_bps" in _cb_hist.columns:
                    _cb_fig.add_trace(_cbgo.Scatter(
                        x=_cb_hist.index, y=_cb_hist["breakeven_hy_bps"],
                        name="HY Breakeven", line=dict(color="#e67e22", width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>HY Breakeven: %{y:.0f} bps<extra></extra>",
                    ))
                if "breakeven_ig_bps" in _cb_hist.columns:
                    _cb_fig.add_trace(_cbgo.Scatter(
                        x=_cb_hist.index, y=_cb_hist["breakeven_ig_bps"],
                        name="IG Breakeven", line=dict(color="#3498db", width=1.5, dash="dash"),
                        hovertemplate="%{x|%Y-%m-%d}<br>IG Breakeven: %{y:.0f} bps<extra></extra>",
                    ))
                _cb_fig.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="bps/yr"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_cb_fig, use_container_width=True)
        else:
            st.info("Carry breakeven unavailable — requires HY/IG yield or spread data.")
    except Exception as _cb_e:
        st.caption(f"Carry breakeven unavailable: {_cb_e}")

    # ── CVaR by Regime ────────────────────────────────────────────────────────
    st.subheader("5. Conditional Value-at-Risk (CVaR) by Regime")
    st.caption(
        "**CVaR** (Expected Shortfall) answers: *given that tomorrow is a bad day, how bad?* "
        "It is the expected loss beyond the VaR threshold and is the preferred tail risk measure under Basel III. "
        "CVaR_95 = average of the worst 5% of daily return observations. "
        "Regimes have dramatically different tail profiles — Risk-Off CVaR is typically 3–5× worse than Risk-On."
    )
    try:
        _cv = load_cvar(df)
        if _cv.get("available"):
            _cv_regime = _cv.get("regime_stats", pd.DataFrame())
            if not _cv_regime.empty:
                import plotly.graph_objects as _cvgo
                _cv_cols = st.columns([2, 1])
                with _cv_cols[0]:
                    _cvar_fig = _cvgo.Figure()
                    _regime_order = ["Risk-On", "Neutral", "Caution", "Risk-Off"]
                    _cv_plot = _cv_regime[_cv_regime.index.isin(_regime_order)].reindex(
                        [r for r in _regime_order if r in _cv_regime.index]
                    )
                    _cv_colors = {"Risk-On": "#27ae60", "Neutral": "#f39c12",
                                  "Caution": "#e67e22", "Risk-Off": "#e74c3c"}
                    for _rg in _cv_plot.index:
                        _bar_color = _cv_colors.get(_rg, "#6b7280")
                        _cvar_fig.add_trace(_cvgo.Bar(
                            x=[_rg],
                            y=[_cv_plot.loc[_rg, "cvar_95"] * 100] if "cvar_95" in _cv_plot.columns else [0],
                            name=_rg, marker_color=_bar_color,
                            hovertemplate=f"{_rg}<br>CVaR 95: %{{y:.2f}}%<extra></extra>",
                        ))
                    _cvar_fig.update_layout(
                        height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11),
                        margin=dict(l=8, r=8, t=30, b=8),
                        title=dict(text="Daily CVaR 95% by Regime (annualised)", font=dict(size=12)),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#6b7280", title="CVaR 95% (daily %)"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        showlegend=False,
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                                        font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_cvar_fig, use_container_width=True)
                with _cv_cols[1]:
                    _cv_disp = _cv_regime.copy()
                    _cv_fmt = {}
                    for _cc in ["mean_return", "volatility", "var_95", "cvar_95", "var_99", "cvar_99", "max_loss"]:
                        if _cc in _cv_disp.columns:
                            _cv_fmt[_cc] = "{:.2%}"
                    for _cc in ["skewness", "kurtosis"]:
                        if _cc in _cv_disp.columns:
                            _cv_fmt[_cc] = "{:.2f}"
                    if "n_obs" in _cv_disp.columns:
                        _cv_fmt["n_obs"] = "{:.0f}"
                    st.dataframe(
                        _cv_disp.style.format(_cv_fmt, na_rep="—"),
                        use_container_width=True,
                    )
                    st.caption("Regime-level tail risk statistics. CVaR = Expected Shortfall.")
        else:
            st.info("CVaR unavailable — requires strategy_daily_return or sp500_daily_return.")
    except Exception as _cv_e:
        st.caption(f"CVaR unavailable: {_cv_e}")

    # ── Credit Portfolio Backtest ─────────────────────────────────────────────
    st.subheader("3. Credit Portfolio Backtest")
    st.caption(
        "Performance of the regime-based HY/IG credit allocation. "
        "Total return = carry (all-in yield / 252) + duration-adjusted price return. "
        f"HY duration ≈ 4y · IG duration ≈ 7y · 10bps one-way transaction cost. "
        f"OOS = from {OOS_CUTOFF} onward."
    )

    try:
        from src.credit_backtest import run_credit_backtest as _run_cb
        _cb_results = _run_cb(_bt_live)
        _cb_returns  = _cb_results.get("returns_df", pd.DataFrame())
        _cb_regimes  = _cb_results.get("regime_stats", pd.DataFrame())
        _cb_buckets  = _cb_results.get("spread_buckets", pd.DataFrame())

        if not _cb_returns.empty and _cb_results.get("has_hy_data"):
            _cr1, _cr2 = st.columns(2)

            with _cr1:
                import plotly.graph_objects as _cgo
                _cb_returns["date"] = pd.to_datetime(_cb_returns["date"])
                _fig_cb = _cgo.Figure()
                _fig_cb.add_trace(_cgo.Scatter(
                    x=_cb_returns["date"],
                    y=(_cb_returns["cum_hy"] - 1) * 100,
                    name="HY", line=dict(color="#e74c3c", width=1.8),
                ))
                if _cb_results.get("has_ig_data"):
                    _fig_cb.add_trace(_cgo.Scatter(
                        x=_cb_returns["date"],
                        y=(_cb_returns["cum_ig"] - 1) * 100,
                        name="IG", line=dict(color="#27ae60", width=1.8),
                    ))
                _fig_cb.add_trace(_cgo.Scatter(
                    x=_cb_returns["date"],
                    y=(_cb_returns["cum_credit"] - 1) * 100,
                    name="Credit Portfolio", line=dict(color="#4f8ef7", width=2.5),
                    fill="tozeroy", fillcolor="rgba(79,142,247,0.07)",
                ))
                _fig_cb.add_trace(_cgo.Scatter(
                    x=_cb_returns["date"],
                    y=(_cb_returns["cum_net_credit"] - 1) * 100,
                    name="Net (after costs)", line=dict(color="#9b59b6", width=1.5, dash="dash"),
                ))
                _fig_cb.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig_cb.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=40, b=8),
                    height=300,
                    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Cumulative Return %"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_fig_cb, use_container_width=True)

            with _cr2:
                if _cb_results.get("ann_turnover") is not None:
                    st.metric("Annualised Credit Turnover",
                              f"{_cb_results['ann_turnover']:.1f}x/yr")
                if not _cb_buckets.empty:
                    st.markdown("**HY Spread Level → 30d Forward Change**")
                    st.caption(
                        "At what spread level does HY become attractive? "
                        "Hit Rate = % of time spreads tightened over next 30 days."
                    )
                    _bkt_disp = _cb_buckets.copy()
                    _bkt_disp.columns = [c.replace("_", " ").title() for c in _bkt_disp.columns]
                    st.dataframe(
                        _bkt_disp.style.format({
                            "Avg Fwd 30D": "{:+.3f}",
                            "Hit Rate Tightening": "{:.0%}",
                            "Pct Risk Off": "{:.0f}%",
                        }, na_rep="—"),
                        use_container_width=True,
                        hide_index=True,
                    )

        if not _cb_regimes.empty:
            st.markdown("**Regime-Conditional Credit Performance**")
            st.caption(
                "Per-regime: avg spread levels, 30d forward HY OAS change, "
                "tightening hit rate, and annualised credit portfolio Sharpe."
            )
            _reg_disp = _cb_regimes.copy()
            _fmt_map  = {}
            for _c in ["avg_hy_spread", "avg_ig_spread", "avg_hy_fwd_30d"]:
                if _c in _reg_disp.columns:
                    _fmt_map[_c] = "{:.2f}"
            for _c in ["hit_rate_tightening", "avg_hy_daily_return", "avg_ig_daily_return"]:
                if _c in _reg_disp.columns:
                    _fmt_map[_c] = "{:.1%}"
            if "credit_sharpe" in _reg_disp.columns:
                _fmt_map["credit_sharpe"] = "{:.2f}"
            _reg_disp.columns = [c.replace("_", " ").title() for c in _reg_disp.columns]
            st.dataframe(
                _reg_disp.style.format({
                    k.replace("_", " ").title(): v for k, v in _fmt_map.items()
                }, na_rep="—"),
                use_container_width=True,
                hide_index=True,
            )
        elif not _cb_results.get("has_hy_data"):
            st.info(
                "HY total return series not available. "
                "Run the data pipeline to fetch HY effective yield (BAMLHYH0A0HYM2EY)."
            )

        # ── Excess Return Decomposition ────────────────────────────────────
        if _cb_results.get("has_hy_data"):
            try:
                import plotly.graph_objects as _cgo
                from src.credit_backtest import decompose_excess_returns as _decomp_fn
                _cb_decomp = _decomp_fn(df)
                if not _cb_decomp.empty and "cum_carry" in _cb_decomp.columns:
                    st.markdown("**Return Attribution: Carry vs. Spread P&L**")
                    st.caption(
                        "Carry = all-in yield / 252 × weight. "
                        "Spread P&L = −duration × Δspread / 100 × weight. "
                        "Together they sum to the total credit portfolio return."
                    )
                    _cb_decomp["date"] = pd.to_datetime(_cb_decomp["date"])
                    _fig_decomp = _cgo.Figure()
                    _fig_decomp.add_trace(_cgo.Scatter(
                        x=_cb_decomp["date"],
                        y=(_cb_decomp["cum_carry"] - 1) * 100,
                        name="Carry (cumulative)", line=dict(color="#27ae60", width=2),
                        fill="tozeroy", fillcolor="rgba(39,174,96,0.07)",
                    ))
                    _fig_decomp.add_trace(_cgo.Scatter(
                        x=_cb_decomp["date"],
                        y=(_cb_decomp["cum_spread_pnl"] - 1) * 100,
                        name="Spread P&L (cumulative)", line=dict(color="#e74c3c", width=2),
                        fill="tozeroy", fillcolor="rgba(231,76,60,0.07)",
                    ))
                    _fig_decomp.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)
                    _fig_decomp.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11),
                        margin=dict(l=8, r=8, t=8, b=8),
                        height=260,
                        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#6b7280", title="Cumulative Return %"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                                        font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig_decomp, use_container_width=True)
            except Exception:
                pass

    except Exception as _cb_err:
        st.warning(f"Credit backtest unavailable: {_cb_err}")

with tab7:
    st.header("Model Run History")

    if not history.empty:
        st.dataframe(history.tail(50), use_container_width=True)

        chart_cols = [
            "macro_risk",
            "credit_risk",
            "complacency",
            "treasury",
        ]

        available_history_cols = [c for c in chart_cols if c in history.columns]

        if "timestamp" in history.columns and available_history_cols:
            history["timestamp"] = pd.to_datetime(history["timestamp"])
            st.line_chart(history.set_index("timestamp")[available_history_cols])
    else:
        st.warning("No run history found yet.")

with _models_sub1:
    st.header("Parameter Sensitivity")

    # ── Live weight optimisation ──────────────────────────────────────────────
    st.subheader("Composite Weight Optimisation")
    st.caption(
        "2,000 weight vectors sampled from the 7-score simplex (Dirichlet uniform). "
        "Strategy return = equity_weight × SP500 daily return, "
        "where equity_weight = 1 − composite/100. "
        "In-sample Sharpe only — treat as directional, not predictive."
    )

    _wopt = load_weight_optimization(df)
    _grid = _wopt["grid_results"]
    _tornado = _wopt["tornado"]
    _curr_s = _wopt["current_sharpe"]
    _opt_w = _wopt["optimal_weights"]
    _opt_s = _wopt["optimal_sharpe"]

    if not _grid.empty:
        # Headline metrics
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Current Weights Sharpe", f"{_curr_s:.2f}" if pd.notna(_curr_s) else "—")
        wc2.metric(
            "Best Found Sharpe",
            f"{_opt_s:.2f}" if pd.notna(_opt_s) else "—",
            delta=f"{_opt_s - _curr_s:+.2f}" if pd.notna(_opt_s) and pd.notna(_curr_s) else None,
        )
        pct_better = (_grid["sharpe"] > _curr_s).mean() if pd.notna(_curr_s) else np.nan
        wc3.metric(
            "Samples Beating Current Weights",
            f"{pct_better:.0%}" if pd.notna(pct_better) else "—",
        )

        # Current vs optimal weights bar chart
        st.markdown("**Current vs Best-Found Weights**")
        _score_display = {
            "treasury":         "Treasury",
            "complacency":      "Complacency",
            "credit_risk":      "Credit Risk",
            "macro_risk":       "Macro Risk",
            "liquidity":        "Liquidity",
            "enhanced_funding": "Funding Stress",
            "fx_commodity":     "FX / Commodity",
        }
        _w_compare = pd.DataFrame({
            "Current": {_score_display.get(k, k): v for k, v in CURRENT_WEIGHTS.items()},
            "Optimal": {_score_display.get(k, k): v for k, v in _opt_w.items()},
        })
        st.bar_chart(_w_compare, use_container_width=True)

        # Sharpe distribution with current marked
        st.markdown("**Sharpe Distribution Across 2,000 Weight Combinations**")
        import plotly.graph_objects as _wgo
        _sharpe_vals = _grid["sharpe"].dropna()
        fig_dist = _wgo.Figure()
        fig_dist.add_trace(_wgo.Histogram(
            x=_sharpe_vals, nbinsx=60,
            marker_color="#3498db", opacity=0.75, name="Sampled Sharpes",
        ))
        if pd.notna(_curr_s):
            fig_dist.add_vline(
                x=_curr_s, line=dict(color="#e74c3c", width=2, dash="dash"),
                annotation_text=f"Current ({_curr_s:.2f})",
                annotation_position="top right",
            )
        fig_dist.update_layout(
            xaxis_title="Annualised Sharpe", yaxis_title="Count",
            height=320, showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        fig_dist.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig_dist.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Tornado chart
        if not _tornado.empty:
            st.markdown(f"**Tornado: Sharpe Sensitivity to ±5% Weight Shift (current weights)**")
            st.caption("Positive delta = increasing that weight improves Sharpe.")
            fig_torn = _wgo.Figure()
            _torn_display = _tornado.copy()
            _torn_display.index = [_score_display.get(i, i) for i in _torn_display.index]
            fig_torn.add_trace(_wgo.Bar(
                y=_torn_display.index.tolist(),
                x=_torn_display["delta_up"],
                name="+5%", orientation="h",
                marker_color="#27ae60", opacity=0.8,
            ))
            fig_torn.add_trace(_wgo.Bar(
                y=_torn_display.index.tolist(),
                x=_torn_display["delta_down"],
                name="−5%", orientation="h",
                marker_color="#e74c3c", opacity=0.8,
            ))
            fig_torn.add_vline(x=0, line=dict(color="#888888", width=1))
            fig_torn.update_layout(
                barmode="overlay",
                xaxis_title="Δ Sharpe vs Current",
                yaxis_title="",
                height=max(280, len(_tornado) * 42 + 80),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=45, b=20),
            )
            fig_torn.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig_torn, use_container_width=True)

        # Top-10 weight combinations
        with st.expander("Top 10 Weight Combinations", expanded=False):
            _top10 = _grid.head(10).copy()
            _top10.index = range(1, 11)
            for k, label in _score_display.items():
                if k in _top10.columns:
                    _top10[label] = _top10[k].map(lambda v: f"{v:.3f}")
                    _top10 = _top10.drop(columns=[k])
            if "sharpe" in _top10.columns:
                _top10["sharpe"] = _top10["sharpe"].map(lambda v: f"{v:.2f}")
            if "corr_30d" in _top10.columns:
                _top10["corr_30d"] = _top10["corr_30d"].map(
                    lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                )
            st.dataframe(_top10, use_container_width=True)
    else:
        st.warning("Insufficient data for weight optimisation (need ≥ 3 score columns and sp500_daily_return).")

    st.divider()

    # ── Legacy CSV-based sensitivity (if outputs exist) ───────────────────────
    sens_df = load_sensitivity()

    if sens_df is None:
        st.warning("No sensitivity results found. Run `python app.py` to generate them.")
    else:
        # ── Plain-text report ────────────────────────────────────────────────
        if SENS_REPORT_PATH.exists():
            with st.expander("Sensitivity Report", expanded=True):
                st.text(SENS_REPORT_PATH.read_text())

        groups = {
            "score_scale":        "Score Scale (±30% per sub-score)",
            "composite_weights":  "Composite Blend Weights (±30%, renormalized)",
            "backtester_weights": "Backtester Decision Weights (±30%)",
        }

        for group_key, group_label in groups.items():
            sub = sens_df[sens_df["group"] == group_key]
            if sub.empty:
                continue

            st.subheader(group_label)

            # ── Summary table: metric range per parameter ────────────────────
            metric_cols = [c for c in ["sharpe", "total_return", "max_drawdown",
                                       "hit_rate", "fwd_mean_return", "fwd_hit_rate"]
                           if c in sub.columns]

            rows = []
            for param, pdata in sub.groupby("param"):
                row = {"param": param}
                for m in metric_cols:
                    vals = pdata[m].dropna()
                    if len(vals):
                        row[f"{m}_min"]   = vals.min()
                        row[f"{m}_max"]   = vals.max()
                        row[f"{m}_range"] = vals.max() - vals.min()
                        mean = vals.mean()
                        row[f"{m}_cv"] = (
                            round(float(vals.std() / abs(mean)), 4)
                            if abs(mean) > 1e-9 else 0.0
                        )
                rows.append(row)

            summary = pd.DataFrame(rows).set_index("param")

            # highlight CV columns — high CV = unstable
            cv_cols = [c for c in summary.columns if c.endswith("_cv")]
            range_cols = [c for c in summary.columns if c.endswith("_range")]

            if cv_cols:
                fmt = {c: "{:.3f}" for c in summary.select_dtypes("number").columns}
                st.dataframe(
                    summary[cv_cols + range_cols].style
                        .format("{:.4f}")
                        .background_gradient(subset=cv_cols, cmap="YlOrRd", axis=None),
                    use_container_width=True,
                )

            # ── Sharpe line chart across perturbation values ─────────────────
            primary_metric = "fwd_mean_return" if group_key == "composite_weights" else "sharpe"
            if primary_metric in sub.columns:
                chart_data = sub.pivot_table(
                    index="param_value", columns="param",
                    values=primary_metric, aggfunc="mean",
                )
                st.caption(f"{primary_metric} vs perturbation value")
                st.line_chart(chart_data, use_container_width=True)


with _models_sub2:
    st.header("Regime Transition Analysis")

    regime_results = load_regime_transition(df)

    VIEW_OPTIONS = {
        "Final Decision": "final_decision",
        "Transition Regime": "transition_regime",
    }
    view_label = st.radio("View regime", list(VIEW_OPTIONS.keys()), horizontal=True)
    view_col = VIEW_OPTIONS[view_label]

    if view_col not in regime_results:
        st.warning(f"No results found for '{view_col}'.")
    else:
        res = regime_results[view_col]

        # ── Transition probability heatmap (live Plotly — dark-theme safe) ──
        st.subheader("Transition Probability Heatmap")
        import plotly.graph_objects as _trans_go
        probs = res["transition_probs"]
        _probs_pct = probs * 100
        _fig_trans = _trans_go.Figure(_trans_go.Heatmap(
            z=_probs_pct.values.tolist(),
            x=_probs_pct.columns.tolist(),
            y=_probs_pct.index.tolist(),
            colorscale="YlOrRd",
            text=[[f"{v:.1f}%" for v in row] for row in _probs_pct.values],
            texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
            hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.1f}%<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="%",
                tickfont=dict(color="#9aa0aa"),
                titlefont=dict(color="#9aa0aa"),
            ),
        ))
        _fig_trans.update_layout(
            height=max(320, len(probs) * 36 + 80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            margin=dict(l=8, r=8, t=16, b=80),
            xaxis=dict(
                title="To →", color="#6b7280", showgrid=False, tickangle=-30,
            ),
            yaxis=dict(
                title="From →", color="#6b7280", showgrid=False, autorange="reversed",
            ),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                            font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(_fig_trans, use_container_width=True)

        # ── Transition counts ─────────────────────────────────────────────────
        with st.expander("Raw transition counts"):
            st.dataframe(res["transition_counts"])

        # ── Regime persistence ────────────────────────────────────────────────
        st.subheader("Regime Persistence (trading days)")
        durations = res["durations"].copy()
        st.dataframe(
            durations.style
            .background_gradient(cmap="Blues", subset=["mean_days", "max_days"])
            .format("{:.1f}")
        )

        # ── Forward returns by regime ─────────────────────────────────────────
        st.subheader("Mean Forward Returns by Regime")
        fwd = res["forward_returns"].copy()
        rename_map = {
            "sp500_forward_30d_return": "SP500 30D Fwd",
            "sp500_forward_60d_return": "SP500 60D Fwd",
            "strategy_forward_30d_return": "Strategy 30D Fwd",
            "sp500_future_drawdown_30d": "Max Drawdown 30D",
        }
        fwd = fwd.rename(columns={k: v for k, v in rename_map.items() if k in fwd.columns})

        pct_cols = [c for c in fwd.columns if c != "Max Drawdown 30D"]
        dd_cols = ["Max Drawdown 30D"] if "Max Drawdown 30D" in fwd.columns else []

        styled = fwd.style
        if pct_cols:
            styled = styled.background_gradient(cmap="RdYlGn", subset=pct_cols)
        if dd_cols:
            styled = styled.background_gradient(cmap="RdYlGn_r", subset=dd_cols)
        styled = styled.format("{:.2%}")
        st.dataframe(styled)

        # ── Transition entry returns ──────────────────────────────────────────
        st.subheader("Entry-Point Returns on Regime Change")
        st.caption("Average forward returns measured on the first day of each new regime.")
        trans = res["transition_returns"].copy()
        trans = trans.rename(columns={k: v for k, v in rename_map.items() if k in trans.columns})
        st.dataframe(
            trans.style
            .background_gradient(cmap="RdYlGn", subset=[c for c in trans.columns if "Drawdown" not in c])
            .format("{:.2%}")
        )

        # ── Hazard Rate / Regime Aging ────────────────────────────────────────
        st.subheader("Regime Transition Forecast")
        st.caption(
            "Hazard rate h(d) = probability of leaving the regime at day d given it has persisted to day d. "
            "Used to estimate P(exit within 20 trading days) from the current regime age."
        )
        import plotly.graph_objects as _hz_go

        _hz_data = res.get("hazard_by_regime", {})
        _cur_fc   = res.get("current_forecast", {})

        if _cur_fc:
            _fc_c1, _fc_c2, _fc_c3 = st.columns(3)
            _fc_c1.metric("Current Regime",        str(_cur_fc.get("current_regime", "—")))
            _fc_c2.metric("Current Streak (days)",  str(_cur_fc.get("current_age_days", "—")))
            _p_exit = _cur_fc.get("prob_exit_20d", float("nan"))
            _fc_c3.metric(
                "P(Exit within 20d)",
                f"{_p_exit:.1%}" if isinstance(_p_exit, float) and not pd.isna(_p_exit) else "—",
            )
            _next_p = _cur_fc.get("transition_probs", {})
            if _next_p:
                st.caption("Most likely next regimes:")
                _np_cols = st.columns(min(len(_next_p), 4))
                for _ci, (_r_name, _r_prob) in enumerate(list(_next_p.items())[:4]):
                    _np_cols[_ci].metric(_r_name[:25], f"{_r_prob:.0%}")

        if _hz_data:
            _hz_regime_sel = st.selectbox(
                "Show hazard rate for regime",
                list(_hz_data.keys()),
                key="hz_regime_sel",
            )
            _hz_df = _hz_data[_hz_regime_sel]["hazard_df"]
            _hz_fig = _hz_go.Figure()
            _hz_fig.add_trace(_hz_go.Bar(
                x=_hz_df["age_days"],
                y=_hz_df["hazard_rate"],
                name="Daily Hazard Rate",
                marker_color="#e67e22",
                opacity=0.7,
            ))
            _hz_fig.add_trace(_hz_go.Scatter(
                x=_hz_df["age_days"],
                y=_hz_df["hazard_rate"].rolling(5, center=True, min_periods=1).mean(),
                mode="lines", name="5-day smoothed",
                line=dict(color="#f1c40f", width=2),
            ))
            _hz_fig.update_layout(
                xaxis_title="Days in Regime",
                yaxis_title="Daily Exit Probability",
                height=300, template="plotly_dark",
                margin=dict(l=50, r=20, t=30, b=40),
            )
            st.plotly_chart(_hz_fig, use_container_width=True)
            st.caption(
                f"Median run length for '{_hz_regime_sel}': "
                f"{_hz_data[_hz_regime_sel]['median_age']:.0f} days. "
                "Increasing hazard = regime becomes less stable over time. "
                "Flat/declining hazard = regime is persistent (low aging effect)."
            )

        st.divider()
        st.subheader("Validation Caveats & Interpretation")
        _n_wf = len(wf_windows) if wf_windows is not None else 0
        _is_days = int((pd.to_datetime(df["date"]) < pd.Timestamp("2020-01-01")).sum())
        _oos_days_val = int((pd.to_datetime(df["date"]) >= pd.Timestamp("2020-01-01")).sum())
        _thin_regimes = _regime_counts_ov[_regime_counts_ov < 30].index.tolist()

        st.warning(
            f"**Research confidence: {_conf_level}**\n\n"
            f"- In-sample period: **{_is_days} trading days** (~{_is_days//252:.1f} years). "
            f"Signal thresholds were calibrated on this window — IS performance is expected to look good.\n"
            f"- Out-of-sample period: **{_oos_days_val} trading days**. This is the only honest test.\n"
            f"- Walk-forward windows available: **{_n_wf}**. "
            f"Few windows = high Sharpe variability = low statistical confidence.\n"
            + (f"- Thinly observed regimes (n<30): **{', '.join(_thin_regimes)}**. "
               f"Stats for these regimes are exploratory only.\n"
               if _thin_regimes else "")
            + "\n**This is a risk management system, not a return-timing model.**\n\n"
            "Signal roles differ by design: treasury stress and complacency are leading "
            "indicators (negative r at 21-63d horizons). Credit and macro are concurrent "
            "stress indicators — they do not predict returns forward, but they confirm "
            "active crises and prevent full re-investment during drawdowns. "
            "The composite therefore has near-zero forward-return correlation, which is "
            "expected and acceptable. Its value is drawdown reduction, not alpha generation."
        )

    # ── Signal Horizon Grid ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Signal Predictive Horizon Grid")
    st.caption(
        "Spearman correlation of each signal vs. forward SP500 return at 6 horizons "
        "(IS = pre-2022, OOS = 2022+). Negative = high risk score preceded lower returns "
        "(correct direction). Green cells = predictive; red = wrong direction or noise."
    )

    try:
        _horizon_grid = load_signal_horizon_grid(df)
        if not _horizon_grid.empty:
            import plotly.graph_objects as _hg_go

            _hz_labels = [f"{h}d" for h in [21, 42, 63, 126, 189, 252]]
            _is_cols  = [(h, "IS")  for h in [21, 42, 63, 126, 189, 252]]
            _oos_cols = [(h, "OOS") for h in [21, 42, 63, 126, 189, 252]]

            # Build display: IS columns only for the heatmap; show OOS in table below
            _grid_is = _horizon_grid[_is_cols].copy()
            _grid_is.columns = _hz_labels

            # Add role column
            _grid_is.insert(0, "Role", [
                SIGNAL_ROLES.get(idx, "—") for idx in _grid_is.index
            ])

            # Color heatmap: green for negative r, red for positive r
            _z = _grid_is.drop(columns=["Role"]).values.tolist()
            _y = _grid_is.index.tolist()
            _fig_hg = _hg_go.Figure(_hg_go.Heatmap(
                z=_z, x=_hz_labels, y=_y,
                colorscale=[
                    [0.0, "#27ae60"], [0.5, "rgba(40,44,60,0.5)"], [1.0, "#e74c3c"]
                ],
                zmid=0, zmin=-0.15, zmax=0.15,
                texttemplate="%{text}",
                text=[[f"{v:+.3f}" if v is not None else "—" for v in row] for row in _z],
                colorbar=dict(title="r", tickfont=dict(color="#9aa0aa"),
                              titlefont=dict(color="#9aa0aa")),
                hovertemplate="Signal: %{y}<br>Horizon: %{x}<br>IS Spearman r: %{text}<extra></extra>",
            ))
            _fig_hg.update_layout(
                height=max(280, len(_y) * 36 + 80),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                margin=dict(l=8, r=8, t=8, b=8),
                xaxis=dict(color="#6b7280", title="Forward horizon"),
                yaxis=dict(color="#6b7280", autorange="reversed"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550",
                                font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig_hg, use_container_width=True)
            st.caption(
                "Green = signal correctly predicts direction at that horizon. "
                "Red = wrong direction. White = no signal. "
                "Banking stress excluded from composite (wrong direction at every horizon)."
            )

            # Role table
            with st.expander("Signal role taxonomy"):
                _role_df = pd.DataFrame([
                    {"Signal": sig, "Role": role}
                    for sig, role in SIGNAL_ROLES.items()
                ])
                st.dataframe(_role_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python app.py` to populate signal columns.")
    except Exception as _val_err:
        st.info(f"Signal horizon grid unavailable. ({_val_err})")

    # ── IS/OOS Summary Table ───────────────────────────────────────────────────
    with st.expander("IS / OOS detail (30d and 60d horizons)"):
        try:
            _sig_val_report = load_signal_validation(df)
            _val_results = _sig_val_report.get("results", {})
            if _val_results:
                _val_rows = []
                for _sig, _ret_dict in _val_results.items():
                    _short = _sig.replace("_score_smooth", "").replace("_smooth", "")
                    for _ret_col, _splits in _ret_dict.items():
                        _short_ret = _ret_col.replace("sp500_forward_", "fwd ").replace("_return", "")
                        _is  = _splits.get("is",  {})
                        _oos = _splits.get("oos", {})
                        _val_rows.append({
                            "Signal":   _short,
                            "Horizon":  _short_ret,
                            "IS r":     _is.get("spearman_r"),
                            "IS hit%":  f"{_is['hit_rate']:.0%}" if _is.get("hit_rate") else "—",
                            "IS n":     _is.get("n"),
                            "OOS r":    _oos.get("spearman_r"),
                            "OOS hit%": f"{_oos['hit_rate']:.0%}" if _oos.get("hit_rate") else "—",
                            "OOS n":    _oos.get("n"),
                        })
                if _val_rows:
                    st.dataframe(
                        pd.DataFrame(_val_rows).style.format({
                            "IS r":  lambda v: f"{v:+.3f}" if v is not None else "—",
                            "OOS r": lambda v: f"{v:+.3f}" if v is not None else "—",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
        except Exception:
            pass

    # ── Stress Episode Stats ───────────────────────────────────────────────────
    st.subheader("Signal Behavior During Named Stress Episodes")
    st.caption(
        "Mean and peak level of each score during seven historical stress episodes. "
        "Scores should be elevated during identified stress periods — low scores during "
        "GFC or COVID indicate the signal was not useful for that stress type."
    )
    try:
        _ep_stats = load_stress_episode_stats(df)
        if not _ep_stats.empty:
            _ep_show = _ep_stats.drop(columns=[("meta", "start"), ("meta", "end")], errors="ignore")
            st.dataframe(_ep_show, use_container_width=True)
        else:
            st.info("Run `python app.py` to generate stress episode data.")
    except Exception as _ep_err:
        st.info(f"Stress episode stats not yet available. ({_ep_err})")

with _analytics_sub2:
    st.header("Signal Attribution")
    st.caption(
        "Explains which sub-scores drive composite risk levels, what triggered each "
        "regime shift, and which scores are systematically elevated for each decision type."
    )

    attr = load_attribution(df)
    rolling = attr["rolling_contributions"]
    shifts  = attr["shift_attribution"]
    elev    = attr["trigger_elevation"]
    drivers = attr["top_drivers_current"]

    # ── Current top drivers ───────────────────────────────────────────────────
    st.subheader("Current Top Drivers")
    col_a, col_b, col_c = st.columns(3)
    for i, col in enumerate([col_a, col_b, col_c]):
        if i < len(drivers):
            d = drivers[i]
            badge = "⬆ Elevated" if d["elevated"] else "Normal"
            col.metric(
                label=d["name"],
                value=f"{d['level']:.1f}",
                delta=f"{d['excess']:+.1f} vs 75th pct — {badge}",
                delta_color="inverse",
            )

    # ── Rolling contributions chart ───────────────────────────────────────────
    st.subheader("Rolling Weighted Contributions to Composite Risk")
    st.caption(
        "Each line = weight × smoothed score. "
        "Mean Reversion is inverted (high MR score = low risk contribution). "
        "Sum approximates the composite risk score."
    )
    contrib_display = rolling.rename(columns={
        f"{k}_contribution": DISPLAY_NAMES[k]
        for k in COMPOSITE_WEIGHTS
        if f"{k}_contribution" in rolling.columns
    })
    if "date" in df.columns:
        contrib_display.index = pd.to_datetime(df["date"])
    st.line_chart(contrib_display, use_container_width=True)

    # ── Shift attribution table ───────────────────────────────────────────────
    st.subheader("Regime Shift Attribution")
    st.caption(
        "Each row = one regime transition. Delta = change in score over the prior "
        "21 trading days. Positive delta = more risk added. "
        "Primary driver = score with the largest absolute move."
    )
    if shifts.empty:
        st.info("No regime transitions found in the dataset.")
    else:
        delta_cols = [c for c in shifts.columns if c.endswith("_delta") and c != "composite_delta"]
        display_cols = ["date", "from_regime", "to_regime",
                        "primary_driver", "secondary_driver",
                        "composite_delta", "direction"] + delta_cols
        # Deduplicate while preserving order (safety net)
        seen: set = set()
        display_cols = [c for c in display_cols if not (c in seen or seen.add(c))]
        shift_disp = shifts[[c for c in display_cols if c in shifts.columns]].copy()
        assert len(shift_disp.columns) == len(set(shift_disp.columns)), "Duplicate columns in shift_disp"
        if "date" in shift_disp.columns:
            shift_disp["date"] = pd.to_datetime(shift_disp["date"]).dt.strftime("%Y-%m-%d")

        # Rename delta cols for readability
        rename_deltas = {
            f"{k}_delta": f"Δ {DISPLAY_NAMES[k]}"
            for k in DISPLAY_NAMES
            if f"{k}_delta" in shift_disp.columns
        }
        shift_disp = shift_disp.rename(columns=rename_deltas)

        st.dataframe(shift_disp, use_container_width=True, hide_index=True)

    # ── Trigger elevation table ───────────────────────────────────────────────
    st.subheader("Score Elevation by Decision Type")
    st.caption(
        "Mean score level while the model was in each decision regime. "
        "Highlighted cells = mean exceeds the global 75th-percentile threshold."
    )
    if elev.empty:
        st.info("No elevation data available.")
    else:
        mean_cols = [c for c in elev.columns if c.endswith("_mean")]
        elev_cols = [c for c in elev.columns if c.endswith("_elevated")]

        # Build a display DataFrame: show means only, with elevation as background
        mean_disp = elev[["n_obs"] + mean_cols].copy()
        mean_disp.columns = ["N Obs"] + [
            DISPLAY_NAMES.get(c.replace("_mean", ""), c) for c in mean_cols
        ]

        # Build boolean mask for styling (same col order, without n_obs)
        bool_mask = elev[elev_cols].copy()
        bool_mask.columns = [
            DISPLAY_NAMES.get(c.replace("_elevated", ""), c) for c in elev_cols
        ]

        score_display_cols = list(bool_mask.columns)

        def highlight_elevated(data):
            style = pd.DataFrame("", index=data.index, columns=data.columns)
            for col in score_display_cols:
                if col in data.columns and col in bool_mask.columns:
                    style[col] = bool_mask[col].map(
                        lambda v: "background-color: #f4c542; font-weight: bold" if v else ""
                    )
            return style

        styled = mean_disp.style.apply(highlight_elevated, axis=None).format(
            {c: "{:.1f}" for c in mean_disp.columns if c != "N Obs"}
        )
        st.dataframe(styled, use_container_width=True)

with _analytics_sub3:
    st.header("Regime Timeline")
    st.caption(
        "Interactive charts with persistent regime color mapping. "
        "Use the date range controls to zoom in on a specific period. "
        "Click legend entries to show/hide individual series."
    )

    # ── Date range selector ───────────────────────────────────────────────────
    if "date" in df.columns:
        min_date = pd.to_datetime(df["date"]).min().date()
        max_date = pd.to_datetime(df["date"]).max().date()
    else:
        min_date = max_date = None

    col_ds, col_de, _ = st.columns([1, 1, 3])
    with col_ds:
        date_start = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="tl_start")
    with col_de:
        date_end = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="tl_end")

    regime_choice = st.radio(
        "Overlay regime",
        ["Final Decision", "Transition Regime"],
        horizontal=True,
        key="tl_regime",
    )
    regime_col_map = {"Final Decision": "final_decision", "Transition Regime": "transition_regime"}
    overlay_col = regime_col_map[regime_choice]

    # ── Decision / Transition regime timeline ────────────────────────────────
    st.subheader("Regime Timeline (Gantt View)")
    st.plotly_chart(
        build_decision_timeline(df, regime_col=overlay_col),
        use_container_width=True,
    )

    # ── SP500 with regime overlay ─────────────────────────────────────────────
    st.subheader("SP500 with Regime Background")
    st.plotly_chart(
        build_sp500_with_regime_overlay(
            df,
            regime_col=overlay_col,
            date_start=date_start,
            date_end=date_end,
        ),
        use_container_width=True,
    )

    # ── Score history ─────────────────────────────────────────────────────────
    st.subheader("Score History")
    st.caption(
        "Macro Risk, Credit Risk, and Composite are visible by default. "
        "Click any legend entry to toggle other scores on or off."
    )
    st.plotly_chart(
        build_score_history(df, date_start=date_start, date_end=date_end),
        use_container_width=True,
    )

    # ── Macro Chronology Overlays ─────────────────────────────────────────────
    st.subheader("Macro Chronology: Composite Score with Historical Event Markers")
    import plotly.graph_objects as _go_chron

    _chron_filter = st.multiselect(
        "Show Fed cycle types",
        ["hike", "cut", "qe", "qt", "zirp"],
        default=["hike", "cut", "qe"],
        key="chron_cycles",
    )

    _chron_evts   = get_events_df()
    _chron_cycles = get_fed_cycles_df()

    _df_chron = df.copy()
    if "date" in _df_chron.columns:
        _chron_x = pd.to_datetime(_df_chron["date"])
    else:
        _chron_x = _df_chron.index

    # Filter by date range selection
    if date_start and date_end:
        _ds = pd.Timestamp(date_start)
        _de = pd.Timestamp(date_end)
        _mask_chron = (_chron_x >= _ds) & (_chron_x <= _de)
        _df_chron = _df_chron[_mask_chron]
        _chron_x  = _chron_x[_mask_chron]
        _chron_evts   = _chron_evts[((_chron_evts.index >= _ds) & (_chron_evts.index <= _de))]
        _chron_cycles = _chron_cycles[(_chron_cycles["end"] >= _ds) & (_chron_cycles["start"] <= _de)]

    _fig_chron = _go_chron.Figure()

    # Fed cycle bands
    for _, _cyc in _chron_cycles.iterrows():
        if _cyc["type"] not in _chron_filter:
            continue
        _fig_chron.add_vrect(
            x0=str(_cyc["start"].date()), x1=str(_cyc["end"].date()),
            fillcolor=_cyc["color"], layer="below", line_width=0,
            annotation_text=_cyc["label"], annotation_position="top left",
            annotation_font_size=8, annotation_font_color="#9ca3af",
        )

    # Composite score
    if "composite_risk_score_smooth" in _df_chron.columns:
        _fig_chron.add_trace(_go_chron.Scatter(
            x=_chron_x, y=_df_chron["composite_risk_score_smooth"],
            mode="lines", name="Composite Risk Score",
            line=dict(color="#4f8ef7", width=2),
        ))

    # Caution / warning lines
    _fig_chron.add_hline(y=50, line_color="rgba(230,126,34,0.5)", line_dash="dot", line_width=1)
    _fig_chron.add_hline(y=70, line_color="rgba(231,76,60,0.5)",  line_dash="dot", line_width=1)

    # Stress event markers
    _cat_colors = {"crisis": "#e74c3c", "episode": "#e67e22", "recovery": "#27ae60"}
    for _ev_date, _ev_row in _chron_evts.iterrows():
        _fig_chron.add_vline(
            x=str(_ev_date.date()),
            line=dict(color=_cat_colors.get(_ev_row["category"], "#888"), width=1, dash="dash"),
            annotation_text=_ev_row["label"],
            annotation_position="top right",
            annotation_font_size=8,
            annotation_font_color=_cat_colors.get(_ev_row["category"], "#888"),
        )

    _fig_chron.update_layout(
        height=420, template="plotly_dark",
        xaxis_title="Date", yaxis_title="Composite Risk Score (0–100)",
        yaxis=dict(range=[0, 100]),
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(_fig_chron, use_container_width=True)

    _cycle_legend = {
        "hike": "Red shading = Fed hiking cycle",
        "cut":  "Green shading = Fed cutting cycle",
        "qe":   "Blue shading = Quantitative Easing",
        "qt":   "Purple shading = Quantitative Tightening",
        "zirp": "Yellow shading = Zero Interest Rate Policy",
    }
    st.caption(" · ".join(_cycle_legend[k] for k in _chron_filter if k in _cycle_legend))

with _analytics_sub4:
    st.header("Signal Decay")
    st.caption(
        "How long does each regime's forward-return edge persist? "
        "Horizons: 7, 14, 30, 60, 90 trading days. "
        "Bootstrap 95% CIs with 500 resamples. "
        "NaN = fewer than 5 clean observations at that horizon."
    )

    _decay = load_signal_decay(df)
    _decay_df = _decay["decay_by_regime"]
    _corr_df  = _decay["score_correlations"]

    if _decay_df.empty:
        st.warning("No decay data — ensure final_decision and sp500 columns are present.")
    else:
        # ── Mean return by horizon (line chart) ──────────────────────────────
        st.subheader("Mean Forward Return by Holding Period")
        st.caption("Each line = one regime. Shaded bands = 95% bootstrap CI.")

        import plotly.graph_objects as go
        from src.regime_charts import DECISION_COLORS

        fig_decay = go.Figure()
        regimes_in_decay = _decay_df.index.get_level_values("regime").unique()

        for regime in regimes_in_decay:
            grp = _decay_df.loc[regime].reset_index()
            color = DECISION_COLORS.get(regime, "#95a5a6")
            has_ci = grp["ci_lower"].notna().any()

            # CI band
            if has_ci:
                x_band = (
                    grp["horizon"].tolist()
                    + grp["horizon"].tolist()[::-1]
                )
                y_band = (
                    grp["ci_upper"].tolist()
                    + grp["ci_lower"].tolist()[::-1]
                )
                fig_decay.add_trace(go.Scatter(
                    x=x_band, y=y_band,
                    fill="toself",
                    fillcolor=color,
                    opacity=0.10,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name=f"{regime}_band",
                ))

            fig_decay.add_trace(go.Scatter(
                x=grp["horizon"],
                y=grp["mean_return"],
                mode="lines+markers",
                name=regime,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                hovertemplate=(
                    f"<b>{regime}</b><br>"
                    "Horizon: %{x}d<br>"
                    "Mean Return: %{y:.2%}<extra></extra>"
                ),
            ))

        fig_decay.add_hline(
            y=0, line=dict(color="#888888", width=1, dash="dash"), opacity=0.5
        )
        fig_decay.update_layout(
            xaxis_title="Holding Period (Trading Days)",
            yaxis_title="Mean Forward Return",
            yaxis_tickformat=".1%",
            height=420,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=45, b=20),
        )
        fig_decay.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                               tickvals=HORIZONS, ticktext=[f"{h}d" for h in HORIZONS])
        fig_decay.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_decay, use_container_width=True)

        # ── Hit rate decay table ──────────────────────────────────────────────
        st.subheader("Hit Rate by Holding Period")
        st.caption("% of periods with positive forward return at each horizon.")

        _hr_pivot = _decay_df["hit_rate"].unstack("horizon")
        _hr_pivot.columns = [f"{h}d" for h in _hr_pivot.columns]

        def _style_hit_rate(val):
            if pd.isna(val):
                return ""
            if val >= 0.60:
                return "background-color: #d5f5e3; font-weight: bold"
            if val >= 0.50:
                return "background-color: #fdfefe"
            return "background-color: #fde8e8"

        styled_hr = _hr_pivot.style.map(_style_hit_rate).format(
            lambda v: f"{v:.0%}" if pd.notna(v) else "—"
        )
        st.dataframe(styled_hr, use_container_width=True)

        # ── N obs table ───────────────────────────────────────────────────────
        with st.expander("Observation counts per (regime, horizon)", expanded=False):
            _n_pivot = _decay_df["n_obs"].unstack("horizon")
            _n_pivot.columns = [f"{h}d" for h in _n_pivot.columns]
            st.dataframe(_n_pivot, use_container_width=True)

    # ── Score correlation decay ───────────────────────────────────────────────
    st.subheader("Score Predictive Power vs Holding Period")
    st.caption(
        "Pearson correlation between each smoothed score and forward SP500 return "
        "at each horizon. Positive = score rise predicts positive returns; "
        "negative = score rise predicts lower returns (risk-off signal working)."
    )

    if _corr_df.empty:
        st.info("No score correlation data available.")
    else:
        _corr_disp = _corr_df.copy()
        _corr_disp.columns = [f"{h}d" for h in _corr_disp.columns]

        def _style_corr(val):
            if pd.isna(val):
                return ""
            if val <= -0.15:
                return "background-color: #d5f5e3; font-weight: bold"
            if val >= 0.15:
                return "background-color: #fde8e8"
            return ""

        styled_corr = _corr_disp.style.map(_style_corr).format(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )
        st.dataframe(styled_corr, use_container_width=True)

with _analytics_sub5:
    st.header("Score Orthogonality")
    st.caption(
        "Measures how independent the six composite sub-scores are from each other. "
        "High multicollinearity reduces the regime engine's discriminating power "
        "and makes composite weights unstable."
    )

    _orth = load_orthogonality(df)
    _corr_m = _orth["correlation_matrix"]
    _vif_df = _orth["vif"]
    _pca    = _orth["pca"]
    _roll   = _orth["rolling_correlations"]
    _osumm  = _orth["summary"]

    # ── Headline metrics ──────────────────────────────────────────────────────
    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric(
        "Effective Rank",
        f"{_osumm.get('effective_rank', '—'):.2f}" if pd.notna(_osumm.get('effective_rank')) else "—",
        help="Shannon-entropy effective dimensionality. 1 = one dominant factor, 6 = fully diverse.",
    )
    oc2.metric(
        "Components for 90% Variance",
        str(_osumm.get("n_components_90pct", "—")),
        help="Fewest PCA components needed to explain 90% of total score variance.",
    )
    oc3.metric(
        "Max Pairwise Correlation",
        f"{_osumm.get('max_pairwise_corr', '—'):.3f}" if pd.notna(_osumm.get('max_pairwise_corr')) else "—",
        help="Highest absolute off-diagonal correlation in the score correlation matrix.",
    )
    oc4.metric(
        "High-VIF Scores",
        str(_osumm.get("n_high_vif", 0)),
        help=f"Scores with VIF ≥ {VIF_HIGH:.0f} (severe multicollinearity).",
    )

    # ── Correlation matrix heatmap ────────────────────────────────────────────
    st.subheader("Pairwise Score Correlations")
    st.caption("Red = strong positive correlation (redundant). Blue = negative. Diagonal = 1.")
    if not _corr_m.empty:
        def _style_corr_cell(val):
            if pd.isna(val) or val == 1.0:
                return ""
            intensity = min(int(abs(val) * 180), 180)
            if val > 0:
                return f"background-color: rgb({255}, {255 - intensity}, {255 - intensity})"
            return f"background-color: rgb({255 - intensity}, {255 - intensity}, {255})"

        styled_matrix = _corr_m.style.map(_style_corr_cell).format("{:.3f}")
        st.dataframe(styled_matrix, use_container_width=True)
    else:
        st.info("Insufficient score columns for correlation matrix.")

    # ── VIF table ─────────────────────────────────────────────────────────────
    st.subheader("Variance Inflation Factors")
    st.caption(
        f"VIF < {VIF_MODERATE:.0f}: Low (acceptable) · "
        f"{VIF_MODERATE:.0f}–{VIF_HIGH:.0f}: Moderate · "
        f"> {VIF_HIGH:.0f}: High (multicollinearity concern)."
    )
    if not _vif_df.empty:
        def _style_vif(row):
            flag = row["flag"]
            color = {"Low": "#d5f5e3", "Moderate": "#fef9e7", "High": "#fde8e8"}.get(flag, "")
            return [f"background-color: {color}"] * len(row)

        vif_disp = _vif_df.copy()
        vif_disp["vif"] = vif_disp["vif"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        styled_vif = vif_disp.style.apply(_style_vif, axis=1)
        st.dataframe(styled_vif, use_container_width=True)
    else:
        st.info("Insufficient scores for VIF computation (need ≥ 3).")

    # ── PCA scree plot ────────────────────────────────────────────────────────
    st.subheader("PCA Explained Variance (Scree Plot)")
    st.caption(
        "How much of total score variance is explained by each principal component. "
        "A single dominant component signals that scores are not truly independent."
    )
    if _pca:
        import plotly.graph_objects as _pgo
        evr = _pca["explained_variance_ratio"]
        cumvar = _pca["cumulative_variance"]
        n_comp = list(range(1, len(evr) + 1))

        fig_pca = _pgo.Figure()
        fig_pca.add_trace(_pgo.Bar(
            x=n_comp, y=[v * 100 for v in evr],
            name="Individual",
            marker_color="#3498db",
            hovertemplate="PC%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig_pca.add_trace(_pgo.Scatter(
            x=n_comp, y=[v * 100 for v in cumvar],
            name="Cumulative",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=6),
            hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
        ))
        fig_pca.add_hline(
            y=90, line=dict(color="#888888", width=1, dash="dash"),
            opacity=0.6, annotation_text="90%", annotation_position="right",
        )
        fig_pca.update_layout(
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance (%)",
            xaxis=dict(tickvals=n_comp, ticktext=[f"PC{i}" for i in n_comp]),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=45, b=20),
        )
        fig_pca.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig_pca.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", range=[0, 105])
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.info("Insufficient data for PCA decomposition.")

    # ── Rolling correlation for top pairs ─────────────────────────────────────
    st.subheader("Rolling Correlations — Most Correlated Pairs")
    st.caption("60-day rolling Pearson correlation. Persistent high values signal redundancy.")
    if _roll:
        import plotly.graph_objects as _rgo
        _PAIR_COLORS = ["#e74c3c", "#e67e22", "#9b59b6"]
        fig_roll = _rgo.Figure()
        for (a, b), series in zip(_roll.keys(), _roll.values()):
            color = _PAIR_COLORS[list(_roll.keys()).index((a, b)) % len(_PAIR_COLORS)]
            fig_roll.add_trace(_rgo.Scatter(
                x=series.index,
                y=series.values,
                name=f"{a} / {b}",
                mode="lines",
                line=dict(color=color, width=1.8),
                hovertemplate=f"{a}/{b}: %{{y:.3f}}<br>%{{x|%Y-%m-%d}}<extra></extra>",
            ))
        fig_roll.add_hline(y=0.7, line=dict(color="#888888", width=1, dash="dot"),
                           opacity=0.5, annotation_text="0.7 threshold", annotation_position="right")
        fig_roll.add_hline(y=0, line=dict(color="#cccccc", width=1), opacity=0.5)
        fig_roll.update_layout(
            xaxis_title="Date", yaxis_title="Correlation",
            yaxis=dict(range=[-1.05, 1.05]),
            height=360, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=45, b=20),
        )
        fig_roll.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig_roll.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("Insufficient data for rolling correlations (need > 60 rows).")

with _analytics_sub6:
    st.header("Tail Risk")
    st.caption(
        "95% CVaR (Conditional Value at Risk / Expected Shortfall) by regime. "
        "CVaR = mean return of the worst 5% of days. "
        "More negative = worse left-tail exposure. "
        "Bootstrap 95% CIs with 500 resamples."
    )

    _tr = load_tail_risk(df)
    _rts  = _tr["regime_tail_stats"]
    _sts  = _tr["strategy_tail_stats"]
    _rcsp = _tr["rolling_cvar_sp500"]
    _rcst = _tr["rolling_cvar_strat"]
    _ddr  = _tr["drawdown_by_regime"]
    _vsb  = _tr["vs_benchmark"]

    # ── Strategy vs SP500 headline ────────────────────────────────────────────
    st.subheader("Strategy vs SP500 — Full Sample Tail")
    if _vsb:
        _labels = {"strategy": "Strategy", "sp500": "SP500 Buy & Hold"}
        _tc = st.columns(len(_vsb) * 3)
        _ci = 0
        for label, stats in _vsb.items():
            _disp = _labels.get(label, label)
            _tc[_ci].metric(f"{_disp} CVaR 95%",
                            f"{stats['cvar']:.2%}" if pd.notna(stats.get('cvar')) else "—")
            _tc[_ci + 1].metric(f"{_disp} VaR 95%",
                                f"{stats['var']:.2%}" if pd.notna(stats.get('var')) else "—")
            _tc[_ci + 2].metric(f"{_disp} Ann. Vol",
                                f"{stats['ann_vol']:.1%}" if pd.notna(stats.get('ann_vol')) else "—")
            _ci += 3

    # ── Rolling CVaR chart ────────────────────────────────────────────────────
    st.subheader("Rolling 60-Day CVaR (95%)")
    st.caption("Mean return of worst 5% of days in each 60-day window. More negative = worse tail risk.")

    import plotly.graph_objects as _trgo
    fig_rcvar = _trgo.Figure()
    if not _rcsp.empty:
        fig_rcvar.add_trace(_trgo.Scatter(
            x=_rcsp.index, y=_rcsp.values,
            name="SP500", mode="lines",
            line=dict(color="#e74c3c", width=1.6),
            hovertemplate="SP500 CVaR: %{y:.2%}<br>%{x|%Y-%m-%d}<extra></extra>",
        ))
    if not _rcst.empty:
        fig_rcvar.add_trace(_trgo.Scatter(
            x=_rcst.index, y=_rcst.values,
            name="Strategy", mode="lines",
            line=dict(color="#27ae60", width=1.6),
            hovertemplate="Strategy CVaR: %{y:.2%}<br>%{x|%Y-%m-%d}<extra></extra>",
        ))
    fig_rcvar.add_hline(y=0, line=dict(color="#cccccc", width=1), opacity=0.6)
    fig_rcvar.update_layout(
        xaxis_title="Date", yaxis_title="CVaR (95%)",
        yaxis_tickformat=".1%", height=360,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    fig_rcvar.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig_rcvar.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    st.plotly_chart(fig_rcvar, use_container_width=True)

    # ── Per-regime tail stats ─────────────────────────────────────────────────
    st.subheader("SP500 Tail Risk by Regime")
    st.caption("Sorted worst-to-best CVaR. CI = 95% bootstrap confidence interval on CVaR.")
    if not _rts.empty:
        def _style_cvar(val):
            if pd.isna(val):
                return ""
            v = pd.to_numeric(val, errors="coerce")
            if pd.isna(v):
                return ""
            if v < -0.025:
                return "background-color: #fde8e8; font-weight: bold"
            if v < -0.015:
                return "background-color: #fef9e7"
            return "background-color: #d5f5e3"

        _pct_cols = ["mean_return", "var_95", "cvar_95", "ci_lower", "ci_upper", "max_loss"]
        _rts_disp = _rts[["n_obs"] + _pct_cols].copy()
        for col in _pct_cols:
            _rts_disp[col] = pd.to_numeric(_rts_disp[col], errors="coerce")
        _rts_disp.columns = ["N Obs", "Mean Return", "VaR 95%", "CVaR 95%",
                              "CI Lower", "CI Upper", "Max Loss"]
        _pct_fmt = {c: lambda v: f"{v:.2%}" if pd.notna(v) else "—"
                    for c in ["Mean Return", "VaR 95%", "CVaR 95%",
                               "CI Lower", "CI Upper", "Max Loss"]}
        st.dataframe(
            _rts_disp.style.map(_style_cvar, subset=["CVaR 95%"]).format(_pct_fmt, na_rep="—"),
            use_container_width=True,
        )
    else:
        st.info("No regime tail stats available.")

    # ── Strategy tail by regime ───────────────────────────────────────────────
    if not _sts.empty:
        with st.expander("Strategy Daily Return Tail by Regime", expanded=False):
            _sts_pct = ["mean_return", "var_95", "cvar_95", "max_loss"]
            _sts_disp = _sts[["n_obs"] + _sts_pct].copy()
            for col in _sts_pct:
                _sts_disp[col] = pd.to_numeric(_sts_disp[col], errors="coerce")
            _sts_disp.columns = ["N Obs", "Mean Return", "VaR 95%", "CVaR 95%", "Max Loss"]
            _sts_fmt = {c: lambda v: f"{v:.2%}" if pd.notna(v) else "—"
                        for c in ["Mean Return", "VaR 95%", "CVaR 95%", "Max Loss"]}
            st.dataframe(
                _sts_disp.style.format(_sts_fmt, na_rep="—"),
                use_container_width=True,
            )

    # ── Forward drawdown by regime ────────────────────────────────────────────
    st.subheader("Future Drawdown at Regime Entry")
    st.caption(
        "Mean and worst SP500 drawdown in the 30/60 days following each regime observation. "
        "Sorted worst mean 30d drawdown first."
    )
    if not _ddr.empty:
        _ddr_disp = _ddr.copy()
        dd_fmt_cols = [c for c in _ddr_disp.columns if "dd" in c]
        for col in dd_fmt_cols:
            _ddr_disp[col] = pd.to_numeric(_ddr_disp[col], errors="coerce")
        col_rename = {
            "n_obs": "N Obs",
            "mean_dd_30d": "Mean DD 30d", "worst_dd_30d": "Worst DD 30d",
            "mean_dd_60d": "Mean DD 60d", "worst_dd_60d": "Worst DD 60d",
        }
        _ddr_disp = _ddr_disp.rename(columns={k: v for k, v in col_rename.items()
                                               if k in _ddr_disp.columns})
        _dd_display_cols = [v for v in col_rename.values() if v in _ddr_disp.columns and v != "N Obs"]
        _dd_fmt = {c: lambda v: f"{v:.2%}" if pd.notna(v) else "—" for c in _dd_display_cols}
        st.dataframe(
            _ddr_disp.style.format(_dd_fmt, na_rep="—"),
            use_container_width=True,
        )
    else:
        st.info("No future drawdown data available.")

with _analytics_sub7:
    st.header("Stress Episode Analysis")
    st.caption(
        "How did the composite risk signal behave during known market stress periods? "
        "Episodes within the dataset range show actual model statistics. "
        "Historical episodes are reference markers only."
    )

    _stress = load_stress_analysis(df)
    _ep_stats = _stress["episode_stats"]
    _n_cov = _stress["n_covered"]
    _n_hist = _stress["n_historical"]

    # ── Summary banner ────────────────────────────────────────────────────────
    _bc1, _bc2 = st.columns(2)
    _bc1.metric("Covered episodes (in dataset)", _n_cov)
    _bc2.metric("Historical reference episodes", _n_hist)

    # ── Timeline chart ────────────────────────────────────────────────────────
    st.subheader("Composite Risk Score — Stress Episode Overlay")
    st.plotly_chart(_stress["timeline_chart"], use_container_width=True)

    # ── Episode stats table ───────────────────────────────────────────────────
    st.subheader("Episode Summary")
    st.caption(
        "Mean/peak composite score, SP500 drawdown, primary model decision, "
        "and % of days in high-alert territory (composite ≥ 60). "
        "Greyed rows = outside dataset range."
    )

    if not _ep_stats.empty:
        _SEVERITY_BADGE = {"Severe": "🔴", "Moderate": "🟠", "Mild": "🟡"}

        _ep_disp = _ep_stats[["label", "severity", "start", "end", "covered",
                               "n_obs", "mean_composite", "peak_composite",
                               "sp500_drawdown", "primary_decision", "high_alert_pct"]].copy()

        # Coerce numerics
        for _col in ["mean_composite", "peak_composite", "sp500_drawdown", "high_alert_pct"]:
            _ep_disp[_col] = pd.to_numeric(_ep_disp[_col], errors="coerce")

        _ep_disp = _ep_disp.rename(columns={
            "label":            "Episode",
            "severity":         "Severity",
            "start":            "Start",
            "end":              "End",
            "covered":          "In Dataset",
            "n_obs":            "N Days",
            "mean_composite":   "Mean Score",
            "peak_composite":   "Peak Score",
            "sp500_drawdown":   "SP500 Drawdown",
            "primary_decision": "Primary Decision",
            "high_alert_pct":   "High Alert %",
        })

        def _style_episode_row(row):
            if not row["In Dataset"]:
                return ["color: #aaa"] * len(row)
            return [""] * len(row)

        _ep_fmt = {
            "Mean Score":     lambda v: f"{v:.1f}" if pd.notna(v) else "—",
            "Peak Score":     lambda v: f"{v:.1f}" if pd.notna(v) else "—",
            "SP500 Drawdown": lambda v: f"{v:.2%}" if pd.notna(v) else "—",
            "High Alert %":   lambda v: f"{v:.0%}" if pd.notna(v) else "—",
        }

        st.dataframe(
            _ep_disp.style.apply(_style_episode_row, axis=1).format(_ep_fmt, na_rep="—"),
            use_container_width=True,
        )

    # ── Per-episode score path drilldown ──────────────────────────────────────
    _paths = _stress["score_paths"]
    if _paths:
        st.subheader("Score Path Drilldown")
        st.caption("Select a covered episode to see the composite score trajectory with ±30-day context.")
        _ep_names = list(_paths.keys())
        _selected = st.selectbox("Episode", _ep_names)
        if _selected and _selected in _paths:
            _path_df = _paths[_selected]
            _ep_meta = next((e for e in STRESS_EPISODES if e["name"] == _selected), None)

            import plotly.graph_objects as _go
            _pfig = _go.Figure()
            _pfig.add_trace(_go.Scatter(
                x=_path_df["date"],
                y=_path_df["composite"] if "composite" in _path_df.columns else [],
                mode="lines",
                name="Composite Score",
                line=dict(color="#1a1a2e", width=2),
            ))
            _pfig.add_hline(y=60, line_dash="dot", line_color="#e74c3c",
                            line_width=1, annotation_text="High Alert (60)")
            if _ep_meta:
                _pfig.add_vrect(
                    x0=_ep_meta["start"], x1=_ep_meta["end"],
                    fillcolor="rgba(230,126,34,0.15)", opacity=1,
                    layer="below", line_width=0,
                    annotation_text=_selected, annotation_position="top left",
                    annotation_font_size=10,
                )
            _pfig.update_layout(
                title=f"Score Path — {_selected}",
                xaxis_title="Date", yaxis_title="Composite Risk Score",
                yaxis=dict(range=[0, 105]),
                height=340, template="plotly_white",
                margin=dict(l=50, r=20, t=50, b=40),
            )
            st.plotly_chart(_pfig, use_container_width=True)

            # Decision breakdown for this episode
            if "decision" in _path_df.columns:
                _ep_start = pd.Timestamp(_ep_meta["start"]) if _ep_meta else _path_df["date"].min()
                _ep_end   = pd.Timestamp(_ep_meta["end"])   if _ep_meta else _path_df["date"].max()
                _in_ep = _path_df[(_path_df["date"] >= _ep_start) & (_path_df["date"] <= _ep_end)]
                if not _in_ep.empty:
                    _dec_counts = _in_ep["decision"].value_counts().reset_index()
                    _dec_counts.columns = ["Decision", "Days"]
                    _dec_counts["% of Episode"] = _dec_counts["Days"] / _dec_counts["Days"].sum()
                    st.dataframe(
                        _dec_counts.style.format({"% of Episode": "{:.0%}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
    else:
        st.info("No covered episodes in the current dataset range.")

with _analytics_sub8:
    import plotly.graph_objects as _go

    st.header("Performance Scorecard")
    st.caption(
        "Full-period and rolling risk-adjusted metrics for the strategy vs. SP500 "
        "buy-and-hold. Rolling windows: 63 trading days (~3 months), "
        "126 trading days (~6 months)."
    )

    _sc = load_performance_scorecard(df)
    _fp = _sc["full_period"]
    _roll = _sc["rolling"]
    _rp = _sc["regime_perf"]
    _cal = _sc["monthly_cal"]
    _wins = _sc["windows"]

    # ── Full-period headline metrics ──────────────────────────────────────────
    st.subheader("Full-Period Summary")

    if _fp:
        _labels = [("strategy", "Strategy"), ("sp500", "SP500 Buy & Hold")]
        _metric_rows = [
            ("Total Return",     "total_return",  "{:.2%}"),
            ("Ann. Return",      "ann_return",    "{:.2%}"),
            ("Ann. Volatility",  "ann_vol",       "{:.2%}"),
            ("Sharpe Ratio",     "sharpe",        "{:.2f}"),
            ("Max Drawdown",     "max_drawdown",  "{:.2%}"),
            ("Calmar Ratio",     "calmar",        "{:.2f}"),
            ("Hit Rate",         "hit_rate",      "{:.1%}"),
            ("Best Day",         "best_day",      "{:.2%}"),
            ("Worst Day",        "worst_day",     "{:.2%}"),
            ("Observations",     "n_obs",         "{:,}"),
        ]

        _fp_rows = []
        for metric_label, key, fmt in _metric_rows:
            row = {"Metric": metric_label}
            for col_key, col_label in _labels:
                val = _fp.get(col_key, {}).get(key)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    row[col_label] = "—"
                else:
                    try:
                        row[col_label] = fmt.format(val)
                    except (ValueError, TypeError):
                        row[col_label] = str(val)
            _fp_rows.append(row)

        _fp_disp = pd.DataFrame(_fp_rows)

        def _style_fp(row):
            styles = ["font-weight:600"] + [""] * (len(row) - 1)
            metric = row["Metric"]
            if metric in ("Sharpe Ratio", "Calmar Ratio", "Total Return",
                          "Ann. Return", "Hit Rate", "Best Day"):
                # higher is better
                s_val = _fp.get("strategy", {}).get(
                    {"Sharpe Ratio": "sharpe", "Calmar Ratio": "calmar",
                     "Total Return": "total_return", "Ann. Return": "ann_return",
                     "Hit Rate": "hit_rate", "Best Day": "best_day"}[metric])
                b_val = _fp.get("sp500", {}).get(
                    {"Sharpe Ratio": "sharpe", "Calmar Ratio": "calmar",
                     "Total Return": "total_return", "Ann. Return": "ann_return",
                     "Hit Rate": "hit_rate", "Best Day": "best_day"}[metric])
                if s_val is not None and b_val is not None and not pd.isna(s_val) and not pd.isna(b_val):
                    if "Strategy" in _fp_disp.columns:
                        idx_s = _fp_disp.columns.get_loc("Strategy")
                        if s_val > b_val:
                            styles[idx_s] = "background-color:#e8f5e9"
                        elif s_val < b_val:
                            styles[idx_s] = "background-color:#fff3e0"
            return styles

        st.dataframe(_fp_disp.style.apply(_style_fp, axis=1),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No return data available.")

    # ── Rolling Sharpe chart ──────────────────────────────────────────────────
    st.subheader("Rolling Sharpe Ratio")
    _win_choice = st.radio("Window", [f"{w}d" for w in _wins],
                           horizontal=True, key="perf_window")
    _w = int(_win_choice.replace("d", ""))

    if _w in _roll and not _roll[_w].empty:
        _rf = _roll[_w]
        _rfig = _go.Figure()
        if "strategy_sharpe" in _rf.columns:
            _rfig.add_trace(_go.Scatter(
                x=_rf.index, y=_rf["strategy_sharpe"],
                mode="lines", name="Strategy",
                line=dict(color="#1a1a2e", width=1.8),
            ))
        if "sp500_sharpe" in _rf.columns:
            _rfig.add_trace(_go.Scatter(
                x=_rf.index, y=_rf["sp500_sharpe"],
                mode="lines", name="SP500",
                line=dict(color="#95a5a6", width=1.2, dash="dot"),
            ))
        _rfig.add_hline(y=0, line_color="#e74c3c", line_width=1, line_dash="dash")
        _rfig.update_layout(
            xaxis_title="Date", yaxis_title=f"Rolling {_w}d Sharpe",
            height=340, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(_rfig, use_container_width=True)

    # ── Rolling max drawdown chart ────────────────────────────────────────────
    st.subheader("Rolling Max Drawdown")
    if _w in _roll and not _roll[_w].empty:
        _rf = _roll[_w]
        _ddfig = _go.Figure()
        if "strategy_max_dd" in _rf.columns:
            _ddfig.add_trace(_go.Scatter(
                x=_rf.index, y=_rf["strategy_max_dd"],
                mode="lines", name="Strategy",
                line=dict(color="#e74c3c", width=1.8),
                fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
            ))
        if "sp500_max_dd" in _rf.columns:
            _ddfig.add_trace(_go.Scatter(
                x=_rf.index, y=_rf["sp500_max_dd"],
                mode="lines", name="SP500",
                line=dict(color="#95a5a6", width=1.2, dash="dot"),
            ))
        _ddfig.update_layout(
            xaxis_title="Date", yaxis_title=f"Rolling {_w}d Max Drawdown",
            height=300, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(_ddfig, use_container_width=True)

    # ── Regime-conditional performance ────────────────────────────────────────
    st.subheader("Performance by Regime")
    st.caption(
        "Strategy and SP500 returns grouped by the model's active regime. "
        "Sorted by strategy Sharpe (best first). Requires ≥20 observations for "
        "annualised metrics."
    )
    if not _rp.empty:
        _rp_disp = _rp.copy()
        _rp_disp = _rp_disp.rename(columns={
            "n_obs":            "N Days",
            "strat_mean":       "Strat Daily Mean",
            "strat_hit_rate":   "Strat Hit Rate",
            "strat_ann_return": "Strat Ann Return",
            "strat_ann_vol":    "Strat Ann Vol",
            "strat_sharpe":     "Strat Sharpe",
            "sp500_mean":       "SP500 Daily Mean",
            "sp500_sharpe":     "SP500 Sharpe",
        })
        _pct_cols = [c for c in _rp_disp.columns
                     if any(k in c for k in ["Mean", "Return", "Vol", "Rate"])]
        _num_cols = ["Strat Sharpe", "SP500 Sharpe"]
        for _col in _pct_cols + _num_cols:
            if _col in _rp_disp.columns:
                _rp_disp[_col] = pd.to_numeric(_rp_disp[_col], errors="coerce")

        _rp_fmt = {c: (lambda v: f"{v:.2%}" if pd.notna(v) else "—")
                   for c in _pct_cols if c in _rp_disp.columns}
        _rp_fmt.update({c: (lambda v: f"{v:.2f}" if pd.notna(v) else "—")
                        for c in _num_cols if c in _rp_disp.columns})

        def _style_sharpe(val):
            if pd.isna(val):
                return ""
            if val > 0.5:
                return "background-color:#e8f5e9"
            if val < 0:
                return "background-color:#fce4e4"
            return ""

        _display_cols = [c for c in ["N Days", "Strat Daily Mean", "Strat Hit Rate",
                                      "Strat Ann Return", "Strat Ann Vol", "Strat Sharpe",
                                      "SP500 Daily Mean", "SP500 Sharpe"]
                         if c in _rp_disp.columns]
        st.dataframe(
            _rp_disp[_display_cols].style
                .map(_style_sharpe, subset=[c for c in ["Strat Sharpe"] if c in _display_cols])
                .format(_rp_fmt, na_rep="—"),
            use_container_width=True,
        )
    else:
        st.info("No regime performance data available.")

    # ── Monthly return calendar ───────────────────────────────────────────────
    st.subheader("Monthly Return Calendar")
    _MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    for _cal_label, _cal_title in [("strategy", "Strategy"), ("sp500", "SP500 Buy & Hold")]:
        if _cal_label not in _cal:
            continue
        _pivot = _cal[_cal_label].copy()
        _pivot.columns = [_MONTH_NAMES.get(c, c) for c in _pivot.columns]
        _pivot.index.name = "Year"

        def _fmt_cal(v):
            return f"{v:.1%}" if pd.notna(v) else "—"

        def _style_cal(val):
            if pd.isna(val):
                return "color:#ccc"
            if val > 0.02:
                return "background-color:#c8e6c9;color:#1b5e20"
            if val > 0:
                return "background-color:#e8f5e9"
            if val > -0.02:
                return "background-color:#fff3e0"
            return "background-color:#ffcdd2;color:#b71c1c"

        with st.expander(f"{_cal_title} — Monthly Returns", expanded=(_cal_label == "strategy")):
            st.dataframe(
                _pivot.style.map(_style_cal).format(_fmt_cal, na_rep="—"),
                use_container_width=True,
            )

with _analytics_sub9:
    import plotly.graph_objects as _go16

    st.header("Factor Exposure & Beta Decomposition")
    st.caption(
        "Single-factor OLS: strategy return = α + β × SP500 return + ε. "
        "Quantifies how much of the strategy's return and risk is market beta "
        "vs. genuine alpha. Rolling windows show how exposure shifts across regimes."
    )

    _fa = load_factor_analysis(df)
    _reg = _fa["regression"]
    _fa_roll = _fa["rolling"]
    _rb = _fa["regime_beta"]
    _decomp = _fa["decomposition"]
    _fa_wins = _fa["windows"]

    # ── Full-period headline metrics ──────────────────────────────────────────
    if _reg:
        st.subheader("Full-Period Regression")
        _fc1, _fc2, _fc3, _fc4, _fc5 = st.columns(5)
        _fc1.metric("Market Beta",      f"{_reg.get('beta', float('nan')):.3f}")
        _fc2.metric("Ann. Alpha",       f"{_reg.get('ann_alpha', float('nan')):.2%}")
        _fc3.metric("R²",               f"{_reg.get('r2', float('nan')):.3f}")
        _fc4.metric("Info Ratio",       f"{_reg.get('info_ratio', float('nan')):.3f}")
        _fc5.metric("Tracking Error",   f"{_reg.get('tracking_error', float('nan')):.2%}")

        with st.expander("Interpretation", expanded=False):
            _beta_v = _reg.get("beta", float("nan"))
            _ir_v   = _reg.get("info_ratio", float("nan"))
            _r2_v   = _reg.get("r2", float("nan"))
            st.markdown(
                f"- **Beta {_beta_v:.2f}**: strategy holds ~{_beta_v:.0%} of market exposure "
                f"on average — lower than 1 confirms the signal actively manages equity weight.\n"
                f"- **R² {_r2_v:.2f}**: {_r2_v:.0%} of strategy variance is explained by SP500 "
                f"moves; the remaining {1-_r2_v:.0%} is idiosyncratic.\n"
                f"- **Info Ratio {_ir_v:.2f}**: annualised alpha per unit of residual risk. "
                f"{'Positive — alpha is additive.' if not pd.isna(_ir_v) and _ir_v > 0 else 'Negative or insufficient data.'}\n"
                f"- **Tracking Error {_reg.get('tracking_error', float('nan')):.2%}**: "
                f"annualised standard deviation of daily return differences vs. buy-and-hold."
            )

    # ── Rolling beta chart ────────────────────────────────────────────────────
    st.subheader("Rolling Market Beta")
    _fa_win_choice = st.radio(
        "Window", [f"{w}d" for w in _fa_wins], horizontal=True, key="fa_window"
    )
    _fa_w = int(_fa_win_choice.replace("d", ""))

    if _fa_w in _fa_roll and not _fa_roll[_fa_w].empty:
        _rfr = _fa_roll[_fa_w]
        _beta_fig = _go16.Figure()
        if "beta" in _rfr.columns:
            _beta_fig.add_trace(_go16.Scatter(
                x=_rfr.index, y=_rfr["beta"],
                mode="lines", name=f"Rolling {_fa_w}d Beta",
                line=dict(color="#1a1a2e", width=1.8),
            ))
        _beta_fig.add_hline(
            y=_reg.get("beta", 0), line_dash="dot",
            line_color="#e67e22", line_width=1,
            annotation_text=f"Full-period β={_reg.get('beta', 0):.2f}",
            annotation_position="top right", annotation_font_size=10,
        )
        _beta_fig.add_hline(y=1.0, line_dash="dash", line_color="#95a5a6",
                             line_width=1, annotation_text="β=1 (Buy & Hold)",
                             annotation_position="bottom right", annotation_font_size=10)
        _beta_fig.add_hline(y=0.0, line_color="#ddd", line_width=1)
        _beta_fig.update_layout(
            xaxis_title="Date", yaxis_title="Rolling Beta",
            height=340, template="plotly_white",
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(_beta_fig, use_container_width=True)

    # ── Rolling alpha chart ───────────────────────────────────────────────────
    st.subheader("Rolling Annualised Alpha")
    if _fa_w in _fa_roll and not _fa_roll[_fa_w].empty:
        _rfr = _fa_roll[_fa_w]
        _alpha_fig = _go16.Figure()
        if "ann_alpha" in _rfr.columns:
            _alpha_vals = _rfr["ann_alpha"]
            _alpha_fig.add_trace(_go16.Scatter(
                x=_rfr.index, y=_alpha_vals,
                mode="lines", name=f"Rolling {_fa_w}d Ann. Alpha",
                line=dict(color="#27ae60", width=1.8),
                fill="tozeroy",
                fillcolor="rgba(39,174,96,0.08)",
            ))
        _alpha_fig.add_hline(y=0, line_color="#e74c3c", line_width=1, line_dash="dash")
        _alpha_fig.update_layout(
            xaxis_title="Date", yaxis_title="Annualised Alpha",
            yaxis_tickformat=".1%",
            height=300, template="plotly_white",
            margin=dict(l=50, r=20, t=40, b=40),
        )
        st.plotly_chart(_alpha_fig, use_container_width=True)

    # ── Regime beta table ─────────────────────────────────────────────────────
    st.subheader("Market Beta by Regime")
    st.caption(
        "Average market exposure in each model regime. "
        "Low beta in risk-off regimes and high beta in entry regimes "
        "validates that the signal correctly modulates market exposure. "
        "Sorted lowest beta first."
    )
    if not _rb.empty:
        _rb_disp = _rb.copy()
        for _col in ["beta", "ann_alpha", "r2", "residual_vol"]:
            if _col in _rb_disp.columns:
                _rb_disp[_col] = pd.to_numeric(_rb_disp[_col], errors="coerce")
        _rb_disp = _rb_disp.rename(columns={
            "n_obs":        "N Days",
            "beta":         "Beta",
            "ann_alpha":    "Ann Alpha",
            "r2":           "R²",
            "residual_vol": "Residual Vol",
        })
        _rb_fmt = {
            "Beta":         lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "Ann Alpha":    lambda v: f"{v:.2%}" if pd.notna(v) else "—",
            "R²":           lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "Residual Vol": lambda v: f"{v:.2%}" if pd.notna(v) else "—",
        }

        def _style_beta(val):
            if pd.isna(val):
                return ""
            if val < 0.35:
                return "background-color:#e3f2fd"   # very low — blue tint
            if val > 0.80:
                return "background-color:#fff3e0"   # high — amber tint
            return ""

        _rb_display_cols = [c for c in ["N Days", "Beta", "Ann Alpha", "R²", "Residual Vol"]
                            if c in _rb_disp.columns]
        st.dataframe(
            _rb_disp[_rb_display_cols].style
                .map(_style_beta, subset=["Beta"])
                .format(_rb_fmt, na_rep="—"),
            use_container_width=True,
        )

    # ── Cumulative return decomposition chart ─────────────────────────────────
    st.subheader("Cumulative Return Decomposition")
    st.caption(
        "Strategy return decomposed into market-beta contribution "
        "(β × SP500) and residual alpha. If alpha curve rises over time, "
        "the signal is generating genuine excess return beyond market exposure."
    )
    if not _decomp.empty:
        _decomp_date = _decomp["date"] if "date" in _decomp.columns else _decomp.index
        _decomp_fig = _go16.Figure()
        _decomp_fig.add_trace(_go16.Scatter(
            x=_decomp_date, y=_decomp["cum_strategy"],
            mode="lines", name="Strategy (total)",
            line=dict(color="#1a1a2e", width=2),
        ))
        _decomp_fig.add_trace(_go16.Scatter(
            x=_decomp_date, y=_decomp["cum_sp500"],
            mode="lines", name="SP500 Buy & Hold",
            line=dict(color="#95a5a6", width=1.2, dash="dot"),
        ))
        _decomp_fig.add_trace(_go16.Scatter(
            x=_decomp_date, y=_decomp["cum_beta"],
            mode="lines", name="Beta contribution (β × SP500)",
            line=dict(color="#e67e22", width=1.5, dash="dash"),
        ))
        _decomp_fig.add_trace(_go16.Scatter(
            x=_decomp_date, y=_decomp["cum_alpha"],
            mode="lines", name="Alpha contribution (residual)",
            line=dict(color="#27ae60", width=1.5),
            fill="tozeroy", fillcolor="rgba(39,174,96,0.07)",
        ))
        _decomp_fig.add_hline(y=1.0, line_color="#ddd", line_width=1)
        _decomp_fig.update_layout(
            xaxis_title="Date", yaxis_title="Cumulative Return (rebased to 1.0)",
            height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=50, r=20, t=50, b=40),
        )
        st.plotly_chart(_decomp_fig, use_container_width=True)

    # ── Multi-factor regression ───────────────────────────────────────────────
    st.subheader("Multi-Factor Regression")
    st.caption(
        "5-factor OLS: strategy ~ α + β_SP500×SP500 + β_VIX×ΔVIX + β_HY×ΔHY + "
        "β_Rates×ΔRates + β_Mom×Momentum. Quantifies whether returns come from "
        "beta reduction, volatility timing, credit avoidance, or duration positioning."
    )
    _mf = _fa.get("multi_factor", {})
    if _mf and "error" not in _mf and "factor_betas" in _mf:
        _mf_c1, _mf_c2, _mf_c3 = st.columns(3)
        _mf_c1.metric("Multi-Factor R²",   f"{_mf.get('r2', float('nan')):.3f}")
        _mf_c2.metric("Ann. Alpha (MF)",   f"{_mf.get('ann_alpha', float('nan')):.2%}")
        _mf_c3.metric("Residual Vol",      f"{_mf.get('residual_vol', float('nan')):.2%}")

        _beta_df = pd.DataFrame(_mf["factor_betas"]).set_index("factor")
        import plotly.graph_objects as _mf_go
        _mf_fig = _mf_go.Figure(_mf_go.Bar(
            x=_beta_df.index.tolist(),
            y=_beta_df["beta"].tolist(),
            marker_color=["#e74c3c" if v < 0 else "#27ae60" for v in _beta_df["beta"]],
            text=[f"{v:+.4f}" for v in _beta_df["beta"]],
            textposition="outside",
        ))
        _mf_fig.update_layout(
            title="Factor Betas (multi-factor OLS)",
            yaxis_title="Beta coefficient",
            height=320, template="plotly_dark",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(_mf_fig, use_container_width=True)
        st.caption(
            "Beta interpretation: SP500 beta = directional market exposure; "
            "VIX Δ beta < 0 = strategy profits when VIX rises (volatility timing); "
            "HY Δ beta < 0 = strategy profits during spread widening (credit avoidance); "
            "negative Rates Δ beta = profits from rate rises (short duration bias)."
        )
    elif _mf and "error" in _mf:
        st.info(f"Multi-factor: {_mf['error']}")
    else:
        st.info("Multi-factor regression not available for this dataset.")

    # ── Performance Attribution Decomposition ────────────────────────────────
    st.subheader("Performance Attribution Decomposition")
    st.caption(
        "Breaks total strategy return into distinct economic contributions: "
        "beta reduction (defensive positioning), volatility/regime timing, "
        "crash avoidance, regime transition P&L, and residual alpha. "
        "Honest accounting — cost of being wrong is reflected in the residuals."
    )
    with st.spinner("Computing attribution…"):
        _attr_decomp = load_performance_attribution(df)

    if _attr_decomp and "summary" in _attr_decomp:
        _attr_s = _attr_decomp["summary"]
        _attr_sc1, _attr_sc2, _attr_sc3 = st.columns(3)
        _attr_sc1.metric(
            "Total Strategy (Ann.)",
            f"{_attr_decomp.get('ann_strategy', 0):.2%}" if _attr_decomp.get('ann_strategy') else "—",
        )
        _attr_sc2.metric(
            "Beta Reduction Share",
            f"{_attr_decomp.get('beta_share_pct', 0):.0f}%" if _attr_decomp.get('beta_share_pct') else "—",
            help="Fraction of total strategy return attributable to holding less-than-market beta",
        )
        _attr_sc3.metric(
            "Timing Share",
            f"{_attr_decomp.get('timing_share_pct', 0):.0f}%" if _attr_decomp.get('timing_share_pct') else "—",
            help="Fraction attributable to varying beta over time vs constant-beta reference",
        )

        import plotly.graph_objects as _go_attr
        _attr_fig = _go_attr.Figure(_go_attr.Bar(
            y=_attr_s["component"].tolist(),
            x=[v * 100 if v is not None and not pd.isna(v) else 0
               for v in _attr_s["ann_return"].tolist()],
            orientation="h",
            marker_color=["#27ae60" if (v or 0) >= 0 else "#e74c3c"
                          for v in _attr_s["ann_return"].tolist()],
            text=[f"{(v or 0):.2%}" for v in _attr_s["ann_return"].tolist()],
            textposition="outside",
        ))
        _attr_fig.add_vline(x=0, line_color="#555", line_width=1)
        _attr_fig.update_layout(
            xaxis_title="Annualised Contribution (%)",
            height=300, template="plotly_dark",
            margin=dict(l=200, r=80, t=30, b=40),
            xaxis_ticksuffix="%",
        )
        st.plotly_chart(_attr_fig, use_container_width=True)

        with st.expander("Attribution details"):
            _attr_display = _attr_s.copy()
            _attr_display["ann_return"] = _attr_display["ann_return"].apply(
                lambda v: f"{v:.2%}" if v is not None and not pd.isna(v) else "—"
            )
            st.dataframe(_attr_display.set_index("component"), use_container_width=True)
    else:
        st.info("Attribution decomposition not available — check strategy return columns.")

with _models_sub3:
    st.header("Regime Probability Nowcast")
    st.caption(
        "Gaussian Naive Bayes posterior over regime labels, fitted in-sample "
        "on all 7 component scores. P(regime | scores) ∝ P(scores | regime) × P(regime)."
    )

    _rp17 = load_regime_probability(df)
    _cur17 = _rp17.get("current", {})
    _hist17 = _rp17.get("history", pd.DataFrame())
    _mdl17  = _rp17.get("model", {})

    if not _cur17:
        st.warning("Could not fit regime probability model — check data.")
    else:
        import plotly.graph_objects as _go17

        # ── Current distribution ─────────────────────────────────────────────
        st.subheader("Current Regime Probabilities")
        _probs_sorted = sorted(_cur17["probs"].items(), key=lambda kv: kv[1], reverse=True)
        _COLORS17 = {
            "Risk-On":  "#27ae60",
            "Neutral":  "#95a5a6",
            "Caution":  "#e67e22",
            "Risk-Off": "#e74c3c",
        }
        _bar_labels = [r for r, _ in _probs_sorted]
        _bar_vals   = [p for _, p in _probs_sorted]
        _bar_colors = [_COLORS17.get(r, "#aaa") for r in _bar_labels]

        _fig_bar = _go17.Figure(_go17.Bar(
            y=_bar_labels,
            x=_bar_vals,
            orientation="h",
            marker_color=_bar_colors,
            text=[f"{v:.1%}" for v in _bar_vals],
            textposition="outside",
        ))
        _fig_bar.update_layout(
            height=max(250, len(_bar_labels) * 40),
            xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=180, r=60, t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        _fig_bar.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(_fig_bar, use_container_width=True)

        _p17c1, _p17c2, _p17c3 = st.columns(3)
        _p17c1.metric("Top Regime",    _cur17["top_regime"])
        _p17c2.metric("Confidence",    f"{_cur17['top_prob']:.1%}")
        _p17c3.metric("Entropy",       f"{_cur17['entropy']:.2f} bits",
                      help="0 bits = certain; log₂(N regimes) = maximum uncertainty")

        if "second" in _cur17:
            st.caption(
                f"Second: **{_cur17['second']}** at {_cur17['second_prob']:.1%}  ·  "
                f"Model covers {len(_mdl17)} regimes "
                f"(min {min(v['n_obs'] for v in _mdl17.values())} obs each)"
            )

        st.divider()

        # ── Probability history ──────────────────────────────────────────────
        if not _hist17.empty:
            st.subheader("Regime Probability Over Time")
            _regime_cols17 = [c for c in _hist17.columns
                              if c not in ("top_regime", "entropy")]
            _sorted_regimes = sorted(
                _regime_cols17,
                key=lambda r: _cur17["probs"].get(r, 0),
                reverse=True,
            )

            _fig_stack = _go17.Figure()
            for _r17 in reversed(_sorted_regimes):
                _fig_stack.add_trace(_go17.Scatter(
                    x=_hist17.index,
                    y=_hist17[_r17],
                    name=_r17,
                    stackgroup="one",
                    mode="none",
                    fillcolor=_COLORS17.get(_r17, "#ccc"),
                    line=dict(width=0),
                ))
            _fig_stack.update_layout(
                height=320,
                yaxis=dict(tickformat=".0%", range=[0, 1], title="Probability"),
                xaxis=dict(title=None, showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                            traceorder="reversed"),
                margin=dict(l=50, r=20, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            st.plotly_chart(_fig_stack, use_container_width=True)

            # ── Entropy trend ────────────────────────────────────────────────
            st.subheader("Model Uncertainty Over Time (Shannon Entropy)")
            _fig_ent = _go17.Figure(_go17.Scatter(
                x=_hist17.index,
                y=_hist17["entropy"],
                mode="lines",
                line=dict(color="#7f7f7f", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(127,127,127,0.08)",
            ))
            _max_ent17 = round(float(pd.Series([1]).apply(
                lambda _: __import__("math").log2(len(_sorted_regimes))
            ).iloc[0]), 2)
            _fig_ent.add_hline(
                y=_max_ent17, line_dash="dot", line_color="#bbb",
                annotation_text=f"Max entropy ({_max_ent17:.2f} bits)",
                annotation_position="bottom right",
            )
            _fig_ent.update_layout(
                height=200,
                yaxis=dict(title="bits", range=[0, _max_ent17 * 1.1]),
                xaxis=dict(title=None, showgrid=False),
                margin=dict(l=50, r=20, t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(_fig_ent, use_container_width=True)
            st.caption(
                "Low entropy = model is confident in a single regime. "
                "High entropy = scores consistent with multiple regimes simultaneously."
            )

        # ── Model parameters ────────────────────────────────────────────────
        with st.expander("Model parameters (regime centroids)"):
            _feat_labels = {
                "macro_risk_score_smooth":          "Macro Risk",
                "credit_market_risk_score_smooth":  "Credit Risk",
                "liquidity_regime_score_smooth":    "Liquidity",
                "complacency_score_smooth":         "Complacency",
                "mean_reversion_score_smooth":      "Mean Reversion",
                "risk_appetite_score_smooth":       "Risk Appetite",
                "treasury_stress_score_smooth":     "Treasury Stress",
            }
            _rows17 = []
            for _r17, _p17 in sorted(_mdl17.items(),
                                     key=lambda kv: kv[1]["prior"], reverse=True):
                _row = {"Regime": _r17, "Prior": f"{_p17['prior']:.1%}",
                        "N": _p17["n_obs"]}
                for _fi, _fn in enumerate(_p17["features"]):
                    _lbl = _feat_labels.get(_fn, _fn)
                    _row[_lbl] = f"{_p17['means'][_fi]:.1f} ± {_p17['stds'][_fi]:.1f}"
                _rows17.append(_row)
            st.dataframe(pd.DataFrame(_rows17).set_index("Regime"),
                         use_container_width=True)

with _models_sub4:
    st.header("Monte Carlo Forward Simulation")
    st.caption(
        "1 000 Markov-chain paths seeded from current regime probabilities. "
        "Regime transitions follow the empirical day-over-day transition matrix "
        "(Laplace-smoothed). Composite scores drawn from per-regime Gaussian distributions."
    )

    import plotly.graph_objects as _go18
    import plotly.graph_objs as _pgo18

    # Build a cache key from the top regime + its probability (changes when model updates)
    _rp18   = load_regime_probability(df)
    _cur18  = _rp18.get("current", {})
    _ck18   = f"{_cur18.get('top_regime','')}:{_cur18.get('top_prob', 0):.4f}"
    _mc18   = load_monte_carlo(df, _ck18)

    if not _mc18:
        st.warning("Monte Carlo simulation unavailable — check input data.")
    else:
        _sim18   = _mc18["simulation"]
        _reg18   = _mc18["regimes"]
        _H_list  = _mc18["horizons"]
        _last_s  = _mc18["last_actual"]
        _last_d  = pd.Timestamp(_mc18["last_date"])

        # ── Fan chart ────────────────────────────────────────────────────────
        st.subheader("Composite Score — Fan Chart")
        _h_sel = st.radio("Horizon", _H_list,
                          format_func=lambda h: f"{h}d (~{h//21}mo)",
                          horizontal=True, key="mc_horizon")

        _pct18  = _sim18["percentiles"][_h_sel]
        _fwd_dates = pd.bdate_range(_last_d + pd.Timedelta(days=1),
                                    periods=_h_sel)

        _hist_n = min(90, len(df))
        _hist_d = df["date"].iloc[-_hist_n:]
        _hist_s = df["composite_risk_score_smooth"].iloc[-_hist_n:]

        _fan = _go18.Figure()

        # Shaded risk bands
        _fan.add_hrect(y0=70, y1=100, fillcolor="rgba(214,39,40,0.06)",
                       line_width=0, annotation_text="Elevated",
                       annotation_position="top left")
        _fan.add_hrect(y0=50, y1=70, fillcolor="rgba(255,127,14,0.06)",
                       line_width=0, annotation_text="Caution",
                       annotation_position="top left")

        # P5–P95 outer band
        _fan.add_trace(_go18.Scatter(
            x=list(_fwd_dates) + list(_fwd_dates[::-1]),
            y=list(_pct18["p95"]) + list(_pct18["p5"][::-1]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.10)",
            line=dict(width=0),
            name="P5–P95",
            hoverinfo="skip",
        ))
        # P25–P75 inner band
        _fan.add_trace(_go18.Scatter(
            x=list(_fwd_dates) + list(_fwd_dates[::-1]),
            y=list(_pct18["p75"]) + list(_pct18["p25"][::-1]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.20)",
            line=dict(width=0),
            name="P25–P75",
            hoverinfo="skip",
        ))
        # Median path
        _fan.add_trace(_go18.Scatter(
            x=_fwd_dates,
            y=_pct18["p50"],
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            name="Median",
        ))
        # Historical actual
        _fan.add_trace(_go18.Scatter(
            x=_hist_d,
            y=_hist_s,
            mode="lines",
            line=dict(color="#333", width=1.5),
            name="Actual (last 90d)",
        ))
        # Connector dot at last actual
        _fan.add_trace(_go18.Scatter(
            x=[_last_d],
            y=[_last_s],
            mode="markers",
            marker=dict(color="#333", size=6),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Vertical separator
        _fan.add_vline(x=_last_d, line_dash="dot", line_color="#bbb", line_width=1)

        _fan.update_layout(
            height=380,
            yaxis=dict(range=[0, 100], title="Composite Score", showgrid=True,
                       gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(showgrid=False, title=None),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=50, r=20, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(_fan, use_container_width=True)

        # ── Summary stats ────────────────────────────────────────────────────
        st.subheader(f"Summary at {_h_sel}-Day Horizon")
        _s18 = _sim18["summary"][_h_sel]
        _sc1, _sc2, _sc3, _sc4, _sc5 = st.columns(5)
        _sc1.metric("Median Score",   f"{_s18['median']:.1f}")
        _sc2.metric("P5 (bear)",      f"{_s18['p5']:.1f}")
        _sc3.metric("P95 (bull-risk)",f"{_s18['p95']:.1f}")
        _sc4.metric("P(score > 50)",  f"{_s18['prob_above_50']:.1%}")
        _sc5.metric("P(score > 70)",  f"{_s18['prob_above_70']:.1%}",
                    help="Probability that composite score enters the elevated band")

        st.divider()

        # ── Regime distribution at horizon ──────────────────────────────────
        st.subheader(f"Regime Distribution at {_h_sel} Days")
        _rpa18 = _sim18["regime_probs_at"][_h_sel]
        _rpa_sorted = sorted(_rpa18.items(), key=lambda kv: kv[1], reverse=True)

        _COLORS18 = {
            "Risk-On":  "#27ae60",
            "Neutral":  "#95a5a6",
            "Caution":  "#e67e22",
            "Risk-Off": "#e74c3c",
        }
        _rpa_labs   = [r for r, _ in _rpa_sorted]
        _rpa_vals   = [p for _, p in _rpa_sorted]
        _rpa_colors = [_COLORS18.get(r, "#aaa") for r in _rpa_labs]

        _fig_rpa = _go18.Figure(_go18.Bar(
            y=_rpa_labs,
            x=_rpa_vals,
            orientation="h",
            marker_color=_rpa_colors,
            text=[f"{v:.1%}" for v in _rpa_vals],
            textposition="outside",
        ))
        _fig_rpa.update_layout(
            height=max(220, len(_rpa_labs) * 38),
            xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=180, r=60, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        _fig_rpa.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(_fig_rpa, use_container_width=True)

        # ── Cross-horizon summary table ──────────────────────────────────────
        with st.expander("Cross-horizon summary"):
            _rows_xh = []
            for _H in _H_list:
                _sx = _sim18["summary"][_H]
                _top_r = max(_sim18["regime_probs_at"][_H].items(),
                             key=lambda kv: kv[1])
                _rows_xh.append({
                    "Horizon": f"{_H}d (~{_H // 21}mo)",
                    "Median": f"{_sx['median']:.1f}",
                    "P5":     f"{_sx['p5']:.1f}",
                    "P95":    f"{_sx['p95']:.1f}",
                    "P(>50)": f"{_sx['prob_above_50']:.1%}",
                    "P(>70)": f"{_sx['prob_above_70']:.1%}",
                    "Top regime": f"{_top_r[0]} ({_top_r[1]:.1%})",
                })
            st.dataframe(
                pd.DataFrame(_rows_xh).set_index("Horizon"),
                use_container_width=True,
            )

with _models_sub5:
    st.header("Sub-period Attribution")
    st.caption(
        "Performance, factor, and regime statistics broken down by calendar year. "
        "Answers whether the model's edge is consistent or concentrated in a single window."
    )

    import plotly.graph_objects as _go19

    _sp19 = load_subperiod_attribution(df)
    _tbl19 = _sp19["table"]
    _roll19 = _sp19["rolling"]
    _rf19  = _sp19["regime_freq"]

    if _tbl19.empty:
        st.warning("Sub-period attribution unavailable — check input data.")
    else:
        # ── Key metrics table ────────────────────────────────────────────────
        st.subheader("Period-by-Period Metrics")

        _DISPLAY_COLS = {
            "n_obs":           "N",
            "strat_ann_ret":   "Strat Ann Ret",
            "strat_ann_vol":   "Strat Ann Vol",
            "strat_sharpe":    "Strat Sharpe",
            "strat_max_dd":    "Strat Max DD",
            "sp500_sharpe":    "SP500 Sharpe",
            "excess_ann_ret":  "Excess Ann Ret",
            "beta":            "Beta",
            "ann_alpha":       "Ann Alpha",
            "r2":              "R²",
            "mean_composite":  "Mean Composite",
        }
        _disp_cols = [c for c in _DISPLAY_COLS if c in _tbl19.columns]
        _disp = _tbl19[_disp_cols].copy().rename(columns=_DISPLAY_COLS)

        # Format for display
        _fmt = {}
        for orig, label in _DISPLAY_COLS.items():
            if "Ret" in label or "Alpha" in label or "Vol" in label or "DD" in label:
                _fmt[label] = "{:.1%}"
            elif label in ("Strat Sharpe", "SP500 Sharpe", "Beta", "R²"):
                _fmt[label] = "{:.2f}"
            elif label == "Mean Composite":
                _fmt[label] = "{:.1f}"

        def _color_sharpe(v):
            if pd.isna(v): return ""
            if v >= 1.0: return "background-color:#d4edda;color:#155724"
            if v >= 0.5: return "background-color:#fff3cd;color:#856404"
            return "background-color:#fee0d2;color:#c0392b"

        def _color_dd(v):
            if pd.isna(v): return ""
            if v >= -0.05: return "background-color:#d4edda;color:#155724"
            if v >= -0.15: return "background-color:#fff3cd;color:#856404"
            return "background-color:#fee0d2;color:#c0392b"

        def _color_alpha(v):
            if pd.isna(v): return ""
            if v > 0.02:  return "background-color:#d4edda;color:#155724"
            if v > -0.02: return "background-color:#fff3cd;color:#856404"
            return "background-color:#fee0d2;color:#c0392b"

        _styled = _disp.style
        if "Strat Sharpe" in _disp.columns:
            _styled = _styled.map(_color_sharpe, subset=["Strat Sharpe"])
        if "SP500 Sharpe" in _disp.columns:
            _styled = _styled.map(_color_sharpe, subset=["SP500 Sharpe"])
        if "Strat Max DD" in _disp.columns:
            _styled = _styled.map(_color_dd, subset=["Strat Max DD"])
        if "Ann Alpha" in _disp.columns:
            _styled = _styled.map(_color_alpha, subset=["Ann Alpha"])
        _styled = _styled.format(_fmt, na_rep="—")

        st.dataframe(_styled, use_container_width=True)

        st.divider()

        # ── Sharpe comparison bar chart ──────────────────────────────────────
        st.subheader("Sharpe Ratio by Period")
        _periods_plot = [p for p in _tbl19.index if p != "Full Period"]
        _strat_sharpes = _tbl19.loc[_periods_plot, "strat_sharpe"] if "strat_sharpe" in _tbl19.columns else pd.Series()
        _sp500_sharpes = _tbl19.loc[_periods_plot, "sp500_sharpe"] if "sp500_sharpe" in _tbl19.columns else pd.Series()

        _fig_sharpe = _go19.Figure()
        if not _strat_sharpes.empty:
            _fig_sharpe.add_trace(_go19.Bar(
                x=_periods_plot, y=_strat_sharpes.values,
                name="Strategy", marker_color="#1f77b4",
            ))
        if not _sp500_sharpes.empty:
            _fig_sharpe.add_trace(_go19.Bar(
                x=_periods_plot, y=_sp500_sharpes.values,
                name="SP500", marker_color="#aaa",
            ))
        _fig_sharpe.add_hline(y=1.0, line_dash="dot", line_color="#e74c3c",
                              annotation_text="Sharpe = 1",
                              annotation_position="bottom right")
        _fig_sharpe.update_layout(
            barmode="group", height=280,
            yaxis=dict(title="Sharpe Ratio", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(title=None),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_fig_sharpe, use_container_width=True)

        # ── Rolling 63-day Sharpe ────────────────────────────────────────────
        if not _roll19.empty and "strat_sharpe" in _roll19.columns:
            st.subheader(f"Rolling {_sp19['roll_window']}-Day Sharpe Ratio")
            _fig_roll = _go19.Figure()
            _fig_roll.add_hline(y=0, line_color="#ccc", line_width=1)
            _fig_roll.add_hline(y=1.0, line_dash="dot", line_color="#e74c3c",
                                line_width=1, annotation_text="Sharpe = 1",
                                annotation_position="bottom right")
            _fig_roll.add_trace(_go19.Scatter(
                x=_roll19.index, y=_roll19["strat_sharpe"],
                name="Strategy", mode="lines",
                line=dict(color="#1f77b4", width=1.5),
            ))
            if "sp500_sharpe" in _roll19.columns:
                _fig_roll.add_trace(_go19.Scatter(
                    x=_roll19.index, y=_roll19["sp500_sharpe"],
                    name="SP500", mode="lines",
                    line=dict(color="#aaa", width=1.5, dash="dot"),
                ))
            _fig_roll.update_layout(
                height=260,
                yaxis=dict(title="Sharpe", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                xaxis=dict(showgrid=False, title=None),
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(l=50, r=20, t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            st.plotly_chart(_fig_roll, use_container_width=True)

        # ── Rolling beta ────────────────────────────────────────────────────
        if not _roll19.empty and "beta" in _roll19.columns:
            st.subheader(f"Rolling {_sp19['roll_window']}-Day Beta")
            _fig_beta19 = _go19.Figure()
            _fig_beta19.add_hline(y=1.0, line_dash="dot", line_color="#aaa",
                                   line_width=1, annotation_text="β = 1 (buy & hold)",
                                   annotation_position="bottom right")
            _fig_beta19.add_trace(_go19.Scatter(
                x=_roll19.index, y=_roll19["beta"],
                mode="lines", name="Beta",
                line=dict(color="#ff7f0e", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,127,14,0.07)",
            ))
            _fig_beta19.update_layout(
                height=220,
                yaxis=dict(title="Beta", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                xaxis=dict(showgrid=False, title=None),
                margin=dict(l=50, r=20, t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(_fig_beta19, use_container_width=True)

        # ── Regime frequency heatmap ─────────────────────────────────────────
        if not _rf19.empty:
            st.subheader("Regime Frequency by Period")
            _heat_z   = _rf19.values * 100          # convert to %
            _heat_x   = list(_rf19.columns)
            _heat_y   = list(_rf19.index)
            _text_z   = [[f"{v:.0f}%" for v in row] for row in _heat_z]

            _fig_heat = _go19.Figure(_go19.Heatmap(
                z=_heat_z, x=_heat_x, y=_heat_y,
                text=_text_z, texttemplate="%{text}",
                colorscale="Blues", showscale=True,
                colorbar=dict(title="% time", ticksuffix="%"),
                zmin=0, zmax=60,
            ))
            _fig_heat.update_layout(
                height=max(200, len(_heat_y) * 55),
                xaxis=dict(tickangle=-30, title=None),
                yaxis=dict(title=None),
                margin=dict(l=130, r=20, t=20, b=80),
            )
            st.plotly_chart(_fig_heat, use_container_width=True)
            st.caption(
                "Each row sums to 100%. "
                "Shift in regime frequency across periods indicates changing market conditions."
            )

with _models_sub6:
    st.header("Position Sizing")
    st.caption(
        "Four complementary position sizing methods applied to the composite risk signal. "
        "Blend = mean of all four non-NaN methods. All weights clipped to [0%, 100%]."
    )

    import plotly.graph_objects as _go20

    _ps20 = load_position_sizing(df)
    _sizes20   = _ps20.get("sizes",   pd.DataFrame())
    _cur20     = _ps20.get("current", {})
    _bt20      = _ps20.get("backtest", pd.DataFrame())

    if _sizes20.empty:
        st.warning("Position sizing unavailable — check input data.")
    else:
        # ── Current weights strip ─────────────────────────────────────────────
        st.subheader("Current Recommended Weights")
        _meth_labels = {
            "score_sizing":      "Score-Based",
            "regime_prob_sizing":"Regime-Prob",
            "kelly_sizing":      "Half-Kelly",
            "vol_target_sizing": "Vol-Target",
            "blend":             "Blend",
        }
        _cur_cols = st.columns(len(_meth_labels))
        for _ci, (_key, _label) in enumerate(_meth_labels.items()):
            _val = _cur20.get(_key)
            if _val is not None:
                _pct = f"{_val:.0%}"
                if _val >= 0.75:   _delta_col = "normal"
                elif _val >= 0.40: _delta_col = "off"
                else:              _delta_col = "inverse"
                _cur_cols[_ci].metric(_label, _pct)
            else:
                _cur_cols[_ci].metric(_label, "—")

        _reg20 = _cur20.get("current_regime", "—")
        _sc20  = _cur20.get("current_composite")
        _sc_str = f"{_sc20:.1f}" if _sc20 is not None else "—"
        st.caption(f"Current regime: **{_reg20}** · Composite score: **{_sc_str}**")

        st.divider()

        # ── Sizing time series ────────────────────────────────────────────────
        st.subheader("Position Weight Over Time")
        _fig_sz = _go20.Figure()
        _sz_colors = {
            "score_sizing":       "#1f77b4",
            "regime_prob_sizing": "#ff7f0e",
            "kelly_sizing":       "#2ca02c",
            "vol_target_sizing":  "#9467bd",
            "blend":              "#d62728",
        }
        _sz_dashes = {
            "score_sizing":       "solid",
            "regime_prob_sizing": "solid",
            "kelly_sizing":       "dot",
            "vol_target_sizing":  "dash",
            "blend":              "solid",
        }
        _sz_widths = {k: (3 if k == "blend" else 1.2) for k in _sz_colors}

        for _col in _sizes20.columns:
            _label = _meth_labels.get(_col, _col)
            _fig_sz.add_trace(_go20.Scatter(
                x=_sizes20.index, y=_sizes20[_col],
                name=_label, mode="lines",
                line=dict(color=_sz_colors.get(_col, "#888"),
                          width=_sz_widths.get(_col, 1.5),
                          dash=_sz_dashes.get(_col, "solid")),
                connectgaps=False,
            ))

        _fig_sz.add_hrect(y0=0.0, y1=0.25, fillcolor="rgba(231,76,60,0.06)",
                          line_width=0, annotation_text="Underweight zone",
                          annotation_position="top left",
                          annotation_font=dict(size=10, color="#c0392b"))
        _fig_sz.add_hline(y=1.0, line_dash="dot", line_color="#aaa", line_width=1)

        _fig_sz.update_layout(
            height=320,
            yaxis=dict(title="Position Weight", tickformat=".0%",
                       range=[-0.05, 1.10], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(showgrid=False, title=None),
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(l=60, r=20, t=50, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(_fig_sz, use_container_width=True)

        st.divider()

        # ── Sized backtest comparison ─────────────────────────────────────────
        if not _bt20.empty:
            st.subheader("Sized Backtest: Cumulative Returns")
            st.caption("Each method's weight applied to strategy returns with a 1-day lag (no look-ahead).")

            _fig_bt = _go20.Figure()
            _bt_traces = {
                "cum_full":              ("Full Allocation", "#aaa",      "dot",   1.5),
                "cum_blend":             ("Blend",           "#d62728",   "solid", 2.5),
                "cum_score_sizing":      ("Score-Based",     "#1f77b4",   "solid", 1.2),
                "cum_regime_prob_sizing":("Regime-Prob",     "#ff7f0e",   "solid", 1.2),
                "cum_kelly_sizing":      ("Half-Kelly",      "#2ca02c",   "dot",   1.2),
                "cum_vol_target_sizing": ("Vol-Target",      "#9467bd",   "dash",  1.2),
                "cum_sp500":             ("SP500",           "#17becf",   "dot",   1.5),
            }
            _date_col20 = _bt20["date"] if "date" in _bt20.columns else _bt20.index

            for _k, (_lbl, _clr, _dsh, _wid) in _bt_traces.items():
                if _k in _bt20.columns:
                    _fig_bt.add_trace(_go20.Scatter(
                        x=_date_col20, y=_bt20[_k],
                        name=_lbl, mode="lines",
                        line=dict(color=_clr, width=_wid, dash=_dsh),
                    ))

            _fig_bt.add_hline(y=1.0, line_dash="dot", line_color="#ccc", line_width=1)
            _fig_bt.update_layout(
                height=340,
                yaxis=dict(title="Cumulative Return (rebased 1.0)",
                           showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                xaxis=dict(showgrid=False, title=None),
                legend=dict(orientation="h", y=1.12, x=0),
                margin=dict(l=60, r=20, t=50, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            st.plotly_chart(_fig_bt, use_container_width=True)

        st.divider()

        # ── Configuration reference ───────────────────────────────────────────
        with st.expander("Sizing Configuration", expanded=False):
            _col_bp, _col_rw = st.columns(2)

            with _col_bp:
                st.markdown("**Score → Weight Breakpoints**")
                _bp_used = _ps20.get("breakpoints", SCORE_BREAKPOINTS)
                _bp_df = pd.DataFrame(_bp_used, columns=["Composite Score", "Weight"])
                _bp_df["Weight"] = _bp_df["Weight"].map(lambda x: f"{x:.0%}")
                st.dataframe(_bp_df, use_container_width=True, hide_index=True)

            with _col_rw:
                st.markdown("**Regime Target Weights**")
                _rw_used = _ps20.get("regime_weights", REGIME_WEIGHTS)
                _rw_df = pd.DataFrame(
                    [{"Regime": k, "Target Weight": f"{v:.0%}"}
                     for k, v in sorted(_rw_used.items(), key=lambda x: -x[1])]
                )
                st.dataframe(_rw_df, use_container_width=True, hide_index=True)

            st.caption(
                f"Vol-target: {_ps20.get('target_vol', 0.10):.0%} annualised · "
                "Half-Kelly uses 63-day rolling window · "
                "Vol-target uses 21-day rolling window"
            )

with _models_sub7:
    st.header("Scenario Analysis")
    st.caption(
        "Apply market shocks to the current observation and trace their effect through the full "
        "risk scoring chain: derived features → component scores → composite → regime → sizing. "
        "Uses a delta-on-smooth approach: shocked scores are the current smoothed value ± the "
        "instantaneous raw-score change, so the existing trend is preserved."
    )

    import plotly.graph_objects as _go21

    # ── Load pre-computed preset grid and tornado (cached) ───────────────────
    _sc21_cache = load_scenario_grid(df)
    _sc21_model  = _sc21_cache.get("model", {})
    _sc21_grid   = _sc21_cache.get("grid",  pd.DataFrame())
    _sc21_tornado = _sc21_cache.get("tornado", pd.DataFrame())

    # ── Scenario selector ────────────────────────────────────────────────────
    _preset_names = list(SCENARIO_PRESETS.keys())
    _sel_preset = st.selectbox(
        "Select scenario",
        _preset_names + ["Custom"],
        index=2,           # default: Moderate Stress
        key="sc21_preset",
    )

    if _sel_preset == "Custom":
        st.markdown("**Custom shocks** (add to current levels):")
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            _vix_sh = st.slider("VIX Δ",          -15.0, +40.0, 0.0, 0.5, key="sc21_vix")
            _hy_sh  = st.slider("HY Spread Δ (pp)", -1.0,  +5.0, 0.0, 0.1, key="sc21_hy")
        with _c2:
            _sp_sh  = st.slider("SP500 Δ (%)",    -40.0,  +20.0, 0.0, 1.0, key="sc21_sp") / 100.0
            _spr_sh = st.slider("2/10 Spread Δ (pp)", -1.5, +1.0, 0.0, 0.05, key="sc21_spr")
        with _c3:
            _nfci_sh = st.slider("NFCI Δ",        -0.5,  +1.5, 0.0, 0.05, key="sc21_nfci")
            _ue_sh   = st.slider("Unemployment Δ (pp)", -0.5, +2.0, 0.0, 0.1, key="sc21_ue")
        _custom_shocks = {
            "vix_shock":          _vix_sh,
            "hy_spread_shock":    _hy_sh,
            "sp500_shock":        _sp_sh,
            "spread_shock":       _spr_sh,
            "nfci_shock":         _nfci_sh,
            "unemployment_shock": _ue_sh,
        }
        _active_shocks = _custom_shocks
    else:
        _active_shocks = SCENARIO_PRESETS[_sel_preset]

    # Run the selected scenario (interactive; not cached)
    _sc21_selected = run_scenario(df, _active_shocks, _sc21_model)
    _sc21_base  = _sc21_selected.get("baseline", {})
    _sc21_scen  = _sc21_selected.get("scenario", {})
    _sc21_delta = _sc21_selected.get("delta",    {})
    _sc21_sz    = _sc21_selected.get("sizing",   {})
    _sc21_bsz   = _sc21_selected.get("baseline_sizing", {})

    st.divider()

    # ── Side-by-side comparison ──────────────────────────────────────────────
    st.subheader("Baseline vs. Scenario")
    _cL, _cR = st.columns(2)

    def _decision_badge_sc(decision: str, env: str = "") -> str:
        _DC = {
            "Risk-On":  "#27ae60",
            "Neutral":  "#95a5a6",
            "Caution":  "#e67e22",
            "Risk-Off": "#e74c3c",
        }
        clr = _DC.get(decision, "#aaa")
        return (
            f"<div style='background:{clr};color:white;padding:8px 14px;"
            f"border-radius:6px;font-weight:600;font-size:1rem;display:inline-block'>"
            f"{decision}</div>"
            + (f"<br><small style='color:#888'>{env}</small>" if env else "")
        )

    with _cL:
        st.markdown("**Current (Baseline)**")
        st.markdown(_decision_badge_sc(
            str(_sc21_base.get("final_decision", "—")), ""
        ), unsafe_allow_html=True)
        st.markdown("")
        _bm1, _bm2, _bm3 = st.columns(3)
        _bm1.metric("Composite", f"{_sc21_base.get('composite_risk_score_smooth', 0):.1f}")
        _bm2.metric("Blend Sizing", f"{_sc21_bsz.get('blend', 0):.0%}")
        _bm3.metric("VIX", f"{_sc21_base.get('vix', 0):.1f}")
        _bm4, _bm5, _bm6 = st.columns(3)
        _bm4.metric("HY Spread", f"{_sc21_base.get('hy_spread', 0):.2f}%")
        _bm5.metric("SP500 DD", f"{_sc21_base.get('sp500_drawdown', 0):.1%}")
        _bm6.metric("NFCI", f"{_sc21_base.get('nfci', 0):.2f}")

    with _cR:
        st.markdown(f"**Scenario: {_sel_preset}**")
        st.markdown(_decision_badge_sc(
            str(_sc21_scen.get("final_decision", "—")),
            str(_sc21_scen.get("final_environment", "")),
        ), unsafe_allow_html=True)
        st.markdown("")
        _sm1, _sm2, _sm3 = st.columns(3)
        _comp_d = _sc21_delta.get("composite_risk_score_smooth", 0)
        _sm1.metric("Composite", f"{_sc21_scen.get('composite_risk_score_smooth', 0):.1f}",
                    delta=f"{_comp_d:+.1f}", delta_color="inverse")
        _sm2.metric("Blend Sizing", f"{_sc21_sz.get('blend', 0):.0%}",
                    delta=f"{(_sc21_sz.get('blend', 0) - _sc21_bsz.get('blend', 0)):+.0%}")
        _sm3.metric("VIX", f"{_sc21_scen.get('vix', 0):.1f}",
                    delta=f"{_active_shocks.get('vix_shock', 0):+.1f}", delta_color="inverse")
        _sm4, _sm5, _sm6 = st.columns(3)
        _sm4.metric("HY Spread", f"{_sc21_scen.get('hy_spread', 0):.2f}%",
                    delta=f"{_active_shocks.get('hy_spread_shock', 0):+.2f}", delta_color="inverse")
        _sm5.metric("SP500 DD", f"{_sc21_scen.get('sp500_drawdown', 0):.1%}",
                    delta=f"{(_sc21_scen.get('sp500_drawdown', 0) - _sc21_base.get('sp500_drawdown', 0)):.1%}",
                    delta_color="inverse")
        _sm6.metric("NFCI", f"{_sc21_scen.get('nfci', 0):.2f}",
                    delta=f"{_active_shocks.get('nfci_shock', 0):+.2f}", delta_color="inverse")

    st.divider()

    # ── Component score comparison chart ────────────────────────────────────
    st.subheader("Component Score Impact")
    _comp_labels = {
        "macro_risk_score_smooth":         "Macro Risk",
        "credit_market_risk_score_smooth": "Credit",
        "liquidity_regime_score_smooth":   "Liquidity",
        "treasury_stress_score_smooth":    "Treasury",
        "complacency_score_smooth":        "Complacency",
        "mean_reversion_score_smooth":     "Mean Reversion",
    }
    _comp_keys   = list(_comp_labels.keys())
    _comp_names  = [_comp_labels[k] for k in _comp_keys]
    _base_vals   = [float(_sc21_base.get(k, 0))  for k in _comp_keys]
    _scen_vals   = [float(_sc21_scen.get(k, 0))  for k in _comp_keys]

    _fig_comp = _go21.Figure()
    _fig_comp.add_trace(_go21.Bar(
        name="Baseline", x=_comp_names, y=_base_vals,
        marker_color="#aaa", opacity=0.8,
    ))
    _fig_comp.add_trace(_go21.Bar(
        name=f"Scenario ({_sel_preset})", x=_comp_names, y=_scen_vals,
        marker_color=[
            "#e74c3c" if s > b else "#2ca02c"
            for s, b in zip(_scen_vals, _base_vals)
        ],
        opacity=0.9,
    ))
    _fig_comp.update_layout(
        barmode="group", height=280,
        yaxis=dict(title="Score (0–100)", range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(title=None),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(_fig_comp, use_container_width=True)

    st.divider()

    # ── Preset grid table ────────────────────────────────────────────────────
    st.subheader("All Preset Scenarios")
    if not _sc21_grid.empty:
        _grid_display_cols = {
            "composite_risk_score_smooth": "Composite",
            "delta_composite":             "Δ Composite",
            "final_decision":              "Decision",
            "blend_sizing":                "Blend Weight",
            "vix":                         "VIX",
            "hy_spread":                   "HY Spread",
            "sp500_drawdown":              "SP500 DD",
            "shock_flag":                  "Shock Flag",
        }
        _disp_cols21 = [c for c in _grid_display_cols if c in _sc21_grid.columns]
        _disp21 = _sc21_grid[_disp_cols21].copy().rename(columns=_grid_display_cols)

        def _color_delta(v):
            if pd.isna(v): return ""
            if v >= 15:  return "background-color:#fee0d2;color:#c0392b"
            if v >= 5:   return "background-color:#fff3cd;color:#856404"
            if v <= -5:  return "background-color:#d4edda;color:#155724"
            return ""

        def _color_blend(v):
            if pd.isna(v): return ""
            if v >= 0.6: return "background-color:#d4edda;color:#155724"
            if v >= 0.3: return "background-color:#fff3cd;color:#856404"
            return "background-color:#fee0d2;color:#c0392b"

        _fmt21 = {
            "Composite":   "{:.1f}",
            "Δ Composite": "{:+.1f}",
            "Blend Weight":"{:.0%}",
            "HY Spread":   "{:.2f}",
            "SP500 DD":    "{:.1%}",
            "VIX":         "{:.1f}",
        }
        _styled21 = _disp21.style
        if "Δ Composite" in _disp21.columns:
            _styled21 = _styled21.map(_color_delta, subset=["Δ Composite"])
        if "Blend Weight" in _disp21.columns:
            _styled21 = _styled21.map(_color_blend, subset=["Blend Weight"])
        _styled21 = _styled21.format(_fmt21, na_rep="—")
        st.dataframe(_styled21, use_container_width=True)

    st.divider()

    # ── Sensitivity tornado ──────────────────────────────────────────────────
    st.subheader("Sensitivity: Single-Variable Impact on Composite Score")
    st.caption(
        "Each bar shows the change in composite risk score when one variable is shocked "
        "independently (HY +1.5pp, VIX +15, SP500 -15%, NFCI +0.3, Spread -0.5pp, Unemp +0.5)."
    )
    if not _sc21_tornado.empty:
        _tornado_colors = [
            "#e74c3c" if d > 0 else "#2ca02c"
            for d in _sc21_tornado["delta_composite"]
        ]
        _fig_tor = _go21.Figure(_go21.Bar(
            x=_sc21_tornado["delta_composite"],
            y=_sc21_tornado["variable"],
            orientation="h",
            marker_color=_tornado_colors,
            text=[f"{v:+.1f}" for v in _sc21_tornado["delta_composite"]],
            textposition="outside",
        ))
        _fig_tor.add_vline(x=0, line_color="#333", line_width=1)
        _fig_tor.update_layout(
            height=max(220, len(_sc21_tornado) * 45),
            xaxis=dict(title="Δ Composite Score", showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                       zeroline=False),
            yaxis=dict(title=None, autorange="reversed"),
            margin=dict(l=160, r=60, t=20, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(_fig_tor, use_container_width=True)
        st.caption(
            "Red bars → shock increases composite risk (bearish). "
            "Green bars → shock reduces composite (e.g., SP500 decline breaks current complacency signal)."
        )

# =============================================================================
# MODELS sub-tab 8: True OOS Splits
# =============================================================================
with _models_sub8:
    import plotly.graph_objects as _go_oos
    import numpy as _np_oos

    st.header("Out-of-Sample Period Splits")
    st.caption(_OOS_CAVEAT)
    st.warning(
        "These splits evaluate strategy performance on held-out time windows. "
        "The scoring rules were designed with awareness of the full dataset, "
        "so these are pseudo-OOS (framework robustness test, not frozen-parameter test).",
        icon="⚠️",
    )

    with st.spinner("Computing OOS splits…"):
        _oos_df = load_frozen_splits(df)

    if _oos_df.empty:
        st.info("OOS splits could not be computed — check required columns.")
    else:
        for _, _sp_row in _oos_df.iterrows():
            st.subheader(_sp_row["name"])
            st.caption(_sp_row["description"])
            _sc1, _sc2, _sc3, _sc4 = st.columns(4)
            _sc1.metric("IS Strategy Sharpe",   f"{_sp_row.get('is_strategy_sharpe', float('nan')):.2f}" if _np_oos.isfinite(float(_sp_row.get('is_strategy_sharpe', float('nan')))) else "—")
            _sc2.metric("OOS Strategy Sharpe",  f"{_sp_row.get('oos_strategy_sharpe', float('nan')):.2f}" if _np_oos.isfinite(float(_sp_row.get('oos_strategy_sharpe', float('nan')))) else "—",
                        delta=f"{_sp_row.get('sharpe_degradation', float('nan')):+.2f}" if _np_oos.isfinite(float(_sp_row.get('sharpe_degradation', float('nan')))) else None)
            _sc3.metric("OOS SP500 Sharpe",     f"{_sp_row.get('oos_sp500_sharpe', float('nan')):.2f}" if _np_oos.isfinite(float(_sp_row.get('oos_sp500_sharpe', float('nan')))) else "—")
            _sc4.metric("OOS Excess Sharpe",    f"{_sp_row.get('oos_excess_sharpe', float('nan')):+.2f}" if _np_oos.isfinite(float(_sp_row.get('oos_excess_sharpe', float('nan')))) else "—")
            with st.expander("Full metrics"):
                _sp_display = {
                    "IS Train Days":          _sp_row.get("n_train_days"),
                    "OOS Test Days":          _sp_row.get("n_test_days"),
                    "OOS Test Period":        f"{_sp_row.get('test_start','?')} → {_sp_row.get('test_end','?')}",
                    "IS Strategy Sharpe":     _sp_row.get("is_strategy_sharpe"),
                    "OOS Strategy Sharpe":    _sp_row.get("oos_strategy_sharpe"),
                    "IS Strategy MaxDD":      f"{float(_sp_row.get('is_strategy_max_drawdown', float('nan'))):.1%}" if _np_oos.isfinite(float(_sp_row.get("is_strategy_max_drawdown", float('nan')))) else "—",
                    "OOS Strategy MaxDD":     f"{float(_sp_row.get('oos_strategy_max_drawdown', float('nan'))):.1%}" if _np_oos.isfinite(float(_sp_row.get("oos_strategy_max_drawdown", float('nan')))) else "—",
                    "OOS SP500 MaxDD":        f"{float(_sp_row.get('oos_sp500_max_drawdown', float('nan'))):.1%}" if _np_oos.isfinite(float(_sp_row.get("oos_sp500_max_drawdown", float('nan')))) else "—",
                    "Sharpe Degradation":     f"{float(_sp_row.get('sharpe_degradation', float('nan'))):+.3f}" if _np_oos.isfinite(float(_sp_row.get("sharpe_degradation", float('nan')))) else "—",
                }
                st.table(pd.Series(_sp_display).rename("Value").to_frame())
            st.divider()

# =============================================================================
# ANALYTICS sub-tab 10: Regime Validity
# =============================================================================
with _analytics_sub10:
    import plotly.graph_objects as _go_rv
    import numpy as _np_rv

    st.header("Regime Validity Testing")
    st.caption(
        "Tests whether the regime framework produces genuinely distinct economic states. "
        "Uses Kruskal-Wallis (non-parametric), one-way ANOVA, and bootstrap pairwise tests "
        "on 21-day forward SP500 returns as the primary outcome. "
        "Separability score (0–100) = eta-squared × 100."
    )

    with st.spinner("Running regime validity tests…"):
        _rv = load_regime_validity(df)

    _rv_stats   = _rv.get("regime_stats", pd.DataFrame())
    _rv_kw      = _rv.get("kw_test", {})
    _rv_anova   = _rv.get("anova", {})
    _rv_pw      = _rv.get("pairwise", pd.DataFrame())
    _rv_sep     = _rv.get("separability", {})
    _rv_cons    = _rv.get("consolidations", [])

    # ── Headline separability metric ─────────────────────────────────────────
    _sep_score = _rv_sep.get("score", float("nan"))
    _sep_color = "#27ae60" if _sep_score >= 15 else "#e67e22" if _sep_score >= 7 else "#e74c3c"
    _rv_h1, _rv_h2, _rv_h3 = st.columns(3)
    _rv_h1.metric(
        "Separability Score",
        f"{_sep_score:.1f} / 100" if _np_rv.isfinite(_sep_score) else "—",
        help="eta-squared × 100: fraction of forward return variance explained by regime",
    )
    _rv_h2.metric(
        "KW H-Statistic",
        f"{_rv_kw.get('H_stat', float('nan')):.2f}" if _rv_kw and "H_stat" in _rv_kw else "—",
        delta="Significant" if _rv_kw.get("significant_05") else "Not significant",
    )
    _rv_h3.metric(
        "ANOVA F-Statistic",
        f"{_rv_anova.get('F_stat', float('nan')):.2f}" if _rv_anova and "F_stat" in _rv_anova else "—",
        delta="Significant" if _rv_anova.get("significant_heuristic") else "Below threshold",
    )

    st.caption(f"**Interpretation:** {_rv_sep.get('interpretation', '—')}")
    if _rv_kw:
        st.caption(f"**KW conclusion:** {_rv_kw.get('conclusion', '—')}")
    if _rv_anova:
        st.caption(f"**ANOVA conclusion:** {_rv_anova.get('conclusion', '—')}")

    # ── Per-regime stats table ────────────────────────────────────────────────
    st.subheader("Per-Regime Economic Summary")
    st.caption("21-day forward statistics for each regime. † = fewer than 30 observations (unreliable).")
    if not _rv_stats.empty:
        _rv_display = _rv_stats.copy()
        _warn_col = "sample_warning"
        for _col in ["fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m"]:
            if _col in _rv_display.columns:
                _rv_display[_col] = _rv_display[_col].map(lambda v: f"{v:.2%}" if _np_rv.isfinite(v) else "—")
        for _col in ["fwd_vol_1m", "fwd_maxdd_1m"]:
            if _col in _rv_display.columns:
                _rv_display[_col] = _rv_display[_col].map(lambda v: f"{v:.2%}" if _np_rv.isfinite(v) else "—")
        for _col in ["fwd_hy_chg", "fwd_vix_chg", "fwd_move_chg", "fwd_rates_chg"]:
            if _col in _rv_display.columns:
                _rv_display[_col] = _rv_display[_col].map(lambda v: f"{v:+.3f}" if _np_rv.isfinite(v) else "—")
        if _warn_col in _rv_display.columns:
            _rv_display.index = [
                f"{r} †" if _rv_stats.at[r, _warn_col] else r
                for r in _rv_stats.index
            ]
            _rv_display = _rv_display.drop(columns=[_warn_col], errors="ignore")
        st.dataframe(_rv_display, use_container_width=True)

    # ── Forward return distribution by regime ────────────────────────────────
    if not _rv_stats.empty and "fwd_ret_1m" not in _rv_stats.columns:
        pass  # raw numeric not available after formatting
    elif not _rv_stats.empty:
        st.subheader("21-Day Forward Return Distribution by Regime")
        if "date" in df.columns:
            _rv_df_aug = df.copy()
        else:
            _rv_df_aug = df.copy()
        from src.regime_validity import _add_forward_cols
        _rv_df_aug = _add_forward_cols(_rv_df_aug)

        if "_fwd_ret_21d" in _rv_df_aug.columns and "final_decision" in _rv_df_aug.columns:
            _rv_box = _go_rv.Figure()
            for _regime in sorted(_rv_df_aug["final_decision"].dropna().unique()):
                _sub_r = _rv_df_aug[_rv_df_aug["final_decision"] == _regime]["_fwd_ret_21d"].dropna()
                if len(_sub_r) >= 5:
                    _rv_box.add_trace(_go_rv.Box(
                        y=_sub_r.values,
                        name=str(_regime),
                        boxpoints="outliers",
                        marker_size=3,
                    ))
            _rv_box.add_hline(y=0, line_color="#999", line_width=1, line_dash="dot")
            _rv_box.update_layout(
                yaxis_title="21-Day Forward Return (sum of daily)",
                height=380, template="plotly_dark",
                margin=dict(l=50, r=20, t=30, b=80),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(_rv_box, use_container_width=True)

    # ── Bootstrap pairwise table ──────────────────────────────────────────────
    st.subheader("Bootstrap Pairwise Separability")
    if not _rv_pw.empty:
        _pw_display = _rv_pw[["regime_a", "regime_b", "n_a", "n_b",
                               "mean_a", "mean_b", "observed_diff",
                               "ci_low_95", "ci_high_95",
                               "p_value_two_sided", "significant_05"]].copy()
        for _c in ["mean_a", "mean_b", "observed_diff", "ci_low_95", "ci_high_95"]:
            if _c in _pw_display.columns:
                _pw_display[_c] = _pw_display[_c].map(lambda v: f"{v:.3%}" if _np_rv.isfinite(v) else "—")

        def _pw_color(v):
            if v is True or v == "True":
                return "background-color:rgba(39,174,96,0.2);color:#27ae60"
            return "background-color:rgba(231,76,60,0.15);color:#e74c3c"

        st.dataframe(
            _pw_display.style.map(_pw_color, subset=["significant_05"]),
            use_container_width=True,
        )
        _n_sig = int(_rv_pw["significant_05"].sum()) if "significant_05" in _rv_pw.columns else 0
        _n_total = len(_rv_pw)
        st.caption(
            f"{_n_sig} of {_n_total} regime pairs are statistically distinct "
            f"(bootstrap p < 0.05 on 21d forward returns). "
            + ("INCONCLUSIVE — most pairs are not separable." if _n_sig < _n_total // 2 else "")
        )

    # ── Consolidation recommendations ─────────────────────────────────────────
    if _rv_cons:
        st.subheader("Consolidation Recommendations")
        for _c in _rv_cons:
            if _c["type"] == "small_sample":
                st.warning(f"**{_c['regime']}**: {_c['reason']} → {_c['action']}")
            else:
                st.info(
                    f"**{_c['regime']}** ↔ **{_c['partner']}**: {_c['reason']} → {_c['action']}"
                )
    else:
        st.success("No consolidation recommendations — all regimes with sufficient data are statistically distinct.")

    # ── Economic ontology table ───────────────────────────────────────────────
    st.subheader("Signal Economic Ontology")
    st.caption("Maps each signal to its economic meaning, transmission mechanism, expected lead time, and known limitations.")
    _onto_df = get_ontology_df()
    _onto_display = _onto_df[["display", "in_composite", "weight", "category",
                               "meaning", "mechanism", "expected_lag",
                               "asset_impact", "sign_conv", "limitations"]].copy()
    _onto_display.columns = ["Signal", "In Composite", "Weight", "Category",
                              "Meaning", "Transmission Mechanism", "Lead Time",
                              "Asset Impact", "Sign Convention", "Limitations"]
    st.dataframe(_onto_display, use_container_width=True, height=420)

# =============================================================================
# ANALYTICS sub-tab 11: Failure Analysis
# =============================================================================
with _analytics_sub11:
    import plotly.graph_objects as _go_fa
    import numpy as _np_fa

    st.header("Model Failure Analysis")
    st.caption(
        "Identifies periods where the regime framework gave wrong signals: "
        "false positives (defensive while market rallied), false negatives (bullish while market crashed), "
        "missed named episodes, unstable rolling windows, and regime confusion periods."
    )
    st.info(
        "Failures are diagnostic, not disqualifying. Every macro risk model has failure modes. "
        "The goal is to understand when and why the model breaks down.",
        icon="ℹ️",
    )

    with st.spinner("Running failure analysis…"):
        _fa_res = load_failure_analysis(df)

    _fa_fp   = _fa_res.get("false_positives",  pd.DataFrame())
    _fa_fn   = _fa_res.get("false_negatives",  pd.DataFrame())
    _fa_ep   = _fa_res.get("missed_episodes",  pd.DataFrame())
    _fa_uw   = _fa_res.get("unstable_windows", pd.DataFrame())
    _fa_rc   = _fa_res.get("regime_confusion", pd.DataFrame())
    _fa_sum  = _fa_res.get("summary", {})

    # ── Headline counts ───────────────────────────────────────────────────────
    _fh1, _fh2, _fh3, _fh4, _fh5 = st.columns(5)
    _fh1.metric("False Positives",    _fa_sum.get("n_false_positives", "—"),   help="Model defensive during >4% rally")
    _fh2.metric("False Negatives",    _fa_sum.get("n_false_negatives", "—"),   help="Model bullish during >5% drawdown")
    _fh3.metric("Missed Episodes",    _fa_sum.get("n_missed_episodes", "—"),   help="Named stress/rally episodes mishandled")
    _fh4.metric("Unstable Windows",   _fa_sum.get("n_unstable_windows", "—"),  help="63d windows with >8% strategy lag vs SP500")
    _fh5.metric("Confusion Periods",  _fa_sum.get("n_confusion_periods", "—"), help="Windows with ≥4 regime transitions in 21 days")

    # ── Named episode scorecard ───────────────────────────────────────────────
    st.subheader("Named Episode Scorecard")
    if not _fa_ep.empty:
        def _ep_color(v):
            if v == "correctly_positioned":
                return "background-color:rgba(39,174,96,0.2);color:#27ae60"
            if v == "partial":
                return "background-color:rgba(230,126,34,0.2);color:#e67e22"
            if v in ("missed_crash", "missed_rally"):
                return "background-color:rgba(231,76,60,0.2);color:#e74c3c"
            return ""
        _ep_show = _fa_ep[["name", "type", "start", "end", "sp500_actual",
                            "mean_equity_weight", "dominant_regime",
                            "composite_mean", "model_assessment"]].copy()
        _ep_show["sp500_actual"] = _ep_show["sp500_actual"].map(lambda v: f"{v:.0%}")
        st.dataframe(
            _ep_show.style.map(_ep_color, subset=["model_assessment"]),
            use_container_width=True,
        )
        _n_correct = (_fa_ep["model_assessment"] == "correctly_positioned").sum()
        _n_missed = _fa_ep["model_assessment"].isin(["missed_crash", "missed_rally"]).sum()
        st.caption(
            f"Correctly positioned: {_n_correct}/{len(_fa_ep)} episodes. "
            f"Missed: {_n_missed}. "
            f"Partial: {len(_fa_ep) - _n_correct - _n_missed}."
        )

    # ── False positives ───────────────────────────────────────────────────────
    st.subheader("False Positives — Defensive While Market Rallied")
    st.caption("Model had low equity exposure (≤ 0.45) while SP500 gained > 4% over the next 30 days.")
    if not _fa_fp.empty:
        _fp_show = _fa_fp[["date", "equity_weight", "composite_score",
                            "fwd_return_30d", "regime", "likely_reason"]].head(20)
        _fp_show["fwd_return_30d"] = _fa_fp["fwd_return_30d"].head(20).map(lambda v: f"{v:.2%}")
        st.dataframe(_fp_show, use_container_width=True)
        with st.expander("Common false positive patterns"):
            st.markdown(
                "- **VIX spike false alarm**: VIX jumped 30%+ briefly but credit didn't confirm\n"
                "- **Rate shock overhang**: Treasury score stayed elevated after the shock resolved\n"
                "- **Late-cycle complacency signal**: Complacency score high but euphoric period extended\n"
                "- **Post-crash slow re-entry**: Model stayed defensive into early recovery phase"
            )
    else:
        st.success("No false positives found at current thresholds.")

    # ── False negatives ───────────────────────────────────────────────────────
    st.subheader("False Negatives — Bullish While Market Crashed")
    st.caption("Model had high equity exposure (≥ 0.65) while SP500 suffered > 5% drawdown over 30 days.")
    if not _fa_fn.empty:
        _fn_show = _fa_fn[["date", "equity_weight", "composite_score",
                            "outcome_30d", "regime", "likely_reason"]].head(20)
        _fn_show["outcome_30d"] = _fa_fn["outcome_30d"].head(20).map(lambda v: f"{v:.2%}")
        st.dataframe(_fn_show, use_container_width=True)
        with st.expander("Common false negative patterns"):
            st.markdown(
                "- **Sudden shock**: Regime framework missed fast-onset crashes (COVID, flash crashes)\n"
                "- **Benign composite, bad outcome**: All signals looked fine until they didn't\n"
                "- **Signal lag**: Shock resolved faster than smoothing windows could respond\n"
                "- **Idiosyncratic events**: External political/geopolitical events not captured by any signal"
            )
    else:
        st.success("No false negatives found at current thresholds.")

    # ── Unstable windows ──────────────────────────────────────────────────────
    st.subheader("Unstable Windows — Strategy Lagging SP500 by > 8% (63-Day Rolling)")
    if not _fa_uw.empty:
        _uw_show = _fa_uw[["start", "end", "strategy_cum", "sp500_cum", "lag", "regime"]].head(20).copy()
        for _c in ["strategy_cum", "sp500_cum", "lag"]:
            if _c in _uw_show.columns:
                _uw_show[_c] = _uw_show[_c].map(lambda v: f"{v:.2%}")
        st.dataframe(_uw_show, use_container_width=True)
    else:
        st.success("No persistent underperformance windows found.")

    # ── Regime confusion ──────────────────────────────────────────────────────
    st.subheader("Regime Confusion Periods — ≥ 4 Transitions in 21 Days")
    if not _fa_rc.empty:
        st.dataframe(_fa_rc.head(20), use_container_width=True)
        st.caption(
            "High regime switching frequency often coincides with market inflection points "
            "or model indecision at composite score boundaries. "
            "Consider smoothing the transition signal further, or widening decision thresholds."
        )
    else:
        st.success("No regime confusion periods found.")

    # ── Cross-asset confirmation ──────────────────────────────────────────────
    st.subheader("Cross-Asset Confirmation — Current State")
    _conf = get_current_confirmation(df)
    if _conf:
        _conf_status = _conf.get("status", "Unknown")
        _conf_color  = "#27ae60" if "Confirmed" == _conf_status else "#e67e22" if "Partial" in _conf_status else "#e74c3c"
        st.markdown(
            f"<div style='font-size:1.4rem;font-weight:700;color:{_conf_color};margin-bottom:8px'>"
            f"Current Confirmation: {_conf_status}</div>",
            unsafe_allow_html=True,
        )
        _domain_results = _conf.get("domain_results", {})
        _dom_cols = st.columns(len(_domain_results))
        for _dc, (_domain, _dr) in zip(_dom_cols, _domain_results.items()):
            _dcolor = "#27ae60" if _dr["confirming"] else "#6b7280"
            _dc.markdown(
                f"<div style='text-align:center;padding:8px;border-radius:6px;"
                f"background:rgba(255,255,255,0.05)'>"
                f"<div style='font-size:0.75rem;color:#9ca3af'>{DOMAIN_LABELS.get(_domain, _domain)}</div>"
                f"<div style='font-size:1.1rem;font-weight:600;color:{_dcolor}'>"
                f"{'✓' if _dr['confirming'] else '✗'}</div></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            f"Confirming domains: {_conf.get('n_confirming', 0)} / {_conf.get('n_domains', 0)}. "
            "Confirmed = 4–5 domains; Partially Confirmed = 2–3; Unconfirmed = 0–1."
        )
    else:
        st.info("Confirmation engine requires at least credit and rates signals.")


# =============================================================================
# ANALYTICS sub-tab 12: Stress Contagion
# =============================================================================
with _analytics_sub12:
    import plotly.graph_objects as _go_cg
    st.header("Stress Contagion")
    st.caption(
        "Rolling 90-day Pearson correlations between sub-scores. "
        "High off-diagonal correlation = stress is spreading across asset classes. "
        "The contagion index (mean absolute off-diagonal) spikes during systemic crises."
    )
    with st.spinner("Computing contagion analysis…"):
        _cg_res = load_contagion(df)

    _cg_matrix = _cg_res.get("matrix", pd.DataFrame())
    _cg_index  = _cg_res.get("index",  pd.Series(dtype=float))
    _cg_cur    = _cg_res.get("current_index", None)

    if _cg_cur is not None:
        _cg_color = "#e74c3c" if _cg_cur > 0.6 else "#e67e22" if _cg_cur > 0.4 else "#27ae60"
        st.metric("Current Contagion Index", f"{_cg_cur:.3f}",
                  help="Mean absolute off-diagonal correlation across all sub-score pairs (90d window). >0.6 = high contagion.")

    _cg_c1, _cg_c2 = st.columns(2)
    with _cg_c1:
        st.subheader("Current Correlation Matrix")
        if not _cg_matrix.empty:
            _cg_heat = _go_cg.Figure(_go_cg.Heatmap(
                z=_cg_matrix.values.tolist(),
                x=_cg_matrix.columns.tolist(),
                y=_cg_matrix.index.tolist(),
                colorscale="RdYlGn_r",
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in _cg_matrix.values],
                texttemplate="%{text}",
                hovertemplate="%{y} × %{x}<br>Corr: %{z:.3f}<extra></extra>",
            ))
            _cg_heat.update_layout(
                height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"), margin=dict(l=8,r=8,t=8,b=8),
                xaxis=dict(tickangle=-30, color="#6b7280"),
                yaxis=dict(color="#6b7280"),
            )
            st.plotly_chart(_cg_heat, use_container_width=True)
        else:
            st.info("Not enough data for correlation matrix.")

    with _cg_c2:
        st.subheader("Contagion Index Over Time")
        if not _cg_index.empty:
            _cg_dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.RangeIndex(len(_cg_index))
            _cg_idx_s = _cg_index.values if hasattr(_cg_index, "values") else _cg_index
            _cg_line = _go_cg.Figure(_go_cg.Scatter(
                x=list(_cg_dates)[-len(_cg_idx_s):],
                y=list(_cg_idx_s),
                line=dict(color="#e67e22", width=1.8),
                fill="tozeroy", fillcolor="rgba(230,126,34,0.08)",
                hovertemplate="Date: %{x}<br>Index: %{y:.3f}<extra></extra>",
            ))
            _cg_line.add_hline(y=0.6, line_color="#e74c3c", line_dash="dot",
                               line_width=1, annotation_text="High (0.6)",
                               annotation_font=dict(color="#e74c3c", size=10))
            _cg_line.add_hline(y=0.4, line_color="#e67e22", line_dash="dot",
                               line_width=1, annotation_text="Elevated (0.4)",
                               annotation_font=dict(color="#e67e22", size=10))
            _cg_line.update_layout(
                height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"), margin=dict(l=8,r=8,t=8,b=8),
                xaxis=dict(showgrid=False, color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Mean |Corr|", range=[0,1]),
                hoverlabel=dict(bgcolor="#1a1f2e", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_cg_line, use_container_width=True)
        else:
            st.info("Not enough data for contagion index.")


# =============================================================================
# ANALYTICS sub-tab 13: Historical Analogs
# =============================================================================
with _analytics_sub13:
    import plotly.graph_objects as _go_ha
    st.header("Historical Analogs")
    st.caption(
        "Finds the 5 closest historical dates to current conditions via cosine similarity "
        "across all 7 composite sub-scores. Shows what happened to HY spreads and SP500 "
        "over the next 30 and 60 days from each analog."
    )
    with st.spinner("Finding historical analogs…"):
        _ha_df = load_historical_analogs(df)
        _ha_summary = get_analog_summary(_ha_df) if not _ha_df.empty else {}

    if _ha_df.empty:
        st.info("Not enough historical data to find analogs.")
    else:
        # Summary metrics
        _ha_c1, _ha_c2, _ha_c3, _ha_c4 = st.columns(4)
        _ha_c1.metric("Avg SP500 30D Fwd",
                      f"{_ha_summary.get('sp500_fwd_30d',{}).get('mean','—'):.1f}%" if _ha_summary.get('sp500_fwd_30d') else "—",
                      help="Mean SP500 30-day forward return across top 5 analogs")
        _ha_c2.metric("Avg SP500 60D Fwd",
                      f"{_ha_summary.get('sp500_fwd_60d',{}).get('mean','—'):.1f}%" if _ha_summary.get('sp500_fwd_60d') else "—")
        _ha_c3.metric("Avg HY Spread 30D Δ",
                      f"{_ha_summary.get('hy_fwd_30d',{}).get('mean','—'):.2f}pp" if _ha_summary.get('hy_fwd_30d') else "—",
                      help="Mean HY spread change (pp) over 30 days. Positive = widening.")
        _ha_c4.metric("Avg HY Spread 60D Δ",
                      f"{_ha_summary.get('hy_fwd_60d',{}).get('mean','—'):.2f}pp" if _ha_summary.get('hy_fwd_60d') else "—")

        # Analog table
        st.subheader("Top 5 Closest Historical Dates")
        _ha_display = _ha_df.copy()
        _ha_display["date"] = _ha_display["date"].astype(str).str[:10]
        _ha_display["similarity"] = _ha_display["similarity"].map(lambda v: f"{v:.3f}")
        _ha_display["composite_risk_score_smooth"] = _ha_display["composite_risk_score_smooth"].map(lambda v: f"{v:.1f}")
        for _hc in ["sp500_fwd_30d","sp500_fwd_60d"]:
            if _hc in _ha_display.columns:
                _ha_display[_hc] = _ha_display[_hc].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        for _hc in ["hy_fwd_30d","hy_fwd_60d"]:
            if _hc in _ha_display.columns:
                _ha_display[_hc] = _ha_display[_hc].map(lambda v: f"{v:+.2f}pp" if pd.notna(v) else "—")
        _ha_display.columns = ["Date","Regime","Similarity","Comp Score",
                               "SP500 30D","SP500 60D","HY Δ 30D","HY Δ 60D"]
        st.dataframe(_ha_display, use_container_width=True)

        # Bar chart of forward outcomes
        _ha_bar = _go_ha.Figure()
        _ha_bar.add_trace(_go_ha.Bar(
            name="SP500 30D %", x=_ha_df["date"].astype(str).str[:10],
            y=_ha_df["sp500_fwd_30d"], marker_color="#4f8ef7",
            hovertemplate="%{x}<br>SP500 30D: %{y:+.2f}%<extra></extra>",
        ))
        _ha_bar.add_trace(_go_ha.Bar(
            name="HY Δ 30D (pp)", x=_ha_df["date"].astype(str).str[:10],
            y=_ha_df["hy_fwd_30d"], marker_color="#e74c3c",
            hovertemplate="%{x}<br>HY 30D Δ: %{y:+.2f}pp<extra></extra>",
        ))
        _ha_bar.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        _ha_bar.update_layout(
            barmode="group", height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), margin=dict(l=8,r=8,t=8,b=60),
            xaxis=dict(showgrid=False, color="#6b7280", tickangle=-20),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor="#1a1f2e", font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(_ha_bar, use_container_width=True)
        st.caption(
            "Analogs exclude the last 252 trading days to avoid near-identical recent dates. "
            "Similarity = cosine similarity of normalized 7-signal vector. "
            "Forward outcomes show what actually happened from each analog date."
        )


# =============================================================================
# ANALYTICS sub-tab 14: Regime Persistence
# =============================================================================
with _analytics_sub14:
    import plotly.graph_objects as _go_rp
    st.header("Regime Persistence")
    st.caption(
        "Empirical survival analysis of regime dwell times. "
        "Given the current regime has lasted N days, estimates the probability it continues for another 10/20/30/60 days."
    )
    with st.spinner("Computing regime persistence…"):
        _rp_res = load_persistence(df)

    _rp_cur     = _rp_res.get("current_regime", "—")
    _rp_days    = _rp_res.get("days_in_regime", 0)
    _rp_start   = _rp_res.get("regime_start_date", "—")
    _rp_surv    = _rp_res.get("survival_probs", {})
    _rp_med     = _rp_res.get("median_dwell", None)
    _rp_mean    = _rp_res.get("mean_dwell", None)
    _rp_curves  = _rp_res.get("survival_curves", {})

    _rp_h1, _rp_h2, _rp_h3, _rp_h4, _rp_h5, _rp_h6 = st.columns(6)
    _rp_h1.metric("Current Regime", _rp_cur)
    _rp_h2.metric("Days Active", _rp_days, help=f"Since {_rp_start}")
    _rp_h3.metric("P(+10d)", f"{_rp_surv.get(10,0):.0%}" if _rp_surv.get(10) is not None else "—")
    _rp_h4.metric("P(+20d)", f"{_rp_surv.get(20,0):.0%}" if _rp_surv.get(20) is not None else "—")
    _rp_h5.metric("P(+30d)", f"{_rp_surv.get(30,0):.0%}" if _rp_surv.get(30) is not None else "—")
    _rp_h6.metric("P(+60d)", f"{_rp_surv.get(60,0):.0%}" if _rp_surv.get(60) is not None else "—")
    if _rp_med:
        st.caption(f"Historical median dwell for **{_rp_cur}**: {_rp_med:.0f} days · mean: {_rp_mean:.0f} days")

    # Survival curves for all regimes
    if _rp_curves:
        st.subheader("Survival Curves by Regime")
        _rp_fig = _go_rp.Figure()
        _rp_colors = {"Risk-On":"#27ae60","Neutral":"#4f8ef7","Caution":"#e67e22","Risk-Off":"#e74c3c"}
        for _reg, _curve in _rp_curves.items():
            if hasattr(_curve, "__len__") and len(_curve) > 0:
                _rp_fig.add_trace(_go_rp.Scatter(
                    x=list(range(len(_curve))),
                    y=list(_curve) if not hasattr(_curve, "values") else list(_curve.values),
                    name=_reg,
                    line=dict(color=_rp_colors.get(_reg,"#9aa0aa"), width=2,
                              dash="solid" if _reg == _rp_cur else "dot"),
                    hovertemplate=f"{_reg}<br>Day %{{x}}: %{{y:.0%}} survive<extra></extra>",
                ))
        if _rp_days > 0:
            _rp_fig.add_vline(x=_rp_days, line_color="rgba(255,255,255,0.3)",
                              line_dash="dash", line_width=1.5,
                              annotation_text=f"Now ({_rp_days}d)",
                              annotation_font=dict(color="#9aa0aa", size=10))
        _rp_fig.update_layout(
            height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), margin=dict(l=8,r=8,t=8,b=8),
            xaxis=dict(showgrid=False, color="#6b7280", title="Days Since Regime Start"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                       color="#6b7280", title="P(regime continues)", tickformat=".0%"),
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            hoverlabel=dict(bgcolor="#1a1f2e", font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(_rp_fig, use_container_width=True)
        st.caption("Solid line = current regime · dotted = others · vertical dashed = today.")

    # Dwell time table
    _rp_dwell = _rp_res.get("dwell_times", pd.DataFrame())
    if not _rp_dwell.empty:
        with st.expander("Historical dwell-time log"):
            st.dataframe(_rp_dwell.tail(50), use_container_width=True)

# =============================================================================
# ANALYTICS sub-tab 15: Merton Distance-to-Default
# =============================================================================
with _analytics_sub15:
    import plotly.graph_objects as _go_mrt
    st.header("Merton Distance-to-Default")
    st.markdown(
        """
        **The Merton (1974) structural model** treats a company's equity as a **call option on its assets**.
        If asset value falls below the face value of debt, the firm defaults (option expires worthless).

        - **Distance-to-Default (DD)** = how many standard deviations of asset value separate the firm from insolvency.
        - **DD > 3**: Low default risk (investment grade).  **DD 1–3**: Elevated. **DD < 1**: Distress territory.
        - Here, DD is estimated *in aggregate* using VIX as a proxy for equity volatility (σ_E)
          and HY spreads to imply leverage (L). This is a macro-level approximation, not a single-issuer model.

        **Formula**: DD = (ln(1/L) − 0.5σ_V²) / σ_V, where σ_V = σ_E × (1 − L)
        """
    )
    try:
        _mrt = load_merton(df)
        if _mrt.get("available"):
            _mc = _mrt["current"]
            _m1, _m2, _m3, _m4 = st.columns(4)
            _m1.metric("Distance-to-Default", f"{_mc.get('dd', float('nan')):.2f}σ",
                       help="Standard deviations from insolvency. Higher = safer.")
            _m2.metric("Default Prob (1y)", f"{_mc.get('default_prob_1y', 0):.2%}",
                       delta_color="inverse", delta=f"{_mc.get('default_prob_1y', 0):.2%}")
            _m3.metric("Implied Leverage", f"{_mc.get('leverage', 0):.1%}",
                       help="D/V ratio implied by HY spread level")
            _m4.metric("Merton Regime", _mc.get("merton_regime", "—"))

            _mrt_df = _mrt.get("df", pd.DataFrame())
            if not _mrt_df.empty and "dd" in _mrt_df.columns:
                _mrt_df = _mrt_df.copy()
                _mrt_df["date"] = pd.to_datetime(_mrt_df["date"])
                _mrt_fig = _go_mrt.Figure()
                _mrt_fig.add_trace(_go_mrt.Scatter(
                    x=_mrt_df["date"], y=_mrt_df["dd"],
                    name="Distance-to-Default", line=dict(color="#4f8ef7", width=2),
                    fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}σ<extra></extra>",
                ))
                _mrt_fig.add_hline(y=3, line=dict(color="#27ae60", dash="dash", width=1),
                                   annotation_text="DD=3 (IG threshold)", annotation_position="top left",
                                   annotation_font=dict(color="#27ae60", size=10))
                _mrt_fig.add_hline(y=1, line=dict(color="#e74c3c", dash="dash", width=1),
                                   annotation_text="DD=1 (Distress)", annotation_position="top left",
                                   annotation_font=dict(color="#e74c3c", size=10))
                _mrt_fig.update_layout(
                    height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Distance-to-Default (σ)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_mrt_fig, use_container_width=True)

            _corr = _mrt.get("correlation_with_composite")
            if _corr is not None:
                st.caption(f"Pearson correlation of DD with composite risk score: **{_corr:.3f}** "
                           "(negative expected — high DD = low risk = low score)")

            _hist_low = _mrt.get("historical_low_dd", pd.DataFrame())
            if not _hist_low.empty:
                with st.expander(f"Periods with DD < 1 (distress episodes, {len(_hist_low)} rows)"):
                    st.dataframe(_hist_low[["date", "dd", "default_prob_1y", "merton_regime"]].head(50),
                                 use_container_width=True, hide_index=True)
        else:
            st.info("Merton model unavailable — requires SP500, VIX, and HY spread in dataset.")
    except Exception as _mrt_e:
        st.caption(f"Merton model unavailable: {_mrt_e}")


# =============================================================================
# ANALYTICS sub-tab 16: Efficient Frontier
# =============================================================================
with _analytics_sub16:
    import plotly.graph_objects as _go_ef
    st.header("Mean-Variance Efficient Frontier")
    st.markdown(
        """
        **Modern Portfolio Theory (Markowitz 1952)**: given a set of assets, the *efficient frontier*
        traces portfolios with the **maximum return for each level of risk**.

        Assets: HY credit, IG credit, Cash (3% p.a. proxy). Portfolios are long-only (weights ≥ 0, sum to 1).
        Monte Carlo simulation draws 5,000 random weight combinations and plots each portfolio's
        (volatility, return) pair. The upper-left envelope is the **efficient frontier**.

        - **Min-Variance portfolio**: lowest possible volatility (safest).
        - **Max-Sharpe portfolio**: best risk-adjusted return (tangency portfolio).
        - Any portfolio *below* the frontier is suboptimal — you could get more return for the same risk.
        """
    )
    try:
        _ef = load_efficient_frontier(df)
        if _ef.get("available"):
            _efm = _ef.get("min_var", {})
            _efs = _ef.get("max_sharpe", {})
            _efc = _ef.get("current_allocation", {})

            _ef1, _ef2, _ef3 = st.columns(3)
            with _ef1:
                st.markdown("**Min-Variance Portfolio**")
                st.caption(f"HY: {_efm.get('hy_weight',0):.0%} · IG: {_efm.get('ig_weight',0):.0%} · Cash: {_efm.get('cash_weight',0):.0%}")
                st.caption(f"Return: {_efm.get('annual_return',0):.1%} · Vol: {_efm.get('annual_vol',0):.1%} · Sharpe: {_efm.get('sharpe',0):.2f}")
            with _ef2:
                st.markdown("**Max-Sharpe Portfolio**")
                st.caption(f"HY: {_efs.get('hy_weight',0):.0%} · IG: {_efs.get('ig_weight',0):.0%} · Cash: {_efs.get('cash_weight',0):.0%}")
                st.caption(f"Return: {_efs.get('annual_return',0):.1%} · Vol: {_efs.get('annual_vol',0):.1%} · Sharpe: {_efs.get('sharpe',0):.2f}")
            with _ef3:
                st.markdown("**Current Allocation**")
                st.caption(f"HY: {_efc.get('hy_weight',0):.0%} · IG: {_efc.get('ig_weight',0):.0%} · Cash: {_efc.get('cash_weight',0):.0%}")
                st.caption(f"Return: {_efc.get('expected_return', _efc.get('annual_return',0)):.1%} · Vol: {_efc.get('annual_vol',0):.1%} · Sharpe: {_efc.get('sharpe',0):.2f}")

            _sim = _ef.get("simulated", pd.DataFrame())
            if not _sim.empty:
                _ef_fig = _go_ef.Figure()
                _ef_fig.add_trace(_go_ef.Scatter(
                    x=_sim["vol"] * 100, y=_sim["ret"] * 100,
                    mode="markers", name="Simulated portfolios",
                    marker=dict(color=_sim["sharpe"], colorscale="Viridis", size=3,
                                opacity=0.5, showscale=True,
                                colorbar=dict(title="Sharpe", len=0.6)),
                    hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
                ))
                for _pt, _lbl, _clr, _sym in [
                    (_efm, "Min-Variance", "#27ae60", "diamond"),
                    (_efs, "Max-Sharpe",   "#f39c12", "star"),
                    (_efc, "Current",      "#4f8ef7", "circle"),
                ]:
                    _pt_vol = _pt.get("annual_vol")
                    _pt_ret = _pt.get("annual_return") or _pt.get("expected_return")
                    if _pt_vol is not None and _pt_ret is not None:
                        _ef_fig.add_trace(_go_ef.Scatter(
                            x=[_pt_vol * 100], y=[_pt_ret * 100],
                            mode="markers+text", name=_lbl,
                            text=[_lbl], textposition="top center",
                            marker=dict(color=_clr, size=12, symbol=_sym,
                                        line=dict(color="white", width=1.5)),
                            hovertemplate=f"{_lbl}<br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<extra></extra>"
                        ))
                _ef_fig.update_layout(
                    height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Annual Volatility (%)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Annual Return (%)"),
                    legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_ef_fig, use_container_width=True)
                st.caption("Each dot = 1 random portfolio. Color = Sharpe ratio. Upper-left = efficient frontier.")
        else:
            st.info("Efficient frontier unavailable — requires HY or IG total return data (hy_total_return_daily / ig_total_return_daily).")
    except Exception as _ef_e:
        st.caption(f"Efficient frontier unavailable: {_ef_e}")


# =============================================================================
# ANALYTICS sub-tab 17: Kelly Criterion
# =============================================================================
with _analytics_sub17:
    st.header("Kelly Criterion — Optimal Position Sizing")
    st.markdown(
        """
        **The Kelly Criterion** (Kelly 1956) gives the mathematically optimal bet size that maximises
        the long-run geometric growth rate of wealth.

        - **Discrete Kelly**: f* = (p × b − q) / b, where p = win probability, b = avg win/avg loss odds, q = 1 − p.
        - **Continuous Kelly**: f* = μ / σ², where μ = mean return, σ = return standard deviation.
        - In practice: **half-Kelly** is standard because full-Kelly has enormous variance and a single bad run
          can wipe out gains. **Quarter-Kelly** is the conservative institutional choice.

        A position *above* full-Kelly destroys geometric growth rate even if it wins on average.
        """
    )
    try:
        _kl = load_kelly(df)
        if _kl.get("available"):
            _klf = _kl.get("full", {})
            _klo = _kl.get("oos", {})
            _klr = _kl.get("regime_kelly", pd.DataFrame())
            _klw = _kl.get("current_weight")
            _klass = _kl.get("kelly_vs_current", "—")

            _k1, _k2, _k3, _k4 = st.columns(4)
            _k1.metric("Full Kelly (OOS)", f"{_klo.get('kelly_fraction', float('nan')):.1%}",
                       help="Optimal equity weight from OOS performance data")
            _k2.metric("Half-Kelly (OOS)", f"{_klo.get('half_kelly', float('nan')):.1%}",
                       help="Practical sizing: half of full Kelly")
            _k3.metric("Current Strategy Weight", f"{_klw:.1%}" if _klw is not None else "—",
                       help="Actual strategy equity weight today")
            _k4.metric("Sizing Assessment", _klass)

            _kl_cols = st.columns(2)
            with _kl_cols[0]:
                st.markdown("**Full History Kelly**")
                st.caption(
                    f"Win rate: {_klf.get('win_rate',0):.1%} · "
                    f"Avg win: {_klf.get('avg_win',0):.2%} · "
                    f"Avg loss: {_klf.get('avg_loss',0):.2%}"
                )
                st.caption(
                    f"Discrete Kelly: {_klf.get('kelly_fraction',0):.1%} · "
                    f"Continuous Kelly: {_klf.get('continuous_kelly',0):.1%}"
                )
            with _kl_cols[1]:
                st.markdown("**OOS Kelly (post-2016)**")
                st.caption(
                    f"Win rate: {_klo.get('win_rate',0):.1%} · "
                    f"Avg win: {_klo.get('avg_win',0):.2%} · "
                    f"Avg loss: {_klo.get('avg_loss',0):.2%}"
                )
                st.caption(
                    f"Discrete Kelly: {_klo.get('kelly_fraction',0):.1%} · "
                    f"Continuous Kelly: {_klo.get('continuous_kelly',0):.1%}"
                )

            if not _klr.empty:
                st.markdown("**Per-Regime Kelly Sizing**")
                st.caption("Optimal position size within each regime based on empirical win/loss stats.")
                _klr_fmt = {}
                for _c in ["win_rate", "avg_win", "avg_loss", "kelly_fraction", "half_kelly"]:
                    if _c in _klr.columns:
                        _klr_fmt[_c] = "{:.1%}"
                for _c in ["odds_ratio", "continuous_kelly"]:
                    if _c in _klr.columns:
                        _klr_fmt[_c] = "{:.2f}"
                st.dataframe(
                    _klr.style.format(_klr_fmt, na_rep="—"),
                    use_container_width=True,
                )
        else:
            st.info("Kelly analysis unavailable — requires strategy_daily_return in backtest data.")
    except Exception as _kl_e:
        st.caption(f"Kelly analysis unavailable: {_kl_e}")


# =============================================================================
# ANALYTICS sub-tab 18: Granger Causality
# =============================================================================
with _analytics_sub18:
    st.header("Granger Causality — Lead/Lag Relationships")
    st.markdown(
        """
        **Granger causality** (Granger 1969) tests whether knowing the history of series X
        improves the forecast of series Y beyond Y's own history.

        It is *not* true causality — it is **predictive precedence**. If HY spreads Granger-cause
        VIX at 5-day lags, it means past spread moves help predict future VIX moves.

        Test statistic: F-test comparing restricted VAR (Y on own lags) vs unrestricted (Y on own + X lags).
        A significant F-statistic (p < 0.05) rejects the null that X does *not* help predict Y.

        Tested pairs: HY Spread, VIX, Composite Risk Score, SP500 Return — all bidirectional at 1, 5, 21 day lags.
        """
    )
    try:
        _gr = load_granger(df)
        if _gr.get("available"):
            _gr_res = _gr.get("results", pd.DataFrame())
            _gr_sum = _gr.get("summary", "")

            if _gr_sum:
                st.info(_gr_sum)

            if not _gr_res.empty:
                _gr_disp = _gr_res.copy()
                _gr_fmt = {}
                if "f_stat" in _gr_disp.columns:
                    _gr_fmt["f_stat"] = "{:.2f}"
                if "p_value" in _gr_disp.columns:
                    _gr_fmt["p_value"] = "{:.4f}"

                def _gr_sig_color(val):
                    if isinstance(val, float) and val < 0.05:
                        return "background-color: rgba(39,174,96,0.15); color: #27ae60"
                    return ""

                _styled = _gr_disp.style.format(_gr_fmt, na_rep="—")
                if "p_value" in _gr_disp.columns:
                    _styled = _styled.applymap(_gr_sig_color, subset=["p_value"])
                st.dataframe(_styled, use_container_width=True, hide_index=True)
                st.caption(
                    "Green = p < 0.05 (significant at 5% level). "
                    "Lags tested: 1d (daily), 5d (weekly), 21d (monthly). "
                    "Data is first-differenced for stationarity (except SP500 return which is already stationary)."
                )
        else:
            st.info("Granger causality unavailable — requires HY spread, VIX, and composite score data.")
    except Exception as _gr_e:
        st.caption(f"Granger analysis unavailable: {_gr_e}")


# =============================================================================
# ANALYTICS sub-tab 19: Default Rate Forecasting
# =============================================================================
with _analytics_sub19:
    import plotly.graph_objects as _go_df
    st.header("Implied Default Rate — Jarrow-Turnbull Model")
    st.markdown(
        """
        **The Jarrow-Turnbull (1995) model** extracts the *market-implied* default rate from credit spreads.

        The core insight: a credit spread over Treasuries compensates investors for **expected losses**.
        If recovery rate R is known, the implied default intensity (hazard rate) λ is:

        **λ = Spread / (1 − R)**  where  LGD = 1 − R  (Loss Given Default)

        - **Recovery rate (R)**: historically ~40% for senior unsecured HY bonds (Moody's LossStats).
        - **LGD** = 60% = the fraction of face value lost in default.
        - A 500bp HY spread implies: λ = 5% / 0.60 = **8.3% annual default probability**.
        - The **break-even spread** is what investors need to exactly cover expected credit losses.

        This is a simplified single-name approximation applied to the aggregate HY index.
        """
    )
    try:
        _dfa = load_default_analysis(df)
        if _dfa.get("available"):
            _dfc = _dfa["current"]
            _df1, _df2, _df3, _df4 = st.columns(4)
            _df1.metric("HY Spread", f"{_dfc.get('hy_spread', float('nan')):.2f}%")
            _df2.metric("Implied Default Rate", f"{_dfc.get('implied_default_rate', 0):.2%}",
                        help="λ = spread / LGD. Market-implied annual default probability.")
            _df3.metric("Excess Spread", f"{_dfc.get('excess_spread', 0):+.2f}pp",
                        help="HY spread above break-even. Positive = investor compensated; negative = underpriced.")
            _df4.metric("Default Regime", _dfc.get("default_regime", "—"))

            if "df" in _dfa and "implied_default_rate" in _dfa["df"].columns:
                _dfa_df = _dfa["df"].copy()
                _dfa_df["date"] = pd.to_datetime(_dfa_df["date"])
                _df_fig = _go_df.Figure()
                _df_fig.add_trace(_go_df.Scatter(
                    x=_dfa_df["date"], y=_dfa_df["implied_default_rate"] * 100,
                    name="Implied Default Rate (%)", line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Implied Default Rate: %{y:.2f}%<extra></extra>",
                ))
                _df_fig.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Implied Default Rate (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_df_fig, use_container_width=True)

            _spread_tbl = _dfa.get("spread_to_default_table", pd.DataFrame())
            if not _spread_tbl.empty:
                with st.expander("Spread → Implied Default Rate reference table"):
                    _tbl_fmt = {}
                    if "implied_default_rate" in _spread_tbl.columns:
                        _tbl_fmt["implied_default_rate"] = "{:.2%}"
                    if "break_even_spread" in _spread_tbl.columns:
                        _tbl_fmt["break_even_spread"] = "{:.2f}"
                    st.dataframe(
                        _spread_tbl.style.format(_tbl_fmt, na_rep="—"),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(f"Assumes Recovery Rate = {_dfa.get('recovery_rate', 0.40):.0%} · LGD = {_dfa.get('lgd', 0.60):.0%}")
        else:
            st.info("Default rate model unavailable — requires HY spread data.")
    except Exception as _dfa_e:
        st.caption(f"Default analysis unavailable: {_dfa_e}")


# =============================================================================
# ANALYTICS sub-tab 20: Regime-Conditioned Forward Simulation
# =============================================================================
with _analytics_sub20:
    import plotly.graph_objects as _go_fs
    st.header("Regime-Conditioned Forward Simulation")
    st.markdown(
        """
        **Monte Carlo fan chart** of portfolio paths starting from today's regime.

        Rather than a single expected return, 500 paths are simulated by:
        1. Drawing regime sequences as a **Markov chain** using the historical transition matrix
        2. Drawing daily returns from each regime's **empirical return distribution** (with skew correction)
        3. Accumulating portfolio value and reporting percentile bands

        The fan chart reveals the full distribution of outcomes — Risk-Off regimes produce
        a wide, left-skewed distribution; Risk-On regimes cluster tightly upward.
        The 5th percentile is the "bad scenario" for risk management purposes.
        """
    )
    try:
        _fs = load_forward_simulation(df)
        if _fs.get("available"):
            _fs_cur = _fs.get("current_regime", "—")
            _fs_hs  = _fs.get("horizon_summary", {})
            _fs_reg = _fs.get("regime_profiles", {})

            _fsc1, _fsc2, _fsc3, _fsc4 = st.columns(4)
            _fsc1.metric("Current Regime", _fs_cur)
            _fsc2.metric("21d Median Return", f"{_fs_hs.get(21,{}).get(50,float('nan')):.2%}" if _fs_hs.get(21) else "—")
            _fsc3.metric("21d 5th Pctile", f"{_fs_hs.get(21,{}).get(5,float('nan')):.2%}" if _fs_hs.get(21) else "—",
                         help="Worst-case 5% outcome over 21 trading days")
            _fsc4.metric("21d 95th Pctile", f"{_fs_hs.get(21,{}).get(95,float('nan')):.2%}" if _fs_hs.get(21) else "—")

            _paths_df = _fs.get("paths_df", pd.DataFrame())
            if not _paths_df.empty:
                _fs_fig = _go_fs.Figure()
                _band_pairs = [(5, 95, "rgba(231,76,60,0.08)"), (10, 90, "rgba(231,76,60,0.10)"),
                               (25, 75, "rgba(79,142,247,0.12)")]
                for _lo, _hi, _fill in _band_pairs:
                    if _lo in _paths_df.columns and _hi in _paths_df.columns:
                        _fs_fig.add_trace(_go_fs.Scatter(
                            x=list(_paths_df.index) + list(_paths_df.index[::-1]),
                            y=list((_paths_df[_hi] - 1) * 100) + list((_paths_df[_lo] - 1) * 100)[::-1],
                            fill="toself", fillcolor=_fill,
                            line=dict(color="rgba(0,0,0,0)"),
                            name=f"P{_lo}–P{_hi}", showlegend=True,
                            hoverinfo="skip",
                        ))
                if 50 in _paths_df.columns:
                    _fs_fig.add_trace(_go_fs.Scatter(
                        x=list(_paths_df.index), y=(_paths_df[50] - 1) * 100,
                        name="Median (P50)", line=dict(color="#4f8ef7", width=2.5),
                        hovertemplate="Day %{x}<br>Median: %{y:+.2f}%<extra></extra>",
                    ))
                _fs_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _fs_fig.update_layout(
                    height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    xaxis=dict(showgrid=False, color="#6b7280", title="Trading Days Forward"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Cumulative Return (%)"),
                    legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fs_fig, use_container_width=True)
                st.caption(f"500 Monte Carlo paths starting from **{_fs_cur}** regime · "
                           "Shaded bands: P5–P95 (outer) / P10–P90 / P25–P75 (inner)")

            # Regime return profiles
            if _fs_reg:
                st.markdown("**Empirical Return Profiles by Regime**")
                _rp_rows = []
                for _rg, _rp in _fs_reg.items():
                    _rp_rows.append({
                        "Regime": _rg,
                        "Mean Daily": f"{_rp.get('mean_daily', 0):.3%}",
                        "Daily Vol": f"{_rp.get('std_daily', 0):.3%}",
                        "Skew": f"{_rp.get('skewness', 0):.2f}",
                        "Annualised Return": f"{_rp.get('mean_daily', 0) * 252:.1%}",
                        "Annualised Vol": f"{_rp.get('std_daily', 0) * (252**0.5):.1%}",
                        "N Obs": _rp.get("n_obs", 0),
                    })
                st.dataframe(pd.DataFrame(_rp_rows), use_container_width=True, hide_index=True)

            _fs_tm = _fs.get("transition_matrix", pd.DataFrame())
            if not _fs_tm.empty:
                with st.expander("Regime transition matrix (historical)"):
                    st.dataframe(_fs_tm.style.format("{:.1%}"), use_container_width=True)
                    st.caption("Row = current regime · Column = next day's regime probability")
        else:
            st.info("Forward simulation unavailable — requires strategy_daily_return with 100+ rows.")
    except Exception as _fs_e:
        st.caption(f"Forward simulation unavailable: {_fs_e}")


# =============================================================================
# ANALYTICS sub-tab 21: Synthetic CDX Proxy
# =============================================================================
with _analytics_sub21:
    import plotly.graph_objects as _go_cdx
    st.header("Synthetic CDX — Credit Default Swap Index Proxy")
    st.markdown(
        """
        **CDX.NA.IG** and **CDX.NA.HY** are OTC credit default swap indices — you pay a spread
        (in bps/year) to insure a basket of 125 IG or 100 HY corporate names against default.
        They are the most liquid instruments for expressing credit views and typically *lead*
        cash bond spreads because derivatives markets price information faster.

        Since CDX data is not freely available, this builds a **synthetic proxy** by normalising
        available credit spread, volatility, and equity drawdown data into a 0–100 stress index:
        - 0–30: Credit markets tight / risk-on
        - 30–60: Normal conditions
        - 60–80: Wide / stress building
        - 80+: Crisis-level stress

        The synthetic index is calibrated so that GFC 2008 and COVID March 2020 register near 100.
        """
    )
    try:
        _cdx = load_cdx_proxy(df)
        if _cdx.get("available"):
            _cdxc = _cdx.get("current", {})
            _cx1, _cx2, _cx3, _cx4 = st.columns(4)
            _cx1.metric("Synthetic CDX-HY", f"{_cdxc.get('cdx_hy_proxy', float('nan')):.1f}/100",
                        help="0=Tight · 100=Crisis. Driven by HY spread, HY momentum, VIX, SP500 drawdown")
            _cx2.metric("Synthetic CDX-IG", f"{_cdxc.get('cdx_ig_proxy', float('nan')):.1f}/100",
                        help="0=Tight · 100=Crisis. Driven by IG spread, IG momentum, yield curve flattening")
            _cx3.metric("Composite", f"{_cdxc.get('cdx_composite', float('nan')):.1f}/100")
            _cx4.metric("Regime", _cdxc.get("cdx_regime", "—"))

            _cdx_hist = _cdx.get("rolling_cdx", pd.DataFrame())
            if not _cdx_hist.empty:
                _cdx_hist = _cdx_hist.copy()
                if "date" in df.columns:
                    _cdx_hist.index = pd.to_datetime(df["date"].values[-len(_cdx_hist):])
                _cdx_fig = _go_cdx.Figure()
                if "cdx_hy_proxy" in _cdx_hist.columns:
                    _cdx_fig.add_trace(_go_cdx.Scatter(
                        x=_cdx_hist.index, y=_cdx_hist["cdx_hy_proxy"],
                        name="Synthetic CDX-HY", line=dict(color="#e74c3c", width=2),
                        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
                    ))
                if "cdx_ig_proxy" in _cdx_hist.columns:
                    _cdx_fig.add_trace(_go_cdx.Scatter(
                        x=_cdx_hist.index, y=_cdx_hist["cdx_ig_proxy"],
                        name="Synthetic CDX-IG", line=dict(color="#4f8ef7", width=1.5, dash="dot"),
                    ))
                for _lv, _clr, _lbl in [(60, "#e67e22", "Stress"), (80, "#e74c3c", "Crisis")]:
                    _cdx_fig.add_hline(y=_lv, line=dict(color=_clr, dash="dash", width=1),
                                       annotation_text=_lbl, annotation_position="top right",
                                       annotation_font=dict(color=_clr, size=10))
                _cdx_fig.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Stress Score (0–100)", range=[0, 105]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_cdx_fig, use_container_width=True)

            _cdx_pct_hy = _cdx.get("historical_percentile_hy")
            _cdx_pct_ig = _cdx.get("historical_percentile_ig")
            _cdx_corr   = _cdx.get("correlation_with_composite")
            _cx_info = []
            if _cdx_pct_hy is not None:
                _cx_info.append(f"CDX-HY at **{_cdx_pct_hy:.0f}th percentile** vs full history")
            if _cdx_pct_ig is not None:
                _cx_info.append(f"CDX-IG at **{_cdx_pct_ig:.0f}th percentile**")
            if _cdx_corr is not None:
                _cx_info.append(f"Correlation with composite risk score: **{_cdx_corr:.3f}**")
            if _cx_info:
                st.caption(" · ".join(_cx_info))

            _cdx_peaks = _cdx.get("stress_peaks", pd.DataFrame())
            if not _cdx_peaks.empty:
                with st.expander("Top stress peaks (historical)"):
                    st.dataframe(_cdx_peaks, use_container_width=True, hide_index=True)
        else:
            st.info("CDX proxy unavailable — requires HY spread data.")
    except Exception as _cdx_e:
        st.caption(f"CDX proxy unavailable: {_cdx_e}")


# =============================================================================
# ANALYTICS sub-tab 22: Equity-Credit Correlation Regime
# =============================================================================
with _analytics_sub22:
    import plotly.graph_objects as _go_cr
    st.header("Equity-Credit Correlation Regime")
    st.markdown(
        """
        **Normally, stocks and credit spreads move in opposite directions**: when equities rally,
        spreads tighten (both signal risk appetite). This negative correlation is the foundation
        of most risk-off hedging strategies.

        When this correlation **breaks down or goes positive**, it signals a **dislocation**:
        both asset classes are moving adversely at the same time. This has historically preceded
        major credit stress episodes by 2–4 weeks.

        - **Normal (negative)**: healthy risk-on / risk-off dynamics
        - **Decoupling** (weakly negative): early warning — monitor closely
        - **Dislocation** (positive): systemic risk elevated — both equities and credit under pressure

        The 90-day rolling window captures medium-term regime shifts rather than daily noise.
        """
    )
    try:
        _cr = load_correlation_regime(df)
        if _cr.get("available"):
            _crc = _cr.get("current", {})
            _cr1, _cr2, _cr3, _cr4 = st.columns(4)
            _cr1.metric("90d EQ-Credit Corr", f"{_crc.get('corr_eq_hy_90d', float('nan')):.3f}",
                        help="Pearson r between SP500 daily return and HY spread daily change. Normally negative.")
            _cr2.metric("Correlation Regime", _crc.get("corr_regime_90d", "—"))
            _cr3.metric("Active Dislocation", f"{_cr.get('current_streak', 0)} days",
                        delta_color="inverse",
                        delta=f"{_cr.get('current_streak', 0)} days" if _cr.get("current_streak", 0) > 0 else None)
            _cr4.metric("21d Corr", f"{_crc.get('corr_eq_hy_21d', float('nan')):.3f}",
                        help="Shorter-term 21d correlation — faster signal")

            if _cr.get("warning"):
                st.warning(_cr["warning"])

            _cr_df = _cr.get("df", pd.DataFrame())
            if not _cr_df.empty and "corr_eq_hy_90d" in _cr_df.columns:
                _cr_df = _cr_df.copy()
                _cr_df["date"] = pd.to_datetime(_cr_df["date"])
                _cr_fig = _go_cr.Figure()
                _cr_fig.add_trace(_go_cr.Scatter(
                    x=_cr_df["date"], y=_cr_df["corr_eq_hy_90d"],
                    name="90d EQ-Credit Correlation",
                    line=dict(color="#4f8ef7", width=2),
                    fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>90d Corr: %{y:.3f}<extra></extra>",
                ))
                if "corr_eq_hy_21d" in _cr_df.columns:
                    _cr_fig.add_trace(_go_cr.Scatter(
                        x=_cr_df["date"], y=_cr_df["corr_eq_hy_21d"],
                        name="21d EQ-Credit Correlation",
                        line=dict(color="#e67e22", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>21d Corr: %{y:.3f}<extra></extra>",
                    ))
                _cr_fig.add_hline(y=0, line_color="rgba(231,76,60,0.6)", line_width=1.5,
                                  annotation_text="Dislocation threshold",
                                  annotation_font=dict(color="#e74c3c", size=10))
                _cr_fig.add_hline(y=-0.3, line_color="rgba(230,126,34,0.4)", line_width=1,
                                  annotation_text="Decoupling threshold",
                                  annotation_font=dict(color="#e67e22", size=10))
                _cr_fig.update_layout(
                    height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Pearson Correlation", range=[-1.05, 1.05]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_cr_fig, use_container_width=True)
                st.caption("Above 0 = dislocation (both assets moving adversely). Red line = warning threshold.")

            _cr_eps = _cr.get("dislocation_episodes", pd.DataFrame())
            if not _cr_eps.empty:
                with st.expander(f"Historical dislocation episodes ({len(_cr_eps)})"):
                    st.dataframe(_cr_eps, use_container_width=True, hide_index=True)
                    st.caption("'Subsequent HY change' = HY spread change in the 4 weeks following dislocation start.")
        else:
            st.info("Correlation analysis unavailable — requires SP500 and HY spread data.")
    except Exception as _cr_e:
        st.caption(f"Correlation analysis unavailable: {_cr_e}")

# =============================================================================
# ANALYTICS sub-tab 23: Regime-Conditional Return Table
# =============================================================================
with _analytics_sub23:
    import plotly.graph_objects as _go_rrt
    st.header("Regime-Conditional Return Table")
    st.markdown(
        """
        **How does each asset class perform in each risk regime?**
        This table shows historical mean returns, hit rates, and volatility
        conditioned on the current regime label — across 1M, 3M, 6M, and 12M horizons.
        Use it to set return expectations and calibrate position sizing by regime.
        """
    )
    try:
        _rrt = load_regime_return_table(df)
        if _rrt.get("available"):
            _tbl_result = _rrt.get("table", {})
            _rrt_tbl = _tbl_result.get("table", None) if isinstance(_tbl_result, dict) else None

            # Regime outlook narrative
            _rrt_outlook = _rrt.get("current_regime_outlook", {})
            if _rrt_outlook.get("available") and _rrt_outlook.get("narrative"):
                st.info(_rrt_outlook["narrative"])

            # Pivot: show mean SP500 return by regime x horizon
            if _rrt_tbl is not None and not _rrt_tbl.empty:
                _assets_avail = _tbl_result.get("assets_available", [])

                # SP500 mean return pivot
                if "sp500" in _assets_avail and "mean_sp500" in _rrt_tbl.columns:
                    _rrt_pivot = _rrt_tbl["mean_sp500"].unstack(level="horizon")
                    _rrt_pivot.columns = ["1M", "3M", "6M", "12M"]
                    _rrt_pivot = _rrt_pivot * 100
                    with st.expander("SP500 Mean Return (%) by Regime & Horizon", expanded=True):
                        st.dataframe(_rrt_pivot.style.format("{:.1f}%"), use_container_width=True)

                # HY spread mean change pivot
                if "hy_spread" in _assets_avail and "mean_hy_spread" in _rrt_tbl.columns:
                    _hy_pivot = _rrt_tbl["mean_hy_spread"].unstack(level="horizon")
                    _hy_pivot.columns = ["1M", "3M", "6M", "12M"]
                    with st.expander("HY Spread Mean Change (bps) by Regime & Horizon"):
                        st.dataframe(_hy_pivot.style.format("{:.0f} bps"), use_container_width=True)

                # Observation counts
                if "n_obs" in _rrt_tbl.columns:
                    _nobs = _rrt_tbl["n_obs"].unstack(level="horizon")
                    _nobs.columns = ["1M", "3M", "6M", "12M"]
                    with st.expander("Observation Counts by Regime & Horizon"):
                        st.dataframe(_nobs, use_container_width=True)

            # Best entry analysis
            _bea = _rrt.get("best_entry_analysis", {})
            _unconditional = _bea.get("unconditional_12m_mean")
            if _unconditional is not None:
                st.caption(f"Unconditional 12M SP500 mean return: {_unconditional*100:.1f}%")
                _regimes_bea = [r for r in ["Risk-On", "Neutral", "Caution", "Risk-Off"] if r in _bea]
                _bea_rows = []
                for _rg in _regimes_bea:
                    _rd = _bea[_rg]
                    _bea_rows.append({
                        "Regime": _rg,
                        "Entry Return 12M": f"{_rd.get('entry_12m_mean', 0)*100:.1f}%",
                        "vs Unconditional": f"{_rd.get('vs_unconditional', 0)*100:+.1f}%",
                        "N Entries": _rd.get("n_entries", 0),
                    })
                if _bea_rows:
                    with st.expander("Best Entry Points by Regime Transition"):
                        st.dataframe(pd.DataFrame(_bea_rows), use_container_width=True, hide_index=True)
                        st.caption("'Entry Return 12M' = SP500 return in the 12M following the first day of a new regime spell.")
        else:
            st.info("Regime return table unavailable — requires at least 4 regimes with 20+ observations each.")
    except Exception as _rrt_e:
        st.caption(f"Regime return table unavailable: {_rrt_e}")

# =============================================================================
# ANALYTICS sub-tab 24: Default Cycle Positioning
# =============================================================================
with _analytics_sub24:
    import plotly.graph_objects as _go_dc
    st.header("Default Cycle Positioning")
    st.markdown(
        """
        **Where are we in the credit default cycle?**
        Compare current Jarrow-Turnbull implied default rates against historical cycle peaks:
        1991 recession, 2001–02 tech bust, 2008–09 GFC, 2015–16 energy/mining, 2020 COVID.
        The cycle position tells you how much additional stress is already priced in
        and what upside-to-peak scenarios look like.
        """
    )
    try:
        _dc = load_default_cycle(df)
        if _dc.get("available"):
            _dcc = _dc.get("current", {})
            _d1, _d2, _d3, _d4 = st.columns(4)
            _d1.metric("Implied Default Rate", f"{_dcc.get('current_implied_pct', 0):.2f}%",
                       help="Jarrow-Turnbull: HY spread / (100 × LGD)")
            _d2.metric("Phase", _dcc.get("current_phase", "—"))
            _d3.metric("% of GFC Peak", f"{_dcc.get('pct_of_gfc_peak', 0):.0f}%")
            _d4.metric("% of COVID Peak", f"{_dcc.get('pct_of_covid_peak', 0):.0f}%")

            if _dcc.get("interpretation"):
                st.info(_dcc["interpretation"])

            # Cycle comparison table
            _dc_cc = _dc.get("cycle_comparison")
            if _dc_cc is not None and not _dc_cc.empty:
                with st.expander("Historical Cycle Comparison", expanded=True):
                    st.dataframe(_dc_cc, use_container_width=True)

            # Time series chart
            _dc_ts = _dc.get("time_series")
            if _dc_ts is not None and not _dc_ts.empty and "implied_default_pct" in _dc_ts.columns:
                _dc_fig = _go_dc.Figure()
                _dc_ts_idx = pd.to_datetime(_dc_ts.index if _dc_ts.index.dtype != object else _dc_ts.get("date", _dc_ts.index))
                _dc_fig.add_trace(_go_dc.Scatter(
                    x=_dc_ts_idx, y=_dc_ts["implied_default_pct"],
                    name="Implied Default Rate", line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Implied Default: %{y:.2f}%<extra></extra>",
                ))
                _dc_fig.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Implied Default Rate (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_dc_fig, use_container_width=True)
        else:
            st.info("Default cycle unavailable — requires HY spread data.")
    except Exception as _dc_e:
        st.caption(f"Default cycle unavailable: {_dc_e}")

# =============================================================================
# ANALYTICS sub-tab 25: Signal Comparison Mode
# =============================================================================
with _analytics_sub25:
    st.header("Signal Comparison Mode")
    st.markdown(
        """
        **Compare the full dashboard signal state between any two dates.**
        Use this to answer: *"What changed since the last Fed meeting?"*
        or *"How does today compare to the 2022 peak stress?"*
        """
    )
    try:
        _avail_dates = get_available_dates(df)
        if len(_avail_dates) >= 2:
            _cmp_col1, _cmp_col2 = st.columns(2)
            with _cmp_col1:
                _cmp_date_a = st.selectbox("Baseline date (Date A)",
                                            options=_avail_dates,
                                            index=max(0, len(_avail_dates) - 252),
                                            key="cmp_date_a")
            with _cmp_col2:
                _cmp_date_b = st.selectbox("Comparison date (Date B)",
                                            options=_avail_dates,
                                            index=len(_avail_dates) - 1,
                                            key="cmp_date_b")

            if st.button("Run Comparison", key="run_comparison"):
                _cmp = compare_dates(df, _cmp_date_a, _cmp_date_b)
                if _cmp.get("available"):
                    # Regime change banner
                    if _cmp.get("regime_changed"):
                        st.warning(
                            f"Regime changed: **{_cmp['regime_a']}** → **{_cmp['regime_b']}**"
                        )
                    else:
                        st.success(f"Regime unchanged: **{_cmp['regime_a']}**")

                    # Risk summary
                    _rs = _cmp.get("risk_summary", "Mixed")
                    _cd = _cmp.get("composite_delta")
                    _rs_str = f"{_rs}" + (f" (composite score {_cd:+.1f})" if _cd is not None else "")
                    st.metric("Risk Direction", _rs_str)

                    # Biggest movers
                    _bm = _cmp.get("biggest_movers", [])
                    if _bm:
                        st.caption("**Top movers:**  " + " · ".join(
                            f"{m['label']} ({m['delta']:+.1f}, {m['direction']})" for m in _bm
                        ))

                    # Full comparison table
                    _cmp_tbl = format_comparison_table(_cmp)
                    if not _cmp_tbl.empty:
                        st.dataframe(_cmp_tbl, use_container_width=True, hide_index=True)
                else:
                    st.warning("Comparison failed — try different dates.")
            else:
                st.caption("Select two dates above and click **Run Comparison** to see the signal diff.")
        else:
            st.info("Comparison mode requires at least 2 dates with valid composite score data.")
    except Exception as _cmp_e:
        st.caption(f"Comparison mode unavailable: {_cmp_e}")

# =============================================================================
# ANALYTICS sub-tab 26: Cross-Asset Correlation Heatmap
# =============================================================================
with _analytics_sub26:
    import plotly.graph_objects as _go_hm
    st.header("Cross-Asset Correlation Heatmap")
    st.markdown(
        """
        Rolling 90-day Pearson correlation matrix across equity, credit, rates, and volatility.
        When all correlations spike toward ±1 during stress, diversification breaks down.
        **Crisis regime** (avg abs correlation > 0.75) historically precedes sharp credit dislocations.
        """
    )
    try:
        _hm = load_correlation_heatmap(df)
        if _hm.get("available"):
            _hm_stress = _hm.get("stress", {})
            _h1, _h2, _h3, _h4 = st.columns(4)
            _h1.metric("Avg |Correlation|", f"{_hm_stress.get('avg_abs_correlation', 0):.3f}")
            _h2.metric("Stress Regime", _hm_stress.get("stress_regime", "—"))
            _h3.metric("Diversification Ratio", f"{_hm_stress.get('diversification_ratio', 0):.3f}",
                       help="1 - avg|corr|. Higher = more diversification remaining")
            _h4.metric("Signals", _hm.get("n_signals", 0))

            _mx_pair = _hm_stress.get("max_pair")
            _mn_pair = _hm_stress.get("min_pair")
            if _mx_pair:
                st.caption(f"Highest correlation: **{_mx_pair[0]} ↔ {_mx_pair[1]}** ({_mx_pair[2]:+.3f})")
            if _mn_pair:
                st.caption(f"Most diversifying: **{_mn_pair[0]} ↔ {_mn_pair[1]}** ({_mn_pair[2]:+.3f})")
            if _hm.get("interpretation"):
                st.info(_hm["interpretation"])

            # Heatmap chart
            _hm_mat = _hm.get("matrix_current")
            if _hm_mat is not None and not _hm_mat.empty:
                _hm_labels = list(_hm_mat.columns)
                _hm_vals = _hm_mat.values.tolist()
                _hm_fig = _go_hm.Figure(data=_go_hm.Heatmap(
                    z=_hm_vals, x=_hm_labels, y=_hm_labels,
                    colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in _hm_vals],
                    texttemplate="%{text}",
                    hovertemplate="%{y} ↔ %{x}<br>Corr: %{z:.3f}<extra></extra>",
                ))
                _hm_fig.update_layout(
                    height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    xaxis=dict(color="#6b7280"), yaxis=dict(color="#6b7280"),
                )
                st.plotly_chart(_hm_fig, use_container_width=True)

            # Rolling avg correlation time series
            _hm_roll = _hm.get("rolling_avg_corr")
            if _hm_roll is not None and len(_hm_roll) > 0:
                _hm_roll_fig = _go_hm.Figure()
                _hm_roll_fig.add_trace(_go_hm.Scatter(
                    x=pd.to_datetime(_hm_roll.index), y=_hm_roll.values,
                    name="Avg |Corr|", line=dict(color="#9b59b6", width=2),
                    fill="tozeroy", fillcolor="rgba(155,89,182,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Avg |Corr|: %{y:.3f}<extra></extra>",
                ))
                for _thresh, _col, _lbl in [(0.75, "#e74c3c", "Crisis"), (0.60, "#e67e22", "Stress"), (0.45, "#f39c12", "Elevated")]:
                    _hm_roll_fig.add_hline(y=_thresh, line=dict(color=_col, dash="dash", width=1),
                                           annotation_text=_lbl, annotation_font=dict(color=_col, size=9))
                _hm_roll_fig.update_layout(
                    height=180, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Avg |Corr|", range=[0, 1]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_hm_roll_fig, use_container_width=True)
        else:
            st.info("Correlation heatmap unavailable — requires at least 4 signals (HY spread, IG spread, VIX, 10y yield, SP500).")
    except Exception as _hm_e:
        st.caption(f"Correlation heatmap unavailable: {_hm_e}")

# =============================================================================
# ANALYTICS sub-tab 27: Spread Volatility Monitor
# =============================================================================
with _analytics_sub27:
    import plotly.graph_objects as _go_sv
    st.header("Credit Spread Volatility Monitor")
    st.markdown(
        """
        Rolling volatility of **daily HY and IG spread changes** — distinct from spread levels.
        Rising spread vol while levels are flat is an early warning: markets are uncertain about direction.
        A GARCH(1,1) model provides a conditional vol estimate that responds faster to recent shocks.
        Regimes are percentile-ranked against a trailing 1-year window.
        """
    )
    try:
        _sv = load_spread_volatility(df)
        if _sv.get("available"):
            _svc = _sv.get("current", {})
            _sv1, _sv2, _sv3, _sv4 = st.columns(4)
            _sv1.metric("HY Spread Vol (21d)", f"{_svc.get('hy_spread_vol_21d', 0):.1f} bps/yr" if _svc.get('hy_spread_vol_21d') else "—")
            _sv2.metric("HY Vol Regime", _svc.get("hy_vol_regime", "—"))
            _sv3.metric("HY Vol Z-Score (1y)", f"{_svc.get('hy_vol_zscore_1y', 0):.2f}" if _svc.get('hy_vol_zscore_1y') is not None else "—")
            _sv4.metric("Vol-of-Vol (63d)", f"{_sv.get('vol_of_vol', 0):.1f}" if _sv.get("vol_of_vol") else "—",
                        help="Std of 21d vol over last 63 days")
            if _sv.get("warning"):
                st.warning(_sv["warning"])
            if _sv.get("interpretation"):
                st.caption(_sv["interpretation"])

            _sv_hist = _sv.get("historical")
            if _sv_hist is not None and not _sv_hist.empty:
                _sv_fig = _go_sv.Figure()
                if "hy_spread_vol_21d" in _sv_hist.columns:
                    _sv_fig.add_trace(_go_sv.Scatter(
                        x=pd.to_datetime(_sv_hist.index), y=_sv_hist["hy_spread_vol_21d"],
                        name="HY Vol 21d", line=dict(color="#e74c3c", width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>HY Vol 21d: %{y:.1f}<extra></extra>",
                    ))
                if "hy_spread_vol_63d" in _sv_hist.columns:
                    _sv_fig.add_trace(_go_sv.Scatter(
                        x=pd.to_datetime(_sv_hist.index), y=_sv_hist["hy_spread_vol_63d"],
                        name="HY Vol 63d", line=dict(color="#e67e22", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>HY Vol 63d: %{y:.1f}<extra></extra>",
                    ))
                if "hy_spread_vol_garch" in _sv_hist.columns:
                    _sv_fig.add_trace(_go_sv.Scatter(
                        x=pd.to_datetime(_sv_hist.index), y=_sv_hist["hy_spread_vol_garch"],
                        name="GARCH(1,1)", line=dict(color="#9b59b6", width=1.5, dash="dash"),
                        hovertemplate="%{x|%Y-%m-%d}<br>GARCH: %{y:.1f}<extra></extra>",
                    ))
                _sv_fig.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Spread Vol (bps/yr)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_sv_fig, use_container_width=True)
                st.caption("Annualised rolling std of daily spread changes (bps). GARCH(1,1): ω=5%×var, α=0.10, β=0.85.")
        else:
            st.info("Spread volatility unavailable — requires HY spread or IG spread data.")
    except Exception as _sv_e:
        st.caption(f"Spread volatility unavailable: {_sv_e}")

# =============================================================================
# ANALYTICS sub-tab 28: Fallen Angel Risk
# =============================================================================
with _analytics_sub28:
    import plotly.graph_objects as _go_fa
    st.header("Fallen Angel Risk Monitor")
    st.markdown(
        """
        **Fallen angels** are bonds downgraded from IG (BBB) to HY — forced selling by IG-mandate funds
        can create sharp spread widening. The **HY/IG spread ratio** (normally 3–4×) is the key cliff-edge signal.
        When the ratio rises sharply, markets are pricing elevated downgrade risk.
        BBB is the largest IG tranche; its downgrade wave in 2020 was the largest in history.
        """
    )
    try:
        _fa = load_fallen_angel(df)
        if _fa.get("available"):
            _fac = _fa.get("current", {})
            _f1, _f2, _f3, _f4 = st.columns(4)
            _f1.metric("HY/IG Ratio", f"{_fac.get('hy_ig_ratio', 0):.2f}×")
            _f2.metric("Z-Score (1y)", f"{_fac.get('hy_ig_ratio_zscore', 0):.2f}")
            _f3.metric("Fallen Angel Regime", _fac.get("fallen_angel_regime", "—"))
            _f4.metric("Signal", f"{_fac.get('fallen_angel_signal', 0):.0f}/100")

            if _fac.get("cliff_risk_flag"):
                st.error("Cliff risk flag active — HY/IG ratio significantly above historical median. Elevated fallen angel risk.")
            if _fa.get("warning"):
                st.warning(_fa["warning"])

            _bb_diff = _fac.get("bbb_bb_differential")
            if _bb_diff is not None:
                st.metric("BBB−BB Differential", f"{_bb_diff:.0f} bps",
                          help="Fallen angel premium — what market charges for BBB-to-junk transition risk")

            if _fa.get("interpretation"):
                st.caption(_fa["interpretation"])

            # Historical ratio chart
            _fa_hist = _fa.get("historical")
            if _fa_hist is not None and not _fa_hist.empty and "hy_ig_ratio" in _fa_hist.columns:
                _fa_fig = _go_fa.Figure()
                _fa_fig.add_trace(_go_fa.Scatter(
                    x=pd.to_datetime(_fa_hist.index), y=_fa_hist["hy_ig_ratio"],
                    name="HY/IG Ratio", line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY/IG: %{y:.2f}×<extra></extra>",
                ))
                _fa_fig.add_hline(y=4.5, line=dict(color="rgba(230,126,34,0.5)", dash="dash", width=1),
                                  annotation_text="Elevated (4.5×)", annotation_font=dict(color="#e67e22", size=10))
                _fa_fig.add_hline(y=6.0, line=dict(color="rgba(231,76,60,0.5)", dash="dot", width=1),
                                  annotation_text="Crisis (6.0×)", annotation_font=dict(color="#e74c3c", size=10))
                _fa_fig.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY/IG Spread Ratio"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fa_fig, use_container_width=True)

            # Historical extremes
            _fa_ext = _fa.get("historical_extremes", {})
            if _fa_ext:
                _pct = _fa.get("percentile_current")
                st.caption(
                    f"Current ratio at **{_pct:.0f}th percentile** of history. "
                    f"Historical peak: {_fa_ext.get('max_ratio', 0):.2f}× ({_fa_ext.get('max_ratio_date', '—')}). "
                    f"Historical trough: {_fa_ext.get('min_ratio', 0):.2f}× ({_fa_ext.get('min_ratio_date', '—')})."
                )
        else:
            st.info("Fallen angel monitor unavailable — requires both HY and IG spread data.")
    except Exception as _fa_e:
        st.caption(f"Fallen angel unavailable: {_fa_e}")

# =============================================================================
# ANALYTICS sub-tab 29: Global Credit Divergence
# =============================================================================
with _analytics_sub29:
    import plotly.graph_objects as _go_gc
    st.header("Global Credit Divergence (US vs International HY)")
    st.markdown(
        """
        Tracks **EMB/HYG** and **HYXU/HYG** price ratios to detect when US and international
        credit markets diverge. US leads in rate-driven stress; Europe/EM leads in
        sovereign/bank stress. A falling ratio signals international credit underperforming
        US — historically precedes contagion into US HY by 4–6 weeks.
        """
    )
    try:
        _gc = load_global_credit(df)
        _gc_snap = _gc.get("snapshot", {})
        _gc_cur = _gc.get("current", {})
        _g1, _g2, _g3, _g4 = st.columns(4)
        _g1.metric("US/Intl Ratio", f"{_gc_cur.get('us_intl_ratio', 0):.3f}" if _gc_cur.get('us_intl_ratio') else "—")
        _g2.metric("63d Divergence", f"{(_gc_cur.get('us_intl_divergence_63d') or 0)*100:+.1f}%")
        _g3.metric("Global Signal", f"{_gc_cur.get('global_credit_signal', 0):.0f}/100")
        _g4.metric("Regime", _gc_snap.get("divergence_regime", "—"))
        if _gc.get("lead_signal"):
            st.warning(_gc["lead_signal"])
        if _gc.get("interpretation"):
            st.caption(_gc["interpretation"])
        _gc_hist = _gc.get("historical")
        if _gc_hist is not None and not _gc_hist.empty and "us_intl_ratio" in _gc_hist.columns:
            _gc_fig = _go_gc.Figure()
            _gc_fig.add_trace(_go_gc.Scatter(
                x=pd.to_datetime(_gc_hist.index), y=_gc_hist["us_intl_ratio"],
                name="HYG/HYXU Ratio", line=dict(color="#3498db", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>US/Intl: %{y:.3f}<extra></extra>",
            ))
            _gc_fig.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Ratio"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_gc_fig, use_container_width=True)
            st.caption("Higher = US HY outperforming international HY. Falling = global stress or contagion risk.")
        if not _gc.get("available"):
            st.info("Global credit requires live market data (HYG, HYXU via yfinance).")
    except Exception as _gc_e:
        st.caption(f"Global credit unavailable: {_gc_e}")

# =============================================================================
# ANALYTICS sub-tab 30: Corporate Leverage Cycle
# =============================================================================
with _analytics_sub30:
    import plotly.graph_objects as _go_cl
    st.header("Corporate Leverage Cycle")
    st.markdown(
        """
        Tracks US nonfinancial corporate leverage — the **fundamental credit quality backdrop**.
        High leverage entering a downturn amplifies HY spread widening; deleveraging enables
        tightening. When fundamental data is unavailable, uses a **synthetic proxy** based on
        the rolling percentile rank of HY spreads (higher spread = higher implied leverage stress).
        """
    )
    try:
        _cl = load_corporate_leverage(df)
        _clc = _cl.get("current", {})
        _l1, _l2, _l3, _l4 = st.columns(4)
        _l1.metric("Leverage Signal", f"{_clc.get('leverage_stress_signal', 0):.0f}/100")
        _l2.metric("Cycle Phase", _cl.get("cycle_phase", "—"))
        _lr = _clc.get("leverage_ratio")
        _l3.metric("Leverage Ratio", f"{_lr:.1f}%" if _lr else "Synthetic proxy")
        _l4.metric("Regime", _clc.get("leverage_regime", "—"))
        if _clc.get("using_synthetic"):
            st.caption("Using synthetic proxy (rolling HY spread percentile rank) — FRED leverage data not in dataset.")
        if _cl.get("warning"):
            st.warning(_cl["warning"])
        if _cl.get("interpretation"):
            st.caption(_cl["interpretation"])
        _cl_hist = _cl.get("historical")
        if _cl_hist is not None and not _cl_hist.empty:
            _cl_fig = _go_cl.Figure()
            if "leverage_stress_signal" in _cl_hist.columns:
                _cl_fig.add_trace(_go_cl.Scatter(
                    x=pd.to_datetime(_cl_hist.index), y=_cl_hist["leverage_stress_signal"],
                    name="Leverage Signal", line=dict(color="#e67e22", width=2),
                    fill="tozeroy", fillcolor="rgba(230,126,34,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Signal: %{y:.0f}<extra></extra>",
                ))
            if "synthetic_leverage_proxy" in _cl_hist.columns:
                _cl_fig.add_trace(_go_cl.Scatter(
                    x=pd.to_datetime(_cl_hist.index), y=_cl_hist["synthetic_leverage_proxy"],
                    name="Synthetic Proxy", line=dict(color="#9b59b6", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Proxy: %{y:.0f}<extra></extra>",
                ))
            _cl_fig.add_hline(y=70, line=dict(color="rgba(231,76,60,0.5)", dash="dash", width=1),
                              annotation_text="Warning threshold",
                              annotation_font=dict(color="#e74c3c", size=9))
            _cl_fig.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Signal (0-100)", range=[0, 100]),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_cl_fig, use_container_width=True)
    except Exception as _cl_e:
        st.caption(f"Corporate leverage unavailable: {_cl_e}")

# =============================================================================
# ANALYTICS sub-tab 31: Credit Spread Seasonality
# =============================================================================
with _analytics_sub31:
    import plotly.graph_objects as _go_sea
    st.header("Credit Spread Seasonality")
    st.markdown(
        """
        Historical average HY and IG spread changes by calendar month.
        Credit has persistent seasonal patterns: typically tightest in **January** (new-year
        risk-on positioning), widest in **October** (year-end risk reduction, thin markets).
        Use this as a ±20bp context signal — not a timing tool, but a calibration layer.
        """
    )
    try:
        _sea = load_seasonality(df)
        if _sea.get("available"):
            _sea_s = _sea.get("seasonality", {})
            _sea_ctx = _sea.get("current_context", {})

            # Current month context
            _sea_bias = _sea_ctx.get("seasonal_bias", "Neutral")
            _sea_col = {"Favorable": "#27ae60", "Unfavorable": "#e74c3c"}.get(_sea_bias, "#f39c12")
            _s1, _s2, _s3, _s4 = st.columns(4)
            _s1.metric("Current Month", _sea_ctx.get("current_month", "—"))
            _s2.metric("Seasonal Bias", _sea_bias)
            _s3.metric("Avg HY Change", f"{_sea_ctx.get('hy_seasonal_avg_bps', 0):+.1f} bps")
            _s4.metric("HY Hit Rate", f"{_sea_ctx.get('hy_hit_rate', 0):.0%}",
                       help="Fraction of years HY tightened this month")
            st.caption(
                f"Best months (HY tightening): **{', '.join(_sea_s.get('best_months_hy', []))}** · "
                f"Worst months: **{', '.join(_sea_s.get('worst_months_hy', []))}** · "
                f"Based on {_sea_s.get('n_years', 0)} years of data"
            )
            if _sea.get("interpretation"):
                st.info(_sea["interpretation"])

            # Bar chart of monthly avg HY change
            _hy_avgs = _sea_s.get("hy_monthly_avg_change", [])
            _ig_avgs = _sea_s.get("ig_monthly_avg_change", [])
            if _hy_avgs:
                from src.seasonality import MONTH_NAMES as _MONTHS
                _sea_fig = _go_sea.Figure()
                _bar_colors = ["#27ae60" if v < 0 else "#e74c3c" for v in _hy_avgs]
                _sea_fig.add_trace(_go_sea.Bar(
                    x=_MONTHS, y=_hy_avgs,
                    name="HY Avg Change (bps)", marker_color=_bar_colors,
                    hovertemplate="%{x}<br>HY avg: %{y:+.1f} bps<extra></extra>",
                ))
                if _ig_avgs:
                    _sea_fig.add_trace(_go_sea.Scatter(
                        x=_MONTHS, y=_ig_avgs,
                        name="IG Avg Change (bps)", line=dict(color="#3498db", width=2),
                        mode="lines+markers",
                        hovertemplate="%{x}<br>IG avg: %{y:+.1f} bps<extra></extra>",
                    ))
                _sea_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _sea_fig.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Avg Monthly Change (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    barmode="overlay",
                )
                st.plotly_chart(_sea_fig, use_container_width=True)
                st.caption("Green bars = historical avg tightening · Red bars = historical avg widening · Requires ≥3 years of data")

            # Monthly hit rate table
            _hy_hr = _sea_s.get("hy_monthly_hit_rate", [])
            if _hy_hr:
                from src.seasonality import MONTH_NAMES as _MN
                _hr_df = pd.DataFrame({
                    "Month": _MN,
                    "HY Avg (bps)": [f"{v:+.1f}" for v in _hy_avgs] if _hy_avgs else ["—"]*12,
                    "HY Hit Rate": [f"{v:.0%}" for v in _hy_hr],
                    "IG Avg (bps)": [f"{v:+.1f}" for v in _ig_avgs] if _ig_avgs else ["—"]*12,
                })
                with st.expander("Full monthly statistics table"):
                    st.dataframe(_hr_df, use_container_width=True, hide_index=True)
        else:
            st.info("Seasonality analysis unavailable — requires HY spread data with ≥3 years of history.")
    except Exception as _sea_e:
        st.caption(f"Seasonality unavailable: {_sea_e}")

# =============================================================================
# ANALYTICS sub-tab 32: Signal Traffic Light
# =============================================================================
with _analytics_sub32:
    import plotly.graph_objects as _go_tl
    st.header("Signal Traffic Light")
    st.markdown(
        "All signals classified **Green / Yellow / Orange / Red** based on historical percentile rank. "
        "Green = bottom 30th percentile (low stress). Red = top 20th percentile (extreme stress). "
        "Arrows show 21-day direction."
    )
    try:
        _tl = load_traffic_light(df)
        if _tl.get("available"):
            _tl_signals = _tl.get("signals")
            _tl_summary = _tl.get("summary", {})
            _tl_overall = _tl.get("overall_color", "Yellow")
            _tl_updated = _tl.get("last_updated", "—")

            _tl_color_map = {"Green": "#27ae60", "Yellow": "#f39c12", "Orange": "#e67e22", "Red": "#e74c3c"}
            _tl_overall_color = _tl_color_map.get(_tl_overall, "#9aa0aa")

            _tl_c1, _tl_c2, _tl_c3, _tl_c4, _tl_c5 = st.columns(5)
            _tl_c1.metric("Overall", _tl_overall)
            _tl_c2.metric("Green", _tl_summary.get("green", 0))
            _tl_c3.metric("Yellow", _tl_summary.get("yellow", 0))
            _tl_c4.metric("Orange", _tl_summary.get("orange", 0))
            _tl_c5.metric("Red", _tl_summary.get("red", 0))

            if _tl_signals is not None and not _tl_signals.empty:
                _tl_avail = _tl_signals[_tl_signals["available"] == True].copy()
                if not _tl_avail.empty:
                    _tl_fig = _go_tl.Figure()
                    for _, _row in _tl_avail.iterrows():
                        _c = _tl_color_map.get(_row["color"], "#9aa0aa")
                        _label = f"{_row['direction']} {_row['label']}"
                        _tl_fig.add_trace(_go_tl.Indicator(
                            mode="number+delta",
                            value=_row["current_val"],
                            title={"text": _label, "font": {"size": 11, "color": "#9aa0aa"}},
                            number={"font": {"size": 18, "color": _c}, "suffix": ""},
                            delta={"reference": _row["current_val"] - 0.001, "relative": False,
                                   "valueformat": ".0f"},
                            domain={"row": 0, "column": 0},
                        ))

                    # Display as styled dataframe instead (cleaner)
                    _tl_display = _tl_avail[["label", "color", "current_val", "percentile", "direction"]].copy()
                    _tl_display.columns = ["Signal", "Status", "Current", "Percentile", "Trend"]
                    _tl_display["Current"] = _tl_display["Current"].apply(lambda x: f"{x:.1f}" if x is not None else "—")
                    _tl_display["Percentile"] = _tl_display["Percentile"].apply(lambda x: f"{x:.0f}th" if x is not None else "—")

                    def _tl_color_cell(val):
                        colors = {"Green": "color: #27ae60", "Yellow": "color: #f39c12",
                                  "Orange": "color: #e67e22", "Red": "color: #e74c3c"}
                        return colors.get(val, "")

                    st.dataframe(
                        _tl_display.style.applymap(_tl_color_cell, subset=["Status"]),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(f"Last updated: {_tl_updated} · Percentile vs full history · Lower risk signals prefer bottom percentiles")
        else:
            st.info("Traffic light unavailable — insufficient signal data.")
    except Exception as _tl_e:
        st.caption(f"Traffic light unavailable: {_tl_e}")

# =============================================================================
# ANALYTICS sub-tab 33: Interactive Shock Simulator
# =============================================================================
with _analytics_sub33:
    import plotly.graph_objects as _go_shk
    st.header("Interactive Shock Simulator")
    st.markdown(
        "Apply hypothetical market shocks and see how each sub-signal and the composite risk score would respond, "
        "based on OLS sensitivities estimated from the past 252 trading days."
    )
    try:
        _shk = load_shock_analysis(df)
        if _shk.get("available"):
            _shk_current = _shk.get("current_values", {})
            _shk_baseline = _shk.get("baseline_composite")
            _shk_regime = _shk.get("baseline_regime", "—")

            st.markdown(f"**Baseline:** Composite = `{_shk_baseline:.1f}` · Regime = **{_shk_regime}**")
            st.divider()

            # Pre-built scenarios
            st.subheader("Pre-Built Scenarios")
            _shk_scenarios = _shk.get("default_scenarios", {})
            _shk_cols = st.columns(min(len(_shk_scenarios), 5))
            for _si, (_sname, _sresult) in enumerate(_shk_scenarios.items()):
                if _sresult.get("available"):
                    _shocked_comp = _sresult.get("shocked", {}).get("composite", 0)
                    _baseline_comp = _sresult.get("baseline", {}).get("composite", 0)
                    _delta_comp = _sresult.get("deltas", {}).get("composite", 0)
                    _new_regime = _sresult.get("regime_shocked", "—")
                    _regime_changed = _sresult.get("regime_baseline") != _new_regime
                    _shk_cols[_si % 5].metric(
                        _sname,
                        f"{_shocked_comp:.1f}",
                        delta=f"{_delta_comp:+.1f}",
                        delta_color="inverse",
                        help=f"New regime: {_new_regime}" + (" ⚠ REGIME CHANGE" if _regime_changed else ""),
                    )

            # Sub-signal impact for the worst scenario
            if _shk_scenarios:
                _worst_name = max(_shk_scenarios, key=lambda k: (
                    _shk_scenarios[k].get("deltas", {}).get("composite", 0)
                    if _shk_scenarios[k].get("available") else -999
                ))
                _worst = _shk_scenarios[_worst_name]
                if _worst.get("available"):
                    st.subheader(f"Signal Impact: {_worst_name}")
                    _w_baseline = _worst.get("baseline", {}).get("sub_signals", {})
                    _w_shocked = _worst.get("shocked", {}).get("sub_signals", {})
                    _w_deltas = _worst.get("deltas", {}).get("sub_signals", {})
                    if _w_deltas:
                        _shk_rows = [
                            {"Signal": k.replace("_score_smooth", "").replace("_", " ").title(),
                             "Baseline": f"{v:.1f}",
                             "Shocked": f"{_w_shocked.get(k, 0):.1f}",
                             "Delta": f"{v:+.1f}"}
                            for k, v in _w_deltas.items()
                        ]
                        st.dataframe(pd.DataFrame(_shk_rows), use_container_width=True, hide_index=True)

            # Sensitivities expander
            _shk_sens = _shk.get("sensitivities", {})
            if _shk_sens:
                with st.expander("OLS Sensitivity Coefficients"):
                    _sens_rows = []
                    for _sig, _betas in _shk_sens.items():
                        if _betas:
                            _row = {"Signal": _sig.replace("_score_smooth", "").replace("_", " ").title()}
                            _row.update({k: f"{v:.3f}" for k, v in _betas.items() if k != "r2"})
                            _row["R²"] = f"{_betas.get('r2', 0):.2f}"
                            _sens_rows.append(_row)
                    if _sens_rows:
                        st.dataframe(pd.DataFrame(_sens_rows), use_container_width=True, hide_index=True)
                        st.caption("Betas: how much each sub-signal moves per unit change in the market variable")
        else:
            st.info("Shock simulator unavailable — requires composite score and market data.")
    except Exception as _shk_e:
        st.caption(f"Shock simulator unavailable: {_shk_e}")

# =============================================================================
# ANALYTICS sub-tab 34: Alert Precision/Recall Backtest
# =============================================================================
with _analytics_sub34:
    import plotly.graph_objects as _go_abt
    st.header("Alert Precision / Recall Backtest")
    st.markdown(
        "For each alert threshold on the composite risk score, how often did it fire historically, "
        "and what happened to HY spreads / SP500 over the next 30 days? "
        "An alert 'fires' when the score crosses **above** a threshold. "
        "A 'true positive' = alert followed by HY widening >25bps or SP500 drop >3%."
    )
    try:
        _abt = load_alert_backtest(df)
        if _abt.get("available"):
            _abt_best = _abt.get("best_threshold")
            _abt_f1 = _abt.get("best_f1")
            _abt_score = _abt.get("current_score")
            _abt_alert = _abt.get("current_alert_level", "No Alert")
            _abt_interp = _abt.get("interpretation", "")

            _ab1, _ab2, _ab3 = st.columns(3)
            _ab1.metric("Current Score", f"{_abt_score:.1f}" if _abt_score else "—")
            _ab2.metric("Current Alert Level", _abt_alert)
            _ab3.metric("Best Threshold (F1)", f"{_abt_best}" if _abt_best else "—",
                        help=f"F1 = {_abt_f1:.2f}" if _abt_f1 else None)

            if _abt_interp:
                st.info(_abt_interp)

            _abt_df = _abt.get("results_df")
            if _abt_df is not None and not _abt_df.empty:
                _abt_display = _abt_df.copy()
                for _col in ["precision", "recall", "f1", "false_alarm_rate"]:
                    if _col in _abt_display.columns:
                        _abt_display[_col] = _abt_display[_col].apply(
                            lambda x: f"{x:.1%}" if x is not None and not pd.isna(x) else "—"
                        )
                for _col in ["avg_hy_change_30d", "avg_sp500_change_30d"]:
                    if _col in _abt_display.columns:
                        _abt_display[_col] = _abt_display[_col].apply(
                            lambda x: f"{x:+.1f}" if x is not None and not pd.isna(x) else "—"
                        )
                st.dataframe(_abt_display, use_container_width=True, hide_index=True)

                # F1 vs threshold chart
                _valid_f1 = [(r["threshold"], r["f1"]) for r in _abt.get("results", [])
                             if r.get("f1") is not None and not pd.isna(r["f1"])]
                if _valid_f1:
                    _f1_thresholds, _f1_vals = zip(*_valid_f1)
                    _f1_fig = _go_abt.Figure()
                    _f1_fig.add_trace(_go_abt.Bar(
                        x=list(_f1_thresholds), y=list(_f1_vals),
                        name="F1 Score",
                        marker_color=["#27ae60" if t == _abt_best else "#3498db" for t in _f1_thresholds],
                        hovertemplate="Threshold: %{x}<br>F1: %{y:.2f}<extra></extra>",
                    ))
                    _f1_fig.update_layout(
                        height=250, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        xaxis=dict(title="Alert Threshold", color="#6b7280"),
                        yaxis=dict(title="F1 Score", color="#6b7280",
                                   showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                    )
                    st.plotly_chart(_f1_fig, use_container_width=True)
        else:
            st.info("Alert backtest unavailable — requires composite score, spread/SP500 data, and ≥504 rows.")
    except Exception as _abt_e:
        st.caption(f"Alert backtest unavailable: {_abt_e}")

# =============================================================================
# ANALYTICS sub-tab 35: PCA Signal Decomposition
# =============================================================================
with _analytics_sub35:
    import plotly.graph_objects as _go_pca
    st.header("PCA Signal Decomposition")
    st.markdown(
        "Principal Component Analysis on the 7 composite sub-scores. "
        "Shows how much of today's composite risk level is driven by each latent factor "
        "(rates-driven, credit-driven, macro-driven). "
        "Implemented from scratch using numpy eigendecomposition — no sklearn."
    )
    try:
        _pca = load_pca_analysis(df)
        if _pca.get("available"):
            _pca_result = _pca.get("pca", {})
            _pca_current = _pca.get("current", {})
            _pca_rolling = _pca.get("rolling_factor_scores")

            _pca_names = _pca_result.get("component_names", [])
            _pca_evr = _pca_result.get("explained_variance_ratio", [])
            _pca_cum = _pca_result.get("cumulative_variance", 0)

            # Variance explained
            _pca_c1, _pca_c2, _pca_c3, _pca_c4 = st.columns(4)
            for _pi, (_pn, _pe) in enumerate(zip(_pca_names, _pca_evr)):
                [_pca_c1, _pca_c2, _pca_c3][_pi].metric(_pn, f"{_pe:.1%}")
            _pca_c4.metric("Cumulative Variance", f"{_pca_cum:.1%}")

            # Current decomposition
            _pca_dom = _pca_current.get("dominant_factor", "—")
            _pca_interp = _pca_current.get("interpretation", "")
            if _pca_interp:
                st.info(_pca_interp)

            _pca_contribs = _pca_current.get("factor_contributions", [])
            if _pca_contribs:
                _pca_contrib_df = pd.DataFrame(_pca_contribs)
                if "pct_contribution" in _pca_contrib_df.columns:
                    _pca_contrib_df["pct_contribution"] = _pca_contrib_df["pct_contribution"].apply(
                        lambda x: f"{x:.1f}%" if x is not None else "—"
                    )
                if "score" in _pca_contrib_df.columns:
                    _pca_contrib_df["score"] = _pca_contrib_df["score"].apply(
                        lambda x: f"{x:.2f}" if x is not None else "—"
                    )
                st.dataframe(_pca_contrib_df, use_container_width=True, hide_index=True)

            # Loadings heatmap
            _pca_loadings = _pca_result.get("loadings")
            if _pca_loadings is not None and not _pca_loadings.empty:
                with st.expander("Factor Loadings Heatmap"):
                    _load_fig = _go_pca.Figure(data=_go_pca.Heatmap(
                        z=_pca_loadings.values.tolist(),
                        x=list(_pca_loadings.columns),
                        y=[r.replace("_score_smooth", "").replace("_", " ").title()
                           for r in _pca_loadings.index],
                        colorscale="RdBu", zmid=0,
                        hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
                    ))
                    _load_fig.update_layout(
                        height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    )
                    st.plotly_chart(_load_fig, use_container_width=True)

            # Rolling factor scores
            if _pca_rolling is not None and not _pca_rolling.empty:
                with st.expander("Rolling Factor Scores (2yr)"):
                    _roll_fig = _go_pca.Figure()
                    _pca_colors = ["#3498db", "#e74c3c", "#27ae60"]
                    for _ci, _col in enumerate(_pca_rolling.columns):
                        _roll_fig.add_trace(_go_pca.Scatter(
                            x=_pca_rolling.index, y=_pca_rolling[_col],
                            name=_col, mode="lines",
                            line=dict(color=_pca_colors[_ci % 3], width=1.5),
                            hovertemplate=f"{_col}: %{{y:.2f}}<extra></extra>",
                        ))
                    _roll_fig.update_layout(
                        height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    )
                    st.plotly_chart(_roll_fig, use_container_width=True)
        else:
            st.info("PCA unavailable — requires ≥3 sub-score columns and ≥252 rows of data.")
    except Exception as _pca_e:
        st.caption(f"PCA decomposition unavailable: {_pca_e}")

# =============================================================================
# ANALYTICS sub-tab 36: Regime Transition Forecast
# =============================================================================
with _analytics_sub36:
    import plotly.graph_objects as _go_rf
    st.header("Regime Transition Forecast")
    st.markdown(
        "Uses a Markov chain transition matrix estimated from weekly regime observations to project "
        "regime probabilities 1, 4, 8, and 12 weeks forward. "
        "The cone shows the most probable regime path and uncertainty."
    )
    try:
        _rf = load_regime_forecast(df)
        if _rf.get("available"):
            _rf_regime = _rf.get("current_regime", "—")
            _rf_stability = _rf.get("regime_stability", 0)
            _rf_change = _rf.get("expected_regime_change", False)
            _rf_ntrans = _rf.get("n_transitions_observed", 0)
            _rf_interp = _rf.get("interpretation", "")

            _rf_c1, _rf_c2, _rf_c3, _rf_c4 = st.columns(4)
            _rf_c1.metric("Current Regime", _rf_regime)
            _rf_c2.metric("Stay Probability", f"{_rf_stability:.0%}",
                          help="P(remain in current regime next week)")
            _rf_c3.metric("Regime Change Expected?", "Yes" if _rf_change else "No",
                          delta="4-week horizon")
            _rf_c4.metric("Transitions Observed", _rf_ntrans)

            if _rf_interp:
                st.info(_rf_interp)

            # Forecast probability table
            _rf_forecast = _rf.get("forecast")
            if _rf_forecast is not None and not _rf_forecast.empty:
                _rf_display = _rf_forecast.copy()
                for _col in _rf_display.columns:
                    _rf_display[_col] = _rf_display[_col].apply(lambda x: f"{x:.0%}")
                _rf_display.index.name = "Week"
                st.dataframe(_rf_display, use_container_width=True)

                # Stacked area chart
                _rf_raw = _rf.get("forecast")
                if _rf_raw is not None:
                    _rf_fig = _go_rf.Figure()
                    _rf_regime_colors = {
                        "Risk-On": "#27ae60", "Neutral": "#3498db",
                        "Caution": "#f39c12", "Risk-Off": "#e74c3c",
                    }
                    for _rc in _rf_raw.columns:
                        _rf_fig.add_trace(_go_rf.Scatter(
                            x=list(_rf_raw.index), y=list(_rf_raw[_rc]),
                            name=_rc, mode="lines+markers", stackgroup="one",
                            line=dict(color=_rf_regime_colors.get(_rc, "#9aa0aa"), width=2),
                            hovertemplate=f"{_rc}: %{{y:.0%}}<extra></extra>",
                        ))
                    _rf_fig.update_layout(
                        height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        xaxis=dict(title="Weeks Forward", color="#6b7280", showgrid=False),
                        yaxis=dict(title="Probability", color="#6b7280", tickformat=".0%",
                                   showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    )
                    st.plotly_chart(_rf_fig, use_container_width=True)

            # Transition matrix
            _rf_tmat = _rf.get("transition_matrix")
            if _rf_tmat is not None:
                with st.expander("Transition Matrix"):
                    _tmat_display = _rf_tmat.copy()
                    for _col in _tmat_display.columns:
                        _tmat_display[_col] = _tmat_display[_col].apply(lambda x: f"{x:.0%}")
                    st.dataframe(_tmat_display, use_container_width=True)
                    st.caption("Row = current regime · Column = next week's regime · Sampled weekly (every 5 trading days)")

            _rf_path = _rf.get("most_likely_path", [])
            if _rf_path:
                st.caption(f"Most likely path: {' → '.join(_rf_path)}")
        else:
            st.info("Regime forecast unavailable — requires ≥100 rows with regime labels and ≥3 regimes observed.")
    except Exception as _rf_e:
        st.caption(f"Regime forecast unavailable: {_rf_e}")

# =============================================================================
# ANALYTICS sub-tab 37: Custom Composite Builder
# =============================================================================
with _analytics_sub37:
    import plotly.graph_objects as _go_cc
    st.header("Custom Composite Builder")
    st.markdown(
        "Adjust the weight of each sub-signal and see how the custom composite score compares "
        "to the default equal-weight composite, including backtest Sharpe ratio."
    )
    try:
        _cc_default = load_custom_composite(df)
        if _cc_default.get("available"):
            _cc_labels = _cc_default.get("signal_labels", [])
            _cc_default_weights = _cc_default.get("weights_used", {})

            st.subheader("Adjust Signal Weights")
            st.caption("Weights are automatically normalized to sum to 1.0")
            _cc_weight_cols = st.columns(min(len(_cc_labels), 4))
            _cc_user_weights = {}
            for _ci, _sig in enumerate(_cc_labels):
                _col_key = _sig["col"]
                _default_w = _sig.get("default_weight", 1/7)
                _user_w = _cc_weight_cols[_ci % 4].slider(
                    _sig["label"],
                    min_value=0.0, max_value=1.0,
                    value=float(_default_w),
                    step=0.05,
                    key=f"cc_weight_{_col_key}",
                )
                _cc_user_weights[_col_key] = _user_w

            # Normalize
            _cc_total = sum(_cc_user_weights.values())
            if _cc_total > 0:
                _cc_normed = {k: v / _cc_total for k, v in _cc_user_weights.items()}
            else:
                _cc_normed = _cc_default_weights

            # Recompute with custom weights (not cached — responds to sliders)
            try:
                _cc_custom = run_custom_composite_analysis(df, weights=_cc_normed)
            except Exception:
                _cc_custom = _cc_default

            if _cc_custom.get("available"):
                _cc_comp = _cc_custom.get("comparison", {})
                _cc_cust_curr = _cc_comp.get("custom_current")
                _cc_def_curr = _cc_comp.get("default_current")
                _cc_cust_regime = _cc_comp.get("custom_regime", "—")
                _cc_def_regime = _cc_comp.get("default_regime", "—")
                _cc_cust_bt = _cc_comp.get("custom_backtest", {})
                _cc_def_bt = _cc_comp.get("default_backtest", {})
                _cc_corr = _cc_comp.get("correlation")
                _cc_sharpe_imp = _cc_comp.get("sharpe_improvement")

                _ccc1, _ccc2, _ccc3, _ccc4 = st.columns(4)
                _ccc1.metric("Custom Score", f"{_cc_cust_curr:.1f}" if _cc_cust_curr else "—",
                             delta=f"{_cc_cust_curr - _cc_def_curr:+.1f} vs default" if _cc_cust_curr and _cc_def_curr else None)
                _ccc2.metric("Custom Regime", _cc_cust_regime)
                _ccc3.metric("Sharpe Improvement",
                             f"{_cc_sharpe_imp:+.2f}" if _cc_sharpe_imp is not None else "—",
                             help="Custom Sharpe minus default Sharpe")
                _ccc4.metric("Correlation vs Default",
                             f"{_cc_corr:.2f}" if _cc_corr is not None else "—")

                # Comparison table
                _cc_bt_rows = [
                    {"Metric": "Sharpe Ratio",
                     "Custom": f"{_cc_cust_bt.get('sharpe', 0):.2f}" if _cc_cust_bt.get('sharpe') is not None else "—",
                     "Default": f"{_cc_def_bt.get('sharpe', 0):.2f}" if _cc_def_bt.get('sharpe') is not None else "—"},
                    {"Metric": "Hit Rate",
                     "Custom": f"{_cc_cust_bt.get('hit_rate', 0):.1%}" if _cc_cust_bt.get('hit_rate') is not None else "—",
                     "Default": f"{_cc_def_bt.get('hit_rate', 0):.1%}" if _cc_def_bt.get('hit_rate') is not None else "—"},
                    {"Metric": "Total Return",
                     "Custom": f"{_cc_cust_bt.get('total_return', 0):.1%}" if _cc_cust_bt.get('total_return') is not None else "—",
                     "Default": f"{_cc_def_bt.get('total_return', 0):.1%}" if _cc_def_bt.get('total_return') is not None else "—"},
                    {"Metric": "Periods",
                     "Custom": str(_cc_cust_bt.get('n_periods', 0)),
                     "Default": str(_cc_def_bt.get('n_periods', 0))},
                ]
                st.dataframe(pd.DataFrame(_cc_bt_rows), use_container_width=True, hide_index=True)

                # Time series comparison
                _cc_cust_series = _cc_comp.get("custom_composite")
                _cc_def_series = _cc_comp.get("default_composite")
                if _cc_cust_series is not None and _cc_def_series is not None:
                    _cc_fig = _go_cc.Figure()
                    _cc_fig.add_trace(_go_cc.Scatter(
                        x=_cc_cust_series.index, y=_cc_cust_series.values,
                        name="Custom Composite", mode="lines",
                        line=dict(color="#3498db", width=2),
                        hovertemplate="Custom: %{y:.1f}<extra></extra>",
                    ))
                    _cc_fig.add_trace(_go_cc.Scatter(
                        x=_cc_def_series.index, y=_cc_def_series.values,
                        name="Default (equal-weight)", mode="lines",
                        line=dict(color="#9aa0aa", width=1.5, dash="dot"),
                        hovertemplate="Default: %{y:.1f}<extra></extra>",
                    ))
                    _cc_fig.update_layout(
                        height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        yaxis=dict(title="Composite Score (0–100)", showgrid=True,
                                   gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    )
                    st.plotly_chart(_cc_fig, use_container_width=True)
                    st.caption("Last 504 trading days (2 years) · Backtest uses HY spread direction as signal target")
        else:
            st.info("Custom composite unavailable — requires ≥3 sub-signal columns and ≥252 rows.")
    except Exception as _cc_e:
        st.caption(f"Custom composite unavailable: {_cc_e}")

# =============================================================================
# ANALYTICS sub-tab 38: Cross-Asset Momentum Scorecard
# =============================================================================
with _analytics_sub38:
    import plotly.graph_objects as _go_xam
    st.header("Cross-Asset Momentum Scorecard")
    st.markdown(
        "Systematic 1M/3M/6M/12M momentum scores across equities, credit, rates, and FX. "
        "**When all directional assets align** (all positive or all negative 3M momentum), "
        "the composite signal is most reliable. Divergence = mixed/uncertain environment."
    )
    try:
        _xam = load_cross_asset_momentum(df)
        if _xam.get("available"):
            _xam_align = _xam.get("alignment", {})
            _xam_strength = _xam.get("signal_strength", "—")
            _xam_regime = _xam.get("current_regime", "—")
            _xam_interp = _xam.get("interpretation", "")

            _xa1, _xa2, _xa3, _xa4 = st.columns(4)
            _xa1.metric("Alignment Score", f"{_xam_align.get('alignment_score', 0)}/4")
            _xa2.metric("Signal Strength", _xam_strength)
            _xa3.metric("Aligned Positive", len(_xam_align.get("aligned_positive", [])))
            _xa4.metric("Current Regime", _xam_regime)

            if _xam_interp:
                st.info(_xam_interp)

            # Momentum table
            _xam_tbl = _xam.get("momentum_table")
            if _xam_tbl is not None and not _xam_tbl.empty:
                _xam_disp = _xam_tbl.copy()
                for _col in ["window_1m", "window_3m", "window_6m", "window_12m"]:
                    if _col in _xam_disp.columns:
                        _xam_disp[_col] = _xam_disp[_col].apply(
                            lambda x: f"{x:+.1f}" if x is not None and not pd.isna(x) else "—"
                        )
                st.dataframe(_xam_disp, use_container_width=True, hide_index=True)
                st.caption("Positive = risk-on momentum · Negative = risk-off · Rates shown as raw yield change")

            # Historical alignment chart
            _xam_hist = _xam.get("historical_alignment")
            if _xam_hist is not None and len(_xam_hist) > 0:
                with st.expander("Historical Alignment Score (2yr)"):
                    _xam_fig = _go_xam.Figure()
                    _xam_fig.add_trace(_go_xam.Scatter(
                        x=list(_xam_hist.index), y=list(_xam_hist.values),
                        mode="lines", name="Alignment (0-4)",
                        line=dict(color="#3498db", width=2),
                        hovertemplate="Week: %{x|%Y-%m-%d}<br>Alignment: %{y}/4<extra></extra>",
                    ))
                    _xam_fig.add_hline(y=2, line_color="rgba(255,255,255,0.2)", line_width=1, line_dash="dot")
                    _xam_fig.update_layout(
                        height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        yaxis=dict(title="Assets Aligned (0-4)", range=[0, 4], showgrid=True,
                                   gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                    )
                    st.plotly_chart(_xam_fig, use_container_width=True)
        else:
            st.info("Cross-asset momentum unavailable — requires ≥2 asset class columns and ≥252 rows.")
    except Exception as _xam_e:
        st.caption(f"Cross-asset momentum unavailable: {_xam_e}")

# =============================================================================
# ANALYTICS sub-tab 39: Volatility Regime Composite
# =============================================================================
with _analytics_sub39:
    import plotly.graph_objects as _go_vrc
    st.header("Volatility Regime Composite")
    st.markdown(
        "Unified volatility regime score (0–100) combining **VIX level**, **VIX term structure slope**, "
        "**Volatility Risk Premium (VRP)**, **CBOE SKEW**, and **MOVE** (bond vol). "
        "Answers the single question: *is vol cheap or expensive right now?*"
    )
    try:
        _vrc = load_vol_regime_composite(df)
        if _vrc.get("available"):
            _vrc_cur = _vrc.get("current", {})
            _vrc_score = _vrc_cur.get("vol_composite_score")
            _vrc_regime = _vrc_cur.get("vol_regime", "—")
            _vrc_pct = _vrc.get("percentile_current")
            _vrc_n = _vrc_cur.get("n_components", 0)
            _vrc_interp = _vrc.get("interpretation", "")

            _vrc_regime_color = {
                "Complacent": "#27ae60", "Normal": "#3498db",
                "Elevated": "#f39c12", "Stressed": "#e74c3c",
            }.get(_vrc_regime, "#9aa0aa")

            _vr1, _vr2, _vr3, _vr4 = st.columns(4)
            _vr1.metric("Vol Composite Score", f"{_vrc_score:.1f}" if _vrc_score is not None else "—")
            _vr2.metric("Vol Regime", _vrc_regime)
            _vr3.metric("Historical Percentile", f"{_vrc_pct:.0f}th" if _vrc_pct is not None else "—")
            _vr4.metric("Components Used", _vrc_n)

            if _vrc_interp:
                st.info(_vrc_interp)

            # Component breakdown
            _vrc_comps = _vrc_cur.get("components", {})
            if _vrc_comps:
                _comp_rows = [
                    {"Component": k.replace("_", " ").title(), "Score (0-100)": f"{v:.1f}" if v is not None else "—"}
                    for k, v in _vrc_comps.items()
                    if k in _vrc_cur.get("components_available", list(_vrc_comps.keys()))
                ]
                if _comp_rows:
                    st.dataframe(pd.DataFrame(_comp_rows), use_container_width=True, hide_index=True)

            # Historical chart
            _vrc_hist = _vrc.get("historical")
            if _vrc_hist is not None and "vol_composite_score" in _vrc_hist.columns:
                _vrc_fig = _go_vrc.Figure()
                _vrc_fig.add_trace(_go_vrc.Scatter(
                    x=_vrc_hist.index, y=_vrc_hist["vol_composite_score"],
                    mode="lines", name="Vol Composite",
                    line=dict(color="#e74c3c", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
                ))
                for _thresh, _col, _lbl in [(25, "#27ae60", "Complacent"), (50, "#3498db", "Normal"),
                                             (75, "#f39c12", "Elevated")]:
                    _vrc_fig.add_hline(y=_thresh, line_color=_col, line_width=1, line_dash="dot",
                                       annotation_text=_lbl, annotation_position="right")
                _vrc_fig.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Vol Composite Score (0-100)", range=[0, 100], showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_vrc_fig, use_container_width=True)
        else:
            st.info("Vol regime composite unavailable — requires VIX data and ≥252 rows.")
    except Exception as _vrc_e:
        st.caption(f"Vol regime composite unavailable: {_vrc_e}")

# =============================================================================
# ANALYTICS sub-tab 40: Credit Quality Migration Monitor
# =============================================================================
with _analytics_sub40:
    import plotly.graph_objects as _go_cqm
    st.header("Credit Quality Migration Monitor")
    st.markdown(
        "Tracks the **upgrade/downgrade dynamic** via HY vs IG spread velocity. "
        "When HY spreads widen faster than IG, downgrades and fallen angel risk are building. "
        "The **BBB cliff proxy** flags when IG is priced like HY — the most dangerous pre-crisis configuration."
    )
    try:
        _cqm = load_credit_quality_migration(df)
        if _cqm.get("available"):
            _cqm_cur = _cqm.get("current", {})
            _cqm_regime = _cqm_cur.get("migration_regime", "—")
            _cqm_zscore = _cqm_cur.get("migration_zscore")
            _cqm_cliff = _cqm_cur.get("cliff_risk_flag", False)
            _cqm_cliff_pct = _cqm_cur.get("cliff_pct")
            _cqm_warn = _cqm_cur.get("warning")
            _cqm_interp = _cqm.get("interpretation", "")

            _cqm_regime_color = {
                "Upgrade Cycle": "#27ae60", "Stable": "#3498db",
                "Downgrade Pressure": "#f39c12", "Fallen Angel Risk": "#e74c3c",
            }.get(_cqm_regime, "#9aa0aa")

            _cq1, _cq2, _cq3, _cq4 = st.columns(4)
            _cq1.metric("Migration Regime", _cqm_regime)
            _cq2.metric("Migration Z-Score", f"{_cqm_zscore:.2f}" if _cqm_zscore is not None else "—",
                        help="Positive = HY underperforming IG = downgrade pressure")
            _cq3.metric("BBB Cliff Percentile", f"{_cqm_cliff_pct:.0f}th" if _cqm_cliff_pct is not None else "—")
            _cq4.metric("Cliff Risk Flag", "⚠ ACTIVE" if _cqm_cliff else "Clear",
                        delta="elevated" if _cqm_cliff else None,
                        delta_color="inverse" if _cqm_cliff else "normal")

            if _cqm_warn:
                st.warning(_cqm_warn)
            elif _cqm_interp:
                st.info(_cqm_interp)

            # Historical migration z-score
            _cqm_series = _cqm.get("signal_series")
            if _cqm_series is not None and len(_cqm_series) > 0:
                _cqm_fig = _go_cqm.Figure()
                _cqm_colors = ["#27ae60" if v < 0 else "#e74c3c" for v in _cqm_series.values]
                _cqm_fig.add_trace(_go_cqm.Bar(
                    x=list(_cqm_series.index), y=list(_cqm_series.values),
                    name="Migration Z-Score", marker_color=_cqm_colors,
                    hovertemplate="%{x|%Y-%m-%d}<br>Z: %{y:.2f}<extra></extra>",
                ))
                _cqm_fig.add_hline(y=1.0, line_color="#f39c12", line_width=1, line_dash="dot",
                                   annotation_text="Downgrade Pressure")
                _cqm_fig.add_hline(y=2.0, line_color="#e74c3c", line_width=1, line_dash="dot",
                                   annotation_text="Fallen Angel Risk")
                _cqm_fig.add_hline(y=-1.0, line_color="#27ae60", line_width=1, line_dash="dot",
                                   annotation_text="Upgrade Cycle")
                _cqm_fig.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Migration Z-Score", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_cqm_fig, use_container_width=True)
                st.caption("Green bars = upgrade pressure (HY tightening vs IG) · Red bars = downgrade pressure · Last 504 days")
        else:
            st.info("Credit quality migration unavailable — requires HY spread and IG spread data with ≥252 rows.")
    except Exception as _cqm_e:
        st.caption(f"Credit quality migration unavailable: {_cqm_e}")

# =============================================================================
# ANALYTICS sub-tab 41: Macro Surprise Index
# =============================================================================
with _analytics_sub41:
    import plotly.graph_objects as _go_msi
    st.header("Macro Surprise Index")
    st.markdown(
        "How is economic data coming in relative to recent trend? "
        "Each indicator's z-score vs its 63-day rolling baseline approximates a **beat/miss** signal. "
        "Positive composite = economy beating recent trend = credit-positive tailwind."
    )
    try:
        _msi = load_macro_surprise_index(df)
        if _msi.get("available"):
            _msi_cur = _msi.get("current", {})
            _msi_score = _msi_cur.get("surprise_composite_smooth")
            _msi_regime = _msi_cur.get("surprise_regime", "—")
            _msi_signal = _msi_cur.get("surprise_credit_signal", "—")
            _msi_n = _msi_cur.get("n_indicators", 0)
            _msi_mom = _msi.get("momentum")
            _msi_interp = _msi.get("interpretation", "")

            _ms1, _ms2, _ms3, _ms4 = st.columns(4)
            _ms1.metric("Surprise Score", f"{_msi_score:.2f}" if _msi_score is not None else "—",
                        delta=f"{_msi_mom:+.2f} vs 1M ago" if _msi_mom is not None else None,
                        help="Positive = beating recent trend")
            _ms2.metric("Macro Regime", _msi_regime)
            _ms3.metric("Credit Signal", _msi_signal)
            _ms4.metric("Indicators Used", _msi_n)

            if _msi_interp:
                st.info(_msi_interp)

            # Per-indicator breakdown
            _msi_ind_scores = _msi_cur.get("indicator_scores", {})
            if _msi_ind_scores:
                _ind_rows = [
                    {"Indicator": k.replace("_", " ").title(),
                     "Z-Score": f"{v:+.2f}" if v is not None and not pd.isna(v) else "—",
                     "Signal": "Positive" if v and v > 0.3 else ("Negative" if v and v < -0.3 else "Neutral")}
                    for k, v in _msi_ind_scores.items()
                ]
                st.dataframe(pd.DataFrame(_ind_rows), use_container_width=True, hide_index=True)

            # Historical composite chart
            _msi_series = _msi.get("signal_series")
            if _msi_series is not None and len(_msi_series) > 0:
                _msi_fig = _go_msi.Figure()
                _msi_fig.add_trace(_go_msi.Scatter(
                    x=list(_msi_series.index), y=list(_msi_series.values),
                    mode="lines", name="Macro Surprise",
                    line=dict(color="#3498db", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(52,152,219,0.1)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Surprise: %{y:.2f}<extra></extra>",
                ))
                _msi_fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
                _msi_fig.add_hline(y=0.3, line_color="#27ae60", line_width=1, line_dash="dot")
                _msi_fig.add_hline(y=-0.3, line_color="#e74c3c", line_width=1, line_dash="dot")
                _msi_fig.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Surprise Index (z-score)", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_msi_fig, use_container_width=True)
                st.caption("10-day EMA smoothed · Green dashed = Positive threshold · Red dashed = Negative threshold")
        else:
            st.info("Macro surprise index unavailable — requires ≥2 macro indicator columns and ≥126 rows.")
    except Exception as _msi_e:
        st.caption(f"Macro surprise index unavailable: {_msi_e}")

# =============================================================================
# ANALYTICS sub-tab 42: Loan Market / CLO Stress Monitor
# =============================================================================
with _analytics_sub42:
    import plotly.graph_objects as _go_lmm
    st.header("Loan Market / CLO Stress Monitor")
    st.markdown(
        "**BKLN vs HYG divergence** as a proxy for CLO stress. "
        "Leveraged loans (floating rate) and HY bonds price differently under stress. "
        "When loans underperform HY bonds, CLO demand is falling — "
        "a leading indicator for broader credit deterioration."
    )
    try:
        _lmm = load_loan_market_monitor(df)
        if _lmm.get("available"):
            _lmm_snap = _lmm.get("snapshot", {})
            _lmm_lead = _lmm.get("lead_signal", "—")
            _lmm_interp = _lmm.get("interpretation", "")

            if _lmm_snap.get("available"):
                _lmm_regime = _lmm_snap.get("loan_market_regime", "—")
                _lmm_ratio = _lmm_snap.get("bkln_hyg_ratio")
                _lmm_chg = _lmm_snap.get("bkln_hyg_ratio_30d_chg")
                _lmm_zscore = _lmm_snap.get("bkln_hyg_zscore_1y")
                _lmm_clo = _lmm_snap.get("clo_stress_flag", False)
                _lmm_asof = _lmm_snap.get("as_of", "—")

                _lm1, _lm2, _lm3, _lm4 = st.columns(4)
                _lm1.metric("BKLN/HYG Ratio", f"{_lmm_ratio:.4f}" if _lmm_ratio else "—",
                            delta=f"{_lmm_chg:+.1f}% (30d)" if _lmm_chg is not None else None,
                            delta_color="normal")
                _lm2.metric("Loan Market Regime", _lmm_regime)
                _lm3.metric("CLO Stress Flag", "⚠ ACTIVE" if _lmm_clo else "Clear")
                _lm4.metric("Lead Signal", _lmm_lead)

                if _lmm_clo:
                    st.warning("CLO stress flag active — BKLN/HYG ratio is below 252d MA with elevated loan vol. Watch for spread contagion.")

            if _lmm_interp:
                st.info(_lmm_interp)

            st.caption(f"Live data via yfinance · As of: {_lmm_snap.get('as_of', '—')}")
            st.caption("BKLN = Invesco Senior Loan ETF · HYG = iShares HY Bond ETF · Ratio falling = loans underperforming")
        else:
            st.info("Loan market monitor unavailable — requires yfinance connection (BKLN, HYG).")
    except Exception as _lmm_e:
        st.caption(f"Loan market monitor unavailable: {_lmm_e}")

# =============================================================================
# ANALYTICS sub-tab 43: Regime Duration & Fatigue Clock
# =============================================================================
with _analytics_sub43:
    import plotly.graph_objects as _go_rd
    st.header("Regime Duration & Fatigue Clock")
    st.markdown(
        "How long has the current regime lasted, and how does that compare to history? "
        "**Regime fatigue** = the duration percentile — if 85, this regime has lasted longer than "
        "85% of historical instances of the same regime. Overdue regimes are fragile."
    )
    try:
        _rd = load_regime_duration(df)
        if _rd.get("available"):
            _rd_spell = _rd.get("current_spell", {})
            _rd_stats = _rd.get("duration_stats", {})
            _rd_warn = _rd.get("warning")
            _rd_interp = _rd.get("interpretation", "")
            _rd_seq = _rd.get("regime_sequence", [])

            _rd_regime = _rd_spell.get("regime", "—")
            _rd_dur = _rd_spell.get("duration_days", 0)
            _rd_fatigue = _rd_spell.get("fatigue_score")
            _rd_age_cat = _rd_spell.get("age_category", "—")
            _rd_start = _rd_spell.get("start_date", "—")
            _rd_to_p90 = _rd_spell.get("days_to_p90")

            _rd_age_color = {
                "Young": "#27ae60", "Mature": "#3498db",
                "Aging": "#f39c12", "Overdue": "#e74c3c",
            }.get(_rd_age_cat, "#9aa0aa")

            _rdc1, _rdc2, _rdc3, _rdc4 = st.columns(4)
            _rdc1.metric("Current Regime", _rd_regime)
            _rdc2.metric("Duration", f"{_rd_dur}d", delta=f"since {_rd_start}")
            _rdc3.metric("Fatigue Score", f"{_rd_fatigue:.0f}th pct" if _rd_fatigue is not None else "—",
                         help="Duration percentile vs historical spells of same regime")
            _rdc4.metric("Age Category", _rd_age_cat,
                         delta=f"{_rd_to_p90}d to Overdue" if _rd_to_p90 and _rd_to_p90 > 0 else None)

            if _rd_warn:
                st.warning(_rd_warn)
            elif _rd_interp:
                st.info(_rd_interp)

            # Per-regime duration stats table
            if _rd_stats:
                _rd_stat_rows = []
                for _rname, _rstat in _rd_stats.items():
                    _rd_stat_rows.append({
                        "Regime": _rname,
                        "N Spells": _rstat.get("n_spells", 0),
                        "Median (d)": f"{_rstat.get('median', 0):.0f}",
                        "P25 (d)": f"{_rstat.get('p25', 0):.0f}",
                        "P75 (d)": f"{_rstat.get('p75', 0):.0f}",
                        "P90 (d)": f"{_rstat.get('p90', 0):.0f}",
                        "Max (d)": f"{_rstat.get('max', 0):.0f}",
                    })
                st.dataframe(pd.DataFrame(_rd_stat_rows), use_container_width=True, hide_index=True)

            # Spell duration bar chart
            _rd_spells = _rd.get("all_spells")
            if _rd_spells is not None and not _rd_spells.empty and "regime" in _rd_spells.columns:
                with st.expander("All Historical Regime Spells"):
                    _rd_colors = {
                        "Risk-On": "#27ae60", "Neutral": "#3498db",
                        "Caution": "#f39c12", "Risk-Off": "#e74c3c",
                    }
                    _rd_spell_fig = _go_rd.Figure()
                    for _rname in _rd_spells["regime"].unique():
                        _mask = _rd_spells["regime"] == _rname
                        _rd_spell_fig.add_trace(_go_rd.Bar(
                            x=list(range(_mask.sum())),
                            y=list(_rd_spells.loc[_mask, "duration_days"]),
                            name=_rname,
                            marker_color=_rd_colors.get(_rname, "#9aa0aa"),
                            hovertemplate=f"{_rname}: %{{y}}d<extra></extra>",
                        ))
                    _rd_spell_fig.update_layout(
                        height=250, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        barmode="overlay",
                        yaxis=dict(title="Duration (trading days)", showgrid=True,
                                   gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                        xaxis=dict(title="Spell #", showgrid=False, color="#6b7280"),
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    )
                    st.plotly_chart(_rd_spell_fig, use_container_width=True)

            # Recent regime sequence
            if _rd_seq:
                st.caption("Recent regime sequence: " + " → ".join(_rd_seq))
        else:
            st.info("Regime duration unavailable — requires `final_decision` column, ≥2 regimes observed, and ≥100 rows.")
    except Exception as _rd_e:
        st.caption(f"Regime duration unavailable: {_rd_e}")

# =============================================================================
# ANALYTICS sub-tab 44: Systematic Deleveraging Detector
# =============================================================================
with _analytics_sub44:
    import plotly.graph_objects as _go_sdl
    st.header("Systematic Deleveraging Detector")
    st.markdown(
        "Identifies when quant/systematic strategies are force-selling via the **correlation-to-1** phenomenon: "
        "assets that normally move independently begin moving together. "
        "Three sub-signals: cross-asset correlation spike + vol expansion + simultaneous drawdown."
    )
    try:
        _sdl = load_systematic_deleveraging(df)
        if _sdl.get("available"):
            _sdl_cur = _sdl.get("current", {})
            _sdl_comp = _sdl_cur.get("delev_composite")
            _sdl_alert = _sdl_cur.get("alert_level", "No Signal")
            _sdl_warn = _sdl.get("warning")
            _sdl_interp = _sdl.get("interpretation", "")

            _sdl_alert_color = {
                "No Signal": "#27ae60", "Elevated Correlation": "#f39c12",
                "Deleveraging Warning": "#e67e22", "Systematic Deleveraging Alert": "#e74c3c",
            }.get(_sdl_alert, "#9aa0aa")

            _sd1, _sd2, _sd3, _sd4 = st.columns(4)
            _sd1.metric("Deleveraging Score", f"{_sdl_comp:.1f}" if _sdl_comp is not None else "—")
            _sd2.metric("Alert Level", _sdl_alert)
            _sd3.metric("Correlation Score", f"{_sdl_cur.get('corr_score', 0):.1f}")
            _sd4.metric("Vol Expansion", f"{_sdl_cur.get('vol_score', 0):.1f}")

            if _sdl_warn:
                st.warning(_sdl_warn)
            elif _sdl_interp:
                st.info(_sdl_interp)

            _sdl_in_dd = _sdl_cur.get("assets_in_drawdown", [])
            if _sdl_in_dd:
                st.caption(f"Assets currently in simultaneous drawdown: **{', '.join(_sdl_in_dd)}**")

            # Historical composite
            _sdl_series = _sdl.get("signal_series")
            if _sdl_series is not None and len(_sdl_series) > 0:
                _sdl_fig = _go_sdl.Figure()
                _sdl_fig.add_trace(_go_sdl.Scatter(
                    x=list(_sdl_series.index), y=list(_sdl_series.values),
                    mode="lines", name="Deleveraging Score",
                    line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
                ))
                for _thresh, _col, _lbl in [(30, "#f39c12", "Elevated"), (60, "#e67e22", "Warning"), (80, "#e74c3c", "Alert")]:
                    _sdl_fig.add_hline(y=_thresh, line_color=_col, line_width=1, line_dash="dot",
                                       annotation_text=_lbl, annotation_position="right")
                _sdl_fig.update_layout(
                    height=270, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Deleveraging Score (0-100)", range=[0, 100], showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_sdl_fig, use_container_width=True)

            # Peak events
            _sdl_peaks = _sdl.get("peak_events", [])
            if _sdl_peaks:
                with st.expander("Top Historical Deleveraging Events"):
                    _peak_rows = [{"Date": p.get("date", "—"), "Score": f"{p.get('composite', 0):.1f}",
                                   "Alert": p.get("alert_level", "—")} for p in _sdl_peaks]
                    st.dataframe(pd.DataFrame(_peak_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Systematic deleveraging detector unavailable — requires ≥2 asset columns and ≥126 rows.")
    except Exception as _sdl_e:
        st.caption(f"Systematic deleveraging unavailable: {_sdl_e}")

# =============================================================================
# ANALYTICS sub-tab 45: Inflation Regime Monitor
# =============================================================================
with _analytics_sub45:
    import plotly.graph_objects as _go_infl
    st.header("Inflation Regime Monitor")
    st.markdown(
        "Decomposes the 10y nominal yield into **real rate + breakeven inflation** and maps today "
        "onto the four-quadrant framework: Stagflation / Tightening / Reflation / Deflation Risk. "
        "Different inflation regimes produce dramatically different credit spread outcomes."
    )
    try:
        _infl = load_inflation_regime(df)
        if _infl.get("available"):
            _infl_cur = _infl.get("current", {})
            _infl_quad = _infl_cur.get("quadrant", "—")
            _infl_outlook = _infl_cur.get("credit_outlook", "—")
            _infl_real = _infl_cur.get("real_rate")
            _infl_be = _infl_cur.get("breakeven")
            _infl_nom = _infl_cur.get("nominal_yield")
            _infl_interp = _infl_cur.get("interpretation", "") or _infl.get("interpretation", "")

            _infl_quad_color = {
                "Reflation": "#27ae60", "Tightening": "#f39c12",
                "Stagflation": "#e74c3c", "Deflation Risk": "#3498db",
            }.get(_infl_quad, "#9aa0aa")
            _infl_outlook_color = {"Bullish": "#27ae60", "Cautious": "#f39c12",
                                    "Bearish": "#e74c3c", "Mixed": "#9aa0aa"}.get(_infl_outlook, "#9aa0aa")

            _if1, _if2, _if3, _if4 = st.columns(4)
            _if1.metric("Inflation Regime", _infl_quad)
            _if2.metric("Credit Outlook", _infl_outlook)
            _if3.metric("Real Rate", f"{_infl_real:.2f}%" if _infl_real is not None else "—",
                        delta=f"{_infl_cur.get('d_real_63d', 0):+.2f}% (63d)" if _infl_cur.get('d_real_63d') is not None else None)
            _if4.metric("Breakeven", f"{_infl_be:.2f}%" if _infl_be is not None else "—",
                        delta=f"{_infl_cur.get('d_breakeven_63d', 0):+.2f}% (63d)" if _infl_cur.get('d_breakeven_63d') is not None else None)

            if _infl_interp:
                st.info(_infl_interp)

            # Credit return by regime
            _infl_cbr = _infl.get("credit_by_regime", {})
            if _infl_cbr:
                _cbr_rows = [
                    {"Regime": q,
                     "Mean HY Δ (63d)": f"{v.get('mean_hy_change', 0):+.0f}bps" if v.get('mean_hy_change') is not None else "—",
                     "Widening Rate": f"{v.get('hit_rate', 0):.0%}" if v.get('hit_rate') is not None else "—",
                     "N Obs": v.get("n_obs", 0)}
                    for q, v in _infl_cbr.items()
                ]
                st.dataframe(pd.DataFrame(_cbr_rows), use_container_width=True, hide_index=True)
                st.caption("Widening Rate = % of periods where HY spreads widened over next 63 days in this regime")

            # Historical real rate & breakeven chart
            _infl_hist = _infl.get("historical")
            if _infl_hist is not None and "infl_real_rate" in _infl_hist.columns:
                with st.expander("Historical Real Rate & Breakeven (2yr)"):
                    _infl_fig = _go_infl.Figure()
                    _infl_fig.add_trace(_go_infl.Scatter(
                        x=_infl_hist.index, y=_infl_hist["infl_real_rate"],
                        name="Real Rate", mode="lines",
                        line=dict(color="#3498db", width=2),
                    ))
                    if "infl_breakeven" in _infl_hist.columns:
                        _infl_fig.add_trace(_go_infl.Scatter(
                            x=_infl_hist.index, y=_infl_hist["infl_breakeven"],
                            name="Breakeven Inflation", mode="lines",
                            line=dict(color="#e74c3c", width=2),
                        ))
                    _infl_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                    _infl_fig.update_layout(
                        height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        yaxis=dict(title="Rate (%)", showgrid=True,
                                   gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    )
                    st.plotly_chart(_infl_fig, use_container_width=True)
        else:
            st.info("Inflation regime unavailable — requires yield_10y and ≥126 rows.")
    except Exception as _infl_e:
        st.caption(f"Inflation regime unavailable: {_infl_e}")

# =============================================================================
# ANALYTICS sub-tab 46: Sector ETF Stress Divergence
# =============================================================================
with _analytics_sub46:
    import plotly.graph_objects as _go_sec
    st.header("Sector ETF Stress Divergence")
    st.markdown(
        "Live sector ETF performance (XLF, XLE, XBI, XLU, XHB, XLI, XLK, XLP) relative to SPY. "
        "**Financials leading down** by 3–4 weeks is historically the most reliable early warning for credit spreads. "
        "≥4 sectors simultaneously leading down = systematic stress."
    )
    try:
        _sec = load_sector_divergence(df)
        if _sec.get("available"):
            _sec_snap = _sec.get("snapshot", {})
            _sec_lead = _sec.get("lead_signal", "—")
            _sec_impl = _sec.get("credit_implication", "")
            _sec_interp = _sec.get("interpretation", "")

            if _sec_snap.get("available"):
                _sec_n_down = _sec_snap.get("n_leading_down", 0)
                _sec_n_up = _sec_snap.get("n_leading_up", 0)
                _sec_stress = _sec_snap.get("stress_sectors", [])
                _sec_xlf = _sec_snap.get("financials_signal", "—")
                _sec_sys = _sec_snap.get("systematic_stress", False)

                _se1, _se2, _se3, _se4 = st.columns(4)
                _se1.metric("Lead Signal", _sec_lead)
                _se2.metric("Financials (XLF)", _sec_xlf)
                _se3.metric("Sectors Leading Down", _sec_n_down,
                            delta="systematic stress" if _sec_sys else None,
                            delta_color="inverse")
                _se4.metric("SPY 1M Return", f"{_sec_snap.get('spy_return_1m', 0):+.1f}%")

                if _sec_sys:
                    st.warning(f"Systematic stress: {_sec_n_down} sectors leading down simultaneously. Stress sectors: {', '.join(_sec_stress)}")
                elif _sec_stress:
                    st.warning(f"Stress sectors: **{', '.join(_sec_stress)}**")

                if _sec_impl:
                    st.info(_sec_impl)

                # Sector table
                _sec_sectors = _sec_snap.get("sectors", [])
                if _sec_sectors:
                    _sec_rows = [
                        {"Sector": s.get("name"), "1M Return": f"{s.get('return_1m', 0):+.1f}%",
                         "3M Return": f"{s.get('return_3m', 0):+.1f}%",
                         "Rel Perf 1M": f"{s.get('rel_perf_1m', 0):+.1f}%",
                         "Z-Score": f"{s.get('z_score_1m', 0):.2f}",
                         "Signal": s.get("signal", "—")}
                        for s in _sec_sectors
                    ]
                    st.dataframe(pd.DataFrame(_sec_rows), use_container_width=True, hide_index=True)
                    st.caption(f"Live data via yfinance · As of: {_sec_snap.get('as_of', '—')}")
            else:
                if _sec_interp:
                    st.info(_sec_interp)
                else:
                    st.info("Live sector data unavailable — check network connection.")
        else:
            st.info("Sector divergence unavailable — requires yfinance connection.")
    except Exception as _sec_e:
        st.caption(f"Sector divergence unavailable: {_sec_e}")

# =============================================================================
# ANALYTICS sub-tab 47: Put/Call Ratio & Sentiment
# =============================================================================
with _analytics_sub47:
    import plotly.graph_objects as _go_pcs
    st.header("Put/Call Ratio & Sentiment Composite")
    st.markdown(
        "**Contrarian sentiment signal**: extreme put buying = crowded short = squeeze risk. "
        "Composite = 60% put/call score + 40% VIX percentile. "
        "Score > 70 = extreme fear → contrarian bullish for credit. Score < 30 = complacency → bearish."
    )
    try:
        _pcs = load_put_call_sentiment(df)
        if _pcs.get("available"):
            _pcs_snap = _pcs.get("snapshot", {})
            _pcs_contra = _pcs.get("contrarian_signal", "—")
            _pcs_interp = _pcs.get("interpretation", "")

            _pcs_score = _pcs_snap.get("sentiment_composite") if _pcs_snap.get("available") else None
            _pcs_sig = _pcs_snap.get("sentiment_signal", "—")
            _pcs_pc = _pcs_snap.get("pc_ratio")
            _pcs_src = _pcs_snap.get("pc_source", "—")
            _pcs_vix = _pcs_snap.get("vix")

            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.metric("Sentiment Score", f"{_pcs_score:.1f}" if _pcs_score is not None else "—",
                        help="0=extreme complacency, 100=extreme fear")
            _pc2.metric("Signal", _pcs_sig)
            _pc3.metric("P/C Ratio", f"{_pcs_pc:.2f}" if _pcs_pc is not None else "—",
                        help=_pcs_src)
            _pc4.metric("Contrarian View", _pcs_contra)

            if _pcs_interp:
                st.info(_pcs_interp)

            st.caption(f"P/C Source: {_pcs_src} · As of: {_pcs_snap.get('as_of', '—')}")

            # Historical sentiment chart
            _pcs_series = _pcs.get("signal_series")
            if _pcs_series is not None and len(_pcs_series) > 0:
                _pcs_fig = _go_pcs.Figure()
                _pcs_fig.add_trace(_go_pcs.Scatter(
                    x=list(_pcs_series.index), y=list(_pcs_series.values),
                    mode="lines", name="Sentiment Score",
                    line=dict(color="#9b59b6", width=2),
                    fill="tozeroy", fillcolor="rgba(155,89,182,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.1f}<extra></extra>",
                ))
                _pcs_fig.add_hline(y=70, line_color="#27ae60", line_width=1, line_dash="dot",
                                   annotation_text="Extreme Fear (Contrarian Bullish)")
                _pcs_fig.add_hline(y=30, line_color="#e74c3c", line_width=1, line_dash="dot",
                                   annotation_text="Complacency (Contrarian Bearish)")
                _pcs_fig.update_layout(
                    height=270, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Sentiment Score (0-100)", range=[0, 100], showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_pcs_fig, use_container_width=True)
                st.caption("Higher = more fear = contrarian bullish · Composite uses VIX percentile as fallback when P/C unavailable")
        else:
            st.info("Put/call sentiment unavailable — requires VIX data.")
    except Exception as _pcs_e:
        st.caption(f"Put/call sentiment unavailable: {_pcs_e}")

# =============================================================================
# ANALYTICS sub-tab 48: Credit Basis Monitor
# =============================================================================
with _analytics_sub48:
    import plotly.graph_objects as _go_cb
    st.header("Credit Basis Monitor")
    st.markdown(
        "**CDX vs cash bond basis**: when cash bonds cheapen relative to CDS (negative basis), "
        "it signals forced cash bond selling — real-money deleveraging, not just hedging. "
        "The most reliable institutional stress signal in credit markets."
    )
    try:
        _cb = load_credit_basis(df)
        if _cb.get("available"):
            _cb_snap = _cb.get("snapshot", {})
            _cb_signal = _cb.get("basis_signal", "—")
            _cb_stress = _cb.get("stress_flag", False)
            _cb_basis = _cb.get("current_basis")
            _cb_pct = _cb.get("percentile_current")
            _cb_warn = _cb.get("warning")
            _cb_interp = _cb.get("interpretation", "")

            _cb_signal_color = {
                "Negative Basis (Forced Selling)": "#e74c3c",
                "Compressed Basis": "#f39c12",
                "Normal": "#27ae60",
                "Rich Basis": "#3498db",
            }.get(_cb_signal, "#9aa0aa")

            _cb1, _cb2, _cb3, _cb4 = st.columns(4)
            _cb1.metric("Basis Signal", _cb_signal)
            _cb2.metric("Current Basis", f"{_cb_basis:+.1f}bps" if _cb_basis is not None else "—")
            _cb3.metric("Stress Flag", "⚠ ACTIVE" if _cb_stress else "Clear")
            _cb4.metric("Historical Percentile", f"{_cb_pct:.0f}th" if _cb_pct is not None else "—")

            if _cb_warn:
                st.warning(_cb_warn)
            elif _cb_interp:
                st.info(_cb_interp)

            if _cb_snap.get("available"):
                st.caption(
                    f"HYG: ${_cb_snap.get('hyg_price', 0):.2f} · "
                    f"JNK: ${_cb_snap.get('jnk_price', 0):.2f} · "
                    f"As of: {_cb_snap.get('as_of', '—')}"
                )

            # Historical basis chart
            _cb_series = _cb.get("signal_series")
            if _cb_series is not None and len(_cb_series) > 0:
                _cb_fig = _go_cb.Figure()
                _cb_colors = ["#e74c3c" if v < -5 else "#27ae60" for v in _cb_series.values]
                _cb_fig.add_trace(_go_cb.Bar(
                    x=list(_cb_series.index), y=list(_cb_series.values),
                    name="Credit Basis", marker_color=_cb_colors,
                    hovertemplate="%{x|%Y-%m-%d}<br>Basis: %{y:+.1f}bps<extra></extra>",
                ))
                _cb_fig.add_hline(y=-20, line_color="#e74c3c", line_width=1, line_dash="dot",
                                  annotation_text="Forced Selling")
                _cb_fig.add_hline(y=15, line_color="#3498db", line_width=1, line_dash="dot",
                                  annotation_text="Rich Basis")
                _cb_fig.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Basis (bps equivalent)", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_cb_fig, use_container_width=True)
                st.caption("Negative basis = cash cheaper than synthetic = forced selling signal · Uses ETF proxy when CDX data unavailable")
        else:
            st.info("Credit basis unavailable — requires yfinance or CDX/spread data.")
    except Exception as _cb_e:
        st.caption(f"Credit basis unavailable: {_cb_e}")

# =============================================================================
# ANALYTICS sub-tab 49: Drawdown Recovery Analyzer
# =============================================================================
with _analytics_sub49:
    import plotly.graph_objects as _go_ddr
    st.header("Drawdown Recovery Analyzer")
    st.markdown(
        "For any ongoing drawdown in HY spreads or the composite risk score, "
        "shows current depth vs all historical episodes and the **empirical recovery time distribution**. "
        "Answers: *how long until this typically normalizes?*"
    )
    try:
        _ddr = load_drawdown_recovery(df)
        if _ddr.get("available"):
            _ddr_primary = _ddr.get("primary_asset", "HY Spread")
            _ddr_depth = _ddr.get("current_depth")
            _ddr_rec = _ddr.get("recovery_estimate", {})
            _ddr_warn = _ddr.get("warning")
            _ddr_interp = _ddr.get("interpretation", "")

            _ddr_assets = _ddr.get("assets", {})
            _primary_cur = _ddr_assets.get(_ddr_primary, {}).get("current", {})
            _in_dd = _primary_cur.get("in_drawdown", False)

            _dd1, _dd2, _dd3, _dd4 = st.columns(4)
            _dd1.metric("Primary Asset", _ddr_primary)
            _dd2.metric("In Drawdown?", "Yes" if _in_dd else "No",
                        delta=f"{_ddr_depth:+.1f} bps/pts" if _ddr_depth else None,
                        delta_color="inverse")
            _dd3.metric("Median Recovery", f"{_ddr_rec.get('median_recovery_days', 0):.0f}d"
                        if _ddr_rec.get('median_recovery_days') is not None else "—")
            _dd4.metric("Comparable Episodes", _ddr_rec.get("n_comparable", 0))

            if _ddr_warn:
                st.warning(_ddr_warn)
            elif _ddr_interp:
                st.info(_ddr_interp)

            # Recovery time distribution
            if _ddr_rec.get("n_comparable", 0) > 0:
                _ddr_p25 = _ddr_rec.get("p25_recovery_days")
                _ddr_p50 = _ddr_rec.get("median_recovery_days")
                _ddr_p75 = _ddr_rec.get("p75_recovery_days")
                _ddr_pct_rec = _ddr_rec.get("pct_recovered", 0)

                _rec_cols = st.columns(4)
                _rec_cols[0].metric("P25 Recovery", f"{_ddr_p25:.0f}d" if _ddr_p25 else "—")
                _rec_cols[1].metric("Median Recovery", f"{_ddr_p50:.0f}d" if _ddr_p50 else "—")
                _rec_cols[2].metric("P75 Recovery", f"{_ddr_p75:.0f}d" if _ddr_p75 else "—")
                _rec_cols[3].metric("Eventually Recovered", f"{_ddr_pct_rec:.0%}")

            # Historical episodes table
            _ddr_episodes = _ddr.get("all_episodes")
            if _ddr_episodes is not None and not _ddr_episodes.empty:
                with st.expander(f"All Historical Drawdown Episodes ({len(_ddr_episodes)} total)"):
                    _ep_disp = _ddr_episodes.copy()
                    for _col in ["depth"]:
                        if _col in _ep_disp.columns:
                            _ep_disp[_col] = _ep_disp[_col].apply(lambda x: f"{x:+.1f}" if x is not None and not pd.isna(x) else "—")
                    if "recovery_days" in _ep_disp.columns:
                        _ep_disp["recovery_days"] = _ep_disp["recovery_days"].apply(
                            lambda x: f"{x:.0f}d" if x is not None and not pd.isna(x) else "Not recovered"
                        )
                    st.dataframe(_ep_disp, use_container_width=True, hide_index=True)

            st.caption("Recovery = return within 10% of pre-drawdown level · Depth in bps for spreads, score points for composite")
        else:
            st.info("Drawdown recovery unavailable — requires HY spread or composite score data with ≥252 rows.")
    except Exception as _ddr_e:
        st.caption(f"Drawdown recovery unavailable: {_ddr_e}")

# =============================================================================
# ANALYTICS sub-tab 50: Signal Move Attribution
# =============================================================================
with _analytics_sub50:
    import plotly.graph_objects as _go_sma
    st.header("Signal Move Attribution")
    st.markdown(
        "**What drove the composite score change?** Brinson-style decomposition: "
        "which sub-signal moved most over the past 1M / 3M / 6M, "
        "and which market variable was the primary cause."
    )
    try:
        _sma = load_signal_move_attribution(df)
        if _sma.get("available"):
            _sma_interp = _sma.get("interpretation", "")
            _sma_comp = _sma.get("current_composite")
            _sma_regime = _sma.get("current_regime", "—")

            _sm1, _sm2 = st.columns(2)
            _sm1.metric("Current Composite", f"{_sma_comp:.1f}" if _sma_comp is not None else "—")
            _sm2.metric("Current Regime", _sma_regime)

            if _sma_interp:
                st.info(_sma_interp)

            # Waterfall-style bar chart for 1M attribution
            _sma_attrs = _sma.get("attributions", {})
            _sma_1m = _sma_attrs.get("1M", {})
            if _sma_1m.get("available"):
                _sma_contribs = _sma_1m.get("sub_signal_contributions", [])
                _sma_delta = _sma_1m.get("composite_delta", 0)

                _sma_c1, _sma_c2, _sma_c3 = st.columns(3)
                _sma_c1.metric("1M Composite Δ", f"{_sma_delta:+.1f}",
                               delta_color="inverse" if _sma_delta > 0 else "normal")
                _sma_c2.metric("Dominant Signal", _sma_1m.get("dominant_sub_signal", "—"))
                _sma_c3.metric("Dominant Driver", _sma_1m.get("dominant_market_driver", "—"))

                if _sma_contribs:
                    _sma_fig = _go_sma.Figure()
                    _sma_labels = [c["label"] for c in _sma_contribs]
                    _sma_pcts = [c.get("contribution_pct", 0) for c in _sma_contribs]
                    _sma_bar_colors = ["#e74c3c" if v > 0 else "#27ae60" for v in _sma_pcts]
                    _sma_fig.add_trace(_go_sma.Bar(
                        x=_sma_labels, y=_sma_pcts,
                        marker_color=_sma_bar_colors,
                        hovertemplate="%{x}<br>Contribution: %{y:+.1f}%<extra></extra>",
                    ))
                    _sma_fig.update_layout(
                        height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                        title=dict(text="1M Contribution to Composite Move (%)", font=dict(size=12, color="#9aa0aa")),
                        yaxis=dict(title="Contribution (%)", showgrid=True,
                                   gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                    )
                    st.plotly_chart(_sma_fig, use_container_width=True)

            # Summary table across all windows
            _sma_tbl = _sma.get("summary_table")
            if _sma_tbl is not None and not _sma_tbl.empty:
                with st.expander("Full attribution across 1M / 3M / 6M"):
                    st.dataframe(_sma_tbl, use_container_width=True)
        else:
            st.info("Signal move attribution unavailable — requires composite score, ≥2 sub-signals, and ≥126 rows.")
    except Exception as _sma_e:
        st.caption(f"Signal move attribution unavailable: {_sma_e}")

# =============================================================================
# ANALYTICS sub-tab 51: Risk Parity Credit Allocation
# =============================================================================
with _analytics_sub51:
    import plotly.graph_objects as _go_rpa
    st.header("Risk Parity Credit Allocation")
    st.markdown(
        "**Equal Risk Contribution (ERC)** weights across IG / HY / Equities / Rates / USD. "
        "Shows how much of each asset to hold so each contributes equally to portfolio volatility. "
        "Regime-adjusted variant scales ERC weights for current market regime."
    )
    try:
        _rpa = load_risk_parity_allocation(df)
        if _rpa.get("available"):
            _rpa_regime = _rpa.get("current_regime", "—")
            _rpa_vol_erc = _rpa.get("portfolio_vol_erc")
            _rpa_vol_eq = _rpa.get("portfolio_vol_equal")
            _rpa_interp = _rpa.get("interpretation", "")
            _rpa_assets = _rpa.get("assets_used", [])

            _rp1, _rp2, _rp3 = st.columns(3)
            _rp1.metric("Current Regime", _rpa_regime)
            _rp2.metric("ERC Portfolio Vol", f"{_rpa_vol_erc:.1%}" if _rpa_vol_erc else "—",
                        delta=f"{(_rpa_vol_erc - _rpa_vol_eq):+.1%} vs equal-weight" if _rpa_vol_erc and _rpa_vol_eq else None)
            _rp3.metric("Assets Used", len(_rpa_assets))

            if _rpa_interp:
                st.info(_rpa_interp)

            _rpa_tbl = _rpa.get("weights_table")
            if _rpa_tbl is not None and not _rpa_tbl.empty:
                st.dataframe(_rpa_tbl, use_container_width=True)

            # ERC vs equal-weight bar chart
            _rpa_erc_w = _rpa.get("erc_weights", {})
            _rpa_eq_w = _rpa.get("equal_weights", {})
            _rpa_reg_w = _rpa.get("regime_weights", {})
            if _rpa_erc_w:
                _rpa_fig = _go_rpa.Figure()
                _rpa_keys = list(_rpa_erc_w.keys())
                _rpa_fig.add_trace(_go_rpa.Bar(name="ERC", x=_rpa_keys,
                    y=[_rpa_erc_w.get(k, 0) for k in _rpa_keys], marker_color="#3498db"))
                _rpa_fig.add_trace(_go_rpa.Bar(name="Equal Weight", x=_rpa_keys,
                    y=[_rpa_eq_w.get(k, 0) for k in _rpa_keys], marker_color="#9aa0aa"))
                _rpa_fig.add_trace(_go_rpa.Bar(name=f"Regime-Adj ({_rpa_regime})", x=_rpa_keys,
                    y=[_rpa_reg_w.get(k, 0) for k in _rpa_keys], marker_color="#e74c3c"))
                _rpa_fig.update_layout(
                    barmode="group", height=260,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Weight", tickformat=".0%", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                )
                st.plotly_chart(_rpa_fig, use_container_width=True)
        else:
            st.info("Risk parity allocation unavailable — requires ≥2 asset proxies and ≥252 rows.")
    except Exception as _rpa_e:
        st.caption(f"Risk parity allocation unavailable: {_rpa_e}")

# =============================================================================
# ANALYTICS sub-tab 52: Tail Dependency Matrix
# =============================================================================
with _analytics_sub52:
    import plotly.graph_objects as _go_td
    st.header("Tail Dependency Matrix")
    st.markdown(
        "Empirical joint tail probability for all asset pairs. "
        "**Tail dependency coefficient > 1** = assets cluster together in stress (diversification fails). "
        "**< 1** = assets actually diversify during extreme moves. Pure empirical — no copula assumptions."
    )
    try:
        _td = load_tail_dependency(df)
        if _td.get("available"):
            _td_matrix = _td.get("matrix", {})
            _td_warn = _td.get("warning")
            _td_interp = _td.get("interpretation", "")
            _td_findings = _td.get("key_findings", [])

            _td_high = _td_matrix.get("highest_dependency")
            _td_low = _td_matrix.get("lowest_dependency")

            _tdc1, _tdc2, _tdc3 = st.columns(3)
            if _td_high:
                _tdc1.metric("Most Clustered Pair", f"{_td_high[0]} / {_td_high[1]}",
                             delta=f"TDC = {_td_high[2]:.1f}x", delta_color="inverse")
            if _td_low:
                _tdc2.metric("Best Diversifier Pair", f"{_td_low[0]} / {_td_low[1]}",
                             delta=f"TDC = {_td_low[2]:.1f}x", delta_color="normal")
            _tdc3.metric("Assets Analyzed", _td_matrix.get("n_assets", 0))

            if _td_warn:
                st.warning(_td_warn)
            if _td_interp:
                st.info(_td_interp)

            for _f in _td_findings:
                st.caption(f"• {_f}")

            # Tail dependency heatmap
            _td_tbl = _td_matrix.get("tail_dep_matrix")
            if _td_tbl is not None and not _td_tbl.empty:
                _td_fig = _go_td.Figure(data=_go_td.Heatmap(
                    z=_td_tbl.values.tolist(),
                    x=list(_td_tbl.columns),
                    y=list(_td_tbl.index),
                    colorscale="RdYlGn_r",
                    zmid=1.0,
                    hovertemplate="%{y} | %{x}<br>TDC: %{z:.2f}x<extra></extra>",
                    text=[[f"{v:.2f}x" for v in row] for row in _td_tbl.values],
                    texttemplate="%{text}",
                ))
                _td_fig.update_layout(
                    height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    title=dict(text="Tail Dependency Coefficient (1.0 = independent)", font=dict(size=11, color="#9aa0aa")),
                )
                st.plotly_chart(_td_fig, use_container_width=True)
                st.caption("Red = clusters in stress · Green = diversifies · 10th/90th percentile tails · Values > 1 = positive tail dependency")

            # Stress vs unconditional correlation
            _td_stress = _td_matrix.get("stress_corr_matrix")
            _td_uncond = _td_matrix.get("unconditional_corr_matrix")
            if _td_stress is not None and _td_uncond is not None:
                with st.expander("Stress Correlation vs Unconditional Correlation"):
                    _tc1, _tc2 = st.columns(2)
                    _tc1.write("**Stress Correlation** (when either asset is in tail)")
                    _tc1.dataframe(_td_stress.applymap(lambda x: f"{x:.2f}"), use_container_width=True)
                    _tc2.write("**Unconditional Correlation**")
                    _tc2.dataframe(_td_uncond.applymap(lambda x: f"{x:.2f}"), use_container_width=True)
        else:
            st.info("Tail dependency unavailable — requires ≥3 asset columns and ≥252 rows.")
    except Exception as _td_e:
        st.caption(f"Tail dependency unavailable: {_td_e}")

# =============================================================================
# ANALYTICS sub-tab 53: Fed Liquidity Plumbing Monitor
# =============================================================================
with _analytics_sub53:
    import plotly.graph_objects as _go_fl
    st.header("Fed Liquidity Plumbing Monitor")
    st.markdown(
        "Fed balance sheet total assets, RRP outstanding, SOFR, and bank reserves — "
        "the **upstream liquidity variables** that lead credit spreads by ~6 weeks. "
        "QT (balance sheet shrinkage) tightens credit conditions before spread moves are visible."
    )
    try:
        _fl = load_fed_liquidity(df)
        if _fl.get("available"):
            _fl_metrics = _fl.get("metrics", {})
            _fl_regime = _fl_metrics.get("fed_assets_regime", "—")
            _fl_signal = _fl_metrics.get("liquidity_composite_signal", "—")
            _fl_credit = _fl.get("credit_implication", "")
            _fl_interp = _fl.get("interpretation", "")
            _fl_lead = _fl.get("lead_weeks", 6)

            _fl_signal_color = {
                "Expanding": "#27ae60", "Neutral": "#f39c12", "Contracting": "#e74c3c"
            }.get(_fl_signal, "#9aa0aa")

            _fc1, _fc2, _fc3, _fc4 = st.columns(4)
            _fc1.metric("Balance Sheet Regime", _fl_regime)
            _fc2.metric("Liquidity Signal", _fl_signal)
            _fc3.metric("Fed Assets", f"${_fl_metrics.get('fed_assets_latest', 0)/1000:.1f}T"
                        if _fl_metrics.get('fed_assets_latest') else "—",
                        delta=f"{_fl_metrics.get('fed_assets_13w_chg_pct', 0):+.1f}% (13w)" if _fl_metrics.get('fed_assets_13w_chg_pct') is not None else None,
                        delta_color="normal")
            _fc4.metric("Lead Time", f"~{_fl_lead} weeks")

            if _fl_credit:
                st.info(_fl_credit)

            # Key metrics
            _fl_rrp = _fl_metrics.get("rrp_latest")
            _fl_res = _fl_metrics.get("reserves_latest")
            _fl_sofr = _fl_metrics.get("sofr_latest")
            _fl_net = _fl_metrics.get("net_liquidity")

            if any(x is not None for x in [_fl_rrp, _fl_res, _fl_sofr]):
                _fm1, _fm2, _fm3, _fm4 = st.columns(4)
                _fm1.metric("RRP Outstanding", f"${_fl_rrp/1000:.1f}T" if _fl_rrp else "—",
                            delta=f"{_fl_metrics.get('rrp_4w_chg', 0)/1000:+.1f}T (4w)" if _fl_metrics.get('rrp_4w_chg') else None)
                _fm2.metric("Bank Reserves", f"${_fl_res/1000:.1f}T" if _fl_res else "—")
                _fm3.metric("SOFR", f"{_fl_sofr:.2f}%" if _fl_sofr else "—")
                _fm4.metric("Net Liquidity", f"${_fl_net/1000:.1f}T" if _fl_net else "—",
                            help="Fed assets minus RRP minus reserves = deployed liquidity")

            # Historical chart
            _fl_hist = _fl.get("historical", {})
            _fl_assets_hist = _fl_hist.get("fed_assets")
            if _fl_assets_hist is not None and len(_fl_assets_hist) > 0:
                _fl_fig = _go_fl.Figure()
                _fl_fig.add_trace(_go_fl.Scatter(
                    x=list(_fl_assets_hist.index), y=list(_fl_assets_hist.values / 1000),
                    name="Fed Assets ($T)", mode="lines",
                    line=dict(color="#3498db", width=2),
                ))
                _fl_rrp_hist = _fl_hist.get("rrp")
                if _fl_rrp_hist is not None:
                    _fl_fig.add_trace(_go_fl.Scatter(
                        x=list(_fl_rrp_hist.index), y=list(_fl_rrp_hist.values / 1000),
                        name="RRP ($T)", mode="lines",
                        line=dict(color="#e74c3c", width=1.5, dash="dot"),
                    ))
                _fl_fig.update_layout(
                    height=270, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="$ Trillion", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                )
                st.plotly_chart(_fl_fig, use_container_width=True)
                st.caption(f"Source: {_fl.get('fed_data', {}).get('source', 'FRED')} · Data weekly · Lead time ~{_fl_lead} weeks ahead of credit spreads")
        else:
            st.info("Fed liquidity monitor unavailable — requires FRED API key or Fed balance sheet columns in data.")
    except Exception as _fl_e:
        st.caption(f"Fed liquidity unavailable: {_fl_e}")

# =============================================================================
# ANALYTICS sub-tab 54: G4 Central Bank Divergence
# =============================================================================
with _analytics_sub54:
    import plotly.graph_objects as _go_g4
    st.header("G4 Central Bank Divergence")
    st.markdown(
        "Fed vs ECB vs BOE vs BOJ policy rates and 10y yields. "
        "Wide Fed-ECB spread = strong USD = EM credit headwind. "
        "Wide Fed-BOJ spread = JPY carry trade active = leverage building in risk assets."
    )
    try:
        _g4 = load_g4_divergence(df)
        if _g4.get("available"):
            _g4_rates = _g4.get("policy_rates", {})
            _g4_live = _g4.get("live_rates", {})
            _g4_metrics = _g4.get("metrics", {})
            _g4_interp = _g4.get("interpretation", "")
            _g4_source = _g4_rates.get("source", "—")

            _g4c1, _g4c2, _g4c3, _g4c4 = st.columns(4)
            _g4c1.metric("Fed", f"{_g4_rates.get('fed', 0):.2f}%")
            _g4c2.metric("ECB", f"{_g4_rates.get('ecb', 0):.2f}%",
                         delta=f"{_g4_metrics.get('fed_ecb_spread', 0):+.2f}% vs Fed")
            _g4c3.metric("BOE", f"{_g4_rates.get('boe', 0):.2f}%",
                         delta=f"{_g4_metrics.get('fed_boe_spread', 0):+.2f}% vs Fed")
            _g4c4.metric("BOJ", f"{_g4_rates.get('boj', 0):.2f}%",
                         delta=f"{_g4_metrics.get('fed_boj_spread', 0):+.2f}% vs Fed")

            _g4m1, _g4m2, _g4m3 = st.columns(3)
            _g4m1.metric("Divergence Regime", _g4_metrics.get("divergence_regime", "—"))
            _g4m2.metric("JPY Carry Pressure", _g4_metrics.get("carry_pressure", "—"))
            _g4m3.metric("EM Credit Signal", _g4_metrics.get("em_credit_signal", "—"))

            if _g4_interp:
                st.info(_g4_interp)

            # Rate table
            _g4_tbl = _g4.get("rate_table")
            if _g4_tbl is not None and not _g4_tbl.empty:
                st.dataframe(_g4_tbl, use_container_width=True, hide_index=True)
                st.caption(f"Policy rate source: {_g4_source}")

            # Live FX rates
            if _g4_live.get("available"):
                st.subheader("Live FX Rates")
                _fx1, _fx2, _fx3 = st.columns(3)
                _fx1.metric("EUR/USD", f"{_g4_live.get('eurusd', 0):.4f}" if _g4_live.get('eurusd') else "—")
                _fx2.metric("GBP/USD", f"{_g4_live.get('gbpusd', 0):.4f}" if _g4_live.get('gbpusd') else "—")
                _fx3.metric("USD/JPY", f"{_g4_live.get('usdjpy', 0):.2f}" if _g4_live.get('usdjpy') else "—")
                st.caption(f"Live via yfinance · As of: {_g4_live.get('as_of', '—')}")

            # Historical Fed rate from df
            _g4_fed_hist = _g4.get("historical_fed_rate")
            if _g4_fed_hist is not None and len(_g4_fed_hist) > 0:
                with st.expander("Historical Fed Funds Rate"):
                    _g4_fig = _go_g4.Figure()
                    _g4_fig.add_trace(_go_g4.Scatter(
                        x=list(_g4_fed_hist.index), y=list(_g4_fed_hist.values),
                        name="Fed Funds Rate", mode="lines",
                        line=dict(color="#3498db", width=2),
                    ))
                    _g4_fig.update_layout(
                        height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                        yaxis=dict(title="Rate (%)", showgrid=True,
                                   gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                    )
                    st.plotly_chart(_g4_fig, use_container_width=True)
        else:
            st.info("G4 divergence unavailable.")
    except Exception as _g4_e:
        st.caption(f"G4 divergence unavailable: {_g4_e}")

# =============================================================================
# ANALYTICS sub-tab 55: Portfolio Stress Test
# =============================================================================
with _analytics_sub55:
    import plotly.graph_objects as _go_pst
    st.header("Portfolio Stress Test")
    st.markdown(
        "Define your portfolio allocation and apply the 5 market shock scenarios. "
        "Shows dollar P&L per $100 invested, broken down by asset class. "
        "Rates and spread durations are fixed assumptions — see captions for details."
    )
    try:
        # Portfolio sliders
        st.subheader("Portfolio Weights")
        st.caption("Weights are normalized to sum to 100%")
        _pst_c = st.columns(5)
        _pst_weights_raw = {}
        _pst_defaults = {"ig": 40, "hy": 30, "loans": 10, "em": 10, "cash": 10}
        _pst_asset_labels = {"ig": "Inv. Grade", "hy": "High Yield", "loans": "Loans", "em": "EM Credit", "cash": "Cash"}
        for _pi, (_pkey, _plabel) in enumerate(_pst_asset_labels.items()):
            _pst_weights_raw[_pkey] = _pst_c[_pi].slider(
                _plabel, 0, 100, _pst_defaults[_pkey], 5, key=f"pst_{_pkey}"
            )

        _pst_total = sum(_pst_weights_raw.values())
        if _pst_total > 0:
            _pst_weights = {k: v / _pst_total for k, v in _pst_weights_raw.items()}
        else:
            _pst_weights = {k: v / 100 for k, v in _pst_defaults.items()}

        # Run stress test
        try:
            _pst = run_portfolio_stress_test(df, portfolio=_pst_weights)
        except Exception:
            _pst = load_portfolio_stress_test(df)

        if _pst.get("available"):
            _pst_worst = _pst.get("worst_scenario", "—")
            _pst_worst_pl = _pst.get("worst_scenario_pl")
            _pst_best = _pst.get("best_scenario", "—")
            _pst_best_pl = _pst.get("best_scenario_pl")
            _pst_regime = _pst.get("current_regime", "—")
            _pst_ctx = _pst.get("regime_context", "")

            _ps1, _ps2, _ps3 = st.columns(3)
            _ps1.metric("Current Regime", _pst_regime)
            _ps2.metric("Worst Scenario", _pst_worst,
                        delta=f"{_pst_worst_pl:+.1f}% P&L" if _pst_worst_pl is not None else None,
                        delta_color="inverse")
            _ps3.metric("Best Scenario", _pst_best,
                        delta=f"{_pst_best_pl:+.1f}% P&L" if _pst_best_pl is not None else None,
                        delta_color="normal")

            if _pst_ctx:
                st.info(_pst_ctx)

            # Summary table
            _pst_tbl = _pst.get("summary_table")
            if _pst_tbl is not None and not _pst_tbl.empty:
                _pst_display = _pst_tbl.copy()
                for _col in _pst_display.columns:
                    _pst_display[_col] = _pst_display[_col].apply(
                        lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else str(x)
                    )
                st.dataframe(_pst_display, use_container_width=True)

            # Bar chart of scenario P&Ls
            _pst_scenarios = _pst.get("scenarios", {})
            if _pst_scenarios:
                _pst_names = list(_pst_scenarios.keys())
                _pst_pls = [_pst_scenarios[n].get("portfolio_pl", 0) for n in _pst_names]
                _pst_colors = ["#e74c3c" if p < 0 else "#27ae60" for p in _pst_pls]
                _pst_fig = _go_pst.Figure()
                _pst_fig.add_trace(_go_pst.Bar(
                    x=_pst_names, y=_pst_pls, marker_color=_pst_colors,
                    hovertemplate="%{x}<br>P&L: %{y:+.1f}%<extra></extra>",
                ))
                _pst_fig.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(title="Portfolio P&L (%)", showgrid=True,
                               gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                )
                st.plotly_chart(_pst_fig, use_container_width=True)
                st.caption("IG duration 7yr · HY duration 4yr · Loans 0.25yr (floating) · EM 5.5yr · Spread shocks in bps · Rate shocks in % · Equity as fraction")
        else:
            st.info("Portfolio stress test unavailable.")
    except Exception as _pst_e:
        st.caption(f"Portfolio stress test unavailable: {_pst_e}")

# --- sub-tab 56: AT1/CoCo Monitor -------------------------------------------
with _analytics_sub56:
    import plotly.graph_objects as _go_at1
    st.header("AT1 / CoCo Monitor")
    st.markdown(
        "Bank hybrid capital stress tracker. AT1 bonds (CoCos) absorb losses before senior debt — "
        "stress here is an early-warning signal for bank credit conditions."
    )
    try:
        _at1 = load_at1_coco_monitor(df)
        if _at1.get("available"):
            _at1c = _at1.get("current", {})
            _c1at1, _c2at1, _c3at1 = st.columns(3)
            _c1at1.metric("Bank Stress Score", f"{_at1c.get('bank_stress_score', 0):.0f}/100")
            _c2at1.metric("AT1 Signal", _at1c.get("at1_signal", "N/A"))
            _c3at1.metric("CoCo Trigger Flag", "YES" if _at1c.get("coco_trigger_flag") else "NO")
            if _at1c.get("warning"):
                st.warning(_at1c["warning"])
            if _at1c.get("interpretation"):
                st.info(_at1c["interpretation"])
            _at1_hist = _at1.get("bank_stress_history")
            if _at1_hist is not None and len(_at1_hist) > 20:
                _fig_at1 = _go_at1.Figure()
                _fig_at1.add_trace(_go_at1.Scatter(
                    x=_at1_hist.index, y=_at1_hist.values,
                    name="Bank Stress Score", line=dict(color="#f59e0b", width=1.5)
                ))
                _fig_at1.add_hline(y=75, line_dash="dot", line_color="#ef4444",
                                   annotation_text="Systemic Risk")
                _fig_at1.update_layout(
                    template="plotly_dark", height=300, title="AT1/Bank Stress Score",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score (0–100)", range=[0, 100]),
                )
                st.plotly_chart(_fig_at1, use_container_width=True)
        else:
            st.info("AT1/CoCo monitor unavailable — requires bank ETF data (KBE, KRE, KBW, AT1).")
    except Exception as _at1_e:
        st.caption(f"AT1/CoCo monitor unavailable: {_at1_e}")

# --- sub-tab 57: Swap Spread Monitor ----------------------------------------
with _analytics_sub57:
    import plotly.graph_objects as _go_swp
    st.header("Swap Spread Monitor")
    st.markdown(
        "Interest rate swap spreads (swap rate minus Treasury yield) reflect bank balance-sheet "
        "constraints and systemic funding pressure. Deeply negative spreads signal dealer stress."
    )
    try:
        _swp = load_swap_spread_monitor(df)
        if _swp.get("available"):
            _swpc = _swp.get("current", {})
            _c1s, _c2s, _c3s, _c4s = st.columns(4)
            _c1s.metric("2y Swap Spread", f"{_swpc.get('spread_2y', float('nan')):.0f}bps" if _swpc.get('spread_2y') is not None else "N/A")
            _c2s.metric("5y Swap Spread", f"{_swpc.get('spread_5y', float('nan')):.0f}bps" if _swpc.get('spread_5y') is not None else "N/A")
            _c3s.metric("10y Swap Spread", f"{_swpc.get('spread_10y', float('nan')):.0f}bps" if _swpc.get('spread_10y') is not None else "N/A")
            _c4s.metric("30y Swap Spread", f"{_swpc.get('spread_30y', float('nan')):.0f}bps" if _swpc.get('spread_30y') is not None else "N/A")
            if _swpc.get("systemic_flag"):
                st.error("SYSTEMIC FLAG: Swap spread(s) deeply negative — dealer balance sheet stress elevated.")
            elif _swpc.get("stress_flag"):
                st.warning("Stress flag: At least one tenor in negative territory.")
            _swp_regime = _swpc.get("regime_10y", "")
            if _swp_regime:
                st.caption(f"10y Regime: {_swp_regime}")
            if _swpc.get("interpretation"):
                st.info(_swpc["interpretation"])
            _swp_hist = _swp.get("historical", {})
            _swp_10y = _swp_hist.get("10y")
            if _swp_10y is not None and len(_swp_10y) > 20:
                _fig_swp = _go_swp.Figure()
                _fig_swp.add_trace(_go_swp.Scatter(
                    x=_swp_10y.index, y=_swp_10y.values,
                    name="10y Swap Spread", line=dict(color="#60a5fa", width=1.5)
                ))
                _fig_swp.add_hline(y=0, line_dash="solid", line_color="#6b7280")
                _fig_swp.add_hline(y=-20, line_dash="dot", line_color="#ef4444",
                                   annotation_text="Systemic threshold")
                _fig_swp.update_layout(
                    template="plotly_dark", height=300, title="10y Swap Spread (bps)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="bps"),
                )
                st.plotly_chart(_fig_swp, use_container_width=True)
        else:
            st.info("Swap spread monitor unavailable — requires FRED API key or swap rate columns.")
    except Exception as _swp_e:
        st.caption(f"Swap spread monitor unavailable: {_swp_e}")

# --- sub-tab 58: Cross-Currency Basis ----------------------------------------
with _analytics_sub58:
    import plotly.graph_objects as _go_xccy
    st.header("Cross-Currency Basis")
    st.markdown(
        "EUR/USD and JPY/USD cross-currency basis swap cost. Negative basis = foreign investors "
        "pay a premium to hedge USD credit exposure → reduced demand for USD credit → spread widening."
    )
    try:
        _xccy = load_cross_currency_basis(df)
        if _xccy.get("available"):
            _xccyc = _xccy.get("current", {})
            _c1x, _c2x, _c3x = st.columns(3)
            _c1x.metric("EUR/USD Basis (proxy)", f"{_xccyc.get('eur_basis_bps', float('nan')):.1f}bps" if _xccyc.get('eur_basis_bps') is not None else "N/A")
            _c2x.metric("JPY/USD Basis (proxy)", f"{_xccyc.get('jpy_basis_bps', float('nan')):.1f}bps" if _xccyc.get('jpy_basis_bps') is not None else "N/A")
            _c3x.metric("Signal", _xccyc.get("signal", "N/A"))
            if _xccyc.get("interpretation"):
                st.info(_xccyc["interpretation"])
            _xccy_hist = _xccy.get("historical")
            if _xccy_hist is not None and len(_xccy_hist) > 20:
                _fig_xccy = _go_xccy.Figure()
                if "eur_basis_bps" in _xccy_hist.columns:
                    _fig_xccy.add_trace(_go_xccy.Scatter(
                        x=_xccy_hist.index, y=_xccy_hist["eur_basis_bps"],
                        name="EUR/USD Basis", line=dict(color="#34d399", width=1.5)
                    ))
                if "jpy_basis_bps" in _xccy_hist.columns:
                    _fig_xccy.add_trace(_go_xccy.Scatter(
                        x=_xccy_hist.index, y=_xccy_hist["jpy_basis_bps"],
                        name="JPY/USD Basis", line=dict(color="#a78bfa", width=1.5)
                    ))
                _fig_xccy.add_hline(y=-30, line_dash="dot", line_color="#ef4444",
                                    annotation_text="Significant Headwind")
                _fig_xccy.add_hline(y=0, line_dash="solid", line_color="#6b7280")
                _fig_xccy.update_layout(
                    template="plotly_dark", height=300,
                    title="Cross-Currency Basis (proxy, bps)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="bps"),
                )
                st.plotly_chart(_fig_xccy, use_container_width=True)
        else:
            st.info("Cross-currency basis unavailable — requires DXY or FX data.")
    except Exception as _xccy_e:
        st.caption(f"Cross-currency basis unavailable: {_xccy_e}")

# --- sub-tab 59: CRE Stress --------------------------------------------------
with _analytics_sub59:
    import plotly.graph_objects as _go_cre
    st.header("Commercial Real Estate (CRE) Stress")
    st.markdown(
        "CRE stress tracker using REIT ETFs (VNQ, IYR) and regional bank ETFs (KRE). "
        "CRE distress transmits to bank balance sheets via loan losses → credit spread widening."
    )
    try:
        _cre = load_cre_stress(df)
        if _cre.get("available"):
            _crec = _cre.get("current", {})
            _c1cre, _c2cre, _c3cre = st.columns(3)
            _c1cre.metric("CRE Stress Score", f"{_crec.get('cre_stress_score', 0):.0f}/100")
            _c2cre.metric("Regime", _crec.get("regime", "N/A"))
            _c3cre.metric("Systemic Flag", "YES" if _crec.get("systemic_flag") else "NO")
            if _crec.get("office_stress_flag"):
                st.warning("Office stress flag: VNQ >20% below 52-week high.")
            if _crec.get("systemic_flag"):
                st.error("SYSTEMIC FLAG: CRE stress score ≥75 — potential bank contagion risk.")
            if _crec.get("interpretation"):
                st.info(_crec["interpretation"])
            _cre_hist = _cre.get("historical")
            if _cre_hist is not None and len(_cre_hist) > 20:
                _fig_cre = _go_cre.Figure()
                _fig_cre.add_trace(_go_cre.Scatter(
                    x=_cre_hist.index, y=_cre_hist.values,
                    name="CRE Stress Score", line=dict(color="#fb923c", width=1.5)
                ))
                _fig_cre.add_hline(y=75, line_dash="dot", line_color="#ef4444",
                                   annotation_text="Systemic Risk")
                _fig_cre.add_hline(y=50, line_dash="dot", line_color="#f59e0b",
                                   annotation_text="Elevated")
                _fig_cre.update_layout(
                    template="plotly_dark", height=300, title="CRE Stress Score (0–100)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_cre, use_container_width=True)
        else:
            st.info("CRE stress monitor unavailable — requires VNQ or CRE spread data.")
    except Exception as _cre_e:
        st.caption(f"CRE stress unavailable: {_cre_e}")

# --- sub-tab 60: Primary Market Issuance ------------------------------------
with _analytics_sub60:
    import plotly.graph_objects as _go_iss
    st.header("Primary Market Issuance")
    st.markdown(
        "New issue concession (NIC) proxy and issuance activity monitor. "
        "Heavy supply / high NIC signals indigestion — spreads tend to widen. "
        "Light supply / low NIC signals favorable technicals."
    )
    try:
        _iss = load_primary_market_issuance(df)
        if _iss.get("available"):
            _issc = _iss.get("current", {})
            _c1i, _c2i, _c3i = st.columns(3)
            _c1i.metric("Issuance Score", f"{_issc.get('issuance_score', 0):.0f}/100")
            _c2i.metric("NIC Signal", _issc.get("nic_signal", "N/A"))
            _c3i.metric("Season", _issc.get("season", "N/A"))
            if _issc.get("interpretation"):
                st.info(_issc["interpretation"])
            if _issc.get("supply_warning"):
                st.warning(_issc["supply_warning"])
            _iss_hist = _iss.get("nic_history")
            if _iss_hist is not None and len(_iss_hist) > 20:
                _fig_iss = _go_iss.Figure()
                _fig_iss.add_trace(_go_iss.Scatter(
                    x=_iss_hist.index, y=_iss_hist.values,
                    name="NIC Proxy (bps)", line=dict(color="#22d3ee", width=1.5)
                ))
                _fig_iss.add_hline(y=0, line_dash="solid", line_color="#6b7280")
                _fig_iss.update_layout(
                    template="plotly_dark", height=300,
                    title="New Issue Concession Proxy (bps)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="bps"),
                )
                st.plotly_chart(_fig_iss, use_container_width=True)
        else:
            st.info("Issuance monitor unavailable — requires hy_spread data.")
    except Exception as _iss_e:
        st.caption(f"Primary market issuance unavailable: {_iss_e}")

# --- sub-tab 61: Distressed Debt ---------------------------------------------
with _analytics_sub61:
    import plotly.graph_objects as _go_dis
    st.header("Distressed Debt Monitor")
    st.markdown(
        "Estimates the fraction of the HY market trading at distressed levels (spread >1000bps) "
        "and the implied recovery rate / LGD cycle. Tracks credit cycle phase from Pristine → Acute."
    )
    try:
        _dis = load_distressed_debt(df)
        if _dis.get("available"):
            _disc = _dis.get("current", {})
            _c1d, _c2d, _c3d, _c4d = st.columns(4)
            _dr = _disc.get("distressed_ratio")
            _c1d.metric("Distressed Ratio", f"{_dr:.1%}" if _dr is not None else "N/A")
            _c2d.metric("Phase", _disc.get("distress_phase", "N/A"))
            _rr = _disc.get("recovery_rate")
            _c3d.metric("Recovery Rate", f"{_rr:.1f}%" if _rr is not None else "N/A")
            _lgd = _disc.get("lgd")
            _c4d.metric("LGD", f"{_lgd:.1%}" if _lgd is not None else "N/A")
            if _disc.get("warning"):
                st.error(_disc["warning"])
            if _disc.get("interpretation"):
                st.info(_disc["interpretation"])
            _dis_ratio = _dis.get("ratio_history")
            _dis_rr = _dis.get("recovery_history")
            if _dis_ratio is not None and len(_dis_ratio.dropna()) > 20:
                _fig_dis = _go_dis.Figure()
                _fig_dis.add_trace(_go_dis.Scatter(
                    x=_dis_ratio.index, y=(_dis_ratio * 100).values,
                    name="Distressed Ratio (%)", line=dict(color="#ef4444", width=1.5)
                ))
                if _dis_rr is not None and len(_dis_rr.dropna()) > 20:
                    _fig_dis.add_trace(_go_dis.Scatter(
                        x=_dis_rr.index, y=_dis_rr.values,
                        name="Recovery Rate (%)", line=dict(color="#34d399", width=1.5),
                        yaxis="y2"
                    ))
                _fig_dis.add_hline(y=22, line_dash="dot", line_color="#f59e0b",
                                   annotation_text="Distressed threshold (22%)")
                _fig_dis.update_layout(
                    template="plotly_dark", height=350,
                    title="Distressed Ratio vs Recovery Rate",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Distressed Ratio (%)", range=[0, 50]),
                    yaxis2=dict(title="Recovery Rate (%)", overlaying="y", side="right",
                                range=[0, 60]),
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(_fig_dis, use_container_width=True)
            _dis_ps = _dis.get("phase_summary", {})
            if _dis_ps:
                _dis_ps_rows = [
                    {"Phase": ph, "Obs": v["n_obs"], "Avg Score": v["mean_score"]}
                    for ph, v in _dis_ps.items()
                ]
                import pandas as _pd_dis
                st.dataframe(_pd_dis.DataFrame(_dis_ps_rows).set_index("Phase"), use_container_width=True)
        else:
            st.info("Distressed debt monitor unavailable — requires hy_spread column.")
    except Exception as _dis_e:
        st.caption(f"Distressed debt monitor unavailable: {_dis_e}")

# =============================================================================
# TAB 8: Allocation (placeholder — content in Tab 3 Portfolio and Tab 4 Backtest)
# =============================================================================
with tab8:
    st.header("Allocation")
    st.info("Allocation content is integrated into the **Portfolio** and **Backtest** tabs.")

# =============================================================================
# TAB 9: Signal Health Monitor
# =============================================================================
with tab9:
    st.header("Signal Health Monitor")
    st.markdown(
        "Meta-view of all signal modules — which data sources loaded, which are stale, "
        "and which are running on synthetic/fallback data."
    )
    try:
        from src.signal_health import check_signal_health, get_health_summary
        _sh = check_signal_health(df)
        _sh_sum = _sh.get("summary", {})

        # Overall health banner
        _oh = _sh.get("overall_health", "Unknown")
        _oh_color = {"Healthy": "success", "Degraded": "warning", "Critical": "error"}.get(_oh, "info")
        getattr(st, _oh_color)(
            f"Overall Health: **{_oh}** · "
            f"{_sh_sum.get('ok', 0)} OK · "
            f"{_sh_sum.get('live_fetch', 0)} Live · "
            f"{_sh_sum.get('degraded', 0)} Degraded · "
            f"{_sh_sum.get('unavailable', 0)} Unavailable · "
            f"Checked at {_sh.get('checked_at', '—')}"
        )

        # Data range
        st.caption(
            f"Dataset: **{_sh.get('data_rows', 0):,} rows** · "
            f"Date range: {_sh.get('date_range', '—')}"
        )

        # Summary table
        import plotly.graph_objects as _sh_go
        _sh_df = get_health_summary(df)
        if not _sh_df.empty:
            # Color-code the Status column
            def _sh_status_color(val):
                return {
                    "OK": "background-color: rgba(39,174,96,0.15)",
                    "Live": "background-color: rgba(52,152,219,0.15)",
                    "Degraded": "background-color: rgba(230,126,34,0.15)",
                    "Unavailable": "background-color: rgba(231,76,60,0.15)",
                }.get(val, "")
            st.dataframe(
                _sh_df.style.applymap(_sh_status_color, subset=["Status"]),
                use_container_width=True, hide_index=True,
            )

        # Category breakdown donut chart
        _sh_cats = {}
        for _m in _sh.get("modules", []):
            _cat = _m.get("category", "Other")
            _st = _m.get("status", "Unknown")
            _sh_cats.setdefault(_cat, {"OK": 0, "Live": 0, "Degraded": 0, "Unavailable": 0})
            _sh_cats[_cat][_st] = _sh_cats[_cat].get(_st, 0) + 1

        # Status breakdown bar
        _sb_fig = _sh_go.Figure()
        _status_colors = {"OK": "#27ae60", "Live": "#3498db", "Degraded": "#e67e22", "Unavailable": "#e74c3c"}
        for _s, _c in _status_colors.items():
            _count = _sh_sum.get(_s.lower().replace(" ", "_"), _sh_sum.get(_s.lower(), 0))
            if _count > 0:
                _sb_fig.add_trace(_sh_go.Bar(
                    x=[_s], y=[_count], name=_s, marker_color=_c,
                    text=[_count], textposition="auto",
                ))
        _sb_fig.update_layout(
            height=180, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
            xaxis=dict(showgrid=False, color="#6b7280"), showlegend=False,
            title=dict(text="Module Status Breakdown", font=dict(size=12, color="#9aa0aa")),
        )
        st.plotly_chart(_sb_fig, use_container_width=True)

        # Unavailable modules detail
        _unavail = [m for m in _sh.get("modules", []) if m.get("status") == "Unavailable"]
        if _unavail:
            with st.expander(f"{len(_unavail)} unavailable modules — missing data columns"):
                for _um in _unavail:
                    st.markdown(
                        f"**{_um['name']}** — missing: `{'`, `'.join(_um.get('cols_missing', []))}`"
                    )
    except ImportError:
        st.info("Signal health monitor is loading — the `src/signal_health.py` module will be available after the next data refresh.")
    except Exception as _sh_e:
        st.caption(f"Signal health unavailable: {_sh_e}")
