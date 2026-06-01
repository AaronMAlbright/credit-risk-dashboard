"""
Credit Markets — analytics section page.
Extracted from streamlit_app.py for performance (single-section reruns).
"""
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
try:
    from src.distressed_debt import run_distressed_debt as run_distressed_debt_analysis
except Exception:
    run_distressed_debt_analysis = None
from src.clo_monitor import run_clo_monitor
from src.financial_conditions import run_fci_analysis
from src.credit_impulse import run_credit_impulse_analysis
from src.etf_premium_discount import run_etf_premium_discount
from src.sovereign_contagion import run_sovereign_contagion
from src.consumer_credit_stress import run_consumer_credit_stress
from src.term_premium import run_term_premium_analysis
from src.yield_curve_butterfly import run_butterfly_analysis
from src.sloos_monitor import run_sloos_monitor
from src.etf_fund_flows import run_etf_fund_flows
from src.corporate_profit_cycle import run_corporate_profit_cycle
from src.cds_implied_pd import run_cds_implied_pd
from src.data_diagnostics import run_diagnostics as _run_data_diagnostics
from src.credit_strategy_memo import generate_credit_strategy_memo
from src.credit_taxonomy import latest_channel_snapshot
from src.credit_regime_performance import summarize_by_regime, latest_regime_performance_note
from src.credit_channel_validation import channel_validation_table, latest_channel_validation_snapshot
from src.credit_positioning import positioning_table, current_positioning
from src.spread_decomposition import latest_spread_snapshot
from src.credit_compensation_scorecard import build_credit_compensation_scorecard
from src.credit_compensation_validation import (
    analyze_scorecard_prediction_errors,
    analyze_scorecard_transitions,
    build_scorecard_validation_report,
    replay_scorecard_stress_episodes,
    validate_scorecard_recommendations,
)
from src.credit_relative_value import latest_relative_value_snapshot, relative_value_table
from src.channel_attribution import channel_contribution_table, top_channel_drivers
from src.credit_presentation import (
    build_credit_brief,
    credit_brief_markdown,
    credit_glossary_table,
    framework_assumptions_table,
)
from src.credit_model_spec import (
    credit_model_spec_table,
    model_spec_markdown,
    model_spec_limitations,
    model_spec_institutional_data,
    validation_boundary_table,
)
from src.credit_view_section import render_credit_view_section
from src.validation_section import render_validation_section
from src.credit_tearsheet import credit_market_tearsheet, credit_tearsheet_markdown
from src.rating_bucket_proxy import rating_bucket_proxy_table, rating_bucket_summary
from src.refinancing_wall import refinancing_wall_table, refinancing_wall_summary
from src.model_governance import (
    governance_markdown,
    governance_status_table,
    known_limitations_table,
    required_institutional_data_table,
)
from src.composite_comparison import composite_comparison_summary, composite_governance_note
from src.credit_case_study import case_study_table


st.set_page_config(
    page_title='Credit Markets — Credit Risk Dashboard',
    page_icon='📊',
    layout='wide',
)

from utils.shared import (
    load_data, _err_track,
    _sig_badge, _pct_clr,
    _VIEW_DESC, _VIEW_INSIGHT,
    _ANALYTICS_VIEWS,
)

# ── load_* aliases (thin wrappers around run_* imports) ──────────────────────
load_default_analysis          = run_default_analysis
load_forward_simulation        = run_forward_simulation
load_cdx_proxy                 = run_cdx_analysis
load_default_cycle             = run_default_cycle_analysis
load_spread_volatility         = run_spread_volatility_analysis
load_fallen_angel              = run_fallen_angel_analysis
load_global_credit             = run_global_credit_analysis
load_corporate_leverage        = run_corporate_leverage_analysis
load_seasonality               = run_seasonality_analysis
load_credit_quality_migration  = run_credit_quality_migration
load_loan_market_monitor       = run_loan_market_monitor
load_credit_basis              = run_credit_basis
load_primary_market_issuance   = run_primary_market_issuance
load_clo_monitor               = run_clo_monitor
load_cds_implied_pd            = run_cds_implied_pd
load_em_credit                 = run_em_credit_analysis
load_carry_breakeven           = run_breakeven_analysis
load_credit_momentum           = run_credit_momentum_analysis
load_quality_curve             = run_quality_curve_analysis


def load_distressed_debt(df):
    if run_distressed_debt_analysis is None:
        return {"available": False}
    try:
        return run_distressed_debt_analysis(df)
    except Exception as _e:
        return {"available": False, "error": str(_e)}


# ── Data & pre-processing ─────────────────────────────────────────────────────
df = load_data()
if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
    df = df.set_index(pd.to_datetime(df['date'])).drop(columns=['date'])

latest      = df.iloc[-1].to_dict()
decision    = str(latest.get('final_decision',    'N/A'))
environment = str(latest.get('final_environment', 'N/A'))
action      = str(latest.get('final_action',      'N/A'))
composite   = float(latest.get('composite_risk_score_smooth', 0))
comp_label  = str(latest.get('composite_risk_label', 'N/A'))

# ── Sidebar nav for this section ─────────────────────────────────────────────
_SECTION_NAME = 'Credit Markets'
_av_list = _ANALYTICS_VIEWS[_SECTION_NAME]
st.sidebar.markdown(f'### {_SECTION_NAME}')
_vsearch = st.sidebar.text_input(
    '_vsearch', placeholder='Search views...', label_visibility='collapsed',
    key='_vsearch_views'
)
_av_list_show = (
    [(n, s) for n, s in _av_list if _vsearch.strip().lower() in n.lower()]
    if _vsearch.strip() else _av_list
) or _av_list
_av_label = st.sidebar.selectbox(
    '_nav_view', [v[0] for v in _av_list_show], label_visibility='collapsed',
)
_active_sub = dict(_av_list_show).get(_av_label, _av_list[0][1])
_vdesc = _VIEW_DESC.get(_active_sub, '')
if _vdesc:
    st.sidebar.caption(_vdesc)

# Ensure analytics content blocks get a DatetimeIndex df
if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
    df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])


# ── Data freshness warning ─────────────────────────────────────────────────
try:
    _data_age = (pd.Timestamp.now() - df.index[-1]).days
    if _data_age > 3:
        st.warning(f'Data is {_data_age} days old — daily refresh may have failed. '
                   f'Last data point: {df.index[-1].date()}')
except Exception:
    pass

# ── Analytics UI: live signal bar + sidebar badges + view insight expander ────
if _active_sub is not None:
    try:
        _lhy = float(df["hy_spread"].dropna().iloc[-1])
        _lhy_bps = _lhy * 100.0 if abs(_lhy) < 50 else _lhy
        _lhy_pct = float((df["hy_spread"].dropna() < _lhy).mean() * 100)
        _lvix = float(df["vix"].dropna().iloc[-1])
        _lvix_pct = float((df["vix"].dropna() < _lvix).mean() * 100)
        _lsc = next(
            (c for c in ["composite_risk_score_smooth", "composite_score_smooth", "credit_risk_score_smooth"]
             if c in df.columns), None
        )
        _lsv = float(df[_lsc].dropna().iloc[-1]) if _lsc else None
        _lsvp = float((df[_lsc].dropna() < _lsv).mean() * 100) if _lsv is not None else 50.0
        _lreg = str(df["final_decision"].dropna().iloc[-1]) if "final_decision" in df.columns else "N/A"
        _lreg_c = {
            "Risk-On": "#22c55e", "Neutral": "#9aa0aa",
            "Caution": "#f59e0b", "Risk-Off": "#ef4444"
        }.get(_lreg, "#9aa0aa")
        _ldate = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else ""
        # ── Main content signal bar ───────────────────────────────────────────
        _bar_parts = [
            (f'<span style="color:#6b7280;font-size:0.71rem;letter-spacing:.4px">HY&nbsp;</span>'
             f'<span style="color:{_pct_clr(_lhy_pct)};font-weight:700">{_lhy_bps:.0f}'
             f'<span style="font-size:0.64rem;font-weight:400"> bps</span></span>'),
        ]
        if _lsv is not None:
            _bar_parts.append(
                f'<span style="color:#6b7280;font-size:0.71rem;letter-spacing:.4px">SCORE&nbsp;</span>'
                f'<span style="color:{_pct_clr(_lsvp)};font-weight:700">{_lsv:.2f}'
                f'<span style="font-size:0.64rem;font-weight:400"> ({_lsvp:.0f}th)</span></span>'
            )
        _bar_parts += [
            (f'<span style="color:#6b7280;font-size:0.71rem;letter-spacing:.4px">VIX&nbsp;</span>'
             f'<span style="color:{_pct_clr(_lvix_pct)};font-weight:700">{_lvix:.1f}</span>'),
            (f'<span style="color:#6b7280;font-size:0.71rem;letter-spacing:.4px">REGIME&nbsp;</span>'
             f'<span style="color:{_lreg_c};font-weight:700">{_lreg}</span>'),
            f'<span style="color:#374151;font-size:0.67rem">{_ldate}</span>',
        ]
        st.markdown(
            '<div style="background:rgba(14,20,40,0.85);border:1px solid #2d3550;border-radius:6px;'
            'padding:6px 18px;margin-bottom:10px;display:flex;gap:22px;align-items:center;flex-wrap:wrap">'
            + "".join(f"<span>{p}</span>" for p in _bar_parts)
            + "</div>",
            unsafe_allow_html=True,
        )
        # ── Sidebar live signal badges ────────────────────────────────────────
        st.sidebar.markdown("---")
        _sb_rows = (
            f'{_sig_badge(_lhy_pct)}&nbsp;<b>HY</b> {_lhy_bps:.0f} bps'
            f'&ensp;<span style="color:{_pct_clr(_lhy_pct)};font-size:0.69rem">{_lhy_pct:.0f}th pct</span><br>'
        )
        if _lsv is not None:
            _sb_rows += (
                f'{_sig_badge(_lsvp)}&nbsp;<b>Score</b> {_lsv:.2f}'
                f'&ensp;<span style="color:{_pct_clr(_lsvp)};font-size:0.69rem">{_lsvp:.0f}th pct</span><br>'
            )
        _sb_rows += (
            f'{_sig_badge(_lvix_pct)}&nbsp;<b>VIX</b> {_lvix:.1f}'
            f'&ensp;<span style="color:{_pct_clr(_lvix_pct)};font-size:0.69rem">{_lvix_pct:.0f}th pct</span><br>'
            f'<span style="color:{_lreg_c}">&#9679;</span>&nbsp;<b>Regime</b>'
            f'&ensp;<span style="color:{_lreg_c};font-weight:600">{_lreg}</span>'
        )
        st.sidebar.markdown(
            f'<div style="font-size:0.78rem;line-height:2.0">{_sb_rows}</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── Sidebar error badge ───────────────────────────────────────────────────
    _n_errs = len(st.session_state.get("_view_errors", {}))
    if _n_errs:
        st.sidebar.warning(f"⚠ {_n_errs} view error(s) this session")
    # ── "What does this show?" expander ──────────────────────────────────────
    _insight_text = _VIEW_INSIGHT.get(_active_sub)
    if _insight_text:
        with st.expander("What does this show?", expanded=False):
            st.markdown(_insight_text)

    try:
        _ccs = build_credit_compensation_scorecard(df)
        if _ccs.get("available"):
            _cur = _ccs["current"]
            _rec = _ccs.get("recommendation", "Hold")
            _rec_color = {
                "Add": "#22c55e",
                "Hold": "#9aa0aa",
                "Upgrade Quality": "#f59e0b",
                "Hedge": "#f97316",
                "De-risk": "#ef4444",
            }.get(_rec, "#9aa0aa")
            st.subheader("Credit Compensation Scorecard")
            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            _m1.metric("HY OAS", f"{_cur.get('hy_oas_bps'):.0f} bps" if _cur.get("hy_oas_bps") is not None else "-")
            _m2.metric("Excess Spread", f"{_cur.get('excess_spread_bps'):.0f} bps" if _cur.get("excess_spread_bps") is not None else "-")
            _m3.metric("Comp Ratio", f"{_cur.get('spread_compensation_ratio'):.2f}x" if _cur.get("spread_compensation_ratio") is not None else "-")
            _m4.metric("HY Breakeven", f"{_cur.get('carry_breakeven_hy_bps'):.0f} bps" if _cur.get("carry_breakeven_hy_bps") is not None else "-")
            _m5.markdown(
                f'<div style="padding:0.45rem 0 0.25rem;color:#6b7280;font-size:0.78rem">Recommendation</div>'
                f'<div style="color:{_rec_color};font-weight:800;font-size:1.55rem;line-height:1.1">{_rec}</div>',
                unsafe_allow_html=True,
            )
            st.caption(_ccs.get("summary", ""))
            _pm_verdict = _ccs.get("pm_final_verdict")
            if _pm_verdict:
                st.markdown("**PM Final Verdict**")
                st.caption(_pm_verdict)
            _memo_table = _ccs.get("memo_table")
            if _memo_table is not None and not _memo_table.empty:
                st.markdown("**PM Trade Memo**")
                st.dataframe(_memo_table, use_container_width=True, hide_index=True)
            _action_table = _ccs.get("action_table")
            if _action_table is not None and not _action_table.empty:
                st.markdown("**Portfolio Expression**")
                st.dataframe(_action_table, use_container_width=True, hide_index=True)
            _rating_table = _ccs.get("rating_bucket_table")
            if _rating_table is not None and not _rating_table.empty:
                st.markdown("**Rating-Bucket Allocation**")
                st.dataframe(_rating_table, use_container_width=True, hide_index=True)
            _risk_reward_table = _ccs.get("risk_reward_table")
            if _risk_reward_table is not None and not _risk_reward_table.empty:
                st.markdown("**Allocation Risk / Reward**")
                st.dataframe(_risk_reward_table, use_container_width=True, hide_index=True)
            _marginal_table = _ccs.get("marginal_allocation_table")
            if _marginal_table is not None and not _marginal_table.empty:
                st.markdown("**Marginal Allocation Advice**")
                st.dataframe(_marginal_table, use_container_width=True, hide_index=True)
            _net_beta_table = _ccs.get("net_spread_beta_table")
            if _net_beta_table is not None and not _net_beta_table.empty:
                st.markdown("**Net Spread Beta**")
                st.caption(_ccs.get("net_spread_beta_summary_text", ""))
                st.dataframe(_net_beta_table, use_container_width=True, hide_index=True)
            _cdx_hedge_table = _ccs.get("cdx_hedge_table")
            if _cdx_hedge_table is not None and not _cdx_hedge_table.empty:
                st.markdown("**CDX Hedge Sizing**")
                st.caption(_ccs.get("cdx_hedge_summary_text", ""))
                st.dataframe(_cdx_hedge_table, use_container_width=True, hide_index=True)
            _constraint_table = _ccs.get("constraint_table")
            if _constraint_table is not None and not _constraint_table.empty:
                st.markdown("**Portfolio Constraints**")
                st.caption(_ccs.get("constraint_summary_text", ""))
                st.dataframe(_constraint_table, use_container_width=True, hide_index=True)
            _bucket_return_table = _ccs.get("bucket_return_table")
            if _bucket_return_table is not None and not _bucket_return_table.empty:
                with st.expander("Expected return & stress by rating bucket", expanded=False):
                    st.caption(_ccs.get("bucket_return_summary_text", ""))
                    st.dataframe(_bucket_return_table, use_container_width=True, hide_index=True)
            _shock_table = _ccs.get("spread_shock_table")
            if _shock_table is not None and not _shock_table.empty:
                with st.expander("Spread shock sensitivity", expanded=False):
                    st.caption(_ccs.get("spread_shock_summary_text", ""))
                    st.dataframe(_shock_table, use_container_width=True, hide_index=True)
            _scenario_table = _ccs.get("scenario_preset_table")
            if _scenario_table is not None and not _scenario_table.empty:
                with st.expander("Scenario preset stress", expanded=False):
                    st.caption(_ccs.get("scenario_preset_summary_text", ""))
                    st.dataframe(_scenario_table, use_container_width=True, hide_index=True)
            _confidence_table = _ccs.get("confidence_table")
            if _confidence_table is not None and not _confidence_table.empty:
                with st.expander("Scorecard confidence flags", expanded=False):
                    st.dataframe(_confidence_table, use_container_width=True, hide_index=True)
            _assumptions_table = _ccs.get("bucket_assumptions_table")
            if _assumptions_table is not None and not _assumptions_table.empty:
                with st.expander("Bucket model assumptions", expanded=False):
                    st.dataframe(_assumptions_table, use_container_width=True, hide_index=True)
            _forward_table = _ccs.get("forward_outcomes_table")
            if _forward_table is not None and not _forward_table.empty:
                with st.expander("Historical forward outcomes", expanded=False):
                    st.caption(_ccs.get("forward_outcomes_summary", ""))
                    st.dataframe(_forward_table, use_container_width=True, hide_index=True)
            _trigger_table = _ccs.get("trigger_table")
            if _trigger_table is not None and not _trigger_table.empty:
                st.markdown("**What Changes Our Mind**")
                st.dataframe(_trigger_table, use_container_width=True, hide_index=True)
            _validation = validate_scorecard_recommendations(df)
            if _validation.get("available"):
                with st.expander("Scorecard validation / backtest", expanded=False):
                    st.caption(_validation.get("summary", ""))
                    st.dataframe(_validation["table"], use_container_width=True, hide_index=True)
                    _validation_report = build_scorecard_validation_report(df)
                    if _validation_report.get("available"):
                        _dl1, _dl2 = st.columns(2)
                        _dl1.download_button(
                            "Download validation report",
                            data=_validation_report["markdown"],
                            file_name="credit_compensation_scorecard_validation.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                        _dl2.download_button(
                            "Download validation CSV",
                            data=_validation_report["csv"],
                            file_name="credit_compensation_scorecard_validation.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
            _transitions = analyze_scorecard_transitions(df)
            if _transitions.get("available"):
                with st.expander("Scorecard transition stability", expanded=False):
                    st.caption(_transitions.get("summary_text", ""))
                    _tm, _td = st.tabs(["Transitions", "Durations"])
                    with _tm:
                        st.dataframe(_transitions["matrix_table"], use_container_width=True, hide_index=True)
                        if not _transitions["transition_outcome_table"].empty:
                            st.dataframe(_transitions["transition_outcome_table"], use_container_width=True, hide_index=True)
                    with _td:
                        st.dataframe(_transitions["duration_table"], use_container_width=True, hide_index=True)
            _errors = analyze_scorecard_prediction_errors(df)
            if _errors.get("available"):
                with st.expander("Scorecard false positive / false negative episodes", expanded=False):
                    st.caption(_errors.get("summary_text", ""))
                    _fp_tab, _fn_tab, _error_summary_tab = st.tabs(["False positives", "False negatives", "Summary"])
                    with _fp_tab:
                        st.dataframe(_errors["false_positive_table"], use_container_width=True, hide_index=True)
                    with _fn_tab:
                        st.dataframe(_errors["false_negative_table"], use_container_width=True, hide_index=True)
                    with _error_summary_tab:
                        st.dataframe(_errors["summary_by_recommendation"], use_container_width=True, hide_index=True)
            _stress_replay = replay_scorecard_stress_episodes(df)
            if _stress_replay.get("available"):
                with st.expander("Scorecard stress episode replay", expanded=False):
                    st.caption(_stress_replay.get("summary_text", ""))
                    st.dataframe(_stress_replay["table"], use_container_width=True, hide_index=True)
            with st.expander("Scorecard detail", expanded=False):
                st.dataframe(_ccs["table"], use_container_width=True, hide_index=True)
    except Exception as _ccs_e:
        st.caption(f"Credit compensation scorecard unavailable: {_ccs_e}")


if _active_sub == 19:
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

if _active_sub == 20:
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

if _active_sub == 21:
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

if _active_sub == 24:
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
            _has_actual = _dcc.get("has_actual_data", False)

            # Metrics row — add actual columns if data is present
            if _has_actual:
                _d1, _d2, _d3, _d4, _d5, _d6 = st.columns(6)
                _d1.metric("Implied Default Rate", f"{_dcc.get('current_implied_pct', 0):.2f}%",
                           help="Jarrow-Turnbull: HY spread / (100 × LGD)")
                _d2.metric("Phase", _dcc.get("current_phase", "—"))
                _co = _dcc.get("actual_chargeoff_pct")
                _dq = _dcc.get("actual_delinq_pct")
                _d3.metric("Charge-Off Rate", f"{_co:.2f}%" if _co and not pd.isna(_co) else "—",
                           help="Fed H.8: Charge-Off Rate on Business Loans (quarterly)")
                _d4.metric("C&I Delinquency", f"{_dq:.2f}%" if _dq and not pd.isna(_dq) else "—",
                           help="Fed H.8: Delinquency Rate on C&I Loans (quarterly)")
                _gap = _dcc.get("implied_vs_actual_gap")
                _d5.metric("Implied vs Actual Gap", f"{_gap:+.2f}pp" if _gap and not pd.isna(_gap) else "—",
                           help="Positive = market pricing more stress than realized; negative = losses outpacing market pricing")
                _d6.metric("% of GFC Peak", f"{_dcc.get('pct_of_gfc_peak', 0):.0f}%")
            else:
                _d1, _d2, _d3, _d4 = st.columns(4)
                _d1.metric("Implied Default Rate", f"{_dcc.get('current_implied_pct', 0):.2f}%",
                           help="Jarrow-Turnbull: HY spread / (100 × LGD)")
                _d2.metric("Phase", _dcc.get("current_phase", "—"))
                _d3.metric("% of GFC Peak", f"{_dcc.get('pct_of_gfc_peak', 0):.0f}%")
                _d4.metric("% of COVID Peak", f"{_dcc.get('pct_of_covid_peak', 0):.0f}%")

            if _dcc.get("interpretation"):
                st.info(_dcc["interpretation"])

            if _has_actual:
                from src.data_sources import format_source_note

                st.caption(
                    "**Observed source status:** "
                    + format_source_note("CORBLACBS", "DRBLACBS")
                    + " "
                    "Implied = Jarrow-Turnbull (HY spread / LGD). "
                    "Gap > 0 = market pricing excess stress; Gap < 0 = realized losses exceeding implied."
                )

            # Cycle comparison table
            _dc_cc = _dc.get("cycle_comparison")
            if _dc_cc is not None and not _dc_cc.empty:
                with st.expander("Historical Cycle Comparison", expanded=True):
                    st.dataframe(_dc_cc, use_container_width=True)

            # Time series chart — implied + actual lines when data is present
            _dc_ts = _dc.get("time_series")
            if _dc_ts is not None and not _dc_ts.empty and "default_cycle_pct" in _dc_ts.columns:
                _dc_fig = _go_dc.Figure()
                _dc_ts_idx = pd.to_datetime(
                    _dc_ts.index if _dc_ts.index.dtype != object
                    else _dc_ts.get("date", _dc_ts.index)
                )
                _dc_fig.add_trace(_go_dc.Scatter(
                    x=_dc_ts_idx, y=_dc_ts["default_cycle_pct"],
                    name="Implied Default Rate", line=dict(color="#e74c3c", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Implied: %{y:.2f}%<extra></extra>",
                ))
                if "actual_chargeoff_pct" in _dc_ts.columns and _dc_ts["actual_chargeoff_pct"].notna().any():
                    _dc_fig.add_trace(_go_dc.Scatter(
                        x=_dc_ts_idx, y=_dc_ts["actual_chargeoff_pct"],
                        name="Actual Charge-Off Rate", line=dict(color="#f59e0b", width=2, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>Charge-Off: %{y:.2f}%<extra></extra>",
                    ))
                if "actual_delinq_pct" in _dc_ts.columns and _dc_ts["actual_delinq_pct"].notna().any():
                    _dc_fig.add_trace(_go_dc.Scatter(
                        x=_dc_ts_idx, y=_dc_ts["actual_delinq_pct"],
                        name="C&I Delinquency Rate", line=dict(color="#8b5cf6", width=1.5, dash="dash"),
                        hovertemplate="%{x|%Y-%m-%d}<br>Delinquency: %{y:.2f}%<extra></extra>",
                    ))
                _dc_fig.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=8, b=8),
                    legend=dict(orientation="h", y=1.08, font=dict(size=10)),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Default Rate (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_dc_fig, use_container_width=True)

            # Gap chart — implied minus actual (shows risk premium vs realized stress)
            if _dc_ts is not None and "implied_vs_actual_gap" in _dc_ts.columns and _dc_ts["implied_vs_actual_gap"].notna().any():
                _gap_fig = _go_dc.Figure()
                _gap_vals = _dc_ts["implied_vs_actual_gap"]
                _gap_fig.add_trace(_go_dc.Bar(
                    x=_dc_ts_idx, y=_gap_vals,
                    name="Implied vs Actual Gap",
                    marker_color=[
                        "rgba(239,68,68,0.6)" if v > 0 else "rgba(245,158,11,0.6)"
                        for v in _gap_vals.fillna(0)
                    ],
                    hovertemplate="%{x|%Y-%m-%d}<br>Gap: %{y:+.2f}pp<extra></extra>",
                ))
                _gap_fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _gap_fig.update_layout(
                    height=150, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=4, b=4),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Gap (pp)", zeroline=False),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_gap_fig, use_container_width=True)
                st.caption("Red = implied > actual (market pricing excess stress / risk premium). Amber = actual > implied (realized losses outpacing market pricing).")
        else:
            st.info("Default cycle unavailable — requires HY spread data.")
    except Exception as _dc_e:
        st.caption(f"Default cycle unavailable: {_dc_e}")

# =============================================================================
# ANALYTICS sub-tab 25: Signal Comparison Mode
# =============================================================================

if _active_sub == 27:
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

if _active_sub == 28:
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

if _active_sub == 29:
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

if _active_sub == 30:
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

if _active_sub == 31:
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

if _active_sub == 40:
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

if _active_sub == 42:
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

if _active_sub == 48:
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

if _active_sub == 60:
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

if _active_sub == 61:
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

# --- sub-tab 62: CLO Monitor ------------------------------------------------

if _active_sub == 62:
    import plotly.graph_objects as _go_clo
    st.header("CLO / Structured Credit Monitor")
    st.markdown(
        "CLO stress tracker. CLOs are the largest buyers of leveraged loans — when CLO "
        "demand is constrained, loan spreads widen and HY technicals deteriorate. "
        "BKLN/HYG divergence and OC cushion proxy signal CLO market health."
    )
    try:
        _clo = load_clo_monitor(df)
        if _clo.get("available"):
            _cloc = _clo.get("current", {})
            _c1clo, _c2clo, _c3clo = st.columns(3)
            _c1clo.metric("CLO Stress Score", f"{_cloc.get('clo_stress_score', 0):.0f}/100")
            _c2clo.metric("Regime", _cloc.get("regime", "N/A"))
            _c3clo.metric("Refi Window", _cloc.get("refi_window", "N/A"))
            _oc = _cloc.get("oc_cushion_proxy")
            if _oc is not None:
                st.metric("OC Cushion Proxy", f"{_oc:.0f}/100 (100=max cushion)")
            if _cloc.get("warning"):
                st.warning(_cloc["warning"])
            if _cloc.get("interpretation"):
                st.info(_cloc["interpretation"])
            _clo_hist = _clo.get("historical_score")
            _clo_loan_hy = _clo.get("loan_vs_hy_history")
            if _clo_hist is not None and len(_clo_hist.dropna()) > 20:
                _fig_clo = _go_clo.Figure()
                _fig_clo.add_trace(_go_clo.Scatter(
                    x=_clo_hist.index, y=_clo_hist.values,
                    name="CLO Stress Score", line=dict(color="#f59e0b", width=1.5)
                ))
                _fig_clo.add_hline(y=50, line_dash="dot", line_color="#f59e0b", annotation_text="Stress")
                _fig_clo.add_hline(y=75, line_dash="dot", line_color="#ef4444", annotation_text="Acute")
                _fig_clo.update_layout(
                    template="plotly_dark", height=300, title="CLO Stress Score (0–100)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_clo, use_container_width=True)
            if _clo_loan_hy is not None and len(_clo_loan_hy.dropna()) > 20:
                _fig_lh = _go_clo.Figure()
                _fig_lh.add_trace(_go_clo.Scatter(
                    x=_clo_loan_hy.index, y=(_clo_loan_hy * 100).values,
                    name="Loan vs HY Relative Return (%)", line=dict(color="#34d399", width=1.2)
                ))
                _fig_lh.add_hline(y=0, line_dash="solid", line_color="#6b7280")
                _fig_lh.update_layout(
                    template="plotly_dark", height=250, title="BKLN/HYG Relative Return (21d, %)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="%"),
                )
                st.plotly_chart(_fig_lh, use_container_width=True)
        else:
            st.info("CLO monitor unavailable — requires BKLN/HYG ETF data or hy_spread column.")
    except Exception as _clo_e:
        st.caption(f"CLO monitor unavailable: {_clo_e}")

# --- sub-tab 63: Financial Conditions Index ----------------------------------

if _active_sub == 73:
    import plotly.graph_objects as _go_pd
    st.header("CDS-Implied Default Probability Surface")
    st.markdown(
        "Converts HY and IG spread levels into implied default probabilities across "
        "1y/3y/5y/10y horizons using standard CDS pricing. Recovery rate is cycle-adjusted "
        "— spreads widening past 450bps compress recoveries, amplifying implied PD nonlinearly."
    )
    try:
        _pd_res = load_cds_implied_pd(df)
        if _pd_res.get("available"):
            _pdc = _pd_res.get("current", {})
            _c1pd, _c2pd, _c3pd, _c4pd = st.columns(4)
            _pd_1y = _pdc.get("pd_surface_hy", {}).get(1)
            _c1pd.metric("HY 1y Implied PD", f"{_pd_1y:.1f}%" if _pd_1y is not None else "N/A")
            _pd_5y = _pdc.get("pd_surface_hy", {}).get(5)
            _c2pd.metric("HY 5y Cumulative PD", f"{_pd_5y:.1f}%" if _pd_5y is not None else "N/A")
            _rr = _pdc.get("recovery_rate_current")
            _c3pd.metric("Cycle-Adj Recovery Rate", f"{_rr:.1f}%" if _rr is not None else "N/A")
            _el = _pdc.get("el_hy_1y")
            _c4pd.metric("Expected Loss (HY 1y)", f"${_el:.2f}/$100" if _el is not None else "N/A")
            if _pdc.get("warning"):
                st.error(_pdc["warning"])
            if _pdc.get("interpretation"):
                st.info(_pdc["interpretation"])
            if _pdc.get("pd_vs_historical"):
                st.caption(_pdc["pd_vs_historical"])
            _hy_surf = _pdc.get("pd_surface_hy", {})
            _ig_surf = _pdc.get("pd_surface_ig", {})
            if _hy_surf:
                import pandas as _pd_surf
                _surf_rows = []
                for _h in [1, 3, 5, 10]:
                    _surf_rows.append({
                        "Horizon": f"{_h}y",
                        "HY PD (%)": f"{_hy_surf.get(_h, float('nan')):.1f}" if _hy_surf.get(_h) is not None else "N/A",
                        "IG PD (%)": f"{_ig_surf.get(_h, float('nan')):.1f}" if _ig_surf.get(_h) is not None else "N/A",
                    })
                st.dataframe(_pd_surf.DataFrame(_surf_rows).set_index("Horizon"), use_container_width=True)
            _pd_hy_hist = _pd_res.get("pd_hy_history")
            _pd_ig_hist = _pd_res.get("pd_ig_history")
            if _pd_hy_hist is not None and len(_pd_hy_hist.dropna()) > 50:
                _fig_pd = _go_pd.Figure()
                _fig_pd.add_trace(_go_pd.Scatter(
                    x=_pd_hy_hist.index, y=_pd_hy_hist.values,
                    name="HY 5y Implied PD (%)", line=dict(color="#ef4444", width=1.5)
                ))
                if _pd_ig_hist is not None and len(_pd_ig_hist.dropna()) > 50:
                    _fig_pd.add_trace(_go_pd.Scatter(
                        x=_pd_ig_hist.index, y=_pd_ig_hist.values,
                        name="IG 5y Implied PD (%)", line=dict(color="#60a5fa", width=1.2, dash="dot")
                    ))
                _fig_pd.add_hline(y=HY_HISTORICAL_AVG_PD * 5 * 0.8, line_dash="dot",
                                  line_color="#f59e0b", annotation_text="Elevated vs history")
                _fig_pd.update_layout(
                    template="plotly_dark", height=320,
                    title="CDS-Implied 5y Cumulative Default Probability (%)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="%"),
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(_fig_pd, use_container_width=True)
        else:
            st.info("CDS-implied PD unavailable — requires hy_spread or ig_spread column.")
    except Exception as _pd_e:
        st.caption(f"CDS-implied PD unavailable: {_pd_e}")

# =============================================================================
# BATCH 8 ANALYTICS: sub74–79
# =============================================================================


if _active_sub == 77:
    import plotly.graph_objects as _go77
    st.header("EM Credit — Emerging Market vs DM HY")
    st.markdown(
        "Tracks **EM vs DM credit performance** using the EMB/HYG ETF ratio as a real-time proxy for "
        "relative stress. EM credit underperformance vs DM HY — driven by DXY strength, EM capital "
        "outflows, or idiosyncratic EM stress — historically **leads DM HY widening by 4–6 weeks**, "
        "making this a useful early warning indicator."
    )
    try:
        _em77 = load_em_credit(df)
        if _em77.get("available"):
            _emc = _em77.get("current", {})
            _snap77 = _em77.get("snapshot", {})
            _em77a, _em77b, _em77c, _em77d = st.columns(4)
            _em77a.metric("EMB/HYG Ratio", f"{_emc.get('em_hyg_ratio', float('nan')):.4f}"
                          if not pd.isna(_emc.get("em_hyg_ratio", float("nan"))) else "—")
            _em77b.metric("30d Change", f"{(_emc.get('em_hyg_ratio_30d_chg') or 0)*100:+.1f}%",
                          delta_color="normal")
            _em77c.metric("EM vs DM Signal", f"{_emc.get('em_vs_dm_signal', 0):.0f}/100",
                          help="Higher = more EM stress relative to DM")
            _em77d.metric("EM Stress Regime", _snap77.get("em_stress_regime", "—"))

            if _em77.get("lead_signal"):
                st.warning(_em77["lead_signal"])
            if _em77.get("interpretation"):
                st.info(_em77["interpretation"])

            # EMB/HYG ratio time series
            _em77_df = _em77.get("df")
            if _em77_df is not None and "em_hyg_ratio" in _em77_df.columns:
                _em77_plot = _em77_df[["em_hyg_ratio"]].dropna().tail(504).copy()
                _em77_plot.index = pd.to_datetime(_em77_plot.index)
                _fig77a = _go77.Figure()
                _fig77a.add_trace(_go77.Scatter(
                    x=_em77_plot.index, y=_em77_plot["em_hyg_ratio"],
                    name="EMB/HYG Ratio", line=dict(color="#10b981", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>EMB/HYG: %{y:.4f}<extra></extra>",
                ))
                _fig77a.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="EMB/HYG Ratio (Rising = EM outperforming DM HY)",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Ratio"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig77a, use_container_width=True)
                st.caption("Falling ratio = EM underperforming DM HY — watch for DM spread widening follow-through.")

            # EM vs DM signal vs HY spread
            if _em77_df is not None and "em_vs_dm_signal" in _em77_df.columns and "hy_spread" in df.columns:
                _em77_sig = _em77_df[["em_vs_dm_signal"]].dropna().tail(504)
                _em77_hy = df[["hy_spread"]].dropna().tail(504)
                _em77_combo = _em77_sig.join(_em77_hy, how="inner").dropna()
                if not _em77_combo.empty:
                    _fig77b = _go77.Figure()
                    _fig77b.add_trace(_go77.Scatter(
                        x=_em77_combo.index, y=_em77_combo["em_vs_dm_signal"],
                        name="EM Stress Signal", line=dict(color="#10b981", width=1.5),
                        yaxis="y1",
                        hovertemplate="%{x|%Y-%m-%d}<br>EM Signal: %{y:.0f}<extra></extra>",
                    ))
                    _fig77b.add_trace(_go77.Scatter(
                        x=_em77_combo.index, y=_em77_combo["hy_spread"],
                        name="HY OAS (%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                        yaxis="y2",
                        hovertemplate="%{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>",
                    ))
                    _fig77b.add_hline(y=65, line=dict(color="#f59e0b", dash="dash", width=1),
                                      annotation_text="Lead signal threshold (65)",
                                      annotation_font=dict(color="#f59e0b", size=9), yref="y")
                    _fig77b.update_layout(
                        height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11),
                        margin=dict(l=8, r=8, t=24, b=8),
                        title=dict(text="EM Stress Signal vs DM HY Spread",
                                   font=dict(size=12, color="#9aa0aa")),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#10b981", title="EM Signal (0–100)"),
                        yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                    title="HY OAS (%)", showgrid=False),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig77b, use_container_width=True)

            # DXY snapshot
            _dxy77 = _emc.get("dxy", float("nan"))
            _dxy_z77 = _emc.get("em_dxy_zscore", float("nan"))
            if not pd.isna(_dxy77):
                st.subheader("Dollar (DXY) Transmission")
                _d77a, _d77b = st.columns(2)
                _d77a.metric("DXY Level", f"{_dxy77:.2f}",
                             help="DXY strength = headwind for EM borrowers with USD debt")
                _d77b.metric("DXY Z-Score (1y)", f"{_dxy_z77:.2f}"
                             if not pd.isna(_dxy_z77) else "—")
                st.caption("DXY > +1σ (z-score) = USD strength amplifying EM credit stress via debt service costs.")
        else:
            st.info("EM credit unavailable — requires EMB/HYG ETF data (via yfinance) or em_hyg_ratio column.")
    except Exception as _e77:
        _err_track(_active_sub, _e77)
        st.caption(f"EM credit unavailable: {_e77}")



if _active_sub == 78:
    import plotly.graph_objects as _go78
    st.header("Carry Breakeven Analysis")
    st.markdown(
        "**Carry breakeven** answers the core credit portfolio question: "
        "_how much can spreads widen before carry income is erased?_ "
        "Formula: **Breakeven Widening (bps/yr) = All-in Yield (%) ÷ Duration (yrs) × 100**. "
        "High carry + short duration (HY) provides a large widening buffer. "
        "Low carry + long duration (IG) is vulnerable to even modest spread widening."
    )
    try:
        _be78 = load_carry_breakeven(df)
        if _be78.get("available"):
            _bec = _be78.get("current", {})
            _be78a, _be78b, _be78c, _be78d = st.columns(4)
            _hy_be = _bec.get("hy_breakeven_annual_bps")
            _ig_be = _bec.get("ig_breakeven_annual_bps")
            _bbb_be = _bec.get("bbb_breakeven_annual_bps")
            _hy_mo = _bec.get("hy_breakeven_monthly_bps")
            _be78a.metric("HY Breakeven (ann)", f"{_hy_be:.0f} bps/yr" if _hy_be else "—",
                          help="HY spreads can widen this much before carry is consumed (annual)")
            _be78b.metric("IG Breakeven (ann)", f"{_ig_be:.0f} bps/yr" if _ig_be else "—")
            _be78c.metric("BBB Breakeven (ann)", f"{_bbb_be:.0f} bps/yr" if _bbb_be else "—")
            _be78d.metric("HY Monthly Buffer", f"{_hy_mo:.1f} bps/mo" if _hy_mo else "—",
                          help="How much HY spreads can widen per month before carry is lost")

            if _be78.get("interpretation"):
                st.info(_be78["interpretation"])

            # Breakeven history chart
            _be78_df = _be78.get("df")
            if _be78_df is not None and "hy_breakeven_annual_bps" in _be78_df.columns:
                _be78_plot = _be78_df[["hy_breakeven_annual_bps"]].copy()
                if "ig_breakeven_annual_bps" in _be78_df.columns:
                    _be78_plot["ig_breakeven_annual_bps"] = _be78_df["ig_breakeven_annual_bps"]
                _be78_plot = _be78_plot.dropna(how="all").tail(504)
                _be78_plot.index = pd.to_datetime(_be78_plot.index)
                _fig78a = _go78.Figure()
                _fig78a.add_trace(_go78.Scatter(
                    x=_be78_plot.index, y=_be78_plot["hy_breakeven_annual_bps"],
                    name="HY Breakeven (bps/yr)", line=dict(color="#ef4444", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>HY BE: %{y:.0f} bps/yr<extra></extra>",
                ))
                if "ig_breakeven_annual_bps" in _be78_plot.columns:
                    _fig78a.add_trace(_go78.Scatter(
                        x=_be78_plot.index, y=_be78_plot["ig_breakeven_annual_bps"],
                        name="IG Breakeven (bps/yr)", line=dict(color="#3b82f6", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>IG BE: %{y:.0f} bps/yr<extra></extra>",
                    ))
                _fig78a.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Breakeven Widening (bps/yr)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig78a, use_container_width=True)

            # Shock scenario table
            st.subheader("Spread Shock Scenarios — HY Carry Buffer")
            _shock_rows = _be78.get("shock_scenarios")
            if _shock_rows:
                import pandas as _pd78
                _shock_df = _pd78.DataFrame(_shock_rows)
                st.dataframe(_shock_df, use_container_width=True, hide_index=True)
            elif _hy_be:
                import pandas as _pd78b
                _shock_rows_fallback = []
                for _shock in [25, 50, 100, 150, 200, 300]:
                    _months = (_hy_be / _shock * 12) if _shock > 0 else float("inf")
                    _verdict = "Safe" if _months > 12 else "Caution" if _months > 6 else "Risky"
                    _shock_rows_fallback.append({
                        "Shock (bps)": _shock,
                        "Months of Carry Remaining": f"{_months:.1f}",
                        "Annual Carry Consumed (%)": f"{min(_shock / _hy_be * 100, 100):.0f}%",
                        "Verdict": _verdict,
                    })
                st.dataframe(_pd78b.DataFrame(_shock_rows_fallback), use_container_width=True, hide_index=True)
                st.caption(f"Based on HY breakeven of {_hy_be:.0f} bps/yr. "
                           f"Months > 12 = carry covers a full year even at that shock level.")
            else:
                st.info("Shock scenario table unavailable — requires hy_yield or hy_spread + yield_10y.")

            # BBB vs HY carry comparison
            if _bbb_be and _ig_be and _hy_be:
                st.subheader("Carry Breakeven Comparison")
                _carr_compare = {
                    "IG (~7yr dur)":  f"{_ig_be:.0f} bps/yr",
                    "BBB (~6yr dur)": f"{_bbb_be:.0f} bps/yr",
                    "HY (~4yr dur)":  f"{_hy_be:.0f} bps/yr",
                }
                _comp_cols = st.columns(3)
                for _ci, (_lbl, _val) in enumerate(zip(
                    ["IG (~7yr dur)", "BBB (~6yr dur)", "HY (~4yr dur)"],
                    [_ig_be, _bbb_be, _hy_be],
                )):
                    _comp_cols[_ci].metric(_lbl, f"{_val:.0f} bps/yr",
                                           help="Annual spread widening this segment can absorb before carry is exhausted")
                st.caption("Higher breakeven = larger cushion. HY typically has the highest carry but also the most spread vol.")
        else:
            st.info("Carry breakeven unavailable — requires hy_yield or ig_yield columns (or hy_spread + yield_10y).")
    except Exception as _e78:
        _err_track(_active_sub, _e78)
        st.caption(f"Carry breakeven unavailable: {_e78}")



if _active_sub == 87:
    import plotly.graph_objects as _go87
    st.header("Credit Spread Momentum")
    st.markdown(
        "Rate-of-change in HY and IG spreads across **1M, 3M, and 6M** horizons. "
        "Spread tightening (negative momentum) = positive credit signal = risk-on. "
        "**Momentum divergence** between HY and IG signals which segment the market favors: "
        "IG tightening while HY widens = flight-to-quality within credit. "
        "Momentum percentile shows where current 3M momentum sits vs full history."
    )
    try:
        _cm87 = load_credit_momentum(df)
        if _cm87.get("available"):
            _cmc = _cm87["current"]
            _cm87a, _cm87b, _cm87c, _cm87d = st.columns(4)
            _cm87a.metric("HY Mom (1M)", f"{_cmc.get('hy_mom_21d', 0):+.0f} bps",
                          delta_color="inverse")
            _cm87b.metric("HY Mom (3M)", f"{_cmc.get('hy_mom_63d', 0):+.0f} bps",
                          delta_color="inverse")
            _cm87c.metric("HY Mom (6M)", f"{_cmc.get('hy_mom_126d', 0):+.0f} bps",
                          delta_color="inverse")
            _cm87d.metric("Momentum Signal", f"{_cmc.get('credit_momentum_signal', 0):.0f}/100",
                          help="0 = very negative momentum, 100 = very positive")

            _div87 = _cmc.get("hy_ig_momentum_divergence", 0) or 0
            _pct87 = _cm87.get("momentum_percentile")
            _cm87_info = (
                f"Trend: **{_cm87.get('trend', '—')}** · "
                f"HY/IG divergence: {_div87:+.0f} bps"
                + (f" · 3M momentum percentile: {_pct87:.0f}th" if _pct87 is not None else "")
            )
            st.caption(_cm87_info)
            if _cm87.get("interpretation"):
                st.info(_cm87["interpretation"])

            # Multi-horizon momentum chart
            _hist87 = _cm87.get("historical")
            if _hist87 is not None and not _hist87.empty:
                _hist87 = _hist87.copy()
                _hist87.index = pd.to_datetime(_hist87.index)
                _fig87a = _go87.Figure()
                _mom_series = [
                    ("hy_mom_21d",  "HY 1M", "#ef4444", False),
                    ("hy_mom_63d",  "HY 3M", "#f59e0b", False),
                    ("ig_mom_63d",  "IG 3M", "#3b82f6", True),
                ]
                for _col87, _lbl87, _clr87, _dot87 in _mom_series:
                    if _col87 in _hist87.columns:
                        _fig87a.add_trace(_go87.Scatter(
                            x=_hist87.index, y=_hist87[_col87],
                            name=_lbl87, line=dict(color=_clr87, width=1.5,
                                                   dash="dot" if _dot87 else "solid"),
                            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{_lbl87}: %{{y:+.0f}} bps<extra></extra>",
                        ))
                _fig87a.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _fig87a.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Spread Change (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig87a, use_container_width=True)
                st.caption("Negative = spreads tightening (positive for credit) · "
                           "Positive = spreads widening (negative for credit)")

                # Momentum signal history
                if "credit_momentum_signal" in _hist87.columns:
                    _fig87b = _go87.Figure()
                    _sig87 = _hist87["credit_momentum_signal"].fillna(50)
                    _sig87_colors = ["#ef4444" if v < 40 else "#27ae60" if v > 60 else "#f59e0b"
                                     for v in _sig87]
                    _fig87b.add_trace(_go87.Bar(
                        x=_hist87.index, y=_sig87,
                        marker_color=_sig87_colors, name="Momentum Signal",
                        hovertemplate="%{x|%Y-%m-%d}<br>Signal: %{y:.0f}<extra></extra>",
                    ))
                    _fig87b.add_hline(y=60, line=dict(color="#27ae60", dash="dash", width=1),
                                      annotation_text="Positive (60)",
                                      annotation_font=dict(color="#27ae60", size=9))
                    _fig87b.add_hline(y=40, line=dict(color="#ef4444", dash="dash", width=1),
                                      annotation_text="Negative (40)",
                                      annotation_font=dict(color="#ef4444", size=9))
                    _fig87b.update_layout(
                        height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                        title=dict(text="Credit Momentum Signal (0–100)",
                                   font=dict(size=12, color="#9aa0aa")),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#6b7280", range=[0, 100]),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig87b, use_container_width=True)

            # Current snapshot table
            st.subheader("Momentum Snapshot")
            import pandas as _pd87
            _snap87_rows = []
            for _hz87, _hy_key87, _ig_key87 in [
                ("1M (21d)", "hy_mom_21d", "ig_mom_21d"),
                ("3M (63d)", "hy_mom_63d", "ig_mom_63d"),
                ("6M (126d)", "hy_mom_126d", "ig_mom_126d"),
            ]:
                _hy87 = _cmc.get(_hy_key87)
                _ig87 = _cmc.get(_ig_key87)
                _snap87_rows.append({
                    "Horizon": _hz87,
                    "HY Mom (bps)": f"{_hy87:+.0f}" if _hy87 is not None else "—",
                    "IG Mom (bps)": f"{_ig87:+.0f}" if _ig87 is not None else "—",
                    "HY/IG Divergence (bps)": f"{(_hy87 or 0) - (_ig87 or 0):+.0f}"
                    if _hy87 is not None and _ig87 is not None else "—",
                })
            st.dataframe(_pd87.DataFrame(_snap87_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Credit momentum unavailable — requires HY or IG spread data.")
    except Exception as _e87:
        _err_track(_active_sub, _e87)
        st.caption(f"Credit momentum unavailable: {_e87}")



if _active_sub == 89:
    import plotly.graph_objects as _go89
    st.header("Credit Quality Curve")
    st.markdown(
        "The **credit quality curve** measures the spread premium demanded at each rating tier: "
        "IG → BBB → HY. A **steep curve** (large HY−IG differential) signals risk aversion — "
        "investors demand extra compensation to move down the quality spectrum. "
        "A **flat or inverted curve** signals complacency — historically a late-cycle warning. "
        "Z-scores are computed over a trailing 252-day window."
    )
    try:
        _qc89 = load_quality_curve(df)
        if _qc89.get("available"):
            _qcc89 = _qc89["current"]
            _qc89a, _qc89b, _qc89c, _qc89d = st.columns(4)
            _hy_sp89 = _qcc89.get("hy_spread", float("nan"))
            _hy_ig89 = _qcc89.get("hy_ig_premium", float("nan"))
            _qc89a.metric("HY Spread", f"{_hy_sp89:.2f}%" if not pd.isna(_hy_sp89) else "—")
            _qc89b.metric("HY−IG Premium", f"{_hy_ig89:.2f}pp" if not pd.isna(_hy_ig89) else "—",
                          help="Extra spread HY pays vs IG — higher = more risk aversion")
            _qc89c.metric("Curve Z-Score (252d)", f"{_qcc89.get('curve_slope_zscore', float('nan')):.2f}"
                          if not pd.isna(_qcc89.get("curve_slope_zscore", float("nan"))) else "—",
                          help="How steep the quality curve is vs trailing year")
            _qc89d.metric("Interpretation", _qcc89.get("interpretation", "—"))

            # Quality curve staircase bar chart (current snapshot)
            _ig_sp89 = _qcc89.get("ig_spread", float("nan"))
            _bbb_sp89 = _qcc89.get("bbb_spread", _qcc89.get("bbb_oas", float("nan")))
            _stair89 = []
            if not pd.isna(_ig_sp89):
                _stair89.append(("IG", _ig_sp89, "#3b82f6"))
            if not pd.isna(_bbb_sp89):
                _stair89.append(("BBB", _bbb_sp89, "#f59e0b"))
            if not pd.isna(_hy_sp89):
                _stair89.append(("HY", _hy_sp89, "#ef4444"))
            if _stair89:
                _fig89a = _go89.Figure()
                _fig89a.add_trace(_go89.Bar(
                    x=[s[0] for s in _stair89],
                    y=[s[1] for s in _stair89],
                    marker_color=[s[2] for s in _stair89],
                    text=[f"{s[1]:.2f}%" for s in _stair89],
                    textposition="outside",
                    hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                ))
                _fig89a.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="Credit Quality Staircase (Current Spreads)",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="OAS / Spread (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig89a, use_container_width=True)

            # HY−IG premium history
            _qc89_df = _qc89.get("df")
            if _qc89_df is not None and "hy_ig_premium" in _qc89_df.columns:
                _qc89_plot = _qc89_df[["hy_ig_premium"]].copy()
                if "bbb_ig_premium" in _qc89_df.columns:
                    _qc89_plot["bbb_ig_premium"] = _qc89_df["bbb_ig_premium"]
                _qc89_plot = _qc89_plot.dropna(how="all").tail(504)
                _qc89_plot.index = pd.to_datetime(_qc89_plot.index)
                _fig89b = _go89.Figure()
                if "bbb_ig_premium" in _qc89_plot.columns:
                    _fig89b.add_trace(_go89.Scatter(
                        x=_qc89_plot.index, y=_qc89_plot["bbb_ig_premium"],
                        name="BBB−IG Premium", line=dict(color="#f59e0b", width=1.5),
                        hovertemplate="%{x|%Y-%m-%d}<br>BBB−IG: %{y:.2f}pp<extra></extra>",
                    ))
                _fig89b.add_trace(_go89.Scatter(
                    x=_qc89_plot.index, y=_qc89_plot["hy_ig_premium"],
                    name="HY−IG Premium", line=dict(color="#ef4444", width=2),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY−IG: %{y:.2f}pp<extra></extra>",
                ))
                _fig89b.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig89b.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="Credit Quality Curve Premium Over Time",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Spread Premium (pp)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig89b, use_container_width=True)
                st.caption("Steep HY−IG premium = elevated risk aversion. "
                           "Flattening/inversion = complacency or late-cycle compression.")
        else:
            st.info("Credit quality curve unavailable — requires hy_spread data in dataset.")
    except Exception as _e89:
        _err_track(_active_sub, _e89)
        st.caption(f"Quality curve unavailable: {_e89}")



if _active_sub == 100:
    import plotly.graph_objects as _go100
    st.header("HY Spread Percentile Monitor")
    st.markdown(
        "Where are current HY spreads relative to their own history? "
        "**Rolling percentile bands** (3Y, 5Y, full history) give a valuation anchor for credit. "
        "Spreads at the **5th percentile** signal euphoria / late-cycle compression risk. "
        "Spreads above the **80th percentile** signal stress — credit is cheap but for a reason. "
        "The percentile rank is a cleaner risk signal than spread level alone because it adjusts for "
        "the structural decline in spreads since the GFC."
    )
    try:
        if "hy_spread" in df.columns and df["hy_spread"].notna().any():
            _hy100 = df[["hy_spread"]].copy()
            _hy100.index = pd.to_datetime(_hy100.index)
            _cur_spread = float(latest.get("hy_spread", float("nan")))

            # Compute rolling percentile ranks
            _hy100["pctile_3y"] = _hy100["hy_spread"].rolling(756, min_periods=126).rank(pct=True) * 100
            _hy100["pctile_5y"] = _hy100["hy_spread"].rolling(1260, min_periods=252).rank(pct=True) * 100
            _hy100["pctile_full"] = _hy100["hy_spread"].rank(pct=True) * 100

            _p3y = _hy100["pctile_3y"].iloc[-1]
            _p5y = _hy100["pctile_5y"].iloc[-1]
            _pfull = _hy100["pctile_full"].iloc[-1]

            _ha, _hb, _hc, _hd = st.columns(4)
            _ha.metric("HY Spread", f"{_cur_spread:.0f} bps")
            _hb.metric("3Y Percentile", f"{_p3y:.0f}th" if pd.notna(_p3y) else "—")
            _hc.metric("5Y Percentile", f"{_p5y:.0f}th" if pd.notna(_p5y) else "—")
            _hd.metric("Full-History Pctile", f"{_pfull:.0f}th" if pd.notna(_pfull) else "—")

            if pd.notna(_pfull):
                if _pfull < 10:
                    st.warning("HY spreads near all-time tights — late-cycle compression risk high. "
                               "Mean-reversion probability elevated.")
                elif _pfull > 80:
                    st.info("HY spreads in distressed territory — forced sellers may create opportunities, "
                            "but fundamental deterioration must be evaluated first.")

            # HY spread with historical percentile bands
            _hy100_tail = _hy100.tail(1260)
            _roll_min_5y = _hy100["hy_spread"].rolling(1260, min_periods=252).min()
            _roll_max_5y = _hy100["hy_spread"].rolling(1260, min_periods=252).max()
            _roll_p25 = _hy100["hy_spread"].rolling(1260, min_periods=252).quantile(0.25)
            _roll_p75 = _hy100["hy_spread"].rolling(1260, min_periods=252).quantile(0.75)
            _roll_med = _hy100["hy_spread"].rolling(1260, min_periods=252).median()

            _fig100a = _go100.Figure()
            _fig100a.add_trace(_go100.Scatter(
                x=_hy100_tail.index, y=_roll_max_5y.loc[_hy100_tail.index],
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            ))
            _fig100a.add_trace(_go100.Scatter(
                x=_hy100_tail.index, y=_roll_min_5y.loc[_hy100_tail.index],
                fill="tonexty", fillcolor="rgba(79,142,247,0.06)",
                line=dict(color="rgba(0,0,0,0)"), name="5Y Min-Max Range", hoverinfo="skip",
            ))
            _fig100a.add_trace(_go100.Scatter(
                x=_hy100_tail.index, y=_roll_p75.loc[_hy100_tail.index],
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            ))
            _fig100a.add_trace(_go100.Scatter(
                x=_hy100_tail.index, y=_roll_p25.loc[_hy100_tail.index],
                fill="tonexty", fillcolor="rgba(79,142,247,0.12)",
                line=dict(color="rgba(0,0,0,0)"), name="5Y IQR (25-75th)", hoverinfo="skip",
            ))
            _fig100a.add_trace(_go100.Scatter(
                x=_hy100_tail.index, y=_roll_med.loc[_hy100_tail.index],
                line=dict(color="#4f8ef7", width=1.5, dash="dot"), name="5Y Median",
                hovertemplate="%{x|%Y-%m-%d}<br>Median: %{y:.0f}bps<extra></extra>",
            ))
            _fig100a.add_trace(_go100.Scatter(
                x=_hy100_tail.index, y=_hy100_tail["hy_spread"],
                line=dict(color="#f59e0b", width=2.5), name="HY Spread",
                hovertemplate="%{x|%Y-%m-%d}<br>HY Spread: %{y:.0f}bps<extra></extra>",
            ))
            _fig100a.update_layout(
                height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="HY Spread vs 5Y Historical Range", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="bps"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            )
            st.plotly_chart(_fig100a, use_container_width=True)

            # Rolling full-history percentile time series
            _fig100b = _go100.Figure()
            _fig100b.add_hrect(y0=0, y1=20, fillcolor="rgba(39,174,96,0.1)", line_width=0)
            _fig100b.add_hrect(y0=80, y1=100, fillcolor="rgba(239,68,68,0.1)", line_width=0)
            _fig100b.add_trace(_go100.Scatter(
                x=_hy100_tail.index, y=_hy100["pctile_full"].loc[_hy100_tail.index],
                line=dict(color="#a78bfa", width=2), name="Full-History Percentile",
                hovertemplate="%{x|%Y-%m-%d}<br>Percentile: %{y:.0f}th<extra></extra>",
            ))
            _fig100b.add_hline(y=50, line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1))
            _fig100b.add_hline(y=80, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig100b.add_hline(y=20, line=dict(color="rgba(39,174,96,0.4)", dash="dot", width=1))
            _fig100b.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="HY Spread Full-History Percentile Rank", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Percentile"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig100b, use_container_width=True)
            st.caption("Green band = tight territory (<20th pctile) · Red band = stressed territory (>80th pctile)")

            st.markdown("**Spread Percentile Interpretation**")
            st.table(pd.DataFrame([
                {"Percentile Band": "< 10th", "Valuation Signal": "Euphoric tights", "Credit Implication": "Compression risk high; spread carry offset by jump risk"},
                {"Percentile Band": "10–30th", "Valuation Signal": "Rich", "Credit Implication": "Overweight only with strong fundamental backdrop"},
                {"Percentile Band": "30–70th", "Valuation Signal": "Fair value range", "Credit Implication": "Normal risk/reward; factor in momentum"},
                {"Percentile Band": "70–85th", "Valuation Signal": "Cheap", "Credit Implication": "Selectively add; distinguish idiosyncratic vs systemic"},
                {"Percentile Band": "> 85th", "Valuation Signal": "Deeply cheap / distressed", "Credit Implication": "Systemic risk elevated; entry size down, quality up"},
            ]))
        else:
            st.info("hy_spread column not available.")
    except Exception as _e100:
        _err_track(_active_sub, _e100)
        st.caption(f"HY Spread Percentile: {_e100}")


if _active_sub == 102:
    import plotly.graph_objects as _go102
    st.header("Credit Cycle Phase")
    st.markdown(
        "The **credit cycle phase** synthesises four leading indicators into a single cycle position: "
        "HY spread momentum (tightening vs widening), credit-equity alignment, NFCI financial conditions, "
        "and the Sahm-like labor signal. Each phase has a distinct risk/return profile for credit. "
        "**Recovery** and **Expansion** favour credit overweights; **Late Cycle** demands quality upgrading; "
        "**Contraction** (rising Sahm + widening spreads + tight NFCI) is the highest-risk phase."
    )
    try:
        _phase_cols = ["hy_change_30d", "credit_equity_divergence", "nfci", "sahm_like",
                       "hy_spread", "spread", "unemployment"]
        _phase_avail = [c for c in _phase_cols if c in df.columns]
        if len(_phase_avail) >= 3:
            _ph102 = df[_phase_avail].copy()
            _ph102.index = pd.to_datetime(_ph102.index)

            def _classify_phase(row):
                hy_mom = row.get("hy_change_30d", 0) if pd.notna(row.get("hy_change_30d")) else 0
                div = str(row.get("credit_equity_divergence", "Neutral"))
                nfci_val = row.get("nfci", 0) if pd.notna(row.get("nfci")) else 0
                sahm_val = row.get("sahm_like", 0) if pd.notna(row.get("sahm_like")) else 0

                stress_score = (
                    (1 if hy_mom > 30 else -1 if hy_mom < -10 else 0) +
                    (1 if div == "Diverging" else -1 if div == "Converging" else 0) +
                    (1 if nfci_val > 0.5 else -1 if nfci_val < -0.5 else 0) +
                    (1 if sahm_val > 0.3 else 0)
                )
                if stress_score >= 3:
                    return "Contraction"
                if stress_score >= 1:
                    return "Late Cycle"
                if stress_score <= -2:
                    return "Recovery"
                return "Expansion"

            _ph102["credit_cycle_phase"] = _ph102.apply(_classify_phase, axis=1)

            _cur_phase = _classify_phase({col: latest.get(col) for col in _phase_avail})
            _phase_colors = {"Recovery": "#27ae60", "Expansion": "#4f8ef7",
                             "Late Cycle": "#f59e0b", "Contraction": "#ef4444"}
            _phase_desc = {
                "Recovery": "Spreads tightening, labor stabilizing, financial conditions easing. "
                            "HY overweight favoured; spread carry attractive.",
                "Expansion": "Risk-on; credit conditions broadly benign. "
                             "Maintain credit exposure; watch for late-cycle signs.",
                "Late Cycle": "Divergence signals emerging; NFCI tightening. "
                              "Reduce HY beta; upgrade quality to IG/BB. Shorten duration.",
                "Contraction": "Multi-indicator stress confirmation. "
                               "Defensive posture: cash, IG, short-dated; reduce HY exposure significantly.",
            }
            _cur_color = _phase_colors.get(_cur_phase, "#6b7280")

            st.markdown(f"### Current Phase: <span style='color:{_cur_color}'>{_cur_phase}</span>", unsafe_allow_html=True)
            st.info(_phase_desc.get(_cur_phase, "Phase undetermined."))

            _pa, _pb, _pc, _pd = st.columns(4)
            _pa.metric("HY 30d Change", f"{latest.get('hy_change_30d', float('nan')):+.0f} bps"
                       if pd.notna(latest.get("hy_change_30d")) else "—")
            _pb.metric("EQ-Credit Signal", str(latest.get("credit_equity_divergence", "—")))
            _pc.metric("NFCI", f"{latest.get('nfci', float('nan')):.2f}"
                       if pd.notna(latest.get("nfci")) else "—")
            _pd.metric("Sahm-Like", f"{latest.get('sahm_like', float('nan')):.2f}pp"
                       if pd.notna(latest.get("sahm_like")) else "—")

            # Phase timeline
            _ph102_tail = _ph102.tail(756)
            _phase_num = {"Recovery": 1, "Expansion": 2, "Late Cycle": 3, "Contraction": 4}
            _ph102_tail = _ph102_tail.copy()
            _ph102_tail["phase_num"] = _ph102_tail["credit_cycle_phase"].map(_phase_num)
            _ph_bar_colors = [_phase_colors.get(p, "#6b7280") for p in _ph102_tail["credit_cycle_phase"]]
            _fig102a = _go102.Figure()
            _fig102a.add_trace(_go102.Bar(
                x=_ph102_tail.index, y=_ph102_tail["phase_num"],
                marker_color=_ph_bar_colors, name="Credit Cycle Phase",
                hovertemplate="%{x|%Y-%m-%d}<br>Phase: %{customdata}<extra></extra>",
                customdata=_ph102_tail["credit_cycle_phase"],
            ))
            _fig102a.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Credit Cycle Phase (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(tickvals=[1, 2, 3, 4], ticktext=["Recovery", "Expansion", "Late Cycle", "Contraction"],
                           showgrid=False, color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig102a, use_container_width=True)

            # Phase frequency over full history
            _phase_freq = _ph102["credit_cycle_phase"].value_counts(normalize=True).mul(100)
            _fig102b = _go102.Figure()
            _fig102b.add_trace(_go102.Bar(
                x=_phase_freq.index.tolist(),
                y=_phase_freq.values.tolist(),
                marker_color=[_phase_colors.get(p, "#6b7280") for p in _phase_freq.index],
                hovertemplate="%{x}<br>%{y:.0f}% of history<extra></extra>",
            ))
            _fig102b.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Phase Frequency (Full History)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="% of days"),
                xaxis=dict(color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig102b, use_container_width=True)

            # Phase × HY spread distribution
            if "hy_spread" in _ph102.columns:
                _fig102c = _go102.Figure()
                for _phase_name, _ph_color in _phase_colors.items():
                    _mask = _ph102["credit_cycle_phase"] == _phase_name
                    _sp_vals = _ph102.loc[_mask, "hy_spread"].dropna()
                    if len(_sp_vals) > 10:
                        _fig102c.add_trace(_go102.Violin(
                            y=_sp_vals, name=_phase_name,
                            fillcolor=f"rgba{tuple(int(_ph_color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.3,)}",
                            line_color=_ph_color, box_visible=True, meanline_visible=True,
                            hovertemplate=f"{_phase_name}<br>HY: %{{y:.0f}}bps<extra></extra>",
                        ))
                _fig102c.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="HY Spread Distribution by Cycle Phase", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="HY Spread (bps)"),
                    xaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig102c, use_container_width=True)
                st.caption("Violin distribution shows HY spread levels historically observed in each cycle phase")
        else:
            st.info("Insufficient columns for credit cycle phase analysis.")
    except Exception as _e102:
        _err_track(_active_sub, _e102)
        st.caption(f"Credit cycle phase: {_e102}")


if _active_sub == 105:
    import plotly.graph_objects as _go105
    st.header("HY Spread Momentum Term Structure")
    st.markdown(
        "**Momentum term structure** compares the sign and magnitude of HY spread changes across 5d, 30d, "
        "and 90d horizons. When **all three horizons are widening**, credit stress is broad-based and "
        "persistent — the most bearish configuration. When **short-term tightening conflicts with longer-term widening**, "
        "it may signal a counter-trend bounce inside a deteriorating trend. "
        "The term structure shape also reveals whether recent spread moves are accelerating or mean-reverting."
    )
    try:
        _mom_cols = ["hy_change_5d", "hy_change_30d", "hy_change_90d", "hy_spread"]
        if all(c in df.columns for c in _mom_cols[:3]):
            _mom105 = df[_mom_cols].copy()
            _mom105.index = pd.to_datetime(_mom105.index)

            _m5 = float(latest.get("hy_change_5d", float("nan")))
            _m30 = float(latest.get("hy_change_30d", float("nan")))
            _m90 = float(latest.get("hy_change_90d", float("nan")))

            def _mom_signal(m5, m30, m90):
                if all(pd.isna(v) for v in [m5, m30, m90]):
                    return "Undetermined"
                pos = sum(1 for v in [m5, m30, m90] if pd.notna(v) and v > 0)
                neg = sum(1 for v in [m5, m30, m90] if pd.notna(v) and v < 0)
                if pos == 3:
                    return "Broad Widening"
                if neg == 3:
                    return "Broad Tightening"
                if pd.notna(m5) and pd.notna(m30) and m5 < 0 < m30:
                    return "Counter-Trend Bounce"
                if pd.notna(m5) and pd.notna(m30) and m5 > 0 > m30:
                    return "Short-Term Surge"
                return "Mixed"

            _cur_signal = _mom_signal(_m5, _m30, _m90)
            _signal_colors = {
                "Broad Widening": "#ef4444", "Broad Tightening": "#27ae60",
                "Counter-Trend Bounce": "#f59e0b", "Short-Term Surge": "#a78bfa", "Mixed": "#6b7280",
            }

            _ma, _mb, _mc, _md = st.columns(4)
            _ma.metric("HY Δ5d", f"{_m5:+.0f} bps" if pd.notna(_m5) else "—",
                       delta_color="inverse")
            _mb.metric("HY Δ30d", f"{_m30:+.0f} bps" if pd.notna(_m30) else "—",
                       delta_color="inverse")
            _mc.metric("HY Δ90d", f"{_m90:+.0f} bps" if pd.notna(_m90) else "—",
                       delta_color="inverse")
            _md.metric("Momentum Signal", _cur_signal)

            _sig_col = _signal_colors.get(_cur_signal, "#6b7280")
            if _cur_signal == "Broad Widening":
                st.error("All three momentum horizons widening: broad-based credit deterioration. "
                         "Reduce HY exposure; favour quality and short duration.")
            elif _cur_signal == "Broad Tightening":
                st.success("All three horizons tightening: sustained credit improvement. "
                           "Risk-on conditions intact.")

            # Current term structure bar chart
            _ts_vals = [_m5, _m30, _m90]
            _ts_labels = ["5d", "30d", "90d"]
            _ts_colors = ["#ef4444" if pd.notna(v) and v > 0 else "#27ae60" if pd.notna(v) else "#6b7280"
                          for v in _ts_vals]
            _fig105a = _go105.Figure()
            _fig105a.add_trace(_go105.Bar(
                x=_ts_labels, y=_ts_vals,
                marker_color=_ts_colors,
                text=[f"{v:+.0f}" if pd.notna(v) else "n/a" for v in _ts_vals],
                textposition="outside",
                hovertemplate="%{x}: %{y:+.0f} bps<extra></extra>",
            ))
            _fig105a.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
            _fig105a.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="HY Spread Momentum Term Structure (current)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="bps"),
                xaxis=dict(color="#6b7280", title="Horizon"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig105a, use_container_width=True)
            st.caption("Red = widening (stress) · Green = tightening (improvement)")

            # Rolling history of all three momentum series
            _mom_tail = _mom105.tail(504)
            _fig105b = _go105.Figure()
            _pal105 = {"hy_change_5d": ("#4f8ef7", "Δ5d"), "hy_change_30d": ("#f59e0b", "Δ30d"),
                       "hy_change_90d": ("#a78bfa", "Δ90d")}
            for _col, (_clr, _lbl) in _pal105.items():
                if _col in _mom_tail.columns:
                    _fig105b.add_trace(_go105.Scatter(
                        x=_mom_tail.index, y=_mom_tail[_col],
                        line=dict(color=_clr, width=1.8), name=_lbl,
                        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{_lbl}: %{{y:+.0f}}bps<extra></extra>",
                    ))
            _fig105b.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig105b.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="HY Spread Momentum History (2Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="bps"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            )
            st.plotly_chart(_fig105b, use_container_width=True)

            # Rolling momentum signal classification
            _mom105["signal_class"] = _mom105.apply(
                lambda r: _mom_signal(r.get("hy_change_5d"), r.get("hy_change_30d"), r.get("hy_change_90d")), axis=1
            )
            _mom105["is_broad_wide"] = (_mom105["signal_class"] == "Broad Widening").astype(int)
            _mom105["broad_wide_freq"] = _mom105["is_broad_wide"].rolling(63).mean() * 100
            _mom_tail2 = _mom105.tail(504)
            _fig105c = _go105.Figure()
            _fig105c.add_trace(_go105.Scatter(
                x=_mom_tail2.index, y=_mom_tail2["broad_wide_freq"],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
                line=dict(color="#ef4444", width=1.5), name="Broad Widening Freq",
                hovertemplate="%{x|%Y-%m-%d}<br>Freq: %{y:.0f}%<extra></extra>",
            ))
            _fig105c.add_hline(y=30, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig105c.update_layout(
                height=180, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Rolling 3M 'Broad Widening' Frequency (%)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig105c, use_container_width=True)
            st.caption("Above 30%: persistent widening pressure — not a one-day spike")
        else:
            st.info("HY momentum columns not available — run the feature pipeline.")
    except Exception as _e105:
        _err_track(_active_sub, _e105)
        st.caption(f"HY momentum term structure: {_e105}")


if _active_sub == 110:
    import plotly.graph_objects as _go110
    st.header("Credit Regime Monitor")
    st.markdown(
        "The **credit regime** classifier combines the yield curve (2s10s spread) and the unemployment rate "
        "into a four-state taxonomy: **Expansion** (steep curve + low unemployment), **Neutral**, "
        "**Late Cycle** (inverted curve + low unemployment — historically the most dangerous combination "
        "because risk premia are compressed just as the curve is signalling tightening), and "
        "**Credit Stress** (inverted curve + elevated unemployment — spread widening underway). "
        "Each regime has a distinct distribution of HY spread outcomes."
    )
    try:
        _cr110_col = "credit_regime"
        if _cr110_col in df.columns:
            _cr110 = df[[_cr110_col, "spread", "unemployment", "hy_spread"]].copy()
            _cr110.index = pd.to_datetime(_cr110.index)
            _cur_cr = str(latest.get(_cr110_col, "Unknown"))

            _regime_colors110 = {
                "Expansion": "#27ae60",
                "Neutral": "#4f8ef7",
                "Late Cycle": "#f59e0b",
                "Credit Stress": "#ef4444",
            }
            _regime_desc110 = {
                "Expansion": "Steep curve + low unemployment — early-to-mid cycle. "
                             "HY spreads historically tight and tightening. Overweight credit.",
                "Neutral": "Mixed signals — balanced risk/reward. Maintain benchmark weight.",
                "Late Cycle": "Inverted curve + low unemployment — historically the highest-risk entry point. "
                              "Spreads still compressed but downside risk rising sharply. Reduce beta.",
                "Credit Stress": "Inverted curve + elevated unemployment — spread widening typically underway. "
                                 "Defensive positioning: IG, short-dated HY, cash.",
            }

            _streak110 = 0
            for _v in reversed((_cr110[_cr110_col] == _cur_cr).tolist()):
                if _v:
                    _streak110 += 1
                else:
                    break

            _ca110, _cb110, _cc110, _cd110 = st.columns(4)
            _ca110.metric("Current Regime", _cur_cr)
            _cb110.metric("2s10s Spread", f"{latest.get('spread', float('nan')):.2f}pp"
                          if pd.notna(latest.get("spread")) else "—")
            _cc110.metric("Unemployment", f"{latest.get('unemployment', float('nan')):.1f}%"
                          if pd.notna(latest.get("unemployment")) else "—")
            _cd110.metric("Streak", f"{_streak110}d")

            _cur_color110 = _regime_colors110.get(_cur_cr, "#6b7280")
            if _cur_cr in _regime_desc110:
                if _cur_cr == "Credit Stress":
                    st.error(_regime_desc110[_cur_cr])
                elif _cur_cr == "Late Cycle":
                    st.warning(_regime_desc110[_cur_cr])
                else:
                    st.info(_regime_desc110[_cur_cr])

            # Regime timeline
            _cr110_tail = _cr110.tail(756)
            _regime_num110 = {"Expansion": 4, "Neutral": 3, "Late Cycle": 2, "Credit Stress": 1}
            _cr110_tail = _cr110_tail.copy()
            _cr110_tail["regime_num"] = _cr110_tail[_cr110_col].map(_regime_num110).fillna(3)
            _bar_colors110 = [_regime_colors110.get(r, "#6b7280") for r in _cr110_tail[_cr110_col]]
            _fig110a = _go110.Figure()
            _fig110a.add_trace(_go110.Bar(
                x=_cr110_tail.index, y=_cr110_tail["regime_num"],
                marker_color=_bar_colors110, name="Credit Regime",
                hovertemplate="%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
                customdata=_cr110_tail[_cr110_col],
            ))
            _fig110a.update_layout(
                height=210, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Credit Regime (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(tickvals=[1, 2, 3, 4],
                           ticktext=["Stress", "Late Cycle", "Neutral", "Expansion"],
                           showgrid=False, color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig110a, use_container_width=True)

            # Full-history regime frequency
            _freq110 = _cr110[_cr110_col].value_counts(normalize=True).mul(100)
            _fig110b = _go110.Figure()
            _fig110b.add_trace(_go110.Bar(
                x=_freq110.index.tolist(), y=_freq110.values.tolist(),
                marker_color=[_regime_colors110.get(r, "#6b7280") for r in _freq110.index],
                text=[f"{v:.0f}%" for v in _freq110.values],
                textposition="auto",
                hovertemplate="%{x}: %{y:.0f}%<extra></extra>",
            ))
            _fig110b.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Regime Frequency (Full History)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="% days"),
                xaxis=dict(color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig110b, use_container_width=True)

            # HY spread distribution by regime
            if "hy_spread" in _cr110.columns:
                _fig110c = _go110.Figure()
                for _reg, _rcol in _regime_colors110.items():
                    _mask = _cr110[_cr110_col] == _reg
                    _hy_v = _cr110.loc[_mask, "hy_spread"].dropna()
                    if len(_hy_v) > 10:
                        _fig110c.add_trace(_go110.Box(
                            y=_hy_v, name=_reg, marker_color=_rcol,
                            line_color=_rcol, boxmean=True,
                            hovertemplate=f"{_reg}<br>HY: %{{y:.0f}}bps<extra></extra>",
                        ))
                _fig110c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="HY Spread by Credit Regime (full history)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread (bps)"),
                    xaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig110c, use_container_width=True)
                st.caption("Late Cycle median HY spread is often tight — the danger is the *transition* to Credit Stress, not the level itself")

            # Transition frequency table
            _cr110_shifted = _cr110[_cr110_col].shift(1)
            _transitions = pd.crosstab(_cr110_shifted, _cr110[_cr110_col], normalize="index").mul(100).round(1)
            if not _transitions.empty:
                st.markdown("**Regime Transition Matrix (% probability)**")
                st.dataframe(_transitions.style.format("{:.0f}%"), use_container_width=True)
                st.caption("Row = current regime · Column = next day's regime")
        else:
            st.info("credit_regime column not found — run the feature pipeline.")
    except Exception as _e110:
        _err_track(_active_sub, _e110)
        st.caption(f"Credit regime: {_e110}")


if _active_sub == 117:
    import plotly.graph_objects as _go117
    from src.regime_attribution import COMPOSITE_WEIGHTS as _CW117, DISPLAY_NAMES as _DN117
    st.header("Credit Market Risk Sub-Score")
    st.markdown(
        "The **credit market risk sub-score** (20% weight — largest single weight) is the most *directly* "
        "credit-native component of the composite. Its drivers are: **HY spread level and 30d/90d change**, "
        "**credit impulse** (second derivative of spread change), **VIX level and 30d change**, "
        "**credit-equity divergence**, and **vol-credit mismatch**. Because it incorporates both level and "
        "momentum, it can be elevated even when absolute spreads are moderate if they are rising rapidly. "
        "When this score leads the composite higher, it signals a *credit-specific* deterioration "
        "rather than broad macro stress."
    )
    try:
        _cmr117_col = "credit_market_risk_score_smooth"
        _cmr117_raw = "credit_market_risk_score"
        _cmr_col = _cmr117_col if _cmr117_col in df.columns else (_cmr117_raw if _cmr117_raw in df.columns else None)
        if _cmr_col:
            _cmr117 = df[[_cmr_col]].copy()
            for _c in ["hy_spread", "composite_risk_score_smooth", "hy_change_30d", "vix"]:
                if _c in df.columns:
                    _cmr117[_c] = df[_c]
            _cmr117.index = pd.to_datetime(_cmr117.index)
            _cur_cmr = float(latest.get(_cmr_col, float("nan")))
            _cmr_pctile = (df[_cmr_col].dropna() < _cur_cmr).mean() * 100 if pd.notna(_cur_cmr) else float("nan")
            _cmr_vel = float(df[_cmr_col].diff(21).iloc[-1]) if df[_cmr_col].notna().any() else float("nan")
            _cmr_contrib = _cur_cmr * _CW117.get("credit_risk", 0.20) if pd.notna(_cur_cmr) else float("nan")

            _ca117, _cb117, _cc117, _cd117 = st.columns(4)
            _ca117.metric("Credit Risk Score", f"{_cur_cmr:.0f}/100" if pd.notna(_cur_cmr) else "—")
            _cb117.metric("Composite Contrib", f"{_cmr_contrib:.1f}pts" if pd.notna(_cmr_contrib) else "—",
                          help=f"Score × {_CW117.get('credit_risk', 0.20):.0%} weight — highest single weight")
            _cc117.metric("Historical Pctile", f"{_cmr_pctile:.0f}th" if pd.notna(_cmr_pctile) else "—")
            _cd117.metric("21d Velocity", f"{_cmr_vel:+.1f}pts" if pd.notna(_cmr_vel) else "—",
                          delta_color="inverse")

            if pd.notna(_cur_cmr) and _cur_cmr >= 65:
                st.error("Credit market risk score at high levels — this is the highest-weight sub-score; "
                         "composite is likely in elevated or high risk territory.")

            _cmr_tail = _cmr117.tail(756)
            _fig117a = _go117.Figure()
            _fig117a.add_hrect(y0=65, y1=105, fillcolor="rgba(239,68,68,0.07)", line_width=0)
            _fig117a.add_hrect(y0=45, y1=65, fillcolor="rgba(245,158,11,0.05)", line_width=0)
            _fig117a.add_trace(_go117.Scatter(
                x=_cmr_tail.index, y=_cmr_tail[_cmr_col],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
                line=dict(color="#ef4444", width=2), name="Credit Risk Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            if "composite_risk_score_smooth" in _cmr_tail.columns:
                _fig117a.add_trace(_go117.Scatter(
                    x=_cmr_tail.index, y=_cmr_tail["composite_risk_score_smooth"],
                    line=dict(color="#e2e8f0", width=1, dash="dot"), name="Composite",
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
            _fig117a.add_hline(y=45, line=dict(color="rgba(245,158,11,0.4)", dash="dot", width=1))
            _fig117a.add_hline(y=65, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig117a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Credit Market Risk Sub-Score vs Composite (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig117a, use_container_width=True)

            # HY spread + VIX dual-axis (primary drivers)
            if "hy_spread" in _cmr_tail.columns and "vix" in _cmr_tail.columns:
                _fig117b = _go117.Figure()
                _fig117b.add_trace(_go117.Scatter(
                    x=_cmr_tail.index, y=_cmr_tail["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig117b.add_trace(_go117.Scatter(
                    x=_cmr_tail.index, y=_cmr_tail["vix"],
                    name="VIX", line=dict(color="#a78bfa", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>VIX: %{y:.1f}<extra></extra>",
                ))
                _fig117b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="Primary Drivers: HY Spread + VIX", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#f59e0b", title="HY (bps)"),
                    yaxis2=dict(overlaying="y", side="right", color="#a78bfa", title="VIX"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig117b, use_container_width=True)

            st.markdown("**Credit Market Risk Score — Driver Breakdown**")
            st.table(pd.DataFrame([
                {"Input": "HY Spread Level", "Pts": "Up to 35", "Trigger": ">600bps → 35pts; 400–600 → 22pts; 350–400 → 12pts"},
                {"Input": "HY Spread Δ30d", "Pts": "Up to 30", "Trigger": ">50bps → 30pts; 25–50bps → 20pts"},
                {"Input": "HY Spread Δ90d", "Pts": "Up to 25", "Trigger": "Sustained widening adds points"},
                {"Input": "Credit Impulse", "Pts": "Up to 15", "Trigger": "Acceleration in widening rate"},
                {"Input": "VIX + VIX Δ30d", "Pts": "Up to 20", "Trigger": "VIX>25 or rapid VIX rise"},
                {"Input": "Credit-Equity Div / Vol Mismatch", "Pts": "Up to 10", "Trigger": "Divergence/mismatch detected"},
            ]))
        else:
            st.info("Credit market risk score not found — run the full scoring pipeline.")
    except Exception as _e117:
        _err_track(_active_sub, _e117)
        st.caption(f"Credit market risk score: {_e117}")


if _active_sub == 130:
    try:
        import plotly.graph_objects as _go130
        import numpy as _np130
        from src.regime_attribution import SCORE_COLS, DISPLAY_NAMES
        if "hy_spread" in df.columns:
            _hy130 = df["hy_spread"].dropna()
            _hy_chg130 = _hy130.diff(21)   # 1M change in HY
            _lags130 = [0, 5, 10, 21, 42]  # lag in trading days
            _sc130_available = [(k, v) for k, v in SCORE_COLS.items() if v in df.columns]
            if _sc130_available:
                _results130 = {}
                for k, v in _sc130_available:
                    _score_chg = df[v].diff(21).dropna()
                    _corrs = []
                    for lag in _lags130:
                        # score leads HY by `lag` days => shift score forward by lag
                        _aligned = _score_chg.shift(-lag).align(_hy_chg130, join="inner")
                        _s, _h = _aligned
                        _valid = _s.dropna().align(_h.dropna(), join="inner")
                        if len(_valid[0]) > 30:
                            _corr_val = float(_valid[0].corr(_valid[1]))
                        else:
                            _corr_val = _np130.nan
                        _corrs.append(_corr_val)
                    _results130[DISPLAY_NAMES.get(k, k)] = _corrs
                # Heatmap of lag vs score
                _rl130 = list(_results130.keys())
                _corr_grid = [list(_results130[s]) for s in _rl130]
                _fig130a = _go130.Figure(data=_go130.Heatmap(
                    z=_corr_grid,
                    x=[f"+{lag}d" for lag in _lags130],
                    y=_rl130,
                    colorscale=[[0.0, "#1e40af"], [0.5, "#1a1f2e"], [1.0, "#dc2626"]],
                    zmin=-1, zmax=1,
                    text=[[f"{v:.2f}" if not _np130.isnan(v) else "" for v in row] for row in _corr_grid],
                    texttemplate="%{text}",
                    hovertemplate="%{y} at lag %{x}: corr=%{z:.2f}<extra></extra>",
                ))
                _fig130a.update_layout(
                    height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                    title=dict(text="Score→HY Lead-Lag Correlation (score change leads HY change by N days)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280", title="Score leads HY by"),
                    yaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig130a, use_container_width=True)
                # Best lag per score bar chart
                _best_lags = {}
                for name, corrs in _results130.items():
                    _best_idx = int(_np130.nanargmax([abs(c) for c in corrs]))
                    _best_lags[name] = (_lags130[_best_idx], corrs[_best_idx])
                _fig130b = _go130.Figure()
                _bl_names = list(_best_lags.keys())
                _bl_vals = [v[1] for v in _best_lags.values()]
                _bl_lags = [v[0] for v in _best_lags.values()]
                _fig130b.add_trace(_go130.Bar(
                    x=_bl_names, y=_bl_vals,
                    marker_color=["#ef4444" if v > 0 else "#3b82f6" for v in _bl_vals],
                    text=[f"lag={lag}d<br>r={corr:.2f}" for lag, corr in zip(_bl_lags, _bl_vals)],
                    textposition="auto", textfont=dict(size=9),
                    hovertemplate="%{x}<br>Best lag: %{text}<extra></extra>",
                ))
                _fig130b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Best Lead Correlation per Score (max |corr| across lags)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280", tickangle=-20),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Correlation"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig130b, use_container_width=True)
                st.caption("Red = positive correlation (score rise → HY widening). Lag = how many days score leads HY. Higher |corr| = stronger leading relationship.")
            else:
                st.info("Sub-score columns not found — run the full scoring pipeline.")
        else:
            st.info("hy_spread not found in dataset.")
    except Exception as _e130:
        _err_track(_active_sub, _e130)
        st.caption(f"HY lead-lag: {_e130}")

# sub131 — Recession Analog (tab_regime)

if _active_sub == 136:
    try:
        import plotly.graph_objects as _go136
        import numpy as _np136
        if "hy_spread" in df.columns:
            _hy136 = df["hy_spread"].dropna()
            # Rolling 30d std of HY spread (spread volatility)
            _spread_vol = _hy136.rolling(30, min_periods=10).std()
            # Rolling percentile of current spread vol
            _sv_pct = _spread_vol.rolling(504, min_periods=126).apply(
                lambda x: float((x[:-1] < x[-1]).mean() * 100) if len(x) > 1 else _np136.nan,
                raw=True
            )
            # Spread vol regime
            def _sv_regime(v):
                if _np136.isnan(v): return "Unknown"
                if v < 10:  return "Calm"
                if v < 20:  return "Normal"
                if v < 40:  return "Elevated"
                return "Turbulent"
            _sv_regime_series = _spread_vol.apply(_sv_regime)
            _sv_colors136 = {"Calm": "#10b981", "Normal": "#3b82f6", "Elevated": "#f59e0b", "Turbulent": "#ef4444", "Unknown": "#6b7280"}
            _fig136a = _go136.Figure()
            _fig136a.add_trace(_go136.Scatter(
                x=_spread_vol.index, y=_spread_vol.values,
                mode="lines", name="30d Spread Vol (bps std)",
                line=dict(color="#f59e0b", width=1.2),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.07)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}bps<extra></extra>",
            ))
            _fig136a.add_hline(y=20, line_color="#3b82f6", line_width=1, line_dash="dot",
                               annotation_text="Normal (20)", annotation_font=dict(color="#3b82f6", size=8))
            _fig136a.add_hline(y=40, line_color="#ef4444", line_width=1, line_dash="dot",
                               annotation_text="Turbulent (40)", annotation_font=dict(color="#ef4444", size=8))
            _fig136a.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="HY Spread 30-Day Rolling Volatility (std dev, bps)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Spread Vol (bps)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig136a, use_container_width=True)
            # Spread vol percentile over time
            _fig136b = _go136.Figure()
            _fig136b.add_trace(_go136.Scatter(
                x=_sv_pct.index, y=_sv_pct.values,
                mode="lines", name="Spread Vol Percentile",
                line=dict(color="#8b5cf6", width=1.0),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.0f}th pct<extra></extra>",
            ))
            _fig136b.add_hline(y=80, line_color="#ef4444", line_width=1, line_dash="dot")
            _fig136b.add_hline(y=20, line_color="#3b82f6", line_width=1, line_dash="dot")
            _fig136b.update_layout(
                height=175, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Spread Vol Percentile vs 2-Year Rolling History", font=dict(size=11, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Percentile", range=[0, 100]),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig136b, use_container_width=True)
            # Spread vol vs VIX scatter
            if "vix" in df.columns:
                _vix136 = df["vix"].dropna()
                _joined136 = _spread_vol.to_frame("sv").join(_vix136.to_frame("vix"), how="inner").dropna()
                if len(_joined136) > 30:
                    _fig136c = _go136.Figure()
                    _fig136c.add_trace(_go136.Scatter(
                        x=_joined136["vix"], y=_joined136["sv"],
                        mode="markers",
                        marker=dict(color="#f59e0b", size=2.5, opacity=0.3),
                        hovertemplate="VIX: %{x:.1f}<br>Spread Vol: %{y:.1f}bps<extra></extra>",
                    ))
                    _fig136c.update_layout(
                        height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                        title=dict(text="Spread Volatility vs VIX — Full History Scatter", font=dict(size=11, color="#9aa0aa")),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="VIX"),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="HY Spread Vol (bps)"),
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig136c, use_container_width=True)
            # Current reading
            _curr_sv = float(_spread_vol.iloc[-1]) if _spread_vol.notna().any() else None
            _curr_sv_pct = float(_sv_pct.iloc[-1]) if _sv_pct.notna().any() else None
            _curr_sv_regime = _sv_regime(_curr_sv if _curr_sv is not None else float("nan"))
            if _curr_sv is not None:
                st.caption(f"Current spread vol: {_curr_sv:.1f}bps std (30d) · {f'{_curr_sv_pct:.0f}th percentile' if _curr_sv_pct else 'N/A'} · Regime: **{_curr_sv_regime}**")
        else:
            st.info("hy_spread not found — run the feature pipeline.")
    except Exception as _e136:
        _err_track(_active_sub, _e136)
        st.caption(f"Spread vol regime: {_e136}")

# sub137 — Score Gradient (tab_siglab)

if _active_sub == 141:
    try:
        import plotly.graph_objects as _go141
        import numpy as _np141
        if "composite_risk_score_smooth" in df.columns and "hy_spread" in df.columns:
            _comp141 = df["composite_risk_score_smooth"].dropna()
            _hy141 = df["hy_spread"].dropna()
            _joined141 = _comp141.to_frame("comp").join(_hy141.to_frame("hy"), how="inner").dropna()
            def _regime141(s):
                if s < 25:  return "Low"
                if s < 40:  return "Moderate"
                if s < 55:  return "Elevated"
                if s < 70:  return "High"
                return "Extreme"
            _joined141["regime"] = _joined141["comp"].apply(_regime141)
            _REGIME_ORDER141 = ["Low", "Moderate", "Elevated", "High", "Extreme"]
            _REGIME_COLORS141 = {"Low": "#10b981", "Moderate": "#3b82f6", "Elevated": "#f59e0b",
                                  "High": "#ef4444", "Extreme": "#7f1d1d"}
            # Box plots: HY spread by regime
            _fig141a = _go141.Figure()
            for reg in _REGIME_ORDER141:
                _sub = _joined141[_joined141["regime"] == reg]["hy"]
                if len(_sub) < 5:
                    continue
                _fig141a.add_trace(_go141.Box(
                    y=_sub.values, name=reg,
                    marker_color=_REGIME_COLORS141[reg],
                    boxmean=True,
                    hovertemplate=f"{reg}: %{{y:.0f}}bps<extra></extra>",
                ))
            _fig141a.update_layout(
                height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="HY Spread Distribution by Composite Risk Regime", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="HY Spread (bps)"),
                xaxis=dict(color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig141a, use_container_width=True)
            # Summary stats table
            import pandas as _pd141
            _stats141 = []
            for reg in _REGIME_ORDER141:
                _sub = _joined141[_joined141["regime"] == reg]["hy"]
                if len(_sub) < 3:
                    continue
                _stats141.append({
                    "Regime": reg,
                    "N days": len(_sub),
                    "% of history": f"{len(_sub)/len(_joined141)*100:.0f}%",
                    "Median HY (bps)": round(float(_sub.median()), 0),
                    "Mean HY (bps)": round(float(_sub.mean()), 0),
                    "P25": round(float(_sub.quantile(0.25)), 0),
                    "P75": round(float(_sub.quantile(0.75)), 0),
                    "Max": round(float(_sub.max()), 0),
                })
            if _stats141:
                st.dataframe(_pd141.DataFrame(_stats141).set_index("Regime"), use_container_width=True)
            # VIX conditional distribution
            if "vix" in df.columns:
                _vix141 = df["vix"].dropna()
                _joined141b = _comp141.to_frame("comp").join(_vix141.to_frame("vix"), how="inner").dropna()
                _joined141b["regime"] = _joined141b["comp"].apply(_regime141)
                _fig141b = _go141.Figure()
                for reg in _REGIME_ORDER141:
                    _sub = _joined141b[_joined141b["regime"] == reg]["vix"]
                    if len(_sub) < 5:
                        continue
                    _fig141b.add_trace(_go141.Box(
                        y=_sub.values, name=reg,
                        marker_color=_REGIME_COLORS141[reg],
                        boxmean=True,
                        hovertemplate=f"{reg} VIX: %{{y:.1f}}<extra></extra>",
                    ))
                _fig141b.update_layout(
                    height=230, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="VIX Distribution by Composite Risk Regime", font=dict(size=11, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="VIX"),
                    xaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig141b, use_container_width=True)
            st.caption("Low <25 · Moderate 25-40 · Elevated 40-55 · High 55-70 · Extreme >70. Box = IQR, mean cross shown.")
        else:
            st.info("composite_risk_score_smooth or hy_spread not found — run the full scoring pipeline.")
    except Exception as _e141:
        _err_track(_active_sub, _e141)
        st.caption(f"Regime spread dist: {_e141}")

# sub142 — Credit Impulse Drill (Rates & Macro)

if _active_sub == 146:
    try:
        import plotly.graph_objects as _go146
        import numpy as _np146
        if "hy_spread" in df.columns and "sp500" in df.columns:
            _hy146 = df["hy_spread"].dropna()
            _sp146 = df["sp500"].dropna()
            _j146 = _hy146.to_frame("hy").join(_sp146.to_frame("sp"), how="inner").dropna()
            _j146["hy_chg"] = _j146["hy"].diff(21)
            _j146["sp_ret"] = _j146["sp"].pct_change(21) * 100
            _j146 = _j146.dropna()

            # Rolling 63d beta: HY change / SP500 return
            def _roll_beta(window=63):
                betas = []
                for i in range(len(_j146)):
                    if i < window:
                        betas.append(_np146.nan)
                        continue
                    _slice = _j146.iloc[i-window:i]
                    _x = _slice["sp_ret"].values
                    _y = _slice["hy_chg"].values
                    _cov = _np146.cov(_x, _y)
                    _var = _np146.var(_x, ddof=1)
                    betas.append(float(_cov[0, 1] / _var) if _var > 0 else _np146.nan)
                return betas
            import pandas as _pd146
            _j146["beta_63d"] = _roll_beta(63)

            _fig146a = _go146.Figure()
            _fig146a.add_trace(_go146.Scatter(
                x=_j146.index, y=_j146["beta_63d"],
                mode="lines", line=dict(color="#f59e0b", width=1.2),
                hovertemplate="%{x|%Y-%m-%d}: β=%{y:.2f}<extra></extra>",
            ))
            _fig146a.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig146a.add_hline(y=-5, line_color="#ef4444", line_width=1, line_dash="dot",
                               annotation_text="Stress zone (β<−5)", annotation_font=dict(color="#ef4444", size=8))
            _fig146a.update_layout(
                height=230, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Rolling 63d Credit Beta: HY Spread Change per 1% SP500 Return (bps/%)",
                           font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="β (bps per %)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig146a, use_container_width=True)

            # Beta by composite regime
            if "composite_risk_score_smooth" in df.columns:
                _comp146 = df["composite_risk_score_smooth"].dropna()
                _j146b = _j146.join(_comp146.to_frame("comp"), how="inner").dropna(subset=["beta_63d", "comp"])
                def _reg146(s):
                    if s < 25: return "Low"
                    if s < 40: return "Moderate"
                    if s < 55: return "Elevated"
                    if s < 70: return "High"
                    return "Extreme"
                _j146b["regime"] = _j146b["comp"].apply(_reg146)
                _REGIME_ORDER146 = ["Low", "Moderate", "Elevated", "High", "Extreme"]
                _REG_COLORS146 = {"Low": "#10b981", "Moderate": "#3b82f6", "Elevated": "#f59e0b",
                                   "High": "#ef4444", "Extreme": "#7f1d1d"}
                _fig146b = _go146.Figure()
                for reg in _REGIME_ORDER146:
                    _sub = _j146b[_j146b["regime"] == reg]["beta_63d"].dropna()
                    if len(_sub) < 5:
                        continue
                    _fig146b.add_trace(_go146.Box(
                        y=_sub.values, name=reg,
                        marker_color=_REG_COLORS146[reg],
                        boxmean=True,
                        hovertemplate=f"{reg}: β=%{{y:.1f}}<extra></extra>",
                    ))
                _fig146b.add_hline(y=0, line_color="#4b5563", line_width=1)
                _fig146b.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Credit Beta Distribution by Composite Regime", font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="β (bps per %)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig146b, use_container_width=True)
                _stats146 = []
                for reg in _REGIME_ORDER146:
                    _sub = _j146b[_j146b["regime"] == reg]["beta_63d"].dropna()
                    if len(_sub) < 3: continue
                    _stats146.append({"Regime": reg, "Median β": round(float(_sub.median()), 1),
                                       "Mean β": round(float(_sub.mean()), 1), "N": len(_sub)})
                if _stats146:
                    st.dataframe(_pd146.DataFrame(_stats146).set_index("Regime"), use_container_width=True)
            _curr_beta = float(_j146["beta_63d"].iloc[-1]) if _j146["beta_63d"].notna().any() else None
            st.caption(
                f"Current 63d credit beta: **{f'{_curr_beta:.1f} bps per 1% SP500 return' if _curr_beta else 'N/A'}**. "
                "Negative beta = equity rallies associated with spread tightening (normal). "
                "More negative in High/Extreme regimes = amplified co-movement."
            )
        else:
            st.info("hy_spread or sp500 not found.")
    except Exception as _e146:
        _err_track(_active_sub, _e146)
        st.caption(f"Credit beta: {_e146}")


# sub147 — Score Seasonality (Signal Lab)

if _active_sub == 149:
    try:
        import plotly.graph_objects as _go149
        import numpy as _np149
        _horizon_cols = [
            ("5d", "hy_change_5d"),
            ("30d", "hy_change_30d"),
            ("90d", "hy_change_90d"),
        ]
        _available149 = [(lbl, col) for lbl, col in _horizon_cols if col in df.columns]
        if _available149:
            _colors149 = {"5d": "#06b6d4", "30d": "#f59e0b", "90d": "#ef4444"}
            # Multi-horizon overlay
            _fig149a = _go149.Figure()
            for lbl, col in _available149:
                _s = df[col].dropna()
                _fig149a.add_trace(_go149.Scatter(
                    x=_s.index, y=_s.values,
                    mode="lines", name=f"HY Δ{lbl}",
                    line=dict(color=_colors149[lbl], width=1.1),
                    hovertemplate=f"Δ{lbl}: %{{y:+.0f}}bps<extra></extra>",
                ))
            _fig149a.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig149a.update_layout(
                height=250, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="HY Spread: Multi-Horizon Momentum (5d / 30d / 90d Change)",
                           font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Spread Change (bps)", zeroline=True, zerolinecolor="#4b5563"),
                legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig149a, use_container_width=True)

            # Momentum alignment: are all horizons pointing same direction?
            _last149 = {}
            for lbl, col in _available149:
                _s = df[col].dropna()
                if len(_s): _last149[lbl] = float(_s.iloc[-1])
            if _last149:
                _all_widen = all(v > 5 for v in _last149.values())
                _all_tight = all(v < -5 for v in _last149.values())
                _mixed = not _all_widen and not _all_tight
                _align_txt = ("🔴 All horizons widening — broad spread pressure" if _all_widen
                               else ("🟢 All horizons tightening — broad credit improvement" if _all_tight
                                     else "🟡 Mixed — horizons not aligned"))
                st.info(_align_txt)
                _import149 = __import__("pandas")
                _snap149 = _import149.DataFrame(
                    [{"Horizon": lbl, "Change (bps)": round(_last149[lbl], 0),
                      "Direction": "Widening" if _last149[lbl] > 5 else ("Tightening" if _last149[lbl] < -5 else "Flat")}
                     for lbl in _last149],
                ).set_index("Horizon")
                st.dataframe(_snap149, use_container_width=True)

            # Scatter: 5d vs 90d — short-term shock vs long-term trend
            if "hy_change_5d" in df.columns and "hy_change_90d" in df.columns:
                _j149 = df[["hy_change_5d","hy_change_90d"]].dropna()
                _fig149b = _go149.Figure()
                _fig149b.add_trace(_go149.Scatter(
                    x=_j149["hy_change_90d"], y=_j149["hy_change_5d"],
                    mode="markers",
                    marker=dict(color="#f59e0b", size=2.5, opacity=0.3),
                    hovertemplate="90d: %{x:+.0f}<br>5d: %{y:+.0f}<extra></extra>",
                ))
                _fig149b.add_hline(y=0, line_color="#4b5563", line_width=1)
                _fig149b.add_vline(x=0, line_color="#4b5563", line_width=1)
                _fig149b.update_layout(
                    height=210, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="5d vs 90d HY Change: Short-Term Shock vs Long-Term Trend (bps)",
                               font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="90d Δ (bps)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="5d Δ (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig149b, use_container_width=True)
                st.caption("Q1 (top-right): short + long widening = momentum. Q3 (bottom-left): both tightening. "
                           "Q2 (top-left): short spike vs long tightening = potential reversal. "
                           "Q4 (bottom-right): short easing vs long widening = late-cycle.")
        else:
            st.info("hy_change_5d / hy_change_30d / hy_change_90d not found — run the feature pipeline.")
    except Exception as _e149:
        _err_track(_active_sub, _e149)
        st.caption(f"HY multi-horizon: {_e149}")


# sub150 — Score Ensemble Conviction (Signal Lab)

if _active_sub == 151:
    try:
        import plotly.graph_objects as _go151
        import numpy as _np151
        import pandas as _pd151
        _df151 = df.copy() if "df" in dir() else None
        if _df151 is None or "hy_spread" not in _df151.columns:
            st.info("hy_spread column required.")
        else:
            st.subheader("Spread Dispersion Monitor")
            st.caption("Rolling 21-day volatility of daily HY spread changes captures whether stress is broad-based or idiosyncratic. High dispersion with moderate mean widening = selective stress; low dispersion with widening = systemic.")
            _hy151 = _df151["hy_spread"].dropna()
            _hy_d151 = _hy151.diff(1)
            _roll_std151 = _hy_d151.rolling(21).std()
            _roll_mean151 = _hy_d151.rolling(21).mean()
            _roll_pct151 = _roll_std151.rolling(252).rank(pct=True) * 100
            _fig151 = _go151.Figure()
            _fig151.add_trace(_go151.Scatter(
                x=_roll_std151.index, y=_roll_std151.values,
                name="21d HY Change Vol", line=dict(color="#ef4444", width=1.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"
            ))
            _fig151.update_layout(
                title="Rolling 21d HY Spread Change Volatility (bps/day)",
                height=300, yaxis_title="Bps",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig151, use_container_width=True)
            _fig151b = _go151.Figure()
            _fig151b.add_trace(_go151.Scatter(
                x=_roll_pct151.index, y=_roll_pct151.values,
                fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
                line=dict(color="#ef4444", width=1), name="Dispersion Percentile"
            ))
            _fig151b.add_hline(y=80, line_dash="dash", line_color="#f59e0b",
                               annotation_text="80th pct", annotation_position="right")
            _fig151b.update_layout(
                title="Dispersion Percentile (vs 252d history)",
                height=240, yaxis_title="Percentile", yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig151b, use_container_width=True)
            _cur_std151 = float(_roll_std151.iloc[-1]) if _roll_std151.notna().any() else float("nan")
            _cur_pct151 = float(_roll_pct151.iloc[-1]) if _roll_pct151.notna().any() else float("nan")
            _cur_mean151 = float(_roll_mean151.iloc[-1]) if _roll_mean151.notna().any() else float("nan")
            if not _np151.isnan(_cur_std151):
                _reg151 = ("Crisis Volatility" if _cur_pct151 > 90 else
                           ("Elevated" if _cur_pct151 > 70 else
                            ("Normal" if _cur_pct151 > 30 else "Suppressed")))
                st.caption(
                    f"Current 21d HY change vol: **{_cur_std151:.2f} bps/day** "
                    f"({_cur_pct151:.0f}th pct) — {_reg151}. "
                    f"Mean daily change: {_cur_mean151:+.2f} bps."
                )
    except Exception as _e151:
        _err_track(_active_sub, _e151)
        st.caption(f"Spread dispersion: {_e151}")


if _active_sub == 156:
    try:
        import plotly.graph_objects as _go156
        import numpy as _np156
        import pandas as _pd156
        _df156 = df.copy() if "df" in dir() else None
        _has156 = (_df156 is not None
                   and "hy_spread" in _df156.columns
                   and "real_yield_proxy" in _df156.columns)
        if not _has156:
            st.info("hy_spread and real_yield_proxy required.")
        else:
            st.subheader("Credit Carry Decomposition")
            st.caption("HY OAS decomposed into an expected-loss proxy (scaled from real yield stress level) and excess carry — the risk premium above expected default losses. Thin excess carry = expensive credit; thick excess carry = better value. Percentile tracks richness/cheapness historically.")
            _hy156 = _df156["hy_spread"].dropna()
            _real156 = _df156["real_yield_proxy"].dropna()
            _j156 = _hy156.to_frame("hy").join(_real156.to_frame("real"), how="inner").dropna().tail(1260)
            _rl_min156 = _j156["real"].quantile(0.05)
            _rl_max156 = _j156["real"].quantile(0.95)
            _rl_norm156 = (_j156["real"] - _rl_min156) / (_rl_max156 - _rl_min156 + 1e-9)
            _exp_loss156 = (_rl_norm156 * 250).clip(0, 400)
            _excess156 = (_j156["hy"] - _exp_loss156).clip(0)
            _fig156 = _go156.Figure()
            _fig156.add_trace(_go156.Scatter(
                x=_j156.index, y=_j156["hy"].values,
                name="HY OAS (total)", line=dict(color="#ef4444", width=1.5)
            ))
            _fig156.add_trace(_go156.Scatter(
                x=_j156.index, y=_exp_loss156.values,
                name="Expected Loss Proxy", line=dict(color="#f59e0b", width=1, dash="dash"),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"
            ))
            _fig156.add_trace(_go156.Scatter(
                x=_j156.index, y=_excess156.values,
                name="Excess Carry", line=dict(color="#22c55e", width=1.5)
            ))
            _fig156.update_layout(
                title="HY Carry Decomposition — Total vs Expected Loss vs Excess (5Y)",
                height=380, yaxis_title="Bps",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig156, use_container_width=True)
            _cur_carry156 = float(_excess156.iloc[-1]) if _excess156.notna().any() else float("nan")
            _carry_pct156 = float((_excess156 < _cur_carry156).mean() * 100) if not _np156.isnan(_cur_carry156) else float("nan")
            _cur_hy156 = float(_j156["hy"].iloc[-1])
            _cur_el156 = float(_exp_loss156.iloc[-1])
            if not _np156.isnan(_cur_carry156):
                _val156 = ("Rich" if _carry_pct156 < 30 else
                           ("Cheap" if _carry_pct156 > 70 else "Fair Value"))
                st.caption(
                    f"Current HY OAS: **{_cur_hy156:.0f} bps** · "
                    f"Expected loss proxy: **{_cur_el156:.0f} bps** · "
                    f"Excess carry: **{_cur_carry156:.0f} bps** ({_carry_pct156:.0f}th pct) · "
                    f"Valuation: **{_val156}**."
                )
    except Exception as _e156:
        _err_track(_active_sub, _e156)
        st.caption(f"Credit carry decomp: {_e156}")


if _active_sub == 157:
    try:
        import plotly.graph_objects as _go157
        import numpy as _np157
        import pandas as _pd157
        _df157 = df.copy() if "df" in dir() else None
        _has157 = (_df157 is not None
                   and "hy_spread" in _df157.columns
                   and "hy_change_90d" in _df157.columns)
        if not _has157:
            st.info("hy_spread and hy_change_90d required.")
        else:
            st.subheader("Credit Cycle Clock")
            st.caption("Phase-space plot tracing the credit cycle: X-axis = HY level (stress), Y-axis = 90d change in HY (momentum). The path over the last 252 days reveals cycle position — Tightening (upper-left→lower-left), Recovery (lower-left), Widening (lower-right→upper-right), Stress (upper-right).")
            _hy157 = _df157["hy_spread"].dropna()
            _mom157 = _df157["hy_change_90d"].dropna()
            _j157 = _hy157.to_frame("hy").join(_mom157.to_frame("mom"), how="inner").dropna().tail(504)
            _path157 = _j157.tail(252)
            _n157 = len(_path157)
            _alphas157 = _np157.linspace(0.15, 1.0, _n157)
            _fig157 = _go157.Figure()
            # Full history scatter (faded)
            _fig157.add_trace(_go157.Scatter(
                x=_j157["hy"], y=_j157["mom"],
                mode="markers", marker=dict(color="#374151", size=3),
                name="History (2Y)", showlegend=True
            ))
            # Animated path — last 252d colored by recency
            for _i157 in range(max(0, _n157 - 50), _n157 - 1):
                _a157 = _alphas157[_i157]
                _fig157.add_trace(_go157.Scatter(
                    x=_path157["hy"].iloc[_i157:_i157+2],
                    y=_path157["mom"].iloc[_i157:_i157+2],
                    mode="lines",
                    line=dict(color=f"rgba(99,102,241,{_a157:.2f})", width=2),
                    showlegend=False
                ))
            # Current position
            _fig157.add_trace(_go157.Scatter(
                x=[_path157["hy"].iloc[-1]], y=[_path157["mom"].iloc[-1]],
                mode="markers+text",
                marker=dict(color="white", size=12, symbol="circle"),
                text=["Now"], textposition="top center",
                name="Current"
            ))
            # Quadrant lines
            _hy_med157 = float(_j157["hy"].median())
            _fig157.add_vline(x=_hy_med157, line_dash="dash", line_color="#4b5563",
                              annotation_text="Median HY")
            _fig157.add_hline(y=0, line_dash="dash", line_color="#4b5563")
            # Quadrant labels
            _hy_max157 = float(_j157["hy"].quantile(0.9))
            _mom_max157 = float(_j157["mom"].quantile(0.85))
            _mom_min157 = float(_j157["mom"].quantile(0.15))
            for _txt157, _x157, _y157 in [
                ("Recovery", _hy_med157 * 0.6, _mom_min157 * 0.6),
                ("Stress Build", _hy_max157 * 0.9, _mom_max157 * 0.6),
                ("Tightening", _hy_med157 * 0.6, _mom_max157 * 0.5),
                ("Stress Peak", _hy_max157 * 0.9, _mom_min157 * 0.6),
            ]:
                _fig157.add_annotation(
                    x=_x157, y=_y157, text=_txt157,
                    showarrow=False, font=dict(color="#6b7280", size=10)
                )
            _fig157.update_layout(
                title="Credit Cycle Phase-Space (252d path)",
                height=450, xaxis_title="HY Spread Level (bps)",
                yaxis_title="90d HY Change (bps, momentum)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig157, use_container_width=True)
            _cur_hy157 = float(_path157["hy"].iloc[-1])
            _cur_mom157 = float(_path157["mom"].iloc[-1])
            _cycle_quad157 = ("Stress" if _cur_hy157 > _hy_med157 and _cur_mom157 > 0 else
                              ("Deteriorating" if _cur_hy157 <= _hy_med157 and _cur_mom157 > 0 else
                               ("Recovery" if _cur_hy157 <= _hy_med157 and _cur_mom157 <= 0 else "Tightening")))
            st.caption(
                f"Current: HY {_cur_hy157:.0f} bps, 90d momentum {_cur_mom157:+.0f} bps → **{_cycle_quad157}** phase."
            )
    except Exception as _e157:
        _err_track(_active_sub, _e157)
        st.caption(f"Credit cycle clock: {_e157}")


if _active_sub == 158:
    try:
        import plotly.graph_objects as _go158
        import numpy as _np158
        import pandas as _pd158
        _df158 = df.copy() if "df" in dir() else None
        _has158 = _df158 is not None and "hy_change_5d" in _df158.columns
        if not _has158:
            st.info("hy_change_5d required.")
        else:
            st.subheader("Spread Compression Velocity")
            st.caption("Acceleration of HY spread changes: the rate of change of the 5-day change (second derivative). Positive velocity = widening is speeding up; negative = tightening gaining pace. Extreme readings identify turning points and momentum exhaustion.")
            _mom158 = _df158["hy_change_5d"].dropna()
            _vel158 = _mom158.diff(5)  # 5d change in the 5d change = acceleration
            _vel_z158 = _vel158.rolling(252).apply(
                lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-9), raw=False)
            _pct158 = _vel158.rolling(252).rank(pct=True) * 100
            _fig158a = _go158.Figure()
            _fig158a.add_trace(_go158.Bar(
                x=_vel158.tail(252).index,
                y=_vel158.tail(252).values,
                marker_color=_np158.where(_vel158.tail(252).values > 0, "#ef4444", "#22c55e"),
                name="Spread Velocity"
            ))
            _fig158a.update_layout(
                title="HY Spread Change Acceleration (last 252d)",
                height=300, yaxis_title="Δ(5d change) in bps",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig158a, use_container_width=True)
            _fig158b = _go158.Figure()
            _fig158b.add_trace(_go158.Scatter(
                x=_pct158.tail(252).index, y=_pct158.tail(252).values,
                line=dict(color="#f59e0b", width=1.5),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"
            ))
            _fig158b.add_hline(y=80, line_dash="dash", line_color="#ef4444",
                               annotation_text="Extreme widening pace")
            _fig158b.add_hline(y=20, line_dash="dash", line_color="#22c55e",
                               annotation_text="Extreme tightening pace")
            _fig158b.update_layout(
                title="Velocity Percentile (vs 252d history)",
                height=240, yaxis_title="Percentile", yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig158b, use_container_width=True)
            _cur_vel158 = float(_vel158.iloc[-1]) if _vel158.notna().any() else float("nan")
            _cur_pct158 = float(_pct158.iloc[-1]) if _pct158.notna().any() else float("nan")
            if not _np158.isnan(_cur_vel158):
                _vdir158 = "accelerating wider" if _cur_vel158 > 0 else "accelerating tighter"
                st.caption(
                    f"Current velocity: **{_cur_vel158:+.1f} bps** ({_cur_pct158:.0f}th pct) — "
                    f"spreads are {_vdir158}."
                )
    except Exception as _e158:
        _err_track(_active_sub, _e158)
        st.caption(f"Spread velocity: {_e158}")


if _active_sub == 163:
    try:
        import plotly.graph_objects as _go163
        import numpy as _np163
        import pandas as _pd163
        _df163 = df.copy() if "df" in dir() else None
        _has163 = (_df163 is not None
                   and "hy_spread" in _df163.columns
                   and "vix" in _df163.columns
                   and "sp500_drawdown" in _df163.columns)
        if not _has163:
            st.info("hy_spread, vix, and sp500_drawdown required.")
        else:
            st.subheader("Credit Risk Appetite Index")
            st.caption("Composite risk appetite indicator: combines HY/VIX ratio (credit vs equity fear) and SP500 drawdown depth. High index = investors are reaching for credit risk (tight spreads relative to equity vol, shallow drawdown). Low index = de-risking. Tracks investor willingness to hold credit risk.")
            _hy163 = _df163["hy_spread"].dropna()
            _vix163 = _df163["vix"].dropna()
            _dd163 = _df163["sp500_drawdown"].dropna()
            _j163 = _hy163.to_frame("hy").join(_vix163.to_frame("vix"), how="inner").join(
                _dd163.to_frame("dd"), how="inner").dropna().tail(1260)
            # HY/VIX ratio — lower = more appetite (tight spreads per unit of VIX)
            _ratio163 = _j163["hy"] / _j163["vix"]
            # Normalize ratio: invert and scale 0-100 (low ratio = high appetite)
            _ratio_z163 = (_ratio163.rolling(252).rank(pct=True) * 100).rsub(100)  # invert
            # Drawdown component: shallow drawdown = high appetite
            _dd_z163 = (_j163["dd"].rolling(252).rank(pct=True) * 100).rsub(100)  # invert (less negative = higher)
            # Composite: 60% ratio, 40% drawdown
            _appetite163 = 0.6 * _ratio_z163 + 0.4 * _dd_z163
            _fig163 = _go163.Figure()
            _fig163.add_trace(_go163.Scatter(
                x=_appetite163.index, y=_appetite163.values,
                line=dict(color="#22c55e", width=1.5),
                fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
                name="Risk Appetite Index"
            ))
            _fig163.add_hline(y=70, line_dash="dash", line_color="#22c55e",
                              annotation_text="High appetite (70)", annotation_position="right")
            _fig163.add_hline(y=30, line_dash="dash", line_color="#ef4444",
                              annotation_text="Low appetite (30)", annotation_position="right")
            _fig163.add_hline(y=50, line_color="#9aa0aa", line_width=0.5)
            _fig163.update_layout(
                title="Credit Risk Appetite Index (0=Max Fear, 100=Max Greed)",
                height=360, yaxis_title="Index", yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig163, use_container_width=True)
            # HY/VIX ratio chart
            _fig163b = _go163.Figure()
            _fig163b.add_trace(_go163.Scatter(
                x=_ratio163.index, y=_ratio163.values,
                line=dict(color="#6366f1", width=1), name="HY/VIX Ratio"
            ))
            _fig163b.update_layout(
                title="HY Spread / VIX Ratio (lower = more appetite)",
                height=220, yaxis_title="Ratio",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig163b, use_container_width=True)
            _cur_app163 = float(_appetite163.iloc[-1]) if _appetite163.notna().any() else float("nan")
            _cur_ratio163 = float(_ratio163.iloc[-1]) if _ratio163.notna().any() else float("nan")
            if not _np163.isnan(_cur_app163):
                _app_label163 = ("Extreme Greed" if _cur_app163 > 80 else
                                 ("Greed" if _cur_app163 > 60 else
                                  ("Neutral" if _cur_app163 > 40 else
                                   ("Fear" if _cur_app163 > 20 else "Extreme Fear"))))
                st.caption(
                    f"Current risk appetite: **{_cur_app163:.0f}/100** — {_app_label163}. "
                    f"HY/VIX ratio: {_cur_ratio163:.1f}."
                )
    except Exception as _e163:
        _err_track(_active_sub, _e163)
        st.caption(f"Risk appetite: {_e163}")


if _active_sub == 164:
    try:
        import plotly.graph_objects as _go164
        import numpy as _np164
        import pandas as _pd164
        _df164 = df.copy() if "df" in dir() else None
        _has164 = _df164 is not None and "hy_spread" in _df164.columns
        if not _has164:
            st.info("hy_spread required.")
        else:
            st.subheader("Carry Efficiency (Sharpe-like)")
            st.caption("HY spread level divided by its rolling 63-day realized volatility — a credit Sharpe ratio. High ratio = spread is large relative to its volatility (attractive carry-per-unit-of-risk). Low ratio = spreads are tight or unusually volatile (poor compensation). Rolling percentile tracks richness/cheapness.")
            _hy164 = _df164["hy_spread"].dropna()
            _hy_vol164 = _hy164.rolling(63).std()
            _carry_eff164 = (_hy164 / (_hy_vol164 + 1e-9)).replace([_np164.inf, -_np164.inf], _np164.nan)
            _eff_pct164 = _carry_eff164.rolling(252).rank(pct=True) * 100
            _c1_164, _c2_164 = st.columns(2)
            with _c1_164:
                _fig164a = _go164.Figure()
                _fig164a.add_trace(_go164.Scatter(
                    x=_carry_eff164.tail(504).index, y=_carry_eff164.tail(504).values,
                    line=dict(color="#22c55e", width=1.5), name="Carry Efficiency"
                ))
                _fig164a.update_layout(
                    title="Carry Efficiency Ratio (HY / 63d Vol)",
                    height=300, yaxis_title="Ratio",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    showlegend=False
                )
                st.plotly_chart(_fig164a, use_container_width=True)
            with _c2_164:
                _fig164b = _go164.Figure()
                _fig164b.add_trace(_go164.Scatter(
                    x=_eff_pct164.tail(504).index, y=_eff_pct164.tail(504).values,
                    fill="tozeroy", fillcolor="rgba(34,197,94,0.1)",
                    line=dict(color="#22c55e", width=1), name="Efficiency Percentile"
                ))
                _fig164b.add_hline(y=70, line_dash="dash", line_color="#22c55e",
                                   annotation_text="Cheap (70)", annotation_position="right")
                _fig164b.add_hline(y=30, line_dash="dash", line_color="#ef4444",
                                   annotation_text="Rich (30)", annotation_position="right")
                _fig164b.update_layout(
                    title="Carry Efficiency Percentile (252d)",
                    height=300, yaxis_title="Pct", yaxis=dict(range=[0, 100]),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    showlegend=False
                )
                st.plotly_chart(_fig164b, use_container_width=True)
            # Distribution
            _eff_hist164 = _carry_eff164.dropna()
            _fig164c = _go164.Figure()
            _fig164c.add_trace(_go164.Histogram(
                x=_eff_hist164.values, nbinsx=40,
                marker_color="#6366f1", opacity=0.7, name="Distribution"
            ))
            _cur_eff164 = float(_carry_eff164.iloc[-1]) if _carry_eff164.notna().any() else float("nan")
            if not _np164.isnan(_cur_eff164):
                _fig164c.add_vline(x=_cur_eff164, line_color="white", line_width=2,
                                   annotation_text=f"Now: {_cur_eff164:.2f}")
            _fig164c.update_layout(
                title="Carry Efficiency Distribution (full history)",
                height=240, xaxis_title="Efficiency Ratio",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig164c, use_container_width=True)
            _cur_pct164 = float(_eff_pct164.iloc[-1]) if _eff_pct164.notna().any() else float("nan")
            if not _np164.isnan(_cur_eff164):
                _val164 = ("Cheap (high carry/vol)" if _cur_pct164 > 70 else
                           ("Rich (low carry/vol)" if _cur_pct164 < 30 else "Fair"))
                st.caption(
                    f"Current carry efficiency: **{_cur_eff164:.2f}** ({_cur_pct164:.0f}th pct) — {_val164}."
                )
    except Exception as _e164:
        _err_track(_active_sub, _e164)
        st.caption(f"Carry efficiency: {_e164}")


if _active_sub == 172:
    try:
        import plotly.graph_objects as _go172
        import numpy as _np172
        import pandas as _pd172
        _df172 = df.copy() if "df" in dir() else None
        _cols172 = ["hy_change_5d", "hy_change_30d", "hy_change_90d"]
        _has172 = _df172 is not None and all(c in _df172.columns for c in _cols172)
        if not _has172:
            st.info("hy_change_5d, hy_change_30d, hy_change_90d required.")
        else:
            st.subheader("HY Momentum Term Structure")
            st.caption("The 'term structure' of HY spread momentum: 5d, 30d, and 90d changes plotted together reveal whether short-term or long-term credit momentum dominates. An upward-sloping structure (5d > 30d > 90d) = recent stress spike. Downward slope = long-term widening, short-term stabilizing.")
            _j172 = _df172[_cols172].dropna().tail(504)
            _fig172a = _go172.Figure()
            for _col172, _name172, _color172 in [
                ("hy_change_5d", "5d Change", "#ef4444"),
                ("hy_change_30d", "30d Change", "#f59e0b"),
                ("hy_change_90d", "90d Change", "#6366f1"),
            ]:
                _fig172a.add_trace(_go172.Scatter(
                    x=_j172.index, y=_j172[_col172],
                    name=_name172, line=dict(color=_color172, width=1.5)
                ))
            _fig172a.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig172a.update_layout(
                title="HY Spread Momentum at 5d / 30d / 90d (bps)",
                height=320, yaxis_title="Change (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig172a, use_container_width=True)
            # Term structure steepness: 5d minus 90d
            _steep172 = _j172["hy_change_5d"] - _j172["hy_change_90d"]
            _fig172b = _go172.Figure()
            _fig172b.add_trace(_go172.Scatter(
                x=_steep172.index, y=_steep172.values,
                fill="tozeroy",
                fillcolor=["rgba(239,68,68,0.1)" if v > 0 else "rgba(34,197,94,0.1)"][0],
                line=dict(color="#f59e0b", width=1.5), name="5d − 90d"
            ))
            _fig172b.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig172b.update_layout(
                title="Term Structure Steepness: 5d Minus 90d Change (bps)",
                height=220, yaxis_title="Bps",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig172b, use_container_width=True)
            # Current snapshot bar
            _cur172 = _j172.iloc[-1]
            _horizons_172 = [5, 30, 90]
            _cur_vals172 = [float(_cur172["hy_change_5d"]), float(_cur172["hy_change_30d"]), float(_cur172["hy_change_90d"])]
            _fig172c = _go172.Figure()
            _fig172c.add_trace(_go172.Bar(
                x=[f"{h}d" for h in _horizons_172], y=_cur_vals172,
                marker_color=["#ef4444" if v > 0 else "#22c55e" for v in _cur_vals172],
                text=[f"{v:+.0f}" for v in _cur_vals172], textposition="outside"
            ))
            _fig172c.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig172c.update_layout(
                title="Current HY Momentum Term Structure",
                height=240, yaxis_title="Bps",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig172c, use_container_width=True)
            _shape172 = ("Upward (recent spike)" if _cur_vals172[0] > _cur_vals172[2] + 5
                         else ("Downward (long-term pressure)" if _cur_vals172[0] < _cur_vals172[2] - 5
                               else "Flat"))
            st.caption(
                f"Current: 5d={_cur_vals172[0]:+.0f} · 30d={_cur_vals172[1]:+.0f} · 90d={_cur_vals172[2]:+.0f} bps · "
                f"Structure: **{_shape172}**."
            )
    except Exception as _e172:
        _err_track(_active_sub, _e172)
        st.caption(f"HY term structure: {_e172}")


if _active_sub == 174:
    try:
        import plotly.graph_objects as _go174
        import numpy as _np174
        import pandas as _pd174
        _df174 = df.copy() if "df" in dir() else None
        _indicators174 = [
            ("hy_spread", "HY Spread", True),           # above MA = stress
            ("vix", "VIX", True),
            ("nfci", "NFCI", True),
            ("unemployment", "Unemployment", True),
            ("sahm_like", "Sahm-like", True),
            ("hy_change_30d", "HY 30d Change", True),
            ("sp500_return_30d", "SP500 30d Return", False),  # below MA = stress
            ("sp500_drawdown", "SP500 Drawdown", True),
            ("real_yield_proxy", "Real Yield", True),
            ("nfci_change_90d", "NFCI Trend", True),
        ]
        _avail174 = [(c, n, d) for c, n, d in _indicators174
                     if _df174 is not None and c in _df174.columns]
        if len(_avail174) < 3:
            st.info("At least 3 indicator columns required.")
        else:
            st.subheader("Credit Conditions Diffusion Index")
            st.caption("Breadth-based credit conditions index: count of indicators above (or below, for inverted signals) their own 63-day moving average. High diffusion = broad deterioration across multiple channels simultaneously. Low diffusion = improvement is broad-based. Unlike level indicators, diffusion captures momentum breadth.")
            _ma_window174 = 63
            _results174 = []
            for _col174, _name174, _high_is_bad174 in _avail174:
                _s174 = _df174[_col174].dropna()
                _ma174 = _s174.rolling(_ma_window174).mean()
                _j174 = _s174.to_frame("v").join(_ma174.to_frame("ma"), how="inner").dropna()
                if _high_is_bad174:
                    _above174 = (_j174["v"] > _j174["ma"]).astype(int)
                else:
                    _above174 = (_j174["v"] < _j174["ma"]).astype(int)
                _results174.append(_above174.rename(_name174))
            _diff_df174 = _pd174.concat(_results174, axis=1).dropna(how="all")
            _diffusion174 = _diff_df174.sum(axis=1)
            _n_avail174 = len(_avail174)
            _diffusion_pct174 = _diffusion174 / _n_avail174 * 100
            _fig174 = _go174.Figure()
            _fig174.add_trace(_go174.Scatter(
                x=_diffusion_pct174.index, y=_diffusion_pct174.values,
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.1)",
                line=dict(color="#ef4444", width=1.5), name="Diffusion %"
            ))
            _fig174.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                              annotation_text="Broad deterioration (70%)", annotation_position="right")
            _fig174.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                              annotation_text="Broad improvement (30%)", annotation_position="right")
            _fig174.add_hline(y=50, line_color="#9aa0aa", line_width=0.5)
            _fig174.update_layout(
                title=f"Credit Conditions Diffusion Index (% of {_n_avail174} indicators deteriorating vs 63d MA)",
                height=340, yaxis_title="% Deteriorating", yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig174, use_container_width=True)
            # Current breakdown table
            _cur_row174 = {}
            for _col174, _name174, _high174 in _avail174:
                _s174b = _df174[_col174].dropna()
                _ma174b = _s174b.rolling(_ma_window174).mean().dropna()
                if len(_ma174b) > 0:
                    _v174 = float(_s174b.iloc[-1])
                    _m174 = float(_ma174b.iloc[-1])
                    _bad174 = (_v174 > _m174) if _high174 else (_v174 < _m174)
                    _cur_row174[_name174] = "⬆ Stress" if _bad174 else "⬇ OK"
            _tbl174 = _pd174.DataFrame.from_dict(
                _cur_row174, orient="index", columns=["vs 63d MA"]).reset_index()
            _tbl174.columns = ["Indicator", "Status"]
            st.dataframe(_tbl174, use_container_width=True, hide_index=True)
            _cur_diff174 = float(_diffusion_pct174.iloc[-1]) if _diffusion_pct174.notna().any() else float("nan")
            _n_stress174 = int(round(_cur_diff174 / 100 * _n_avail174)) if not _np174.isnan(_cur_diff174) else 0
            _breadth174 = ("Broad Deterioration" if _cur_diff174 > 70 else
                           ("Improving Breadth" if _cur_diff174 < 30 else "Mixed"))
            st.caption(
                f"Current diffusion: **{_cur_diff174:.0f}%** ({_n_stress174}/{_n_avail174} indicators deteriorating) — **{_breadth174}**."
            )
    except Exception as _e174:
        _err_track(_active_sub, _e174)
        st.caption(f"Credit diffusion: {_e174}")

# ── Section Overview Pages ───────────────────────────────────────────────────


if _active_sub == "ov_cm":
    try:
        import plotly.graph_objects as _go_ov_cm
        import numpy as _np_ov_cm
        st.subheader("Credit Markets — Section Overview")
        st.caption("Current snapshot across key credit market indicators. Select any sub-view from the sidebar to drill in.")
        _d = df
        def _pct_rank(col):
            s = _d[col].dropna()
            if len(s) < 10: return float("nan")
            return float((s < float(s.iloc[-1])).mean() * 100)
        def _last(col):
            s = _d[col].dropna()
            return float(s.iloc[-1]) if len(s) else float("nan")
        def _chg(col, n=21):
            s = _d[col].dropna()
            if len(s) < n + 1: return float("nan")
            return float(s.iloc[-1]) - float(s.iloc[-n - 1])
        _c1, _c2, _c3, _c4 = st.columns(4)
        _hy_lvl = _last("hy_spread"); _hy_pct = _pct_rank("hy_spread"); _hy_chg = _chg("hy_spread")
        _c1.metric("HY OAS", f"{_hy_lvl:.0f} bps" if not _np_ov_cm.isnan(_hy_lvl) else "—",
                   delta=f"{_hy_chg:+.0f} 21d" if not _np_ov_cm.isnan(_hy_chg) else None,
                   delta_color="inverse")
        _ig_lvl = _last("ig_spread"); _ig_chg = _chg("ig_spread")
        _c2.metric("IG OAS", f"{_ig_lvl:.0f} bps" if not _np_ov_cm.isnan(_ig_lvl) else "—",
                   delta=f"{_ig_chg:+.0f} 21d" if not _np_ov_cm.isnan(_ig_chg) else None,
                   delta_color="inverse")
        _hyi_lvl = _last("hy_ig_ratio"); _hyi_pct = _pct_rank("hy_ig_ratio")
        _c3.metric("HY/IG Ratio", f"{_hyi_lvl:.2f}" if not _np_ov_cm.isnan(_hyi_lvl) else "—",
                   delta=f"{_hyi_pct:.0f}th pct" if not _np_ov_cm.isnan(_hyi_pct) else None,
                   delta_color="off")
        _cs_lvl = _last("credit_market_risk_score_smooth")
        _cs_chg = _chg("credit_market_risk_score_smooth")
        _c4.metric("Credit Risk Score", f"{_cs_lvl:.1f}" if not _np_ov_cm.isnan(_cs_lvl) else "—",
                   delta=f"{_cs_chg:+.1f} 21d" if not _np_ov_cm.isnan(_cs_chg) else None,
                   delta_color="inverse")
        st.divider()
        # HY sparkline + regime
        _hy_s = _d["hy_spread"].dropna().tail(252)
        _ig_s = _d["ig_spread"].dropna().tail(252) if "ig_spread" in _d.columns else None
        _fig_ov_cm = _go_ov_cm.Figure()
        _fig_ov_cm.add_trace(_go_ov_cm.Scatter(x=_hy_s.index, y=_hy_s.values,
            name="HY OAS", line=dict(color="#ef4444", width=2)))
        if _ig_s is not None:
            _fig_ov_cm.add_trace(_go_ov_cm.Scatter(x=_ig_s.index, y=_ig_s.values,
                name="IG OAS", line=dict(color="#6366f1", width=1.5)))
        _fig_ov_cm.update_layout(
            title="HY & IG Spreads — Last 252 Trading Days",
            height=280, yaxis_title="Bps",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=40, b=20))
        st.plotly_chart(_fig_ov_cm, use_container_width=True)
        # Valuation summary
        _hy_val = "Rich" if _hy_pct < 30 else ("Cheap" if _hy_pct > 70 else "Fair")
        st.info(f"HY at **{_hy_pct:.0f}th** historical percentile — **{_hy_val}**. "
                f"30 sub-views available: default cycle, carry decomp, credit cycle clock, momentum, and more.")
    except Exception as _e_ov_cm:
        _err_track(_active_sub, _e_ov_cm)
        st.caption(f"Credit Markets overview: {_e_ov_cm}")


if _active_sub == 175:
    try:
        import plotly.graph_objects as _go175
        import numpy as _np175
        import pandas as _pd175
        _df175 = df.copy() if "df" in dir() else None
        _has175 = (_df175 is not None
                   and "bbb_ig_ratio" in _df175.columns
                   and "hy_spread" in _df175.columns)
        if not _has175:
            st.info("bbb_ig_ratio and hy_spread required.")
        else:
            st.subheader("BBB Cliff Risk Monitor")
            st.caption("BBB-rated debt sits one notch above junk. When BBB/IG spread ratios surge, fallen angel risk rises — forced selling as bonds drop to HY triggers wave widening. This view tracks BBB stress relative to IG and HY as an early warning of credit quality deterioration cascades.")
            _ratio175 = _df175["bbb_ig_ratio"].dropna()
            _hy175 = _df175["hy_spread"].dropna()
            _j175 = _ratio175.to_frame("ratio").join(_hy175.to_frame("hy"), how="inner").dropna().tail(1260)
            _ratio_pct175 = _ratio175.rolling(252).rank(pct=True) * 100
            _c1, _c2, _c3, _c4 = st.columns(4)
            _cur_ratio175 = float(_ratio175.iloc[-1])
            _cur_rpct175 = float(_ratio_pct175.iloc[-1]) if _ratio_pct175.notna().any() else float("nan")
            _cur_hy175 = float(_hy175.iloc[-1])
            _c1.metric("BBB/IG Ratio", f"{_cur_ratio175:.2f}")
            _c2.metric("Ratio Percentile", f"{_cur_rpct175:.0f}th" if not _np175.isnan(_cur_rpct175) else "—",
                       delta_color="inverse")
            _c3.metric("HY OAS", f"{_cur_hy175:.0f} bps")
            _cliff_risk = "Elevated" if _cur_rpct175 > 70 else ("Moderate" if _cur_rpct175 > 40 else "Low")
            _c4.metric("Cliff Risk", _cliff_risk,
                       delta_color="inverse" if _cliff_risk == "Elevated" else "off")
            st.divider()
            _fig175 = _go175.Figure()
            _fig175.add_trace(_go175.Scatter(
                x=_j175.index, y=_j175["ratio"],
                name="BBB/IG Ratio", line=dict(color="#f59e0b", width=2)))
            _fig175.add_trace(_go175.Scatter(
                x=_ratio_pct175.tail(1260).index,
                y=_ratio_pct175.tail(1260).values / 100 * float(_j175["ratio"].max()),
                name="Percentile (scaled)", line=dict(color="#6366f1", width=1, dash="dot"),
                yaxis="y2"))
            _fig175.update_layout(
                title="BBB/IG Spread Ratio (5Y)",
                height=300,
                yaxis=dict(title="Ratio"),
                yaxis2=dict(title="Percentile (scaled)", overlaying="y", side="right"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(_fig175, use_container_width=True)
            # Scatter: BBB/IG ratio vs HY
            _fig175b = _go175.Figure()
            _fig175b.add_trace(_go175.Scatter(
                x=_j175["ratio"], y=_j175["hy"],
                mode="markers", marker=dict(color="#f59e0b", size=3, opacity=0.4),
                name="History"))
            _fig175b.add_trace(_go175.Scatter(
                x=[_cur_ratio175], y=[_cur_hy175],
                mode="markers+text", marker=dict(color="white", size=12, symbol="star"),
                text=["Now"], textposition="top center", name="Current"))
            _fig175b.update_layout(
                title="BBB/IG Ratio vs HY Spread (5Y)",
                height=280, xaxis_title="BBB/IG Ratio", yaxis_title="HY Spread (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False)
            st.plotly_chart(_fig175b, use_container_width=True)
            st.caption(
                f"BBB/IG ratio at **{_cur_ratio175:.2f}** ({_cur_rpct175:.0f}th pct vs 252d history). "
                "Rising ratio = BBB stress outpacing IG; watch for fallen angel acceleration above 80th pct.")
    except Exception as _e175:
        _err_track(_active_sub, _e175)
        st.caption(f"BBB cliff risk: {_e175}")


if _active_sub == 183:
    st.subheader("Banking Flow Monitor")
    st.caption("Deposit and loan growth as leading credit stress indicators — early warning from banking system flows")
    try:
        import plotly.graph_objects as _go183
        import numpy as _np183
        import pandas as _pd183
        _df183 = df[["deposit_growth_90d","loan_growth_90d","bank_deposits","hy_spread","nfci"]].dropna().copy()
        _last183 = _df183.iloc[-1]
        _c1_183, _c2_183, _c3_183, _c4_183 = st.columns(4)
        _c1_183.metric("Deposit Growth (90d)", f"{_last183['deposit_growth_90d']:+.2f}%")
        _c2_183.metric("Loan Growth (90d)", f"{_last183['loan_growth_90d']:+.2f}%")
        _c3_183.metric("Bank Deposits", f"${_last183['bank_deposits']:,.0f}bn" if _last183['bank_deposits'] > 100 else f"{_last183['bank_deposits']:,.1f}")
        _stress183 = _last183["deposit_growth_90d"] < -2 or _last183["loan_growth_90d"] < -3
        _c4_183.metric("Flow Signal", "Stress" if _stress183 else "Normal", delta=None)
        st.divider()
        # Dual panel: deposit/loan growth + HY
        _fig183a = _go183.Figure()
        _fig183a.add_trace(_go183.Scatter(
            x=_df183.index, y=_df183["deposit_growth_90d"],
            name="Deposit Growth 90d (%)", line=dict(color="#3b82f6", width=1.5), yaxis="y1"
        ))
        _fig183a.add_trace(_go183.Scatter(
            x=_df183.index, y=_df183["loan_growth_90d"],
            name="Loan Growth 90d (%)", line=dict(color="#10b981", width=1.5), yaxis="y1"
        ))
        _fig183a.add_trace(_go183.Scatter(
            x=_df183.index, y=_df183["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#ef4444", width=1.2, dash="dot"), yaxis="y2"
        ))
        _fig183a.add_hline(y=0, line_color="#4b5563", line_dash="dash", line_width=1)
        _fig183a.update_layout(
            title="Deposit/Loan Growth vs HY Spread",
            height=360,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Growth (%)", side="left"),
            yaxis2=dict(title="HY Spread (bps)", side="right", overlaying="y", color="#ef4444", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig183a, use_container_width=True)
        # Bank deposits level
        _fig183b = _go183.Figure()
        _fig183b.add_trace(_go183.Scatter(
            x=_df183.index, y=_df183["bank_deposits"],
            name="Bank Deposits", fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)", line=dict(color="#3b82f6", width=1.5)
        ))
        _fig183b.update_layout(
            title="Total Bank Deposits",
            height=200,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), yaxis_title="Deposits",
            margin=dict(t=40, b=30),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig183b, use_container_width=True)
        # Rolling correlation between deposit growth and HY spread
        _roll_dep183 = _df183["deposit_growth_90d"].rolling(90).corr(_df183["hy_spread"])
        _roll_loan183 = _df183["loan_growth_90d"].rolling(90).corr(_df183["hy_spread"])
        _fig183c = _go183.Figure()
        _fig183c.add_trace(_go183.Scatter(x=_df183.index, y=_roll_dep183, name="Deposit↔HY (90d roll)", line=dict(color="#3b82f6", width=1.5)))
        _fig183c.add_trace(_go183.Scatter(x=_df183.index, y=_roll_loan183, name="Loan↔HY (90d roll)", line=dict(color="#10b981", width=1.5)))
        _fig183c.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)
        _fig183c.update_layout(
            title="Rolling 90d Correlation: Banking Flows ↔ HY Spread",
            height=200,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), yaxis_title="Correlation",
            margin=dict(t=40, b=30))
        st.plotly_chart(_fig183c, use_container_width=True)
        _corr_dep183 = round(float(_df183["deposit_growth_90d"].corr(_df183["hy_spread"])), 2)
        _corr_loan183 = round(float(_df183["loan_growth_90d"].corr(_df183["hy_spread"])), 2)
        st.caption(
            f"Deposit growth {_last183['deposit_growth_90d']:+.2f}% / Loan growth {_last183['loan_growth_90d']:+.2f}% (90d). "
            f"Full-history deposit↔HY corr: {_corr_dep183:+.2f}; loan↔HY corr: {_corr_loan183:+.2f}. "
            f"{'Banking flow stress detected.' if _stress183 else 'Banking flows within normal range.'}")
    except Exception as _e183:
        _err_track(_active_sub, _e183)
        st.caption(f"Banking flow monitor: {_e183}")

# --- sub184: Forward Return Calibration ---

if _active_sub == 185:
    st.subheader("Default Probability Monitor")
    st.caption("Cycle-based default probability, CDS-implied PD, and default cycle score — credit stress through the default lens")
    try:
        import plotly.graph_objects as _go185
        import numpy as _np185
        import pandas as _pd185
        _df185 = df[["default_cycle_score","default_probability","cds_implied_pd_score","hy_spread","nfci"]].dropna().copy()
        _last185 = _df185.iloc[-1]
        _dp_pct185 = float((_df185["default_probability"] < _last185["default_probability"]).mean() * 100)
        _dc_pct185 = float((_df185["default_cycle_score"] < _last185["default_cycle_score"]).mean() * 100)
        _c1_185, _c2_185, _c3_185, _c4_185 = st.columns(4)
        _c1_185.metric("Default Prob", f"{_last185['default_probability']:.2%}", f"{_dp_pct185:.0f}th pct")
        _c2_185.metric("Default Cycle", f"{_last185['default_cycle_score']:.2f}", f"{_dc_pct185:.0f}th pct")
        _c3_185.metric("CDS Implied PD", f"{_last185['cds_implied_pd_score']:.2f}")
        _alert185 = _last185["default_probability"] > _df185["default_probability"].quantile(0.80)
        _c4_185.metric("Signal", "Elevated" if _alert185 else "Normal")
        st.divider()
        # Main time series: all three on same chart
        _fig185a = _go185.Figure()
        _fig185a.add_trace(_go185.Scatter(
            x=_df185.index, y=_df185["default_probability"],
            name="Default Probability", line=dict(color="#ef4444", width=2), yaxis="y1"
        ))
        _fig185a.add_trace(_go185.Scatter(
            x=_df185.index, y=_df185["default_cycle_score"],
            name="Default Cycle Score", line=dict(color="#f59e0b", width=1.5, dash="dot"), yaxis="y2"
        ))
        _fig185a.add_trace(_go185.Scatter(
            x=_df185.index, y=_df185["cds_implied_pd_score"],
            name="CDS Implied PD Score", line=dict(color="#a78bfa", width=1.5, dash="dash"), yaxis="y2"
        ))
        _fig185a.add_hline(y=_df185["default_probability"].quantile(0.80), line_dash="dot",
                           line_color="#ef4444", line_width=0.8, yref="y1", annotation_text="80th pct")
        _fig185a.update_layout(
            title="Default Probability & Cycle Score",
            height=360,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Default Probability", side="left", color="#ef4444", tickformat=".1%"),
            yaxis2=dict(title="Score (normalized)", side="right", overlaying="y", color="#f59e0b", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig185a, use_container_width=True)
        # HY spread vs default probability scatter
        _c1b_185, _c2b_185 = st.columns(2)
        with _c1b_185:
            _fig185b = _go185.Figure()
            _fig185b.add_trace(_go185.Scatter(
                x=_df185["default_probability"], y=_df185["hy_spread"],
                mode="markers", marker=dict(size=3, color="#ef4444", opacity=0.4), name="Obs"
            ))
            _fig185b.add_trace(_go185.Scatter(
                x=[_last185["default_probability"]], y=[_last185["hy_spread"]],
                mode="markers", marker=dict(size=12, color="#f59e0b", symbol="star"), name="Now"
            ))
            _fig185b.update_layout(
                title="Default Prob vs HY Spread",
                height=260,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                xaxis_title="Default Prob", yaxis_title="HY Spread (bps)",
                xaxis_tickformat=".1%",
                margin=dict(t=40, b=30))
            st.plotly_chart(_fig185b, use_container_width=True)
        with _c2b_185:
            # Distribution of default probability
            _fig185c = _go185.Figure()
            _fig185c.add_trace(_go185.Histogram(
                x=_df185["default_probability"], nbinsx=50,
                marker_color="#ef4444", opacity=0.7, name="Distribution"
            ))
            _fig185c.add_vline(x=_last185["default_probability"], line_dash="dash",
                               line_color="#f59e0b", annotation_text=f"Now: {_last185['default_probability']:.2%}")
            _fig185c.update_layout(
                title="Default Prob Distribution",
                height=260,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                xaxis_title="Default Probability", yaxis_title="Count",
                xaxis_tickformat=".1%",
                margin=dict(t=40, b=30))
            st.plotly_chart(_fig185c, use_container_width=True)
        # Rolling correlation: default prob vs HY
        _roll_dp_hy185 = _df185["default_probability"].rolling(90).corr(_df185["hy_spread"])
        _fig185d = _go185.Figure()
        _fig185d.add_trace(_go185.Scatter(x=_df185.index, y=_roll_dp_hy185,
                                          name="DefaultProb↔HY (90d)", line=dict(color="#ef4444", width=1.5)))
        _fig185d.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)
        _fig185d.update_layout(
            title="Rolling 90d: Default Probability ↔ HY Spread Correlation",
            height=180,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), yaxis_title="Correlation",
            margin=dict(t=40, b=25))
        st.plotly_chart(_fig185d, use_container_width=True)
        _corr_dp185 = round(float(_df185["default_probability"].corr(_df185["hy_spread"])), 2)
        st.caption(
            f"Default probability at {_last185['default_probability']:.2%} ({_dp_pct185:.0f}th percentile). "
            f"Default cycle score: {_last185['default_cycle_score']:.2f} ({_dc_pct185:.0f}th pct). "
            f"Full-history default prob↔HY spread correlation: {_corr_dp185:+.2f}. "
            f"{'Elevated default risk — spread widening risk elevated.' if _alert185 else 'Default probability within normal historical range.'}")
    except Exception as _e185:
        _err_track(_active_sub, _e185)
        st.caption(f"Default probability monitor: {_e185}")

# --- sub186: ANFCI & Banking Stress Suite ---

if _active_sub == 187:
    st.subheader("Spread Decomposition")
    st.caption("HY spread broken into expected loss, excess spread, and compensation ratio — how much are investors paid above fair value?")
    try:
        import plotly.graph_objects as _go187
        import numpy as _np187
        import pandas as _pd187
        _df187 = df[["hy_spread","excess_spread_bps","expected_loss_bps","spread_compensation_ratio","excess_spread_percentile"]].dropna().copy()
        _last187 = _df187.iloc[-1]
        _scr_pct187 = float((_df187["spread_compensation_ratio"] < _last187["spread_compensation_ratio"]).mean() * 100)
        _exc_pct187 = float((_df187["excess_spread_bps"] < _last187["excess_spread_bps"]).mean() * 100)
        _c1_187, _c2_187, _c3_187, _c4_187 = st.columns(4)
        _c1_187.metric("HY Spread", f"{_last187['hy_spread']:.0f} bps")
        _c2_187.metric("Expected Loss", f"{_last187['expected_loss_bps']:.0f} bps", f"{_last187['expected_loss_bps']/max(_last187['hy_spread'],1)*100:.0f}% of spread")
        _c3_187.metric("Excess Spread", f"{_last187['excess_spread_bps']:.0f} bps", f"{_exc_pct187:.0f}th pct")
        _c4_187.metric("Compensation Ratio", f"{_last187['spread_compensation_ratio']:.2f}x", f"{_scr_pct187:.0f}th pct")
        st.divider()
        # Stacked area: EL + excess = total spread
        _fig187a = _go187.Figure()
        _fig187a.add_trace(_go187.Scatter(
            x=_df187.index, y=_df187["expected_loss_bps"],
            name="Expected Loss", stackgroup="spread",
            fillcolor="rgba(239,68,68,0.6)", line=dict(color="#ef4444", width=0)
        ))
        _fig187a.add_trace(_go187.Scatter(
            x=_df187.index, y=_df187["excess_spread_bps"].clip(lower=0),
            name="Excess Spread (compensation)", stackgroup="spread",
            fillcolor="rgba(16,185,129,0.5)", line=dict(color="#10b981", width=0)
        ))
        _fig187a.add_trace(_go187.Scatter(
            x=_df187.index, y=_df187["hy_spread"],
            name="HY Spread (total)", line=dict(color="#f59e0b", width=1.5, dash="dot"), stackgroup=None
        ))
        _fig187a.update_layout(
            title="HY Spread Decomposition: Expected Loss vs Excess Spread",
            height=360,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis_title="bps",
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig187a, use_container_width=True)
        # Compensation ratio and excess spread percentile
        _c1b_187, _c2b_187 = st.columns(2)
        with _c1b_187:
            _fig187b = _go187.Figure()
            _fig187b.add_trace(_go187.Scatter(
                x=_df187.index, y=_df187["spread_compensation_ratio"],
                name="Compensation Ratio", fill="tozeroy",
                fillcolor="rgba(16,185,129,0.12)", line=dict(color="#10b981", width=1.5)
            ))
            _fig187b.add_hline(y=1.0, line_dash="dash", line_color="#f59e0b", line_width=1,
                               annotation_text="Fair value (1.0x)")
            _fig187b.add_hline(y=float(_last187["spread_compensation_ratio"]),
                               line_dash="dot", line_color="#3b82f6", line_width=1,
                               annotation_text=f"Now: {_last187['spread_compensation_ratio']:.2f}x")
            _fig187b.update_layout(
                title="Spread Compensation Ratio",
                height=240,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                yaxis_title="Ratio (>1 = compensated)",
                margin=dict(t=40, b=30))
            st.plotly_chart(_fig187b, use_container_width=True)
        with _c2b_187:
            _fig187c = _go187.Figure()
            _fig187c.add_trace(_go187.Scatter(
                x=_df187.index, y=_df187["excess_spread_percentile"],
                name="Excess Spread Percentile", fill="tozeroy",
                fillcolor="rgba(59,130,246,0.12)", line=dict(color="#3b82f6", width=1.5)
            ))
            _fig187c.add_hline(y=50, line_dash="dash", line_color="#4b5563", line_width=0.8)
            _fig187c.add_hline(y=float(_last187["excess_spread_percentile"]),
                               line_dash="dot", line_color="#f59e0b", line_width=1,
                               annotation_text=f"Now: {_last187['excess_spread_percentile']:.0f}th")
            _fig187c.update_layout(
                title="Excess Spread Percentile",
                height=240,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                yaxis_title="Percentile", yaxis_range=[0, 100],
                margin=dict(t=40, b=30))
            st.plotly_chart(_fig187c, use_container_width=True)
        # Valuation summary
        _val_signal187 = "Rich (below fair value)" if _last187["spread_compensation_ratio"] < 0.9 else ("Cheap (above fair value)" if _last187["spread_compensation_ratio"] > 1.1 else "Fair value")
        st.info(
            f"**Spread valuation: {_val_signal187}**  \n"
            f"Of the {_last187['hy_spread']:.0f} bps total HY spread, {_last187['expected_loss_bps']:.0f} bps compensates for expected credit losses "
            f"and {_last187['excess_spread_bps']:.0f} bps is excess (risk premium). "
            f"Compensation ratio {_last187['spread_compensation_ratio']:.2f}x is at the {_scr_pct187:.0f}th historical percentile."
        )
    except Exception as _e187:
        _err_track(_active_sub, _e187)
        st.caption(f"Spread decomposition: {_e187}")

# --- sub188: Corporate Health Suite ---

if _active_sub == 188:
    st.subheader("Corporate Health Suite")
    st.caption("Corporate leverage cycle and profit cycle scores — fundamental credit quality drivers")
    try:
        import plotly.graph_objects as _go188
        import numpy as _np188
        import pandas as _pd188
        _df188 = df[["corporate_leverage_score","corporate_profit_cycle_score","hy_spread","ig_spread"]].dropna().copy()
        _last188 = _df188.iloc[-1]
        _lev_pct188 = float((_df188["corporate_leverage_score"] < _last188["corporate_leverage_score"]).mean() * 100)
        _pft_pct188 = float((_df188["corporate_profit_cycle_score"] < _last188["corporate_profit_cycle_score"]).mean() * 100)
        _c1_188, _c2_188, _c3_188, _c4_188 = st.columns(4)
        _c1_188.metric("Leverage Score", f"{_last188['corporate_leverage_score']:.2f}", f"{_lev_pct188:.0f}th pct")
        _c2_188.metric("Profit Cycle", f"{_last188['corporate_profit_cycle_score']:.2f}", f"{_pft_pct188:.0f}th pct")
        _c3_188.metric("HY Spread", f"{_last188['hy_spread']:.0f} bps")
        _combined188 = (_last188["corporate_leverage_score"] + (1 - _last188["corporate_profit_cycle_score"])) / 2
        _c4_188.metric("Combined Stress", f"{_combined188:.2f}", "high=stress")
        st.divider()
        # Time series: both scores + HY
        _fig188a = _go188.Figure()
        _fig188a.add_trace(_go188.Scatter(
            x=_df188.index, y=_df188["corporate_leverage_score"],
            name="Leverage Score (high=stressed)", line=dict(color="#ef4444", width=1.8), yaxis="y1"
        ))
        _fig188a.add_trace(_go188.Scatter(
            x=_df188.index, y=_df188["corporate_profit_cycle_score"],
            name="Profit Cycle Score (high=healthy)", line=dict(color="#10b981", width=1.8), yaxis="y1"
        ))
        _fig188a.add_trace(_go188.Scatter(
            x=_df188.index, y=_df188["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#9aa0aa", width=1, dash="dot"), yaxis="y2"
        ))
        _fig188a.update_layout(
            title="Corporate Leverage & Profit Cycle vs HY Spread",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Score", side="left"),
            yaxis2=dict(title="HY Spread (bps)", side="right", overlaying="y", color="#9aa0aa", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig188a, use_container_width=True)
        # Quadrant scatter: leverage vs profit cycle
        _c1b_188, _c2b_188 = st.columns(2)
        with _c1b_188:
            _fig188b = _go188.Figure()
            _color_scale188 = _df188["hy_spread"].values
            _fig188b.add_trace(_go188.Scatter(
                x=_df188["corporate_leverage_score"], y=_df188["corporate_profit_cycle_score"],
                mode="markers",
                marker=dict(size=3, color=_color_scale188, colorscale=[[0,"#3b82f6"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                            colorbar=dict(title="HY bps", len=0.6), opacity=0.5),
                name="History"
            ))
            _fig188b.add_trace(_go188.Scatter(
                x=[_last188["corporate_leverage_score"]], y=[_last188["corporate_profit_cycle_score"]],
                mode="markers", marker=dict(size=14, color="#f59e0b", symbol="star"), name="Now"
            ))
            _lev_med188 = float(_df188["corporate_leverage_score"].median())
            _pft_med188 = float(_df188["corporate_profit_cycle_score"].median())
            _fig188b.add_vline(x=_lev_med188, line_dash="dash", line_color="#4b5563", line_width=0.8)
            _fig188b.add_hline(y=_pft_med188, line_dash="dash", line_color="#4b5563", line_width=0.8)
            _fig188b.update_layout(
                title="Corporate Health Quadrant",
                height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                xaxis_title="Leverage (high=stressed)", yaxis_title="Profit Cycle (high=healthy)",
                margin=dict(t=40, b=30))
            st.plotly_chart(_fig188b, use_container_width=True)
        with _c2b_188:
            # Rolling correlation
            _roll_lev188 = _df188["corporate_leverage_score"].rolling(90).corr(_df188["hy_spread"])
            _roll_pft188 = _df188["corporate_profit_cycle_score"].rolling(90).corr(_df188["hy_spread"])
            _fig188c = _go188.Figure()
            _fig188c.add_trace(_go188.Scatter(x=_df188.index, y=_roll_lev188,
                                              name="Leverage↔HY (90d)", line=dict(color="#ef4444", width=1.5)))
            _fig188c.add_trace(_go188.Scatter(x=_df188.index, y=_roll_pft188,
                                              name="Profit↔HY (90d)", line=dict(color="#10b981", width=1.5)))
            _fig188c.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=0.8)
            _fig188c.update_layout(
                title="Rolling 90d Correlation with HY Spread",
                height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                yaxis_title="Correlation", margin=dict(t=40, b=30))
            st.plotly_chart(_fig188c, use_container_width=True)
        # Quadrant label
        _q188 = ("High Leverage / Low Profits — worst fundamentals" if _last188["corporate_leverage_score"] > _lev_med188 and _last188["corporate_profit_cycle_score"] < _pft_med188
                 else "Low Leverage / High Profits — best fundamentals" if _last188["corporate_leverage_score"] <= _lev_med188 and _last188["corporate_profit_cycle_score"] >= _pft_med188
                 else "Mixed signals")
        _corr_lev188 = round(float(_df188["corporate_leverage_score"].corr(_df188["hy_spread"])), 2)
        _corr_pft188 = round(float(_df188["corporate_profit_cycle_score"].corr(_df188["hy_spread"])), 2)
        st.caption(
            f"Quadrant: {_q188}. "
            f"Leverage↔HY corr: {_corr_lev188:+.2f}; Profit↔HY corr: {_corr_pft188:+.2f}. "
            f"Leverage {_lev_pct188:.0f}th pct, profit cycle {_pft_pct188:.0f}th pct.")
    except Exception as _e188:
        _err_track(_active_sub, _e188)
        st.caption(f"Corporate health: {_e188}")

# --- sub189: Absolute Yield Monitor ---

if _active_sub == 191:
    st.subheader("Primary & Loan Market Monitor")
    st.caption("New issuance conditions, loan market health, and institutional credit demand — supply-side credit signals")
    try:
        import plotly.graph_objects as _go191
        import numpy as _np191
        import pandas as _pd191
        _df191 = df[["primary_market_score","loan_market_score","institutional_credit_score","hy_spread","ig_spread"]].dropna().copy()
        _last191 = _df191.iloc[-1]
        _pm_pct191 = float((_df191["primary_market_score"] < _last191["primary_market_score"]).mean() * 100)
        _lm_pct191 = float((_df191["loan_market_score"] < _last191["loan_market_score"]).mean() * 100)
        _ic_pct191 = float((_df191["institutional_credit_score"] < _last191["institutional_credit_score"]).mean() * 100)
        _c1_191, _c2_191, _c3_191, _c4_191 = st.columns(4)
        _c1_191.metric("Primary Market", f"{_last191['primary_market_score']:.2f}", f"{_pm_pct191:.0f}th pct")
        _c2_191.metric("Loan Market", f"{_last191['loan_market_score']:.2f}", f"{_lm_pct191:.0f}th pct")
        _c3_191.metric("Institutional Credit", f"{_last191['institutional_credit_score']:.2f}", f"{_ic_pct191:.0f}th pct")
        _avg_supply191 = (_last191["primary_market_score"] + _last191["loan_market_score"] + _last191["institutional_credit_score"]) / 3
        _c4_191.metric("Avg Supply Signal", f"{_avg_supply191:.2f}")
        st.divider()
        # All three vs HY spread
        _fig191a = _go191.Figure()
        _fig191a.add_trace(_go191.Scatter(
            x=_df191.index, y=_df191["primary_market_score"],
            name="Primary Market Score", line=dict(color="#3b82f6", width=1.5), yaxis="y1"
        ))
        _fig191a.add_trace(_go191.Scatter(
            x=_df191.index, y=_df191["loan_market_score"],
            name="Loan Market Score", line=dict(color="#10b981", width=1.5), yaxis="y1"
        ))
        _fig191a.add_trace(_go191.Scatter(
            x=_df191.index, y=_df191["institutional_credit_score"],
            name="Institutional Credit Score", line=dict(color="#a78bfa", width=1.5), yaxis="y1"
        ))
        _fig191a.add_trace(_go191.Scatter(
            x=_df191.index, y=_df191["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#ef4444", width=1, dash="dot"), yaxis="y2"
        ))
        _fig191a.update_layout(
            title="Primary Market, Loan Market & Institutional Credit vs HY Spread",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Score", side="left"),
            yaxis2=dict(title="HY Spread (bps)", side="right", overlaying="y", color="#ef4444", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig191a, use_container_width=True)
        # Composite supply signal vs HY
        _df191["supply_composite"] = (_df191["primary_market_score"] + _df191["loan_market_score"] + _df191["institutional_credit_score"]) / 3
        _c1b_191, _c2b_191 = st.columns(2)
        with _c1b_191:
            _fig191b = _go191.Figure()
            _fig191b.add_trace(_go191.Scatter(
                x=_df191.index, y=_df191["supply_composite"],
                name="Supply Composite", fill="tozeroy",
                fillcolor="rgba(59,130,246,0.12)", line=dict(color="#3b82f6", width=1.5)
            ))
            _fig191b.update_layout(
                title="Supply Composite Score",
                height=230,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                yaxis_title="Score", margin=dict(t=40, b=30))
            st.plotly_chart(_fig191b, use_container_width=True)
        with _c2b_191:
            # Correlation table
            _corr191 = _df191[["primary_market_score","loan_market_score","institutional_credit_score","hy_spread"]].corr()[["hy_spread"]].round(2)
            _corr191.index = ["Primary Market", "Loan Market", "Institutional Credit", "HY Spread"]
            _corr191.columns = ["↔ HY Spread"]
            st.markdown("**Correlation with HY Spread**")
            st.dataframe(_corr191, use_container_width=True)
            # Regime: all three above/below median
            _pm_high191 = _last191["primary_market_score"] > _df191["primary_market_score"].median()
            _lm_high191 = _last191["loan_market_score"] > _df191["loan_market_score"].median()
            _ic_high191 = _last191["institutional_credit_score"] > _df191["institutional_credit_score"].median()
            _n_elevated191 = sum([_pm_high191, _lm_high191, _ic_high191])
            _supply_label191 = "All elevated — tight supply conditions" if _n_elevated191 == 3 else f"{_n_elevated191}/3 signals elevated"
            st.info(f"Supply signal: {_supply_label191}")
        st.caption(
            f"Primary market {_last191['primary_market_score']:.2f} ({_pm_pct191:.0f}th pct), "
            f"loan market {_last191['loan_market_score']:.2f} ({_lm_pct191:.0f}th pct), "
            f"institutional credit {_last191['institutional_credit_score']:.2f} ({_ic_pct191:.0f}th pct). "
            f"Composite supply signal: {_avg_supply191:.2f}.")
    except Exception as _e191:
        _err_track(_active_sub, _e191)
        st.caption(f"Primary & loan market: {_e191}")

# --- sub192: Model Confidence Monitor ---
