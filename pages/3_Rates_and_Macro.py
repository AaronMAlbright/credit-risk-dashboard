"""
Rates & Macro — analytics section page.
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
from src.distressed_debt import run_distressed_debt as run_distressed_debt_analysis
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
    page_title='Rates & Macro — Credit Risk Dashboard',
    page_icon='📈',
    layout='wide',
)

from utils.shared import (
    load_data, _err_track,
    _sig_badge, _pct_clr,
    _VIEW_DESC, _VIEW_INSIGHT,
    _ANALYTICS_VIEWS,
)


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
_SECTION_NAME = 'Rates & Macro'
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
if _nav_type == "Analytics" and _active_sub is not None:
    try:
        _lhy = float(df["hy_spread"].dropna().iloc[-1])
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
             f'<span style="color:{_pct_clr(_lhy_pct)};font-weight:700">{_lhy:.0f}'
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
            f'{_sig_badge(_lhy_pct)}&nbsp;<b>HY</b> {_lhy:.0f} bps'
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


if _active_sub == 23:
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

if _active_sub == 38:
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

if _active_sub == 41:
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

if _active_sub == 45:
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

if _active_sub == 53:
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

if _active_sub == 54:
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

if _active_sub == 57:
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

if _active_sub == 58:
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

if _active_sub == 63:
    import plotly.graph_objects as _go_fci
    st.header("Financial Conditions Index (FCI)")
    st.markdown(
        "Single 0–100 score capturing how tight or loose overall financial conditions are, "
        "credit-tilted. Higher = tighter. Leads economic activity by ~6–12 months. "
        "A 100bps tightening historically adds ~0.3–0.5% to HY default rates."
    )
    try:
        _fci = load_financial_conditions(df)
        if _fci.get("available"):
            _fcic = _fci.get("current", {})
            _c1f, _c2f, _c3f, _c4f = st.columns(4)
            _fci_score = _fcic.get("fci_score")
            _c1f.metric("FCI Score", f"{_fci_score:.0f}/100" if _fci_score is not None else "N/A")
            _c2f.metric("Regime", _fcic.get("fci_regime", "N/A"))
            _f1m = _fcic.get("fci_1m_change")
            _c3f.metric("1M Change", f"{'+' if _f1m and _f1m > 0 else ''}{_f1m:.1f}" if _f1m is not None else "N/A")
            _c4f.metric("Tightest Component", _fcic.get("tightest_component", "N/A"))
            if _fcic.get("interpretation"):
                st.info(_fcic["interpretation"])
            _fci_hist = _fci.get("fci_history")
            if _fci_hist is not None and len(_fci_hist.dropna()) > 50:
                _fig_fci = _go_fci.Figure()
                _fig_fci.add_trace(_go_fci.Scatter(
                    x=_fci_hist.index, y=_fci_hist.values,
                    name="FCI Score", line=dict(color="#a78bfa", width=1.5)
                ))
                _fig_fci.add_hrect(y0=75, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0)
                _fig_fci.add_hrect(y0=0, y1=25, fillcolor="rgba(52,211,153,0.08)", line_width=0)
                _fig_fci.add_hline(y=60, line_dash="dot", line_color="#f59e0b", annotation_text="Tight")
                _fig_fci.update_layout(
                    template="plotly_dark", height=320, title="Financial Conditions Index (0–100, higher=tighter)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="FCI Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_fci, use_container_width=True)
            _fci_comp = _fci.get("component_df")
            if _fci_comp is not None and not _fci_comp.empty:
                st.subheader("Component Z-Scores")
                _fig_comp = _go_fci.Figure()
                _comp_colors = {"hy_spread": "#ef4444", "ig_spread": "#f97316", "vix": "#f59e0b",
                                "yield_10y": "#60a5fa", "usd": "#34d399", "sp500": "#a78bfa"}
                for _col in _fci_comp.columns:
                    _fig_comp.add_trace(_go_fci.Scatter(
                        x=_fci_comp.index, y=_fci_comp[_col],
                        name=_col, line=dict(color=_comp_colors.get(_col, "#9aa0aa"), width=1.2)
                    ))
                _fig_comp.add_hline(y=0, line_dash="solid", line_color="#6b7280")
                _fig_comp.update_layout(
                    template="plotly_dark", height=260, title="FCI Component Z-Scores",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Z-score"),
                )
                st.plotly_chart(_fig_comp, use_container_width=True)
        else:
            st.info("FCI unavailable — requires hy_spread or ig_spread columns.")
    except Exception as _fci_e:
        st.caption(f"Financial Conditions Index unavailable: {_fci_e}")

# --- sub-tab 64: Credit Impulse ---------------------------------------------

if _active_sub == 64:
    import plotly.graph_objects as _go_ci
    st.header("Credit Impulse")
    st.markdown(
        "Rate of change of new credit flow relative to GDP (ΔCredit/GDP). "
        "Positive impulse supports spreads; contraction leads default cycles by 6–18 months. "
        "Requires FRED API key for TOTLL + CONSUMER + GDP series."
    )
    try:
        _ci = load_credit_impulse(df)
        if _ci.get("available"):
            _cic = _ci.get("current", {})
            _c1ci, _c2ci, _c3ci = st.columns(3)
            _ci_score = _cic.get("credit_impulse_score")
            _c1ci.metric("Impulse Score", f"{_ci_score:.0f}/100" if _ci_score is not None else "N/A")
            _c2ci.metric("Direction", _cic.get("impulse_direction", "N/A"))
            _c3ci.metric("Momentum", _cic.get("impulse_momentum", "N/A"))
            if _cic.get("lead_signal"):
                st.info(_cic["lead_signal"])
            if _cic.get("interpretation"):
                st.caption(_cic["interpretation"])
            if _cic.get("warning"):
                st.warning(_cic["warning"])
            _lc = _ci.get("lead_correlation")
            if _lc is not None:
                st.metric("Lead Correlation (impulse → fwd HY spread, 126d)", f"{_lc:.3f}")
            _ci_hist = _ci.get("impulse_score_history")
            if _ci_hist is not None and len(_ci_hist.dropna()) > 30:
                _fig_ci = _go_ci.Figure()
                _fig_ci.add_trace(_go_ci.Scatter(
                    x=_ci_hist.index, y=_ci_hist.values,
                    name="Credit Impulse Score", line=dict(color="#22d3ee", width=1.5)
                ))
                _fig_ci.add_hline(y=50, line_dash="dot", line_color="#6b7280", annotation_text="Neutral")
                _fig_ci.add_hline(y=30, line_dash="dot", line_color="#ef4444", annotation_text="Contraction Risk")
                _fig_ci.update_layout(
                    template="plotly_dark", height=300, title="Credit Impulse Score (0=most contractionary, 100=most expansionary)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_ci, use_container_width=True)
        else:
            st.info("Credit impulse unavailable — requires FRED API key (TOTLL, CONSUMER, GDP) or total_credit/gdp columns.")
    except Exception as _ci_e:
        st.caption(f"Credit impulse unavailable: {_ci_e}")

# --- sub-tab 65: ETF Premium/Discount ----------------------------------------

if _active_sub == 68:
    import plotly.graph_objects as _go_tp
    st.header("Term Premium Monitor")
    st.markdown(
        "Decomposes the 10y Treasury yield into **expected short rate** (where markets "
        "expect rates to go) and **term premium** (extra yield for duration/uncertainty risk). "
        "Rising term premium → duration costly → bad for IG; "
        "rising rate expectations → growth optimism → mildly positive for HY."
    )
    try:
        _tp = load_term_premium(df)
        if _tp.get("available"):
            _tpc = _tp.get("current", {})
            _c1tp, _c2tp, _c3tp, _c4tp = st.columns(4)
            _tp_val = _tpc.get("term_premium")
            _c1tp.metric("Term Premium", f"{_tp_val:.2f}%" if _tp_val is not None else "N/A")
            _c2tp.metric("Regime", _tpc.get("term_premium_regime", "N/A"))
            _c3tp.metric("Direction", _tpc.get("term_premium_direction", "N/A"))
            _dur_score = _tpc.get("duration_risk_score")
            _c4tp.metric("Duration Risk Score", f"{_dur_score:.0f}/100" if _dur_score is not None else "N/A")
            if _tpc.get("warning"):
                st.warning(_tpc["warning"])
            if _tpc.get("credit_implication"):
                st.info(_tpc["credit_implication"])
            _tp_hist = _tp.get("term_premium_history")
            _er_hist = _tp.get("expected_rate_history")
            if _tp_hist is not None and len(_tp_hist.dropna()) > 50:
                _fig_tp = _go_tp.Figure()
                _fig_tp.add_trace(_go_tp.Scatter(
                    x=_tp_hist.index, y=_tp_hist.values,
                    name="Term Premium (%)", line=dict(color="#a78bfa", width=1.5)
                ))
                if _er_hist is not None and len(_er_hist.dropna()) > 50:
                    _fig_tp.add_trace(_go_tp.Scatter(
                        x=_er_hist.index, y=_er_hist.values,
                        name="Expected Short Rate (%)", line=dict(color="#60a5fa", width=1.2, dash="dot")
                    ))
                _fig_tp.add_hline(y=0, line_dash="solid", line_color="#6b7280")
                _fig_tp.add_hline(y=1.5, line_dash="dot", line_color="#f59e0b", annotation_text="Elevated")
                _fig_tp.update_layout(
                    template="plotly_dark", height=320, title="Term Premium Decomposition",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="%"),
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(_fig_tp, use_container_width=True)
            _tp_r2 = _tp.get("tp_fraction_of_yield_move")
            if _tp_r2 is not None:
                st.metric("Term Premium Share of 10y Yield Moves (rolling R²)", f"{_tp_r2:.1%}")
        else:
            st.info("Term premium unavailable — requires yield_10y and yield_3m or yield_2y columns.")
    except Exception as _tp_e:
        st.caption(f"Term premium unavailable: {_tp_e}")

# --- sub-tab 69: Yield Curve Butterfly ---------------------------------------

if _active_sub == 69:
    import plotly.graph_objects as _go_fly
    st.header("Yield Curve Butterfly (2s5s10s)")
    st.markdown(
        "2s5s10s butterfly = 2×yield_5y − yield_2y − yield_10y. "
        "Measures curve curvature — negative butterfly means the belly is cheap = "
        "curve flattening = historically precedes recession and spread widening by 3–6 months."
    )
    try:
        _fly = load_yield_curve_butterfly(df)
        if _fly.get("available"):
            _flyc = _fly.get("current", {})
            _c1fly, _c2fly, _c3fly, _c4fly = st.columns(4)
            _fly_bps = _flyc.get("butterfly_bps")
            _c1fly.metric("Butterfly (2s5s10s)", f"{_fly_bps:.0f}bps" if _fly_bps is not None else "N/A")
            _slope_2s10s = _flyc.get("slope_2s10s_bps")
            _c2fly.metric("2s10s Slope", f"{_slope_2s10s:.0f}bps" if _slope_2s10s is not None else "N/A")
            _c3fly.metric("Curve Regime", _flyc.get("curve_regime", "N/A"))
            _inv_days = _flyc.get("inversion_duration_days", 0)
            _c4fly.metric("Inversion Duration", f"{_inv_days}d")
            if _flyc.get("warning"):
                st.warning(_flyc["warning"])
            if _flyc.get("credit_implication"):
                st.info(_flyc["credit_implication"])
            _fly_hist = _fly.get("butterfly_history")
            _slope_hist = _fly.get("slope_2s10s_history")
            if _fly_hist is not None and len(_fly_hist.dropna()) > 50:
                _fig_fly = _go_fly.Figure()
                _fig_fly.add_trace(_go_fly.Scatter(
                    x=_fly_hist.index, y=_fly_hist.values,
                    name="2s5s10s Butterfly (bps)", line=dict(color="#34d399", width=1.5)
                ))
                if _slope_hist is not None and len(_slope_hist.dropna()) > 50:
                    _fig_fly.add_trace(_go_fly.Scatter(
                        x=_slope_hist.index, y=_slope_hist.values,
                        name="2s10s Slope (bps)", line=dict(color="#60a5fa", width=1.2, dash="dot")
                    ))
                _fig_fly.add_hline(y=0, line_dash="solid", line_color="#6b7280")
                _fig_fly.add_hline(y=-50, line_dash="dot", line_color="#ef4444",
                                   annotation_text="Recession Warning")
                _fig_fly.update_layout(
                    template="plotly_dark", height=320, title="Yield Curve Butterfly & 2s10s Slope (bps)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="bps"),
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(_fig_fly, use_container_width=True)
            _fly_corr = _fly.get("credit_lead_correlation")
            if _fly_corr is not None:
                st.metric("Lead Correlation (butterfly → HY spread 6m forward)", f"{_fly_corr:.3f}")
        else:
            st.info("Yield curve butterfly unavailable — requires yield_2y, yield_5y, yield_10y columns.")
    except Exception as _fly_e:
        st.caption(f"Yield curve butterfly unavailable: {_fly_e}")

# --- sub-tab 70: SLOOS -------------------------------------------------------

if _active_sub == 70:
    import plotly.graph_objects as _go_sl
    st.header("Bank Lending Standards (SLOOS)")
    st.markdown(
        "Senior Loan Officer Opinion Survey — net % of banks tightening C&I loan standards. "
        "One of the most powerful leading indicators for corporate credit spreads, "
        "with a 2–4 quarter lead time. Requires FRED API key."
    )
    try:
        _sl = load_sloos_monitor(df)
        if _sl.get("available"):
            _slc = _sl.get("current", {})
            _c1sl, _c2sl, _c3sl = st.columns(3)
            _sl_score = _slc.get("sloos_score")
            _c1sl.metric("SLOOS Score", f"{_sl_score:.0f}/100" if _sl_score is not None else "N/A")
            _c2sl.metric("Regime", _slc.get("sloos_regime", "N/A"))
            _c3sl.metric("Direction", _slc.get("sloos_direction", "N/A"))
            _nt_ci = _slc.get("net_tightening_ci_large")
            if _nt_ci is not None:
                st.metric("C&I Net Tightening (large firms)", f"{_nt_ci:+.1f}%")
            if _slc.get("warning"):
                st.warning(_slc["warning"])
            if _slc.get("lead_signal"):
                st.info(_slc["lead_signal"])
            if _slc.get("interpretation"):
                st.caption(_slc["interpretation"])
            _sl_hist = _sl.get("sloos_history")
            if _sl_hist is not None and len(_sl_hist.dropna()) > 10:
                _fig_sl = _go_sl.Figure()
                _fig_sl.add_trace(_go_sl.Scatter(
                    x=_sl_hist.index, y=_sl_hist.values,
                    name="SLOOS Score", line=dict(color="#f59e0b", width=1.5)
                ))
                _fig_sl.add_hline(y=65, line_dash="dot", line_color="#ef4444",
                                  annotation_text="Tightening")
                _fig_sl.add_hline(y=35, line_dash="dot", line_color="#34d399",
                                  annotation_text="Easing")
                _fig_sl.update_layout(
                    template="plotly_dark", height=300,
                    title="Bank Lending Standards Score (0=Easing, 100=Tightening)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_sl, use_container_width=True)
            _sl_corr = _sl.get("lead_correlation_6m")
            if _sl_corr is not None:
                st.metric("Lead Correlation (SLOOS → HY spread 6m forward)", f"{_sl_corr:.3f}")
        else:
            st.info("SLOOS monitor unavailable — requires FRED API key (DRTSCILM, DRTSCIS, SUBLPPDCNOT).")
    except Exception as _sl_e:
        st.caption(f"SLOOS monitor unavailable: {_sl_e}")

# --- sub-tab 71: ETF Fund Flows ----------------------------------------------

if _active_sub == 72:
    import plotly.graph_objects as _go_cp
    st.header("Corporate Profit Cycle")
    st.markdown(
        "Tracks corporate profit margins (profits/GDP) as a macro-credit link. "
        "Margin compression → deteriorating debt coverage → rising default probability → "
        "HY spread widening. Leads credit spreads by 2–4 quarters. Requires FRED API key."
    )
    try:
        _cp = load_corporate_profit_cycle(df)
        if _cp.get("available"):
            _cpc = _cp.get("current", {})
            _c1cp, _c2cp, _c3cp, _c4cp = st.columns(4)
            _pm = _cpc.get("profit_margin")
            _c1cp.metric("Profit Margin", f"{_pm:.1%}" if _pm is not None else "N/A")
            _pm_yoy = _cpc.get("profit_margin_yoy")
            _c2cp.metric("YoY Change", f"{_pm_yoy:+.1f}%" if _pm_yoy is not None else "N/A")
            _ps = _cpc.get("profit_stress_score")
            _c3cp.metric("Profit Stress Score", f"{_ps:.0f}/100" if _ps is not None else "N/A")
            _c4cp.metric("Regime", _cpc.get("profit_cycle_regime", "N/A"))
            if _cpc.get("leverage_flag"):
                st.warning("Leverage flag: Margin compression + rising rates — debt service stress.")
            if _cpc.get("warning"):
                st.error(_cpc["warning"])
            if _cpc.get("lead_signal"):
                st.info(_cpc["lead_signal"])
            if _cpc.get("interpretation"):
                st.caption(_cpc["interpretation"])
            _cp_hist = _cp.get("profit_margin_history")
            _ps_hist = _cp.get("profit_stress_history")
            if _cp_hist is not None and len(_cp_hist.dropna()) > 10:
                _fig_cp = _go_cp.Figure()
                _fig_cp.add_trace(_go_cp.Scatter(
                    x=_cp_hist.index, y=(_cp_hist * 100).values,
                    name="Profit Margin (% of GDP)", line=dict(color="#34d399", width=1.5)
                ))
                _fig_cp.update_layout(
                    template="plotly_dark", height=280,
                    title="Corporate Profit Margin (% of GDP)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="%"),
                )
                st.plotly_chart(_fig_cp, use_container_width=True)
            if _ps_hist is not None and len(_ps_hist.dropna()) > 10:
                _fig_ps = _go_cp.Figure()
                _fig_ps.add_trace(_go_cp.Scatter(
                    x=_ps_hist.index, y=_ps_hist.values,
                    name="Profit Stress Score", line=dict(color="#f87171", width=1.5)
                ))
                _fig_ps.add_hline(y=60, line_dash="dot", line_color="#f59e0b",
                                  annotation_text="Compression Zone")
                _fig_ps.update_layout(
                    template="plotly_dark", height=260,
                    title="Corporate Profit Stress Score (0=Boom, 100=Recession Risk)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_ps, use_container_width=True)
            _cp_lc = _cp.get("lead_correlation")
            if _cp_lc is not None:
                st.metric("Lead Correlation (profit stress → HY spread 6m forward)", f"{_cp_lc:.3f}")
        else:
            st.info("Corporate profit cycle unavailable — requires FRED API key (CP, NFCPATAX, GDP).")
    except Exception as _cp_e:
        st.caption(f"Corporate profit cycle unavailable: {_cp_e}")

# --- sub-tab 73: CDS-Implied Default Probability -----------------------------

if _active_sub == 74:
    import plotly.graph_objects as _go74
    st.header("Recession Model (Estrella-Mishkin)")
    st.markdown(
        "The **Estrella-Mishkin (1998)** probit model estimates 12-month-ahead recession probability "
        "from the 10y−3m yield curve spread. Calibrated on post-war US data; has signaled every "
        "recession since 1969. **Threshold:** p > 25% = elevated concern; p > 40% = high probability. "
        "Credit spreads typically widen 3–6 months after recession probability crosses 25%."
    )
    try:
        _rec74 = load_recession(df)
        if _rec74.get("available"):
            _rc74 = _rec74["current"]
            _r74a, _r74b, _r74c, _r74d = st.columns(4)
            _r74a.metric("10y−3m Spread", f"{_rc74['spread_10y3m']:.2f}%",
                         help="Negative = inverted yield curve")
            _r74b.metric("10y−2y Spread", f"{_rc74['spread_10y2y']:.2f}%")
            _r74c.metric("Recession Prob (12m)", f"{_rc74['recession_prob_12m']:.1%}",
                         delta_color="inverse",
                         delta=f"{_rc74['recession_prob_12m']:.1%}")
            _r74d.metric("Signal", _rc74.get("signal", "—"))

            if "df" in _rec74 and "recession_prob_12m" in _rec74["df"].columns:
                _rec74_df = _rec74["df"].copy()
                _rec74_df["date"] = pd.to_datetime(_rec74_df["date"])
                _fig74a = _go74.Figure()
                _fig74a.add_trace(_go74.Scatter(
                    x=_rec74_df["date"], y=_rec74_df["recession_prob_12m"] * 100,
                    name="Recession Prob (12m)", line=dict(color="#e67e22", width=2),
                    fill="tozeroy", fillcolor="rgba(230,126,34,0.12)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Prob: %{y:.1f}%<extra></extra>",
                ))
                _fig74a.add_hline(y=25, line=dict(color="#e74c3c", dash="dash", width=1),
                                  annotation_text="25% — Elevated",
                                  annotation_font=dict(color="#e74c3c", size=10))
                _fig74a.add_hline(y=40, line=dict(color="#9b59b6", dash="dot", width=1),
                                  annotation_text="40% — High",
                                  annotation_font=dict(color="#9b59b6", size=10))
                _fig74a.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Probability (%)", range=[0, 100]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig74a, use_container_width=True)

            # Yield curve slope chart
            if "df" in _rec74:
                _rec74_df2 = _rec74["df"].copy()
                _rec74_df2["date"] = pd.to_datetime(_rec74_df2["date"])
                _fig74b = _go74.Figure()
                if "spread_10y3m" in _rec74_df2.columns:
                    _fig74b.add_trace(_go74.Scatter(
                        x=_rec74_df2["date"], y=_rec74_df2["spread_10y3m"],
                        name="10y−3m", line=dict(color="#4f8ef7", width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>10y−3m: %{y:+.2f}%<extra></extra>",
                    ))
                if "spread_10y2y" in _rec74_df2.columns:
                    _fig74b.add_trace(_go74.Scatter(
                        x=_rec74_df2["date"], y=_rec74_df2["spread_10y2y"],
                        name="10y−2y", line=dict(color="#27ae60", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>10y−2y: %{y:+.2f}%<extra></extra>",
                    ))
                _fig74b.add_hline(y=0, line_color="rgba(231,76,60,0.5)", line_width=1.5,
                                  annotation_text="Inversion threshold",
                                  annotation_font=dict(color="#e74c3c", size=10))
                _fig74b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="Yield Curve Slope", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Spread (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig74b, use_container_width=True)
                st.caption("Every US recession since 1969 was preceded by a 10y−3m inversion.")

            if _rec74.get("historical_inversions") is not None and not _rec74["historical_inversions"].empty:
                with st.expander("Historical yield curve inversion episodes"):
                    st.dataframe(_rec74["historical_inversions"], use_container_width=True, hide_index=True)

            # Credit spread implication table by recession probability bucket
            _prob_now = _rc74.get("recession_prob_12m", 0)
            st.subheader("Credit Spread Implications")
            _impl_rows = [
                {"Recession Prob Range": "< 10%",  "HY Spread Direction": "Tightening / Neutral", "Typical HY OAS": "250–400 bps", "Posture": "Risk-on"},
                {"Recession Prob Range": "10–25%", "HY Spread Direction": "Neutral / Watch",       "Typical HY OAS": "350–500 bps", "Posture": "Cautious"},
                {"Recession Prob Range": "25–40%", "HY Spread Direction": "Widening risk",          "Typical HY OAS": "450–650 bps", "Posture": "Defensive"},
                {"Recession Prob Range": "> 40%",  "HY Spread Direction": "Significant widening",  "Typical HY OAS": "600–900+ bps", "Posture": "Risk-off"},
            ]
            import pandas as _pd74
            _impl_df = _pd74.DataFrame(_impl_rows)
            st.dataframe(_impl_df, use_container_width=True, hide_index=True)
            st.caption(f"Current recession probability: **{_prob_now:.1%}** — "
                       f"{'elevated concern' if _prob_now > 0.25 else 'within normal range' if _prob_now < 0.10 else 'monitoring zone'}.")
        else:
            st.info("Recession model unavailable — requires 10y and 3m Treasury yields in dataset.")
    except Exception as _e74:
        _err_track(_active_sub, _e74)
        st.caption(f"Recession model unavailable: {_e74}")



if _active_sub == 75:
    import plotly.graph_objects as _go75
    st.header("Real Rates Monitor")
    st.markdown(
        "**Real rates** = nominal yield − inflation breakeven (TIPS-implied). "
        "Rising real rates increase debt service costs for leveraged borrowers, compress credit multiples, "
        "and historically precede HY spread widening. "
        "**Financial repression** (real rates < −1%) supports credit by reducing refinancing pressure. "
        "**Restrictive territory** (real rates > +1.5%) tightens credit conditions materially."
    )
    try:
        _rr75 = load_real_rates(df)
        if _rr75.get("available"):
            _rc75 = _rr75["current"]
            _rr75a, _rr75b, _rr75c, _rr75d = st.columns(4)
            _rr75a.metric("10y Real Rate", f"{_rc75.get('real_rate_10y', float('nan')):.2f}%",
                          delta=f"{_rc75.get('real_rate_change_1m', 0):+.2f}pp 1M",
                          delta_color="inverse",
                          help="10y nominal yield minus 10y breakeven inflation")
            _rr75b.metric("10y Breakeven Inflation", f"{_rc75.get('breakeven_10y', float('nan')):.2f}%",
                          help="Market-implied 10y avg inflation from TIPS")
            _rr75c.metric("10y Nominal Yield", f"{_rc75.get('yield_10y', float('nan')):.2f}%")
            _rr75d.metric("Real Rate Regime", _rc75.get("real_rate_regime", "—"))

            if _rr75.get("rising_flag"):
                st.warning("Real rates rising rapidly (+0.5pp/month) — refinancing pressure building for leveraged credit.")
            if _rr75.get("interpretation"):
                st.info(_rr75["interpretation"])

            # Real rate + breakeven decomposition chart
            _hist75 = _rr75.get("historical")
            if _hist75 is not None and not _hist75.empty:
                _hist75 = _hist75.copy()
                _hist75.index = pd.to_datetime(_hist75.index)
                _fig75 = _go75.Figure()
                if "real_rate_10y" in _hist75.columns:
                    _fig75.add_trace(_go75.Scatter(
                        x=_hist75.index, y=_hist75["real_rate_10y"],
                        name="10y Real Rate", line=dict(color="#f59e0b", width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>Real: %{y:.2f}%<extra></extra>",
                    ))
                if "breakeven_10y" in _hist75.columns:
                    _fig75.add_trace(_go75.Scatter(
                        x=_hist75.index, y=_hist75["breakeven_10y"],
                        name="10y Breakeven Inflation", line=dict(color="#e74c3c", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>BE: %{y:.2f}%<extra></extra>",
                    ))
                if "yield_10y" in _hist75.columns:
                    _fig75.add_trace(_go75.Scatter(
                        x=_hist75.index, y=_hist75["yield_10y"],
                        name="10y Nominal Yield", line=dict(color="#4f8ef7", width=1.5, dash="dash"),
                        hovertemplate="%{x|%Y-%m-%d}<br>Nominal: %{y:.2f}%<extra></extra>",
                    ))
                _fig75.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _fig75.add_hrect(y0=1.5, y1=6, fillcolor="rgba(231,76,60,0.05)", line_width=0,
                                 annotation_text="Restrictive", annotation_position="top right",
                                 annotation_font=dict(color="#e74c3c", size=9))
                _fig75.add_hrect(y0=-6, y1=-1, fillcolor="rgba(39,174,96,0.05)", line_width=0,
                                 annotation_text="Financial repression", annotation_position="bottom right",
                                 annotation_font=dict(color="#27ae60", size=9))
                _fig75.update_layout(
                    height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Rate (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig75, use_container_width=True)

            # Real rate vs HY spread scatter (if HY data available)
            st.subheader("Real Rate → Credit Spread Linkage")
            _rr75_df = _rr75.get("df")
            if _rr75_df is not None and "real_rate_10y" in _rr75_df.columns and "hy_spread" in _rr75_df.columns:
                _scat75 = _rr75_df[["real_rate_10y", "hy_spread"]].dropna().tail(504)
                _fig75b = _go75.Figure()
                _fig75b.add_trace(_go75.Scatter(
                    x=_scat75["real_rate_10y"], y=_scat75["hy_spread"],
                    mode="markers",
                    marker=dict(size=4, color="#4f8ef7", opacity=0.5),
                    hovertemplate="Real Rate: %{x:.2f}%<br>HY OAS: %{y:.2f}%<extra></extra>",
                ))
                _fig75b.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="10y Real Rate vs HY Spread (2yr rolling window)",
                               font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="10y Real Rate (%)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="HY OAS (%)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig75b, use_container_width=True)
                _corr75 = float(_scat75["real_rate_10y"].corr(_scat75["hy_spread"]))
                st.caption(f"Correlation (2yr): {_corr75:+.2f}. "
                           f"{'Positive correlation: rising real rates associated with wider spreads.' if _corr75 > 0.2 else 'Negative correlation: real rate regime not currently driving spreads.' if _corr75 < -0.2 else 'Low correlation in current window.'}")
            else:
                st.info("Scatter unavailable — requires both real_rate_10y and hy_spread columns.")

            # Regime summary table
            st.subheader("Regime Implications")
            _regime_tbl75 = [
                {"Real Rate Regime": "Financial Repression (< −1%)", "Credit Implication": "Spread compression supported; carry trade attractive", "Posture": "Risk-on"},
                {"Real Rate Regime": "Neutral (−1% to +1.5%)",        "Credit Implication": "Credit conditions broadly normal",                   "Posture": "Neutral"},
                {"Real Rate Regime": "Restrictive (> +1.5%)",         "Credit Implication": "Refinancing pressure; HY risk elevated",             "Posture": "Defensive"},
                {"Real Rate Regime": "Highly Restrictive (> +3%)",    "Credit Implication": "Significant default risk uplift in levered credit",  "Posture": "Risk-off"},
            ]
            import pandas as _pd75
            st.dataframe(_pd75.DataFrame(_regime_tbl75), use_container_width=True, hide_index=True)
        else:
            _miss75 = _rr75.get("missing_columns", [])
            st.info(f"Real rates unavailable — missing columns: {', '.join(_miss75) if _miss75 else 'yield_10y and/or breakeven_10y'}.")
    except Exception as _e75:
        _err_track(_active_sub, _e75)
        st.caption(f"Real rates unavailable: {_e75}")



if _active_sub == 81:
    import plotly.graph_objects as _go81
    st.header("Fed Communication Sentiment")
    st.markdown(
        "Scores each **FOMC post-meeting statement** 0–100 for monetary policy tone: "
        "0 = very dovish · 50 = neutral · 100 = very hawkish. "
        "Hawkish drift → tighter financial conditions → wider credit spreads. "
        "Dovish pivot → risk-on rally → spread compression. "
        "The **Taylor Rule** (see next tab) tells you where rates *should* be; "
        "this module tells you what the Fed is *signaling* about where they're heading."
    )
    try:
        _fs81 = load_fed_sentiment(df)
        _cur81 = _fs81.get("current", {})
        _fs81a, _fs81b, _fs81c, _fs81d = st.columns(4)
        _score81 = _cur81.get("score")
        _fs81a.metric("Sentiment Score", f"{_score81:.0f}/100" if _score81 is not None else "—",
                      help="0 = very dovish, 50 = neutral, 100 = very hawkish")
        _fs81b.metric("Tone", _cur81.get("label", "—"))
        _fs81c.metric("Meeting Date", _cur81.get("date", "—") or "—")
        _fs81d.metric("Trend (last 3)", _fs81.get("trend", "—"))

        if _cur81.get("reasoning"):
            st.info(f"**Reasoning:** {_cur81['reasoning']}")

        if not _fs81.get("api_key_present"):
            st.warning("ANTHROPIC_API_KEY not set — sentiment scoring requires LLM. "
                       "Using cached history if available.")

        # History bar chart
        _hist81 = _fs81.get("history", [])
        if _hist81:
            import pandas as _pd81
            _h81_df = _pd81.DataFrame(_hist81)
            _h81_df["date"] = _pd81.to_datetime(_h81_df["date"], format="%Y%m%d", errors="coerce")
            _h81_df = _h81_df.dropna(subset=["date", "score"]).sort_values("date")
            _colors81 = [
                "#ef4444" if s >= 65 else "#27ae60" if s <= 35 else "#f59e0b"
                for s in _h81_df["score"]
            ]
            _fig81 = _go81.Figure()
            _fig81.add_trace(_go81.Bar(
                x=_h81_df["date"], y=_h81_df["score"],
                marker_color=_colors81, name="FOMC Sentiment",
                text=_h81_df["label"], textposition="outside",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<br>%{text}<extra></extra>",
            ))
            _fig81.add_hline(y=50, line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
                             annotation_text="Neutral (50)",
                             annotation_font=dict(color="#9aa0aa", size=9))
            _fig81.add_hrect(y0=65, y1=100, fillcolor="rgba(231,76,60,0.05)", line_width=0,
                             annotation_text="Hawkish zone",
                             annotation_font=dict(color="#e74c3c", size=9))
            _fig81.add_hrect(y0=0, y1=35, fillcolor="rgba(39,174,96,0.05)", line_width=0,
                             annotation_text="Dovish zone",
                             annotation_font=dict(color="#27ae60", size=9))
            _fig81.update_layout(
                height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=24),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Sentiment Score (0–100)", range=[0, 110]),
                xaxis=dict(showgrid=False, color="#6b7280", title="FOMC Meeting Date"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig81, use_container_width=True)
            st.caption("Red = hawkish (≥ 65) · Amber = neutral (35–65) · Green = dovish (≤ 35). "
                       "Credit spreads typically widen in hawkish regimes and tighten on dovish pivots.")

            # Summary statistics
            import numpy as _np81
            _scores81 = [h["score"] for h in _hist81 if h.get("score") is not None]
            if _scores81:
                _s81a, _s81b, _s81c = st.columns(3)
                _s81a.metric("Avg Score (history)", f"{_np81.mean(_scores81):.1f}")
                _s81b.metric("Last 3 Meetings Avg", f"{_np81.mean(_scores81[-3:]):.1f}" if len(_scores81) >= 3 else "—")
                _s81c.metric("Min / Max", f"{min(_scores81):.0f} / {max(_scores81):.0f}")
        else:
            st.info("No FOMC sentiment history available — requires ANTHROPIC_API_KEY and Fed statement access.")
    except Exception as _e81:
        _err_track(_active_sub, _e81)
        st.caption(f"Fed sentiment unavailable: {_e81}")



if _active_sub == 84:
    import plotly.graph_objects as _go84
    st.header("Taylor Rule — Monetary Policy Positioning")
    st.markdown(
        "The **Taylor Rule (1993)** estimates the appropriate Fed funds rate: "
        "r = r\\* + π + 0.5(π − π\\*) + 0.5(y − y\\*). "
        "A **positive policy gap** (actual rate > Taylor prescription) = policy is *overly restrictive* "
        "— historically associated with credit spread widening, slowing loan growth, and rising default risk. "
        "A **negative gap** = policy is accommodative relative to fundamentals."
    )
    try:
        _tr84 = load_taylor(df)
        if _tr84.get("available"):
            _trc = _tr84["current"]
            _tr84a, _tr84b, _tr84c, _tr84d = st.columns(4)
            _tr84a.metric("Fed Funds Rate", f"{_trc.get('fed_funds', float('nan')):.2f}%")
            _tr84b.metric("Taylor Rule Rate", f"{_trc.get('taylor_rate', float('nan')):.2f}%",
                          help="Estimated neutral rate given inflation and output gap")
            _tr84c.metric("Policy Gap", f"{_trc.get('policy_gap', 0):+.2f}pp",
                          delta=f"{_trc.get('policy_gap', 0):+.2f}pp",
                          delta_color="inverse",
                          help="Actual − Taylor rate. Positive = too tight.")
            _tr84d.metric("Policy Stance", _trc.get("stance", "—"))

            # Policy gap history bar chart
            if "df" in _tr84 and "policy_gap" in _tr84["df"].columns:
                _tr84_df = _tr84["df"].copy()
                _tr84_df["date"] = pd.to_datetime(_tr84_df["date"])
                _fig84a = _go84.Figure()
                _pg84 = _tr84_df["policy_gap"].fillna(0)
                _fig84a.add_trace(_go84.Bar(
                    x=_tr84_df["date"], y=_pg84,
                    marker_color=["#ef4444" if v > 0 else "#27ae60" for v in _pg84],
                    name="Policy Gap",
                    hovertemplate="%{x|%Y-%m-%d}<br>Policy Gap: %{y:+.2f}pp<extra></extra>",
                ))
                _fig84a.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
                _fig84a.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Actual − Taylor Rate (pp)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig84a, use_container_width=True)
                st.caption("Red = policy tighter than Taylor Rule prescribes (restrictive) · "
                           "Green = policy looser than prescribed (accommodative)")

            # Fed funds vs Taylor rate overlay
            if "df" in _tr84 and "fed_funds" in _tr84["df"].columns and "taylor_rate" in _tr84["df"].columns:
                _tr84_df2 = _tr84["df"].copy()
                _tr84_df2["date"] = pd.to_datetime(_tr84_df2["date"])
                _fig84b = _go84.Figure()
                _fig84b.add_trace(_go84.Scatter(
                    x=_tr84_df2["date"], y=_tr84_df2["fed_funds"],
                    name="Actual Fed Funds", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Fed Funds: %{y:.2f}%<extra></extra>",
                ))
                _fig84b.add_trace(_go84.Scatter(
                    x=_tr84_df2["date"], y=_tr84_df2["taylor_rate"],
                    name="Taylor Rate", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Taylor Rate: %{y:.2f}%<extra></extra>",
                ))
                _fig84b.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="Actual Fed Funds vs Taylor Rule Prescription",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Rate (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig84b, use_container_width=True)

            # Credit implications table
            st.subheader("Policy Gap → Credit Spread Implications")
            import pandas as _pd84
            _taylor_tbl = _pd84.DataFrame([
                {"Policy Gap Range":   "< −2pp (very accommodative)", "HY Spread Impact": "Significant compression",  "Default Risk": "Low", "Posture": "Risk-on"},
                {"Policy Gap Range":   "−2 to 0pp (accommodative)",   "HY Spread Impact": "Modest compression",       "Default Risk": "Low–Moderate", "Posture": "Risk-on"},
                {"Policy Gap Range":   "0 to +2pp (mildly tight)",     "HY Spread Impact": "Stable / slight widening", "Default Risk": "Moderate", "Posture": "Neutral"},
                {"Policy Gap Range":   "+2 to +4pp (restrictive)",     "HY Spread Impact": "Widening risk elevated",   "Default Risk": "Elevated", "Posture": "Defensive"},
                {"Policy Gap Range":   "> +4pp (very restrictive)",    "HY Spread Impact": "Significant widening",     "Default Risk": "High", "Posture": "Risk-off"},
            ])
            st.dataframe(_taylor_tbl, use_container_width=True, hide_index=True)
            _gap84 = _trc.get("policy_gap", 0)
            st.caption(f"Current policy gap: **{_gap84:+.2f}pp** ({_trc.get('stance', '—')})")
        else:
            st.info("Taylor Rule unavailable — requires FEDFUNDS, CPIAUCSL / T10YIE, UNRATE in dataset.")
    except Exception as _e84:
        _err_track(_active_sub, _e84)
        st.caption(f"Taylor Rule unavailable: {_e84}")



if _active_sub == 85:
    import plotly.graph_objects as _go85
    st.header("Macro Nowcast")
    st.markdown(
        "Real-time GDP growth signal derived from a weighted composite of weekly/monthly indicators: "
        "unemployment, initial claims, equity momentum, yield curve slope, and PMI. "
        "Each indicator is 252-day z-scored and averaged into a **Nowcast Score 0–100**: "
        "above 55 = expansion signal · below 45 = contraction signal. "
        "Credit spreads tend to widen 1–2 quarters after the nowcast moves into contraction."
    )
    try:
        _nc85 = load_macro_nowcast(df)
        if _nc85.get("available"):
            _ncc85 = _nc85["current"]
            _nc85a, _nc85b, _nc85c, _nc85d = st.columns(4)
            _nc85a.metric("Nowcast Score", f"{_ncc85.get('nowcast_score', 0):.1f}/100",
                          delta=f"{_ncc85.get('nowcast_change_1m', 0):+.1f} 1M",
                          help="0–100; above 55 = expansion, below 45 = contraction signal")
            _nc85b.metric("Regime", _ncc85.get("nowcast_regime", "—"))
            _nc85c.metric("Momentum", _nc85.get("momentum", "—"))
            _nc85d.metric("Recession Prob", f"{_ncc85.get('nowcast_recession_prob', 0):.1%}")

            _indicators85 = _ncc85.get("indicators_used", [])
            if _indicators85:
                st.caption(f"Indicators in model: **{', '.join(_indicators85)}**")

            if _nc85.get("interpretation"):
                st.info(_nc85["interpretation"])

            # Nowcast score history
            _nc85_hist = _nc85.get("historical")
            if _nc85_hist is not None and not _nc85_hist.empty and "nowcast_score" in _nc85_hist.columns:
                _nc85_hist = _nc85_hist.copy()
                _nc85_hist.index = pd.to_datetime(_nc85_hist.index)
                _fig85a = _go85.Figure()
                _fig85a.add_trace(_go85.Scatter(
                    x=_nc85_hist.index, y=_nc85_hist["nowcast_score"],
                    name="Nowcast Score", line=dict(color="#27ae60", width=2),
                    fill="tozeroy", fillcolor="rgba(39,174,96,0.10)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Nowcast: %{y:.1f}<extra></extra>",
                ))
                _fig85a.add_hline(y=55, line=dict(color="rgba(39,174,96,0.5)", dash="dash", width=1),
                                  annotation_text="Expansion threshold (55)",
                                  annotation_font=dict(color="#27ae60", size=9))
                _fig85a.add_hline(y=45, line=dict(color="rgba(231,76,60,0.5)", dash="dash", width=1),
                                  annotation_text="Contraction signal (45)",
                                  annotation_font=dict(color="#e74c3c", size=9))
                _fig85a.add_hrect(y0=0, y1=45, fillcolor="rgba(231,76,60,0.04)", line_width=0)
                _fig85a.add_hrect(y0=55, y1=100, fillcolor="rgba(39,174,96,0.04)", line_width=0)
                _fig85a.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Score (0–100)", range=[0, 100]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig85a, use_container_width=True)

                # Nowcast vs HY spread overlay (credit lead-lag)
                if "hy_spread" in df.columns:
                    _nc85_hy = _nc85_hist[["nowcast_score"]].join(
                        df[["hy_spread"]].dropna(), how="inner"
                    ).dropna().tail(504)
                    if not _nc85_hy.empty:
                        _fig85b = _go85.Figure()
                        _fig85b.add_trace(_go85.Scatter(
                            x=_nc85_hy.index, y=_nc85_hy["nowcast_score"],
                            name="Nowcast Score", line=dict(color="#27ae60", width=1.5),
                            yaxis="y1",
                            hovertemplate="%{x|%Y-%m-%d}<br>Nowcast: %{y:.1f}<extra></extra>",
                        ))
                        _fig85b.add_trace(_go85.Scatter(
                            x=_nc85_hy.index, y=_nc85_hy["hy_spread"],
                            name="HY OAS (%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                            yaxis="y2",
                            hovertemplate="%{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>",
                        ))
                        _fig85b.update_layout(
                            height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#9aa0aa", size=11),
                            margin=dict(l=8, r=8, t=24, b=8),
                            title=dict(text="Nowcast Score vs HY Spread (falling nowcast leads widening)",
                                       font=dict(size=12, color="#9aa0aa")),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                       color="#27ae60", title="Nowcast Score"),
                            yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                        title="HY OAS (%)", showgrid=False),
                            xaxis=dict(showgrid=False, color="#6b7280"),
                            legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                        )
                        st.plotly_chart(_fig85b, use_container_width=True)
                        _nc_corr85 = float(_nc85_hy["nowcast_score"].corr(_nc85_hy["hy_spread"]))
                        st.caption(f"Nowcast vs HY spread correlation (2yr): {_nc_corr85:+.2f}. "
                                   f"{'Negative correlation expected: lower nowcast → wider spreads.' if _nc_corr85 < -0.2 else ''}")

            # Regime implication table
            st.subheader("Nowcast → Credit Regime Mapping")
            import pandas as _pd85
            _nc_tbl85 = _pd85.DataFrame([
                {"Nowcast Score": "> 65",    "Macro Regime": "Strong expansion",   "HY Spread Direction": "Tightening", "Posture": "Risk-on"},
                {"Nowcast Score": "55–65",   "Macro Regime": "Expansion",          "HY Spread Direction": "Stable/tightening", "Posture": "Risk-on"},
                {"Nowcast Score": "45–55",   "Macro Regime": "Neutral/transition", "HY Spread Direction": "Stable", "Posture": "Neutral"},
                {"Nowcast Score": "35–45",   "Macro Regime": "Contraction signal", "HY Spread Direction": "Widening risk", "Posture": "Defensive"},
                {"Nowcast Score": "< 35",    "Macro Regime": "Contraction",        "HY Spread Direction": "Significant widening", "Posture": "Risk-off"},
            ])
            st.dataframe(_nc_tbl85, use_container_width=True, hide_index=True)
            st.caption(f"Current nowcast: **{_ncc85.get('nowcast_score', 0):.1f}** ({_ncc85.get('nowcast_regime', '—')})")
        else:
            st.info("Macro nowcast unavailable — requires at least 2 indicators (unemployment/claims/SP500/yield curve).")
    except Exception as _e85:
        _err_track(_active_sub, _e85)
        st.caption(f"Macro nowcast unavailable: {_e85}")


# =============================================================================
# BATCH 10 ANALYTICS: sub86–91
# =============================================================================


if _active_sub == 90:
    import plotly.graph_objects as _go90
    st.header("Rates & Credit Term Structure")
    st.markdown(
        "The joint view of the **Treasury yield curve** (3m/2y/10y slopes) and the "
        "**credit quality slope** (IG → HY spread differential) describes where stress "
        "sits in the maturity spectrum. "
        "Front-end Treasury inversion (3m > 10y) has preceded every US recession since 1969. "
        "Front-end **credit** stress = near-term liquidity fear. Back-end = solvency concerns."
    )
    try:
        _ts90 = load_term_structure(df)
        if _ts90.get("available"):
            _tsc90 = _ts90.get("current", {})
            _ts90a, _ts90b, _ts90c, _ts90d = st.columns(4)
            _ts90a.metric("2s10s Slope", f"{_tsc90.get('ts_curve_slope_2s10s', float('nan')):+.2f}pp",
                          help="10y − 2y Treasury. Negative = inverted.")
            _ts90b.metric("3m10y Slope", f"{_tsc90.get('ts_curve_slope_3m10y', float('nan')):+.2f}pp",
                          help="10y − 3m Treasury. Estrella-Mishkin recession predictor.")
            _ts90c.metric("HY−IG Credit Slope", f"{_tsc90.get('ts_credit_slope_hy_ig', float('nan')):+.2f}pp",
                          help="HY OAS minus IG OAS. High = steep quality curve = risk aversion.")
            _ts90d.metric("Curve Regime", _tsc90.get("ts_curve_regime", "—"))

            if _tsc90.get("interpretation"):
                st.info(_tsc90["interpretation"])

            # Percentile ranks row
            _ts90_pct = _ts90.get("history_percentiles", {})
            if _ts90_pct:
                _pct90_cols = st.columns(len(_ts90_pct))
                for _pc90, (_pk90, _pv90) in zip(_pct90_cols, _ts90_pct.items()):
                    _short90 = _pk90.replace("ts_", "").replace("_zscore", " z").replace("_", " ").title()
                    _pc90.metric(f"{_short90} Pctile", f"{_pv90:.0f}th" if _pv90 is not None else "—")

            # Treasury curve slopes over time
            _ts90_df = _ts90.get("df")
            if _ts90_df is not None and "ts_curve_slope_2s10s" in _ts90_df.columns:
                _ts90_plot = _ts90_df.copy()
                _ts90_plot.index = pd.to_datetime(_ts90_plot.index)
                _ts90_plot = _ts90_plot.tail(504)
                _fig90a = _go90.Figure()
                _fig90a.add_trace(_go90.Scatter(
                    x=_ts90_plot.index, y=_ts90_plot["ts_curve_slope_2s10s"],
                    name="2s10s (Treasury)", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>2s10s: %{y:+.2f}pp<extra></extra>",
                ))
                if "ts_curve_slope_3m10y" in _ts90_plot.columns:
                    _fig90a.add_trace(_go90.Scatter(
                        x=_ts90_plot.index, y=_ts90_plot["ts_curve_slope_3m10y"],
                        name="3m10y (Treasury)", line=dict(color="#27ae60", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>3m10y: %{y:+.2f}pp<extra></extra>",
                    ))
                if "ts_credit_slope_hy_ig" in _ts90_plot.columns:
                    _fig90a.add_trace(_go90.Scatter(
                        x=_ts90_plot.index, y=_ts90_plot["ts_credit_slope_hy_ig"],
                        name="HY−IG (Credit Slope)", line=dict(color="#ef4444", width=2),
                        yaxis="y2",
                        hovertemplate="%{x|%Y-%m-%d}<br>HY−IG: %{y:+.2f}pp<extra></extra>",
                    ))
                _fig90a.add_hline(y=0, line_color="rgba(231,76,60,0.5)", line_width=1.5,
                                  annotation_text="Inversion",
                                  annotation_font=dict(color="#e74c3c", size=10))
                _fig90a.update_layout(
                    height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#4f8ef7", title="Treasury Slope (pp)"),
                    yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                title="HY−IG Slope (pp)", showgrid=False),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig90a, use_container_width=True)
                st.caption("Below 0 = inverted. 3m10y inversion has preceded every US recession since 1969.")

            # Regime implication table
            st.subheader("Curve Shape → Credit Implication")
            import pandas as _pd90
            _ts_tbl90 = _pd90.DataFrame([
                {"Curve Shape":         "Steep (2s10s > +1.5pp)",   "Economic Signal": "Growth expected",     "Credit Signal": "Spreads likely tightening", "Posture": "Risk-on"},
                {"Curve Shape":         "Flat (0 to +1.5pp)",       "Economic Signal": "Late cycle",          "Credit Signal": "Watch for deterioration",   "Posture": "Neutral"},
                {"Curve Shape":         "Inverted (2s10s < 0)",     "Economic Signal": "Recession signal",    "Credit Signal": "Widening risk elevated",    "Posture": "Defensive"},
                {"Curve Shape":         "Deep inversion (< −1pp)",  "Economic Signal": "Recession likely",    "Credit Signal": "Significant widening",      "Posture": "Risk-off"},
            ])
            st.dataframe(_ts_tbl90, use_container_width=True, hide_index=True)
        else:
            st.info("Term structure unavailable — requires Treasury yield columns (yield_3m, yield_2y, yield_10y).")
    except Exception as _e90:
        _err_track(_active_sub, _e90)
        st.caption(f"Term structure unavailable: {_e90}")



if _active_sub == 92:
    import plotly.graph_objects as _go92
    st.header("NFCI — National Financial Conditions Index")
    st.markdown(
        "The **Chicago Fed NFCI** is a weekly index of US financial conditions across "
        "money markets, debt and equity markets, and the traditional and shadow banking systems. "
        "**Positive values** = tighter-than-average financial conditions (stress). "
        "**Negative values** = looser-than-average (accommodative). "
        "NFCI > 0.5 historically precedes HY spread widening; NFCI > 1.0 = significant tightening."
    )
    try:
        if "nfci" in df.columns and df["nfci"].notna().sum() > 20:
            _nfci_df = df[["nfci"]].copy()
            if "nfci_90d_avg" in df.columns:
                _nfci_df["nfci_90d_avg"] = df["nfci_90d_avg"]
            if "nfci_change_90d" in df.columns:
                _nfci_df["nfci_change_90d"] = df["nfci_change_90d"]
            _nfci_df.index = pd.to_datetime(_nfci_df.index)
            _nfci_latest = float(_nfci_df["nfci"].dropna().iloc[-1])
            _nfci_90d_latest = float(_nfci_df["nfci_90d_avg"].dropna().iloc[-1]) if "nfci_90d_avg" in _nfci_df.columns else float("nan")
            _nfci_chg = float(_nfci_df["nfci_change_90d"].dropna().iloc[-1]) if "nfci_change_90d" in _nfci_df.columns else float("nan")

            _nfci_regime = (
                "Tightening (Stress)" if _nfci_latest > 0.5
                else "Elevated" if _nfci_latest > 0.0
                else "Neutral" if _nfci_latest > -0.5
                else "Loose (Accommodative)"
            )

            _n92a, _n92b, _n92c, _n92d = st.columns(4)
            _n92a.metric("NFCI (latest)", f"{_nfci_latest:.3f}",
                         help="Positive = tighter than average. Normal range: −1 to +1.")
            _n92b.metric("90d Average", f"{_nfci_90d_latest:.3f}" if not pd.isna(_nfci_90d_latest) else "—")
            _n92c.metric("90d Change", f"{_nfci_chg:+.3f}" if not pd.isna(_nfci_chg) else "—",
                         delta_color="inverse")
            _n92d.metric("Regime", _nfci_regime)

            if _nfci_latest > 0.5:
                st.warning(f"NFCI above 0.5 ({_nfci_latest:.3f}) — financial conditions tightening materially. "
                           "Historically associated with HY spread widening within 4–8 weeks.")
            elif _nfci_latest < -0.5:
                st.success(f"NFCI below −0.5 ({_nfci_latest:.3f}) — accommodative conditions. "
                           "Favorable for credit carry strategies.")

            # NFCI time series with regime bands
            _nfci_plot = _nfci_df.tail(756)
            _fig92a = _go92.Figure()
            _fig92a.add_trace(_go92.Scatter(
                x=_nfci_plot.index, y=_nfci_plot["nfci"],
                name="NFCI", line=dict(color="#06b6d4", width=2),
                fill="tozeroy", fillcolor="rgba(6,182,212,0.10)",
                hovertemplate="%{x|%Y-%m-%d}<br>NFCI: %{y:.3f}<extra></extra>",
            ))
            if "nfci_90d_avg" in _nfci_plot.columns:
                _fig92a.add_trace(_go92.Scatter(
                    x=_nfci_plot.index, y=_nfci_plot["nfci_90d_avg"],
                    name="90d Average", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>90d Avg: %{y:.3f}<extra></extra>",
                ))
            _fig92a.add_hline(y=0,   line=dict(color="rgba(255,255,255,0.3)", width=1))
            _fig92a.add_hline(y=0.5, line=dict(color="#ef4444", dash="dash", width=1),
                              annotation_text="Tightening (0.5)",
                              annotation_font=dict(color="#ef4444", size=9))
            _fig92a.add_hline(y=1.0, line=dict(color="#9b59b6", dash="dot", width=1),
                              annotation_text="Stress (1.0)",
                              annotation_font=dict(color="#9b59b6", size=9))
            _fig92a.update_layout(
                height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="NFCI"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig92a, use_container_width=True)
            st.caption("Above 0 = tighter than average · Below 0 = looser than average · "
                       "Weekly frequency (Chicago Fed release).")

            # NFCI vs HY spread
            if "hy_spread" in df.columns:
                _nfci_hy = _nfci_df[["nfci"]].join(df[["hy_spread"]].dropna(), how="inner").dropna().tail(504)
                if not _nfci_hy.empty:
                    _fig92b = _go92.Figure()
                    _fig92b.add_trace(_go92.Scatter(
                        x=_nfci_hy.index, y=_nfci_hy["nfci"],
                        name="NFCI", line=dict(color="#06b6d4", width=1.5),
                        yaxis="y1",
                        hovertemplate="%{x|%Y-%m-%d}<br>NFCI: %{y:.3f}<extra></extra>",
                    ))
                    _fig92b.add_trace(_go92.Scatter(
                        x=_nfci_hy.index, y=_nfci_hy["hy_spread"],
                        name="HY OAS (%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                        yaxis="y2",
                        hovertemplate="%{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>",
                    ))
                    _fig92b.update_layout(
                        height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11),
                        margin=dict(l=8, r=8, t=24, b=8),
                        title=dict(text="NFCI vs HY Spread", font=dict(size=12, color="#9aa0aa")),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#06b6d4", title="NFCI"),
                        yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                    title="HY OAS (%)", showgrid=False),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig92b, use_container_width=True)
                    _nfci_corr = float(_nfci_hy["nfci"].corr(_nfci_hy["hy_spread"]))
                    st.caption(f"NFCI vs HY spread correlation (2yr): {_nfci_corr:+.2f}.")

            # Implication table
            import pandas as _pd92
            st.subheader("NFCI → Credit Implication")
            st.dataframe(_pd92.DataFrame([
                {"NFCI Range": "< −0.5",        "Financial Conditions": "Loose/accommodative",  "Credit Impact": "Spread compression", "Posture": "Risk-on"},
                {"NFCI Range": "−0.5 to 0",     "Financial Conditions": "Slightly easy",        "Credit Impact": "Neutral/stable",     "Posture": "Neutral"},
                {"NFCI Range": "0 to +0.5",     "Financial Conditions": "Slightly tight",       "Credit Impact": "Watch for widening", "Posture": "Cautious"},
                {"NFCI Range": "+0.5 to +1.0",  "Financial Conditions": "Tightening",           "Credit Impact": "Widening likely",    "Posture": "Defensive"},
                {"NFCI Range": "> +1.0",         "Financial Conditions": "Significant stress",  "Credit Impact": "Material widening",  "Posture": "Risk-off"},
            ]), use_container_width=True, hide_index=True)
        else:
            st.info("NFCI unavailable — requires `nfci` column in dataset (FRED series NFCI).")
    except Exception as _e92:
        _err_track(_active_sub, _e92)
        st.caption(f"NFCI unavailable: {_e92}")



if _active_sub == 93:
    import plotly.graph_objects as _go93
    st.header("Sahm Rule — Labor Market Early Warning")
    st.markdown(
        "The **Sahm Rule** (Claudia Sahm, 2019) triggers when the 3-month average unemployment "
        "rate rises ≥ 0.5pp above its 12-month low — a real-time recession signal with near-perfect "
        "historical accuracy. "
        "For credit: rising unemployment → rising default rates with a 2–4 quarter lag. "
        "The `sahm_like` indicator here uses the daily dataset's unemployment interpolation "
        "as a proxy for the monthly Sahm calculation."
    )
    try:
        _sahm_avail = "sahm_like" in df.columns and df["sahm_like"].notna().sum() > 10
        _unemp_avail = "unemployment" in df.columns and df["unemployment"].notna().sum() > 10
        if _sahm_avail or _unemp_avail:
            _s93_df = df[[]].copy()
            if _sahm_avail:
                _s93_df["sahm_like"] = df["sahm_like"]
            if _unemp_avail:
                _s93_df["unemployment"] = df["unemployment"]
            if "unemployment_change_90d" in df.columns:
                _s93_df["unemployment_change_90d"] = df["unemployment_change_90d"]
            if "labor_warning" in df.columns:
                _s93_df["labor_warning"] = df["labor_warning"]
            _s93_df.index = pd.to_datetime(_s93_df.index)

            _sahm_now = float(_s93_df["sahm_like"].dropna().iloc[-1]) if _sahm_avail else float("nan")
            _unemp_now = float(_s93_df["unemployment"].dropna().iloc[-1]) if _unemp_avail else float("nan")
            _unemp_chg = float(_s93_df["unemployment_change_90d"].dropna().iloc[-1]) if "unemployment_change_90d" in _s93_df.columns else float("nan")
            _labor_warn = str(_s93_df["labor_warning"].dropna().iloc[-1]) if "labor_warning" in _s93_df.columns else "—"

            _s93a, _s93b, _s93c, _s93d = st.columns(4)
            _s93a.metric("Sahm-Like Indicator", f"{_sahm_now:.2f}pp" if not pd.isna(_sahm_now) else "—",
                         delta_color="inverse",
                         help="Unemployment minus 12m rolling min. ≥ 0.5pp = Sahm Rule triggered.")
            _s93b.metric("Unemployment Rate", f"{_unemp_now:.1f}%" if not pd.isna(_unemp_now) else "—")
            _s93c.metric("Unemp 90d Change", f"{_unemp_chg:+.2f}pp" if not pd.isna(_unemp_chg) else "—",
                         delta_color="inverse")
            _s93d.metric("Labor Warning", _labor_warn)

            _sahm_triggered = not pd.isna(_sahm_now) and _sahm_now >= 0.5
            if _sahm_triggered:
                st.error(f"Sahm Rule TRIGGERED: indicator at {_sahm_now:.2f}pp (threshold: 0.5pp). "
                         "Historical recession signal with near-perfect accuracy. "
                         "Default rates typically rise 2–4 quarters after trigger.")
            elif not pd.isna(_sahm_now) and _sahm_now >= 0.3:
                st.warning(f"Sahm indicator approaching threshold ({_sahm_now:.2f}pp / 0.5pp threshold). "
                           "Monitor unemployment trend closely.")

            # Sahm-like history chart
            if _sahm_avail:
                _s93_plot = _s93_df[["sahm_like"]].dropna().tail(756)
                _fig93a = _go93.Figure()
                _s93_colors = ["#ef4444" if v >= 0.5 else "#f59e0b" if v >= 0.3 else "#27ae60"
                               for v in _s93_plot["sahm_like"].fillna(0)]
                _fig93a.add_trace(_go93.Bar(
                    x=_s93_plot.index, y=_s93_plot["sahm_like"],
                    marker_color=_s93_colors, name="Sahm-Like",
                    hovertemplate="%{x|%Y-%m-%d}<br>Sahm: %{y:.2f}pp<extra></extra>",
                ))
                _fig93a.add_hline(y=0.5, line=dict(color="#ef4444", dash="dash", width=1.5),
                                  annotation_text="Recession signal (0.5pp)",
                                  annotation_font=dict(color="#ef4444", size=10))
                _fig93a.add_hline(y=0.3, line=dict(color="#f59e0b", dash="dot", width=1),
                                  annotation_text="Watch zone (0.3pp)",
                                  annotation_font=dict(color="#f59e0b", size=9))
                _fig93a.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Unemployment Rise from 12M Low (pp)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig93a, use_container_width=True)

            # Unemployment level + HY spread lead-lag
            if _unemp_avail and "hy_spread" in df.columns:
                _s93_hy = _s93_df[["unemployment"]].join(df[["hy_spread"]].dropna(), how="inner").dropna().tail(756)
                if not _s93_hy.empty:
                    _fig93b = _go93.Figure()
                    _fig93b.add_trace(_go93.Scatter(
                        x=_s93_hy.index, y=_s93_hy["unemployment"],
                        name="Unemployment (%)", line=dict(color="#e67e22", width=2),
                        yaxis="y1",
                        hovertemplate="%{x|%Y-%m-%d}<br>Unemp: %{y:.1f}%<extra></extra>",
                    ))
                    _fig93b.add_trace(_go93.Scatter(
                        x=_s93_hy.index, y=_s93_hy["hy_spread"],
                        name="HY OAS (%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                        yaxis="y2",
                        hovertemplate="%{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>",
                    ))
                    _fig93b.update_layout(
                        height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa", size=11),
                        margin=dict(l=8, r=8, t=24, b=8),
                        title=dict(text="Unemployment vs HY Spread (unemployment lags credit by ~2 qtrs)",
                                   font=dict(size=12, color="#9aa0aa")),
                        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                   color="#e67e22", title="Unemployment (%)"),
                        yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                    title="HY OAS (%)", showgrid=False),
                        xaxis=dict(showgrid=False, color="#6b7280"),
                        legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    )
                    st.plotly_chart(_fig93b, use_container_width=True)
                    st.caption("HY spreads lead unemployment by ~2 quarters: spreads widen first, "
                               "then defaults rise, then unemployment increases.")

            # Labor warning implication table
            import pandas as _pd93
            st.subheader("Labor Warning → Credit Cycle Implication")
            st.dataframe(_pd93.DataFrame([
                {"Sahm Level": "< 0.2pp",    "Labor Signal": "Strong labor market",   "Default Outlook": "Low",       "Credit Posture": "Risk-on"},
                {"Sahm Level": "0.2–0.3pp",  "Labor Signal": "Softening",             "Default Outlook": "Rising",    "Credit Posture": "Neutral"},
                {"Sahm Level": "0.3–0.5pp",  "Labor Signal": "Deteriorating",         "Default Outlook": "Elevated",  "Credit Posture": "Defensive"},
                {"Sahm Level": "≥ 0.5pp",    "Labor Signal": "Recession signal",      "Default Outlook": "High",      "Credit Posture": "Risk-off"},
            ]), use_container_width=True, hide_index=True)
            st.caption(f"Current: Sahm-like = **{_sahm_now:.2f}pp** · Unemployment = **{_unemp_now:.1f}%** · "
                       f"Labor warning: **{_labor_warn}**")
        else:
            st.info("Sahm Rule unavailable — requires `unemployment` or `sahm_like` columns in dataset.")
    except Exception as _e93:
        _err_track(_active_sub, _e93)
        st.caption(f"Sahm Rule unavailable: {_e93}")



if _active_sub == 103:
    import plotly.graph_objects as _go103
    st.header("Yield Curve Velocity")
    st.markdown(
        "**Curve velocity** measures the 90-day rate-of-change of the 10y-2y yield spread — how fast the curve "
        "is steepening or flattening. Bear steepening (long rates rising faster than short) compresses credit "
        "valuations and widens spreads. Bull flattening (short rates falling as recession risk rises) is a "
        "late-cycle signal. Velocity above +50 bps/quarter warrants caution on duration; below -50 bps/quarter "
        "signals a policy pivot may be underway."
    )
    try:
        _cv103_col = "curve_steepening_velocity_90d"
        if _cv103_col in df.columns:
            _cv103 = df[[_cv103_col, "spread", "yield_10y", "yield_2y"]].copy()
            _cv103.index = pd.to_datetime(_cv103.index)

            _cur_vel = float(latest.get(_cv103_col, float("nan")))
            _cur_spread = float(latest.get("spread", float("nan")))
            _cur_10y = float(latest.get("yield_10y", float("nan")))
            _cur_2y = float(latest.get("yield_2y", float("nan")))

            def _curve_vel_regime(v):
                if pd.isna(v):
                    return "Unknown"
                if v > 50:
                    return "Bear Steepening"
                if v > 15:
                    return "Steepening"
                if v >= -15:
                    return "Stable"
                if v >= -50:
                    return "Flattening"
                return "Rapid Flattening"

            _vel_regime = _curve_vel_regime(_cur_vel)

            _va, _vb, _vc, _vd = st.columns(4)
            _va.metric("Curve Velocity (90d)", f"{_cur_vel:+.0f} bps/qtr" if pd.notna(_cur_vel) else "—",
                       delta="steepening" if pd.notna(_cur_vel) and _cur_vel > 0 else "flattening" if pd.notna(_cur_vel) else None,
                       delta_color="normal" if pd.notna(_cur_vel) and _cur_vel > 0 else "inverse")
            _vb.metric("2s10s Spread", f"{_cur_spread:.0f} bps" if pd.notna(_cur_spread) else "—")
            _vc.metric("Velocity Regime", _vel_regime)
            _vd.metric("10y Yield", f"{_cur_10y:.2f}%" if pd.notna(_cur_10y) else "—")

            if pd.notna(_cur_vel) and abs(_cur_vel) > 50:
                if _cur_vel > 0:
                    st.warning("Bear steepening alert: long rates rising rapidly faster than short rates. "
                               "Credit duration risk elevated; expect spread widening pressure on long-dated bonds.")
                else:
                    st.info("Rapid curve flattening: often precedes a bull market in credit as recession risk rises "
                            "and central bank cuts are priced in. Favour short-dated HY / IG.")

            # Curve velocity time series with regime colors
            _cv103_tail = _cv103.tail(756)
            _vel_colors = [
                "#ef4444" if v > 50 else "#f59e0b" if v > 15 else "#27ae60" if v > -15 else
                "#a78bfa" if v >= -50 else "#4f8ef7"
                for v in _cv103_tail[_cv103_col].fillna(0)
            ]
            _fig103a = _go103.Figure()
            _fig103a.add_trace(_go103.Bar(
                x=_cv103_tail.index, y=_cv103_tail[_cv103_col],
                marker_color=_vel_colors, name="Curve Velocity",
                hovertemplate="%{x|%Y-%m-%d}<br>Velocity: %{y:+.0f}bps/qtr<extra></extra>",
            ))
            _fig103a.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
            _fig103a.add_hline(y=50, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig103a.add_hline(y=-50, line=dict(color="rgba(79,142,247,0.4)", dash="dot", width=1))
            _fig103a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Yield Curve 90-Day Velocity (bps/quarter)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Δ Spread (bps, 90d)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig103a, use_container_width=True)
            st.caption("Red = bear steepening (>50 bps/qtr) · Blue = rapid flattening (<-50 bps/qtr)")

            # Dual-axis: curve velocity vs HY spread
            if "hy_spread" in df.columns:
                _cv103_dual = _cv103_tail.copy()
                _cv103_dual["hy_spread"] = df["hy_spread"].reindex(_cv103_dual.index)
                _fig103b = _go103.Figure()
                _fig103b.add_trace(_go103.Scatter(
                    x=_cv103_dual.index, y=_cv103_dual["spread"],
                    name="2s10s Spread (bps)", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Curve: %{y:.0f}bps<extra></extra>",
                ))
                _fig103b.add_trace(_go103.Scatter(
                    x=_cv103_dual.index, y=_cv103_dual["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig103b.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=60),
                    title=dict(text="2s10s Spread vs HY Spread", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#4f8ef7",
                               title="2s10s (bps)"),
                    yaxis2=dict(overlaying="y", side="right", color="#f59e0b",
                                title="HY Spread (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig103b, use_container_width=True)

            st.markdown("**Curve Velocity Regime Implications**")
            st.table(pd.DataFrame([
                {"Regime": "Bear Steepening (>50)", "Driver": "Long rates rising, inflation/supply fears",
                 "Credit Impact": "Negative: duration compression, potential spread widening on long bonds"},
                {"Regime": "Steepening (15–50)", "Driver": "Normalisation / recovery pricing",
                 "Credit Impact": "Neutral to mild negative for long duration; short-dated OK"},
                {"Regime": "Stable (±15)", "Driver": "Range-bound curve",
                 "Credit Impact": "Benign; carry trade conditions intact"},
                {"Regime": "Flattening (-50 to -15)", "Driver": "Front-end pricing cuts or long-end anchoring",
                 "Credit Impact": "Positive for long credit; late cycle signal worth monitoring"},
                {"Regime": "Rapid Flattening (<-50)", "Driver": "Recession pricing / aggressive easing expectations",
                 "Credit Impact": "IG positive; HY negative — quality rotation historically observed"},
            ]))
        else:
            st.info("curve_steepening_velocity_90d column not found — run the feature pipeline first.")
    except Exception as _e103:
        _err_track(_active_sub, _e103)
        st.caption(f"Curve velocity: {_e103}")

# =============================================================================
# BATCH 13 ANALYTICS — sub104–109
# sub104  Yield Curve Regime Deep Dive     → tab_macro
# sub105  HY Momentum Term Structure       → tab_credit
# sub106  Real Yield Z-Score Monitor       → tab_macro
# sub107  Liquidity Sub-Score Deep Dive    → tab_risk
# sub108  Cross-Asset Divergence Detail    → tab_siglab
# sub109  Unemployment Momentum            → tab_macro
# =============================================================================


if _active_sub == 104:
    import plotly.graph_objects as _go104
    st.header("Yield Curve Regime Deep Dive")
    st.markdown(
        "The **yield curve regime** (2s10s spread level) is one of the most durable leading indicators in credit. "
        "A sustained **inversion** (2y > 10y) has preceded every U.S. recession since 1970 with a typical lag of "
        "6–18 months. Credit spreads tend to tighten *during* inversion (Fed still hiking) then widen sharply "
        "*after* disinversion as recession risk crystallises. "
        "**Regimes:** Inverted (<0) · Flat/Inversion Risk (0–0) · Normal (0–1%) · Steep Expansion (>1%)."
    )
    try:
        _cr104_col = "yield_curve_regime"
        if _cr104_col in df.columns and "spread" in df.columns:
            _cr104 = df[["spread", _cr104_col, "hy_spread"]].copy()
            _cr104.index = pd.to_datetime(_cr104.index)
            _cur_regime = str(latest.get(_cr104_col, "Unknown"))
            _cur_spread = float(latest.get("spread", float("nan")))

            _regime_colors = {
                "Steep (Expansion)": "#27ae60",
                "Normal": "#4f8ef7",
                "Flat / Inversion Risk": "#f59e0b",
                "Inverted (Recession Risk)": "#ef4444",
            }
            _cur_color = _regime_colors.get(_cur_regime, "#6b7280")

            _ra, _rb, _rc, _rd = st.columns(4)
            _ra.metric("2s10s Spread", f"{_cur_spread:.2f}pp" if pd.notna(_cur_spread) else "—",
                       delta=f"{latest.get('spread_change_90d', float('nan')):+.2f}pp 90d" if pd.notna(latest.get("spread_change_90d")) else None,
                       delta_color="normal")
            _rb.metric("Curve Regime", _cur_regime)
            _rc_val = (_cr104[_cr104_col] == "Inverted (Recession Risk)").rolling(252).mean().iloc[-1]
            _rc.metric("Inversion Freq (1Y)", f"{_rc_val:.0%}" if pd.notna(_rc_val) else "—")
            _cr104["days_in_regime"] = (_cr104[_cr104_col] == _cur_regime).astype(int)
            _streak = 0
            for _v in reversed(_cr104["days_in_regime"].tolist()):
                if _v == 1:
                    _streak += 1
                else:
                    break
            _rd.metric("Current Streak", f"{_streak}d")

            # Regime timeline
            _regime_num_map = {
                "Steep (Expansion)": 4, "Normal": 3,
                "Flat / Inversion Risk": 2, "Inverted (Recession Risk)": 1,
            }
            _cr104["regime_num"] = _cr104[_cr104_col].map(_regime_num_map).fillna(2)
            _cr104_tail = _cr104.tail(756)
            _bar_colors104 = [_regime_colors.get(r, "#6b7280") for r in _cr104_tail[_cr104_col]]
            _fig104a = _go104.Figure()
            _fig104a.add_trace(_go104.Bar(
                x=_cr104_tail.index, y=_cr104_tail["regime_num"],
                marker_color=_bar_colors104, name="Curve Regime",
                hovertemplate="%{x|%Y-%m-%d}<br>Regime: %{customdata}<extra></extra>",
                customdata=_cr104_tail[_cr104_col],
            ))
            _fig104a.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Yield Curve Regime (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(tickvals=[1, 2, 3, 4],
                           ticktext=["Inverted", "Flat", "Normal", "Steep"],
                           showgrid=False, color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig104a, use_container_width=True)

            # 2s10s spread time series with zero line
            _fig104b = _go104.Figure()
            _spread104 = _cr104_tail["spread"]
            _pos_color = "rgba(39,174,96,0.4)"
            _neg_color = "rgba(239,68,68,0.4)"
            _fig104b.add_trace(_go104.Scatter(
                x=_cr104_tail.index, y=_spread104.clip(lower=0),
                fill="tozeroy", fillcolor=_pos_color,
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            ))
            _fig104b.add_trace(_go104.Scatter(
                x=_cr104_tail.index, y=_spread104.clip(upper=0),
                fill="tozeroy", fillcolor=_neg_color,
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
            ))
            _fig104b.add_trace(_go104.Scatter(
                x=_cr104_tail.index, y=_spread104,
                line=dict(color="#e2e8f0", width=1.5), name="2s10s Spread",
                hovertemplate="%{x|%Y-%m-%d}<br>2s10s: %{y:.2f}pp<extra></extra>",
            ))
            _fig104b.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
            _fig104b.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="pp"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig104b, use_container_width=True)
            st.caption("Green = positive (normal/steep) · Red = negative (inverted)")

            # HY spread distribution by curve regime
            if "hy_spread" in _cr104.columns:
                _fig104c = _go104.Figure()
                for _reg, _rcol in _regime_colors.items():
                    _mask = _cr104[_cr104_col] == _reg
                    _hy_vals = _cr104.loc[_mask, "hy_spread"].dropna()
                    if len(_hy_vals) > 10:
                        _fig104c.add_trace(_go104.Box(
                            y=_hy_vals, name=_reg.split(" ")[0],
                            marker_color=_rcol, line_color=_rcol,
                            boxmean=True,
                            hovertemplate=f"{_reg}<br>HY: %{{y:.0f}}bps<extra></extra>",
                        ))
                _fig104c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="HY Spread Distribution by Curve Regime (full history)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="HY Spread (bps)"),
                    xaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig104c, use_container_width=True)

            st.markdown("**Curve Regime → Credit Implications**")
            st.table(pd.DataFrame([
                {"Regime": "Steep (>1pp)", "Typical Credit Environment": "Early-cycle; spreads tightening",
                 "HY Action": "Overweight; carry attractive"},
                {"Regime": "Normal (0–1pp)", "Typical Credit Environment": "Mid-cycle; balanced conditions",
                 "HY Action": "Neutral weight; monitor momentum"},
                {"Regime": "Flat (0 to −0.5pp)", "Typical Credit Environment": "Late-cycle warning; compression risk",
                 "HY Action": "Reduce beta; upgrade quality"},
                {"Regime": "Inverted (<−0.5pp)", "Typical Credit Environment": "Recession risk; lagged spread widening expected",
                 "HY Action": "Defensive: IG, cash, short-dated HY"},
            ]))
        else:
            st.info("yield_curve_regime or spread column not found — run the feature pipeline.")
    except Exception as _e104:
        _err_track(_active_sub, _e104)
        st.caption(f"Curve regime: {_e104}")


if _active_sub == 106:
    import plotly.graph_objects as _go106
    st.header("Real Yield Z-Score Monitor")
    st.markdown(
        "The **real yield z-score** normalises the 10y real yield (nominal minus TIPS breakeven) against "
        "its trailing 1-year distribution. A z-score above **+1.5** (real yields sharply above recent average) "
        "historically correlates with HY spread widening of 50–150 bps over the following 3–6 months. "
        "The z-score strips out the level effect — useful for detecting *rate shocks* even when the absolute "
        "level of real yields is ambiguous (e.g. -0.5% rising to +0.3% is a large shock regardless of level)."
    )
    try:
        _rz106_col = "real_yield_z"
        _rp106_col = "real_yield_proxy"
        if _rz106_col in df.columns:
            _rz106 = df[[_rz106_col, _rp106_col, "real_yield_change_90d", "hy_spread",
                          "breakeven_10y", "yield_10y"]].copy()
            _rz106.index = pd.to_datetime(_rz106.index)
            _cur_z = float(latest.get(_rz106_col, float("nan")))
            _cur_ry = float(latest.get(_rp106_col, float("nan")))
            _cur_ryc = float(latest.get("real_yield_change_90d", float("nan")))

            def _real_yield_z_regime(z):
                if pd.isna(z):
                    return "Unknown"
                if z > 1.5:
                    return "Sharply Elevated"
                if z > 0.5:
                    return "Elevated"
                if z >= -0.5:
                    return "Neutral"
                if z >= -1.5:
                    return "Suppressed"
                return "Sharply Suppressed"

            _rz_regime = _real_yield_z_regime(_cur_z)

            _za, _zb, _zc, _zd = st.columns(4)
            _za.metric("Real Yield (10y)", f"{_cur_ry:.2f}%" if pd.notna(_cur_ry) else "—",
                       delta=f"{_cur_ryc:+.2f}pp 90d" if pd.notna(_cur_ryc) else None,
                       delta_color="inverse")
            _zb.metric("Real Yield Z-Score", f"{_cur_z:+.2f}" if pd.notna(_cur_z) else "—")
            _zc.metric("Z-Score Regime", _rz_regime)
            _zd.metric("10y Breakeven", f"{latest.get('breakeven_10y', float('nan')):.2f}%"
                       if pd.notna(latest.get("breakeven_10y")) else "—")

            if pd.notna(_cur_z) and _cur_z > 1.5:
                st.warning("Real yield z-score sharply elevated — historically associated with "
                           "HY spread widening of 50–150 bps over the following 3–6 months.")

            # Z-score time series with bands
            _rz_tail = _rz106.tail(756)
            _fig106a = _go106.Figure()
            _fig106a.add_hrect(y0=1.5, y1=4, fillcolor="rgba(239,68,68,0.08)", line_width=0)
            _fig106a.add_hrect(y0=-4, y1=-1.5, fillcolor="rgba(39,174,96,0.08)", line_width=0)
            _fig106a.add_trace(_go106.Scatter(
                x=_rz_tail.index, y=_rz_tail[_rz106_col],
                line=dict(color="#f59e0b", width=2), name="Real Yield Z-Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Z: %{y:+.2f}<extra></extra>",
            ))
            for _zt, _zc_line in [(1.5, "rgba(239,68,68,0.5)"), (0.5, "rgba(239,68,68,0.25)"),
                                   (-0.5, "rgba(39,174,96,0.25)"), (-1.5, "rgba(39,174,96,0.5)")]:
                _fig106a.add_hline(y=_zt, line=dict(color=_zc_line, dash="dot", width=1))
            _fig106a.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig106a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="10y Real Yield Z-Score (1Y Rolling)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Z-Score", range=[-4, 4]),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig106a, use_container_width=True)
            st.caption("Red band = sharply elevated (>+1.5σ) · Green band = suppressed (<-1.5σ)")

            # Real yield level + HY spread dual-axis
            if "hy_spread" in _rz106.columns:
                _fig106b = _go106.Figure()
                _fig106b.add_trace(_go106.Scatter(
                    x=_rz_tail.index, y=_rz_tail[_rp106_col],
                    name="10y Real Yield (%)", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Real Yield: %{y:.2f}%<extra></extra>",
                ))
                _fig106b.add_trace(_go106.Scatter(
                    x=_rz_tail.index, y=_rz_tail["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig106b.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
                _fig106b.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="Real Yield Level vs HY Spread", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#4f8ef7",
                               title="Real Yield (%)"),
                    yaxis2=dict(overlaying="y", side="right", color="#f59e0b",
                                title="HY Spread (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig106b, use_container_width=True)

            st.markdown("**Real Yield Z-Score Regime Implications**")
            st.table(pd.DataFrame([
                {"Z-Score Range": "> +1.5", "Regime": "Sharply Elevated",
                 "Credit Implication": "Rate shock risk — HY widening of 50–150bps historically follows within 3–6M"},
                {"Z-Score Range": "+0.5 to +1.5", "Regime": "Elevated",
                 "Credit Implication": "Refinancing pressure rising; monitor shorter-duration spreads"},
                {"Z-Score Range": "−0.5 to +0.5", "Regime": "Neutral",
                 "Credit Implication": "Balanced; real yields within normal recent range"},
                {"Z-Score Range": "−1.5 to −0.5", "Regime": "Suppressed",
                 "Credit Implication": "Financial repression supports levered borrowers; risk-on tailwind"},
                {"Z-Score Range": "< −1.5", "Regime": "Sharply Suppressed",
                 "Credit Implication": "ZIRP/QE regime; credit spreads structurally compressed"},
            ]))
        else:
            st.info("real_yield_z column not found — run the full feature pipeline including treasury_engine.")
    except Exception as _e106:
        _err_track(_active_sub, _e106)
        st.caption(f"Real yield Z-score: {_e106}")


if _active_sub == 109:
    import plotly.graph_objects as _go109
    st.header("Unemployment Momentum Monitor")
    st.markdown(
        "**Unemployment momentum** — the 90-day rate-of-change in the unemployment rate — is a mid-cycle "
        "recession indicator distinct from the Sahm Rule threshold trigger. While the Sahm Rule fires at a "
        "0.5pp rise from the 12-month low, unemployment momentum catches the *acceleration phase* "
        "earlier: a sustained 90-day rise of +0.3pp often precedes the Sahm threshold by 4–8 weeks. "
        "Combined with HY spread momentum, this produces an early-warning composite for credit deterioration."
    )
    try:
        _uc109_col = "unemployment_change_90d"
        if _uc109_col in df.columns and "unemployment" in df.columns:
            _uc109 = df[[_uc109_col, "unemployment", "sahm_like", "hy_spread",
                          "hy_change_90d", "labor_warning"]].copy()
            _uc109.index = pd.to_datetime(_uc109.index)
            _cur_uc = float(latest.get(_uc109_col, float("nan")))
            _cur_ur = float(latest.get("unemployment", float("nan")))
            _cur_sahm = float(latest.get("sahm_like", float("nan")))
            _cur_lw = str(latest.get("labor_warning", "—"))

            def _unemp_momentum_regime(delta90):
                if pd.isna(delta90):
                    return "Unknown"
                if delta90 >= 0.5:
                    return "Rapid Deterioration"
                if delta90 >= 0.2:
                    return "Deteriorating"
                if delta90 >= -0.1:
                    return "Stable"
                return "Improving"

            _ur_regime = _unemp_momentum_regime(_cur_uc)
            _ur_colors = {"Rapid Deterioration": "#ef4444", "Deteriorating": "#f59e0b",
                          "Stable": "#4f8ef7", "Improving": "#27ae60", "Unknown": "#6b7280"}

            _ua, _ub, _uc109a, _ud = st.columns(4)
            _ua.metric("Unemployment Rate", f"{_cur_ur:.1f}%" if pd.notna(_cur_ur) else "—")
            _ub.metric("90d Change", f"{_cur_uc:+.2f}pp" if pd.notna(_cur_uc) else "—",
                       delta_color="inverse")
            _uc109a.metric("Momentum Regime", _ur_regime)
            _ud.metric("Sahm-Like Signal", f"{_cur_sahm:.2f}pp" if pd.notna(_cur_sahm) else "—",
                       help="Unemployment minus 12M rolling min — triggers at 0.5pp")

            if pd.notna(_cur_uc) and _cur_uc >= 0.5:
                st.error("Unemployment rising rapidly (+0.5pp/quarter) — recession signal approaching. "
                         "Historical pattern: HY spreads widen 100–300 bps within 6 months.")
            elif pd.notna(_cur_uc) and _cur_uc >= 0.2:
                st.warning("Unemployment momentum deteriorating — early-warning phase. "
                           "Monitor for Sahm Rule trigger and HY spread widening acceleration.")

            # 90d change bar chart
            _uc_tail = _uc109.tail(756)
            _uc_bar_colors = [
                "#ef4444" if v >= 0.5 else "#f59e0b" if v >= 0.2 else
                "#4f8ef7" if v >= -0.1 else "#27ae60"
                for v in _uc_tail[_uc109_col].fillna(0)
            ]
            _fig109a = _go109.Figure()
            _fig109a.add_trace(_go109.Bar(
                x=_uc_tail.index, y=_uc_tail[_uc109_col],
                marker_color=_uc_bar_colors, name="Unemp Δ90d",
                hovertemplate="%{x|%Y-%m-%d}<br>Δ90d: %{y:+.2f}pp<extra></extra>",
            ))
            _fig109a.add_hline(y=0.2, line=dict(color="rgba(245,158,11,0.5)", dash="dot", width=1))
            _fig109a.add_hline(y=0.5, line=dict(color="rgba(239,68,68,0.5)", dash="dot", width=1))
            _fig109a.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig109a.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Unemployment 90-Day Change (pp)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Δ pp (90d)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig109a, use_container_width=True)
            st.caption("Orange threshold: +0.2pp/qtr (Deteriorating) · Red threshold: +0.5pp/qtr (Rapid)")

            # Unemployment level + HY spread dual-axis
            if "hy_spread" in _uc109.columns:
                _fig109b = _go109.Figure()
                _fig109b.add_trace(_go109.Scatter(
                    x=_uc_tail.index, y=_uc_tail["unemployment"],
                    name="Unemployment (%)", line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Unemp: %{y:.1f}%<extra></extra>",
                ))
                _fig109b.add_trace(_go109.Scatter(
                    x=_uc_tail.index, y=_uc_tail["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig109b.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="Unemployment Rate vs HY Spread (3Y)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#f59e0b",
                               title="Unemployment (%)"),
                    yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                title="HY Spread (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig109b, use_container_width=True)

            # Scatter: unemployment 90d change vs HY 90d change
            if "hy_change_90d" in _uc109.columns:
                _sc109 = _uc109.dropna(subset=[_uc109_col, "hy_change_90d"]).tail(756)
                _sc109_colors = [
                    "#ef4444" if u >= 0.5 else "#f59e0b" if u >= 0.2 else
                    "#4f8ef7" if u >= -0.1 else "#27ae60"
                    for u in _sc109[_uc109_col]
                ]
                _fig109c = _go109.Figure()
                _fig109c.add_trace(_go109.Scatter(
                    x=_sc109[_uc109_col], y=_sc109["hy_change_90d"],
                    mode="markers",
                    marker=dict(color=_sc109_colors, size=4, opacity=0.5),
                    hovertemplate="Unemp Δ90d: %{x:+.2f}pp<br>HY Δ90d: %{y:+.0f}bps<extra></extra>",
                ))
                _fig109c.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _fig109c.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _fig109c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Unemployment Momentum vs HY Spread Momentum (90d scatter)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Unemployment Δ90d (pp)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread Δ90d (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig109c, use_container_width=True)
                st.caption("Top-right quadrant: rising unemployment + widening credit — confirmed deterioration. "
                           "Bottom-left: improving labor + tightening spreads — confirmed recovery.")
        else:
            st.info("unemployment_change_90d column not found — run the feature pipeline.")
    except Exception as _e109:
        _err_track(_active_sub, _e109)
        st.caption(f"Unemployment momentum: {_e109}")

# =============================================================================
# BATCH 14 ANALYTICS — sub110–115
# sub110  Credit Regime Monitor            → tab_credit
# sub111  NFCI Trend Change               → tab_macro
# sub112  VIX Momentum Deep Dive          → tab_risk
# sub113  FX/Commodity Sub-Score          → tab_risk
# sub114  Labor Warning Episodes          → tab_macro
# sub115  Complacency Sub-Score           → tab_siglab
# =============================================================================


if _active_sub == 111:
    import plotly.graph_objects as _go111
    st.header("NFCI Trend Change Monitor")
    st.markdown(
        "The **NFCI trend change** (90-day change in the 90-day NFCI moving average) measures how fast "
        "financial conditions are tightening or easing relative to recent trend — not just the level. "
        "NFCI level tells you *where* conditions are; trend change tells you *which direction they're moving*. "
        "A rapid tightening in financial conditions (+0.2 in 90 days) has led HY spread widening by "
        "3–6 weeks historically. Trend easing below -0.15 is a tailwind for credit compression."
    )
    try:
        _nfci111_col = "nfci_change_90d"
        if _nfci111_col in df.columns and "nfci" in df.columns:
            _nfci111 = df[[_nfci111_col, "nfci", "nfci_90d_avg", "hy_spread"]].copy()
            _nfci111.index = pd.to_datetime(_nfci111.index)
            _cur_nc = float(latest.get(_nfci111_col, float("nan")))
            _cur_nfci = float(latest.get("nfci", float("nan")))

            def _nfci_trend_regime(delta):
                if pd.isna(delta):
                    return "Unknown"
                if delta > 0.2:
                    return "Rapid Tightening"
                if delta > 0.05:
                    return "Tightening"
                if delta >= -0.05:
                    return "Stable"
                if delta >= -0.15:
                    return "Easing"
                return "Rapid Easing"

            _nfci_tr = _nfci_trend_regime(_cur_nc)
            _nfci_tr_colors = {
                "Rapid Tightening": "#ef4444", "Tightening": "#f59e0b",
                "Stable": "#4f8ef7", "Easing": "#27ae60", "Rapid Easing": "#10b981",
            }

            _na111, _nb111, _nc111, _nd111 = st.columns(4)
            _na111.metric("NFCI (Current)", f"{_cur_nfci:.3f}" if pd.notna(_cur_nfci) else "—",
                          help="0 = historical average. Positive = tighter than average.")
            _nb111.metric("NFCI Trend Δ90d", f"{_cur_nc:+.3f}" if pd.notna(_cur_nc) else "—",
                          delta_color="inverse")
            _nc111.metric("Trend Regime", _nfci_tr)
            _nfci_1y_chg = float(df["nfci_90d_avg"].diff(252).iloc[-1]) if "nfci_90d_avg" in df.columns else float("nan")
            _nd111.metric("NFCI Trend Δ1Y", f"{_nfci_1y_chg:+.3f}" if pd.notna(_nfci_1y_chg) else "—",
                          delta_color="inverse")

            if pd.notna(_cur_nc) and _cur_nc > 0.2:
                st.error("Financial conditions tightening rapidly — lead signal for HY spread widening in 3–6 weeks.")
            elif pd.notna(_cur_nc) and _cur_nc > 0.05:
                st.warning("Financial conditions tightening — monitor HY spread momentum for follow-through.")

            # NFCI 90d avg + trend change dual chart
            _nfci_tail = _nfci111.tail(756)
            _fig111a = _go111.Figure()
            _fig111a.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig111a.add_trace(_go111.Scatter(
                x=_nfci_tail.index, y=_nfci_tail["nfci_90d_avg"],
                line=dict(color="#4f8ef7", width=2), name="NFCI 90d Avg",
                hovertemplate="%{x|%Y-%m-%d}<br>NFCI Avg: %{y:.3f}<extra></extra>",
            ))
            _fig111a.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="NFCI 90-Day Average (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig111a, use_container_width=True)

            # Trend change bar chart
            _trend_colors111 = [
                "#ef4444" if v > 0.2 else "#f59e0b" if v > 0.05 else
                "#4f8ef7" if v >= -0.05 else "#27ae60" if v >= -0.15 else "#10b981"
                for v in _nfci_tail[_nfci111_col].fillna(0)
            ]
            _fig111b = _go111.Figure()
            _fig111b.add_trace(_go111.Bar(
                x=_nfci_tail.index, y=_nfci_tail[_nfci111_col],
                marker_color=_trend_colors111, name="NFCI Trend Δ90d",
                hovertemplate="%{x|%Y-%m-%d}<br>Δ90d: %{y:+.3f}<extra></extra>",
            ))
            _fig111b.add_hline(y=0.2, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig111b.add_hline(y=-0.15, line=dict(color="rgba(39,174,96,0.4)", dash="dot", width=1))
            _fig111b.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig111b.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="NFCI 90-Day Trend Change", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Δ NFCI (90d)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig111b, use_container_width=True)
            st.caption("Red = rapid tightening (>+0.2) · Green = rapid easing (<-0.15)")

            # NFCI trend vs HY spread dual-axis
            if "hy_spread" in _nfci111.columns:
                _fig111c = _go111.Figure()
                _fig111c.add_trace(_go111.Scatter(
                    x=_nfci_tail.index, y=_nfci_tail[_nfci111_col],
                    name="NFCI Trend Δ90d", line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>NFCI Δ: %{y:+.3f}<extra></extra>",
                ))
                _fig111c.add_trace(_go111.Scatter(
                    x=_nfci_tail.index, y=_nfci_tail["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig111c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="NFCI Trend Change vs HY Spread", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#f59e0b",
                               title="NFCI Δ90d"),
                    yaxis2=dict(overlaying="y", side="right", color="#ef4444", title="HY (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig111c, use_container_width=True)
        else:
            st.info("nfci_change_90d column not found — run the feature pipeline.")
    except Exception as _e111:
        _err_track(_active_sub, _e111)
        st.caption(f"NFCI trend: {_e111}")


if _active_sub == 114:
    import plotly.graph_objects as _go114
    st.header("Labor Warning Episodes")
    st.markdown(
        "The **labor warning signal** has three states: **Low** (no concern), **Watch** "
        "(Sahm-like ≥ 0.3pp or unemployment rising ≥ 0.2pp/quarter — precautionary monitoring), "
        "and **Triggered** (Sahm-like ≥ 0.5pp — historical recession onset indicator). "
        "This tab analyses *episodes* rather than levels: how long do Warning/Triggered periods last, "
        "what HY spread moves accompanied them, and what was the subsequent 3-month credit outcome."
    )
    try:
        _lw114_col = "labor_warning"
        if _lw114_col in df.columns:
            _lw114 = df[[_lw114_col, "sahm_like", "unemployment_change_90d",
                          "unemployment", "hy_spread", "hy_change_90d"]].copy()
            _lw114.index = pd.to_datetime(_lw114.index)
            _cur_lw = str(latest.get(_lw114_col, "Low"))
            _cur_sahm = float(latest.get("sahm_like", float("nan")))

            _lw_colors = {"Triggered": "#ef4444", "Watch": "#f59e0b", "Low": "#27ae60"}
            _lw_num = {"Triggered": 3, "Watch": 2, "Low": 1}

            _la114, _lb114, _lc114, _ld114 = st.columns(4)
            _la114.metric("Warning Signal", _cur_lw)
            _lb114.metric("Sahm-Like (pp)", f"{_cur_sahm:.2f}" if pd.notna(_cur_sahm) else "—",
                          help="≥0.3 = Watch, ≥0.5 = Triggered")
            _watch_freq = ((_lw114[_lw114_col] != "Low").rolling(252).mean().iloc[-1]) * 100
            _lc114.metric("Non-Low Freq (1Y)", f"{_watch_freq:.0f}%" if pd.notna(_watch_freq) else "—")
            _triggered_total = (_lw114[_lw114_col] == "Triggered").sum()
            _ld114.metric("Triggered Days (History)", str(_triggered_total))

            if _cur_lw == "Triggered":
                st.error("Labor warning TRIGGERED — Sahm-like indicator at or above 0.5pp. "
                         "Recession onset signal active. Historically: HY spreads widen 150–400bps over subsequent 6M.")
            elif _cur_lw == "Watch":
                st.warning("Labor warning in Watch state — labor market showing early deterioration signs. "
                           "Monitor for Sahm trigger and NFCI tightening confirmation.")

            # Warning signal timeline
            _lw_tail = _lw114.tail(756)
            _lw_tail = _lw_tail.copy()
            _lw_tail["signal_num"] = _lw_tail[_lw114_col].map(_lw_num).fillna(1)
            _bar_colors114 = [_lw_colors.get(s, "#6b7280") for s in _lw_tail[_lw114_col]]
            _fig114a = _go114.Figure()
            _fig114a.add_trace(_go114.Bar(
                x=_lw_tail.index, y=_lw_tail["signal_num"],
                marker_color=_bar_colors114, name="Labor Warning",
                hovertemplate="%{x|%Y-%m-%d}<br>%{customdata}<extra></extra>",
                customdata=_lw_tail[_lw114_col],
            ))
            _fig114a.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Labor Warning Signal (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(tickvals=[1, 2, 3], ticktext=["Low", "Watch", "Triggered"],
                           showgrid=False, color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig114a, use_container_width=True)

            # Sahm-like with threshold bands
            _fig114b = _go114.Figure()
            _fig114b.add_hrect(y0=0.5, y1=3.0, fillcolor="rgba(239,68,68,0.08)", line_width=0)
            _fig114b.add_hrect(y0=0.3, y1=0.5, fillcolor="rgba(245,158,11,0.07)", line_width=0)
            _fig114b.add_trace(_go114.Scatter(
                x=_lw_tail.index, y=_lw_tail["sahm_like"],
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                line=dict(color="#f59e0b", width=2), name="Sahm-Like",
                hovertemplate="%{x|%Y-%m-%d}<br>Sahm: %{y:.2f}pp<extra></extra>",
            ))
            _fig114b.add_hline(y=0.3, line=dict(color="rgba(245,158,11,0.5)", dash="dot", width=1))
            _fig114b.add_hline(y=0.5, line=dict(color="rgba(239,68,68,0.5)", dash="dot", width=1))
            _fig114b.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Sahm-Like Indicator with Warning Thresholds", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="pp above 12M min"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig114b, use_container_width=True)
            st.caption("Orange band = Watch (0.3–0.5pp) · Red band = Triggered (>0.5pp)")

            # HY spread in each warning state
            if "hy_spread" in _lw114.columns:
                _fig114c = _go114.Figure()
                for _state, _sc in _lw_colors.items():
                    _mask = _lw114[_lw114_col] == _state
                    _hy_v = _lw114.loc[_mask, "hy_spread"].dropna()
                    if len(_hy_v) > 5:
                        _fig114c.add_trace(_go114.Box(
                            y=_hy_v, name=_state, marker_color=_sc,
                            line_color=_sc, boxmean=True,
                            hovertemplate=f"{_state}<br>HY: %{{y:.0f}}bps<extra></extra>",
                        ))
                _fig114c.update_layout(
                    height=230, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="HY Spread Distribution by Labor Warning State", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread (bps)"),
                    xaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig114c, use_container_width=True)
        else:
            st.info("labor_warning column not found — run the feature pipeline.")
    except Exception as _e114:
        _err_track(_active_sub, _e114)
        st.caption(f"Labor warning: {_e114}")


if _active_sub == 118:
    import plotly.graph_objects as _go118
    from src.regime_attribution import COMPOSITE_WEIGHTS as _CW118
    st.header("Treasury Stress Sub-Score")
    st.markdown(
        "The **treasury stress sub-score** (20% weight — equal to the largest) captures rate-driven credit "
        "risk: how hard the risk-free rate is pressing on credit valuations. Its three inputs are the "
        "**real yield z-score** (are real rates sharply above their own recent average?), the "
        "**90d change in real yields** (velocity of rate shock), and the **curve steepening velocity** "
        "(bear steepening = long rates rising fast = duration compression). It is the strongest *leading* "
        "sub-score in the composite, with historical lead times of 6–12 weeks ahead of spread widening."
    )
    try:
        _ts118_col = "treasury_stress_score_smooth"
        _ts118_raw = "treasury_stress_score"
        _ts_col = _ts118_col if _ts118_col in df.columns else (_ts118_raw if _ts118_raw in df.columns else None)
        if _ts_col:
            _ts118 = df[[_ts_col]].copy()
            for _c in ["real_yield_proxy", "real_yield_z", "curve_steepening_velocity_90d",
                        "composite_risk_score_smooth", "hy_spread"]:
                if _c in df.columns:
                    _ts118[_c] = df[_c]
            _ts118.index = pd.to_datetime(_ts118.index)
            _cur_ts = float(latest.get(_ts_col, float("nan")))
            _ts_pctile = (df[_ts_col].dropna() < _cur_ts).mean() * 100 if pd.notna(_cur_ts) else float("nan")
            _ts_vel = float(df[_ts_col].diff(21).iloc[-1]) if df[_ts_col].notna().any() else float("nan")
            _ts_contrib = _cur_ts * _CW118.get("treasury", 0.20) if pd.notna(_cur_ts) else float("nan")

            _ta118, _tb118, _tc118, _td118 = st.columns(4)
            _ta118.metric("Treasury Stress Score", f"{_cur_ts:.0f}/100" if pd.notna(_cur_ts) else "—")
            _tb118.metric("Composite Contrib", f"{_ts_contrib:.1f}pts" if pd.notna(_ts_contrib) else "—",
                          help="Score × 20% weight — strongest leading signal")
            _tc118.metric("Historical Pctile", f"{_ts_pctile:.0f}th" if pd.notna(_ts_pctile) else "—")
            _td118.metric("21d Velocity", f"{_ts_vel:+.1f}pts" if pd.notna(_ts_vel) else "—",
                          delta_color="inverse")

            if pd.notna(_cur_ts) and _cur_ts >= 55:
                st.warning("Treasury stress score elevated — real yields sharply above recent average "
                           "and/or curve velocity accelerating. Historically leads HY spread widening by 6–12 weeks.")

            _ts_tail = _ts118.tail(756)
            _fig118a = _go118.Figure()
            _fig118a.add_hrect(y0=55, y1=105, fillcolor="rgba(239,68,68,0.07)", line_width=0)
            _fig118a.add_trace(_go118.Scatter(
                x=_ts_tail.index, y=_ts_tail[_ts_col],
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
                line=dict(color="#f59e0b", width=2), name="Treasury Stress Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            if "composite_risk_score_smooth" in _ts_tail.columns:
                _fig118a.add_trace(_go118.Scatter(
                    x=_ts_tail.index, y=_ts_tail["composite_risk_score_smooth"],
                    line=dict(color="#e2e8f0", width=1, dash="dot"), name="Composite",
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
            _fig118a.add_hline(y=55, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig118a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Treasury Stress Sub-Score vs Composite (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig118a, use_container_width=True)

            # Real yield z + curve velocity dual-axis
            if "real_yield_z" in _ts_tail.columns and "curve_steepening_velocity_90d" in _ts_tail.columns:
                _fig118b = _go118.Figure()
                _fig118b.add_trace(_go118.Scatter(
                    x=_ts_tail.index, y=_ts_tail["real_yield_z"],
                    name="Real Yield Z-Score", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Z: %{y:+.2f}<extra></extra>",
                ))
                _fig118b.add_trace(_go118.Scatter(
                    x=_ts_tail.index, y=_ts_tail["curve_steepening_velocity_90d"] * 100,
                    name="Curve Velocity (×100)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>Velocity: %{y:+.0f}bps<extra></extra>",
                ))
                _fig118b.add_hline(y=1.5, line=dict(color="rgba(239,68,68,0.35)", dash="dot", width=1))
                _fig118b.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig118b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="Real Yield Z-Score and Curve Velocity", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#4f8ef7",
                               title="Real Yield Z"),
                    yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                title="Curve Velocity (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig118b, use_container_width=True)

            st.markdown("**Treasury Stress Score — Driver Breakdown**")
            st.table(pd.DataFrame([
                {"Input": "Real Yield Z-Score", "Pts": "Up to 35",
                 "Trigger": ">+1.5σ → 35pts; >+1.0σ → 20pts"},
                {"Input": "Real Yield Δ90d", "Pts": "Up to 30",
                 "Trigger": ">+0.5pp/qtr → 30pts; >+0.25pp → 15pts"},
                {"Input": "Curve Steepening Velocity", "Pts": "Up to 20",
                 "Trigger": "Bear steepening >50bps/qtr → 20pts"},
            ]))
        else:
            st.info("Treasury stress score not found — run the full scoring pipeline.")
    except Exception as _e118:
        _err_track(_active_sub, _e118)
        st.caption(f"Treasury stress score: {_e118}")


if _active_sub == 121:
    import plotly.graph_objects as _go121
    st.header("Yield Curve Spread Changes — Multi-Horizon")
    st.markdown(
        "Rates of change in the **2s10s yield spread** across 5-day and 90-day windows. "
        "The 90-day change captures the slow-moving structural steepening or flattening; "
        "the 5-day change catches acute near-term disruptions (Fed meetings, macro data). "
        "When both are negative (5d and 90d flattening simultaneously), the curve is in "
        "sustained compression — the most historically credit-negative configuration. "
        "When the 5d reverses positive while 90d remains negative, it may signal "
        "a temporary bear-steepening counter-move inside a longer flattening trend."
    )
    try:
        _sc121_cols = ["spread_change_90d", "spread_change_5d", "spread", "yield_10y", "yield_2y"]
        if all(c in df.columns for c in ["spread_change_90d", "spread_change_5d", "spread"]):
            _sc121 = df[[c for c in _sc121_cols if c in df.columns]].copy()
            _sc121.index = pd.to_datetime(_sc121.index)
            _cur_sc90 = float(latest.get("spread_change_90d", float("nan")))
            _cur_sc5 = float(latest.get("spread_change_5d", float("nan")))
            _cur_sp = float(latest.get("spread", float("nan")))

            def _spread_chg_signal(sc5, sc90):
                if pd.isna(sc5) or pd.isna(sc90):
                    return "Unknown"
                if sc5 < 0 and sc90 < 0:
                    return "Sustained Flattening"
                if sc5 > 0 and sc90 > 0:
                    return "Sustained Steepening"
                if sc5 > 0 and sc90 < 0:
                    return "Bear Steepen / Flattening Pause"
                return "Short Flatten / Steepen Pullback"

            _chg_signal = _spread_chg_signal(_cur_sc5, _cur_sc90)
            _chg_colors = {
                "Sustained Flattening": "#ef4444",
                "Sustained Steepening": "#27ae60",
                "Bear Steepen / Flattening Pause": "#f59e0b",
                "Short Flatten / Steepen Pullback": "#4f8ef7",
                "Unknown": "#6b7280",
            }

            _sca, _scb, _scc, _scd = st.columns(4)
            _sca.metric("2s10s Spread", f"{_cur_sp:.2f}pp" if pd.notna(_cur_sp) else "—")
            _scb.metric("Δ5d", f"{_cur_sc5:+.2f}pp" if pd.notna(_cur_sc5) else "—",
                        delta_color="normal" if pd.notna(_cur_sc5) and _cur_sc5 > 0 else "inverse")
            _scc.metric("Δ90d", f"{_cur_sc90:+.2f}pp" if pd.notna(_cur_sc90) else "—",
                        delta_color="normal" if pd.notna(_cur_sc90) and _cur_sc90 > 0 else "inverse")
            _scd.metric("Signal", _chg_signal)

            if _chg_signal == "Sustained Flattening":
                st.warning("Both 5d and 90d curve spread changes are negative — sustained flattening/inversion "
                           "pressure. Historically the most negative credit configuration from the curve.")

            # 90d spread change with regime coloring
            _sc_tail = _sc121.tail(756)
            _c90_colors = ["#ef4444" if v < -0.05 else "#27ae60" if v > 0.05 else "#4f8ef7"
                           for v in _sc_tail["spread_change_90d"].fillna(0)]
            _fig121a = _go121.Figure()
            _fig121a.add_trace(_go121.Bar(
                x=_sc_tail.index, y=_sc_tail["spread_change_90d"],
                marker_color=_c90_colors, name="Δ90d",
                hovertemplate="%{x|%Y-%m-%d}<br>Δ90d: %{y:+.2f}pp<extra></extra>",
            ))
            _fig121a.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig121a.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="2s10s Spread 90-Day Change (pp)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig121a, use_container_width=True)

            # 5d change for short-horizon moves
            _c5_colors = ["#ef4444" if v < 0 else "#27ae60"
                          for v in _sc_tail["spread_change_5d"].fillna(0)]
            _fig121b = _go121.Figure()
            _fig121b.add_trace(_go121.Bar(
                x=_sc_tail.index, y=_sc_tail["spread_change_5d"],
                marker_color=_c5_colors, name="Δ5d", opacity=0.7,
                hovertemplate="%{x|%Y-%m-%d}<br>Δ5d: %{y:+.3f}pp<extra></extra>",
            ))
            _fig121b.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig121b.update_layout(
                height=180, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="2s10s Spread 5-Day Change (pp) — short-horizon moves", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig121b, use_container_width=True)

            # Term structure of changes: current snapshot
            _ts_vals121 = [_cur_sc5, _cur_sc90]
            _ts_labs121 = ["5d", "90d"]
            _ts_colors121 = ["#ef4444" if pd.notna(v) and v < 0 else "#27ae60" if pd.notna(v) else "#6b7280"
                             for v in _ts_vals121]
            _fig121c = _go121.Figure()
            _fig121c.add_trace(_go121.Bar(
                x=_ts_labs121, y=_ts_vals121,
                marker_color=_ts_colors121,
                text=[f"{v:+.3f}pp" if pd.notna(v) else "n/a" for v in _ts_vals121],
                textposition="outside",
                hovertemplate="%{x}: %{y:+.3f}pp<extra></extra>",
            ))
            _fig121c.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
            _fig121c.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Current Curve Change Term Structure", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Δ spread (pp)"),
                xaxis=dict(color="#6b7280", title="Horizon"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig121c, use_container_width=True)
            st.caption("Both negative = Sustained Flattening (most credit-negative) · "
                       "Both positive = Sustained Steepening · Mixed = transition / counter-move")
        else:
            st.info("Spread change columns not found — run the feature pipeline.")
    except Exception as _e121:
        _err_track(_active_sub, _e121)
        st.caption(f"Spread changes: {_e121}")

# =============================================================================
# BATCH 16 ANALYTICS — sub122–127
# sub122  Breakeven Inflation Monitor      → tab_macro
# sub123  Score Consensus Monitor          → tab_siglab
# sub124  Equity Return Context            → tab_risk
# sub125  10y Yield Level Context          → tab_macro
# sub126  Composite Score History          → tab_siglab
# sub127  VIX Level Context               → tab_risk
# =============================================================================


if _active_sub == 122:
    import plotly.graph_objects as _go122
    st.header("Breakeven Inflation Monitor")
    st.markdown(
        "The **10-year breakeven inflation rate** (TIPS-implied) is the market's expectation of average "
        "CPI over the next decade. Rising breakevens signal **reflation risk**: the Fed must stay "
        "restrictive longer, keeping real yields high and pressuring credit spreads. Falling breakevens "
        "signal **deflation/recession risk**: demand destruction arriving, which typically leads "
        "HY spread widening as default expectations rise. The 'sweet spot' for credit is moderate, "
        "stable breakevens (2–2.5%) — neither deflationary collapse nor inflationary overshooting."
    )
    try:
        _be122_col = "breakeven_10y"
        if _be122_col in df.columns and df[_be122_col].notna().any():
            _be122 = df[[_be122_col]].copy()
            for _c in ["real_yield_proxy", "yield_10y", "hy_spread"]:
                if _c in df.columns:
                    _be122[_c] = df[_c]
            _be122.index = pd.to_datetime(_be122.index)
            _cur_be = float(latest.get(_be122_col, float("nan")))
            _be_1m = float(df[_be122_col].diff(21).iloc[-1]) if df[_be122_col].notna().any() else float("nan")
            _be_3m = float(df[_be122_col].diff(63).iloc[-1]) if df[_be122_col].notna().any() else float("nan")
            _be_pctile = (df[_be122_col].dropna() < _cur_be).mean() * 100 if pd.notna(_cur_be) else float("nan")

            def _be_regime(be):
                if pd.isna(be):
                    return "Unknown"
                if be > 2.75:
                    return "Elevated (Reflationary)"
                if be >= 2.0:
                    return "Moderate (Sweet Spot)"
                if be >= 1.5:
                    return "Low (Disinflation)"
                return "Very Low (Deflation Risk)"

            _be_reg = _be_regime(_cur_be)
            _be_reg_colors = {
                "Elevated (Reflationary)": "#ef4444",
                "Moderate (Sweet Spot)": "#27ae60",
                "Low (Disinflation)": "#f59e0b",
                "Very Low (Deflation Risk)": "#a78bfa",
            }

            _ba, _bb, _bc, _bd = st.columns(4)
            _ba.metric("10y Breakeven", f"{_cur_be:.2f}%" if pd.notna(_cur_be) else "—",
                       delta=f"{_be_1m:+.2f}pp 1M" if pd.notna(_be_1m) else None)
            _bb.metric("3M Change", f"{_be_3m:+.2f}pp" if pd.notna(_be_3m) else "—",
                       delta_color="inverse" if pd.notna(_be_3m) and _be_3m > 0.25 else "normal")
            _bc.metric("Regime", _be_reg)
            _bd.metric("Historical Pctile", f"{_be_pctile:.0f}th" if pd.notna(_be_pctile) else "—")

            if pd.notna(_cur_be):
                if _cur_be > 2.75:
                    st.warning("Breakeven inflation elevated — reflationary pressures keeping Fed hawkish. "
                               "Real yields likely to remain high; credit spread pressure building.")
                elif _cur_be < 1.5:
                    st.warning("Breakeven inflation very low — deflation/recession risk priced in. "
                               "Credit default expectations rising; HY spread widening typically follows.")

            # Breakeven time series with sweet-spot band
            _be_tail = _be122.tail(756)
            _fig122a = _go122.Figure()
            _fig122a.add_hrect(y0=2.0, y1=2.75, fillcolor="rgba(39,174,96,0.08)", line_width=0)
            _fig122a.add_trace(_go122.Scatter(
                x=_be_tail.index, y=_be_tail[_be122_col],
                line=dict(color="#f59e0b", width=2.5), name="10y Breakeven",
                hovertemplate="%{x|%Y-%m-%d}<br>Breakeven: %{y:.2f}%<extra></extra>",
            ))
            _fig122a.add_hline(y=2.0, line=dict(color="rgba(39,174,96,0.4)", dash="dot", width=1))
            _fig122a.add_hline(y=2.75, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig122a.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="10Y Breakeven Inflation (TIPS-implied, 3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="%"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig122a, use_container_width=True)
            st.caption("Green band = sweet spot for credit (2.0–2.75%) · Above = reflationary risk · Below = deflation risk")

            # Nominal = Real + Breakeven decomposition
            if "real_yield_proxy" in _be_tail.columns and "yield_10y" in _be_tail.columns:
                _fig122b = _go122.Figure()
                _fig122b.add_trace(_go122.Scatter(
                    x=_be_tail.index, y=_be_tail["yield_10y"],
                    name="10y Nominal Yield", line=dict(color="#e2e8f0", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Nominal: %{y:.2f}%<extra></extra>",
                ))
                _fig122b.add_trace(_go122.Scatter(
                    x=_be_tail.index, y=_be_tail["real_yield_proxy"],
                    name="10y Real Yield", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Real: %{y:.2f}%<extra></extra>",
                ))
                _fig122b.add_trace(_go122.Scatter(
                    x=_be_tail.index, y=_be_tail[_be122_col],
                    name="Breakeven (Inflation Exp)", line=dict(color="#f59e0b", width=2, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>BE: %{y:.2f}%<extra></extra>",
                ))
                _fig122b.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig122b.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Nominal = Real Yield + Breakeven Decomposition", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="%"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig122b, use_container_width=True)

            # Breakeven vs HY spread
            if "hy_spread" in _be_tail.columns:
                _be_sc = _be_tail.dropna(subset=[_be122_col, "hy_spread"])
                _fig122c = _go122.Figure()
                _fig122c.add_trace(_go122.Scatter(
                    x=_be_sc[_be122_col], y=_be_sc["hy_spread"],
                    mode="markers",
                    marker=dict(color="#f59e0b", size=3, opacity=0.4),
                    hovertemplate="Breakeven: %{x:.2f}%<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig122c.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Breakeven Inflation vs HY Spread (3Y scatter)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="10Y Breakeven (%)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig122c, use_container_width=True)
                st.caption("U-shaped relationship expected: very low OR very high breakevens both widen HY spreads. "
                           "Moderate 2–2.5% breakevens correlate with tightest credit spreads.")
        else:
            st.info("breakeven_10y column not found — requires T10YIE FRED series.")
    except Exception as _e122:
        _err_track(_active_sub, _e122)
        st.caption(f"Breakeven monitor: {_e122}")


if _active_sub == 125:
    import plotly.graph_objects as _go125
    st.header("10-Year Yield Level Context")
    st.markdown(
        "The **10-year Treasury yield** is the risk-free rate anchor for all credit pricing. "
        "When the 10y yield rises sharply (bear market in rates), investment-grade credit "
        "suffers most from duration losses; HY spreads may initially be stable but widen "
        "as refinancing costs rise. When the 10y falls rapidly (flight to safety), "
        "it signals risk-off — spreads typically widen simultaneously despite lower base rates. "
        "This tab tracks the 10y yield level, its rolling percentile, and co-movement with HY spreads."
    )
    try:
        _y10_col = "yield_10y"
        _y2_col = "yield_2y"
        if _y10_col in df.columns and df[_y10_col].notna().any():
            _y10125 = df[[_y10_col]].copy()
            for _c in [_y2_col, "hy_spread", "spread", "breakeven_10y", "real_yield_proxy"]:
                if _c in df.columns:
                    _y10125[_c] = df[_c]
            _y10125.index = pd.to_datetime(_y10125.index)
            _cur_y10 = float(latest.get(_y10_col, float("nan")))
            _cur_y2 = float(latest.get(_y2_col, float("nan"))) if _y2_col in df.columns else float("nan")
            _y10_1m = float(df[_y10_col].diff(21).iloc[-1]) if df[_y10_col].notna().any() else float("nan")
            _y10_3m = float(df[_y10_col].diff(63).iloc[-1]) if df[_y10_col].notna().any() else float("nan")
            _y10_pctile = (df[_y10_col].dropna() < _cur_y10).mean() * 100 if pd.notna(_cur_y10) else float("nan")

            _ya, _yb, _yc, _yd = st.columns(4)
            _ya.metric("10y Yield", f"{_cur_y10:.2f}%" if pd.notna(_cur_y10) else "—",
                       delta=f"{_y10_1m:+.2f}pp 1M" if pd.notna(_y10_1m) else None,
                       delta_color="inverse")
            _yb.metric("2y Yield", f"{_cur_y2:.2f}%" if pd.notna(_cur_y2) else "—")
            _yc.metric("3M Change", f"{_y10_3m:+.2f}pp" if pd.notna(_y10_3m) else "—",
                       delta_color="inverse")
            _yd.metric("Full-History Pctile", f"{_y10_pctile:.0f}th" if pd.notna(_y10_pctile) else "—")

            if pd.notna(_y10_3m) and _y10_3m > 0.75:
                st.warning(f"10y yield rising rapidly (+{_y10_3m:.2f}pp in 3M) — rate shock risk. "
                           "Duration-sensitive IG credit under pressure; HY refinancing costs rising.")
            elif pd.notna(_y10_3m) and _y10_3m < -0.50:
                st.info(f"10y yield falling sharply ({_y10_3m:.2f}pp in 3M) — flight-to-safety signal. "
                        "Typically coincides with or leads HY spread widening.")

            # 10y + 2y yield time series
            _y10_tail = _y10125.tail(756)
            _fig125a = _go125.Figure()
            _fig125a.add_trace(_go125.Scatter(
                x=_y10_tail.index, y=_y10_tail[_y10_col],
                name="10y Yield", line=dict(color="#4f8ef7", width=2.5),
                hovertemplate="%{x|%Y-%m-%d}<br>10y: %{y:.2f}%<extra></extra>",
            ))
            if _y2_col in _y10_tail.columns:
                _fig125a.add_trace(_go125.Scatter(
                    x=_y10_tail.index, y=_y10_tail[_y2_col],
                    name="2y Yield", line=dict(color="#a78bfa", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>2y: %{y:.2f}%<extra></extra>",
                ))
            _fig125a.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="10y and 2y Treasury Yields (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="%"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig125a, use_container_width=True)

            # Rolling percentile of 10y yield
            _y10125["y10_pctile"] = _y10125[_y10_col].expanding().rank(pct=True) * 100
            _fig125b = _go125.Figure()
            _fig125b.add_hrect(y0=80, y1=100, fillcolor="rgba(239,68,68,0.07)", line_width=0)
            _fig125b.add_hrect(y0=0, y1=20, fillcolor="rgba(39,174,96,0.07)", line_width=0)
            _fig125b.add_trace(_go125.Scatter(
                x=_y10_tail.index, y=_y10125["y10_pctile"].loc[_y10_tail.index],
                line=dict(color="#4f8ef7", width=2), name="10y Yield Pctile",
                hovertemplate="%{x|%Y-%m-%d}<br>Pctile: %{y:.0f}th<extra></extra>",
            ))
            _fig125b.add_hline(y=50, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
            _fig125b.update_layout(
                height=180, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="10y Yield Historical Percentile Rank", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Pctile"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig125b, use_container_width=True)

            # 10y yield vs HY spread dual-axis
            if "hy_spread" in _y10_tail.columns:
                _fig125c = _go125.Figure()
                _fig125c.add_trace(_go125.Scatter(
                    x=_y10_tail.index, y=_y10_tail[_y10_col],
                    name="10y Yield (%)", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>10y: %{y:.2f}%<extra></extra>",
                ))
                _fig125c.add_trace(_go125.Scatter(
                    x=_y10_tail.index, y=_y10_tail["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig125c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="10y Treasury Yield vs HY Spread", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#4f8ef7", title="10y Yield (%)"),
                    yaxis2=dict(overlaying="y", side="right", color="#f59e0b", title="HY Spread (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig125c, use_container_width=True)
                st.caption("Both rising = stagflation / rate-shock risk (worst for total return) · "
                           "Yield falling + spreads rising = flight-to-safety episode (risk-off)")
        else:
            st.info("yield_10y column not found — run the feature pipeline.")
    except Exception as _e125:
        _err_track(_active_sub, _e125)
        st.caption(f"10y yield context: {_e125}")


if _active_sub == 138:
    try:
        import plotly.graph_objects as _go138
        import numpy as _np138
        _macro_col = "macro_risk_score_smooth"
        _credit_col = "credit_market_risk_score_smooth"
        if _macro_col in df.columns and _credit_col in df.columns:
            _mac138 = df[_macro_col].dropna()
            _crd138 = df[_credit_col].dropna()
            _joined138 = _mac138.to_frame("macro").join(_crd138.to_frame("credit"), how="inner").dropna()
            # Rolling 63d correlation
            _roll_corr138 = _joined138["macro"].rolling(63, min_periods=21).corr(_joined138["credit"])
            # Spread between scores (macro - credit)
            _spread138 = _joined138["macro"] - _joined138["credit"]
            # Chart 1: both scores over time
            _fig138a = _go138.Figure()
            _fig138a.add_trace(_go138.Scatter(
                x=_joined138.index, y=_joined138["macro"],
                mode="lines", name="Macro Risk",
                line=dict(color="#f59e0b", width=1.2),
                hovertemplate="Macro: %{y:.0f}<extra></extra>",
            ))
            _fig138a.add_trace(_go138.Scatter(
                x=_joined138.index, y=_joined138["credit"],
                mode="lines", name="Credit Risk",
                line=dict(color="#3b82f6", width=1.2),
                hovertemplate="Credit: %{y:.0f}<extra></extra>",
            ))
            _fig138a.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Macro Risk Score vs Credit Market Risk Score", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Score (0–100)", range=[0, 100]),
                legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig138a, use_container_width=True)
            # Chart 2: rolling 63d correlation
            _fig138b = _go138.Figure()
            _fig138b.add_trace(_go138.Scatter(
                x=_roll_corr138.index, y=_roll_corr138.values,
                mode="lines", name="63d Rolling Corr",
                line=dict(color="#10b981", width=1.2),
                fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
                hovertemplate="%{x|%Y-%m-%d}: corr=%{y:.2f}<extra></extra>",
            ))
            _fig138b.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig138b.add_hline(y=0.5, line_color="#6b7280", line_width=1, line_dash="dot")
            _fig138b.update_layout(
                height=170, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Rolling 63d Correlation: Macro vs Credit Risk Scores", font=dict(size=11, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Correlation", range=[-1, 1]),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig138b, use_container_width=True)
            # Chart 3: score spread (macro - credit) — divergence episodes
            _fig138c = _go138.Figure()
            _fig138c.add_trace(_go138.Scatter(
                x=_spread138.index, y=_spread138.values,
                mode="lines", name="Macro − Credit",
                line=dict(color="#8b5cf6", width=1.0),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.07)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.1f} pts<extra></extra>",
            ))
            _fig138c.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig138c.add_hline(y=15, line_color="#f59e0b", line_width=1, line_dash="dot",
                               annotation_text="Macro leading", annotation_font=dict(color="#f59e0b", size=8))
            _fig138c.add_hline(y=-15, line_color="#3b82f6", line_width=1, line_dash="dot",
                               annotation_text="Credit leading", annotation_font=dict(color="#3b82f6", size=8))
            _fig138c.update_layout(
                height=170, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Score Divergence: Macro Risk − Credit Risk (>0 = macro leading)", font=dict(size=11, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Score Gap (pts)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig138c, use_container_width=True)
            _curr_mac = float(_joined138["macro"].iloc[-1])
            _curr_crd = float(_joined138["credit"].iloc[-1])
            _curr_corr = float(_roll_corr138.iloc[-1]) if _roll_corr138.notna().any() else None
            st.caption(
                f"Current — Macro: {_curr_mac:.0f} · Credit: {_curr_crd:.0f} · Gap: {_curr_mac - _curr_crd:+.0f} pts · "
                f"63d Corr: {f'{_curr_corr:.2f}' if _curr_corr is not None else 'N/A'}. "
                "Decoupling (low corr) signals regime transition risk."
            )
        else:
            st.info("macro_risk_score_smooth or credit_market_risk_score_smooth not found — run the full scoring pipeline.")
    except Exception as _e138:
        _err_track(_active_sub, _e138)
        st.caption(f"Macro-credit corr: {_e138}")

# sub139 — Alert History (tab_risk)

if _active_sub == 142:
    try:
        import plotly.graph_objects as _go142
        import numpy as _np142
        if "credit_impulse" in df.columns:
            _ci142 = df["credit_impulse"].dropna()
            _hy142 = df["hy_spread"].dropna() if "hy_spread" in df.columns else None
            # Time series with regime shading
            def _ci_regime(v):
                if _np142.isnan(v): return "Unknown"
                if v < -20: return "Strong Easing"
                if v < -5:  return "Easing"
                if v < 5:   return "Neutral"
                if v < 20:  return "Tightening"
                return "Strong Tightening"
            _ci_reg_colors = {"Strong Easing": "#10b981", "Easing": "#3b82f6",
                               "Neutral": "#6b7280", "Tightening": "#f59e0b",
                               "Strong Tightening": "#ef4444", "Unknown": "#4b5563"}
            _fig142a = _go142.Figure()
            _fig142a.add_trace(_go142.Scatter(
                x=_ci142.index, y=_ci142.values,
                mode="lines", line=dict(color="#8b5cf6", width=1.2),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.1f}bps<extra></extra>",
            ))
            _fig142a.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig142a.add_hline(y=20, line_color="#ef4444", line_width=1, line_dash="dot",
                               annotation_text="Strong Tight", annotation_font=dict(color="#ef4444", size=8))
            _fig142a.add_hline(y=-20, line_color="#10b981", line_width=1, line_dash="dot",
                               annotation_text="Strong Ease", annotation_font=dict(color="#10b981", size=8))
            _fig142a.update_layout(
                height=250, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Credit Impulse (ΔHY 30d − ΔHY 30d prior): Acceleration/Deceleration of Spread Change",
                           font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Credit Impulse (bps)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig142a, use_container_width=True)
            # Current regime
            _curr_ci = float(_ci142.iloc[-1])
            _curr_ci_reg = _ci_regime(_curr_ci)
            st.caption(f"Current: **{_curr_ci:+.1f} bps** — Regime: **{_curr_ci_reg}**. "
                       "Positive = spread widening is accelerating. Negative = widening is decelerating (or tightening).")
            # Lead-lag: impulse vs HY level
            if _hy142 is not None:
                _lags142 = list(range(-21, 22, 3))
                _corrs142 = []
                for lag in _lags142:
                    _ci_shifted = _ci142.shift(-lag)
                    _j = _ci_shifted.to_frame("ci").join(_hy142.to_frame("hy"), how="inner").dropna()
                    _corrs142.append(float(_j["ci"].corr(_j["hy"])) if len(_j) > 20 else _np142.nan)
                _fig142b = _go142.Figure()
                _fig142b.add_trace(_go142.Scatter(
                    x=_lags142, y=_corrs142,
                    mode="lines+markers",
                    line=dict(color="#8b5cf6", width=1.5),
                    marker=dict(size=5, color="#8b5cf6"),
                    hovertemplate="Lag %{x}d: corr=%{y:.2f}<extra></extra>",
                ))
                _fig142b.add_hline(y=0, line_color="#4b5563", line_width=1)
                _fig142b.add_vline(x=0, line_color="#6b7280", line_width=1, line_dash="dot")
                _fig142b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Credit Impulse Lead-Lag vs HY Spread Level (negative lag = impulse leads HY)",
                               font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Lag (days) — negative = impulse leads"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Correlation"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig142b, use_container_width=True)
                # Distribution
                _fig142c = _go142.Figure()
                _fig142c.add_trace(_go142.Histogram(
                    x=_ci142.values, nbinsx=50,
                    marker_color="#8b5cf6", opacity=0.6,
                    hovertemplate="Impulse %{x:.0f}bps: %{y} days<extra></extra>",
                ))
                _curr_pct142 = float((_ci142 < _curr_ci).mean() * 100)
                _fig142c.add_vline(x=_curr_ci, line_color="#ffffff", line_width=2,
                                   annotation_text=f"Now: {_curr_ci:+.0f} ({_curr_pct142:.0f}th pct)",
                                   annotation_font=dict(color="#ffffff", size=9))
                _fig142c.add_vline(x=0, line_color="#6b7280", line_width=1, line_dash="dot")
                _fig142c.update_layout(
                    height=180, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Credit Impulse Distribution (full history)", font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Impulse (bps)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig142c, use_container_width=True)
        else:
            st.info("credit_impulse column not found — run the feature pipeline.")
    except Exception as _e142:
        _err_track(_active_sub, _e142)
        st.caption(f"Credit impulse drill: {_e142}")

# sub143 — Sahm Episodes (Rates & Macro)

if _active_sub == 143:
    try:
        import plotly.graph_objects as _go143
        import numpy as _np143
        import pandas as _pd143
        if "sahm_like" in df.columns:
            _sahm143 = df["sahm_like"].dropna()
            _unemp143 = df["unemployment"].dropna() if "unemployment" in df.columns else None
            _SAHM_TRIGGER = 0.50  # Claudia Sahm's threshold
            # Time series with trigger line and shading
            _fig143a = _go143.Figure()
            _fig143a.add_trace(_go143.Scatter(
                x=_sahm143.index, y=_sahm143.values,
                mode="lines", name="Sahm-like",
                line=dict(color="#f59e0b", width=1.5),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}pp<extra></extra>",
            ))
            _fig143a.add_hline(y=_SAHM_TRIGGER, line_color="#ef4444", line_width=1.5, line_dash="dash",
                               annotation_text="Sahm Trigger (0.50pp)", annotation_position="top left",
                               annotation_font=dict(color="#ef4444", size=9))
            _fig143a.add_hline(y=0.25, line_color="#f59e0b", line_width=1, line_dash="dot",
                               annotation_text="Watch (0.25pp)", annotation_font=dict(color="#f59e0b", size=8))
            # Shade trigger episodes
            _above143 = _sahm143 >= _SAHM_TRIGGER
            _in_ep143 = False
            _ep_s143 = None
            for _d, _v in _above143.items():
                if _v and not _in_ep143:
                    _in_ep143 = True; _ep_s143 = _d
                elif not _v and _in_ep143:
                    _in_ep143 = False
                    _fig143a.add_vrect(x0=_ep_s143, x1=_d, fillcolor="rgba(239,68,68,0.1)",
                                       layer="below", line_width=0)
            if _in_ep143:
                _fig143a.add_vrect(x0=_ep_s143, x1=_sahm143.index[-1],
                                   fillcolor="rgba(239,68,68,0.1)", layer="below", line_width=0)
            _fig143a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Sahm-like Indicator: Unemployment Rise from 12m Low (red = trigger zone)",
                           font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="pp above 12m low"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig143a, use_container_width=True)
            # Episode table
            _episodes143 = []
            _in_ep143 = False; _ep_s143 = None; _ep_peak = 0.0
            for _d, _v in _sahm143.items():
                if _v >= _SAHM_TRIGGER and not _in_ep143:
                    _in_ep143 = True; _ep_s143 = _d; _ep_peak = _v
                elif _v >= _SAHM_TRIGGER and _in_ep143:
                    _ep_peak = max(_ep_peak, _v)
                elif _v < _SAHM_TRIGGER and _in_ep143:
                    _in_ep143 = False
                    _episodes143.append({"Start": str(_ep_s143.date()), "End": str(_d.date()),
                                          "Duration (days)": (_d - _ep_s143).days, "Peak": round(_ep_peak, 2)})
                    _ep_peak = 0.0
            if _in_ep143:
                _episodes143.append({"Start": str(_ep_s143.date()), "End": "ongoing",
                                      "Duration (days)": (_sahm143.index[-1] - _ep_s143).days,
                                      "Peak": round(_ep_peak, 2)})
            if _episodes143:
                st.markdown("**Trigger Episodes (Sahm ≥ 0.50pp)**")
                st.dataframe(_pd143.DataFrame(_episodes143).set_index("Start"), use_container_width=True)
            else:
                st.info("No Sahm trigger episodes found in the dataset.")
            # Sahm vs unemployment level dual axis
            if _unemp143 is not None:
                _j143 = _sahm143.to_frame("sahm").join(_unemp143.to_frame("unemp"), how="inner").dropna()
                _fig143b = _go143.Figure()
                _fig143b.add_trace(_go143.Scatter(
                    x=_j143.index, y=_j143["sahm"],
                    mode="lines", name="Sahm-like",
                    line=dict(color="#f59e0b", width=1.2),
                    hovertemplate="Sahm: %{y:.2f}pp<extra></extra>",
                    yaxis="y1",
                ))
                _fig143b.add_trace(_go143.Scatter(
                    x=_j143.index, y=_j143["unemp"],
                    mode="lines", name="Unemployment Rate",
                    line=dict(color="#6b7280", width=1.0, dash="dot"),
                    hovertemplate="UNRATE: %{y:.1f}%<extra></extra>",
                    yaxis="y2",
                ))
                _fig143b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Sahm-like vs Unemployment Rate", font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Sahm (pp)"),
                    yaxis2=dict(overlaying="y", side="right", color="#6b7280", title="UNRATE (%)", showgrid=False),
                    legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig143b, use_container_width=True)
            _curr_sahm = float(_sahm143.iloc[-1])
            _sahm_pct143 = float((_sahm143 < _curr_sahm).mean() * 100)
            st.caption(
                f"Current: **{_curr_sahm:.2f}pp** ({_sahm_pct143:.0f}th historical percentile). "
                f"{'🔴 TRIGGERED — recession signal historically reliable with ≤1 false positive since 1970.' if _curr_sahm >= 0.50 else ('🟡 Watch zone — approaching trigger.' if _curr_sahm >= 0.25 else '🟢 Below watch threshold.')}"
            )
        else:
            st.info("sahm_like column not found — run the feature pipeline.")
    except Exception as _e143:
        _err_track(_active_sub, _e143)
        st.caption(f"Sahm episodes: {_e143}")

# sub144 — Score Momentum vs Level (Signal Lab)

if _active_sub == 145:
    try:
        import plotly.graph_objects as _go145
        import numpy as _np145
        import pandas as _pd145
        if "real_yield_proxy" in df.columns:
            _ry145 = df["real_yield_proxy"].dropna()
            _ryz145 = df["real_yield_z"].dropna() if "real_yield_z" in df.columns else None
            _hy145 = df["hy_spread"].dropna() if "hy_spread" in df.columns else None
            _sp145 = df["sp500"].dropna() if "sp500" in df.columns else None

            def _ry_regime(v):
                if _np145.isnan(v): return "Unknown"
                if v < -1.0:  return "Deeply Negative"
                if v < 0.0:   return "Negative"
                if v < 1.0:   return "Low Positive"
                if v < 2.5:   return "Normal"
                return "Restrictive"
            _RY_COLORS = {"Deeply Negative": "#1e40af", "Negative": "#3b82f6",
                          "Low Positive": "#10b981", "Normal": "#6b7280",
                          "Restrictive": "#ef4444", "Unknown": "#4b5563"}

            # Real yield time series
            _fig145a = _go145.Figure()
            _fig145a.add_trace(_go145.Scatter(
                x=_ry145.index, y=_ry145.values,
                mode="lines", line=dict(color="#06b6d4", width=1.3),
                fill="tozeroy", fillcolor="rgba(6,182,212,0.07)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>",
            ))
            _fig145a.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig145a.add_hline(y=2.5, line_color="#ef4444", line_width=1, line_dash="dot",
                               annotation_text="Restrictive (2.5%)", annotation_font=dict(color="#ef4444", size=8))
            _fig145a.add_hline(y=-1.0, line_color="#1e40af", line_width=1, line_dash="dot",
                               annotation_text="Deeply Neg (−1%)", annotation_font=dict(color="#1e40af", size=8))
            _fig145a.update_layout(
                height=230, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Real Yield Proxy (10y Nominal − 10y Breakeven)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Real Yield (%)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig145a, use_container_width=True)

            # Z-score overlay
            if _ryz145 is not None:
                _fig145b = _go145.Figure()
                _fig145b.add_trace(_go145.Scatter(
                    x=_ryz145.index, y=_ryz145.values,
                    mode="lines", line=dict(color="#8b5cf6", width=1.0),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}σ<extra></extra>",
                ))
                _fig145b.add_hline(y=0, line_color="#4b5563", line_width=1)
                _fig145b.add_hline(y=2, line_color="#ef4444", line_width=1, line_dash="dot")
                _fig145b.add_hline(y=-2, line_color="#1e40af", line_width=1, line_dash="dot")
                _fig145b.update_layout(
                    height=170, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Real Yield Rolling Z-Score vs 252d History", font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Z-Score (σ)", zeroline=True, zerolinecolor="#4b5563"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig145b, use_container_width=True)

            # HY spread by real yield regime
            if _hy145 is not None:
                _ry_reg_series = _ry145.apply(_ry_regime)
                _j145 = _ry_reg_series.to_frame("regime").join(_hy145.to_frame("hy"), how="inner").dropna()
                _REGIME_ORDER145 = ["Deeply Negative", "Negative", "Low Positive", "Normal", "Restrictive"]
                _fig145c = _go145.Figure()
                for reg in _REGIME_ORDER145:
                    _sub = _j145[_j145["regime"] == reg]["hy"]
                    if len(_sub) < 5:
                        continue
                    _fig145c.add_trace(_go145.Box(
                        y=_sub.values, name=reg,
                        marker_color=_RY_COLORS.get(reg, "#6b7280"),
                        boxmean=True,
                        hovertemplate=f"{reg}: %{{y:.0f}}bps<extra></extra>",
                    ))
                _fig145c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="HY Spread Distribution by Real Yield Regime", font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="HY Spread (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig145c, use_container_width=True)
                _curr_ry = float(_ry145.iloc[-1])
                _curr_reg = _ry_regime(_curr_ry)
                _ry_pct = float((_ry145 < _curr_ry).mean() * 100)
                st.caption(f"Current real yield: **{_curr_ry:.2f}%** ({_ry_pct:.0f}th pct) — Regime: **{_curr_reg}**. "
                           "Restrictive real yields historically correlate with wider credit spreads and equity stress.")
        else:
            st.info("real_yield_proxy not found — run the feature pipeline.")
    except Exception as _e145:
        _err_track(_active_sub, _e145)
        st.caption(f"Real yield episodes: {_e145}")


# sub146 — Credit Beta by Regime (Credit Markets)

if _active_sub == 152:
    try:
        import plotly.graph_objects as _go152
        import numpy as _np152
        import pandas as _pd152
        _df152 = df.copy() if "df" in dir() else None
        _has152 = (_df152 is not None
                   and "sahm_like" in _df152.columns
                   and "hy_change_30d" in _df152.columns)
        if not _has152:
            st.info("sahm_like and hy_change_30d required.")
        else:
            st.subheader("Credit-Labor Divergence")
            st.caption("Cross-signal divergence between labor market stress (Sahm-like) and 30d HY spread change. Four quadrants: when credit tightens before labor weakens, credit is leading. When labor weakens but credit holds, expect delayed credit repricing.")
            _sahm152 = _df152["sahm_like"].dropna()
            _hy152 = _df152["hy_change_30d"].dropna()
            _j152 = _sahm152.to_frame("sahm").join(_hy152.to_frame("hy_chg"), how="inner").dropna().tail(1260)
            def _quad152(row):
                if row["sahm"] < 0.3 and row["hy_chg"] < 0:
                    return "Both Calm"
                elif row["sahm"] >= 0.3 and row["hy_chg"] >= 0:
                    return "Both Stressed"
                elif row["sahm"] < 0.3 and row["hy_chg"] >= 0:
                    return "Credit Leads"
                else:
                    return "Labor Leads"
            _j152["quadrant"] = _j152.apply(_quad152, axis=1)
            _colors152 = {"Both Calm": "#22c55e", "Both Stressed": "#ef4444",
                          "Credit Leads": "#f59e0b", "Labor Leads": "#8b5cf6"}
            _fig152 = _go152.Figure()
            for _q152, _c152 in _colors152.items():
                _sub152 = _j152[_j152["quadrant"] == _q152]
                _fig152.add_trace(_go152.Scatter(
                    x=_sub152["sahm"], y=_sub152["hy_chg"],
                    mode="markers", marker=dict(color=_c152, size=4, opacity=0.55),
                    name=_q152
                ))
            _cur152 = _j152.iloc[-1]
            _fig152.add_trace(_go152.Scatter(
                x=[_cur152["sahm"]], y=[_cur152["hy_chg"]],
                mode="markers", marker=dict(color="white", size=12, symbol="star"),
                name="Current"
            ))
            _fig152.add_vline(x=0.3, line_dash="dash", line_color="#9aa0aa")
            _fig152.add_hline(y=0, line_dash="dash", line_color="#9aa0aa")
            _fig152.update_layout(
                title="Credit vs Labor Stress Quadrants (5Y)",
                height=400,
                xaxis_title="Sahm-like (0.3 = recession warning)",
                yaxis_title="HY 30d Change (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig152, use_container_width=True)
            _qdist152 = _j152["quadrant"].value_counts()
            _cur_quad152 = str(_cur152["quadrant"])
            st.caption(
                f"Current: **{_cur_quad152}** · "
                + " · ".join(f"{k}: {v}d ({v/len(_j152)*100:.0f}%)" for k, v in _qdist152.items())
            )
    except Exception as _e152:
        _err_track(_active_sub, _e152)
        st.caption(f"Credit-labor divergence: {_e152}")


if _active_sub == 161:
    try:
        import plotly.graph_objects as _go161
        import numpy as _np161
        import pandas as _pd161
        _df161 = df.copy() if "df" in dir() else None
        _has161 = (_df161 is not None
                   and "nfci_90d_avg" in _df161.columns
                   and "hy_spread" in _df161.columns)
        if not _has161:
            st.info("nfci_90d_avg and hy_spread required.")
        else:
            st.subheader("Liquidity-Credit Nexus")
            st.caption("NFCI (National Financial Conditions Index) as a systemic liquidity proxy vs HY spreads. Tighter NFCI (more positive) = tighter financial conditions → tends to lead HY widening. Rolling correlation tracks how tightly the two are coupled in the current environment.")
            _nfci161 = _df161["nfci_90d_avg"].dropna()
            _hy161 = _df161["hy_spread"].dropna()
            _j161 = _nfci161.to_frame("nfci").join(_hy161.to_frame("hy"), how="inner").dropna().tail(1260)
            _fig161 = _go161.Figure()
            _fig161.add_trace(_go161.Scatter(
                x=_j161.index, y=_j161["hy"],
                name="HY Spread (bps)", line=dict(color="#ef4444", width=1.5),
                yaxis="y1"
            ))
            _fig161.add_trace(_go161.Scatter(
                x=_j161.index, y=_j161["nfci"],
                name="NFCI 90d Avg", line=dict(color="#6366f1", width=1.5, dash="dot"),
                yaxis="y2"
            ))
            _fig161.update_layout(
                title="HY Spread vs NFCI (5Y)",
                height=350,
                yaxis=dict(title="HY Spread (bps)", side="left"),
                yaxis2=dict(title="NFCI", side="right", overlaying="y"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig161, use_container_width=True)
            # Rolling correlation
            _roll_corr161 = _j161["nfci"].rolling(63).corr(_j161["hy"])
            _fig161b = _go161.Figure()
            _fig161b.add_trace(_go161.Scatter(
                x=_roll_corr161.index, y=_roll_corr161.values,
                line=dict(color="#6366f1", width=1.5),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)"
            ))
            _fig161b.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig161b.update_layout(
                title="Rolling 63d NFCI–HY Correlation",
                height=220, yaxis_title="Correlation", yaxis=dict(range=[-1, 1]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig161b, use_container_width=True)
            # Scatter
            _fig161c = _go161.Figure()
            _fig161c.add_trace(_go161.Scatter(
                x=_j161["nfci"], y=_j161["hy"],
                mode="markers", marker=dict(color="#6366f1", size=3, opacity=0.4),
                name="History"
            ))
            _cur161 = _j161.iloc[-1]
            _fig161c.add_trace(_go161.Scatter(
                x=[_cur161["nfci"]], y=[_cur161["hy"]],
                mode="markers+text", marker=dict(color="white", size=10, symbol="star"),
                text=["Now"], textposition="top center", name="Current"
            ))
            _fig161c.update_layout(
                title="NFCI vs HY Spread — Scatter (5Y)",
                height=300, xaxis_title="NFCI 90d Avg", yaxis_title="HY Spread (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig161c, use_container_width=True)
            _cur_corr161 = float(_roll_corr161.iloc[-1]) if _roll_corr161.notna().any() else float("nan")
            _coupling161 = "tightly coupled" if abs(_cur_corr161) > 0.5 else ("moderately coupled" if abs(_cur_corr161) > 0.25 else "decoupled")
            st.caption(
                f"Current NFCI: **{float(_cur161['nfci']):.3f}** · "
                f"63d NFCI-HY correlation: **{_cur_corr161:.2f}** ({_coupling161}). "
                "Positive correlation = tighter financial conditions historically associated with higher spreads."
            )
    except Exception as _e161:
        _err_track(_active_sub, _e161)
        st.caption(f"Liquidity-credit nexus: {_e161}")


if _active_sub == "ov_rm":
    try:
        import plotly.graph_objects as _go_ov_rm
        import numpy as _np_ov_rm
        st.subheader("Rates & Macro — Section Overview")
        st.caption("Current snapshot of yield curve, inflation, labor, and financial conditions. Select any sub-view from the sidebar to drill in.")
        _d = df
        def _last_rm(col):
            s = _d[col].dropna(); return float(s.iloc[-1]) if len(s) else float("nan")
        def _chg_rm(col, n=21):
            s = _d[col].dropna()
            if len(s) < n + 1: return float("nan")
            return float(s.iloc[-1]) - float(s.iloc[-n - 1])
        def _pct_rm(col):
            s = _d[col].dropna()
            if len(s) < 10: return float("nan")
            return float((s < float(s.iloc[-1])).mean() * 100)
        _c1, _c2, _c3, _c4 = st.columns(4)
        _curve = _last_rm("spread"); _curve_chg = _chg_rm("spread")
        _c1.metric("2s10s Curve", f"{_curve:.2f}%" if not _np_ov_rm.isnan(_curve) else "—",
                   delta=f"{_curve_chg:+.2f}pp 21d" if not _np_ov_rm.isnan(_curve_chg) else None)
        _be = _last_rm("breakeven_10y"); _be_chg = _chg_rm("breakeven_10y")
        _c2.metric("10y Breakeven", f"{_be:.2f}%" if not _np_ov_rm.isnan(_be) else "—",
                   delta=f"{_be_chg:+.2f}pp 21d" if not _np_ov_rm.isnan(_be_chg) else None)
        _nfci = _last_rm("nfci"); _nfci_pct = _pct_rm("nfci")
        _c3.metric("NFCI", f"{_nfci:.3f}" if not _np_ov_rm.isnan(_nfci) else "—",
                   delta=f"{_nfci_pct:.0f}th pct" if not _np_ov_rm.isnan(_nfci_pct) else None,
                   delta_color="inverse")
        _sahm = _last_rm("sahm_like")
        _c4.metric("Sahm-like", f"{_sahm:.2f}" if not _np_ov_rm.isnan(_sahm) else "—",
                   delta="⚠ Elevated" if not _np_ov_rm.isnan(_sahm) and _sahm >= 0.3 else "Normal",
                   delta_color="inverse" if not _np_ov_rm.isnan(_sahm) and _sahm >= 0.3 else "off")
        st.divider()
        # Yield curve + real yield chart
        _fig_ov_rm = _go_ov_rm.Figure()
        _curve_s = _d["spread"].dropna().tail(504)
        _fig_ov_rm.add_trace(_go_ov_rm.Scatter(x=_curve_s.index, y=_curve_s.values,
            name="2s10s", line=dict(color="#6366f1", width=2), fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)"))
        _fig_ov_rm.add_hline(y=0, line_color="#ef4444", line_dash="dash", annotation_text="Inversion")
        if "real_yield_proxy" in _d.columns:
            _ry_s = _d["real_yield_proxy"].dropna().tail(504)
            _fig_ov_rm.add_trace(_go_ov_rm.Scatter(x=_ry_s.index, y=_ry_s.values,
                name="Real Yield", line=dict(color="#f59e0b", width=1.5), yaxis="y2"))
        _fig_ov_rm.update_layout(
            title="Yield Curve (2s10s) & Real Yield — Last 2 Years",
            height=280, yaxis=dict(title="Curve (%)"),
            yaxis2=dict(title="Real Yield (%)", overlaying="y", side="right"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=40, b=20))
        st.plotly_chart(_fig_ov_rm, use_container_width=True)
        _regime_label = "Inverted" if not _np_ov_rm.isnan(_curve) and _curve < 0 else "Normal"
        st.info(f"Yield curve: **{_regime_label}** ({_curve:.2f}%). "
                f"37 sub-views available: real rates, NFCI, Sahm rule, credit impulse, Taylor rule, and more.")
    except Exception as _e_ov_rm:
        _err_track(_active_sub, _e_ov_rm)
        st.caption(f"Rates & Macro overview: {_e_ov_rm}")


if _active_sub == 176:
    try:
        import plotly.graph_objects as _go176
        import numpy as _np176
        import pandas as _pd176
        _df176 = df.copy() if "df" in dir() else None
        _has176 = (_df176 is not None
                   and "initial_claims" in _df176.columns
                   and "hy_spread" in _df176.columns)
        if not _has176:
            st.info("initial_claims and hy_spread required.")
        else:
            st.subheader("Initial Claims — Credit Leading Indicator")
            st.caption("Weekly initial jobless claims are the earliest available labor market signal. Claims typically lead HY spread widening by 4–8 weeks as rising layoffs foreshadow deteriorating credit quality. This view tracks the claims-credit relationship and flags when claims are breaking out of their normal range.")
            _claims176 = _df176["initial_claims"].dropna()
            _hy176 = _df176["hy_spread"].dropna()
            _j176 = _claims176.to_frame("claims").join(_hy176.to_frame("hy"), how="inner").dropna().tail(1260)
            _claims_ma176 = _j176["claims"].rolling(13).mean()  # 13-week MA
            _claims_z176 = ((_j176["claims"] - _claims_ma176) /
                            (_j176["claims"].rolling(52).std() + 1e-9))
            _fig176 = _go176.Figure()
            _fig176.add_trace(_go176.Scatter(
                x=_j176.index, y=_j176["claims"] / 1000,
                name="Initial Claims (k)", line=dict(color="#8b5cf6", width=1.5)))
            _fig176.add_trace(_go176.Scatter(
                x=_claims_ma176.index, y=_claims_ma176.values / 1000,
                name="13-week MA", line=dict(color="#f59e0b", width=1.5, dash="dot")))
            _fig176.update_layout(
                title="Initial Jobless Claims (thousands)",
                height=280, yaxis_title="Claims (000s)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(_fig176, use_container_width=True)
            # Dual-axis: claims z-score vs HY
            _fig176b = _go176.Figure()
            _fig176b.add_trace(_go176.Scatter(
                x=_j176.index, y=_j176["hy"],
                name="HY Spread (bps)", line=dict(color="#ef4444", width=1.5)))
            _fig176b.add_trace(_go176.Scatter(
                x=_claims_z176.index, y=_claims_z176.values,
                name="Claims Z-score", line=dict(color="#8b5cf6", width=1.5, dash="dot"),
                yaxis="y2"))
            _fig176b.add_hline(y=0, line_color="#9aa0aa", line_width=0.5, yref="y2")
            _fig176b.update_layout(
                title="HY Spread vs Claims Z-score (claims leads by ~6 weeks)",
                height=300,
                yaxis=dict(title="HY Spread (bps)"),
                yaxis2=dict(title="Claims Z-score", overlaying="y", side="right"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(_fig176b, use_container_width=True)
            # Forward correlation at 6-week lag
            _fwd_hy176 = _j176["hy"].diff(30).shift(-30)
            _j176b = _claims_z176.to_frame("cz").join(_fwd_hy176.to_frame("fwd"), how="inner").dropna()
            _lead_corr176 = float(_j176b["cz"].corr(_j176b["fwd"]))
            _cur_cz176 = float(_claims_z176.iloc[-1]) if _claims_z176.notna().any() else float("nan")
            _cur_claims176 = float(_claims176.iloc[-1])
            _claims_signal176 = "Elevated" if not _np176.isnan(_cur_cz176) and _cur_cz176 > 1 else (
                "Rising" if not _np176.isnan(_cur_cz176) and _cur_cz176 > 0.3 else "Normal")
            st.caption(
                f"Current claims: **{_cur_claims176:,.0f}** (z={_cur_cz176:.2f} vs 52w baseline) — {_claims_signal176}. "
                f"30d lead correlation with HY widening: **{_lead_corr176:.2f}**.")
    except Exception as _e176:
        _err_track(_active_sub, _e176)
        st.caption(f"Initial claims: {_e176}")


if _active_sub == 177:
    try:
        import plotly.graph_objects as _go177
        import numpy as _np177
        import pandas as _pd177
        _df177 = df.copy() if "df" in dir() else None
        _has177 = (_df177 is not None
                   and "oil_wti" in _df177.columns
                   and "hy_spread" in _df177.columns)
        if not _has177:
            st.info("oil_wti and hy_spread required.")
        else:
            st.subheader("Oil-Credit Nexus")
            st.caption("WTI crude oil price changes and HY credit spreads are tightly linked through the energy sector's large HY market share (~15% of index). Oil shocks widen HY via energy defaults; oil rallies compress energy-sector spreads. This view tracks the oil-credit relationship and identifies divergence periods.")
            _oil177 = _df177["oil_wti"].dropna()
            _hy177 = _df177["hy_spread"].dropna()
            _j177 = _oil177.to_frame("oil").join(_hy177.to_frame("hy"), how="inner").dropna().tail(1260)
            _oil_ret177 = _j177["oil"].pct_change(21)
            _hy_chg177 = _j177["hy"].diff(21)
            _roll_corr177 = _oil_ret177.rolling(63).corr(_hy_chg177)
            # Dual axis: oil vs HY
            _fig177 = _go177.Figure()
            _fig177.add_trace(_go177.Scatter(
                x=_j177.index, y=_j177["hy"],
                name="HY Spread (bps)", line=dict(color="#ef4444", width=1.5)))
            _fig177.add_trace(_go177.Scatter(
                x=_j177.index, y=_j177["oil"],
                name="WTI ($/bbl)", line=dict(color="#22c55e", width=1.5, dash="dot"),
                yaxis="y2"))
            _fig177.update_layout(
                title="HY Spread vs WTI Oil (5Y)",
                height=300,
                yaxis=dict(title="HY Spread (bps)"),
                yaxis2=dict(title="WTI ($/bbl)", overlaying="y", side="right"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(_fig177, use_container_width=True)
            # Rolling correlation
            _fig177b = _go177.Figure()
            _fig177b.add_trace(_go177.Scatter(
                x=_roll_corr177.index, y=_roll_corr177.values,
                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                line=dict(color="#6366f1", width=1.5)))
            _fig177b.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig177b.update_layout(
                title="Rolling 63d Correlation: Oil Returns vs HY Spread Change",
                height=220, yaxis_title="Correlation", yaxis=dict(range=[-1, 1]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False)
            st.plotly_chart(_fig177b, use_container_width=True)
            # Scatter with quadrant
            _fig177c = _go177.Figure()
            _j177c = _oil_ret177.to_frame("oil_r").join(_hy_chg177.to_frame("hy_c"), how="inner").dropna()
            _fig177c.add_trace(_go177.Scatter(
                x=_j177c["oil_r"] * 100, y=_j177c["hy_c"],
                mode="markers", marker=dict(color="#6366f1", size=3, opacity=0.35),
                name="History"))
            _cur_oil_r177 = float(_oil_ret177.iloc[-1]) * 100 if _oil_ret177.notna().any() else float("nan")
            _cur_hy_c177 = float(_hy_chg177.iloc[-1]) if _hy_chg177.notna().any() else float("nan")
            _fig177c.add_trace(_go177.Scatter(
                x=[_cur_oil_r177], y=[_cur_hy_c177],
                mode="markers+text", marker=dict(color="white", size=12, symbol="star"),
                text=["Now"], textposition="top center"))
            _fig177c.add_vline(x=0, line_color="#9aa0aa", line_width=0.5)
            _fig177c.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig177c.update_layout(
                title="21d Oil Return vs 21d HY Change",
                height=280, xaxis_title="Oil 21d Return (%)", yaxis_title="HY 21d Change (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False)
            st.plotly_chart(_fig177c, use_container_width=True)
            _cur_corr177 = float(_roll_corr177.iloc[-1]) if _roll_corr177.notna().any() else float("nan")
            _cur_oil177 = float(_oil177.iloc[-1])
            _oil_regime177 = "Rising" if not _np177.isnan(_cur_oil_r177) and _cur_oil_r177 > 5 else (
                "Falling" if not _np177.isnan(_cur_oil_r177) and _cur_oil_r177 < -5 else "Stable")
            st.caption(
                f"WTI: **${_cur_oil177:.1f}/bbl** ({_oil_regime177}). "
                f"63d oil-HY correlation: **{_cur_corr177:.2f}** "
                f"({'negative = inverse relationship' if not _np177.isnan(_cur_corr177) and _cur_corr177 < 0 else 'positive = co-movement'}).")
    except Exception as _e177:
        _err_track(_active_sub, _e177)
        st.caption(f"Oil-credit: {_e177}")


if _active_sub == 181:
    st.subheader("FX-Credit Nexus")
    st.caption("USD strength vs credit spreads — EUR/USD and USD/JPY as leading risk indicators")
    try:
        import plotly.graph_objects as _go181
        import numpy as _np181
        import pandas as _pd181
        _df181 = df[["hy_spread","eurusd","usdjpy","eurusd_change_30d","usdjpy_change_30d","hy_spread"]].dropna().copy()
        _df181["hy_change_30d"] = _df181["hy_spread"].diff(30)
        _df181 = _df181.dropna()
        # Header metrics
        _last181 = _df181.iloc[-1]
        _c1_181, _c2_181, _c3_181, _c4_181 = st.columns(4)
        _c1_181.metric("EUR/USD", f"{_last181['eurusd']:.4f}", f"{_last181['eurusd_change_30d']:+.4f} 30d")
        _c2_181.metric("USD/JPY", f"{_last181['usdjpy']:.2f}", f"{_last181['usdjpy_change_30d']:+.2f} 30d")
        _c3_181.metric("HY Spread", f"{_last181['hy_spread']:.0f} bps", f"{_last181['hy_change_30d']:+.0f} 30d")
        _corr_eur181 = round(float(_df181["eurusd_change_30d"].corr(_df181["hy_change_30d"])), 2)
        _c4_181.metric("EUR↔HY Corr", f"{_corr_eur181:+.2f}", "30d changes")
        st.divider()
        # Dual-axis time series: EUR/USD (inverted, USD strength) + HY spread
        _fig181a = _go181.Figure()
        _fig181a.add_trace(_go181.Scatter(
            x=_df181.index, y=_df181["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#ef4444", width=1.5), yaxis="y1"
        ))
        _fig181a.add_trace(_go181.Scatter(
            x=_df181.index, y=_df181["eurusd"],
            name="EUR/USD", line=dict(color="#3b82f6", width=1.2), yaxis="y2"
        ))
        _fig181a.add_trace(_go181.Scatter(
            x=_df181.index, y=_df181["usdjpy"],
            name="USD/JPY", line=dict(color="#f59e0b", width=1.2), yaxis="y3", visible="legendonly"
        ))
        _fig181a.update_layout(
            title="HY Spread vs FX (EUR/USD & USD/JPY)",
            height=360,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="HY Spread (bps)", side="left", color="#ef4444"),
            yaxis2=dict(title="EUR/USD", side="right", overlaying="y", color="#3b82f6", showgrid=False),
            yaxis3=dict(title="USD/JPY", side="right", overlaying="y", position=0.95, color="#f59e0b", showgrid=False, visible=False),
            legend=dict(orientation="h", y=1.05),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig181a, use_container_width=True)
        # Scatter: 30d EUR/USD change vs 30d HY change
        _c1b_181, _c2b_181 = st.columns(2)
        with _c1b_181:
            _fig181b = _go181.Figure()
            _fig181b.add_trace(_go181.Scatter(
                x=_df181["eurusd_change_30d"], y=_df181["hy_change_30d"],
                mode="markers", marker=dict(size=3, color="#3b82f6", opacity=0.5), name="EUR/USD vs HY"
            ))
            # Regression line
            _m181, _b181 = _np181.polyfit(_df181["eurusd_change_30d"], _df181["hy_change_30d"], 1)
            _x181 = _np181.linspace(_df181["eurusd_change_30d"].min(), _df181["eurusd_change_30d"].max(), 50)
            _fig181b.add_trace(_go181.Scatter(x=_x181, y=_m181*_x181+_b181, mode="lines", line=dict(color="#ef4444", width=1.5), name="Trend"))
            _fig181b.update_layout(
                title=f"EUR/USD Δ30d vs HY Δ30d (r={_corr_eur181:+.2f})",
                height=280,
                xaxis_title="EUR/USD Change (30d)", yaxis_title="HY Spread Change (30d bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(t=40, b=30))
            st.plotly_chart(_fig181b, use_container_width=True)
        with _c2b_181:
            _corr_jpy181 = round(float(_df181["usdjpy_change_30d"].corr(_df181["hy_change_30d"])), 2)
            _fig181c = _go181.Figure()
            _fig181c.add_trace(_go181.Scatter(
                x=_df181["usdjpy_change_30d"], y=_df181["hy_change_30d"],
                mode="markers", marker=dict(size=3, color="#f59e0b", opacity=0.5), name="USD/JPY vs HY"
            ))
            _m181j, _b181j = _np181.polyfit(_df181["usdjpy_change_30d"], _df181["hy_change_30d"], 1)
            _x181j = _np181.linspace(_df181["usdjpy_change_30d"].min(), _df181["usdjpy_change_30d"].max(), 50)
            _fig181c.add_trace(_go181.Scatter(x=_x181j, y=_m181j*_x181j+_b181j, mode="lines", line=dict(color="#ef4444", width=1.5), name="Trend"))
            _fig181c.update_layout(
                title=f"USD/JPY Δ30d vs HY Δ30d (r={_corr_jpy181:+.2f})",
                height=280,
                xaxis_title="USD/JPY Change (30d)", yaxis_title="HY Spread Change (30d bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(t=40, b=30))
            st.plotly_chart(_fig181c, use_container_width=True)
        # Rolling 90d correlation
        _roll_corr_eur181 = _df181["eurusd_change_30d"].rolling(90).corr(_df181["hy_change_30d"])
        _roll_corr_jpy181 = _df181["usdjpy_change_30d"].rolling(90).corr(_df181["hy_change_30d"])
        _fig181d = _go181.Figure()
        _fig181d.add_trace(_go181.Scatter(x=_df181.index, y=_roll_corr_eur181, name="EUR/USD↔HY (90d)", line=dict(color="#3b82f6", width=1.5)))
        _fig181d.add_trace(_go181.Scatter(x=_df181.index, y=_roll_corr_jpy181, name="USD/JPY↔HY (90d)", line=dict(color="#f59e0b", width=1.5)))
        _fig181d.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)
        _fig181d.update_layout(
            title="Rolling 90d FX-HY Correlation",
            height=240,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), yaxis_title="Correlation",
            margin=dict(t=40, b=30),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig181d, use_container_width=True)
        _risk_flag181 = abs(_last181["eurusd_change_30d"]) > 0.02 and _last181["hy_change_30d"] > 20
        st.caption(
            f"EUR/USD 30d change {_last181['eurusd_change_30d']:+.4f}; USD/JPY 30d change {_last181['usdjpy_change_30d']:+.2f}. "
            f"EUR↔HY corr {_corr_eur181:+.2f}, JPY↔HY corr {_corr_jpy181:+.2f} over full history. "
            f"{'Risk-off FX + spread widening alignment flagged.' if _risk_flag181 else 'FX-credit alignment within normal range.'}")
    except Exception as _e181:
        _err_track(_active_sub, _e181)
        st.caption(f"FX-Credit nexus: {_e181}")

# --- sub182: Fed Balance Sheet QT Tracker ---

if _active_sub == 182:
    st.subheader("Fed Balance Sheet / QT Tracker")
    st.caption("QE/QT cycle impact on credit spreads — balance sheet expansion vs contraction regimes")
    try:
        import plotly.graph_objects as _go182
        import numpy as _np182
        import pandas as _pd182
        _df182 = df[["fed_balance_sheet","fed_bs_change_90d","hy_spread","ig_spread","nfci"]].dropna().copy()
        _last182 = _df182.iloc[-1]
        _bs_pct182 = (_df182["fed_balance_sheet"].iloc[-1] / _df182["fed_balance_sheet"].max()) * 100
        _c1_182, _c2_182, _c3_182, _c4_182 = st.columns(4)
        _c1_182.metric("Fed BS (bn)", f"${_last182['fed_balance_sheet']:,.0f}", f"{_last182['fed_bs_change_90d']:+,.0f} 90d")
        _c2_182.metric("% of Peak", f"{_bs_pct182:.1f}%")
        _c3_182.metric("HY Spread", f"{_last182['hy_spread']:.0f} bps")
        _regime182 = "QT" if _last182["fed_bs_change_90d"] < 0 else "QE/Neutral"
        _c4_182.metric("Regime", _regime182)
        st.divider()
        # Dual axis: Fed BS + HY spread
        _fig182a = _go182.Figure()
        _fig182a.add_trace(_go182.Scatter(
            x=_df182.index, y=_df182["fed_balance_sheet"],
            name="Fed Balance Sheet (bn)", fill="tozeroy",
            fillcolor="rgba(59,130,246,0.15)", line=dict(color="#3b82f6", width=1.5), yaxis="y2"
        ))
        _fig182a.add_trace(_go182.Scatter(
            x=_df182.index, y=_df182["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#ef4444", width=1.5), yaxis="y1"
        ))
        _fig182a.add_trace(_go182.Scatter(
            x=_df182.index, y=_df182["ig_spread"],
            name="IG Spread (bps)", line=dict(color="#f59e0b", width=1.2, dash="dot"), yaxis="y1"
        ))
        # Shade QT periods (90d change < -100bn)
        _qt_mask182 = _df182["fed_bs_change_90d"] < -100
        _qt_starts182 = _df182.index[_qt_mask182 & (~_qt_mask182.shift(1, fill_value=False))]
        _qt_ends182 = _df182.index[_qt_mask182 & (~_qt_mask182.shift(-1, fill_value=False))]
        for _s182, _e182 in zip(_qt_starts182, _qt_ends182):
            _fig182a.add_vrect(x0=_s182, x1=_e182, fillcolor="rgba(239,68,68,0.08)", line_width=0, annotation_text="QT")
        _fig182a.update_layout(
            title="Fed Balance Sheet vs Credit Spreads (QT periods shaded)",
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Spread (bps)", side="left", color="#ef4444"),
            yaxis2=dict(title="Fed BS ($bn)", side="right", overlaying="y", color="#3b82f6", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig182a, use_container_width=True)
        # 90d change rate
        _fig182b = _go182.Figure()
        _fig182b.add_trace(_go182.Bar(
            x=_df182.index, y=_df182["fed_bs_change_90d"],
            marker_color=_df182["fed_bs_change_90d"].apply(lambda v: "#3b82f6" if v >= 0 else "#ef4444"),
            name="Fed BS 90d Change"
        ))
        _fig182b.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)
        _fig182b.update_layout(
            title="Fed Balance Sheet 90d Change (QE=blue, QT=red)",
            height=220,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), yaxis_title="$bn Change",
            margin=dict(t=40, b=30),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig182b, use_container_width=True)
        # Stats by regime
        _df182["bs_regime"] = _df182["fed_bs_change_90d"].apply(lambda v: "QT" if v < -100 else ("QE" if v > 100 else "Neutral"))
        _regime_stats182 = _df182.groupby("bs_regime")["hy_spread"].agg(["mean","std","count"]).round(1)
        _regime_stats182.columns = ["Avg HY Spread", "Std", "Observations"]
        st.markdown("**HY Spread by Fed Balance Sheet Regime**")
        st.dataframe(_regime_stats182, use_container_width=True)
        _corr_bs182 = round(float(_df182["fed_balance_sheet"].corr(_df182["hy_spread"])), 2)
        st.caption(
            f"Fed BS {_regime182} regime — current 90d change ${_last182['fed_bs_change_90d']:+,.0f}bn. "
            f"Full-history BS↔HY correlation: {_corr_bs182:+.2f}. "
            f"Balance sheet at {_bs_pct182:.1f}% of its all-time peak.")
    except Exception as _e182:
        _err_track(_active_sub, _e182)
        st.caption(f"Fed balance sheet: {_e182}")

# --- sub183: Banking Flow Monitor ---

if _active_sub == 189:
    st.subheader("Absolute Yield Monitor")
    st.caption("HY and IG absolute yield levels vs fed funds — carry above cash, real yield, and yield curve context")
    try:
        import plotly.graph_objects as _go189
        import numpy as _np189
        import pandas as _pd189
        _df189 = df[["hy_yield","ig_yield","fed_funds_rate","yield_10y","yield_2y","breakeven_10y"]].dropna().copy()
        _df189["hy_carry_over_cash"] = _df189["hy_yield"] - _df189["fed_funds_rate"]
        _df189["ig_carry_over_cash"] = _df189["ig_yield"] - _df189["fed_funds_rate"]
        _df189["hy_real_yield"] = _df189["hy_yield"] - _df189["breakeven_10y"]
        _df189["ig_real_yield"] = _df189["ig_yield"] - _df189["breakeven_10y"]
        _last189 = _df189.iloc[-1]
        _c1_189, _c2_189, _c3_189, _c4_189 = st.columns(4)
        _c1_189.metric("HY Yield", f"{_last189['hy_yield']:.2f}%", f"{_last189['hy_carry_over_cash']:+.2f}% over cash")
        _c2_189.metric("IG Yield", f"{_last189['ig_yield']:.2f}%", f"{_last189['ig_carry_over_cash']:+.2f}% over cash")
        _c3_189.metric("Fed Funds", f"{_last189['fed_funds_rate']:.2f}%")
        _c4_189.metric("HY Real Yield", f"{_last189['hy_real_yield']:.2f}%")
        st.divider()
        # Absolute yield time series
        _fig189a = _go189.Figure()
        _fig189a.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["hy_yield"],
            name="HY Yield", line=dict(color="#ef4444", width=2)
        ))
        _fig189a.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["ig_yield"],
            name="IG Yield", line=dict(color="#3b82f6", width=1.5)
        ))
        _fig189a.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["fed_funds_rate"],
            name="Fed Funds Rate", line=dict(color="#10b981", width=1.5, dash="dash")
        ))
        _fig189a.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["yield_10y"],
            name="10y Treasury", line=dict(color="#f59e0b", width=1, dash="dot")
        ))
        _fig189a.update_layout(
            title="Absolute Yield Levels: HY / IG / Fed Funds / 10y",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis_title="Yield (%)",
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig189a, use_container_width=True)
        # Carry over cash
        _fig189b = _go189.Figure()
        _fig189b.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["hy_carry_over_cash"],
            name="HY carry over cash", fill="tozeroy",
            fillcolor="rgba(239,68,68,0.15)", line=dict(color="#ef4444", width=1.5)
        ))
        _fig189b.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["ig_carry_over_cash"],
            name="IG carry over cash", fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)", line=dict(color="#3b82f6", width=1.5)
        ))
        _fig189b.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)
        _fig189b.update_layout(
            title="Carry Over Cash (Yield - Fed Funds Rate)",
            height=220,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis_title="Carry (%)",
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=30),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig189b, use_container_width=True)
        # Real yield
        _fig189c = _go189.Figure()
        _fig189c.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["hy_real_yield"],
            name="HY Real Yield", line=dict(color="#ef4444", width=1.5)
        ))
        _fig189c.add_trace(_go189.Scatter(
            x=_df189.index, y=_df189["ig_real_yield"],
            name="IG Real Yield", line=dict(color="#3b82f6", width=1.5)
        ))
        _fig189c.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1, annotation_text="Zero real yield")
        _fig189c.update_layout(
            title="Real Yield (Yield - 10y Breakeven Inflation)",
            height=200,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis_title="Real Yield (%)",
            margin=dict(t=40, b=25),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig189c, use_container_width=True)
        _hy_carry_pct189 = float((_df189["hy_carry_over_cash"] < _last189["hy_carry_over_cash"]).mean() * 100)
        st.caption(
            f"HY yield {_last189['hy_yield']:.2f}% / IG {_last189['ig_yield']:.2f}% vs fed funds {_last189['fed_funds_rate']:.2f}%. "
            f"HY carry over cash {_last189['hy_carry_over_cash']:+.2f}% ({_hy_carry_pct189:.0f}th pct). "
            f"HY real yield {_last189['hy_real_yield']:.2f}%.")
    except Exception as _e189:
        _err_track(_active_sub, _e189)
        st.caption(f"Absolute yield monitor: {_e189}")

# --- sub190: ETF Flow & Dislocation ---
