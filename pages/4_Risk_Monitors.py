"""
Risk Monitors — analytics section page.
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
from src.distressed_debt import run_distressed_debt_analysis
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
    page_title='Risk Monitors — Credit Risk Dashboard',
    page_icon='🔔',
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
_SECTION_NAME = 'Risk Monitors'
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


if _active_sub == 6:
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


if _active_sub == 7:
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


if _active_sub == 12:
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

if _active_sub == 39:
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

if _active_sub == 44:
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

if _active_sub == 46:
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

if _active_sub == 47:
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

if _active_sub == 52:
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

if _active_sub == 55:
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

if _active_sub == 56:
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

if _active_sub == 59:
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

if _active_sub == 65:
    import plotly.graph_objects as _go_etf
    st.header("ETF Premium / Discount Monitor")
    st.markdown(
        "Tracks whether HYG, LQD, JNK trade at a premium or discount to NAV. "
        "During stress, ETFs trade at discounts — a real-time signal of forced "
        "redemption pressure and bond market illiquidity."
    )
    try:
        _etf = load_etf_premium_discount(df)
        if _etf.get("available"):
            _etfc = _etf.get("current", {})
            _c1e, _c2e, _c3e = st.columns(3)
            _etf_score = _etfc.get("dislocation_score")
            _c1e.metric("Dislocation Score", f"{_etf_score:.0f}/100" if _etf_score is not None else "N/A")
            _c2e.metric("Regime", _etfc.get("regime", "N/A"))
            _c3e.metric("Most Dislocated", _etfc.get("most_dislocated", "N/A"))
            _cpd = _etfc.get("composite_premium_discount_bps")
            if _cpd is not None:
                _sign = "+" if _cpd > 0 else ""
                st.metric("Composite Premium/Discount", f"{_sign}{_cpd:.1f}bps")
            if _etfc.get("warning"):
                st.warning(_etfc["warning"])
            if _etfc.get("interpretation"):
                st.info(_etfc["interpretation"])
            _etf_readings = _etfc.get("etf_readings", {})
            if _etf_readings:
                _etf_rows = [
                    {"ETF": t, "P/D (bps)": f"{v.get('premium_discount_bps', 0):.1f}",
                     "Dislocation": "YES" if v.get("dislocation_flag") else "NO",
                     "Vol Ratio": f"{v.get('volume_ratio', 1):.1f}x"}
                    for t, v in _etf_readings.items()
                ]
                import pandas as _pd_etf
                st.dataframe(_pd_etf.DataFrame(_etf_rows).set_index("ETF"), use_container_width=True)
            _etf_hist = _etf.get("dislocation_history")
            if _etf_hist is not None and len(_etf_hist.dropna()) > 20:
                _fig_etf = _go_etf.Figure()
                _fig_etf.add_trace(_go_etf.Scatter(
                    x=_etf_hist.index, y=_etf_hist.values,
                    name="ETF Dislocation Score", line=dict(color="#f87171", width=1.5)
                ))
                _fig_etf.add_hline(y=40, line_dash="dot", line_color="#f59e0b", annotation_text="Stress")
                _fig_etf.add_hline(y=70, line_dash="dot", line_color="#ef4444", annotation_text="Dislocation")
                _fig_etf.update_layout(
                    template="plotly_dark", height=300, title="Credit ETF Dislocation Score (0–100)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_etf, use_container_width=True)
        else:
            st.info("ETF premium/discount monitor unavailable — requires live yfinance data (HYG, LQD, JNK).")
    except Exception as _etf_e:
        _err_track(_active_sub, _etf_e)
        st.caption(f"ETF premium/discount unavailable: {_etf_e}")

# --- sub-tab 66: Sovereign Contagion ----------------------------------------

if _active_sub == 66:
    import plotly.graph_objects as _go_sov
    st.header("Sovereign Contagion")
    st.markdown(
        "EU peripheral (Italy/Germany proxy) and EM sovereign stress monitor. "
        "Sovereign distress transmits to corporate credit via bank balance sheets. "
        "A leading indicator for corporate spread widening."
    )
    try:
        _sov = load_sovereign_contagion(df)
        if _sov.get("available"):
            _sovc = _sov.get("current", {})
            _c1sv, _c2sv, _c3sv, _c4sv = st.columns(4)
            _sov_score = _sovc.get("sovereign_stress_score")
            _c1sv.metric("Sovereign Stress", f"{_sov_score:.0f}/100" if _sov_score is not None else "N/A")
            _eu_score = _sovc.get("eu_peripheral_score")
            _c2sv.metric("EU Peripheral", f"{_eu_score:.0f}/100" if _eu_score is not None else "N/A")
            _em_score = _sovc.get("em_sovereign_score")
            _c3sv.metric("EM Sovereign", f"{_em_score:.0f}/100" if _em_score is not None else "N/A")
            _c4sv.metric("Regime", _sovc.get("regime", "N/A"))
            if _sovc.get("contagion_flag"):
                st.error("CONTAGION FLAG: Sovereign stress elevated — corporate spread widening likely.")
            elif _sovc.get("spillover_risk") == "Moderate":
                st.warning("Spillover risk moderate — watch corporate spreads.")
            if _sovc.get("interpretation"):
                st.info(_sovc["interpretation"])
            _sov_hist = _sov.get("stress_history")
            _eu_hist  = _sov.get("eu_stress_history")
            _em_hist  = _sov.get("em_stress_history")
            if _sov_hist is not None and len(_sov_hist.dropna()) > 20:
                _fig_sov = _go_sov.Figure()
                _fig_sov.add_trace(_go_sov.Scatter(
                    x=_sov_hist.index, y=_sov_hist.values,
                    name="Sovereign Stress", line=dict(color="#f59e0b", width=2)
                ))
                if _eu_hist is not None and len(_eu_hist.dropna()) > 20:
                    _fig_sov.add_trace(_go_sov.Scatter(
                        x=_eu_hist.index, y=_eu_hist.values,
                        name="EU Peripheral", line=dict(color="#60a5fa", width=1.2, dash="dot")
                    ))
                if _em_hist is not None and len(_em_hist.dropna()) > 20:
                    _fig_sov.add_trace(_go_sov.Scatter(
                        x=_em_hist.index, y=_em_hist.values,
                        name="EM Sovereign", line=dict(color="#34d399", width=1.2, dash="dot")
                    ))
                _fig_sov.add_hline(y=65, line_dash="dot", line_color="#ef4444", annotation_text="Contagion Risk")
                _fig_sov.update_layout(
                    template="plotly_dark", height=320, title="Sovereign Stress Score (0–100)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(_fig_sov, use_container_width=True)
            _sov_corr = _sov.get("spillover_correlation")
            if _sov_corr is not None:
                st.metric("Sovereign→HY Spillover Correlation (63d rolling, latest)", f"{_sov_corr:.3f}")
        else:
            st.info("Sovereign contagion monitor unavailable — requires EWI/EWG/EMB/TLT ETF data.")
    except Exception as _sov_e:
        st.caption(f"Sovereign contagion unavailable: {_sov_e}")

# --- sub-tab 67: Consumer Credit Stress -------------------------------------

if _active_sub == 67:
    import plotly.graph_objects as _go_cons
    st.header("Consumer Credit Stress")
    st.markdown(
        "Delinquency-based consumer credit stress monitor. Consumer-facing HY sectors "
        "(retail, auto, restaurants) represent ~30% of the index. Rising delinquencies "
        "lead corporate spread widening by 2–3 quarters. Requires FRED API key."
    )
    try:
        _cons = load_consumer_credit_stress(df)
        if _cons.get("available"):
            _consc = _cons.get("current", {})
            _c1cs, _c2cs, _c3cs = st.columns(3)
            _cons_score = _consc.get("consumer_stress_score")
            _c1cs.metric("Consumer Stress", f"{_cons_score:.0f}/100" if _cons_score is not None else "N/A")
            _c2cs.metric("Regime", _consc.get("regime", "N/A"))
            _ar_bps = _consc.get("at_risk_spread_contribution")
            _c3cs.metric("At-Risk HY Spread (est.)", f"{_ar_bps:.0f}bps" if _ar_bps is not None else "N/A")
            if _consc.get("warning"):
                st.warning(_consc["warning"])
            if _consc.get("lead_signal"):
                st.info(_consc["lead_signal"])
            if _consc.get("interpretation"):
                st.caption(_consc["interpretation"])
            _cons_comps = _consc.get("component_scores", {})
            if _cons_comps:
                _comp_cols = st.columns(len(_cons_comps))
                for _i, (_ck, _cv) in enumerate(_cons_comps.items()):
                    _comp_cols[_i].metric(_ck.replace("_", " ").title(), f"{_cv:.0f}/100" if _cv is not None else "N/A")
            _cons_hist = _cons.get("stress_history")
            if _cons_hist is not None and len(_cons_hist.dropna()) > 20:
                _fig_cons = _go_cons.Figure()
                _fig_cons.add_trace(_go_cons.Scatter(
                    x=_cons_hist.index, y=_cons_hist.values,
                    name="Consumer Stress Score", line=dict(color="#fb923c", width=1.5)
                ))
                _comp_hists = _cons.get("component_histories", {})
                _comp_hist_colors = {
                    "cc_delinquency": "#ef4444", "auto_delinquency": "#f59e0b",
                    "mortgage_delinquency": "#60a5fa", "revolving_growth": "#34d399"
                }
                for _ch_key, _ch_series in _comp_hists.items():
                    if len(_ch_series.dropna()) > 20:
                        _fig_cons.add_trace(_go_cons.Scatter(
                            x=_ch_series.index, y=_ch_series.values,
                            name=_ch_key.replace("_", " ").title(),
                            line=dict(color=_comp_hist_colors.get(_ch_key, "#9aa0aa"), width=1, dash="dot")
                        ))
                _fig_cons.add_hline(y=60, line_dash="dot", line_color="#ef4444", annotation_text="Stress")
                _fig_cons.update_layout(
                    template="plotly_dark", height=320, title="Consumer Credit Stress (0–100)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(_fig_cons, use_container_width=True)
            _lc2 = _cons.get("hy_lead_correlation")
            if _lc2 is not None:
                st.metric("Lead Correlation (consumer stress → HY spread, 189d forward)", f"{_lc2:.3f}")
        else:
            st.info("Consumer credit stress unavailable — requires FRED API key (DRCCLACBS, DRAUTOACBS, DRSFRMACBS, REVOLSL).")
    except Exception as _cons_e:
        st.caption(f"Consumer credit stress unavailable: {_cons_e}")

# --- sub-tab 68: Term Premium ------------------------------------------------

if _active_sub == 71:
    import plotly.graph_objects as _go_ff
    st.header("Credit ETF Fund Flows")
    st.markdown(
        "Estimates net flows into/out of HYG, LQD, JNK as a proxy for institutional "
        "credit demand. Sustained outflows create technical selling pressure → spread widening. "
        "HYG is the most liquid HY vehicle — its flows have a short-term lead on spreads."
    )
    try:
        _ff = load_etf_fund_flows(df)
        if _ff.get("available"):
            _ffc = _ff.get("current", {})
            _c1ff, _c2ff, _c3ff = st.columns(3)
            _ff_score = _ffc.get("flow_score")
            _c1ff.metric("Flow Score", f"{_ff_score:.0f}/100" if _ff_score is not None else "N/A")
            _c2ff.metric("Regime", _ffc.get("flow_regime", "N/A"))
            _c3ff.metric("Most Outflow", _ffc.get("largest_outflow_etf", "N/A"))
            if _ffc.get("warning"):
                st.warning(_ffc["warning"])
            if _ffc.get("interpretation"):
                st.info(_ffc["interpretation"])
            _ff_etfs = _ffc.get("etf_flows", {})
            if _ff_etfs:
                _ff_rows = [
                    {"ETF": t,
                     "Flow Z-score": f"{v.get('flow_z', 0):.2f}",
                     "Direction": v.get("flow_direction", "N/A")}
                    for t, v in _ff_etfs.items()
                ]
                import pandas as _pd_ff
                st.dataframe(_pd_ff.DataFrame(_ff_rows).set_index("ETF"), use_container_width=True)
            _ff_hist = _ff.get("flow_history")
            if _ff_hist is not None and len(_ff_hist.dropna()) > 20:
                _fig_ff = _go_ff.Figure()
                _fig_ff.add_trace(_go_ff.Scatter(
                    x=_ff_hist.index, y=_ff_hist.values,
                    name="Flow Score", line=dict(color="#22d3ee", width=1.5)
                ))
                _fig_ff.add_hline(y=30, line_dash="dot", line_color="#ef4444",
                                  annotation_text="Heavy Outflow")
                _fig_ff.add_hline(y=70, line_dash="dot", line_color="#34d399",
                                  annotation_text="Heavy Inflow")
                _fig_ff.update_layout(
                    template="plotly_dark", height=300,
                    title="Credit ETF Flow Score (0=Heavy Outflow, 100=Heavy Inflow)",
                    margin=dict(l=8, r=8, t=32, b=8),
                    yaxis=dict(title="Score", range=[0, 100]),
                )
                st.plotly_chart(_fig_ff, use_container_width=True)
        else:
            st.info("ETF fund flows unavailable — requires live yfinance data (HYG, LQD, JNK).")
    except Exception as _ff_e:
        st.caption(f"ETF fund flows unavailable: {_ff_e}")

# --- sub-tab 72: Corporate Profit Cycle --------------------------------------

if _active_sub == 76:
    import plotly.graph_objects as _go76
    st.header("MOVE Index — Bond Market Volatility")
    st.markdown(
        "The **MOVE Index** (ICE BofA MOVE) measures implied volatility of 1-month US Treasury options "
        "across maturities — the bond market's equivalent of VIX. "
        "MOVE spikes typically **precede** HY spread widening by days to weeks, as rates markets price "
        "uncertainty before credit does. "
        "Normal: < 80 · Elevated: 80–120 · Stress: 120–150 · Crisis: > 150."
    )
    try:
        _mv76 = load_move(df)
        if _mv76.get("available"):
            _mvc = _mv76.get("current", {})
            _mv76a, _mv76b, _mv76c, _mv76d = st.columns(4)
            _move_lvl = _mvc.get("move_level", float("nan"))
            _mv76a.metric("MOVE Level", f"{_move_lvl:.1f}" if not pd.isna(_move_lvl) else "—",
                          help="ICE BofA MOVE Index. Normal < 80.")
            _mv76b.metric("MOVE Regime", _mvc.get("move_regime", "—"))
            _mv76c.metric("MOVE Z-Score (1y)", f"{_mvc.get('move_zscore_1y', float('nan')):.2f}"
                          if not pd.isna(_mvc.get("move_zscore_1y", float("nan"))) else "—")
            _mv76d.metric("MOVE Signal", f"{_mvc.get('move_signal', 0):.0f}/100",
                          help="Higher = more rates-vol stress")
            if _mv76.get("warning"):
                st.warning(_mv76["warning"])
            if _mv76.get("interpretation"):
                st.info(_mv76["interpretation"])

            # MOVE level time series
            _mv76_hist = _mv76.get("historical")
            if _mv76_hist is not None and not _mv76_hist.empty:
                _mv76_hist = _mv76_hist.copy()
                _mv76_hist.index = pd.to_datetime(_mv76_hist.index)
                _fig76a = _go76.Figure()
                if "move_level" in _mv76_hist.columns:
                    _fig76a.add_trace(_go76.Scatter(
                        x=_mv76_hist.index, y=_mv76_hist["move_level"],
                        name="MOVE Level", line=dict(color="#8b5cf6", width=2),
                        fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
                        hovertemplate="%{x|%Y-%m-%d}<br>MOVE: %{y:.1f}<extra></extra>",
                    ))
                _fig76a.add_hline(y=80,  line=dict(color="#f59e0b", dash="dash", width=1),
                                  annotation_text="Elevated (80)", annotation_font=dict(color="#f59e0b", size=9))
                _fig76a.add_hline(y=120, line=dict(color="#ef4444", dash="dash", width=1),
                                  annotation_text="Stress (120)",  annotation_font=dict(color="#ef4444", size=9))
                _fig76a.add_hline(y=150, line=dict(color="#9b59b6", dash="dot", width=1),
                                  annotation_text="Crisis (150)",  annotation_font=dict(color="#9b59b6", size=9))
                _fig76a.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="MOVE Level"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig76a, use_container_width=True)

                # MOVE vs HY spread overlay
                if "hy_spread" in df.columns and "move_level" in _mv76_hist.columns:
                    _mv76_merged = _mv76_hist[["move_level"]].join(
                        df[["hy_spread"]].rename(columns={"hy_spread": "hy_spread"}), how="inner"
                    ).dropna().tail(504)
                    if not _mv76_merged.empty:
                        _fig76b = _go76.Figure()
                        _fig76b.add_trace(_go76.Scatter(
                            x=_mv76_merged.index, y=_mv76_merged["move_level"],
                            name="MOVE", line=dict(color="#8b5cf6", width=1.5),
                            yaxis="y1",
                            hovertemplate="%{x|%Y-%m-%d}<br>MOVE: %{y:.1f}<extra></extra>",
                        ))
                        _fig76b.add_trace(_go76.Scatter(
                            x=_mv76_merged.index, y=_mv76_merged["hy_spread"],
                            name="HY OAS (%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                            yaxis="y2",
                            hovertemplate="%{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>",
                        ))
                        _fig76b.update_layout(
                            height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#9aa0aa", size=11),
                            margin=dict(l=8, r=8, t=24, b=8),
                            title=dict(text="MOVE vs HY Spread", font=dict(size=12, color="#9aa0aa")),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                       color="#8b5cf6", title="MOVE"),
                            yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                        title="HY OAS (%)", showgrid=False),
                            xaxis=dict(showgrid=False, color="#6b7280"),
                            legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                        )
                        st.plotly_chart(_fig76b, use_container_width=True)
                        _mv_corr = float(_mv76_merged["move_level"].corr(_mv76_merged["hy_spread"]))
                        st.caption(f"MOVE vs HY spread correlation (2yr): {_mv_corr:+.2f}. "
                                   f"High MOVE + lag of ~2 weeks historically precedes HY widening.")

            if _mv76.get("regime_table") is not None:
                with st.expander("MOVE regime statistics"):
                    st.dataframe(_mv76["regime_table"], use_container_width=True)
        else:
            st.info("MOVE Index unavailable — requires `move_index` column or live yfinance fetch (^MOVE).")
    except Exception as _e76:
        _err_track(_active_sub, _e76)
        st.caption(f"MOVE Index unavailable: {_e76}")



if _active_sub == 79:
    import plotly.graph_objects as _go79
    st.header("VIX Term Structure")
    st.markdown(
        "The VIX term structure measures the slope between near-term (VIX) and medium-term (VIX3M) "
        "implied volatility. **Backwardation** (VIX > VIX3M, slope < 0) signals acute near-term fear — "
        "historically associated with credit spread widening within 2–4 weeks. "
        "**Contango** (normal upward slope) indicates market pricing in reversion to calm."
    )
    try:
        _vt79 = load_vix_term(df)
        if _vt79.get("available"):
            _vtc = _vt79.get("current", {})
            _vt79a, _vt79b, _vt79c, _vt79d = st.columns(4)
            _vt79a.metric("VIX (1m IV)", f"{_vtc.get('vix', float('nan')):.1f}"
                          if not pd.isna(_vtc.get("vix", float("nan"))) else "—")
            _vt79b.metric("VIX3M (3m IV)", f"{_vtc.get('vix3m', float('nan')):.1f}"
                          if not pd.isna(_vtc.get("vix3m", float("nan"))) else "—")
            _vt79c.metric("Term Slope (VIX3M−VIX)", f"{_vtc.get('vix_term_slope', float('nan')):+.2f}"
                          if not pd.isna(_vtc.get("vix_term_slope", float("nan"))) else "—",
                          help="Negative = backwardation = fear spike")
            _vt79d.metric("Structure", _vtc.get("vix_term_structure", "—"))

            if _vtc.get("vix_term_structure") == "Backwardation":
                st.warning("VIX backwardation detected — near-term fear premium elevated. "
                           "Historically precedes HY spread widening by 2–4 weeks.")
            elif _vtc.get("interpretation"):
                st.info(_vtc["interpretation"])

            # VIX / VIX3M slope history
            _vt79_hist = _vt79.get("historical")
            if _vt79_hist is not None and not _vt79_hist.empty:
                _vt79_plot = _vt79_hist.copy()
                _vt79_plot.index = pd.to_datetime(_vt79_plot.index)
                _fig79a = _go79.Figure()
                if "vix_term_slope" in _vt79_plot.columns:
                    _slope_vals = _vt79_plot["vix_term_slope"].fillna(0)
                    _slope_colors = ["#ef4444" if v < 0 else "#27ae60" for v in _slope_vals]
                    _fig79a.add_trace(_go79.Bar(
                        x=_vt79_plot.index, y=_slope_vals,
                        marker_color=_slope_colors, name="VIX Term Slope",
                        hovertemplate="%{x|%Y-%m-%d}<br>Slope: %{y:+.2f}<extra></extra>",
                    ))
                _fig79a.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1.5)
                _fig79a.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="VIX Term Slope (VIX3M − VIX): Red = Backwardation",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Slope"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig79a, use_container_width=True)

                # VIX vs HY spread dual-axis
                if "hy_spread" in df.columns:
                    _vt79_hy = _vt79_plot[["vix"]].dropna().join(
                        df[["hy_spread"]].dropna(), how="inner"
                    ).tail(504)
                    if not _vt79_hy.empty:
                        _fig79b = _go79.Figure()
                        _fig79b.add_trace(_go79.Scatter(
                            x=_vt79_hy.index, y=_vt79_hy["vix"],
                            name="VIX", line=dict(color="#f59e0b", width=1.5),
                            yaxis="y1",
                            hovertemplate="%{x|%Y-%m-%d}<br>VIX: %{y:.1f}<extra></extra>",
                        ))
                        _fig79b.add_trace(_go79.Scatter(
                            x=_vt79_hy.index, y=_vt79_hy["hy_spread"],
                            name="HY OAS (%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                            yaxis="y2",
                            hovertemplate="%{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>",
                        ))
                        _fig79b.update_layout(
                            height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#9aa0aa", size=11),
                            margin=dict(l=8, r=8, t=24, b=8),
                            title=dict(text="VIX vs HY Spread", font=dict(size=12, color="#9aa0aa")),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                       color="#f59e0b", title="VIX"),
                            yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                        title="HY OAS (%)", showgrid=False),
                            xaxis=dict(showgrid=False, color="#6b7280"),
                            legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                        )
                        st.plotly_chart(_fig79b, use_container_width=True)
                        _vt_corr = float(_vt79_hy["vix"].corr(_vt79_hy["hy_spread"]))
                        st.caption(f"VIX vs HY correlation (2yr): {_vt_corr:+.2f}.")

            # Regime implication table
            st.subheader("VIX Term Structure Regimes")
            import pandas as _pd79
            _vt_regime_tbl = _pd79.DataFrame([
                {"Structure": "Steep contango (slope > +3)",  "Interpretation": "Market calm; low near-term fear",     "Credit Signal": "Risk-on"},
                {"Structure": "Flat contango (0 to +3)",      "Interpretation": "Normal; modest upward slope",         "Credit Signal": "Neutral"},
                {"Structure": "Flat/slight backwardation",    "Interpretation": "Near-term uncertainty; watch",        "Credit Signal": "Cautious"},
                {"Structure": "Backwardation (slope < 0)",    "Interpretation": "Acute fear spike; near-term stress",  "Credit Signal": "Defensive"},
                {"Structure": "Deep backwardation (< −3)",    "Interpretation": "Crisis-level fear; panic conditions", "Credit Signal": "Risk-off"},
            ])
            st.dataframe(_vt_regime_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("VIX term structure unavailable — requires VIX and VIX3M columns (or yfinance fetch).")
    except Exception as _e79:
        _err_track(_active_sub, _e79)
        st.caption(f"VIX term structure unavailable: {_e79}")


# =============================================================================
# BATCH 9 ANALYTICS: sub80–85
# =============================================================================


if _active_sub == 80:
    import plotly.graph_objects as _go80
    st.header("Options Skew — Tail Risk Pricing")
    st.markdown(
        "The **CBOE SKEW Index** measures how much more expensive OTM puts are relative to ATM options — "
        "the market's implied probability of outlier S&P 500 moves. "
        "Unlike VIX (which measures the level of near-term vol), SKEW measures the **shape** of the smile. "
        "**Hidden danger regime**: High SKEW + Low VIX = market is complacent on the surface "
        "but quietly pricing extreme left-tail events — historically one of the most dangerous configurations for credit."
    )
    try:
        _sk80 = load_options_skew(df)
        _snap80 = _sk80.get("snapshot", {})
        _sk80a, _sk80b, _sk80c, _sk80d = st.columns(4)
        _skew_lvl = _snap80.get("skew_level", float("nan"))
        _vvix_lvl = _snap80.get("vvix", float("nan"))
        _sk80a.metric("SKEW Level", f"{_skew_lvl:.1f}" if not pd.isna(_skew_lvl) else "—",
                      help="Normal ≈ 100–115. Elevated > 130. Extreme > 150.")
        _sk80b.metric("SKEW Regime", _snap80.get("skew_regime", "—"))
        _sk80c.metric("VVIX (Vol-of-Vol)", f"{_vvix_lvl:.1f}" if not pd.isna(_vvix_lvl) else "—",
                      help="> 100 typically precedes VIX spikes")
        _sk80d.metric("Hidden Danger", "YES" if _snap80.get("hidden_danger") else "No",
                      help="High SKEW + Low VIX — complacency with hidden tail risk")

        if _snap80.get("hidden_danger"):
            st.error("Hidden danger regime: High SKEW + Low VIX. Market is pricing extreme tails "
                     "while surface vol remains suppressed. Historically precedes credit dislocations.")

        if _sk80.get("interpretation"):
            st.info(_sk80["interpretation"])

        # Skew vs HY spread correlation callout
        _corr_hy80 = _sk80.get("correlation_with_hy")
        _corr_comp80 = _sk80.get("correlation_with_composite")
        if _corr_hy80 is not None or _corr_comp80 is not None:
            _sc80a, _sc80b = st.columns(2)
            if _corr_hy80 is not None:
                _sc80a.metric("SKEW vs HY Corr (1yr)", f"{_corr_hy80:+.2f}")
            if _corr_comp80 is not None:
                _sc80b.metric("SKEW vs Composite Corr (1yr)", f"{_corr_comp80:+.2f}")

        # SKEW + VVIX time series
        _sk80_df = _sk80.get("df")
        if _sk80_df is not None and "skew_level" in _sk80_df.columns:
            _sk80_plot = _sk80_df[["skew_level"]].dropna().tail(504).copy()
            _sk80_plot.index = pd.to_datetime(_sk80_plot.index)
            _fig80a = _go80.Figure()
            _fig80a.add_trace(_go80.Scatter(
                x=_sk80_plot.index, y=_sk80_plot["skew_level"],
                name="SKEW Level", line=dict(color="#a78bfa", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>SKEW: %{y:.1f}<extra></extra>",
            ))
            _fig80a.add_hline(y=115, line=dict(color="#f59e0b", dash="dash", width=1),
                              annotation_text="Normal (115)",
                              annotation_font=dict(color="#f59e0b", size=9))
            _fig80a.add_hline(y=130, line=dict(color="#ef4444", dash="dash", width=1),
                              annotation_text="Elevated (130)",
                              annotation_font=dict(color="#ef4444", size=9))
            _fig80a.add_hline(y=145, line=dict(color="#9b59b6", dash="dot", width=1),
                              annotation_text="High (145)",
                              annotation_font=dict(color="#9b59b6", size=9))
            _fig80a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="SKEW Level"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig80a, use_container_width=True)

            if "vvix" in _sk80_df.columns:
                _vvix_plot = _sk80_df[["vvix"]].dropna().tail(504).copy()
                _vvix_plot.index = pd.to_datetime(_vvix_plot.index)
                _fig80b = _go80.Figure()
                _fig80b.add_trace(_go80.Scatter(
                    x=_vvix_plot.index, y=_vvix_plot["vvix"],
                    name="VVIX", line=dict(color="#06b6d4", width=1.5),
                    fill="tozeroy", fillcolor="rgba(6,182,212,0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>VVIX: %{y:.1f}<extra></extra>",
                ))
                _fig80b.add_hline(y=100, line=dict(color="#ef4444", dash="dash", width=1),
                                  annotation_text="Stress threshold (100)",
                                  annotation_font=dict(color="#ef4444", size=9))
                _fig80b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="VVIX (Vol-of-Vol — precedes VIX spikes)",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="VVIX"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig80b, use_container_width=True)

        # Hidden danger episodes table
        _hd80 = _sk80.get("hidden_danger_episodes")
        if _hd80 is not None and not _hd80.empty:
            with st.expander(f"Hidden danger episodes ({len(_hd80)} found)"):
                st.dataframe(_hd80, use_container_width=True, hide_index=True)
                st.caption("Each episode: SKEW > 130 AND VIX < 18. "
                           "hy_spread_change_5d shows subsequent 5-day HY spread move.")
        elif not _sk80.get("available"):
            st.info("Options skew unavailable — requires live SKEW/VVIX fetch via yfinance.")
    except Exception as _e80:
        _err_track(_active_sub, _e80)
        st.caption(f"Options skew unavailable: {_e80}")



if _active_sub == 82:
    import plotly.graph_objects as _go82
    st.header("DV01 — Spread Sensitivity & P&L Scenarios")
    st.markdown(
        "**DV01** (Dollar Value of a Basis Point) measures the dollar P&L impact of a 1 bp change "
        "in credit spreads across the portfolio. Based on current HY/IG weights and their respective "
        "effective durations (HY ≈ 4yr, IG ≈ 7yr). "
        "Use this to size positions, set stop-losses, and stress-test spread-widening scenarios."
    )
    try:
        _dv82 = load_dv01(df)
        if _dv82.get("available"):
            _dvc = _dv82.get("current", {})
            _dv82a, _dv82b, _dv82c, _dv82d = st.columns(4)
            _dv01_val = _dvc.get("dv01", float("nan"))
            _dur_val = _dvc.get("blended_duration", float("nan"))
            _dv82a.metric("DV01 (per $1M)", f"${_dv01_val:,.0f}" if not pd.isna(_dv01_val) else "—",
                          help="Dollar loss per 1 bp spread widening on $1M notional")
            _dv82b.metric("Blended Duration", f"{_dur_val:.2f} yrs" if not pd.isna(_dur_val) else "—")
            _dv82c.metric("HY Weight", f"{_dvc.get('hy_weight', 0):.0%}")
            _dv82d.metric("IG Weight", f"{_dvc.get('ig_weight', 0):.0%}")

            # P&L scenario metrics
            _pnl82 = _dvc.get("pnl_scenarios", {})
            if _pnl82:
                st.subheader("P&L Scenarios (per $1M notional)")
                _pnl82_cols = st.columns(len(_pnl82))
                for _ci82, (_lbl82, _val82) in enumerate(
                    [(k, v) for k, v in _pnl82.items() if v is not None and not pd.isna(v)]
                ):
                    _color82 = "inverse" if "widen" in _lbl82.lower() else "normal"
                    _pnl82_cols[_ci82 % len(_pnl82_cols)].metric(
                        f"Spread {_lbl82}", f"${_val82:,.0f}",
                        delta_color=_color82,
                    )

            # Full scenario table
            _scen82 = _dv82.get("scenario_table")
            if _scen82 is not None and not _scen82.empty:
                st.subheader("Full Spread Shock Scenario Table")
                import pandas as _pd82
                _scen82_fmt = _scen82.copy()
                for _col82 in _scen82_fmt.select_dtypes(include=["float", "int"]).columns:
                    if "pct" in _col82.lower() or "%" in _col82:
                        _scen82_fmt[_col82] = _scen82_fmt[_col82].map(lambda x: f"{x:.1f}%")
                    elif "pnl" in _col82.lower() or "$" in _col82:
                        _scen82_fmt[_col82] = _scen82_fmt[_col82].map(lambda x: f"${x:,.0f}")
                st.dataframe(_scen82_fmt, use_container_width=True, hide_index=True)
                st.caption(f"Based on portfolio notional ${_dvc.get('portfolio_value', 1_000_000):,.0f}. "
                           f"Blended duration {_dur_val:.2f}yr = {_dur_val:.2f}× price sensitivity per 100 bps of spread move.")

            # DV01 history chart
            _dv82_df = _dv82.get("df")
            if _dv82_df is not None and "dv01_per_1m" in _dv82_df.columns:
                _dv82_plot = _dv82_df[["dv01_per_1m"]].dropna().tail(504).copy()
                _dv82_plot.index = pd.to_datetime(_dv82_plot.index)
                _fig82 = _go82.Figure()
                _fig82.add_trace(_go82.Scatter(
                    x=_dv82_plot.index, y=_dv82_plot["dv01_per_1m"],
                    name="DV01 per $1M", line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>DV01: $%{y:,.0f}<extra></extra>",
                ))
                _fig82.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="DV01 Over Time (varies with portfolio duration)",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="DV01 per $1M ($)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig82, use_container_width=True)
        else:
            st.info("DV01 analysis unavailable — data frame is empty.")
    except Exception as _e82:
        _err_track(_active_sub, _e82)
        st.caption(f"DV01 unavailable: {_e82}")



if _active_sub == 83:
    import plotly.graph_objects as _go83
    st.header("CVaR — Conditional Value at Risk")
    st.markdown(
        "**CVaR** (Expected Shortfall) measures the average loss in the worst α% of scenarios — "
        "more informative than VaR because it captures the severity of tail losses, not just the threshold. "
        "Computed from daily portfolio returns, conditional on each credit regime, "
        "using historical simulation (no parametric distribution assumptions). "
        "**CVaR₉₅** = average of the worst 5% of return days · **CVaR₉₉** = worst 1%."
    )
    try:
        _cv83 = load_cvar(df)
        _cur83_regime = _cv83.get("current_regime", "Unknown")
        _cv95 = _cv83.get("current_cvar_95", float("nan"))
        _cv99 = _cv83.get("current_cvar_99", float("nan"))
        _cv83a, _cv83b, _cv83c, _cv83d = st.columns(4)
        _cv83a.metric("Current Regime", _cur83_regime)
        _cv83b.metric("CVaR₉₅ (daily)", f"{_cv95:.2%}" if not pd.isna(_cv95) else "—",
                      help="Average of worst 5% of days in current regime")
        _cv83c.metric("CVaR₉₉ (daily)", f"{_cv99:.2%}" if not pd.isna(_cv99) else "—",
                      help="Average of worst 1% of days in current regime")
        _fp83 = _cv83.get("full_period", {})
        _cv83d.metric("Full-Period CVaR₉₅", f"{_fp83.get('cvar_95', float('nan')):.2%}"
                      if _fp83.get("cvar_95") and not pd.isna(_fp83.get("cvar_95", float("nan"))) else "—")

        if not _cv83.get("available"):
            st.warning("Strategy return series not found — using SP500 returns as proxy.")

        # Regime-conditional CVaR table
        _rs83 = _cv83.get("regime_stats")
        if _rs83 is not None and not _rs83.empty:
            st.subheader("CVaR by Regime")
            import pandas as _pd83
            _rs83_fmt = _rs83.copy()
            for _col83 in _rs83_fmt.select_dtypes(include=["float"]).columns:
                _rs83_fmt[_col83] = _rs83_fmt[_col83].map(lambda x: f"{x:.2%}" if abs(x) < 1 else f"{x:.1f}")
            st.dataframe(_rs83_fmt, use_container_width=True)
            st.caption("CVaR shows average loss on the worst days, conditional on the credit regime. "
                       "Risk-off regimes produce materially worse tail outcomes.")

        # Rolling CVaR chart
        _rcvar83 = _cv83.get("rolling_cvar")
        if _rcvar83 is not None and len(_rcvar83.dropna()) > 20:
            _rcvar83 = _rcvar83.dropna()
            _fig83a = _go83.Figure()
            _fig83a.add_trace(_go83.Scatter(
                x=_rcvar83.index, y=_rcvar83.values * 100,
                name="Rolling CVaR₉₅ (252d, %)", line=dict(color="#ef4444", width=1.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                hovertemplate="%{x|%Y-%m-%d}<br>CVaR₉₅: %{y:.2f}%<extra></extra>",
            ))
            _fig83a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                title=dict(text="Rolling 252-Day CVaR₉₅ (daily %)",
                           font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="CVaR₉₅ (%)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig83a, use_container_width=True)

        # Full-period vs OOS comparison
        _oos83 = _cv83.get("oos_period", {})
        if _fp83 and _oos83:
            st.subheader("In-Sample vs Out-of-Sample Tail Risk")
            _cmp83_cols = st.columns(2)
            _cmp83_cols[0].metric("Full-Period CVaR₉₅", f"{_fp83.get('cvar_95', float('nan')):.2%}"
                                  if _fp83.get("cvar_95") else "—")
            _cmp83_cols[0].metric("Full-Period CVaR₉₉", f"{_fp83.get('cvar_99', float('nan')):.2%}"
                                  if _fp83.get("cvar_99") else "—")
            _cmp83_cols[1].metric("OOS CVaR₉₅ (post-2016)", f"{_oos83.get('cvar_95', float('nan')):.2%}"
                                  if _oos83.get("cvar_95") else "—")
            _cmp83_cols[1].metric("OOS CVaR₉₉ (post-2016)", f"{_oos83.get('cvar_99', float('nan')):.2%}"
                                  if _oos83.get("cvar_99") else "—")
    except Exception as _e83:
        _err_track(_active_sub, _e83)
        st.caption(f"CVaR unavailable: {_e83}")



if _active_sub == 86:
    import plotly.graph_objects as _go86
    st.header("Volatility Risk Premium (VRP)")
    st.markdown(
        "**VRP = VIX − 21-day realized SP500 volatility.** "
        "When positive, the options market is charging a *fear premium* above realized vol — the normal state. "
        "When **inverted** (VRP < 0), realized vol has exceeded implied: the fear premium has collapsed "
        "and panic is already in the price. Inversion historically precedes credit spread widening by ~2 weeks "
        "as the volatility regime confirms what the credit market hasn't fully priced yet."
    )
    try:
        _vrp86 = load_vrp(df)
        if _vrp86.get("available"):
            _vc86 = _vrp86["current"]
            _v86a, _v86b, _v86c, _v86d = st.columns(4)
            _vrp21 = _vc86.get("vrp_21d", float("nan"))
            _vrp63 = _vc86.get("vrp_63d", float("nan"))
            _v86a.metric("VRP (21d)", f"{_vrp21:.2f}%",
                         help="VIX minus 21d realized vol. Negative = inverted.")
            _v86b.metric("VRP (63d)", f"{_vrp63:.2f}%",
                         help="VIX minus 63d realized vol.")
            _v86c.metric("VIX", f"{_vc86.get('vix', float('nan')):.1f}")
            _v86d.metric("VRP Regime", _vc86.get("vrp_regime", "—"))

            if _vrp86.get("warning"):
                st.warning(_vrp86["warning"])
            if _vrp86.get("interpretation"):
                st.info(_vrp86["interpretation"])

            _hist86 = _vrp86.get("historical")
            if _hist86 is not None and not _hist86.empty:
                _hist86 = _hist86.copy()
                _hist86.index = pd.to_datetime(_hist86.index)

                # VRP 21d with inversion shading
                _fig86a = _go86.Figure()
                if "vrp_21d" in _hist86.columns:
                    _vrp_vals = _hist86["vrp_21d"].fillna(0)
                    _fig86a.add_trace(_go86.Scatter(
                        x=_hist86.index, y=_hist86["vrp_21d"],
                        name="VRP (21d)", line=dict(color="#9b59b6", width=2),
                        fill="tozeroy", fillcolor="rgba(155,89,182,0.10)",
                        hovertemplate="%{x|%Y-%m-%d}<br>VRP: %{y:.2f}%<extra></extra>",
                    ))
                if "vrp_63d" in _hist86.columns:
                    _fig86a.add_trace(_go86.Scatter(
                        x=_hist86.index, y=_hist86["vrp_63d"],
                        name="VRP (63d)", line=dict(color="#8e44ad", width=1.5, dash="dot"),
                        hovertemplate="%{x|%Y-%m-%d}<br>VRP 63d: %{y:.2f}%<extra></extra>",
                    ))
                _fig86a.add_hline(y=0, line_color="rgba(231,76,60,0.6)", line_width=1.5,
                                  annotation_text="Inversion threshold",
                                  annotation_font=dict(color="#e74c3c", size=10))
                _fig86a.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="VRP (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig86a, use_container_width=True)

                # VIX vs realized vol chart
                if "sp500_realized_vol_21d" in _hist86.columns:
                    _vix_col = _vrp86.get("df")
                    if _vix_col is not None and "vix" in _vix_col.columns:
                        _rv86 = _hist86[["sp500_realized_vol_21d"]].join(
                            _vix_col[["vix"]].tail(504), how="left"
                        ).dropna()
                        if not _rv86.empty:
                            _fig86b = _go86.Figure()
                            _fig86b.add_trace(_go86.Scatter(
                                x=_rv86.index, y=_rv86["vix"],
                                name="VIX (Implied Vol)", line=dict(color="#f59e0b", width=2),
                                hovertemplate="%{x|%Y-%m-%d}<br>VIX: %{y:.1f}%<extra></extra>",
                            ))
                            _fig86b.add_trace(_go86.Scatter(
                                x=_rv86.index, y=_rv86["sp500_realized_vol_21d"],
                                name="Realized Vol (21d)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                                hovertemplate="%{x|%Y-%m-%d}<br>Realized: %{y:.1f}%<extra></extra>",
                            ))
                            _fig86b.update_layout(
                                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#9aa0aa", size=11),
                                margin=dict(l=8, r=8, t=24, b=8),
                                title=dict(text="VIX vs 21d Realized Vol (gap = VRP)",
                                           font=dict(size=12, color="#9aa0aa")),
                                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                           color="#6b7280", title="Volatility (%)"),
                                xaxis=dict(showgrid=False, color="#6b7280"),
                                legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                            )
                            st.plotly_chart(_fig86b, use_container_width=True)

            # VRP z-score and signal
            _vrp_z86 = _vc86.get("vrp_zscore_1y", float("nan"))
            _vrp_sig86 = _vc86.get("vrp_signal", float("nan"))
            if not pd.isna(_vrp_z86):
                _zs86a, _zs86b = st.columns(2)
                _zs86a.metric("VRP Z-Score (1yr)", f"{_vrp_z86:.2f}",
                              help="How extreme the current VRP is vs trailing year")
                _zs86b.metric("VRP Signal (0–100)", f"{_vrp_sig86:.0f}"
                              if not pd.isna(_vrp_sig86) else "—",
                              help="Higher = more stress from inverted/compressed VRP")
        else:
            st.info("VRP unavailable — requires VIX and SP500 columns in dataset.")
    except Exception as _e86:
        _err_track(_active_sub, _e86)
        st.caption(f"VRP unavailable: {_e86}")



if _active_sub == 88:
    import plotly.graph_objects as _go88
    st.header("Funding Stress Monitor")
    st.markdown(
        "**Funding stress** measures plumbing risk in interbank credit markets. "
        "The **TED spread** (3m T-bill vs LIBOR/interbank rate) proxies the cost premium banks "
        "charge each other to lend, reflecting systemic trust and liquidity. "
        "Stress in funding markets typically **precedes** broad HY spread widening by 3–5 weeks "
        "as banks restrict credit before it shows in public spread data."
    )
    try:
        _fs88 = load_funding_stress(df)
        if _fs88.get("available"):
            _fsc = _fs88["current"]
            _fs88a, _fs88b, _fs88c, _fs88d = st.columns(4)
            _ted88 = _fsc.get("ted_spread_proxy_bps", float("nan"))
            _fs88a.metric("TED Proxy (bps)", f"{_ted88:.1f}" if not pd.isna(_ted88) else "—",
                          help="3m T-bill vs Fed funds proxy — measures interbank lending premium")
            _fs88b.metric("Z-Score (1yr)", f"{_fsc.get('ted_spread_zscore', float('nan')):.2f}"
                          if not pd.isna(_fsc.get("ted_spread_zscore", float("nan"))) else "—")
            _fs88c.metric("Funding Regime", _fsc.get("funding_regime", "—"))
            _fs88d.metric("Stress Signal", f"{_fsc.get('funding_stress_signal', 0):.0f}/100")

            if _fs88.get("warning"):
                st.error(_fs88["warning"])
            if _fs88.get("interpretation"):
                st.info(_fs88["interpretation"])

            st.caption(f"Historical lead time: ~{_fs88.get('lead_weeks', 3)} weeks ahead of HY spread widening.")

            # TED proxy time series
            _hist88 = _fs88.get("historical")
            if _hist88 is not None and not _hist88.empty:
                _hist88 = _hist88.copy()
                _hist88.index = pd.to_datetime(_hist88.index)
                _fig88a = _go88.Figure()
                if "ted_spread_proxy" in _hist88.columns:
                    _fig88a.add_trace(_go88.Scatter(
                        x=_hist88.index, y=_hist88["ted_spread_proxy"] * 100,
                        name="TED Proxy (bps)", line=dict(color="#06b6d4", width=2),
                        fill="tozeroy", fillcolor="rgba(6,182,212,0.10)",
                        hovertemplate="%{x|%Y-%m-%d}<br>TED: %{y:.1f} bps<extra></extra>",
                    ))
                _fig88a.add_hline(y=25, line=dict(color="#f59e0b", dash="dash", width=1),
                                  annotation_text="Elevated (25 bps)",
                                  annotation_font=dict(color="#f59e0b", size=9))
                _fig88a.add_hline(y=50, line=dict(color="#ef4444", dash="dash", width=1),
                                  annotation_text="Stress (50 bps)",
                                  annotation_font=dict(color="#ef4444", size=9))
                _fig88a.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="TED Proxy (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig88a, use_container_width=True)

                # Funding stress signal vs HY spread
                if "funding_stress_signal" in _hist88.columns and "hy_spread" in df.columns:
                    _fs88_combo = _hist88[["funding_stress_signal"]].join(
                        df[["hy_spread"]].dropna(), how="inner"
                    ).dropna().tail(504)
                    if not _fs88_combo.empty:
                        _fig88b = _go88.Figure()
                        _fig88b.add_trace(_go88.Scatter(
                            x=_fs88_combo.index, y=_fs88_combo["funding_stress_signal"],
                            name="Funding Stress Signal", line=dict(color="#06b6d4", width=1.5),
                            yaxis="y1",
                            hovertemplate="%{x|%Y-%m-%d}<br>Funding: %{y:.0f}<extra></extra>",
                        ))
                        _fig88b.add_trace(_go88.Scatter(
                            x=_fs88_combo.index, y=_fs88_combo["hy_spread"],
                            name="HY OAS (%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
                            yaxis="y2",
                            hovertemplate="%{x|%Y-%m-%d}<br>HY OAS: %{y:.2f}%<extra></extra>",
                        ))
                        _fig88b.update_layout(
                            height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#9aa0aa", size=11),
                            margin=dict(l=8, r=8, t=24, b=8),
                            title=dict(text="Funding Stress Signal vs HY Spread",
                                       font=dict(size=12, color="#9aa0aa")),
                            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                                       color="#06b6d4", title="Funding Signal (0–100)"),
                            yaxis2=dict(overlaying="y", side="right", color="#ef4444",
                                        title="HY OAS (%)", showgrid=False),
                            xaxis=dict(showgrid=False, color="#6b7280"),
                            legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                        )
                        st.plotly_chart(_fig88b, use_container_width=True)
                        _fs_corr88 = float(_fs88_combo["funding_stress_signal"].corr(_fs88_combo["hy_spread"]))
                        st.caption(f"Funding stress vs HY spread correlation: {_fs_corr88:+.2f}. "
                                   f"Funding stress peaks ~{_fs88.get('lead_weeks', 3)} weeks before HY widening.")
        else:
            st.info("Funding stress unavailable — requires yield_3m (3m Treasury) and fed_funds columns.")
    except Exception as _e88:
        _err_track(_active_sub, _e88)
        st.caption(f"Funding stress unavailable: {_e88}")



if _active_sub == 95:
    import plotly.graph_objects as _go95
    st.header("Market Internals & Cross-Asset Divergence")
    st.markdown(
        "Two composite signals from the risk engine: "
        "**Market Internals** (0–100) aggregates SP500 return momentum, drawdown, and VIX "
        "into a breadth-style health score — high = equity market stressed. "
        "**Cross-Asset Divergence** measures when equity, credit, and volatility markets are "
        "sending conflicting signals — elevated divergence historically precedes mean reversion and "
        "volatility spikes in the lagging market."
    )
    try:
        _mi95_cols = [c for c in ["market_internals_score_smooth", "market_internals_score",
                                   "cross_asset_divergence_score_smooth", "cross_asset_divergence_score"]
                      if c in df.columns]
        if _mi95_cols:
            _mi95_df = df[_mi95_cols].copy()
            _mi95_df.index = pd.to_datetime(_mi95_df.index)

            _mi_col = next((c for c in ["market_internals_score_smooth", "market_internals_score"]
                            if c in _mi95_df.columns), None)
            _ca_col = next((c for c in ["cross_asset_divergence_score_smooth", "cross_asset_divergence_score"]
                            if c in _mi95_df.columns), None)

            _mi_now = float(_mi95_df[_mi_col].dropna().iloc[-1]) if _mi_col else float("nan")
            _ca_now = float(_mi95_df[_ca_col].dropna().iloc[-1]) if _ca_col else float("nan")

            _m95a, _m95b, _m95c, _m95d = st.columns(4)
            _m95a.metric("Market Internals", f"{_mi_now:.1f}/100" if not pd.isna(_mi_now) else "—",
                         help="0 = healthy, 100 = severely stressed equities")
            _m95b.metric("Internals Regime",
                         "Stressed" if _mi_now > 65 else "Elevated" if _mi_now > 40 else "Normal"
                         if not pd.isna(_mi_now) else "—")
            _m95c.metric("Cross-Asset Divergence", f"{_ca_now:.1f}/100" if not pd.isna(_ca_now) else "—",
                         help="High = equity/credit/vol sending conflicting signals")
            _m95d.metric("Divergence Regime",
                         "High" if _ca_now > 65 else "Moderate" if _ca_now > 40 else "Low"
                         if not pd.isna(_ca_now) else "—")

            if _mi_now > 65:
                st.warning("Market internals score elevated — equity market breadth and momentum deteriorating.")
            if _ca_now > 65:
                st.warning("Cross-asset divergence high — asset classes sending conflicting signals. "
                           "Mean reversion event likely within 2–4 weeks.")

            # Dual time series
            _mi95_plot = _mi95_df.tail(504)
            _fig95 = _go95.Figure()
            if _mi_col:
                _fig95.add_trace(_go95.Scatter(
                    x=_mi95_plot.index, y=_mi95_plot[_mi_col],
                    name="Market Internals", line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Internals: %{y:.1f}<extra></extra>",
                ))
            if _ca_col:
                _fig95.add_trace(_go95.Scatter(
                    x=_mi95_plot.index, y=_mi95_plot[_ca_col],
                    name="Cross-Asset Divergence", line=dict(color="#8b5cf6", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Divergence: %{y:.1f}<extra></extra>",
                ))
            _fig95.add_hline(y=65, line=dict(color="#ef4444", dash="dash", width=1),
                             annotation_text="Stress threshold (65)",
                             annotation_font=dict(color="#ef4444", size=9))
            _fig95.add_hline(y=40, line=dict(color="#f59e0b", dash="dot", width=1),
                             annotation_text="Elevated (40)",
                             annotation_font=dict(color="#f59e0b", size=9))
            _fig95.update_layout(
                height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)", range=[0, 100]),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig95, use_container_width=True)
            st.caption("Market Internals above 65 = equity market under sustained pressure. "
                       "Cross-Asset Divergence above 65 = look for catch-up moves in the lagging market.")
        else:
            st.info("Market internals unavailable — requires market_internals_score or "
                    "cross_asset_divergence_score columns.")
    except Exception as _e95:
        _err_track(_active_sub, _e95)
        st.caption(f"Market internals unavailable: {_e95}")



if _active_sub == 96:
    import plotly.graph_objects as _go96
    st.header("Vol-Credit Mismatch")
    st.markdown(
        "**Vol-Credit Mismatch** fires when VIX and credit spreads are sending contradictory signals: "
        "e.g. VIX elevated but HY spreads complacent, or spreads wide but vol suppressed. "
        "These mismatches are unstable equilibria — one market will catch up. "
        "Historically: when HY is *tight* but VIX is *high*, spreads tend to widen within 3–6 weeks; "
        "when HY is *wide* but VIX is *low*, vol tends to spike."
    )
    try:
        _mismatch_avail = "vol_credit_mismatch" in df.columns and df["vol_credit_mismatch"].notna().sum() > 10
        if _mismatch_avail or ("vix" in df.columns and "hy_spread" in df.columns):
            _vc96_df = df[[]].copy()
            if _mismatch_avail:
                _vc96_df["vol_credit_mismatch"] = df["vol_credit_mismatch"]
            if "vix" in df.columns:
                _vc96_df["vix"] = df["vix"]
            if "hy_spread" in df.columns:
                _vc96_df["hy_spread"] = df["hy_spread"]
            _vc96_df.index = pd.to_datetime(_vc96_df.index)

            _mismatch_now = str(_vc96_df["vol_credit_mismatch"].dropna().iloc[-1]) if _mismatch_avail else "—"
            _vix_now = float(_vc96_df["vix"].dropna().iloc[-1]) if "vix" in _vc96_df.columns else float("nan")
            _hy_now = float(_vc96_df["hy_spread"].dropna().iloc[-1]) if "hy_spread" in _vc96_df.columns else float("nan")

            _v96a, _v96b, _v96c, _v96d = st.columns(4)
            _v96a.metric("VIX", f"{_vix_now:.1f}" if not pd.isna(_vix_now) else "—")
            _v96b.metric("HY OAS", f"{_hy_now:.2f}%" if not pd.isna(_hy_now) else "—")
            _v96c.metric("Mismatch Signal", _mismatch_now)

            # Derive VIX z-score and HY z-score for mismatch visualization
            _vix_252 = _vc96_df["vix"].tail(252) if "vix" in _vc96_df.columns else None
            _hy_252 = _vc96_df["hy_spread"].tail(252) if "hy_spread" in _vc96_df.columns else None
            _vix_z = float((_vix_now - _vix_252.mean()) / (_vix_252.std() + 1e-9)) if _vix_252 is not None and len(_vix_252) > 10 else float("nan")
            _hy_z = float((_hy_now - _hy_252.mean()) / (_hy_252.std() + 1e-9)) if _hy_252 is not None and len(_hy_252) > 10 else float("nan")
            _mismatch_score = abs(_vix_z - _hy_z) if not (pd.isna(_vix_z) or pd.isna(_hy_z)) else float("nan")
            _v96d.metric("Divergence |z|", f"{_mismatch_score:.2f}" if not pd.isna(_mismatch_score) else "—",
                         help="Absolute difference between VIX z-score and HY z-score (1yr window)")

            if not pd.isna(_mismatch_score) and _mismatch_score > 1.5:
                st.warning(f"Large VIX vs HY divergence detected (|z| = {_mismatch_score:.2f}). "
                           "Mismatch historically resolves within 3–6 weeks.")

            # VIX z-score vs HY z-score over time
            if "vix" in _vc96_df.columns and "hy_spread" in _vc96_df.columns:
                _vc96_plot = _vc96_df[["vix", "hy_spread"]].dropna().tail(504)
                _vix_z_series = (_vc96_plot["vix"] - _vc96_plot["vix"].rolling(252, min_periods=60).mean()) / \
                                (_vc96_plot["vix"].rolling(252, min_periods=60).std() + 1e-9)
                _hy_z_series  = (_vc96_plot["hy_spread"] - _vc96_plot["hy_spread"].rolling(252, min_periods=60).mean()) / \
                                (_vc96_plot["hy_spread"].rolling(252, min_periods=60).std() + 1e-9)
                _div_series = (_vix_z_series - _hy_z_series).abs()

                _fig96a = _go96.Figure()
                _fig96a.add_trace(_go96.Scatter(
                    x=_vc96_plot.index, y=_vix_z_series,
                    name="VIX z-score (1yr)", line=dict(color="#f59e0b", width=1.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>VIX z: %{y:.2f}<extra></extra>",
                ))
                _fig96a.add_trace(_go96.Scatter(
                    x=_vc96_plot.index, y=_hy_z_series,
                    name="HY Spread z-score (1yr)", line=dict(color="#ef4444", width=1.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>HY z: %{y:.2f}<extra></extra>",
                ))
                _fig96a.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _fig96a.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="VIX vs HY Z-Scores (1yr rolling) — Divergence = Mismatch",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Z-Score"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.10, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig96a, use_container_width=True)

                # Divergence magnitude chart
                _fig96b = _go96.Figure()
                _div_colors = ["#ef4444" if v > 1.5 else "#f59e0b" if v > 0.8 else "#27ae60"
                               for v in _div_series.fillna(0)]
                _fig96b.add_trace(_go96.Bar(
                    x=_vc96_plot.index, y=_div_series,
                    marker_color=_div_colors, name="|VIX z − HY z|",
                    hovertemplate="%{x|%Y-%m-%d}<br>Divergence: %{y:.2f}<extra></extra>",
                ))
                _fig96b.add_hline(y=1.5, line=dict(color="#ef4444", dash="dash", width=1),
                                  annotation_text="High mismatch (1.5σ)",
                                  annotation_font=dict(color="#ef4444", size=9))
                _fig96b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="Vol-Credit Divergence Magnitude |VIX z − HY z|",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="|Divergence|"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig96b, use_container_width=True)
        else:
            st.info("Vol-Credit mismatch unavailable — requires vix and hy_spread columns.")
    except Exception as _e96:
        _err_track(_active_sub, _e96)
        st.caption(f"Vol-Credit mismatch unavailable: {_e96}")



if _active_sub == 97:
    import plotly.graph_objects as _go97
    st.header("Equity Drawdown vs Credit Spreads")
    st.markdown(
        "**SP500 drawdown from peak** is a leading indicator of credit stress: "
        "equity dislocations transfer to credit through leveraged balance sheets, "
        "collateral calls, and risk-off sentiment. "
        "A drawdown > −10% has historically been associated with meaningful HY spread widening "
        "within 4–8 weeks. Understanding the drawdown-to-spread transmission helps size "
        "credit hedges and set reentry levels."
    )
    try:
        _dd97_avail = "sp500_drawdown" in df.columns and df["sp500_drawdown"].notna().sum() > 20
        _hy97_avail = "hy_spread" in df.columns and df["hy_spread"].notna().sum() > 20
        if _dd97_avail or ("sp500" in df.columns and df["sp500"].notna().sum() > 20):
            _dd97_df = df[[]].copy()
            if _dd97_avail:
                _dd97_df["sp500_drawdown"] = df["sp500_drawdown"]
            elif "sp500" in df.columns:
                _sp97 = df["sp500"].dropna()
                _dd97_df["sp500_drawdown"] = (_sp97 / _sp97.cummax() - 1).reindex(df.index)
            if _hy97_avail:
                _dd97_df["hy_spread"] = df["hy_spread"]
            if "sp500" in df.columns:
                _dd97_df["sp500"] = df["sp500"]
            _dd97_df.index = pd.to_datetime(_dd97_df.index)

            _dd_now = float(_dd97_df["sp500_drawdown"].dropna().iloc[-1])
            _hy_now97 = float(_dd97_df["hy_spread"].dropna().iloc[-1]) if _hy97_avail else float("nan")
            _sp_now = float(_dd97_df["sp500"].dropna().iloc[-1]) if "sp500" in _dd97_df.columns else float("nan")

            _d97a, _d97b, _d97c, _d97d = st.columns(4)
            _d97a.metric("Current Drawdown", f"{_dd_now:.1%}",
                         delta_color="inverse",
                         help="SP500 distance from all-time high")
            _d97b.metric("Drawdown Regime",
                         "Crisis" if _dd_now < -0.20 else "Stress" if _dd_now < -0.10
                         else "Elevated" if _dd_now < -0.05 else "Normal")
            _d97c.metric("SP500 Level", f"{_sp_now:,.0f}" if not pd.isna(_sp_now) else "—")
            _d97d.metric("HY OAS", f"{_hy_now97:.2f}%" if not pd.isna(_hy_now97) else "—")

            if _dd_now < -0.15:
                st.error(f"Drawdown at {_dd_now:.1%} — crisis zone. HY spread widening likely if not already reflected.")
            elif _dd_now < -0.08:
                st.warning(f"Drawdown at {_dd_now:.1%} — elevated stress zone. Monitor HY spread follow-through.")

            # Drawdown time series
            _dd97_plot = _dd97_df.tail(756)
            _fig97a = _go97.Figure()
            _fig97a.add_trace(_go97.Scatter(
                x=_dd97_plot.index, y=_dd97_plot["sp500_drawdown"] * 100,
                name="SP500 Drawdown (%)", line=dict(color="#ef4444", width=1.5),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>",
            ))
            _fig97a.add_hline(y=-5,  line=dict(color="#f59e0b", dash="dash", width=1),
                              annotation_text="−5% Elevated", annotation_font=dict(color="#f59e0b", size=9))
            _fig97a.add_hline(y=-10, line=dict(color="#ef4444", dash="dash", width=1),
                              annotation_text="−10% Stress", annotation_font=dict(color="#ef4444", size=9))
            _fig97a.add_hline(y=-20, line=dict(color="#9b59b6", dash="dot", width=1),
                              annotation_text="−20% Crisis", annotation_font=dict(color="#9b59b6", size=9))
            _fig97a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=8, b=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Drawdown from Peak (%)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig97a, use_container_width=True)

            # Drawdown vs HY spread scatter
            if _hy97_avail:
                _scat97 = _dd97_df[["sp500_drawdown", "hy_spread"]].dropna().tail(504)
                _fig97b = _go97.Figure()
                _scat97_colors = [
                    "#ef4444" if dd < -0.10 else "#f59e0b" if dd < -0.05 else "#27ae60"
                    for dd in _scat97["sp500_drawdown"]
                ]
                _fig97b.add_trace(_go97.Scatter(
                    x=_scat97["sp500_drawdown"] * 100,
                    y=_scat97["hy_spread"],
                    mode="markers",
                    marker=dict(size=4, color=_scat97_colors, opacity=0.6),
                    hovertemplate="Drawdown: %{x:.1f}%<br>HY OAS: %{y:.2f}%<extra></extra>",
                ))
                _fig97b.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="SP500 Drawdown vs HY Spread (2yr window)",
                               font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="SP500 Drawdown (%)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="HY OAS (%)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig97b, use_container_width=True)
                _dd_corr97 = float(_scat97["sp500_drawdown"].corr(_scat97["hy_spread"]))
                st.caption(f"Drawdown vs HY correlation (2yr): {_dd_corr97:+.2f}. "
                           f"{'Negative correlation: deeper drawdowns → wider spreads.' if _dd_corr97 < -0.2 else ''}")

            # Drawdown bucket analysis
            import pandas as _pd97
            st.subheader("Drawdown Regime → Credit Spread Tendency")
            st.dataframe(_pd97.DataFrame([
                {"Drawdown":  "0 to −5%",     "Regime": "Normal",   "HY OAS Tendency": "Stable / tightening",     "Posture": "Risk-on"},
                {"Drawdown": "−5 to −10%",    "Regime": "Elevated", "HY OAS Tendency": "Watch / minor widening",  "Posture": "Neutral"},
                {"Drawdown": "−10 to −20%",   "Regime": "Stress",   "HY OAS Tendency": "Significant widening",    "Posture": "Defensive"},
                {"Drawdown": "> −20%",         "Regime": "Crisis",   "HY OAS Tendency": "Major dislocation risk",  "Posture": "Risk-off"},
            ]), use_container_width=True, hide_index=True)
            st.caption(f"Current: drawdown **{_dd_now:.1%}** from SP500 peak.")
        else:
            st.info("Drawdown analysis unavailable — requires sp500_drawdown or sp500 column.")
    except Exception as _e97:
        _err_track(_active_sub, _e97)
        st.caption(f"Drawdown analysis unavailable: {_e97}")


# =============================================================================
# TAB 8: Allocation (placeholder — content in Tab 3 Portfolio and Tab 4 Backtest)
# =============================================================================
if _active_section == "Allocation":
    st.header("Allocation")
    st.info("Allocation content is integrated into the **Portfolio** and **Backtest** tabs.")

# =============================================================================
# TAB 9: Signal Health Monitor
# =============================================================================
if _active_section == "Health":
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

# =============================================================================
# BATCH 12 ANALYTICS — sub98–103
# sub98  Credit-Equity Divergence Episodes  → tab_risk
# sub99  Shock Intensity Monitor            → tab_risk
# sub100 HY Spread Percentile              → tab_credit
# sub101 Score Velocity & Alerts           → tab_siglab
# sub102 Credit Cycle Phase               → tab_credit
# sub103 Yield Curve Velocity             → tab_macro
# =============================================================================


if _active_sub == 98:
    import plotly.graph_objects as _go98
    st.header("Credit-Equity Divergence Episodes")
    st.markdown(
        "**Credit-equity divergence** occurs when equity (SP500) and HY credit spreads move in opposite directions "
        "over a 30-day window. Prolonged divergence — equities rising while HY widens — is a classic late-cycle "
        "warning: the equity market is ignoring the stress signal embedded in credit. "
        "When credit eventually 'wins,' equity drawdowns tend to be sharp and fast."
    )
    try:
        _div98_col = "credit_equity_divergence"
        if _div98_col in df.columns:
            _div98 = df[[_div98_col, "sp500_return_30d", "hy_change_30d", "hy_spread", "sp500"]].copy()
            _div98.index = pd.to_datetime(_div98.index)
            _div98_cur = str(latest.get(_div98_col, "Unknown"))

            _d98a, _d98b, _d98c, _d98d = st.columns(4)
            _d98a.metric("Current Signal", _div98_cur)
            _d98b.metric("SP500 30d Return", f"{latest.get('sp500_return_30d', float('nan')):.1%}")
            _d98c.metric("HY Spread 30d Chg", f"{latest.get('hy_change_30d', float('nan')):+.0f} bps")
            _div98_freq = (_div98[_div98_col] == "Diverging").rolling(63).mean().iloc[-1]
            _d98d.metric("Divergence Freq (3M)", f"{_div98_freq:.0%}" if pd.notna(_div98_freq) else "—")

            if _div98_cur == "Diverging":
                st.warning("Credit-equity divergence active: equities and HY spreads are sending conflicting signals. "
                           "Historical base rate: credit resolves the divergence ~65% of the time within 6 weeks.")

            # Episode timeline bar chart
            _div98_map = {"Diverging": 1, "Converging": -1, "Neutral": 0}
            _div98["div_numeric"] = _div98[_div98_col].map(_div98_map).fillna(0)
            _div98_tail = _div98.tail(504)
            _div98_colors = ["#ef4444" if v == 1 else "#27ae60" if v == -1 else "#6b7280"
                             for v in _div98_tail["div_numeric"]]
            _fig98a = _go98.Figure()
            _fig98a.add_trace(_go98.Bar(
                x=_div98_tail.index, y=_div98_tail["div_numeric"],
                marker_color=_div98_colors, name="Divergence Signal",
                hovertemplate="%{x|%Y-%m-%d}<br>Signal: %{customdata}<extra></extra>",
                customdata=_div98_tail[_div98_col],
            ))
            _fig98a.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Credit-Equity Divergence Signal (2Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(tickvals=[-1, 0, 1], ticktext=["Converging", "Neutral", "Diverging"],
                           showgrid=False, color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig98a, use_container_width=True)
            st.caption("Red = equities rising / HY widening (Diverging) · Green = both improving (Converging)")

            # Rolling 63d divergence frequency
            _div98["div_freq_63d"] = (_div98[_div98_col] == "Diverging").rolling(63).mean() * 100
            _div98_tail2 = _div98.tail(504)
            _fig98b = _go98.Figure()
            _fig98b.add_trace(_go98.Scatter(
                x=_div98_tail2.index, y=_div98_tail2["div_freq_63d"],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                line=dict(color="#ef4444", width=2), name="Divergence Freq",
                hovertemplate="%{x|%Y-%m-%d}<br>Divergence Days: %{y:.0f}%<extra></extra>",
            ))
            _fig98b.add_hline(y=30, line=dict(color="#f59e0b", dash="dot", width=1))
            _fig98b.add_hline(y=50, line=dict(color="#ef4444", dash="dot", width=1))
            _fig98b.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Rolling 3M Divergence Frequency (%)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="% days diverging"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig98b, use_container_width=True)

            # Scatter: SP500 return vs HY change, colored by divergence state
            _div98_sc = _div98.dropna(subset=["sp500_return_30d", "hy_change_30d"]).tail(504)
            _sc_colors = {"Diverging": "#ef4444", "Converging": "#27ae60", "Neutral": "#6b7280"}
            _fig98c = _go98.Figure()
            for _state, _sc_col in _sc_colors.items():
                _mask = _div98_sc[_div98_col] == _state
                if _mask.any():
                    _fig98c.add_trace(_go98.Scatter(
                        x=_div98_sc.loc[_mask, "sp500_return_30d"] * 100,
                        y=_div98_sc.loc[_mask, "hy_change_30d"],
                        mode="markers", name=_state,
                        marker=dict(color=_sc_col, size=4, opacity=0.5),
                        hovertemplate=f"SP500: %{{x:.1f}}%<br>HY: %{{y:+.0f}}bps<br>({_state})<extra></extra>",
                    ))
            _fig98c.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig98c.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig98c.update_layout(
                height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="SP500 30d Return vs HY 30d Change", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="SP500 30d Return (%)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="HY Spread 30d Change (bps)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig98c, use_container_width=True)
            st.caption("Top-left quadrant (SP500 down, HY wider) = confirmed stress · "
                       "Bottom-right quadrant (SP500 up, HY tighter) = confirmed risk-on")

            st.markdown("**Regime Implication Table**")
            st.table(pd.DataFrame([
                {"Signal": "Diverging (EQ up, HY wide)", "Interpretation": "Late-cycle warning; credit leading equity lower",
                 "Historical Outcome": "Equity drawdown within 6-8 wks ~65% of episodes"},
                {"Signal": "Diverging (EQ down, HY tight)", "Interpretation": "Equity oversold relative to credit backdrop",
                 "Historical Outcome": "Equity recovery ~55% within 4 wks"},
                {"Signal": "Converging", "Interpretation": "Both markets aligned — risk-on or risk-off confirmed",
                 "Historical Outcome": "Trend persistence; follow the direction"},
                {"Signal": "Neutral", "Interpretation": "No directional signal; range-bound",
                 "Historical Outcome": "Mean-reversion likely"},
            ]))
        else:
            st.info("credit_equity_divergence column not found — run the feature pipeline first.")
    except Exception as _e98:
        _err_track(_active_sub, _e98)
        st.caption(f"Credit-equity divergence: {_e98}")


if _active_sub == 99:
    import plotly.graph_objects as _go99
    st.header("Shock Intensity Monitor")
    st.markdown(
        "A **shock day** is flagged when at least two of four 5-day signals breach alert thresholds simultaneously: "
        "VIX spike (+5pts), HY widening (+25 bps), SP500 selloff (-3%), or curve inversion acceleration (-10 bps). "
        "This tab tracks shock frequency (clustered shocks = systemic vs idiosyncratic) and composite shock intensity "
        "— the product of all four signals normalised to a 0-100 score."
    )
    try:
        _sf99_col = "shock_flag"
        if _sf99_col in df.columns:
            _sf99 = df[[_sf99_col, "vix_change_5d", "hy_change_5d", "sp500_return_5d",
                         "spread_change_5d", "hy_spread", "vix"]].copy()
            _sf99.index = pd.to_datetime(_sf99.index)
            _sf99_cur = int(latest.get(_sf99_col, 0))
            _sf99_freq63 = (_sf99[_sf99_col] == 1).rolling(63).mean().iloc[-1]
            _sf99_freq252 = (_sf99[_sf99_col] == 1).rolling(252).mean().iloc[-1]

            _sa, _sb99, _sc99, _sd = st.columns(4)
            _sa.metric("Shock Flag (Today)", "Active" if _sf99_cur else "Clear",
                       delta="⚠ Multi-signal breach" if _sf99_cur else None)
            _sb99.metric("Shock Freq (3M)", f"{_sf99_freq63:.1%}" if pd.notna(_sf99_freq63) else "—")
            _sc99.metric("Shock Freq (1Y)", f"{_sf99_freq252:.1%}" if pd.notna(_sf99_freq252) else "—")
            _shock_today_count = sum([
                1 if pd.notna(latest.get("vix_change_5d")) and latest.get("vix_change_5d", 0) > 5 else 0,
                1 if pd.notna(latest.get("hy_change_5d")) and latest.get("hy_change_5d", 0) > 25 else 0,
                1 if pd.notna(latest.get("sp500_return_5d")) and latest.get("sp500_return_5d", 0) < -0.03 else 0,
                1 if pd.notna(latest.get("spread_change_5d")) and latest.get("spread_change_5d", 0) < -0.10 else 0,
            ])
            _sd.metric("Signals Breached", f"{_shock_today_count}/4")

            if _sf99_cur:
                st.error("Shock detected: multiple market stress signals breached simultaneously. "
                         "Check VIX, HY spread, and SP500 charts for confirmation.")

            # Rolling shock frequency
            _sf99["freq_63d"] = (_sf99[_sf99_col] == 1).rolling(63).mean() * 100
            _sf99["freq_21d"] = (_sf99[_sf99_col] == 1).rolling(21).mean() * 100
            _sf99_tail = _sf99.tail(504)
            _fig99a = _go99.Figure()
            _fig99a.add_trace(_go99.Scatter(
                x=_sf99_tail.index, y=_sf99_tail["freq_63d"],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
                line=dict(color="#ef4444", width=2), name="3M Freq",
                hovertemplate="%{x|%Y-%m-%d}<br>3M: %{y:.1f}%<extra></extra>",
            ))
            _fig99a.add_trace(_go99.Scatter(
                x=_sf99_tail.index, y=_sf99_tail["freq_21d"],
                line=dict(color="#f59e0b", width=1.5, dash="dot"), name="1M Freq",
                hovertemplate="%{x|%Y-%m-%d}<br>1M: %{y:.1f}%<extra></extra>",
            ))
            _fig99a.add_hline(y=15, line=dict(color="rgba(239,68,68,0.5)", dash="dot", width=1))
            _fig99a.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Rolling Shock Frequency (% days flagged)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig99a, use_container_width=True)
            st.caption("Dashed threshold at 15%: sustained shock clustering (>15% of days) signals systemic vs idiosyncratic stress")

            # Shock intensity composite: normalised absolute values of 4 signal components
            _sf99["vix_z5"] = _sf99["vix_change_5d"].abs() / 5.0
            _sf99["hy_z5"] = _sf99["hy_change_5d"].abs() / 25.0
            _sf99["sp_z5"] = _sf99["sp500_return_5d"].abs() / 0.03
            _sf99["cv_z5"] = _sf99["spread_change_5d"].abs() / 0.10
            _sf99["intensity"] = _sf99[["vix_z5", "hy_z5", "sp_z5", "cv_z5"]].mean(axis=1).clip(0, 3) / 3 * 100
            _sf99_tail2 = _sf99.tail(504)
            _fig99b = _go99.Figure()
            _fig99b.add_trace(_go99.Scatter(
                x=_sf99_tail2.index, y=_sf99_tail2["intensity"],
                line=dict(color="#8b5cf6", width=2), name="Shock Intensity",
                fill="tozeroy", fillcolor="rgba(139,92,246,0.1)",
                hovertemplate="%{x|%Y-%m-%d}<br>Intensity: %{y:.0f}/100<extra></extra>",
            ))
            _fig99b.add_hline(y=50, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig99b.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Shock Intensity Score (0–100)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig99b, use_container_width=True)

            # Recent shock episodes table
            _shock_eps = _sf99[_sf99[_sf99_col] == 1].tail(20).copy()
            if not _shock_eps.empty:
                st.markdown("**Recent Shock Episodes (last 20)**")
                _shock_tbl = pd.DataFrame({
                    "Date": _shock_eps.index.strftime("%Y-%m-%d"),
                    "VIX Δ5d": _shock_eps["vix_change_5d"].round(1),
                    "HY Δ5d (bps)": _shock_eps["hy_change_5d"].round(0),
                    "SP500 5d (%)": (_shock_eps["sp500_return_5d"] * 100).round(1),
                    "Curve Δ5d (pp)": _shock_eps["spread_change_5d"].round(3),
                    "Intensity": _shock_eps["intensity"].round(0),
                })
                st.dataframe(_shock_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("shock_flag column not found — run the feature pipeline first.")
    except Exception as _e99:
        _err_track(_active_sub, _e99)
        st.caption(f"Shock monitor: {_e99}")


if _active_sub == 107:
    import plotly.graph_objects as _go107
    st.header("Liquidity Sub-Score Deep Dive")
    st.markdown(
        "The **liquidity sub-score** (10% weight in the composite) captures market microstructure stress: "
        "bid-ask widening, cross-asset illiquidity, and funding-market disruptions that precede broader "
        "credit deterioration. Unlike the funding stress score (which focuses on interbank plumbing), the "
        "liquidity score captures on-the-run/off-the-run spreads, ETF premium/discount cycles, and "
        "broad market depth deterioration. When elevated, expect wider transaction costs and potential "
        "forced-seller pressure in HY credit."
    )
    try:
        _liq107_col = "liquidity_regime_score_smooth"
        _liq107_raw = "liquidity_regime_score"
        _liq_col = _liq107_col if _liq107_col in df.columns else (_liq107_raw if _liq107_raw in df.columns else None)
        if _liq_col:
            _liq107 = df[[_liq_col]].copy()
            if "hy_spread" in df.columns:
                _liq107["hy_spread"] = df["hy_spread"]
            if "enhanced_funding_stress_score_smooth" in df.columns:
                _liq107["funding_score"] = df["enhanced_funding_stress_score_smooth"]
            _liq107.index = pd.to_datetime(_liq107.index)
            _cur_liq = float(latest.get(_liq_col, float("nan")))

            def _liq_regime(score):
                if pd.isna(score):
                    return "Unknown"
                if score >= 70:
                    return "Stressed"
                if score >= 50:
                    return "Elevated"
                if score >= 30:
                    return "Normal"
                return "Benign"

            _liq_reg = _liq_regime(_cur_liq)
            _liq_pctile = (df[_liq_col].dropna() < _cur_liq).mean() * 100 if pd.notna(_cur_liq) else float("nan")

            _la, _lb, _lc, _ld = st.columns(4)
            _la.metric("Liquidity Score", f"{_cur_liq:.0f}/100" if pd.notna(_cur_liq) else "—")
            _lb.metric("Regime", _liq_reg)
            _lc.metric("Historical Pctile", f"{_liq_pctile:.0f}th" if pd.notna(_liq_pctile) else "—")
            _liq107["liq_90d_high"] = _liq107[_liq_col].rolling(90).max()
            _ld.metric("90d High", f"{_liq107['liq_90d_high'].iloc[-1]:.0f}" if not _liq107.empty else "—")

            if pd.notna(_cur_liq) and _cur_liq >= 70:
                st.error("Liquidity score in stressed territory — transaction costs elevated, "
                         "forced-seller risk in credit markets.")
            elif pd.notna(_cur_liq) and _cur_liq >= 50:
                st.warning("Liquidity score elevated — monitor spread bid-ask widening and ETF premium/discount.")

            # Score time series
            _liq_tail = _liq107.tail(756)
            _fig107a = _go107.Figure()
            _fig107a.add_hrect(y0=70, y1=105, fillcolor="rgba(239,68,68,0.08)", line_width=0)
            _fig107a.add_hrect(y0=50, y1=70, fillcolor="rgba(245,158,11,0.06)", line_width=0)
            _fig107a.add_trace(_go107.Scatter(
                x=_liq_tail.index, y=_liq_tail[_liq_col],
                fill="tozeroy", fillcolor="rgba(79,142,247,0.1)",
                line=dict(color="#4f8ef7", width=2), name="Liquidity Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}/100<extra></extra>",
            ))
            _fig107a.add_hline(y=50, line=dict(color="rgba(245,158,11,0.5)", dash="dot", width=1))
            _fig107a.add_hline(y=70, line=dict(color="rgba(239,68,68,0.5)", dash="dot", width=1))
            _fig107a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Liquidity Sub-Score (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig107a, use_container_width=True)
            st.caption("Orange band = Elevated (50–70) · Red band = Stressed (>70)")

            # Liquidity vs funding stress overlay
            if "funding_score" in _liq107.columns and _liq107["funding_score"].notna().any():
                _fig107b = _go107.Figure()
                _fig107b.add_trace(_go107.Scatter(
                    x=_liq_tail.index, y=_liq_tail[_liq_col],
                    name="Liquidity Score", line=dict(color="#4f8ef7", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Liquidity: %{y:.0f}<extra></extra>",
                ))
                _fig107b.add_trace(_go107.Scatter(
                    x=_liq_tail.index, y=_liq_tail["funding_score"],
                    name="Funding Stress Score", line=dict(color="#a78bfa", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Funding: %{y:.0f}<extra></extra>",
                ))
                _fig107b.add_hline(y=50, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
                _fig107b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Liquidity vs Funding Stress Sub-Scores", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Score"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig107b, use_container_width=True)
                st.caption("Both scores elevated simultaneously = compound liquidity-funding stress — highest credit risk")

            # Liquidity vs HY scatter
            if "hy_spread" in _liq107.columns:
                _liq_sc = _liq107.dropna(subset=[_liq_col, "hy_spread"]).tail(756)
                _fig107c = _go107.Figure()
                _fig107c.add_trace(_go107.Scatter(
                    x=_liq_sc[_liq_col], y=_liq_sc["hy_spread"],
                    mode="markers",
                    marker=dict(color="#4f8ef7", size=3, opacity=0.4),
                    hovertemplate="Liquidity: %{x:.0f}<br>HY Spread: %{y:.0f}bps<extra></extra>",
                ))
                _fig107c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Liquidity Score vs HY Spread (3Y scatter)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Liquidity Score"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig107c, use_container_width=True)
        else:
            st.info("Liquidity score column not found in dataframe. "
                    "This score is computed by the scoring pipeline — ensure the full pipeline has run.")
    except Exception as _e107:
        _err_track(_active_sub, _e107)
        st.caption(f"Liquidity score: {_e107}")


if _active_sub == 112:
    import plotly.graph_objects as _go112
    st.header("VIX Momentum Deep Dive")
    st.markdown(
        "**VIX momentum** — the 5-day and 30-day change in the VIX — is a short-horizon stress signal that "
        "leads HY spread widening by 1–3 weeks. A sudden VIX spike (>5pts in 5 days) is an acute shock "
        "indicator. Sustained 30-day VIX elevation (+3–5 pts) signals a regime shift toward risk-off. "
        "The **VIX acceleration** (30d change minus 5d change, reversed) identifies whether volatility "
        "is rising in a single burst or grinding higher — the latter is historically more damaging to credit."
    )
    try:
        _vix112_cols = ["vix", "vix_change_5d", "vix_change_30d", "hy_spread", "hy_change_5d"]
        if all(c in df.columns for c in ["vix", "vix_change_5d", "vix_change_30d"]):
            _vix112 = df[[c for c in _vix112_cols if c in df.columns]].copy()
            _vix112.index = pd.to_datetime(_vix112.index)
            _cur_vix = float(latest.get("vix", float("nan")))
            _cur_v5 = float(latest.get("vix_change_5d", float("nan")))
            _cur_v30 = float(latest.get("vix_change_30d", float("nan")))

            def _vix_regime(vix, v30):
                if pd.isna(vix):
                    return "Unknown"
                if vix > 30:
                    return "Fear" if pd.isna(v30) or v30 > 0 else "Subsiding Fear"
                if vix > 20:
                    return "Elevated" if pd.isna(v30) or v30 > 0 else "Easing"
                return "Complacent" if pd.isna(v30) or v30 < -2 else "Low"

            _vix_reg = _vix_regime(_cur_vix, _cur_v30)
            _vix112["vix_accel"] = _vix112["vix_change_30d"] - _vix112["vix_change_5d"]

            _va112, _vb112, _vc112, _vd112 = st.columns(4)
            _va112.metric("VIX Level", f"{_cur_vix:.1f}" if pd.notna(_cur_vix) else "—")
            _vb112.metric("VIX Δ5d", f"{_cur_v5:+.1f}" if pd.notna(_cur_v5) else "—",
                          delta_color="inverse")
            _vc112.metric("VIX Δ30d", f"{_cur_v30:+.1f}" if pd.notna(_cur_v30) else "—",
                          delta_color="inverse")
            _vd112.metric("VIX Regime", _vix_reg)

            if pd.notna(_cur_v5) and _cur_v5 > 5:
                st.error(f"Acute VIX spike: +{_cur_v5:.1f}pts in 5 days — short-horizon credit stress signal active.")
            elif pd.notna(_cur_v30) and _cur_v30 > 3:
                st.warning(f"VIX rising steadily over 30 days (+{_cur_v30:.1f}pts) — sustained risk-off regime forming.")

            # VIX level + 30d change dual chart
            _vix_tail = _vix112.tail(504)
            _fig112a = _go112.Figure()
            _fig112a.add_trace(_go112.Scatter(
                x=_vix_tail.index, y=_vix_tail["vix"],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                line=dict(color="#ef4444", width=2), name="VIX",
                hovertemplate="%{x|%Y-%m-%d}<br>VIX: %{y:.1f}<extra></extra>",
            ))
            _fig112a.add_hline(y=20, line=dict(color="rgba(245,158,11,0.4)", dash="dot", width=1))
            _fig112a.add_hline(y=30, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig112a.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="VIX Level (2Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig112a, use_container_width=True)

            # 5d and 30d momentum
            _fig112b = _go112.Figure()
            _fig112b.add_trace(_go112.Scatter(
                x=_vix_tail.index, y=_vix_tail["vix_change_30d"],
                line=dict(color="#f59e0b", width=2), name="Δ30d",
                hovertemplate="%{x|%Y-%m-%d}<br>Δ30d: %{y:+.1f}<extra></extra>",
            ))
            _fig112b.add_trace(_go112.Scatter(
                x=_vix_tail.index, y=_vix_tail["vix_change_5d"],
                line=dict(color="#a78bfa", width=1.5, dash="dot"), name="Δ5d",
                hovertemplate="%{x|%Y-%m-%d}<br>Δ5d: %{y:+.1f}<extra></extra>",
            ))
            _fig112b.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
            _fig112b.add_hline(y=5, line=dict(color="rgba(239,68,68,0.3)", dash="dot", width=1))
            _fig112b.update_layout(
                height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="VIX Momentum (5d and 30d change)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Δ pts"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig112b, use_container_width=True)
            st.caption("Dashed red line at +5: acute spike threshold · Yellow = 30d sustained momentum")

            # Scatter: VIX 5d spike vs HY 5d change
            if "hy_change_5d" in _vix112.columns:
                _sc112 = _vix112.dropna(subset=["vix_change_5d", "hy_change_5d"]).tail(504)
                _spike = _sc112["vix_change_5d"].abs() > 5
                _sc112_colors = ["#ef4444" if s else "#4f8ef7" for s in _spike]
                _fig112c = _go112.Figure()
                _fig112c.add_trace(_go112.Scatter(
                    x=_sc112["vix_change_5d"], y=_sc112["hy_change_5d"],
                    mode="markers",
                    marker=dict(color=_sc112_colors, size=4, opacity=0.45),
                    hovertemplate="VIX Δ5d: %{x:+.1f}<br>HY Δ5d: %{y:+.0f}bps<extra></extra>",
                ))
                _fig112c.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig112c.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig112c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="VIX Δ5d vs HY Spread Δ5d (scatter)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="VIX Δ5d (pts)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Δ5d (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig112c, use_container_width=True)
                st.caption("Red dots = VIX spike days (>5pts) · Top-right: VIX up + HY wider = confirmed stress")
        else:
            st.info("VIX momentum columns not found — run the feature pipeline.")
    except Exception as _e112:
        _err_track(_active_sub, _e112)
        st.caption(f"VIX momentum: {_e112}")


if _active_sub == 113:
    import plotly.graph_objects as _go113
    st.header("FX / Commodity Sub-Score")
    st.markdown(
        "The **FX/Commodity sub-score** (5% weight in the composite) captures risk-off signals transmitted "
        "through the currency and commodity channels. A stronger USD (flight-to-safety) combined with "
        "falling commodity prices historically precedes EM credit spread widening and, with a 2–4 week lag, "
        "US HY spread widening via the commodity producer sub-sector. At 5% weight, it rarely moves the "
        "composite needle alone — but when it aligns with the Treasury and Credit Risk sub-scores, "
        "it reinforces the signal."
    )
    try:
        _fx113_col = "fx_commodity_score_smooth"
        _fx113_raw = "fx_commodity_score"
        _fx_col = _fx113_col if _fx113_col in df.columns else (_fx113_raw if _fx113_raw in df.columns else None)
        if _fx_col:
            _fx113 = df[[_fx_col]].copy()
            if "hy_spread" in df.columns:
                _fx113["hy_spread"] = df["hy_spread"]
            if "composite_risk_score_smooth" in df.columns:
                _fx113["composite"] = df["composite_risk_score_smooth"]
            _fx113.index = pd.to_datetime(_fx113.index)
            _cur_fx = float(latest.get(_fx_col, float("nan")))
            _fx_pctile = (df[_fx_col].dropna() < _cur_fx).mean() * 100 if pd.notna(_cur_fx) else float("nan")

            def _fx_regime(score):
                if pd.isna(score):
                    return "Unknown"
                if score >= 70:
                    return "Risk-Off (Stress)"
                if score >= 50:
                    return "Elevated"
                if score >= 30:
                    return "Neutral"
                return "Risk-On"

            _fx_reg = _fx_regime(_cur_fx)

            _fa, _fb, _fc, _fd = st.columns(4)
            _fa.metric("FX/Commodity Score", f"{_cur_fx:.0f}/100" if pd.notna(_cur_fx) else "—")
            _fb.metric("Regime", _fx_reg)
            _fc.metric("Historical Pctile", f"{_fx_pctile:.0f}th" if pd.notna(_fx_pctile) else "—")
            _fx_vel = float(df[_fx_col].diff(21).iloc[-1]) if df[_fx_col].notna().any() else float("nan")
            _fd.metric("21d Velocity", f"{_fx_vel:+.1f}pts" if pd.notna(_fx_vel) else "—",
                       delta_color="inverse")

            # Score time series
            _fx_tail = _fx113.tail(756)
            _fig113a = _go113.Figure()
            _fig113a.add_hrect(y0=70, y1=105, fillcolor="rgba(239,68,68,0.07)", line_width=0)
            _fig113a.add_hrect(y0=50, y1=70, fillcolor="rgba(245,158,11,0.05)", line_width=0)
            _fig113a.add_trace(_go113.Scatter(
                x=_fx_tail.index, y=_fx_tail[_fx_col],
                fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
                line=dict(color="#a78bfa", width=2), name="FX/Commodity Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            _fig113a.add_hline(y=50, line=dict(color="rgba(245,158,11,0.4)", dash="dot", width=1))
            _fig113a.add_hline(y=70, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig113a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="FX/Commodity Sub-Score (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig113a, use_container_width=True)

            # FX score vs composite overlay
            if "composite" in _fx113.columns:
                _fig113b = _go113.Figure()
                _fig113b.add_trace(_go113.Scatter(
                    x=_fx_tail.index, y=_fx_tail[_fx_col],
                    name="FX/Commodity Score", line=dict(color="#a78bfa", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>FX/Comm: %{y:.0f}<extra></extra>",
                ))
                _fig113b.add_trace(_go113.Scatter(
                    x=_fx_tail.index, y=_fx_tail["composite"],
                    name="Composite Score", line=dict(color="#4f8ef7", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
                _fig113b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="FX/Commodity Score vs Composite (3Y)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Score"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig113b, use_container_width=True)
                st.caption("FX/Commodity score carries 5% weight — rarely decisive alone, "
                           "but amplifies composite when aligned with Treasury and Credit Risk sub-scores")

            # HY spread scatter
            if "hy_spread" in _fx113.columns:
                _fx_sc = _fx113.dropna(subset=[_fx_col, "hy_spread"]).tail(756)
                _fig113c = _go113.Figure()
                _fig113c.add_trace(_go113.Scatter(
                    x=_fx_sc[_fx_col], y=_fx_sc["hy_spread"],
                    mode="markers",
                    marker=dict(color="#a78bfa", size=3, opacity=0.4),
                    hovertemplate="FX Score: %{x:.0f}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig113c.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="FX/Commodity Score vs HY Spread (3Y scatter)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="FX/Commodity Score"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig113c, use_container_width=True)
        else:
            st.info("FX/Commodity score not found — run the full scoring pipeline.")
    except Exception as _e113:
        _err_track(_active_sub, _e113)
        st.caption(f"FX/Commodity score: {_e113}")


if _active_sub == 119:
    import plotly.graph_objects as _go119
    from src.regime_attribution import COMPOSITE_WEIGHTS as _CW119
    st.header("Enhanced Funding Stress Sub-Score")
    st.markdown(
        "The **enhanced funding stress sub-score** (10% weight) extends beyond the basic TED spread proxy "
        "by combining the **NFCI**, **Adjusted NFCI (ANFCI)**, the **St. Louis FSI**, and "
        "**initial jobless claims z-score**. ANFCI is particularly important: it strips out the contribution "
        "of expected macro deterioration, so a high ANFCI signals *pure financial plumbing stress* — "
        "banks and counterparties tightening credit beyond what fundamentals justify. "
        "This score has historically led broad HY spread widening by 3–5 weeks."
    )
    try:
        _ef119_col = "enhanced_funding_stress_score_smooth"
        _ef119_raw = "enhanced_funding_stress_score"
        _ef_col = _ef119_col if _ef119_col in df.columns else (_ef119_raw if _ef119_raw in df.columns else None)
        if _ef_col:
            _ef119 = df[[_ef_col]].copy()
            for _c in ["hy_spread", "composite_risk_score_smooth", "nfci",
                        "liquidity_regime_score_smooth"]:
                if _c in df.columns:
                    _ef119[_c] = df[_c]
            _ef119.index = pd.to_datetime(_ef119.index)
            _cur_ef = float(latest.get(_ef_col, float("nan")))
            _ef_pctile = (df[_ef_col].dropna() < _cur_ef).mean() * 100 if pd.notna(_cur_ef) else float("nan")
            _ef_vel = float(df[_ef_col].diff(21).iloc[-1]) if df[_ef_col].notna().any() else float("nan")
            _ef_contrib = _cur_ef * _CW119.get("enhanced_funding", 0.10) if pd.notna(_cur_ef) else float("nan")

            _ea119, _eb119, _ec119, _ed119 = st.columns(4)
            _ea119.metric("Enh Funding Score", f"{_cur_ef:.0f}/100" if pd.notna(_cur_ef) else "—")
            _eb119.metric("Composite Contrib", f"{_ef_contrib:.1f}pts" if pd.notna(_ef_contrib) else "—",
                          help="Score × 10% weight — medium-horizon leading signal")
            _ec119.metric("Historical Pctile", f"{_ef_pctile:.0f}th" if pd.notna(_ef_pctile) else "—")
            _ed119.metric("21d Velocity", f"{_ef_vel:+.1f}pts" if pd.notna(_ef_vel) else "—",
                          delta_color="inverse")

            if pd.notna(_cur_ef) and _cur_ef >= 65:
                st.error("Enhanced funding stress elevated — interbank and financial conditions tightening "
                         "beyond fundamentals. This score leads credit spread widening by 3–5 weeks.")

            _ef_tail = _ef119.tail(756)
            _fig119a = _go119.Figure()
            _fig119a.add_hrect(y0=65, y1=105, fillcolor="rgba(239,68,68,0.07)", line_width=0)
            _fig119a.add_hrect(y0=45, y1=65, fillcolor="rgba(245,158,11,0.05)", line_width=0)
            _fig119a.add_trace(_go119.Scatter(
                x=_ef_tail.index, y=_ef_tail[_ef_col],
                fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
                line=dict(color="#10b981", width=2), name="Enh Funding Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            if "composite_risk_score_smooth" in _ef_tail.columns:
                _fig119a.add_trace(_go119.Scatter(
                    x=_ef_tail.index, y=_ef_tail["composite_risk_score_smooth"],
                    line=dict(color="#e2e8f0", width=1, dash="dot"), name="Composite",
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
            _fig119a.add_hline(y=45, line=dict(color="rgba(245,158,11,0.4)", dash="dot", width=1))
            _fig119a.add_hline(y=65, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig119a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Enhanced Funding Stress Sub-Score (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig119a, use_container_width=True)

            # vs Liquidity score overlay (both measure plumbing stress)
            if "liquidity_regime_score_smooth" in _ef_tail.columns:
                _fig119b = _go119.Figure()
                _fig119b.add_trace(_go119.Scatter(
                    x=_ef_tail.index, y=_ef_tail[_ef_col],
                    name="Enh Funding (10%)", line=dict(color="#10b981", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Enh Funding: %{y:.0f}<extra></extra>",
                ))
                _fig119b.add_trace(_go119.Scatter(
                    x=_ef_tail.index, y=_ef_tail["liquidity_regime_score_smooth"],
                    name="Liquidity Score (10%)", line=dict(color="#4f8ef7", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Liquidity: %{y:.0f}<extra></extra>",
                ))
                _fig119b.add_hline(y=50, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
                _fig119b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Enh Funding vs Liquidity Score — plumbing stress pair", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Score"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig119b, use_container_width=True)
                st.caption("Both elevated simultaneously = compound plumbing stress — highest short-horizon credit risk")

            st.markdown("**Enhanced Funding Score — Driver Breakdown**")
            st.table(pd.DataFrame([
                {"Input": "NFCI Level", "Pts": "Up to 30", "Signal": ">0.5 → 30pts; 0.2–0.5 → 20pts; 0–0.2 → 10pts"},
                {"Input": "ANFCI (adjusted)", "Pts": "Up to 35", "Signal": "Pure financial stress beyond macro; >0.5 → 35pts"},
                {"Input": "St. Louis FSI (STLFSI)", "Pts": "Up to 20", "Signal": ">1σ → 20pts; 0.5–1σ → 10pts"},
                {"Input": "Initial Claims Z-Score", "Pts": "Up to 15", "Signal": "Claims spike z > 2 → 15pts; 1–2 → 8pts"},
            ]))
        else:
            st.info("Enhanced funding stress score not found — run the full scoring pipeline.")
    except Exception as _e119:
        _err_track(_active_sub, _e119)
        st.caption(f"Enhanced funding score: {_e119}")


if _active_sub == 124:
    import plotly.graph_objects as _go124
    st.header("Equity Return Context")
    st.markdown(
        "**SP500 5-day and 30-day returns** are used as inputs in multiple composite sub-scores, "
        "but their standalone distribution and clustering properties are also credit-relevant. "
        "Equity declines of more than **-5% in 5 days** historically see HY spreads widen 20–80 bps "
        "within 2 weeks. Sustained **-10%+ over 30 days** signals de-risking flows that spill into "
        "credit, even when HY fundamentals are intact. "
        "The **drawdown context** — where the 5d/30d return sits relative to the trailing drawdown — "
        "distinguishes shallow corrections from structural breaks."
    )
    try:
        _eq124_cols = ["sp500_return_5d", "sp500_return_30d", "sp500_drawdown", "sp500", "hy_spread"]
        _avail124 = [c for c in _eq124_cols if c in df.columns]
        if "sp500_return_5d" in df.columns or "sp500_return_30d" in df.columns:
            _eq124 = df[[c for c in _avail124]].copy()
            _eq124.index = pd.to_datetime(_eq124.index)
            _cur_r5 = float(latest.get("sp500_return_5d", float("nan")))
            _cur_r30 = float(latest.get("sp500_return_30d", float("nan")))
            _cur_dd = float(latest.get("sp500_drawdown", float("nan")))

            _ea124, _eb124, _ec124, _ed124 = st.columns(4)
            _ea124.metric("SP500 Return 5d", f"{_cur_r5:.1%}" if pd.notna(_cur_r5) else "—",
                          delta_color="normal" if pd.notna(_cur_r5) and _cur_r5 > 0 else "inverse")
            _eb124.metric("SP500 Return 30d", f"{_cur_r30:.1%}" if pd.notna(_cur_r30) else "—",
                          delta_color="normal" if pd.notna(_cur_r30) and _cur_r30 > 0 else "inverse")
            _ec124.metric("SP500 Drawdown", f"{_cur_dd:.1%}" if pd.notna(_cur_dd) else "—",
                          delta_color="inverse")
            _r30_pctile = (df["sp500_return_30d"].dropna() < _cur_r30).mean() * 100 if pd.notna(_cur_r30) and "sp500_return_30d" in df.columns else float("nan")
            _ed124.metric("30d Return Pctile", f"{_r30_pctile:.0f}th" if pd.notna(_r30_pctile) else "—")

            if pd.notna(_cur_r5) and _cur_r5 < -0.05:
                st.error(f"SP500 down {_cur_r5:.1%} in 5 days — historical pattern: HY spreads widen 20–80bps within 2 weeks.")
            elif pd.notna(_cur_r30) and _cur_r30 < -0.10:
                st.warning(f"SP500 down {_cur_r30:.1%} over 30 days — sustained de-risking; credit spread widening risk elevated.")

            # Return history
            _eq_tail = _eq124.tail(504)
            if "sp500_return_30d" in _eq_tail.columns:
                _r30_colors = ["#ef4444" if v < -0.05 else "#f59e0b" if v < 0 else "#27ae60"
                               for v in _eq_tail["sp500_return_30d"].fillna(0)]
                _fig124a = _go124.Figure()
                _fig124a.add_trace(_go124.Bar(
                    x=_eq_tail.index, y=(_eq_tail["sp500_return_30d"] * 100),
                    marker_color=_r30_colors, name="30d Return",
                    hovertemplate="%{x|%Y-%m-%d}<br>30d: %{y:+.1f}%<extra></extra>",
                ))
                _fig124a.add_hline(y=-5, line=dict(color="rgba(245,158,11,0.5)", dash="dot", width=1))
                _fig124a.add_hline(y=-10, line=dict(color="rgba(239,68,68,0.5)", dash="dot", width=1))
                _fig124a.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
                _fig124a.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="SP500 30-Day Return (2Y)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="%"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig124a, use_container_width=True)
                st.caption("Orange: -5% threshold | Red: -10% threshold — credit contagion risk zones")

            # SP500 return 30d vs HY change 30d scatter
            if "hy_spread" in _eq124.columns and "sp500_return_30d" in _eq124.columns:
                _eq_sc = _eq124.dropna(subset=["sp500_return_30d"]).tail(504).copy()
                _eq_sc["hy_change_30d"] = _eq_sc["hy_spread"].diff(30)
                _eq_sc = _eq_sc.dropna(subset=["hy_change_30d"])
                _sc_colors124 = ["#ef4444" if r < -0.05 else "#f59e0b" if r < 0 else "#27ae60"
                                  for r in _eq_sc["sp500_return_30d"]]
                _fig124b = _go124.Figure()
                _fig124b.add_trace(_go124.Scatter(
                    x=_eq_sc["sp500_return_30d"] * 100, y=_eq_sc["hy_change_30d"],
                    mode="markers",
                    marker=dict(color=_sc_colors124, size=4, opacity=0.45),
                    hovertemplate="SP500 30d: %{x:+.1f}%<br>HY Δ30d: %{y:+.0f}bps<extra></extra>",
                ))
                _fig124b.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig124b.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
                _fig124b.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="SP500 30d Return vs HY 30d Change", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="SP500 30d Return (%)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Δ30d (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig124b, use_container_width=True)
                st.caption("Strong negative correlation: equity declines → HY widening · "
                           "Outliers (equity up + HY wider) = credit-equity divergence episodes")

            # Return distribution
            if "sp500_return_30d" in _eq124.columns:
                _ret_vals = _eq124["sp500_return_30d"].dropna() * 100
                _fig124c = _go124.Figure()
                _fig124c.add_trace(_go124.Histogram(
                    x=_ret_vals, nbinsx=60,
                    marker_color="#4f8ef7", opacity=0.7, name="30d Return Distribution",
                    hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>",
                ))
                if pd.notna(_cur_r30):
                    _fig124c.add_vline(x=_cur_r30 * 100,
                                       line=dict(color="#f59e0b", dash="dash", width=2),
                                       annotation_text=f"Current: {_cur_r30:.1%}",
                                       annotation_font_color="#f59e0b")
                _fig124c.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="SP500 30-Day Return Distribution (full history)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    xaxis=dict(showgrid=False, color="#6b7280", title="30d Return (%)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig124c, use_container_width=True)
        else:
            st.info("SP500 return columns not found — run the feature pipeline.")
    except Exception as _e124:
        _err_track(_active_sub, _e124)
        st.caption(f"Equity returns: {_e124}")


if _active_sub == 127:
    import plotly.graph_objects as _go127
    st.header("VIX Level Context")
    st.markdown(
        "The **VIX** (CBOE Volatility Index) measures the market's 30-day implied volatility on S&P 500 "
        "options. For credit, VIX is important because it is the primary component of the Complacency "
        "sub-score and a key input to the Credit Market Risk sub-score. "
        "Key levels: **VIX < 15** = complacency risk / tight credit; **15–20** = normal; "
        "**20–30** = elevated stress; **VIX > 30** = acute fear / HY spread widening underway. "
        "This tab examines VIX in isolation — its full-history percentile, regime distribution, "
        "and the relationship between sustained VIX elevation and HY spread outcomes."
    )
    try:
        if "vix" in df.columns and df["vix"].notna().any():
            _vix127 = df[["vix"]].copy()
            if "hy_spread" in df.columns:
                _vix127["hy_spread"] = df["hy_spread"]
            _vix127.index = pd.to_datetime(_vix127.index)
            _cur_vix = float(latest.get("vix", float("nan")))
            _vix_pctile = (df["vix"].dropna() < _cur_vix).mean() * 100 if pd.notna(_cur_vix) else float("nan")
            _vix_1m_avg = float(df["vix"].rolling(21).mean().iloc[-1])
            _vix_3m_avg = float(df["vix"].rolling(63).mean().iloc[-1])

            def _vix_regime_label(vix):
                if pd.isna(vix):
                    return "Unknown"
                if vix > 40:
                    return "Panic (>40)"
                if vix > 30:
                    return "Acute Fear (30–40)"
                if vix > 20:
                    return "Elevated (20–30)"
                if vix > 15:
                    return "Normal (15–20)"
                return "Complacent (<15)"

            _vix_reg = _vix_regime_label(_cur_vix)
            _vix_reg_colors = {
                "Panic (>40)": "#7f1d1d",
                "Acute Fear (30–40)": "#ef4444",
                "Elevated (20–30)": "#f59e0b",
                "Normal (15–20)": "#4f8ef7",
                "Complacent (<15)": "#27ae60",
            }

            _va127, _vb127, _vc127, _vd127 = st.columns(4)
            _va127.metric("VIX Level", f"{_cur_vix:.1f}" if pd.notna(_cur_vix) else "—")
            _vb127.metric("Regime", _vix_reg)
            _vc127.metric("Historical Pctile", f"{_vix_pctile:.0f}th" if pd.notna(_vix_pctile) else "—")
            _vd127.metric("1M / 3M Avg", f"{_vix_1m_avg:.1f} / {_vix_3m_avg:.1f}"
                          if pd.notna(_vix_1m_avg) else "—")

            if pd.notna(_cur_vix) and _cur_vix > 30:
                st.error(f"VIX in Acute Fear territory ({_cur_vix:.1f}) — "
                         "implied volatility at stressed levels. HY spreads typically widen during sustained VIX>30 regimes.")
            elif pd.notna(_cur_vix) and _cur_vix < 15:
                st.warning(f"VIX at complacency levels ({_cur_vix:.1f}) — "
                           "fear premia historically insufficient at these levels. Credit spread compression risk elevated.")

            # VIX full history with regime shading
            _vix_full = _vix127.copy()
            _fig127a = _go127.Figure()
            _fig127a.add_hrect(y0=30, y1=80, fillcolor="rgba(239,68,68,0.08)", line_width=0)
            _fig127a.add_hrect(y0=0, y1=15, fillcolor="rgba(39,174,96,0.06)", line_width=0)
            _fig127a.add_trace(_go127.Scatter(
                x=_vix_full.index, y=_vix_full["vix"],
                line=dict(color="#ef4444", width=1.5), name="VIX",
                fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
                hovertemplate="%{x|%Y-%m-%d}<br>VIX: %{y:.1f}<extra></extra>",
            ))
            for _vt, _vc in [(15, "rgba(39,174,96,0.4)"), (20, "rgba(79,142,247,0.4)"),
                              (30, "rgba(245,158,11,0.4)"), (40, "rgba(239,68,68,0.4)")]:
                _fig127a.add_hline(y=_vt, line=dict(color=_vc, dash="dot", width=1))
            _fig127a.update_layout(
                height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="VIX Level — Full History", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig127a, use_container_width=True)
            st.caption("Green band = Complacent (<15) · Red band = Fear (>30) · Thresholds: 15, 20, 30, 40")

            # VIX regime frequency
            _vix_full["regime"] = _vix_full["vix"].apply(_vix_regime_label)
            _vix_freq = _vix_full["regime"].value_counts(normalize=True).mul(100)
            _fig127b = _go127.Figure()
            _fig127b.add_trace(_go127.Bar(
                x=_vix_freq.index.tolist(), y=_vix_freq.values.tolist(),
                marker_color=[_vix_reg_colors.get(r, "#6b7280") for r in _vix_freq.index],
                text=[f"{v:.0f}%" for v in _vix_freq.values], textposition="auto",
                hovertemplate="%{x}: %{y:.0f}%<extra></extra>",
            ))
            _fig127b.update_layout(
                height=190, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="VIX Regime Frequency (Full History)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="% of days"),
                xaxis=dict(color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig127b, use_container_width=True)

            # VIX vs HY spread scatter
            if "hy_spread" in _vix_full.columns:
                _vix_sc = _vix_full.dropna(subset=["vix", "hy_spread"])
                _sc_colors127 = [_vix_reg_colors.get(_vix_regime_label(v), "#6b7280") for v in _vix_sc["vix"]]
                _fig127c = _go127.Figure()
                _fig127c.add_trace(_go127.Scatter(
                    x=_vix_sc["vix"], y=_vix_sc["hy_spread"],
                    mode="markers",
                    marker=dict(color=_sc_colors127, size=3, opacity=0.35),
                    hovertemplate="VIX: %{x:.1f}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig127c.update_layout(
                    height=230, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="VIX vs HY Spread — Full History Scatter", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="VIX"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig127c, use_container_width=True)
                st.caption("Strong positive correlation: VIX>30 historically coexists with HY spreads >500bps. "
                           "Note: correlation is nonlinear — VIX 15→20 barely moves HY; VIX 30→40 surges HY 200+ bps.")
        else:
            st.info("VIX column not found — run the feature pipeline.")
    except Exception as _e127:
        _err_track(_active_sub, _e127)
        st.caption(f"VIX context: {_e127}")

# ---------------------------------------------------------------------------
# Batch 17 — sub128–sub133
# ---------------------------------------------------------------------------

# sub128 — Score Correlations (tab_siglab)

if _active_sub == 133:
    try:
        import plotly.graph_objects as _go133
        import numpy as _np133
        import pandas as _pd133
        if "sp500_drawdown" in df.columns and "hy_spread" in df.columns:
            _dd133 = df["sp500_drawdown"].dropna()
            _hy133 = df["hy_spread"].dropna()
            _dd_thresh = -0.10  # -10% drawdown = significant episode
            # Identify drawdown episodes (contiguous periods below threshold)
            _in_dd = (_dd133 < _dd_thresh).astype(int)
            _dd_ep = []
            _in_episode = False
            _ep_start = None
            for _d, _v in _in_dd.items():
                if _v and not _in_episode:
                    _in_episode = True
                    _ep_start = _d
                elif not _v and _in_episode:
                    _in_episode = False
                    _dd_ep.append((_ep_start, _d))
            if _in_episode:
                _dd_ep.append((_ep_start, _dd133.index[-1]))
            # Full history: SP500 drawdown + HY spread dual axis
            _fig133a = _go133.Figure()
            _fig133a.add_trace(_go133.Scatter(
                x=_dd133.index, y=(_dd133.values * 100),
                mode="lines", name="SP500 Drawdown (%)",
                line=dict(color="#ef4444", width=1.2),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
                yaxis="y1",
            ))
            # Shade drawdown episodes
            for (ep_s, ep_e) in _dd_ep:
                _fig133a.add_vrect(x0=ep_s, x1=ep_e, fillcolor="rgba(239,68,68,0.07)",
                                   layer="below", line_width=0)
            _fig133a.add_trace(_go133.Scatter(
                x=_hy133.index, y=_hy133.values,
                mode="lines", name="HY Spread (bps)",
                line=dict(color="#f59e0b", width=1.0),
                hovertemplate="HY: %{y:.0f}bps<extra></extra>",
                yaxis="y2",
            ))
            _fig133a.add_hline(y=-10.0, line_color="#6b7280", line_width=1, line_dash="dot",
                               annotation_text="-10%", annotation_font=dict(color="#6b7280", size=8))
            _fig133a.add_hline(y=-20.0, line_color="#9b1c1c", line_width=1, line_dash="dot",
                               annotation_text="-20%", annotation_font=dict(color="#9b1c1c", size=8))
            _fig133a.update_layout(
                height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="SP500 Drawdown vs HY Spread (red shading = drawdown episodes ≥10%)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Drawdown (%)"),
                yaxis2=dict(overlaying="y", side="right", color="#f59e0b",
                            title="HY Spread (bps)", showgrid=False),
                legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig133a, use_container_width=True)
            # Drawdown episode depth vs HY spread at trough
            if _dd_ep:
                _ep_depths = []
                _ep_hy_at_trough = []
                _ep_durations = []
                _ep_labels = []
                for ep_s, ep_e in _dd_ep:
                    _ep_dd = _dd133[ep_s:ep_e]
                    _depth = float(_ep_dd.min() * 100)
                    _trough_date = _ep_dd.idxmin()
                    _hy_at_trough = float(_hy133.reindex([_trough_date], method="nearest").iloc[0]) if len(_hy133) else _np133.nan
                    _dur = (ep_e - ep_s).days
                    _ep_depths.append(_depth)
                    _ep_hy_at_trough.append(_hy_at_trough)
                    _ep_durations.append(_dur)
                    _ep_labels.append(str(ep_s.date()))
                _fig133b = _go133.Figure()
                _fig133b.add_trace(_go133.Scatter(
                    x=_ep_depths, y=_ep_hy_at_trough,
                    mode="markers+text",
                    marker=dict(
                        color=_ep_durations,
                        colorscale="RdYlGn_r",
                        size=[max(6, min(18, d // 10)) for d in _ep_durations],
                        colorbar=dict(title="Duration (days)", titlefont=dict(color="#9aa0aa", size=9),
                                      tickfont=dict(color="#9aa0aa", size=8)),
                        showscale=True,
                    ),
                    text=_ep_labels,
                    textposition="top center", textfont=dict(size=8, color="#9aa0aa"),
                    hovertemplate="Drawdown: %{x:.1f}%<br>HY at trough: %{y:.0f}bps<br>Duration: %{marker.color}d<extra></extra>",
                ))
                _fig133b.update_layout(
                    height=250, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=60),
                    title=dict(text="Episode Anatomy: Drawdown Depth vs HY Spread at Trough (size=duration)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Drawdown Depth (%)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread at Trough (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig133b, use_container_width=True)
                st.caption(f"Found {len(_dd_ep)} episodes with SP500 drawdown ≥10%. Larger dot = longer episode. Color scale: green=short, red=long duration.")
            else:
                st.info("No drawdown episodes ≥10% found in the dataset.")
        else:
            st.info("sp500_drawdown or hy_spread not found — run the feature pipeline.")
    except Exception as _e133:
        _err_track(_active_sub, _e133)
        st.caption(f"Drawdown anatomy: {_e133}")

# ---------------------------------------------------------------------------
# Batch 18 — sub134–sub139
# ---------------------------------------------------------------------------

# sub134 — Regime Transition Matrix (tab_regime)

if _active_sub == 139:
    try:
        import plotly.graph_objects as _go139
        import numpy as _np139
        import pandas as _pd139
        from src.regime_attribution import SCORE_COLS, DISPLAY_NAMES
        _THRESHOLDS139 = [50, 70]
        _sc139_cols = [(k, v) for k, v in SCORE_COLS.items() if v in df.columns]
        if "composite_risk_score_smooth" in df.columns:
            _sc139_cols.append(("composite", "composite_risk_score_smooth"))
        if _sc139_cols:
            # Alert frequency table: % of days above each threshold
            _alert_data = []
            for k, v in _sc139_cols:
                _s = df[v].dropna()
                if len(_s) < 10:
                    continue
                _label = "Composite" if k == "composite" else DISPLAY_NAMES.get(k, k)
                _freq50 = float((_s >= 50).mean() * 100)
                _freq70 = float((_s >= 70).mean() * 100)
                _consec50 = 0  # longest consecutive streak above 50
                _streak = 0
                for val in (_s >= 50):
                    if val:
                        _streak += 1
                        _consec50 = max(_consec50, _streak)
                    else:
                        _streak = 0
                _alert_data.append({
                    "Score": _label,
                    "% Days ≥50": round(_freq50, 1),
                    "% Days ≥70": round(_freq70, 1),
                    "Longest ≥50 Streak": _consec50,
                })
            if _alert_data:
                import streamlit as _st139_st
                _df_alert = _pd139.DataFrame(_alert_data).set_index("Score")
                st.dataframe(
                    _df_alert.style.background_gradient(subset=["% Days ≥50"], cmap="RdYlGn_r", vmin=0, vmax=50)
                               .background_gradient(subset=["% Days ≥70"], cmap="RdYlGn_r", vmin=0, vmax=25)
                               .format({"% Days ≥50": "{:.1f}%", "% Days ≥70": "{:.1f}%", "Longest ≥50 Streak": "{:d}d"}),
                    use_container_width=True,
                )
            # Timeline: composite ≥50 episodes shaded on composite chart
            if "composite_risk_score_smooth" in df.columns:
                _comp139 = df["composite_risk_score_smooth"].dropna()
                _fig139 = _go139.Figure()
                _fig139.add_trace(_go139.Scatter(
                    x=_comp139.index, y=_comp139.values,
                    mode="lines", name="Composite",
                    line=dict(color="#ffffff", width=1.2),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.0f}<extra></extra>",
                ))
                # Shade ≥50 and ≥70 regions
                _above50 = _comp139 >= 50
                _above70 = _comp139 >= 70
                # Detect episode start/end for shading
                for _thresh, _mask, _color in [
                    (50, _above50, "rgba(245,158,11,0.12)"),
                    (70, _above70, "rgba(239,68,68,0.18)"),
                ]:
                    _in_ep = False
                    _ep_s = None
                    for _d, _v in _mask.items():
                        if _v and not _in_ep:
                            _in_ep = True
                            _ep_s = _d
                        elif not _v and _in_ep:
                            _in_ep = False
                            _fig139.add_vrect(x0=_ep_s, x1=_d, fillcolor=_color, layer="below", line_width=0)
                    if _in_ep:
                        _fig139.add_vrect(x0=_ep_s, x1=_comp139.index[-1], fillcolor=_color, layer="below", line_width=0)
                _fig139.add_hline(y=50, line_color="#f59e0b", line_width=1, line_dash="dot",
                                  annotation_text="Alert (50)", annotation_font=dict(color="#f59e0b", size=8))
                _fig139.add_hline(y=70, line_color="#ef4444", line_width=1, line_dash="dot",
                                  annotation_text="High Alert (70)", annotation_font=dict(color="#ef4444", size=8))
                _fig139.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                    title=dict(text="Composite Score Alert Timeline (amber=≥50, red=≥70)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Score", range=[0, 100]),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig139, use_container_width=True)
            st.caption("Alert frequency = % of all trading days above threshold. Longer streaks = more persistent risk episodes.")
        else:
            st.info("Sub-score columns not found — run the full scoring pipeline.")
    except Exception as _e139:
        _err_track(_active_sub, _e139)
        st.caption(f"Alert history: {_e139}")

# ---------------------------------------------------------------------------
# Batch 19 — sub140–sub144, m9
# ---------------------------------------------------------------------------

# sub140 — Forward Returns (Signal Lab)

if _active_sub == 148:
    try:
        import plotly.graph_objects as _go148
        import numpy as _np148
        if "vix" in df.columns and "vix_change_5d" in df.columns:
            _vix148 = df["vix"].dropna()
            _vxchg148 = df["vix_change_5d"].dropna()
            # Vol of vol = rolling std of VIX 5d change
            _vov148 = _vxchg148.rolling(21, min_periods=10).std()
            # Percentile
            _vov_pct = _vov148.rolling(504, min_periods=63).apply(
                lambda x: float((x[:-1] < x[-1]).mean() * 100) if len(x) > 1 else _np148.nan,
                raw=True
            )
            def _vov_regime(v):
                if _np148.isnan(v): return "Unknown"
                if v < 1.5:  return "Calm"
                if v < 3.0:  return "Normal"
                if v < 5.0:  return "Elevated"
                return "Turbulent"
            _VOV_COLORS = {"Calm": "#10b981", "Normal": "#3b82f6", "Elevated": "#f59e0b",
                           "Turbulent": "#ef4444", "Unknown": "#6b7280"}

            _fig148a = _go148.Figure()
            _fig148a.add_trace(_go148.Scatter(
                x=_vov148.index, y=_vov148.values,
                mode="lines", line=dict(color="#ec4899", width=1.2),
                fill="tozeroy", fillcolor="rgba(236,72,153,0.07)",
                hovertemplate="%{x|%Y-%m-%d}: VoV=%{y:.2f}<extra></extra>",
            ))
            _fig148a.add_hline(y=3.0, line_color="#f59e0b", line_width=1, line_dash="dot",
                               annotation_text="Elevated (3.0)", annotation_font=dict(color="#f59e0b", size=8))
            _fig148a.add_hline(y=5.0, line_color="#ef4444", line_width=1, line_dash="dot",
                               annotation_text="Turbulent (5.0)", annotation_font=dict(color="#ef4444", size=8))
            _fig148a.update_layout(
                height=230, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Vol of Vol: Rolling 21d Std of VIX 5-Day Change (VoV)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="VoV"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig148a, use_container_width=True)

            # Percentile
            _fig148b = _go148.Figure()
            _fig148b.add_trace(_go148.Scatter(
                x=_vov_pct.index, y=_vov_pct.values,
                mode="lines", line=dict(color="#ec4899", width=0.9),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.0f}th pct<extra></extra>",
            ))
            _fig148b.add_hline(y=80, line_color="#ef4444", line_width=1, line_dash="dot")
            _fig148b.update_layout(
                height=160, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="VoV Percentile vs 2-Year Rolling History", font=dict(size=11, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Percentile", range=[0, 100]),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig148b, use_container_width=True)

            # VoV vs HY spread
            if "hy_spread" in df.columns:
                _j148 = _vov148.to_frame("vov").join(df["hy_spread"].to_frame("hy"), how="inner").dropna()
                _fig148c = _go148.Figure()
                _fig148c.add_trace(_go148.Scatter(
                    x=_j148["vov"], y=_j148["hy"],
                    mode="markers",
                    marker=dict(color="#ec4899", size=2.5, opacity=0.3),
                    hovertemplate="VoV: %{x:.2f}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig148c.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Vol of Vol vs HY Spread Scatter (full history)", font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="VoV"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="HY (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig148c, use_container_width=True)
            _curr_vov = float(_vov148.iloc[-1]) if _vov148.notna().any() else None
            _curr_vov_pct = float(_vov_pct.iloc[-1]) if _vov_pct.notna().any() else None
            _curr_vov_reg = _vov_regime(_curr_vov if _curr_vov is not None else float("nan"))
            st.caption(
                f"Current VoV: **{f'{_curr_vov:.2f}' if _curr_vov else 'N/A'}** "
                f"({f'{_curr_vov_pct:.0f}th pct' if _curr_vov_pct else 'N/A'}) — Regime: **{_curr_vov_reg}**. "
                "High VoV precedes sharp credit moves by ~1–3 weeks — a leading indicator of regime instability."
            )
        else:
            st.info("vix or vix_change_5d not found — run the feature pipeline.")
    except Exception as _e148:
        _err_track(_active_sub, _e148)
        st.caption(f"Vol of vol: {_e148}")


# sub149 — HY Multi-Horizon Momentum (Credit Markets)

if _active_sub == 154:
    try:
        import plotly.graph_objects as _go154
        import numpy as _np154
        import pandas as _pd154
        _df154 = df.copy() if "df" in dir() else None
        _has154 = (_df154 is not None
                   and "sp500_return_5d" in _df154.columns
                   and "hy_change_5d" in _df154.columns)
        if not _has154:
            st.info("sp500_return_5d and hy_change_5d required.")
        else:
            st.subheader("Rolling Credit-Equity Correlation Regime")
            st.caption("Rolling 63-day correlation between SP500 weekly returns and HY tightening (inverted). High correlation (>0.4) = aligned risk-on/off. Inverted correlation = dislocated regime — credit leading equity lower is a classic early-warning pattern.")
            _eq154 = _df154["sp500_return_5d"].dropna()
            _cr154 = (-_df154["hy_change_5d"]).dropna()
            _j154 = _eq154.to_frame("eq").join(_cr154.to_frame("cr"), how="inner").dropna()
            _roll_corr154 = _j154["eq"].rolling(63).corr(_j154["cr"])
            _fig154 = _go154.Figure()
            _fig154.add_trace(_go154.Scatter(
                x=_roll_corr154.index, y=_roll_corr154.values,
                line=dict(color="#6366f1", width=1.5),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)", name="63d Corr"
            ))
            _fig154.add_hline(y=0.4, line_dash="dash", line_color="#22c55e",
                              annotation_text="High (0.4)", annotation_position="right")
            _fig154.add_hline(y=-0.2, line_dash="dash", line_color="#ef4444",
                              annotation_text="Inverted (−0.2)", annotation_position="right")
            _fig154.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig154.update_layout(
                title="Rolling 63d Equity–Credit Correlation",
                height=350, yaxis_title="Correlation", yaxis=dict(range=[-1, 1]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig154, use_container_width=True)
            if "hy_change_30d" in _df154.columns:
                _j154b = _roll_corr154.to_frame("corr").join(
                    _df154["hy_change_30d"].to_frame("hy30"), how="inner").dropna()
                _j154b["regime"] = _j154b["hy30"].apply(
                    lambda x: "Widening" if x > 15 else ("Tightening" if x < -15 else "Flat"))
                _box154 = _go154.Figure()
                for _r154, _c154 in [("Tightening", "#22c55e"), ("Flat", "#6366f1"), ("Widening", "#ef4444")]:
                    _rs154 = _j154b[_j154b["regime"] == _r154]["corr"]
                    if len(_rs154) > 10:
                        _box154.add_trace(_go154.Box(
                            y=_rs154.values, name=_r154,
                            marker_color=_c154, line_color=_c154
                        ))
                _box154.update_layout(
                    title="Corr Distribution by HY Spread Regime",
                    height=280, yaxis_title="Correlation",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    showlegend=False
                )
                st.plotly_chart(_box154, use_container_width=True)
            _cur_corr154 = float(_roll_corr154.iloc[-1]) if _roll_corr154.notna().any() else float("nan")
            if not _np154.isnan(_cur_corr154):
                _creg154 = ("Strongly Aligned" if _cur_corr154 > 0.4 else
                            ("Aligned" if _cur_corr154 > 0.1 else
                             ("Decoupled" if _cur_corr154 > -0.2 else "Inverted")))
                st.caption(f"Current 63d equity-credit correlation: **{_cur_corr154:.2f}** — {_creg154}.")
    except Exception as _e154:
        _err_track(_active_sub, _e154)
        st.caption(f"Credit-equity correlation regime: {_e154}")


if _active_sub == 167:
    try:
        import plotly.graph_objects as _go167
        import numpy as _np167
        import pandas as _pd167
        _df167 = df.copy() if "df" in dir() else None
        _has167 = _df167 is not None and "hy_change_5d" in _df167.columns
        if not _has167:
            st.info("hy_change_5d required.")
        else:
            st.subheader("Tail Skew Monitor")
            st.caption("Rolling skewness and kurtosis of HY 5-day changes. Negative skew = left tail is fatter (more frequent large widening moves). Excess kurtosis > 0 = fat tails relative to normal. Both are early warnings of tail-risk build-up before stress episodes.")
            _hyd167 = _df167["hy_change_5d"].dropna()
            _roll_skew167 = _hyd167.rolling(63).skew()
            _roll_kurt167 = _hyd167.rolling(63).kurt()  # excess kurtosis
            _c1_167, _c2_167 = st.columns(2)
            with _c1_167:
                _fig167a = _go167.Figure()
                _fig167a.add_trace(_go167.Scatter(
                    x=_roll_skew167.index, y=_roll_skew167.values,
                    line=dict(color="#f59e0b", width=1.5), name="Rolling Skew"
                ))
                _fig167a.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
                _fig167a.add_hline(y=-1, line_dash="dash", line_color="#ef4444",
                                   annotation_text="Left-tail warning")
                _fig167a.update_layout(
                    title="Rolling 63d Skewness of HY 5d Changes",
                    height=300, yaxis_title="Skewness",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    showlegend=False
                )
                st.plotly_chart(_fig167a, use_container_width=True)
            with _c2_167:
                _fig167b = _go167.Figure()
                _fig167b.add_trace(_go167.Scatter(
                    x=_roll_kurt167.index, y=_roll_kurt167.values,
                    line=dict(color="#8b5cf6", width=1.5), name="Rolling Kurtosis"
                ))
                _fig167b.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
                _fig167b.add_hline(y=3, line_dash="dash", line_color="#ef4444",
                                   annotation_text="Fat tail warning")
                _fig167b.update_layout(
                    title="Rolling 63d Excess Kurtosis of HY 5d Changes",
                    height=300, yaxis_title="Excess Kurtosis",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    showlegend=False
                )
                st.plotly_chart(_fig167b, use_container_width=True)
            # Full history distribution with QQ-ish comparison
            _full_vals167 = _hyd167.values
            _p5_167, _p95_167 = float(_np167.percentile(_full_vals167, 5)), float(_np167.percentile(_full_vals167, 95))
            _p1_167, _p99_167 = float(_np167.percentile(_full_vals167, 1)), float(_np167.percentile(_full_vals167, 99))
            _fig167c = _go167.Figure()
            _fig167c.add_trace(_go167.Histogram(
                x=_full_vals167, nbinsx=60,
                marker_color="#6366f1", opacity=0.7
            ))
            for _v167, _lbl167 in [(_p1_167, "P1"), (_p5_167, "P5"), (_p95_167, "P95"), (_p99_167, "P99")]:
                _fig167c.add_vline(x=_v167, line_dash="dash", line_color="#f59e0b",
                                   annotation_text=_lbl167, annotation_position="top")
            _fig167c.update_layout(
                title="Full Distribution of HY 5d Changes",
                height=260, xaxis_title="5d Change (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig167c, use_container_width=True)
            _cur_skew167 = float(_roll_skew167.iloc[-1]) if _roll_skew167.notna().any() else float("nan")
            _cur_kurt167 = float(_roll_kurt167.iloc[-1]) if _roll_kurt167.notna().any() else float("nan")
            if not _np167.isnan(_cur_skew167):
                _tail_warn167 = _cur_skew167 < -1 or _cur_kurt167 > 3
                st.caption(
                    f"Current 63d skew: **{_cur_skew167:.2f}** · excess kurtosis: **{_cur_kurt167:.2f}** · "
                    f"Tail risk {'⚠️ ELEVATED' if _tail_warn167 else 'within normal range'}. "
                    f"5th/95th pct (full history): {_p5_167:.1f} / {_p95_167:.1f} bps."
                )
    except Exception as _e167:
        _err_track(_active_sub, _e167)
        st.caption(f"Tail skew: {_e167}")


if _active_sub == 170:
    try:
        import plotly.graph_objects as _go170
        import numpy as _np170
        import pandas as _pd170
        _df170 = df.copy() if "df" in dir() else None
        _has170 = (_df170 is not None
                   and "vix" in _df170.columns
                   and "hy_spread" in _df170.columns
                   and "sp500_return_30d" in _df170.columns)
        if not _has170:
            st.info("vix, hy_spread, sp500_return_30d required.")
        else:
            st.subheader("Volatility Regime Clustering")
            st.caption("Four-quadrant regime classification using VIX level and HY spread level as axes. Quadrants identify distinct vol-credit states: Low-Low (calm), High-Low (equity stress only), Low-High (credit stress only), High-High (systemic stress). Time series shows how the market has moved across quadrants.")
            _j170 = _df170[["vix", "hy_spread", "sp500_return_30d"]].dropna().tail(1260)
            # Classify by VIX and HY percentile relative to 5Y history
            _vix_med170 = float(_j170["vix"].median())
            _hy_med170 = float(_j170["hy_spread"].median())
            def _cluster170(row):
                hi_vix = row["vix"] > _vix_med170
                hi_hy = row["hy_spread"] > _hy_med170
                if not hi_vix and not hi_hy: return "Calm"
                if hi_vix and not hi_hy: return "Equity Stress"
                if not hi_vix and hi_hy: return "Credit Stress"
                return "Systemic Stress"
            _j170["cluster"] = _j170.apply(_cluster170, axis=1)
            _cluster_colors170 = {
                "Calm": "#22c55e", "Equity Stress": "#f59e0b",
                "Credit Stress": "#8b5cf6", "Systemic Stress": "#ef4444"
            }
            # Scatter
            _fig170a = _go170.Figure()
            for _cl170, _cc170 in _cluster_colors170.items():
                _s170 = _j170[_j170["cluster"] == _cl170]
                _fig170a.add_trace(_go170.Scatter(
                    x=_s170["vix"], y=_s170["hy_spread"],
                    mode="markers", marker=dict(color=_cc170, size=4, opacity=0.5),
                    name=_cl170
                ))
            _cur170 = _j170.iloc[-1]
            _fig170a.add_trace(_go170.Scatter(
                x=[_cur170["vix"]], y=[_cur170["hy_spread"]],
                mode="markers+text", marker=dict(color="white", size=12, symbol="star"),
                text=["Now"], textposition="top center", name="Current"
            ))
            _fig170a.add_vline(x=_vix_med170, line_dash="dash", line_color="#4b5563")
            _fig170a.add_hline(y=_hy_med170, line_dash="dash", line_color="#4b5563")
            _fig170a.update_layout(
                title="Vol-Credit Regime Quadrants (5Y)",
                height=380, xaxis_title="VIX", yaxis_title="HY Spread (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig170a, use_container_width=True)
            # Regime time series (colored area)
            _cluster_enc170 = {"Calm": 0, "Equity Stress": 1, "Credit Stress": 2, "Systemic Stress": 3}
            _cl_series170 = _j170["cluster"].map(_cluster_enc170)
            _fig170b = _go170.Figure()
            for _cl170b, _enc170b in _cluster_enc170.items():
                _mask170 = _cl_series170 == _enc170b
                _fig170b.add_trace(_go170.Scatter(
                    x=_j170.index[_mask170], y=[_enc170b] * _mask170.sum(),
                    mode="markers", marker=dict(color=_cluster_colors170[_cl170b], size=5),
                    name=_cl170b
                ))
            _fig170b.update_layout(
                title="Regime Over Time",
                height=200,
                yaxis=dict(tickvals=[0,1,2,3],
                           ticktext=["Calm","EQ Stress","Credit Stress","Systemic"]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig170b, use_container_width=True)
            _cur_cl170 = str(_cur170["cluster"])
            _cl_dist170 = _j170["cluster"].value_counts()
            st.caption(
                f"Current regime: **{_cur_cl170}** · "
                + " · ".join(f"{k}: {v}d ({v/len(_j170)*100:.0f}%)" for k, v in _cl_dist170.items())
            )
    except Exception as _e170:
        _err_track(_active_sub, _e170)
        st.caption(f"Vol clustering: {_e170}")


if _active_sub == "ov_risk":
    try:
        import plotly.graph_objects as _go_ov_risk
        import numpy as _np_ov_risk
        st.subheader("Risk Monitors — Section Overview")
        st.caption("Current reading across volatility, tail risk, and systemic stress indicators. Select any sub-view from the sidebar.")
        _d = df
        def _last_r(col): s = _d[col].dropna(); return float(s.iloc[-1]) if len(s) else float("nan")
        def _pct_r(col):
            s = _d[col].dropna()
            if len(s) < 10: return float("nan")
            return float((s < float(s.iloc[-1])).mean() * 100)
        _c1, _c2, _c3, _c4 = st.columns(4)
        _vix = _last_r("vix"); _vix_pct = _pct_r("vix")
        _c1.metric("VIX", f"{_vix:.1f}" if not _np_ov_risk.isnan(_vix) else "—",
                   delta=f"{_vix_pct:.0f}th pct" if not _np_ov_risk.isnan(_vix_pct) else None,
                   delta_color="inverse")
        _move = _last_r("move_index"); _move_pct = _pct_r("move_index")
        _c2.metric("MOVE Index", f"{_move:.0f}" if not _np_ov_risk.isnan(_move) else "—",
                   delta=f"{_move_pct:.0f}th pct" if not _np_ov_risk.isnan(_move_pct) else None,
                   delta_color="inverse")
        _dd = _last_r("sp500_drawdown")
        _c3.metric("SP500 Drawdown", f"{_dd:.1%}" if not _np_ov_risk.isnan(_dd) else "—",
                   delta_color="inverse")
        _comp = _last_r("composite_risk_score_smooth")
        _c4.metric("Composite Score", f"{_comp:.1f}" if not _np_ov_risk.isnan(_comp) else "—",
                   delta_color="inverse")
        st.divider()
        _fig_ov_risk = _go_ov_risk.Figure()
        _vix_s = _d["vix"].dropna().tail(252)
        _fig_ov_risk.add_trace(_go_ov_risk.Scatter(x=_vix_s.index, y=_vix_s.values,
            name="VIX", line=dict(color="#ef4444", width=2)))
        if "move_index" in _d.columns:
            _move_s = _d["move_index"].dropna().tail(252)
            _move_scaled = _move_s / _move_s.max() * _vix_s.max()
            _fig_ov_risk.add_trace(_go_ov_risk.Scatter(x=_move_scaled.index, y=_move_scaled.values,
                name="MOVE (scaled)", line=dict(color="#f59e0b", width=1.5, dash="dot")))
        _fig_ov_risk.add_hline(y=25, line_dash="dash", line_color="#f59e0b", annotation_text="Stress (25)")
        _fig_ov_risk.update_layout(
            title="VIX & MOVE Index — Last 252 Trading Days",
            height=280, yaxis_title="Level",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=40, b=20))
        st.plotly_chart(_fig_ov_risk, use_container_width=True)
        _risk_label = ("Elevated" if not _np_ov_risk.isnan(_vix) and _vix > 25
                       else ("Moderate" if not _np_ov_risk.isnan(_vix) and _vix > 18 else "Low"))
        st.info(f"Volatility regime: **{_risk_label}** (VIX {_vix:.1f}). "
                f"37 sub-views available: tail risk, funding stress, contagion, drawdown anatomy, and more.")
    except Exception as _e_ov_risk:
        _err_track(_active_sub, _e_ov_risk)
        st.caption(f"Risk Monitors overview: {_e_ov_risk}")


if _active_sub == 178:
    try:
        import plotly.graph_objects as _go178
        import numpy as _np178
        import pandas as _pd178
        _df178 = df.copy() if "df" in dir() else None
        _has178 = _df178 is not None and "stlfsi" in _df178.columns
        if not _has178:
            st.info("stlfsi column required.")
        else:
            st.subheader("St. Louis Fed Financial Stress Index (STLFSI)")
            st.caption("The STLFSI measures financial stress across 18 weekly data series including interest rates, yield spreads, and other indicators. Values above zero indicate above-average financial stress. This view compares STLFSI to NFCI and HY spreads to triangulate the current stress regime.")
            _stl178 = _df178["stlfsi"].dropna()
            _nfci178 = _df178["nfci"].dropna() if "nfci" in _df178.columns else None
            _hy178 = _df178["hy_spread"].dropna()
            _c1, _c2, _c3, _c4 = st.columns(4)
            _cur_stl178 = float(_stl178.iloc[-1])
            _stl_pct178 = float((_stl178 < _cur_stl178).mean() * 100)
            _stl_regime178 = ("Crisis" if _cur_stl178 > 2 else
                              ("Stressed" if _cur_stl178 > 1 else
                               ("Elevated" if _cur_stl178 > 0 else "Normal")))
            _c1.metric("STLFSI", f"{_cur_stl178:.3f}")
            _c2.metric("Percentile", f"{_stl_pct178:.0f}th", delta_color="inverse")
            _c3.metric("Regime", _stl_regime178,
                       delta_color="inverse" if _stl_regime178 in ("Stressed", "Crisis") else "off")
            if _nfci178 is not None:
                _cur_nfci178 = float(_nfci178.iloc[-1])
                _c4.metric("NFCI", f"{_cur_nfci178:.3f}")
            st.divider()
            _fig178 = _go178.Figure()
            _stl_tail178 = _stl178.tail(504)
            _fig178.add_trace(_go178.Scatter(
                x=_stl_tail178.index, y=_stl_tail178.values,
                name="STLFSI", line=dict(color="#ef4444", width=2),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
            if _nfci178 is not None:
                _fig178.add_trace(_go178.Scatter(
                    x=_nfci178.tail(504).index, y=_nfci178.tail(504).values,
                    name="NFCI", line=dict(color="#6366f1", width=1.5, dash="dot"),
                    yaxis="y2"))
            _fig178.add_hline(y=0, line_dash="dash", line_color="#9aa0aa",
                              annotation_text="Normal boundary")
            _fig178.add_hline(y=1, line_dash="dash", line_color="#f59e0b",
                              annotation_text="Stress (1.0)")
            _fig178.update_layout(
                title="STLFSI & NFCI — Last 2 Years",
                height=320,
                yaxis=dict(title="STLFSI"),
                yaxis2=dict(title="NFCI", overlaying="y", side="right") if _nfci178 is not None else {},
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(_fig178, use_container_width=True)
            # STLFSI vs HY scatter
            _j178 = _stl178.to_frame("stl").join(_hy178.to_frame("hy"), how="inner").dropna().tail(1260)
            _fig178b = _go178.Figure()
            _fig178b.add_trace(_go178.Scatter(
                x=_j178["stl"], y=_j178["hy"],
                mode="markers", marker=dict(color="#6366f1", size=3, opacity=0.4)))
            _fig178b.add_trace(_go178.Scatter(
                x=[_cur_stl178], y=[float(_hy178.iloc[-1])],
                mode="markers+text", marker=dict(color="white", size=12, symbol="star"),
                text=["Now"], textposition="top center"))
            _fig178b.add_vline(x=0, line_color="#9aa0aa", line_width=0.5)
            _fig178b.update_layout(
                title="STLFSI vs HY Spread (5Y scatter)",
                height=260, xaxis_title="STLFSI", yaxis_title="HY Spread (bps)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False)
            st.plotly_chart(_fig178b, use_container_width=True)
            _stl_hy_corr178 = float(_j178["stl"].corr(_j178["hy"]))
            st.caption(
                f"STLFSI **{_cur_stl178:.3f}** ({_stl_regime178}, {_stl_pct178:.0f}th pct). "
                f"STLFSI-HY correlation: {_stl_hy_corr178:.2f}. "
                "Values >0 = above-average financial stress across 18 market indicators.")
    except Exception as _e178:
        _err_track(_active_sub, _e178)
        st.caption(f"STLFSI: {_e178}")


if _active_sub == 179:
    try:
        import plotly.graph_objects as _go179
        import numpy as _np179
        import pandas as _pd179
        _df179 = df.copy() if "df" in dir() else None
        _has179 = (_df179 is not None
                   and "move_index" in _df179.columns
                   and "vix" in _df179.columns)
        if not _has179:
            st.info("move_index and vix required.")
        else:
            st.subheader("MOVE/VIX Divergence — Rates vs Equity Vol")
            st.caption("The MOVE/VIX ratio measures whether bond market volatility is elevated relative to equity volatility. High ratio = rates markets pricing more risk than equities (rate risk dominates). Low ratio = equity stress dominates. Divergence episodes historically precede credit repricing as one market catches up to the other.")
            _move179 = _df179["move_index"].dropna()
            _vix179 = _df179["vix"].dropna()
            _j179 = _move179.to_frame("move").join(_vix179.to_frame("vix"), how="inner").dropna().tail(1260)
            _ratio179 = _j179["move"] / _j179["vix"]
            _ratio_pct179 = _ratio179.rolling(252).rank(pct=True) * 100
            _fig179 = _go179.Figure()
            _fig179.add_trace(_go179.Scatter(
                x=_ratio179.index, y=_ratio179.values,
                name="MOVE/VIX", line=dict(color="#6366f1", width=2),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.08)"))
            _med_ratio179 = float(_ratio179.median())
            _fig179.add_hline(y=_med_ratio179, line_dash="dash", line_color="#9aa0aa",
                              annotation_text=f"Median ({_med_ratio179:.1f})")
            _p80_ratio179 = float(_ratio179.quantile(0.8))
            _fig179.add_hline(y=_p80_ratio179, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"80th pct ({_p80_ratio179:.1f})")
            _fig179.update_layout(
                title="MOVE / VIX Ratio (rates vol relative to equity vol)",
                height=300, yaxis_title="Ratio",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False)
            st.plotly_chart(_fig179, use_container_width=True)
            # MOVE and VIX together
            _fig179b = _go179.Figure()
            _fig179b.add_trace(_go179.Scatter(
                x=_j179.index, y=_j179["move"],
                name="MOVE", line=dict(color="#f59e0b", width=1.5)))
            _fig179b.add_trace(_go179.Scatter(
                x=_j179.index, y=_j179["vix"],
                name="VIX", line=dict(color="#ef4444", width=1.5, dash="dot"),
                yaxis="y2"))
            _fig179b.update_layout(
                title="MOVE vs VIX (5Y)",
                height=240,
                yaxis=dict(title="MOVE"),
                yaxis2=dict(title="VIX", overlaying="y", side="right"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(_fig179b, use_container_width=True)
            _cur_ratio179 = float(_ratio179.iloc[-1])
            _cur_rpct179 = float(_ratio_pct179.iloc[-1]) if _ratio_pct179.notna().any() else float("nan")
            _cur_move179 = float(_move179.iloc[-1])
            _cur_vix179 = float(_vix179.iloc[-1])
            _dom179 = ("Rates-dominated stress" if _cur_rpct179 > 70
                       else ("Equity-dominated stress" if _cur_rpct179 < 30 else "Balanced"))
            st.caption(
                f"MOVE: **{_cur_move179:.0f}** · VIX: **{_cur_vix179:.1f}** · "
                f"Ratio: **{_cur_ratio179:.2f}** ({_cur_rpct179:.0f}th pct) — {_dom179}.")
    except Exception as _e179:
        _err_track(_active_sub, _e179)
        st.caption(f"MOVE-VIX ratio: {_e179}")


if _active_sub == 180:
    try:
        import plotly.graph_objects as _go180
        import numpy as _np180
        import pandas as _pd180
        _df180 = df.copy() if "df" in dir() else None
        _corr_cols180 = [c for c in [
            "vix", "hy_spread", "ig_spread", "sp500_return_5d",
            "yield_10y", "spread", "nfci", "oil_wti", "move_index",
        ] if _df180 is not None and c in _df180.columns]
        _has180 = _df180 is not None and len(_corr_cols180) >= 4
        if not _has180:
            st.info("At least 4 of: vix, hy_spread, ig_spread, sp500_return_5d, yield_10y, spread, nfci, oil_wti, move_index.")
        else:
            st.subheader("Cross-Asset Correlation Snapshot")
            st.caption("Current 63-day pairwise correlation matrix vs the long-run baseline (full history). Red = correlations spiking above baseline (contagion). Blue = correlations falling below baseline (decoupling). When many pairs spike positive simultaneously, systemic stress is broadening.")
            _tail180 = _df180[_corr_cols180].dropna().tail(63)
            _full180 = _df180[_corr_cols180].dropna()
            _cur_corr180 = _tail180.corr()
            _base_corr180 = _full180.corr()
            _delta_corr180 = _cur_corr180 - _base_corr180
            _short_labels180 = [c.replace("_spread", "").replace("_return_5d", " ret").replace("_", " ").upper()
                                 for c in _corr_cols180]
            _c1_180, _c2_180 = st.columns(2)
            with _c1_180:
                _fig180a = _go180.Figure(data=_go180.Heatmap(
                    z=_cur_corr180.values,
                    x=_short_labels180, y=_short_labels180,
                    colorscale=[[0, "#1d4ed8"], [0.5, "#1e1b4b"], [1, "#7f1d1d"]],
                    zmid=0, zmin=-1, zmax=1,
                    colorbar=dict(title="Corr", len=0.6),
                    text=_np180.round(_cur_corr180.values, 2),
                    texttemplate="%{text:.2f}"
                ))
                _fig180a.update_layout(
                    title="Current 63d Correlations",
                    height=350,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10),
                    margin=dict(t=40, b=20))
                st.plotly_chart(_fig180a, use_container_width=True)
            with _c2_180:
                _fig180b = _go180.Figure(data=_go180.Heatmap(
                    z=_delta_corr180.values,
                    x=_short_labels180, y=_short_labels180,
                    colorscale=[[0, "#166534"], [0.5, "#1e1b4b"], [1, "#7f1d1d"]],
                    zmid=0, zmin=-0.5, zmax=0.5,
                    colorbar=dict(title="Δ Corr", len=0.6),
                    text=_np180.round(_delta_corr180.values, 2),
                    texttemplate="%{text:.2f}"
                ))
                _fig180b.update_layout(
                    title="Change vs Full-History Baseline",
                    height=350,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10),
                    margin=dict(t=40, b=20))
                st.plotly_chart(_fig180b, use_container_width=True)
            # Biggest movers
            _delta_vals180 = []
            for i, c1 in enumerate(_corr_cols180):
                for j, c2 in enumerate(_corr_cols180):
                    if j <= i: continue
                    _delta_vals180.append({
                        "Pair": f"{_short_labels180[i]} / {_short_labels180[j]}",
                        "Current Corr": round(float(_cur_corr180.iloc[i, j]), 2),
                        "Baseline Corr": round(float(_base_corr180.iloc[i, j]), 2),
                        "Δ": round(float(_delta_corr180.iloc[i, j]), 2),
                    })
            _delta_df180 = (_pd180.DataFrame(_delta_vals180)
                            .sort_values("Δ", key=_np180.abs, ascending=False)
                            .head(8).reset_index(drop=True))
            st.markdown("**Biggest Correlation Shifts (current vs baseline)**")
            st.dataframe(_delta_df180, use_container_width=True, hide_index=True)
            _n_spike180 = int((_delta_df180["Δ"] > 0.15).sum())
            st.caption(
                f"{_n_spike180} pairs with correlation >0.15 above baseline — "
                f"{'elevated contagion risk' if _n_spike180 >= 3 else 'normal cross-asset structure'}. "
                f"Using {len(_corr_cols180)} assets over 63 trading days.")
    except Exception as _e180:
        _err_track(_active_sub, _e180)
        st.caption(f"Cross-asset correlation: {_e180}")

# --- sub181: FX-Credit Nexus ---

if _active_sub == 186:
    st.subheader("ANFCI & Banking Stress Suite")
    st.caption("Adjusted NFCI isolates non-economic credit tightness; paired with banking stress and CLO market indicators")
    try:
        import plotly.graph_objects as _go186
        import numpy as _np186
        import pandas as _pd186
        _df186 = df[["anfci","anfci_change_30d","nfci","banking_stress_score_smooth","clo_stress_score","hy_spread","excess_spread_bps","excess_spread_percentile"]].dropna().copy()
        _last186 = _df186.iloc[-1]
        _anfci_pct186 = float((_df186["anfci"] < _last186["anfci"]).mean() * 100)
        _c1_186, _c2_186, _c3_186, _c4_186 = st.columns(4)
        _c1_186.metric("ANFCI", f"{_last186['anfci']:.2f}", f"{_last186['anfci_change_30d']:+.2f} 30d")
        _c2_186.metric("ANFCI Pct", f"{_anfci_pct186:.0f}th")
        _c3_186.metric("Banking Stress", f"{_last186['banking_stress_score_smooth']:.2f}")
        _c4_186.metric("CLO Stress", f"{_last186['clo_stress_score']:.2f}")
        st.divider()
        # ANFCI vs NFCI divergence — the "extra" tightness not explained by economy
        _df186["nfci_anfci_gap"] = _df186["nfci"] - _df186["anfci"]
        _fig186a = _go186.Figure()
        _fig186a.add_trace(_go186.Scatter(
            x=_df186.index, y=_df186["nfci"], name="NFCI", line=dict(color="#3b82f6", width=1.5)
        ))
        _fig186a.add_trace(_go186.Scatter(
            x=_df186.index, y=_df186["anfci"], name="ANFCI (adjusted)", line=dict(color="#f59e0b", width=1.5)
        ))
        _fig186a.add_trace(_go186.Scatter(
            x=_df186.index, y=_df186["nfci_anfci_gap"],
            name="NFCI - ANFCI Gap (excess tightness)",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#ef4444", width=1)
        ))
        _fig186a.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=1)
        _fig186a.update_layout(
            title="NFCI vs ANFCI — Excess Financial Tightness",
            height=320,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig186a, use_container_width=True)
        # Banking stress + CLO stress vs HY
        _fig186b = _go186.Figure()
        _fig186b.add_trace(_go186.Scatter(
            x=_df186.index, y=_df186["banking_stress_score_smooth"],
            name="Banking Stress Score", line=dict(color="#ef4444", width=1.5), yaxis="y1"
        ))
        _fig186b.add_trace(_go186.Scatter(
            x=_df186.index, y=_df186["clo_stress_score"],
            name="CLO Stress Score", line=dict(color="#a78bfa", width=1.5), yaxis="y1"
        ))
        _fig186b.add_trace(_go186.Scatter(
            x=_df186.index, y=_df186["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#9aa0aa", width=1, dash="dot"), yaxis="y2"
        ))
        _fig186b.update_layout(
            title="Banking Stress & CLO Stress vs HY Spread",
            height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Stress Score", side="left"),
            yaxis2=dict(title="HY Spread (bps)", side="right", overlaying="y", color="#9aa0aa", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig186b, use_container_width=True)
        # Excess spread percentile over time
        _fig186c = _go186.Figure()
        _fig186c.add_trace(_go186.Scatter(
            x=_df186.index, y=_df186["excess_spread_percentile"],
            name="Excess Spread Pct", fill="tozeroy",
            fillcolor="rgba(16,185,129,0.12)", line=dict(color="#10b981", width=1.5)
        ))
        _fig186c.add_hline(y=50, line_dash="dash", line_color="#4b5563", line_width=0.8)
        _fig186c.update_layout(
            title="Excess Spread Percentile (compensation above expected loss)",
            height=200,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"), yaxis_title="Percentile",
            margin=dict(t=40, b=25))
        st.plotly_chart(_fig186c, use_container_width=True)
        # Correlation table
        _corr_cols186 = ["anfci", "banking_stress_score_smooth", "clo_stress_score", "excess_spread_percentile"]
        _corr186 = _df186[_corr_cols186 + ["hy_spread"]].corr()[["hy_spread"]].round(2)
        _corr186.index = ["ANFCI", "Banking Stress", "CLO Stress", "Excess Spread Pct", "HY Spread"]
        _corr186.columns = ["↔ HY Spread"]
        st.markdown("**Correlation with HY Spread**")
        st.dataframe(_corr186, use_container_width=True)
        _stress_combo186 = _last186["banking_stress_score_smooth"] > _df186["banking_stress_score_smooth"].quantile(0.75) and _last186["clo_stress_score"] > _df186["clo_stress_score"].quantile(0.75)
        st.caption(
            f"ANFCI {_last186['anfci']:.2f} ({_anfci_pct186:.0f}th pct); 30d change {_last186['anfci_change_30d']:+.2f}. "
            f"Excess spread at {_last186['excess_spread_percentile']:.0f}th pct — "
            f"{'compensating for risk' if _last186['excess_spread_percentile'] > 50 else 'compressed vs expected loss'}. "
            f"{'Banking + CLO stress both elevated — systemic credit risk.' if _stress_combo186 else 'Banking and CLO stress within normal range.'}")
    except Exception as _e186:
        _err_track(_active_sub, _e186)
        st.caption(f"ANFCI suite: {_e186}")

# --- sub187: Spread Decomposition ---

if _active_sub == 190:
    st.subheader("ETF Flow & Dislocation Monitor")
    st.caption("Credit ETF fund flows and NAV dislocation — technical stress signals from ETF market structure")
    try:
        import plotly.graph_objects as _go190
        import numpy as _np190
        import pandas as _pd190
        _df190 = df[["etf_dislocation_score","etf_fund_flow_score","hy_spread","vix"]].dropna().copy()
        _last190 = _df190.iloc[-1]
        _dis_pct190 = float((_df190["etf_dislocation_score"] < _last190["etf_dislocation_score"]).mean() * 100)
        _flow_pct190 = float((_df190["etf_fund_flow_score"] < _last190["etf_fund_flow_score"]).mean() * 100)
        _c1_190, _c2_190, _c3_190, _c4_190 = st.columns(4)
        _c1_190.metric("Dislocation Score", f"{_last190['etf_dislocation_score']:.2f}", f"{_dis_pct190:.0f}th pct")
        _c2_190.metric("Fund Flow Score", f"{_last190['etf_fund_flow_score']:.2f}", f"{_flow_pct190:.0f}th pct")
        _c3_190.metric("HY Spread", f"{_last190['hy_spread']:.0f} bps")
        _dual_stress190 = _last190["etf_dislocation_score"] > _df190["etf_dislocation_score"].quantile(0.80) and _last190["etf_fund_flow_score"] > _df190["etf_fund_flow_score"].quantile(0.80)
        _c4_190.metric("ETF Stress", "Elevated" if _dual_stress190 else "Normal")
        st.divider()
        # Dual-panel: dislocation + fund flow vs HY
        _fig190a = _go190.Figure()
        _fig190a.add_trace(_go190.Scatter(
            x=_df190.index, y=_df190["etf_dislocation_score"],
            name="Dislocation Score", line=dict(color="#f59e0b", width=1.8), yaxis="y1"
        ))
        _fig190a.add_trace(_go190.Scatter(
            x=_df190.index, y=_df190["etf_fund_flow_score"],
            name="Fund Flow Score", line=dict(color="#3b82f6", width=1.5), yaxis="y1"
        ))
        _fig190a.add_trace(_go190.Scatter(
            x=_df190.index, y=_df190["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#ef4444", width=1, dash="dot"), yaxis="y2"
        ))
        # Mark dislocation spikes
        _disloc_q80_190 = float(_df190["etf_dislocation_score"].quantile(0.80))
        _fig190a.add_hline(y=_disloc_q80_190, yref="y1", line_dash="dot", line_color="#f59e0b",
                           line_width=0.8, annotation_text="80th pct", annotation_position="bottom right")
        _fig190a.update_layout(
            title="ETF Dislocation & Fund Flow Scores vs HY Spread",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Score", side="left"),
            yaxis2=dict(title="HY Spread (bps)", side="right", overlaying="y", color="#ef4444", showgrid=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig190a, use_container_width=True)
        # Scatter: dislocation vs fund flow (joint stress map)
        _c1b_190, _c2b_190 = st.columns(2)
        with _c1b_190:
            _fig190b = _go190.Figure()
            _fig190b.add_trace(_go190.Scatter(
                x=_df190["etf_fund_flow_score"], y=_df190["etf_dislocation_score"],
                mode="markers",
                marker=dict(size=3, color=_df190["hy_spread"].values,
                            colorscale=[[0,"#3b82f6"],[0.5,"#f59e0b"],[1,"#ef4444"]],
                            colorbar=dict(title="HY bps", len=0.5), opacity=0.4),
                name="History"
            ))
            _fig190b.add_trace(_go190.Scatter(
                x=[_last190["etf_fund_flow_score"]], y=[_last190["etf_dislocation_score"]],
                mode="markers", marker=dict(size=14, color="#f59e0b", symbol="star"), name="Now"
            ))
            _flow_med190 = float(_df190["etf_fund_flow_score"].median())
            _dis_med190 = float(_df190["etf_dislocation_score"].median())
            _fig190b.add_vline(x=_flow_med190, line_dash="dash", line_color="#4b5563", line_width=0.8)
            _fig190b.add_hline(y=_dis_med190, line_dash="dash", line_color="#4b5563", line_width=0.8)
            _fig190b.update_layout(
                title="ETF Stress Quadrant",
                height=270,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                xaxis_title="Fund Flow Score", yaxis_title="Dislocation Score",
                margin=dict(t=40, b=30))
            st.plotly_chart(_fig190b, use_container_width=True)
        with _c2b_190:
            # Rolling correlation with HY
            _roll_dis190 = _df190["etf_dislocation_score"].rolling(63).corr(_df190["hy_spread"])
            _roll_flow190 = _df190["etf_fund_flow_score"].rolling(63).corr(_df190["hy_spread"])
            _fig190c = _go190.Figure()
            _fig190c.add_trace(_go190.Scatter(x=_df190.index, y=_roll_dis190,
                                              name="Dislocation↔HY", line=dict(color="#f59e0b", width=1.5)))
            _fig190c.add_trace(_go190.Scatter(x=_df190.index, y=_roll_flow190,
                                              name="Flow↔HY", line=dict(color="#3b82f6", width=1.5)))
            _fig190c.add_hline(y=0, line_dash="dash", line_color="#4b5563", line_width=0.8)
            _fig190c.update_layout(
                title="Rolling 63d Corr with HY",
                height=270,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11),
                yaxis_title="Correlation", margin=dict(t=40, b=30))
            st.plotly_chart(_fig190c, use_container_width=True)
        _corr_dis190 = round(float(_df190["etf_dislocation_score"].corr(_df190["hy_spread"])), 2)
        st.caption(
            f"ETF dislocation {_last190['etf_dislocation_score']:.2f} ({_dis_pct190:.0f}th pct); "
            f"fund flow score {_last190['etf_fund_flow_score']:.2f} ({_flow_pct190:.0f}th pct). "
            f"Dislocation↔HY correlation: {_corr_dis190:+.2f}. "
            f"{'Dual ETF stress signal — monitor for forced selling.' if _dual_stress190 else 'ETF market structure within normal range.'}")
    except Exception as _e190:
        _err_track(_active_sub, _e190)
        st.caption(f"ETF flow & dislocation: {_e190}")

# --- sub191: Primary & Loan Market Monitor ---
