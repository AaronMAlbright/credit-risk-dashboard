"""
Signal Lab — analytics section page.
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
    page_title='Signal Lab — Credit Risk Dashboard',
    page_icon='🔬',
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
_SECTION_NAME = 'Signal Lab'
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


if _active_sub == 1:
    wf_windows, wf_regimes = load_walk_forward()
    render_validation_section(
        df,
        wf_windows,
        wf_regimes,
        load_regime_transition,
        load_validation_audit,
        load_bootstrap,
        check_missing_values,
        check_score_bounds,
        check_sample_sizes,
        _cfg_equity_floor,
        _cfg_equity_cap,
        _cfg_target_vol,
        _cfg_momentum_lookback,
    )

if _active_section == "Backtest":
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

if _active_section == "History":
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


if _active_sub == 2:
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


if _active_sub == 3:
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


if _active_sub == 4:
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


if _active_sub == 5:
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


if _active_sub == 9:
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


if _active_sub == 18:
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

if _active_sub == 22:
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

if _active_sub == 26:
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

if _active_sub == 35:
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

if _active_sub == 37:
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

if _active_sub == 50:
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

if _active_sub == 91:
    import plotly.graph_objects as _go91
    st.header("Data Quality Diagnostics")
    st.markdown(
        "Automated checks on the dataset powering this dashboard: "
        "date continuity, duplicate rows, missing value rates, and lookahead leakage detection. "
        "Run after the pipeline assembles data but before scoring. "
        "Green = passed · Red = issue found."
    )
    try:
        _dq91 = load_data_diagnostics(df)
        _overall91 = _dq91.get("overall_passed", False)
        _nr91 = _dq91.get("n_rows", 0)
        _dr91 = _dq91.get("date_range", {})

        if _overall91:
            st.success(f"All checks passed · {_nr91:,} rows · "
                       f"{_dr91.get('start', '—')} → {_dr91.get('end', '—')}")
        else:
            st.warning(f"One or more checks flagged · {_nr91:,} rows · "
                       f"{_dr91.get('start', '—')} → {_dr91.get('end', '—')}")

        # Date continuity
        _dc91 = _dq91.get("date_continuity", {})
        st.subheader("Date Continuity")
        _dc91a, _dc91b, _dc91c = st.columns(3)
        _dc91a.metric("Status", "Pass" if _dc91.get("passed") else "FAIL",
                      delta_color="normal" if _dc91.get("passed") else "inverse")
        _dc91b.metric("Gaps Found", _dc91.get("n_gaps", 0),
                      help="Business-day gaps > 5 days (excluding holidays)")
        _dc91c.metric("Max Gap (days)", _dc91.get("max_gap_days", 0))
        if _dc91.get("gaps"):
            with st.expander(f"{len(_dc91['gaps'])} gap(s) detected"):
                import pandas as _pd91a
                st.dataframe(_pd91a.DataFrame(_dc91["gaps"]), use_container_width=True, hide_index=True)

        # Duplicates
        _dup91 = _dq91.get("duplicates", {})
        st.subheader("Duplicate Rows")
        _dup91a, _dup91b = st.columns(2)
        _dup91a.metric("Status", "Pass" if _dup91.get("passed") else "FAIL")
        _dup91b.metric("Duplicate Count", _dup91.get("n_duplicates", 0))

        # Missing values
        _mv91 = _dq91.get("missing_values", {})
        st.subheader("Missing Values")
        _mv91a, _mv91b = st.columns(2)
        _mv91a.metric("Status", "Pass" if _mv91.get("passed") else "FAIL")
        _mv91b.metric("Columns with > 20% Missing", _mv91.get("n_high_missing", 0))
        _high91 = _mv91.get("high_missing_cols", {})
        if _high91:
            import pandas as _pd91b
            _hm91_rows = [{"Column": k, "Missing %": f"{v:.1f}%"} for k, v in _high91.items()]
            with st.expander(f"{len(_hm91_rows)} column(s) with high missing rates"):
                st.dataframe(_pd91b.DataFrame(_hm91_rows), use_container_width=True, hide_index=True)

        # Missing rate bar chart (top 20 columns)
        _all_missing91 = _mv91.get("missing_pct_by_col", {})
        if _all_missing91:
            import pandas as _pd91c
            _miss91_series = _pd91c.Series(_all_missing91).sort_values(ascending=False).head(20)
            _miss91_series = _miss91_series[_miss91_series > 0]
            if not _miss91_series.empty:
                _fig91 = _go91.Figure()
                _miss_colors91 = ["#ef4444" if v > 20 else "#f59e0b" if v > 5 else "#27ae60"
                                  for v in _miss91_series.values]
                _fig91.add_trace(_go91.Bar(
                    x=_miss91_series.index, y=_miss91_series.values,
                    marker_color=_miss_colors91, name="Missing %",
                    hovertemplate="%{x}<br>Missing: %{y:.1f}%<extra></extra>",
                ))
                _fig91.add_hline(y=20, line=dict(color="#ef4444", dash="dash", width=1),
                                 annotation_text="High missing threshold (20%)",
                                 annotation_font=dict(color="#ef4444", size=9))
                _fig91.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="Missing Value Rate by Column (top 20)",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Missing (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280", tickangle=-45),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig91, use_container_width=True)

        # Lookahead leakage
        _la91 = _dq91.get("lookahead_leakage", {})
        st.subheader("Lookahead Leakage Check")
        _la91a, _la91b = st.columns(2)
        _la91a.metric("Status", "Pass" if _la91.get("passed") else "WARNING")
        _la91b.metric("Forward-Return Cols Checked", _la91.get("n_cols_checked", 0))
        if _la91.get("issues"):
            st.warning("Potential lookahead leakage detected in forward-return columns.")
            with st.expander("Leakage details"):
                for _issue91 in _la91.get("issues", []):
                    st.markdown(f"- `{_issue91}`")
        else:
            st.caption("No lookahead leakage detected in forward-return columns.")
    except Exception as _e91:
        _err_track(_active_sub, _e91)
        st.caption(f"Data diagnostics unavailable: {_e91}")


# =============================================================================
# BATCH 11 ANALYTICS: sub92–97
# =============================================================================


if _active_sub == 94:
    import plotly.graph_objects as _go94
    st.header("Score Decomposition")
    st.markdown(
        "The composite risk signal is built from **7 weighted sub-scores**, each measuring a "
        "different dimension of systemic credit risk. This tab shows the current contribution "
        "of each component, their history, and which sub-scores are driving the signal today. "
        "Understanding decomposition helps identify *what* is causing elevated readings "
        "and which signals may be lagging."
    )
    try:
        # Collect current sub-score values
        _sd94_components = []
        for _key94, _col94 in SCORE_COLS.items():
            _w94 = COMPOSITE_WEIGHTS.get(_key94, 0)
            _lbl94 = DISPLAY_NAMES.get(_key94, _key94)
            _val94 = float(latest.get(_col94, float("nan"))) if _col94 in df.columns else float("nan")
            _contribution94 = _val94 * _w94 if not pd.isna(_val94) and _w94 > 0 else float("nan")
            _sd94_components.append({
                "key": _key94, "label": _lbl94, "weight": _w94,
                "score": _val94, "contribution": _contribution94, "col": _col94,
            })
        _sd94_valid = [c for c in _sd94_components if not pd.isna(c["score"]) and c["weight"] > 0]

        if _sd94_valid:
            # KPI row — composite vs sub-score breakdown
            _comp_now = float(latest.get("composite_risk_score_smooth", float("nan")))
            _sd94a, _sd94b, _sd94c = st.columns(3)
            _sd94a.metric("Composite Score", f"{_comp_now:.1f}/100" if not pd.isna(_comp_now) else "—")
            _top_driver = max(_sd94_valid, key=lambda c: c.get("contribution", 0))
            _sd94b.metric("Top Driver", _top_driver["label"],
                          help="Sub-score with highest weighted contribution today")
            _sd94c.metric("Top Driver Score", f"{_top_driver['score']:.1f}",
                          delta=f"{_top_driver['weight']:.0%} weight")

            # Radar chart of sub-scores
            _radar_labels = [c["label"] for c in _sd94_valid] + [_sd94_valid[0]["label"]]
            _radar_vals   = [c["score"] for c in _sd94_valid] + [_sd94_valid[0]["score"]]
            _fig94a = _go94.Figure()
            _fig94a.add_trace(_go94.Scatterpolar(
                r=_radar_vals, theta=_radar_labels, fill="toself",
                fillcolor="rgba(239,68,68,0.15)",
                line=dict(color="#ef4444", width=2),
                name="Current",
                hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            ))
            _fig94a.add_trace(_go94.Scatterpolar(
                r=[50] * len(_radar_labels), theta=_radar_labels, fill=None,
                line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
                name="Neutral (50)",
            ))
            _fig94a.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=9, color="#6b7280"),
                                    gridcolor="rgba(255,255,255,0.08)"),
                    angularaxis=dict(tickfont=dict(size=10, color="#9aa0aa")),
                    bgcolor="rgba(0,0,0,0)",
                ),
                height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig94a, use_container_width=True)

            # Waterfall: weighted contributions
            _wf94_labels = [c["label"] for c in _sd94_valid]
            _wf94_contribs = [c["contribution"] for c in _sd94_valid]
            _wf94_colors = ["#ef4444" if v > 15 else "#f59e0b" if v > 8 else "#27ae60"
                            for v in _wf94_contribs]
            _fig94b = _go94.Figure()
            _fig94b.add_trace(_go94.Bar(
                x=_wf94_labels, y=_wf94_contribs,
                marker_color=_wf94_colors,
                text=[f"{v:.1f}" for v in _wf94_contribs],
                textposition="outside",
                hovertemplate="%{x}<br>Contribution: %{y:.1f} pts<extra></extra>",
            ))
            if not pd.isna(_comp_now):
                _fig94b.add_hline(y=_comp_now / len(_sd94_valid),
                                  line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1),
                                  annotation_text="Equal-split baseline",
                                  annotation_font=dict(color="#9aa0aa", size=9))
            _fig94b.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                title=dict(text="Weighted Score Contribution (score × weight)",
                           font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Contribution (pts)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig94b, use_container_width=True)

            # Sub-score history stacked area
            _hist_cols94 = [c["col"] for c in _sd94_valid if c["col"] in df.columns]
            if _hist_cols94:
                _hist94 = df[_hist_cols94].tail(504).copy()
                _hist94.index = pd.to_datetime(_hist94.index)
                _fig94c = _go94.Figure()
                _palette94 = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#06b6d4", "#ec4899"]
                for _i94, _c94 in enumerate(_sd94_valid):
                    if _c94["col"] in _hist94.columns:
                        _fig94c.add_trace(_go94.Scatter(
                            x=_hist94.index, y=_hist94[_c94["col"]],
                            name=_c94["label"],
                            line=dict(color=_palette94[_i94 % len(_palette94)], width=1.5),
                            hovertemplate=f"%{{x|%Y-%m-%d}}<br>{_c94['label']}: %{{y:.1f}}<extra></extra>",
                        ))
                _fig94c.add_hline(y=50, line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1))
                _fig94c.update_layout(
                    height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=24, b=8),
                    title=dict(text="Sub-Score History (0–100)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Score (0–100)", range=[0, 100]),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(orientation="h", y=1.15, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig94c, use_container_width=True)

            # Summary table
            import pandas as _pd94
            _sd94_tbl = _pd94.DataFrame([{
                "Component": c["label"],
                "Weight": f"{c['weight']:.0%}",
                "Current Score": f"{c['score']:.1f}" if not pd.isna(c["score"]) else "—",
                "Contribution": f"{c['contribution']:.1f} pts" if not pd.isna(c.get("contribution", float("nan"))) else "—",
                "Column": c["col"],
            } for c in _sd94_components if c["weight"] > 0])
            with st.expander("Component detail table"):
                st.dataframe(_sd94_tbl, use_container_width=True, hide_index=True)
        else:
            st.info("Score decomposition unavailable — sub-score columns not found in dataset.")
    except Exception as _e94:
        _err_track(_active_sub, _e94)
        st.caption(f"Score decomposition unavailable: {_e94}")



if _active_sub == 101:
    import plotly.graph_objects as _go101
    from src.regime_attribution import SCORE_COLS as _SC101, COMPOSITE_WEIGHTS as _CW101, DISPLAY_NAMES as _DN101
    st.header("Score Velocity & Alert Monitor")
    st.markdown(
        "**Score velocity** measures the 21-day change in each composite sub-score — a leading indicator of "
        "which risk dimensions are accelerating or decelerating. Rapid acceleration (+15+ pts/month) in any "
        "sub-score is an early warning even before the overall composite reaches alert thresholds. "
        "The alert matrix flags when sub-scores cross the 50 (elevated) and 70 (high alert) levels."
    )
    try:
        _sc101_present = {k: v for k, v in _SC101.items() if v in df.columns}
        if _sc101_present:
            _sv101 = df[[v for v in _sc101_present.values()]].copy()
            _sv101.index = pd.to_datetime(_sv101.index)

            # 21-day velocity for each score
            _vel_data = {}
            for _key, _col in _sc101_present.items():
                _vel_data[_DN101.get(_key, _key)] = _sv101[_col].diff(21)

            _vel_df = pd.DataFrame(_vel_data)
            _vel_cur = _vel_df.iloc[-1]

            # Current velocity metrics
            st.markdown("**Current 21-Day Score Velocity (pts/month)**")
            _vel_cols = st.columns(min(len(_vel_cur), 4))
            for _i, (_name, _val) in enumerate(sorted(_vel_cur.items(), key=lambda x: abs(x[1]) if pd.notna(x[1]) else 0, reverse=True)[:4]):
                _vc = _vel_cols[_i % 4]
                _vc.metric(
                    _name[:14],
                    f"{_val:+.1f} pts" if pd.notna(_val) else "—",
                    delta=f"{'rising fast' if _val > 15 else 'falling fast' if _val < -15 else 'stable'}" if pd.notna(_val) else None,
                    delta_color="inverse" if pd.notna(_val) and _val > 0 else "normal",
                )

            # Velocity heatmap: last 12 months × all sub-scores
            _vel_tail = _vel_df.tail(252)
            _vel_monthly = _vel_tail.resample("ME").last()
            if not _vel_monthly.empty:
                _z_vals = _vel_monthly.values.T.tolist()
                _fig101a = _go101.Figure(data=_go101.Heatmap(
                    z=_z_vals,
                    x=[str(d)[:7] for d in _vel_monthly.index],
                    y=list(_vel_monthly.columns),
                    colorscale=[[0, "#1a3a2e"], [0.35, "#166534"], [0.5, "#1a1f2e"], [0.65, "#7f1d1d"], [1, "#ef4444"]],
                    zmid=0, zmin=-30, zmax=30,
                    text=[[f"{v:+.0f}" if pd.notna(v) else "" for v in row] for row in _z_vals],
                    texttemplate="%{text}",
                    hovertemplate="Month: %{x}<br>Score: %{y}<br>Δ: %{z:+.1f}pts<extra></extra>",
                    colorbar=dict(title="Δ pts", tickfont=dict(color="#9aa0aa", size=10),
                                  titlefont=dict(color="#9aa0aa")),
                ))
                _fig101a.update_layout(
                    height=max(200, len(_sc101_present) * 32 + 60),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Score Velocity Heatmap (Monthly, last 12M)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280"),
                    yaxis=dict(color="#6b7280"),
                )
                st.plotly_chart(_fig101a, use_container_width=True)
                st.caption("Red = score accelerating (rising stress) · Green = score decelerating (improving)")

            # Alert matrix: current score level vs thresholds
            _score_now = _sv101.iloc[-1]
            _alert_rows = []
            for _key, _col in _sc101_present.items():
                _val = float(_score_now.get(_col, float("nan")))
                _vel_val = float(_vel_cur.get(_DN101.get(_key, _key), float("nan")))
                _wt = _CW101.get(_key, 0)
                _alert = "🔴 High Alert" if _val >= 70 else "🟡 Elevated" if _val >= 50 else "🟢 Normal"
                _momentum = "↑ Accelerating" if pd.notna(_vel_val) and _vel_val > 5 else (
                            "↓ Decelerating" if pd.notna(_vel_val) and _vel_val < -5 else "→ Stable")
                _alert_rows.append({
                    "Sub-Score": _DN101.get(_key, _key),
                    "Current": f"{_val:.0f}" if pd.notna(_val) else "—",
                    "Alert": _alert,
                    "Velocity (21d)": f"{_vel_val:+.1f}pts" if pd.notna(_vel_val) else "—",
                    "Momentum": _momentum,
                    "Weight": f"{_wt:.0%}" if _wt else "—",
                })
            st.markdown("**Alert Matrix — Sub-Score Status**")
            st.dataframe(pd.DataFrame(_alert_rows), use_container_width=True, hide_index=True)

            # Velocity time series for top 3 sub-scores by current absolute velocity
            _top3 = _vel_cur.abs().nlargest(3).index.tolist()
            if _top3:
                _fig101b = _go101.Figure()
                _colors101 = ["#4f8ef7", "#f59e0b", "#27ae60"]
                for _ci, _cname in enumerate(_top3):
                    _fig101b.add_trace(_go101.Scatter(
                        x=_vel_tail.index, y=_vel_tail[_cname],
                        line=dict(color=_colors101[_ci % 3], width=1.8), name=_cname,
                        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{_cname}: %{{y:+.1f}}pts<extra></extra>",
                    ))
                _fig101b.add_hline(y=15, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
                _fig101b.add_hline(y=-15, line=dict(color="rgba(39,174,96,0.4)", dash="dot", width=1))
                _fig101b.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1))
                _fig101b.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Top 3 Sub-Scores by Velocity (21d Change)", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Δ pts/month"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                )
                st.plotly_chart(_fig101b, use_container_width=True)
        else:
            st.info("No smoothed score columns found in dataframe.")
    except Exception as _e101:
        _err_track(_active_sub, _e101)
        st.caption(f"Score velocity: {_e101}")


if _active_sub == 108:
    import plotly.graph_objects as _go108
    from src.regime_attribution import SUPPLEMENTAL_SCORES as _SS108, DISPLAY_NAMES as _DN108
    st.header("Cross-Asset Divergence — Supplemental Score")
    st.markdown(
        "The **cross-asset divergence score** measures the degree to which equity, credit, and vol signals "
        "are pointing in conflicting directions. It was assessed for the composite but excluded "
        "because it showed correct-direction signals at short horizons but mean-reverted at the 6–12 month "
        "horizon that matters most for credit positioning. "
        "It remains informative as a **tactical** signal: high divergence often resolves within 4–6 weeks, "
        "creating short-term opportunity. The score is compared here against the composite to identify "
        "periods where the two most disagree."
    )
    try:
        _ca108_smooth = "cross_asset_divergence_score_smooth"
        _ca108_raw = "cross_asset_divergence_score"
        _comp108 = "composite_risk_score_smooth" if "composite_risk_score_smooth" in df.columns else None

        _ca_col = _ca108_smooth if _ca108_smooth in df.columns else (_ca108_raw if _ca108_raw in df.columns else None)
        if _ca_col:
            _ca108 = df[[_ca_col]].copy()
            if _comp108:
                _ca108["composite"] = df[_comp108]
            if "hy_spread" in df.columns:
                _ca108["hy_spread"] = df["hy_spread"]
            _ca108.index = pd.to_datetime(_ca108.index)
            _cur_ca = float(latest.get(_ca_col, float("nan")))
            _cur_comp = float(latest.get(_comp108, float("nan"))) if _comp108 else float("nan")

            _aa, _ab, _ac, _ad = st.columns(4)
            _aa.metric("X-Asset Div Score", f"{_cur_ca:.0f}/100" if pd.notna(_cur_ca) else "—")
            if pd.notna(_cur_comp):
                _ab.metric("Composite Score", f"{_cur_comp:.0f}/100")
                _divergence_delta = _cur_ca - _cur_comp
                _ac.metric("Score Delta (Div − Comp)", f"{_divergence_delta:+.0f}pts" if pd.notna(_divergence_delta) else "—")
            _ca_pctile = (df[_ca_col].dropna() < _cur_ca).mean() * 100 if pd.notna(_cur_ca) else float("nan")
            _ad.metric("Historical Pctile", f"{_ca_pctile:.0f}th" if pd.notna(_ca_pctile) else "—")

            if pd.notna(_cur_ca) and _cur_ca >= 60:
                st.warning("Cross-asset divergence elevated — equity, credit, and vol signals in conflict. "
                           "Historically resolves within 4–6 weeks: credit signal tends to dominate.")

            # Score time series
            _ca_tail = _ca108.tail(756)
            _fig108a = _go108.Figure()
            _fig108a.add_trace(_go108.Scatter(
                x=_ca_tail.index, y=_ca_tail[_ca_col],
                line=dict(color="#27ae60", width=2), name="X-Asset Divergence",
                fill="tozeroy", fillcolor="rgba(39,174,96,0.08)",
                hovertemplate="%{x|%Y-%m-%d}<br>Div Score: %{y:.0f}<extra></extra>",
            ))
            if "composite" in _ca_tail.columns:
                _fig108a.add_trace(_go108.Scatter(
                    x=_ca_tail.index, y=_ca_tail["composite"],
                    line=dict(color="#4f8ef7", width=1.5, dash="dot"), name="Composite Score",
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
            _fig108a.add_hline(y=50, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
            _fig108a.add_hline(y=70, line=dict(color="rgba(239,68,68,0.3)", dash="dot", width=1))
            _fig108a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Cross-Asset Divergence Score vs Composite (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig108a, use_container_width=True)

            # Score delta: when do they diverge the most?
            if "composite" in _ca108.columns:
                _ca108["score_delta"] = _ca108[_ca_col] - _ca108["composite"]
                _fig108b = _go108.Figure()
                _delta_series = _ca108["score_delta"].tail(756)
                _delta_colors = ["#ef4444" if v > 10 else "#27ae60" if v < -10 else "#6b7280"
                                  for v in _delta_series.fillna(0)]
                _fig108b.add_trace(_go108.Bar(
                    x=_delta_series.index, y=_delta_series,
                    marker_color=_delta_colors, name="Div − Composite",
                    hovertemplate="%{x|%Y-%m-%d}<br>Delta: %{y:+.0f}pts<extra></extra>",
                ))
                _fig108b.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
                _fig108b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Score Delta: X-Asset Div minus Composite", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="pts"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig108b, use_container_width=True)
                st.caption("Red = X-Asset div score >> composite (divergence signal more alarmed) · "
                           "Green = X-Asset score << composite (composite more alarmed than divergence signal)")

            st.markdown("**Why Cross-Asset Divergence is Supplemental (not composite)**")
            st.info(
                "The cross-asset divergence score was assessed for inclusion in the 7-factor composite "
                "but excluded because backtesting showed it had correct leading signals at 2–4 week horizons "
                "but underperformed at the 3–6 month horizon that drives credit positioning decisions. "
                "It remains in the dashboard as a *tactical* signal for short-term monitoring "
                "and a useful cross-check on the composite."
            )
        else:
            st.info("Cross-asset divergence score not found. Run the full scoring pipeline.")
    except Exception as _e108:
        _err_track(_active_sub, _e108)
        st.caption(f"Cross-asset divergence: {_e108}")


if _active_sub == 115:
    import plotly.graph_objects as _go115
    st.header("Complacency Sub-Score")
    st.markdown(
        "The **complacency sub-score** (20% weight in the composite — tied with Treasury and Credit Risk for "
        "the largest weight) captures the market's tendency to under-price risk when conditions have been "
        "benign for an extended period. High complacency (VIX low, VRP compressed, SP500 drawdown minimal, "
        "credit-equity aligned) creates fragility: small shocks generate outsized spread moves because "
        "risk premia were not being paid. Low complacency scores historically co-occur with the "
        "tightest credit spreads — and the most dangerous entry points."
    )
    try:
        _comp115_col = "complacency_score_smooth"
        _comp115_raw = "complacency_score"
        _cmp_col = _comp115_col if _comp115_col in df.columns else (_comp115_raw if _comp115_raw in df.columns else None)
        if _cmp_col:
            _cmp115 = df[[_cmp_col]].copy()
            if "hy_spread" in df.columns:
                _cmp115["hy_spread"] = df["hy_spread"]
            if "composite_risk_score_smooth" in df.columns:
                _cmp115["composite"] = df["composite_risk_score_smooth"]
            if "vix" in df.columns:
                _cmp115["vix"] = df["vix"]
            _cmp115.index = pd.to_datetime(_cmp115.index)
            _cur_cmp = float(latest.get(_cmp_col, float("nan")))
            _cmp_pctile = (df[_cmp_col].dropna() < _cur_cmp).mean() * 100 if pd.notna(_cur_cmp) else float("nan")

            def _cmp_regime(score):
                if pd.isna(score):
                    return "Unknown"
                if score >= 70:
                    return "Acute Complacency Risk"
                if score >= 50:
                    return "Complacent"
                if score >= 30:
                    return "Neutral"
                return "Alert / Risk-Off"

            _cmp_reg = _cmp_regime(_cur_cmp)
            _cmp_vel = float(df[_cmp_col].diff(21).iloc[-1]) if df[_cmp_col].notna().any() else float("nan")

            _ca115, _cb115, _cc115, _cd115 = st.columns(4)
            _ca115.metric("Complacency Score", f"{_cur_cmp:.0f}/100" if pd.notna(_cur_cmp) else "—",
                          help="High = market under-pricing risk (complacent). Low = risk-off / fear premia elevated.")
            _cb115.metric("Regime", _cmp_reg)
            _cc115.metric("Historical Pctile", f"{_cmp_pctile:.0f}th" if pd.notna(_cmp_pctile) else "—")
            _cd115.metric("21d Velocity", f"{_cmp_vel:+.1f}pts" if pd.notna(_cmp_vel) else "—",
                          help="Positive = complacency rising (risk building). Negative = fear re-entering market.")

            if pd.notna(_cur_cmp) and _cur_cmp >= 70:
                st.warning("Complacency at elevated levels — risk premia compressed, market fragile to shocks. "
                           "Historical pattern: HY spread widening of 30–100bps follows within 4–8 weeks after "
                           "complacency peaks.")
            elif pd.notna(_cur_cmp) and _cur_cmp <= 25:
                st.info("Complacency score low — fear premia elevated. Historically a favorable entry for credit.")

            # Score time series
            _cmp_tail = _cmp115.tail(756)
            _fig115a = _go115.Figure()
            _fig115a.add_hrect(y0=70, y1=105, fillcolor="rgba(245,158,11,0.08)", line_width=0)
            _fig115a.add_hrect(y0=0, y1=25, fillcolor="rgba(39,174,96,0.06)", line_width=0)
            _fig115a.add_trace(_go115.Scatter(
                x=_cmp_tail.index, y=_cmp_tail[_cmp_col],
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                line=dict(color="#f59e0b", width=2), name="Complacency Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            _fig115a.add_hline(y=50, line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1))
            _fig115a.add_hline(y=70, line=dict(color="rgba(245,158,11,0.5)", dash="dot", width=1))
            _fig115a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Complacency Sub-Score (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig115a, use_container_width=True)
            st.caption("Yellow band = complacency elevated (>70) · Green band = fear premia healthy (<25)")

            # Complacency vs composite
            if "composite" in _cmp115.columns:
                _fig115b = _go115.Figure()
                _fig115b.add_trace(_go115.Scatter(
                    x=_cmp_tail.index, y=_cmp_tail[_cmp_col],
                    name="Complacency (20% wt)", line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Complacency: %{y:.0f}<extra></extra>",
                ))
                _fig115b.add_trace(_go115.Scatter(
                    x=_cmp_tail.index, y=_cmp_tail["composite"],
                    name="Composite Score", line=dict(color="#4f8ef7", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
                _fig115b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Complacency vs Composite Score", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#6b7280", title="Score"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig115b, use_container_width=True)

            # Complacency vs HY spread inverse relationship
            if "hy_spread" in _cmp115.columns:
                _sc115 = _cmp115.dropna(subset=[_cmp_col, "hy_spread"]).tail(756)
                _fig115c = _go115.Figure()
                _fig115c.add_trace(_go115.Scatter(
                    x=_sc115[_cmp_col], y=_sc115["hy_spread"],
                    mode="markers",
                    marker=dict(
                        color=_sc115[_cmp_col], colorscale=[[0, "#27ae60"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                        size=3, opacity=0.4,
                    ),
                    hovertemplate="Complacency: %{x:.0f}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig115c.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Complacency Score vs HY Spread — inverse relationship", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Complacency Score"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="HY Spread (bps)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig115c, use_container_width=True)
                st.caption("Negative correlation expected: low complacency = high spreads (fear priced in) · "
                           "High complacency = tight spreads (risk under-priced) — the dangerous quadrant")
        else:
            st.info("Complacency score not found — run the full scoring pipeline.")
    except Exception as _e115:
        _err_track(_active_sub, _e115)
        st.caption(f"Complacency score: {_e115}")

# =============================================================================
# BATCH 15 ANALYTICS — sub116–121
# sub116  Macro Risk Sub-Score            → tab_siglab
# sub117  Credit Market Risk Sub-Score    → tab_credit
# sub118  Treasury Stress Sub-Score       → tab_macro
# sub119  Enhanced Funding Sub-Score      → tab_risk
# sub120  Mean Reversion Sub-Score        → tab_siglab
# sub121  Spread Change Deep Dive         → tab_macro
# =============================================================================


if _active_sub == 116:
    import plotly.graph_objects as _go116
    from src.regime_attribution import COMPOSITE_WEIGHTS as _CW116, DISPLAY_NAMES as _DN116
    st.header("Macro Risk Sub-Score")
    st.markdown(
        "The **macro risk sub-score** (15% weight) is the broadest recession-risk component in the composite. "
        "It synthesises: the **yield curve spread** (flattening = restrictive policy), **unemployment level** "
        "and **90d change**, the **NFCI** level and trend, and the **Sahm-like** trigger. "
        "It is primarily a *coincident-to-lagging* signal — it confirms deterioration already visible in "
        "credit and treasury scores — but its 15% weight means it can push the composite through key "
        "thresholds when the other signals are borderline."
    )
    try:
        _mr116_col = "macro_risk_score_smooth"
        _mr116_raw = "macro_risk_score"
        _mr_col = _mr116_col if _mr116_col in df.columns else (_mr116_raw if _mr116_raw in df.columns else None)
        if _mr_col:
            _mr116 = df[[_mr_col]].copy()
            for _c in ["hy_spread", "composite_risk_score_smooth", "nfci", "unemployment"]:
                if _c in df.columns:
                    _mr116[_c] = df[_c]
            _mr116.index = pd.to_datetime(_mr116.index)
            _cur_mr = float(latest.get(_mr_col, float("nan")))
            _mr_pctile = (df[_mr_col].dropna() < _cur_mr).mean() * 100 if pd.notna(_cur_mr) else float("nan")
            _mr_vel = float(df[_mr_col].diff(21).iloc[-1]) if df[_mr_col].notna().any() else float("nan")
            _mr_contrib = _cur_mr * _CW116.get("macro_risk", 0.15) if pd.notna(_cur_mr) else float("nan")

            _ma116, _mb116, _mc116, _md116 = st.columns(4)
            _ma116.metric("Macro Risk Score", f"{_cur_mr:.0f}/100" if pd.notna(_cur_mr) else "—")
            _mb116.metric("Composite Contrib", f"{_mr_contrib:.1f}pts" if pd.notna(_mr_contrib) else "—",
                          help=f"Score × {_CW116.get('macro_risk', 0.15):.0%} weight")
            _mc116.metric("Historical Pctile", f"{_mr_pctile:.0f}th" if pd.notna(_mr_pctile) else "—")
            _md116.metric("21d Velocity", f"{_mr_vel:+.1f}pts" if pd.notna(_mr_vel) else "—",
                          delta_color="inverse")

            if pd.notna(_cur_mr) and _cur_mr >= 65:
                st.error("Macro risk score elevated — recession indicators active across multiple inputs. "
                         "This score's 15% weight is now materially pushing the composite toward High Risk territory.")

            # Score time series with driver input overlay
            _mr_tail = _mr116.tail(756)
            _fig116a = _go116.Figure()
            _fig116a.add_hrect(y0=65, y1=105, fillcolor="rgba(239,68,68,0.07)", line_width=0)
            _fig116a.add_hrect(y0=45, y1=65, fillcolor="rgba(245,158,11,0.05)", line_width=0)
            _fig116a.add_trace(_go116.Scatter(
                x=_mr_tail.index, y=_mr_tail[_mr_col],
                fill="tozeroy", fillcolor="rgba(79,142,247,0.1)",
                line=dict(color="#4f8ef7", width=2), name="Macro Risk Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            if "composite_risk_score_smooth" in _mr_tail.columns:
                _fig116a.add_trace(_go116.Scatter(
                    x=_mr_tail.index, y=_mr_tail["composite_risk_score_smooth"],
                    line=dict(color="#e2e8f0", width=1, dash="dot"), name="Composite",
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
            _fig116a.add_hline(y=45, line=dict(color="rgba(245,158,11,0.4)", dash="dot", width=1))
            _fig116a.add_hline(y=65, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig116a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Macro Risk Sub-Score vs Composite (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig116a, use_container_width=True)

            # NFCI + Unemployment dual-axis (the two primary drivers)
            if "nfci" in _mr_tail.columns and "unemployment" in _mr_tail.columns:
                _fig116b = _go116.Figure()
                _fig116b.add_trace(_go116.Scatter(
                    x=_mr_tail.index, y=_mr_tail["nfci"],
                    name="NFCI", line=dict(color="#f59e0b", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>NFCI: %{y:.3f}<extra></extra>",
                ))
                _fig116b.add_trace(_go116.Scatter(
                    x=_mr_tail.index, y=_mr_tail["unemployment"],
                    name="Unemployment (%)", line=dict(color="#a78bfa", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>Unemp: %{y:.1f}%<extra></extra>",
                ))
                _fig116b.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
                _fig116b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="Key Macro Risk Drivers", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#f59e0b", title="NFCI"),
                    yaxis2=dict(overlaying="y", side="right", color="#a78bfa", title="Unemp (%)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig116b, use_container_width=True)

            st.markdown("**Macro Risk Score — Driver Breakdown**")
            st.table(pd.DataFrame([
                {"Input": "2s10s Spread", "Weight in Score": "Up to 30pts", "Signal Direction": "Low/negative spread → higher score"},
                {"Input": "Unemployment Level", "Weight in Score": "Up to 30pts", "Signal Direction": ">6% → 30pts; 5–6% → 15pts"},
                {"Input": "Unemployment Δ90d", "Weight in Score": "Up to 25pts", "Signal Direction": "Rising → proportional addition"},
                {"Input": "Sahm-Like", "Weight in Score": "Up to 30pts", "Signal Direction": ">0.5pp → 30pts; >0.3pp → 15pts"},
                {"Input": "NFCI Level", "Weight in Score": "Up to 35pts", "Signal Direction": "Positive NFCI → proportional"},
                {"Input": "NFCI Δ90d", "Weight in Score": "Up to 20pts", "Signal Direction": "Trend tightening → add pts"},
            ]))
        else:
            st.info("Macro risk score not found — run the full scoring pipeline.")
    except Exception as _e116:
        _err_track(_active_sub, _e116)
        st.caption(f"Macro risk score: {_e116}")


if _active_sub == 120:
    import plotly.graph_objects as _go120
    st.header("Mean Reversion Sub-Score (Supplemental)")
    st.markdown(
        "The **mean reversion score** is a supplemental signal that was assessed for the composite but "
        "excluded because it fires *after* stress has already peaked — useful for identifying when "
        "conditions are ripe for a spread compression bounce, not for warning of deterioration. "
        "It combines: overall stress level (are macro or credit risk scores elevated?), "
        "HY spread level (is HY cheap relative to history?), and short-term momentum. "
        "When elevated, it signals that a mean-reversion trade in credit may be setting up — "
        "spreads have widened far enough that dip-buyers historically emerge."
    )
    try:
        _mv120_col = "mean_reversion_score_smooth"
        _mv120_raw = "mean_reversion_score"
        _mv_col = _mv120_col if _mv120_col in df.columns else (_mv120_raw if _mv120_raw in df.columns else None)
        if _mv_col:
            _mv120 = df[[_mv_col]].copy()
            for _c in ["hy_spread", "composite_risk_score_smooth", "credit_market_risk_score_smooth"]:
                if _c in df.columns:
                    _mv120[_c] = df[_c]
            _mv120.index = pd.to_datetime(_mv120.index)
            _cur_mv = float(latest.get(_mv_col, float("nan")))
            _mv_pctile = (df[_mv_col].dropna() < _cur_mv).mean() * 100 if pd.notna(_cur_mv) else float("nan")
            _mv_vel = float(df[_mv_col].diff(21).iloc[-1]) if df[_mv_col].notna().any() else float("nan")

            _ma120, _mb120, _mc120, _md120 = st.columns(4)
            _ma120.metric("Mean Rev Score", f"{_cur_mv:.0f}/100" if pd.notna(_cur_mv) else "—",
                          help="High = conditions ripe for spread compression / bounce")
            _mb120.metric("Historical Pctile", f"{_mv_pctile:.0f}th" if pd.notna(_mv_pctile) else "—")
            _mc120.metric("21d Velocity", f"{_mv_vel:+.1f}pts" if pd.notna(_mv_vel) else "—")
            _md120.metric("Status", "Supplemental only — not in composite")

            if pd.notna(_cur_mv) and _cur_mv >= 60:
                st.info("Mean reversion score elevated — spreads may have overshot fundamentals. "
                        "This signal historically precedes short-term credit compression as dip-buyers enter. "
                        "Not a primary risk signal; useful for timing credit re-entry after selloffs.")

            _mv_tail = _mv120.tail(756)
            _fig120a = _go120.Figure()
            _fig120a.add_hrect(y0=60, y1=105, fillcolor="rgba(39,174,96,0.07)", line_width=0)
            _fig120a.add_trace(_go120.Scatter(
                x=_mv_tail.index, y=_mv_tail[_mv_col],
                fill="tozeroy", fillcolor="rgba(39,174,96,0.08)",
                line=dict(color="#27ae60", width=2), name="Mean Reversion Score",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            if "composite_risk_score_smooth" in _mv_tail.columns:
                _fig120a.add_trace(_go120.Scatter(
                    x=_mv_tail.index, y=_mv_tail["composite_risk_score_smooth"],
                    line=dict(color="#ef4444", width=1.5, dash="dot"), name="Composite (risk)",
                    hovertemplate="%{x|%Y-%m-%d}<br>Composite: %{y:.0f}<extra></extra>",
                ))
            _fig120a.add_hline(y=60, line=dict(color="rgba(39,174,96,0.4)", dash="dot", width=1))
            _fig120a.update_layout(
                height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Mean Reversion Score vs Composite Risk (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig120a, use_container_width=True)
            st.caption("Green band = mean reversion opportunity zone (>60). "
                       "Note: score is high *because* risk is elevated — it signals dip-buy opportunity, not safety.")

            # Mean reversion vs HY spread — the timing relationship
            if "hy_spread" in _mv_tail.columns:
                _fig120b = _go120.Figure()
                _fig120b.add_trace(_go120.Scatter(
                    x=_mv_tail.index, y=_mv_tail[_mv_col],
                    name="Mean Rev Score", line=dict(color="#27ae60", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>MR Score: %{y:.0f}<extra></extra>",
                ))
                _fig120b.add_trace(_go120.Scatter(
                    x=_mv_tail.index, y=_mv_tail["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig120b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="Mean Reversion Score vs HY Spread — timing alignment", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#27ae60",
                               title="MR Score"),
                    yaxis2=dict(overlaying="y", side="right", color="#f59e0b", title="HY (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig120b, use_container_width=True)
                st.caption("MR Score peaks near HY spread peaks — confirms the lagging nature of this signal")

            st.info("**Why excluded from composite:** Mean reversion score showed correct *contrarian* signals "
                    "at 2–4 week horizons (good for timing) but at 6–12 month horizons it was noise. "
                    "Including it reduced composite Sharpe ratio by dampening high-risk warnings. "
                    "Kept as supplemental for short-horizon tactical use only.")
        else:
            st.info("Mean reversion score not found — run the full scoring pipeline.")
    except Exception as _e120:
        _err_track(_active_sub, _e120)
        st.caption(f"Mean reversion score: {_e120}")


if _active_sub == 123:
    import plotly.graph_objects as _go123
    from src.regime_attribution import SCORE_COLS as _SC123, COMPOSITE_WEIGHTS as _CW123, DISPLAY_NAMES as _DN123
    st.header("Score Consensus Monitor")
    st.markdown(
        "**Score consensus** measures how many of the 7 composite sub-scores agree on direction. "
        "When all 7 are elevated simultaneously, the signal is the strongest — multi-dimensional stress "
        "confirmation with no offsetting factors. When scores diverge widely, the composite may be "
        "masking internal disagreement. This tab tracks: breadth of sub-score elevation, the "
        "count of scores above 50 (elevated), above 70 (high-stress), and the inter-score "
        "standard deviation (high dispersion = mixed signals)."
    )
    try:
        _sc123_present = {k: v for k, v in _SC123.items() if v in df.columns and k in _CW123}
        if len(_sc123_present) >= 3:
            _cons123 = df[[v for v in _sc123_present.values()]].copy()
            _cons123.index = pd.to_datetime(_cons123.index)

            # Compute consensus metrics per row
            _cons123["n_above_50"] = (_cons123 >= 50).sum(axis=1)
            _cons123["n_above_70"] = (_cons123 >= 70).sum(axis=1)
            _cons123["score_std"] = _cons123[[v for v in _sc123_present.values()]].std(axis=1)
            _cons123["score_mean"] = _cons123[[v for v in _sc123_present.values()]].mean(axis=1)

            _cur_n50 = int(_cons123["n_above_50"].iloc[-1])
            _cur_n70 = int(_cons123["n_above_70"].iloc[-1])
            _cur_std = float(_cons123["score_std"].iloc[-1])
            _n_total = len(_sc123_present)

            _ca123, _cb123, _cc123, _cd123 = st.columns(4)
            _ca123.metric("Scores ≥ 50 (Elevated)", f"{_cur_n50}/{_n_total}")
            _cb123.metric("Scores ≥ 70 (High Stress)", f"{_cur_n70}/{_n_total}")
            _cc123.metric("Score Std Dev", f"{_cur_std:.1f}pts",
                          help="High = sub-scores diverging (mixed signal). Low = all agree.")
            _consensus_pct = _cur_n50 / _n_total * 100
            _cd123.metric("Consensus (%)", f"{_consensus_pct:.0f}%")

            if _cur_n50 >= 5:
                st.error(f"{_cur_n50}/{_n_total} sub-scores elevated — broad consensus that risk conditions are stressed. "
                         "Multi-dimensional confirmation: not a false positive.")
            elif _cur_std > 20:
                st.info("High score dispersion — sub-scores disagree significantly. "
                        "Mixed environment: some dimensions stressed, others benign. Interpret composite with caution.")

            # Rolling count of elevated scores
            _cons_tail = _cons123.tail(756)
            _fig123a = _go123.Figure()
            _fig123a.add_trace(_go123.Scatter(
                x=_cons_tail.index, y=_cons_tail["n_above_50"],
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)",
                line=dict(color="#f59e0b", width=2), name="Scores ≥ 50",
                hovertemplate="%{x|%Y-%m-%d}<br>Elevated: %{y}/" + str(_n_total) + "<extra></extra>",
            ))
            _fig123a.add_trace(_go123.Scatter(
                x=_cons_tail.index, y=_cons_tail["n_above_70"],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
                line=dict(color="#ef4444", width=2), name="Scores ≥ 70",
                hovertemplate="%{x|%Y-%m-%d}<br>High Stress: %{y}/" + str(_n_total) + "<extra></extra>",
            ))
            _fig123a.add_hline(y=_n_total * 0.5, line=dict(color="rgba(245,158,11,0.4)", dash="dot", width=1))
            _fig123a.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Count of Elevated Sub-Scores (3Y)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, _n_total + 0.5], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="# Sub-Scores"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig123a, use_container_width=True)
            st.caption(f"Dashed line = majority threshold ({_n_total//2 +1}/{_n_total} scores elevated = broad consensus)")

            # Score dispersion (std dev)
            _fig123b = _go123.Figure()
            _fig123b.add_trace(_go123.Scatter(
                x=_cons_tail.index, y=_cons_tail["score_std"],
                line=dict(color="#a78bfa", width=2), name="Score Std Dev",
                hovertemplate="%{x|%Y-%m-%d}<br>Std Dev: %{y:.1f}pts<extra></extra>",
            ))
            _fig123b.add_hline(y=20, line=dict(color="rgba(167,139,250,0.4)", dash="dot", width=1))
            _fig123b.update_layout(
                height=180, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Inter-Score Dispersion (Std Dev, pts)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig123b, use_container_width=True)
            st.caption("High dispersion (>20pts) = mixed signal environment — composite may understate or overstate risk")

            # Current score bar chart (all sub-scores)
            _cur_scores = {_DN123.get(k, k): float(latest.get(v, float("nan")))
                           for k, v in _sc123_present.items()}
            _sorted_scores = sorted(_cur_scores.items(), key=lambda x: x[1] if pd.notna(x[1]) else 0, reverse=True)
            _fig123c = _go123.Figure()
            _fig123c.add_trace(_go123.Bar(
                x=[s[0] for s in _sorted_scores],
                y=[s[1] for s in _sorted_scores],
                marker_color=["#ef4444" if s[1] >= 70 else "#f59e0b" if s[1] >= 50 else "#27ae60"
                              if pd.notna(s[1]) else "#6b7280" for s in _sorted_scores],
                text=[f"{s[1]:.0f}" if pd.notna(s[1]) else "—" for s in _sorted_scores],
                textposition="auto",
                hovertemplate="%{x}<br>Score: %{y:.0f}<extra></extra>",
            ))
            _fig123c.add_hline(y=50, line=dict(color="rgba(245,158,11,0.4)", dash="dot", width=1))
            _fig123c.add_hline(y=70, line=dict(color="rgba(239,68,68,0.4)", dash="dot", width=1))
            _fig123c.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="All Sub-Scores — Current Snapshot (ranked)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score"),
                xaxis=dict(color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig123c, use_container_width=True)
        else:
            st.info("Not enough sub-score columns in dataframe for consensus analysis.")
    except Exception as _e123:
        _err_track(_active_sub, _e123)
        st.caption(f"Score consensus: {_e123}")


if _active_sub == 126:
    import plotly.graph_objects as _go126
    st.header("Composite Risk Score — Full History")
    st.markdown(
        "The **composite risk score** is the primary output of this dashboard — a 0–100 index weighting "
        "seven sub-scores into a single credit stress signal. This tab provides the full historical "
        "context: where is the current score relative to all prior readings, how persistent are "
        "high-risk regimes, and what was the HY spread outcome following each major score peak. "
        "**Score thresholds:** 0–30 = Low Risk · 30–50 = Moderate · 50–70 = Elevated · 70–100 = High Risk."
    )
    try:
        _comp126_col = "composite_risk_score_smooth"
        if _comp126_col in df.columns and df[_comp126_col].notna().any():
            _comp126 = df[[_comp126_col]].copy()
            if "hy_spread" in df.columns:
                _comp126["hy_spread"] = df["hy_spread"]
            _comp126.index = pd.to_datetime(_comp126.index)
            _cur_comp = float(latest.get(_comp126_col, float("nan")))
            _comp_pctile = (df[_comp126_col].dropna() < _cur_comp).mean() * 100 if pd.notna(_cur_comp) else float("nan")
            _comp_max = float(df[_comp126_col].max())
            _comp_mean = float(df[_comp126_col].mean())

            def _comp_regime(score):
                if pd.isna(score):
                    return "Unknown"
                if score >= 70:
                    return "High Risk"
                if score >= 50:
                    return "Elevated Risk"
                if score >= 30:
                    return "Moderate Risk"
                return "Low Risk"

            _comp_reg = _comp_regime(_cur_comp)
            _comp_reg_colors = {"High Risk": "#ef4444", "Elevated Risk": "#f59e0b",
                                "Moderate Risk": "#4f8ef7", "Low Risk": "#27ae60"}

            _ca126, _cb126, _cc126, _cd126 = st.columns(4)
            _ca126.metric("Composite Score", f"{_cur_comp:.0f}/100" if pd.notna(_cur_comp) else "—")
            _cb126.metric("Regime", _comp_reg)
            _cc126.metric("Historical Pctile", f"{_comp_pctile:.0f}th" if pd.notna(_comp_pctile) else "—")
            _cd126.metric("All-Time High", f"{_comp_max:.0f}" if pd.notna(_comp_max) else "—")

            if pd.notna(_cur_comp) and _cur_comp >= 70:
                st.error(f"Composite score in High Risk territory ({_cur_comp:.0f}/100) — "
                         "broad multi-dimensional stress confirmed. Full defensive posture warranted.")

            # Full-history composite with regime bands
            _comp_full = _comp126.copy()
            _fig126a = _go126.Figure()
            _fig126a.add_hrect(y0=70, y1=105, fillcolor="rgba(239,68,68,0.08)", line_width=0)
            _fig126a.add_hrect(y0=50, y1=70, fillcolor="rgba(245,158,11,0.06)", line_width=0)
            _fig126a.add_hrect(y0=30, y1=50, fillcolor="rgba(79,142,247,0.04)", line_width=0)
            _fig126a.add_trace(_go126.Scatter(
                x=_comp_full.index, y=_comp_full[_comp126_col],
                line=dict(color="#e2e8f0", width=1.5), name="Composite Score",
                fill="tozeroy", fillcolor="rgba(226,232,240,0.05)",
                hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
            ))
            for _thresh, _tc in [(30, "rgba(79,142,247,0.3)"), (50, "rgba(245,158,11,0.4)"),
                                  (70, "rgba(239,68,68,0.5)")]:
                _fig126a.add_hline(y=_thresh, line=dict(color=_tc, dash="dot", width=1))
            _fig126a.add_hline(y=_comp_mean, line=dict(color="rgba(255,255,255,0.2)", dash="dash", width=1),
                               annotation_text=f"Mean: {_comp_mean:.0f}",
                               annotation_font_color="rgba(255,255,255,0.5)")
            _fig126a.update_layout(
                height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Composite Risk Score — Full History", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           color="#6b7280", title="Score (0–100)"),
                xaxis=dict(showgrid=False, color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig126a, use_container_width=True)
            st.caption("Red band = High Risk (≥70) · Orange = Elevated (50–70) · Blue = Moderate (30–50) · Dashed = historical mean")

            # Rolling % time in each regime
            _comp126["regime"] = _comp126[_comp126_col].apply(_comp_regime)
            _regime_freq126 = _comp126["regime"].value_counts(normalize=True).mul(100)
            _fig126b = _go126.Figure()
            _fig126b.add_trace(_go126.Bar(
                x=_regime_freq126.index.tolist(), y=_regime_freq126.values.tolist(),
                marker_color=[_comp_reg_colors.get(r, "#6b7280") for r in _regime_freq126.index],
                text=[f"{v:.0f}%" for v in _regime_freq126.values], textposition="auto",
                hovertemplate="%{x}: %{y:.0f}%<extra></extra>",
            ))
            _fig126b.update_layout(
                height=190, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Time Spent in Each Risk Regime (Full History)", font=dict(size=12, color="#9aa0aa")),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="% of days"),
                xaxis=dict(color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig126b, use_container_width=True)

            # Composite vs HY spread full history
            if "hy_spread" in _comp_full.columns:
                _fig126c = _go126.Figure()
                _fig126c.add_trace(_go126.Scatter(
                    x=_comp_full.index, y=_comp_full[_comp126_col],
                    name="Composite Score", line=dict(color="#e2e8f0", width=1.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.0f}<extra></extra>",
                ))
                _fig126c.add_trace(_go126.Scatter(
                    x=_comp_full.index, y=_comp_full["hy_spread"],
                    name="HY Spread (bps)", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    yaxis="y2",
                    hovertemplate="%{x|%Y-%m-%d}<br>HY: %{y:.0f}bps<extra></extra>",
                ))
                _fig126c.update_layout(
                    height=260, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=30, b=50),
                    title=dict(text="Composite Score vs HY Spread — Full History", font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               color="#e2e8f0", title="Composite Score"),
                    yaxis2=dict(overlaying="y", side="right", color="#f59e0b", title="HY Spread (bps)"),
                    xaxis=dict(showgrid=False, color="#6b7280"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10),
                                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig126c, use_container_width=True)
        else:
            st.info("Composite risk score not found — run the full scoring pipeline.")
    except Exception as _e126:
        _err_track(_active_sub, _e126)
        st.caption(f"Composite history: {_e126}")


if _active_sub == 128:
    try:
        import plotly.graph_objects as _go128
        from src.regime_attribution import SCORE_COLS, DISPLAY_NAMES
        _sc128_cols = [v for v in SCORE_COLS.values() if v in df.columns]
        if len(_sc128_cols) >= 2:
            _sc128_df = df[_sc128_cols].dropna(how="all")
            _sc128_labels = [DISPLAY_NAMES.get(k, k) for k, v in SCORE_COLS.items() if v in _sc128_cols]
            # Rolling 252d correlation (last available snapshot)
            _corr128 = _sc128_df.rolling(252, min_periods=63).corr().dropna(how="all")
            # Take the most recent correlation slice
            _last_date128 = _corr128.index.get_level_values(0).max()
            _corr_latest = _corr128.loc[_last_date128]
            # Reindex to align labels
            _corr_mat = _corr_latest.reindex(index=_sc128_cols, columns=_sc128_cols)
            _mat_vals = _corr_mat.values.tolist()
            _fig128a = _go128.Figure(data=_go128.Heatmap(
                z=_mat_vals,
                x=_sc128_labels, y=_sc128_labels,
                colorscale=[
                    [0.0, "#1e40af"], [0.5, "#1a1f2e"], [1.0, "#dc2626"]
                ],
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in _mat_vals],
                texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            ))
            _fig128a.update_layout(
                height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Sub-Score Rolling 252d Correlation Matrix (current)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(color="#6b7280", tickangle=-30),
                yaxis=dict(color="#6b7280"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig128a, use_container_width=True)
            # Rolling correlation of each score vs composite over time
            if "composite_risk_score_smooth" in df.columns:
                _fig128b = _go128.Figure()
                _comp128 = df["composite_risk_score_smooth"]
                _colors128 = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#64748b","#ec4899"]
                for i, (col, label) in enumerate(zip(_sc128_cols, _sc128_labels)):
                    _roll_corr = df[col].rolling(252, min_periods=63).corr(_comp128).dropna()
                    _fig128b.add_trace(_go128.Scatter(
                        x=_roll_corr.index, y=_roll_corr.values,
                        mode="lines", name=label,
                        line=dict(color=_colors128[i % len(_colors128)], width=1.2),
                        hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
                    ))
                _fig128b.add_hline(y=0, line_color="#4b5563", line_width=1, line_dash="dot")
                _fig128b.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Rolling 252d Corr: Each Sub-Score vs Composite", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Correlation"),
                    legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig128b, use_container_width=True)
            st.caption("Red = high positive correlation; Blue = negative. Sub-scores that de-correlate signal divergent market stress.")
        else:
            st.info("Need at least 2 sub-score columns — run the full scoring pipeline.")
    except Exception as _e128:
        _err_track(_active_sub, _e128)
        st.caption(f"Score correlations: {_e128}")

# sub129 — Score Z-Scores (tab_siglab)

if _active_sub == 129:
    try:
        import plotly.graph_objects as _go129
        import numpy as _np129
        from src.regime_attribution import SCORE_COLS, DISPLAY_NAMES
        _sc129_cols = [v for v in SCORE_COLS.values() if v in df.columns]
        if _sc129_cols:
            _fig129a = _go129.Figure()
            _colors129 = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#64748b","#ec4899"]
            _current_zscores = {}
            for i, (k, v) in enumerate(SCORE_COLS.items()):
                if v not in df.columns:
                    continue
                _s = df[v].dropna()
                if len(_s) < 63:
                    continue
                _roll = _s.rolling(252, min_periods=63)
                _z = (_s - _roll.mean()) / _roll.std(ddof=1).replace(0, _np129.nan)
                _label = DISPLAY_NAMES.get(k, k)
                _fig129a.add_trace(_go129.Scatter(
                    x=_z.index, y=_z.values,
                    mode="lines", name=_label,
                    line=dict(color=_colors129[i % len(_colors129)], width=1.2),
                    hovertemplate=f"{_label}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}σ<extra></extra>",
                ))
                _current_zscores[_label] = float(_z.iloc[-1]) if _z.notna().any() else None
            # Reference bands
            for _lvl, _col in [(2.0, "rgba(220,38,38,0.15)"), (-2.0, "rgba(37,99,235,0.15)")]:
                _fig129a.add_hline(y=_lvl, line_color="#4b5563", line_width=1, line_dash="dot")
            _fig129a.update_layout(
                height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Sub-Score Z-Scores vs Rolling 252d History", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Z-Score (σ)", zeroline=True, zerolinecolor="#4b5563"),
                legend=dict(orientation="h", y=-0.35, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig129a, use_container_width=True)
            # Current z-score snapshot bar chart
            if _current_zscores:
                _z_labels = list(_current_zscores.keys())
                _z_vals = [v if v is not None else 0.0 for v in _current_zscores.values()]
                _z_bar_colors = ["#ef4444" if v > 1.0 else ("#f59e0b" if v > 0.0 else "#3b82f6") for v in _z_vals]
                _fig129b = _go129.Figure()
                _fig129b.add_trace(_go129.Bar(
                    x=_z_labels, y=_z_vals,
                    marker_color=_z_bar_colors,
                    text=[f"{v:+.2f}σ" for v in _z_vals], textposition="auto",
                    hovertemplate="%{x}: %{y:+.2f}σ<extra></extra>",
                ))
                _fig129b.add_hline(y=0, line_color="#4b5563", line_width=1)
                _fig129b.add_hline(y=2, line_color="#dc2626", line_width=1, line_dash="dash")
                _fig129b.add_hline(y=-2, line_color="#1d4ed8", line_width=1, line_dash="dash")
                _fig129b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text="Current Z-Score Snapshot — Which Scores Are Anomalous?", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280", tickangle=-20),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Z-Score (σ)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig129b, use_container_width=True)
            st.caption(">+2σ = historically elevated; <−2σ = historically suppressed. Dotted lines at ±2σ.")
        else:
            st.info("Sub-score columns not found — run the full scoring pipeline.")
    except Exception as _e129:
        _err_track(_active_sub, _e129)
        st.caption(f"Score z-scores: {_e129}")

# sub130 — HY Lead-Lag (tab_credit)

if _active_sub == 132:
    try:
        import plotly.graph_objects as _go132
        import numpy as _np132
        from src.regime_attribution import SCORE_COLS, DISPLAY_NAMES
        _sc132_cols = [(k, v) for k, v in SCORE_COLS.items() if v in df.columns]
        if _sc132_cols:
            _lags132 = [1, 5, 21, 63]
            _autocorrs = {}
            for k, v in _sc132_cols:
                _s = df[v].dropna()
                if len(_s) < 126:
                    continue
                _label = DISPLAY_NAMES.get(k, k)
                _acs = []
                for lag in _lags132:
                    _s_lag = _s.shift(lag)
                    _valid = _s.align(_s_lag, join="inner")
                    _combined = _valid[0].dropna().align(_valid[1].dropna(), join="inner")
                    if len(_combined[0]) > 20:
                        _ac = float(_combined[0].corr(_combined[1]))
                    else:
                        _ac = _np132.nan
                    _acs.append(_ac)
                _autocorrs[_label] = _acs
            if _autocorrs:
                _names132 = list(_autocorrs.keys())
                _fig132a = _go132.Figure(data=_go132.Heatmap(
                    z=[list(_autocorrs[n]) for n in _names132],
                    x=[f"Lag {lag}d" for lag in _lags132],
                    y=_names132,
                    colorscale=[[0.0, "#1e40af"], [0.5, "#1a1f2e"], [1.0, "#dc2626"]],
                    zmin=0, zmax=1,
                    text=[[f"{v:.2f}" if not _np132.isnan(v) else "" for v in _autocorrs[n]] for n in _names132],
                    texttemplate="%{text}",
                    hovertemplate="%{y} at %{x}: %{z:.2f}<extra></extra>",
                ))
                _fig132a.update_layout(
                    height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                    title=dict(text="Score Autocorrelation by Lag (persistence = high autocorr)", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280"),
                    yaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig132a, use_container_width=True)
                # Bar chart: autocorr at 21d lag per score (persistence summary)
                _pers_vals = [_autocorrs[n][2] for n in _names132]  # index 2 = 21d lag
                _fig132b = _go132.Figure()
                _fig132b.add_trace(_go132.Bar(
                    x=_names132, y=_pers_vals,
                    marker_color=["#3b82f6" if v > 0.7 else ("#f59e0b" if v > 0.5 else "#ef4444") for v in _pers_vals],
                    text=[f"{v:.2f}" if not _np132.isnan(v) else "N/A" for v in _pers_vals],
                    textposition="auto",
                    hovertemplate="%{x}: autocorr(21d)=%{y:.2f}<extra></extra>",
                ))
                _fig132b.add_hline(y=0.7, line_color="#3b82f6", line_width=1, line_dash="dash",
                                   annotation_text="Sticky (0.7)", annotation_font=dict(color="#3b82f6", size=8))
                _fig132b.add_hline(y=0.5, line_color="#f59e0b", line_width=1, line_dash="dash",
                                   annotation_text="Moderate (0.5)", annotation_font=dict(color="#f59e0b", size=8))
                _fig132b.update_layout(
                    height=210, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                    title=dict(text="21-Day Autocorrelation — Which Scores Are Stickiest?", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280", tickangle=-20),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               title="Autocorr(21d)", range=[0, 1]),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig132b, use_container_width=True)
                st.caption("Blue >0.7 = sticky/persistent signal. Red <0.5 = mean-reverting/noisy. Sticky scores confirm regime; noisy scores provide contrast signals.")
            else:
                st.info("Insufficient history to compute autocorrelations (need 126+ observations).")
        else:
            st.info("Sub-score columns not found — run the full scoring pipeline.")
    except Exception as _e132:
        _err_track(_active_sub, _e132)
        st.caption(f"Score persistence: {_e132}")

# sub133 — Drawdown Anatomy (tab_risk)

if _active_sub == 135:
    try:
        import plotly.graph_objects as _go135
        import numpy as _np135
        from src.regime_attribution import SCORE_COLS, DISPLAY_NAMES
        _sc135_cols = [(k, v) for k, v in SCORE_COLS.items() if v in df.columns]
        if _sc135_cols:
            # Show distributions for the 4 highest-weight scores + composite
            _priority_keys = ["treasury", "complacency", "credit_risk", "macro_risk"]
            _show135 = [(k, v) for k, v in _sc135_cols if k in _priority_keys]
            if "composite_risk_score_smooth" in df.columns:
                _show135.append(("composite", "composite_risk_score_smooth"))
            _colors135 = {"treasury": "#3b82f6", "complacency": "#10b981",
                          "credit_risk": "#ef4444", "macro_risk": "#f59e0b", "composite": "#ffffff"}
            for k, v in _show135:
                _s = df[v].dropna()
                if len(_s) < 30:
                    continue
                _label = "Composite" if k == "composite" else DISPLAY_NAMES.get(k, k)
                _current_val = float(_s.iloc[-1])
                _pct = float((_s < _current_val).mean() * 100)
                _fig135 = _go135.Figure()
                # Histogram
                _fig135.add_trace(_go135.Histogram(
                    x=_s.values, nbinsx=40,
                    marker_color=_colors135.get(k, "#6b7280"),
                    opacity=0.55, name="Historical",
                    hovertemplate="Score %{x:.0f}: %{y} days<extra></extra>",
                ))
                # Current value vline
                _fig135.add_vline(
                    x=_current_val,
                    line_color="#ffffff", line_width=2,
                    annotation_text=f"Now: {_current_val:.0f} ({_pct:.0f}th pct)",
                    annotation_position="top right",
                    annotation_font=dict(color="#ffffff", size=9),
                )
                # Threshold bands
                _fig135.add_vrect(x0=50, x1=100, fillcolor="rgba(239,68,68,0.06)", layer="below", line_width=0)
                _fig135.add_vline(x=50, line_color="#6b7280", line_width=1, line_dash="dot")
                _fig135.add_vline(x=70, line_color="#9b1c1c", line_width=1, line_dash="dot")
                _fig135.update_layout(
                    height=175, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text=f"{_label} — Distribution vs Current ({_current_val:.0f}, {_pct:.0f}th pct)", font=dict(size=11, color="#9aa0aa")),
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", range=[0, 100]),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    showlegend=False,
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig135, use_container_width=True)
            st.caption("White line = current reading. Red shading >50. Dotted lines at 50 and 70.")
        else:
            st.info("Sub-score columns not found — run the full scoring pipeline.")
    except Exception as _e135:
        _err_track(_active_sub, _e135)
        st.caption(f"Score distributions: {_e135}")

# sub136 — Spread Vol Regime (tab_credit)

if _active_sub == 137:
    try:
        import plotly.graph_objects as _go137
        import numpy as _np137
        if "composite_risk_score_smooth" in df.columns:
            _comp137 = df["composite_risk_score_smooth"].dropna()
            # Daily change and rolling windows
            _daily_chg = _comp137.diff(1)
            _weekly_chg = _comp137.diff(5)
            _monthly_chg = _comp137.diff(21)
            # Rolling percentile of the daily change
            _chg_pct = _daily_chg.rolling(504, min_periods=63).apply(
                lambda x: float((x[:-1] < x[-1]).mean() * 100) if len(x) > 1 else _np137.nan,
                raw=True
            )
            # Streak analysis: consecutive days rising vs falling
            _direction = _daily_chg.apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
            _streaks = []
            _streak_len = 1
            for i in range(1, len(_direction)):
                if _direction.iloc[i] == _direction.iloc[i-1] and _direction.iloc[i] != 0:
                    _streak_len += 1
                else:
                    _streaks.append(_streak_len * _direction.iloc[i-1])
                    _streak_len = 1
            _streaks.append(_streak_len * _direction.iloc[-1])
            _current_streak = _streaks[-1] if _streaks else 0
            # Chart: composite daily change bar (last 252d)
            _recent137 = _daily_chg.tail(252)
            _bar_colors137 = ["#ef4444" if v > 0 else "#3b82f6" for v in _recent137.values]
            _fig137a = _go137.Figure()
            _fig137a.add_trace(_go137.Bar(
                x=_recent137.index, y=_recent137.values,
                marker_color=_bar_colors137,
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f} pts<extra></extra>",
            ))
            _fig137a.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig137a.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Composite Score Daily Change (last 252 days) — Red=Rising, Blue=Falling", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Daily Δ (pts)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig137a, use_container_width=True)
            # Rolling 21d composite change with percentile band
            _fig137b = _go137.Figure()
            _fig137b.add_trace(_go137.Scatter(
                x=_monthly_chg.index, y=_monthly_chg.values,
                mode="lines", name="21d Change",
                line=dict(color="#8b5cf6", width=1.2),
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.1f} pts<extra></extra>",
            ))
            # Rolling ±1σ band
            _m_roll = _monthly_chg.rolling(252, min_periods=63)
            _m_mean = _m_roll.mean()
            _m_std = _m_roll.std(ddof=1)
            _fig137b.add_trace(_go137.Scatter(
                x=_m_mean.index, y=(_m_mean + _m_std).values,
                mode="lines", line=dict(color="rgba(139,92,246,0.2)", width=0.5),
                showlegend=False, hoverinfo="skip",
            ))
            _fig137b.add_trace(_go137.Scatter(
                x=_m_mean.index, y=(_m_mean - _m_std).values,
                mode="lines", line=dict(color="rgba(139,92,246,0.2)", width=0.5),
                fill="tonexty", fillcolor="rgba(139,92,246,0.07)",
                showlegend=False, hoverinfo="skip",
            ))
            _fig137b.add_hline(y=0, line_color="#4b5563", line_width=1)
            _fig137b.update_layout(
                height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="Composite Score 21-Day Change with ±1σ Band", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="21d Δ (pts)"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig137b, use_container_width=True)
            # Current stats
            _curr_d = float(_daily_chg.iloc[-1]) if _daily_chg.notna().any() else None
            _curr_w = float(_weekly_chg.iloc[-1]) if _weekly_chg.notna().any() else None
            _curr_m = float(_monthly_chg.iloc[-1]) if _monthly_chg.notna().any() else None
            _streak_dir = "rising" if _current_streak > 0 else ("falling" if _current_streak < 0 else "flat")
            st.caption(
                f"Daily Δ: {f'{_curr_d:+.2f}' if _curr_d is not None else 'N/A'} pts · "
                f"5d Δ: {f'{_curr_w:+.1f}' if _curr_w is not None else 'N/A'} pts · "
                f"21d Δ: {f'{_curr_m:+.1f}' if _curr_m is not None else 'N/A'} pts · "
                f"Streak: {abs(_current_streak)}d {_streak_dir}"
            )
        else:
            st.info("composite_risk_score_smooth not found — run the full scoring pipeline.")
    except Exception as _e137:
        _err_track(_active_sub, _e137)
        st.caption(f"Score gradient: {_e137}")

# sub138 — Macro-Credit Decoupling (tab_macro)

if _active_sub == 140:
    try:
        import plotly.graph_objects as _go140
        import numpy as _np140
        import pandas as _pd140
        if "composite_risk_score_smooth" in df.columns and "hy_spread" in df.columns:
            _comp140 = df["composite_risk_score_smooth"].dropna()
            _hy140 = df["hy_spread"].dropna()
            _sp500140 = df["sp500"].dropna() if "sp500" in df.columns else None
            _horizons140 = [21, 63, 126]
            _thresholds140 = [(50, "≥50 Alert", "#f59e0b"), (70, "≥70 High Alert", "#ef4444")]
            st.markdown("#### Signal-to-Outcome: What Happens After Composite Crosses Thresholds?")
            for _thresh, _label, _color in _thresholds140:
                # Find crossing days: composite first hits threshold from below
                _above = (_comp140 >= _thresh).astype(int)
                _cross_days = _comp140.index[(_above.diff() == 1)]
                if len(_cross_days) < 3:
                    st.caption(f"{_label}: not enough crossing events.")
                    continue
                st.markdown(f"**{_label}** ({len(_cross_days)} crossing events)")
                _fwd_hy = {h: [] for h in _horizons140}
                _fwd_sp = {h: [] for h in _horizons140}
                for _d in _cross_days:
                    _idx = _hy140.index.searchsorted(_d)
                    for h in _horizons140:
                        if _idx + h < len(_hy140):
                            _fwd_hy[h].append(float(_hy140.iloc[_idx + h] - _hy140.iloc[_idx]))
                        if _sp500140 is not None:
                            _idx_sp = _sp500140.index.searchsorted(_d)
                            if _idx_sp + h < len(_sp500140):
                                _fwd_sp[h].append(
                                    float(_sp500140.iloc[_idx_sp + h] / _sp500140.iloc[_idx_sp] - 1) * 100
                                )
                # Box plots for HY forward change
                _fig140 = _go140.Figure()
                for h in _horizons140:
                    if _fwd_hy[h]:
                        _fig140.add_trace(_go140.Box(
                            y=_fwd_hy[h], name=f"HY +{h}d",
                            marker_color=_color,
                            boxmean=True,
                            hovertemplate=f"HY Δ at +{h}d: %{{y:+.0f}}bps<extra></extra>",
                        ))
                if _sp500140 is not None:
                    for h in _horizons140:
                        if _fwd_sp[h]:
                            _fig140.add_trace(_go140.Box(
                                y=_fwd_sp[h], name=f"SP500 +{h}d%",
                                marker_color="#3b82f6",
                                boxmean=True,
                                hovertemplate=f"SP500 ret at +{h}d: %{{y:+.1f}}%<extra></extra>",
                            ))
                _fig140.add_hline(y=0, line_color="#4b5563", line_width=1)
                _fig140.update_layout(
                    height=270, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                    title=dict(text=f"Forward Returns After {_label} Crossing — HY Δ (bps) and SP500 (%)",
                               font=dict(size=12, color="#9aa0aa")),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                               zeroline=True, zerolinecolor="#4b5563"),
                    xaxis=dict(color="#6b7280"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig140, use_container_width=True)
                # Summary table
                _rows140 = []
                for h in _horizons140:
                    if _fwd_hy[h]:
                        _arr = _np140.array(_fwd_hy[h])
                        _rows140.append({
                            "Horizon": f"+{h}d",
                            "HY median Δ (bps)": round(float(_np140.median(_arr)), 0),
                            "HY mean Δ (bps)": round(float(_np140.mean(_arr)), 0),
                            "HY % positive": f"{(_arr > 0).mean()*100:.0f}%",
                            "N": len(_arr),
                        })
                if _rows140:
                    st.dataframe(_pd140.DataFrame(_rows140).set_index("Horizon"), use_container_width=True)
            st.caption("Positive HY Δ = spreads widened (worse credit). Box: median line, mean cross, whiskers=1.5×IQR.")
        else:
            st.info("composite_risk_score_smooth or hy_spread not found — run the full scoring pipeline.")
    except Exception as _e140:
        _err_track(_active_sub, _e140)
        st.caption(f"Forward returns: {_e140}")

# sub141 — Regime-Conditional Spread Distribution (Credit Markets)

if _active_sub == 144:
    try:
        import plotly.graph_objects as _go144
        import numpy as _np144
        import pandas as _pd144
        if "composite_risk_score_smooth" in df.columns and "hy_spread" in df.columns:
            _comp144 = df["composite_risk_score_smooth"].dropna()
            _hy144 = df["hy_spread"].dropna()
            _j144 = _comp144.to_frame("score").join(_hy144.to_frame("hy"), how="inner").dropna()
            _j144["score_21d_chg"] = _j144["score"].diff(21)
            _j144["hy_fwd_21d"] = _j144["hy"].diff(21).shift(-21)  # forward 21d HY change
            _j144 = _j144.dropna()
            # Quadrant analysis: high level vs rising momentum
            _med_score = float(_j144["score"].median())
            _med_chg = float(_j144["score_21d_chg"].median())
            _j144["quadrant"] = _j144.apply(
                lambda r: (
                    "High+Rising" if r["score"] >= _med_score and r["score_21d_chg"] >= _med_chg
                    else ("High+Falling" if r["score"] >= _med_score
                          else ("Low+Rising" if r["score_21d_chg"] >= _med_chg else "Low+Falling"))
                ), axis=1
            )
            _QUAD_COLORS = {"High+Rising": "#ef4444", "High+Falling": "#f59e0b",
                            "Low+Rising": "#8b5cf6", "Low+Falling": "#10b981"}
            # Scatter: score level vs score 21d change, colored by forward HY change
            _fig144a = _go144.Figure()
            for quad, color in _QUAD_COLORS.items():
                _sub = _j144[_j144["quadrant"] == quad]
                if len(_sub) < 3:
                    continue
                _fig144a.add_trace(_go144.Scatter(
                    x=_sub["score"], y=_sub["score_21d_chg"],
                    mode="markers",
                    marker=dict(
                        color=_sub["hy_fwd_21d"],
                        colorscale=[[0, "#1e40af"], [0.5, "#f59e0b"], [1, "#dc2626"]],
                        size=3, opacity=0.5,
                        cmin=-50, cmax=50,
                        showscale=(quad == list(_QUAD_COLORS.keys())[0]),
                        colorbar=dict(title="Fwd HY Δ<br>(bps)", titlefont=dict(color="#9aa0aa", size=8),
                                      tickfont=dict(color="#9aa0aa", size=8)) if quad == list(_QUAD_COLORS.keys())[0] else None,
                    ),
                    name=quad,
                    hovertemplate=f"{quad}<br>Score: %{{x:.0f}} · Δ21d: %{{y:+.1f}} · Fwd HY: %{{marker.color:+.0f}}bps<extra></extra>",
                ))
            _fig144a.add_vline(x=_med_score, line_color="#6b7280", line_width=1, line_dash="dot")
            _fig144a.add_hline(y=_med_chg, line_color="#6b7280", line_width=1, line_dash="dot")
            _fig144a.update_layout(
                height=310, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Score Level vs 21d Momentum — Dot color = forward 21d HY Δ (bps)",
                           font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Score Level"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="21d Score Change", zeroline=True, zerolinecolor="#4b5563"),
                legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig144a, use_container_width=True)
            # Quadrant forward HY stats
            _quad_rows = []
            for quad in _QUAD_COLORS:
                _sub = _j144[_j144["quadrant"] == quad]["hy_fwd_21d"].dropna()
                if len(_sub) < 5:
                    continue
                _quad_rows.append({
                    "Quadrant": quad,
                    "N": len(_sub),
                    "Median Fwd HY Δ (bps)": round(float(_sub.median()), 0),
                    "Mean Fwd HY Δ (bps)": round(float(_sub.mean()), 0),
                    "% Positive (widening)": f"{(_sub > 0).mean()*100:.0f}%",
                })
            if _quad_rows:
                st.markdown("**Quadrant Summary: Forward 21d HY Spread Change**")
                st.dataframe(_pd144.DataFrame(_quad_rows).set_index("Quadrant"), use_container_width=True)
                st.caption(
                    "High+Rising = high score AND accelerating → strongest widening signal. "
                    "High+Falling = elevated but improving → mean reversion. "
                    "Low+Rising = watch — momentum turning. "
                    "Low+Falling = benign. Dots left of median = momentum dominates level; right = level dominates."
                )
        else:
            st.info("composite_risk_score_smooth or hy_spread not found — run the full scoring pipeline.")
    except Exception as _e144:
        _err_track(_active_sub, _e144)
        st.caption(f"Score momentum vs level: {_e144}")

# m9 — Thresholds (Models)

if _active_sub == 147:
    try:
        import plotly.graph_objects as _go147
        import numpy as _np147
        import pandas as _pd147
        _MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        _cols_to_check = []
        if "composite_risk_score_smooth" in df.columns:
            _cols_to_check.append(("Composite", "composite_risk_score_smooth"))
        if "hy_spread" in df.columns:
            _cols_to_check.append(("HY Spread", "hy_spread"))
        if "vix" in df.columns:
            _cols_to_check.append(("VIX", "vix"))
        if not _cols_to_check:
            st.info("No series found for seasonality analysis.")
        else:
            for _label, _col in _cols_to_check:
                _s147 = df[_col].dropna()
                if len(_s147) < 252:
                    continue
                _s147.index = _pd147.to_datetime(_s147.index)
                _monthly = _s147.groupby(_s147.index.month)
                _fig147 = _go147.Figure()
                for m in range(1, 13):
                    _vals = _monthly.get_group(m).values if m in _monthly.groups else []
                    if len(_vals) < 5:
                        continue
                    _fig147.add_trace(_go147.Box(
                        y=_vals, name=_MONTH_NAMES[m-1],
                        marker_color="#3b82f6",
                        boxmean=True,
                        hovertemplate=f"{_MONTH_NAMES[m-1]}: %{{y:.1f}}<extra></extra>",
                    ))
                _monthly_means = [float(_s147[_s147.index.month == m].mean()) for m in range(1, 13)]
                _fig147.add_trace(_go147.Scatter(
                    x=_MONTH_NAMES, y=_monthly_means,
                    mode="lines+markers",
                    line=dict(color="#ef4444", width=1.5),
                    marker=dict(size=6, color="#ef4444"),
                    name="Monthly Mean",
                    hovertemplate="%{x}: mean=%{y:.1f}<extra></extra>",
                ))
                _fig147.update_layout(
                    height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=35, b=8),
                    title=dict(text=f"{_label} — Seasonal Distribution by Calendar Month",
                               font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                    showlegend=False,
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig147, use_container_width=True)
            # Month-of-year mean table for composite
            if "composite_risk_score_smooth" in df.columns:
                _s147c = df["composite_risk_score_smooth"].dropna()
                _s147c.index = _pd147.to_datetime(_s147c.index)
                _rows147 = []
                for m in range(1, 13):
                    _vals = _s147c[_s147c.index.month == m]
                    if len(_vals) < 3: continue
                    _rows147.append({
                        "Month": _MONTH_NAMES[m-1],
                        "Mean Score": round(float(_vals.mean()), 1),
                        "Median": round(float(_vals.median()), 1),
                        "P25": round(float(_vals.quantile(0.25)), 1),
                        "P75": round(float(_vals.quantile(0.75)), 1),
                        "N days": len(_vals),
                    })
                if _rows147:
                    st.dataframe(_pd147.DataFrame(_rows147).set_index("Month"), use_container_width=True)
                    _high_month = max(_rows147, key=lambda x: x["Mean Score"])
                    _low_month = min(_rows147, key=lambda x: x["Mean Score"])
                    st.caption(
                        f"Historically highest-stress month: **{_high_month['Month']}** ({_high_month['Mean Score']:.1f} avg). "
                        f"Lowest: **{_low_month['Month']}** ({_low_month['Mean Score']:.1f} avg). "
                        "Seasonal patterns in credit often reflect earnings calendar, fiscal year-end, and summer liquidity."
                    )
    except Exception as _e147:
        _err_track(_active_sub, _e147)
        st.caption(f"Score seasonality: {_e147}")


# sub148 — Vol of Vol (Risk Monitors)

if _active_sub == 150:
    try:
        import plotly.graph_objects as _go150
        import numpy as _np150
        import pandas as _pd150
        from src.regime_attribution import SCORE_COLS, COMPOSITE_WEIGHTS, DISPLAY_NAMES
        _sc150_cols = [(k, v) for k, v in SCORE_COLS.items() if v in df.columns]
        if len(_sc150_cols) >= 3 and "hy_spread" in df.columns:
            _hy150 = df["hy_spread"].dropna()
            # Agreement score: count of sub-scores >= 50
            _agree_df = df[[v for _, v in _sc150_cols]].dropna(how="all")
            _n_above50 = (_agree_df >= 50).sum(axis=1)
            _n_above70 = (_agree_df >= 70).sum(axis=1)
            _n_total = len(_sc150_cols)

            _fig150a = _go150.Figure()
            _fig150a.add_trace(_go150.Scatter(
                x=_n_above50.index, y=_n_above50.values,
                mode="lines", name="Scores ≥50",
                line=dict(color="#f59e0b", width=1.2),
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                hovertemplate="%{x|%Y-%m-%d}: %{y} of " + str(_n_total) + " scores ≥50<extra></extra>",
            ))
            _fig150a.add_trace(_go150.Scatter(
                x=_n_above70.index, y=_n_above70.values,
                mode="lines", name="Scores ≥70",
                line=dict(color="#ef4444", width=1.0),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.06)",
                hovertemplate="%{x|%Y-%m-%d}: %{y} of " + str(_n_total) + " scores ≥70<extra></extra>",
            ))
            _fig150a.add_hline(y=_n_total // 2, line_color="#6b7280", line_width=1, line_dash="dot",
                               annotation_text="Majority", annotation_font=dict(color="#6b7280", size=8))
            _fig150a.update_layout(
                height=210, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text=f"Score Ensemble Agreement: # of {_n_total} Sub-Scores Above Threshold",
                           font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Count", range=[0, _n_total + 1]),
                legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig150a, use_container_width=True)

            # Forward HY outcomes by agreement level
            _j150 = _n_above50.to_frame("n_agree").join(_hy150.to_frame("hy"), how="inner").dropna()
            _j150["hy_fwd_21d"] = _j150["hy"].diff(21).shift(-21)
            _j150 = _j150.dropna()
            _bins150 = [(0, 1, "0–1 (No consensus)"), (2, 3, "2–3 (Weak)"),
                        (4, 5, "4–5 (Majority)"), (6, _n_total, f"6–{_n_total} (Strong)")]
            _rows150 = []
            for lo, hi, label in _bins150:
                _sub = _j150[(_j150["n_agree"] >= lo) & (_j150["n_agree"] <= hi)]["hy_fwd_21d"].dropna()
                if len(_sub) < 5: continue
                _rows150.append({
                    "Agreement": label, "N": len(_sub),
                    "Median Fwd HY Δ (bps)": round(float(_sub.median()), 0),
                    "Mean Fwd HY Δ (bps)": round(float(_sub.mean()), 0),
                    "% Widening": f"{(_sub > 0).mean()*100:.0f}%",
                })
            if _rows150:
                st.markdown("**Ensemble Conviction → Forward 21d HY Spread Change**")
                st.dataframe(_pd150.DataFrame(_rows150).set_index("Agreement"), use_container_width=True)

            # Current snapshot
            _curr_agree50 = int(_n_above50.iloc[-1]) if _n_above50.notna().any() else 0
            _curr_agree70 = int(_n_above70.iloc[-1]) if _n_above70.notna().any() else 0
            _agree_pct50 = _curr_agree50 / _n_total * 100
            st.caption(
                f"Current: **{_curr_agree50}/{_n_total}** scores ≥50 ({_agree_pct50:.0f}% consensus) · "
                f"**{_curr_agree70}/{_n_total}** ≥70 (high-stress). "
                "Strong ensemble agreement historically leads to more reliable HY spread widening signals."
            )
        else:
            st.info("Need at least 3 sub-score columns and hy_spread — run the full scoring pipeline.")
    except Exception as _e150:
        _err_track(_active_sub, _e150)
        st.caption(f"Score ensemble: {_e150}")


if _active_sub == 153:
    try:
        import plotly.graph_objects as _go153
        import numpy as _np153
        import pandas as _pd153
        _df153 = df.copy() if "df" in dir() else None
        _score_cols153 = [c for c in [
            "macro_risk_score_smooth", "credit_market_risk_score_smooth",
            "complacency_score_smooth", "liquidity_regime_score_smooth",
            "treasury_stress_score_smooth", "fx_commodity_score_smooth",
            "enhanced_funding_stress_score_smooth", "cross_asset_divergence_score_smooth",
        ] if _df153 is not None and c in _df153.columns]
        if len(_score_cols153) < 3:
            st.info("Need at least 3 sub-score columns for bootstrap CI.")
        else:
            st.subheader("Score Bootstrap Confidence Interval")
            st.caption("Bootstrap uncertainty bands around the composite score: resample sub-scores (with replacement) 200 times to estimate the range of plausible composite values. Wide bands = high model uncertainty; narrow bands = robust signal.")
            _sub153 = _df153[_score_cols153].tail(504).dropna(how="all")
            _comp153 = _sub153.mean(axis=1)
            _n_boot153 = 200
            _boot153 = _np153.zeros((_n_boot153, len(_sub153)))
            _rng153 = _np153.random.default_rng(42)
            _ncols153 = len(_score_cols153)
            for _b153 in range(_n_boot153):
                _bidx153 = _rng153.integers(0, _ncols153, size=_ncols153)
                _boot153[_b153] = _sub153.iloc[:, _bidx153].mean(axis=1).values
            _p5_153 = _np153.percentile(_boot153, 5, axis=0)
            _p25_153 = _np153.percentile(_boot153, 25, axis=0)
            _p75_153 = _np153.percentile(_boot153, 75, axis=0)
            _p95_153 = _np153.percentile(_boot153, 95, axis=0)
            _idx153 = _sub153.index
            _fig153 = _go153.Figure()
            _fig153.add_trace(_go153.Scatter(
                x=list(_idx153) + list(_idx153[::-1]),
                y=list(_p5_153) + list(_p95_153[::-1]),
                fill="toself", fillcolor="rgba(99,102,241,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="90% CI"
            ))
            _fig153.add_trace(_go153.Scatter(
                x=list(_idx153) + list(_idx153[::-1]),
                y=list(_p25_153) + list(_p75_153[::-1]),
                fill="toself", fillcolor="rgba(99,102,241,0.18)",
                line=dict(color="rgba(0,0,0,0)"), name="50% CI"
            ))
            _fig153.add_trace(_go153.Scatter(
                x=_idx153, y=_comp153.values,
                line=dict(color="#6366f1", width=2), name="Composite Score"
            ))
            _fig153.add_hline(y=50, line_dash="dash", line_color="#9aa0aa",
                              annotation_text="Stress threshold")
            _fig153.update_layout(
                title="Composite Score — Bootstrap Uncertainty Bands",
                height=400, yaxis_title="Score (0–100)", yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig153, use_container_width=True)
            _cur_p5_153 = float(_p5_153[-1])
            _cur_p95_153 = float(_p95_153[-1])
            _cur_comp153 = float(_comp153.iloc[-1])
            _ci_width153 = _cur_p95_153 - _cur_p5_153
            _uncert153 = "High" if _ci_width153 > 20 else ("Moderate" if _ci_width153 > 10 else "Low")
            st.caption(
                f"Current score: **{_cur_comp153:.1f}** · 90% CI: [{_cur_p5_153:.1f}, {_cur_p95_153:.1f}] "
                f"(width {_ci_width153:.1f}) · Uncertainty: **{_uncert153}** · "
                f"{len(_score_cols153)} sub-scores, n=200 resamples."
            )
    except Exception as _e153:
        _err_track(_active_sub, _e153)
        st.caption(f"Score bootstrap CI: {_e153}")


if _active_sub == 159:
    try:
        import plotly.graph_objects as _go159
        import numpy as _np159
        import pandas as _pd159
        _df159 = df.copy() if "df" in dir() else None
        _has159 = _df159 is not None and "composite_risk_score_smooth" in _df159.columns
        if not _has159:
            st.info("composite_risk_score_smooth required.")
        else:
            st.subheader("Score Calendar Heatmap")
            st.caption("Monthly average composite risk score by year and month. Dark red = high stress; dark green = low stress. Reveals seasonal patterns and year-over-year changes at a glance.")
            _comp159 = _df159["composite_risk_score_smooth"].dropna()
            _cal159 = _comp159.to_frame("score")
            _cal159["year"] = _cal159.index.year
            _cal159["month"] = _cal159.index.month
            _pivot159 = _cal159.groupby(["year", "month"])["score"].mean().unstack(level=1)
            _pivot159.columns = [_pd159.Timestamp(2000, m, 1).strftime("%b") for m in _pivot159.columns]
            _fig159 = _go159.Figure(data=_go159.Heatmap(
                z=_pivot159.values,
                x=list(_pivot159.columns),
                y=[str(y) for y in _pivot159.index],
                colorscale=[
                    [0.0, "#166534"], [0.3, "#22c55e"],
                    [0.5, "#6366f1"], [0.7, "#f59e0b"],
                    [1.0, "#7f1d1d"]
                ],
                zmin=0, zmax=100,
                colorbar=dict(title="Score", tickvals=[0, 25, 50, 75, 100]),
                text=_np159.round(_pivot159.values, 0),
                texttemplate="%{text:.0f}",
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Score: %{z:.1f}<extra></extra>"
            ))
            _fig159.update_layout(
                title="Monthly Average Composite Score by Year",
                height=max(300, len(_pivot159) * 28 + 100),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                xaxis=dict(side="top")
            )
            st.plotly_chart(_fig159, use_container_width=True)
            # Seasonal average
            _seas159 = _cal159.groupby("month")["score"].mean()
            _hot159 = int(_seas159.idxmax())
            _cold159 = int(_seas159.idxmin())
            _hot_name159 = _pd159.Timestamp(2000, _hot159, 1).strftime("%B")
            _cold_name159 = _pd159.Timestamp(2000, _cold159, 1).strftime("%B")
            st.caption(
                f"Seasonal: highest avg stress month = **{_hot_name159}** ({_seas159[_hot159]:.1f}) · "
                f"lowest = **{_cold_name159}** ({_seas159[_cold159]:.1f})."
            )
    except Exception as _e159:
        _err_track(_active_sub, _e159)
        st.caption(f"Score calendar: {_e159}")


if _active_sub == 160:
    try:
        import plotly.graph_objects as _go160
        import numpy as _np160
        import pandas as _pd160
        _df160 = df.copy() if "df" in dir() else None
        _has160 = _df160 is not None and "hy_spread" in _df160.columns
        if not _has160:
            st.info("hy_spread required.")
        else:
            st.subheader("Macro Factor Correlation Scan")
            st.caption("Rolling 63-day correlation of each macro input vs forward 21-day HY spread change. Identifies which indicators are currently the strongest leading signals for credit conditions. Sorted by absolute correlation — top rows = most predictive right now.")
            _fwd_hy160 = _df160["hy_spread"].diff(21).shift(-21)
            _candidates160 = [
                ("vix", "VIX Level"),
                ("vix_change_30d", "VIX 30d Change"),
                ("spread", "Yield Curve (2s10s)"),
                ("sp500_return_30d", "SP500 30d Return"),
                ("sp500_drawdown", "SP500 Drawdown"),
                ("nfci", "NFCI"),
                ("nfci_change_90d", "NFCI 90d Change"),
                ("unemployment", "Unemployment"),
                ("sahm_like", "Sahm-like"),
                ("real_yield_proxy", "Real Yield"),
                ("breakeven_10y", "Breakeven Inflation"),
                ("credit_impulse", "Credit Impulse"),
                ("hy_change_30d", "HY 30d Change"),
            ]
            _rows160 = []
            for _col160, _label160 in _candidates160:
                if _col160 not in _df160.columns:
                    continue
                _ser160 = _df160[_col160].dropna()
                _j160 = _ser160.to_frame("x").join(_fwd_hy160.to_frame("y"), how="inner").dropna()
                if len(_j160) < 126:
                    continue
                # Current rolling 63d corr (last window)
                _tail160 = _j160.tail(63)
                if len(_tail160) < 30:
                    continue
                _c160 = float(_tail160["x"].corr(_tail160["y"]))
                # Full history corr
                _c_full160 = float(_j160["x"].corr(_j160["y"]))
                _rows160.append({
                    "Factor": _label160,
                    "Current 63d Corr": round(_c160, 3),
                    "Full History Corr": round(_c_full160, 3),
                    "Signal": "Leading Higher" if _c160 > 0.2 else ("Leading Lower" if _c160 < -0.2 else "Neutral"),
                })
            if _rows160:
                _scan_df160 = (_pd160.DataFrame(_rows160)
                               .sort_values("Current 63d Corr", key=_np160.abs, ascending=False)
                               .reset_index(drop=True))
                # Bar chart
                _fig160 = _go160.Figure()
                _fig160.add_trace(_go160.Bar(
                    x=_scan_df160["Factor"],
                    y=_scan_df160["Current 63d Corr"],
                    marker_color=_np160.where(_scan_df160["Current 63d Corr"].values > 0,
                                              "#ef4444", "#22c55e"),
                    name="Current 63d Corr"
                ))
                _fig160.add_trace(_go160.Scatter(
                    x=_scan_df160["Factor"],
                    y=_scan_df160["Full History Corr"],
                    mode="markers", marker=dict(color="#6366f1", size=8, symbol="diamond"),
                    name="Full History Corr"
                ))
                _fig160.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
                _fig160.update_layout(
                    title="Factor Correlation vs Forward 21d HY Change",
                    height=380, yaxis_title="Correlation",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(tickangle=-30)
                )
                st.plotly_chart(_fig160, use_container_width=True)
                st.dataframe(_scan_df160, use_container_width=True, hide_index=True)
                _top160 = _scan_df160.iloc[0]
                st.caption(
                    f"Strongest current leading indicator: **{_top160['Factor']}** "
                    f"(r={_top160['Current 63d Corr']:.2f}) → {_top160['Signal']}."
                )
            else:
                st.info("Not enough data for factor correlation scan.")
    except Exception as _e160:
        _err_track(_active_sub, _e160)
        st.caption(f"Factor correlation scan: {_e160}")


if _active_sub == 162:
    try:
        import plotly.graph_objects as _go162
        import numpy as _np162
        import pandas as _pd162
        _df162 = df.copy() if "df" in dir() else None
        _comp_col162 = "composite_risk_score_smooth"
        _sub_cols162 = [c for c in [
            "macro_risk_score_smooth", "credit_market_risk_score_smooth",
            "complacency_score_smooth", "liquidity_regime_score_smooth",
            "treasury_stress_score_smooth", "fx_commodity_score_smooth",
            "enhanced_funding_stress_score_smooth", "cross_asset_divergence_score_smooth",
            "mean_reversion_score_smooth",
        ] if _df162 is not None and c in _df162.columns]
        _has162 = _df162 is not None and _comp_col162 in _df162.columns and len(_sub_cols162) >= 2
        if not _has162:
            st.info("composite_risk_score_smooth and sub-score columns required.")
        else:
            st.subheader("Score Lead-Lag Map")
            st.caption("Cross-correlation of each sub-score vs the composite at lags from −21 to +21 trading days. A peak at negative lag = sub-score leads the composite (early warning). Peak at positive lag = sub-score lags (confirming). Diagonal = contemporaneous.")
            _comp162 = _df162[_comp_col162].dropna()
            _lags162 = list(range(-21, 22))
            _corr_matrix162 = {}
            for _sc162 in _sub_cols162:
                _row162 = []
                _s162 = _df162[_sc162].dropna()
                _j162 = _comp162.to_frame("comp").join(_s162.to_frame("sub"), how="inner").dropna()
                for _lag162 in _lags162:
                    if _lag162 < 0:
                        _shifted162 = _j162["sub"].shift(-_lag162)
                    else:
                        _shifted162 = _j162["sub"].shift(-_lag162)
                    _aligned162 = _j162["comp"].to_frame("comp").join(
                        _shifted162.to_frame("sub"), how="inner").dropna()
                    if len(_aligned162) > 30:
                        _row162.append(float(_aligned162["comp"].corr(_aligned162["sub"])))
                    else:
                        _row162.append(float("nan"))
                _corr_matrix162[_sc162] = _row162
            _heat_z162 = [_corr_matrix162[sc] for sc in _sub_cols162]
            _short_names162 = [c.replace("_score_smooth", "").replace("_risk", "").replace("_", " ").title()
                               for c in _sub_cols162]
            _fig162 = _go162.Figure(data=_go162.Heatmap(
                z=_heat_z162,
                x=[str(l) for l in _lags162],
                y=_short_names162,
                colorscale=[[0.0, "#1d4ed8"], [0.5, "#1e1b4b"], [1.0, "#7f1d1d"]],
                zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title="Corr"),
                hovertemplate="Sub-score: %{y}<br>Lag: %{x}d<br>Corr: %{z:.2f}<extra></extra>"
            ))
            _fig162.add_vline(x="0", line_color="#9aa0aa", line_dash="dash")
            _fig162.update_layout(
                title="Sub-score vs Composite Cross-Correlation (lag in trading days)",
                height=max(280, len(_sub_cols162) * 35 + 100),
                xaxis_title="Lag (negative = sub-score leads composite)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa")
            )
            st.plotly_chart(_fig162, use_container_width=True)
            # Peak lag per sub-score
            _peak_rows162 = []
            for _sc162, _row162 in _corr_matrix162.items():
                _arr162 = _np162.array(_row162, dtype=float)
                if _np162.all(_np162.isnan(_arr162)):
                    continue
                _peak_idx162 = int(_np162.nanargmax(_np162.abs(_arr162)))
                _peak_lag162 = _lags162[_peak_idx162]
                _peak_corr162 = _arr162[_peak_idx162]
                _role162 = ("Leads" if _peak_lag162 < -2 else ("Lags" if _peak_lag162 > 2 else "Contemporaneous"))
                _peak_rows162.append({
                    "Sub-score": _sc162.replace("_score_smooth", "").replace("_", " ").title(),
                    "Peak Lag (d)": _peak_lag162,
                    "Peak Corr": round(_peak_corr162, 2),
                    "Role": _role162,
                })
            if _peak_rows162:
                _peak_df162 = (_pd162.DataFrame(_peak_rows162)
                               .sort_values("Peak Lag (d)").reset_index(drop=True))
                st.dataframe(_peak_df162, use_container_width=True, hide_index=True)
                _leaders162 = [r["Sub-score"] for r in _peak_rows162 if r["Role"] == "Leads"]
                if _leaders162:
                    st.caption("Leading sub-scores: " + ", ".join(_leaders162))
    except Exception as _e162:
        _err_track(_active_sub, _e162)
        st.caption(f"Score lead-lag map: {_e162}")


if _active_sub == 165:
    try:
        import plotly.graph_objects as _go165
        import numpy as _np165
        import pandas as _pd165
        _df165 = df.copy() if "df" in dir() else None
        _has165 = _df165 is not None and "composite_risk_score_smooth" in _df165.columns
        if not _has165:
            st.info("composite_risk_score_smooth required.")
        else:
            st.subheader("Score Drawdown Profile")
            st.caption("Drawdown analysis applied to the composite risk score itself: how far does the score fall from its rolling peak, and how long does recovery take? Identifies 'score exhaustion' periods — after stress spikes, does the score recover quickly or stay elevated?")
            _comp165 = _df165["composite_risk_score_smooth"].dropna()
            _peak165 = _comp165.cummax()
            _dd165 = (_comp165 - _peak165)  # always ≤ 0
            # For score: drawdown means score fell from its high
            _fig165a = _go165.Figure()
            _fig165a.add_trace(_go165.Scatter(
                x=_comp165.index, y=_comp165.values,
                line=dict(color="#6366f1", width=1.5), name="Composite Score"
            ))
            _fig165a.add_trace(_go165.Scatter(
                x=_peak165.index, y=_peak165.values,
                line=dict(color="#ef4444", width=1, dash="dot"), name="Rolling Peak"
            ))
            _fig165a.add_hline(y=50, line_dash="dash", line_color="#9aa0aa")
            _fig165a.update_layout(
                title="Composite Score vs Rolling Peak",
                height=300, yaxis_title="Score",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(_fig165a, use_container_width=True)
            _fig165b = _go165.Figure()
            _fig165b.add_trace(_go165.Scatter(
                x=_dd165.index, y=_dd165.values,
                fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
                line=dict(color="#ef4444", width=1), name="Score Drawdown"
            ))
            _fig165b.update_layout(
                title="Score Drawdown from Peak (points below rolling high)",
                height=220, yaxis_title="Drawdown (pts)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig165b, use_container_width=True)
            # Identify drawdown episodes (where score retreated >10 pts from peak)
            _ep_rows165 = []
            _in_dd165 = False
            _ep_start165 = None
            _ep_min165 = 0.0
            for _dt165, _v165 in _dd165.items():
                if _v165 < -10:
                    if not _in_dd165:
                        _in_dd165 = True
                        _ep_start165 = _dt165
                        _ep_min165 = float(_v165)
                    else:
                        _ep_min165 = min(_ep_min165, float(_v165))
                else:
                    if _in_dd165:
                        _ep_rows165.append({
                            "Start": str(_ep_start165.date()),
                            "End": str(_dt165.date()),
                            "Max Drawdown (pts)": round(_ep_min165, 1),
                            "Duration (d)": ((_dt165 - _ep_start165).days),
                        })
                        _in_dd165 = False
            if _ep_rows165:
                _ep_df165 = _pd165.DataFrame(_ep_rows165).sort_values("Max Drawdown (pts)").reset_index(drop=True)
                st.markdown("**Score Drawdown Episodes (>10 pts from peak)**")
                st.dataframe(_ep_df165, use_container_width=True, hide_index=True)
            _cur_dd165 = float(_dd165.iloc[-1])
            _max_dd165 = float(_dd165.min())
            st.caption(
                f"Current score drawdown: **{_cur_dd165:.1f} pts** from peak. "
                f"Worst ever: {_max_dd165:.1f} pts. "
                f"Episodes >10 pts: {len(_ep_rows165)}."
            )
    except Exception as _e165:
        _err_track(_active_sub, _e165)
        st.caption(f"Score drawdown: {_e165}")


if _active_sub == 169:
    try:
        import plotly.graph_objects as _go169
        import numpy as _np169
        import pandas as _pd169
        _df169 = df.copy() if "df" in dir() else None
        _sub_cols169 = [c for c in [
            "macro_risk_score_smooth", "credit_market_risk_score_smooth",
            "complacency_score_smooth", "liquidity_regime_score_smooth",
            "treasury_stress_score_smooth", "fx_commodity_score_smooth",
            "enhanced_funding_stress_score_smooth", "cross_asset_divergence_score_smooth",
            "mean_reversion_score_smooth",
        ] if _df169 is not None and c in _df169.columns]
        _has169 = len(_sub_cols169) >= 3
        if not _has169:
            st.info("At least 3 sub-score columns required.")
        else:
            st.subheader("Score Factor Decomposition")
            st.caption("How much did each sub-score contribute to the 21-day change in the composite? Equal-weighted composite means each sub-score's 21d change contributes 1/N of the total composite change. Waterfall chart shows which channels drove the composite higher or lower.")
            _data169 = _df169[_sub_cols169].tail(63).dropna(how="all")
            _n169 = len(_sub_cols169)
            _weight169 = 1.0 / _n169
            # 21d change in each sub-score
            _chg169 = {}
            for _c169 in _sub_cols169:
                _s169 = _df169[_c169].dropna()
                if len(_s169) >= 22:
                    _chg169[_c169] = float(_s169.iloc[-1]) - float(_s169.iloc[-22])
                else:
                    _chg169[_c169] = float("nan")
            _contrib169 = {k: v * _weight169 for k, v in _chg169.items() if not _np169.isnan(v)}
            _total169 = sum(_contrib169.values())
            _labels169 = [k.replace("_score_smooth", "").replace("_risk", "").replace("_", " ").title()
                          for k in _contrib169]
            _vals169 = list(_contrib169.values())
            # Sort by absolute contribution
            _order169 = sorted(range(len(_vals169)), key=lambda i: abs(_vals169[i]), reverse=True)
            _labels169 = [_labels169[i] for i in _order169]
            _vals169 = [_vals169[i] for i in _order169]
            # Waterfall
            _running169 = 0.0
            _bases169 = []
            for v in _vals169:
                _bases169.append(_running169)
                _running169 += v
            _colors169 = ["#22c55e" if v < 0 else "#ef4444" for v in _vals169]
            _fig169 = _go169.Figure()
            _fig169.add_trace(_go169.Bar(
                x=_labels169 + ["Total Δ"],
                y=_vals169 + [_total169],
                base=_bases169 + [0],
                marker_color=_colors169 + ["#6366f1"],
                text=[f"{v:+.1f}" for v in _vals169] + [f"{_total169:+.1f}"],
                textposition="outside"
            ))
            _fig169.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig169.update_layout(
                title="21-Day Sub-Score Contribution to Composite Change",
                height=380, yaxis_title="Points",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False, xaxis=dict(tickangle=-25)
            )
            st.plotly_chart(_fig169, use_container_width=True)
            # Summary table
            _tbl169 = _pd169.DataFrame({
                "Sub-score": _labels169,
                "21d Raw Change": [round(_chg169[list(_contrib169.keys())[_order169[i]]], 1) for i in range(len(_labels169))],
                "Contribution (pts)": [round(v, 2) for v in _vals169],
            })
            st.dataframe(_tbl169, use_container_width=True, hide_index=True)
            _top_driver169 = _labels169[0] if _labels169 else "—"
            st.caption(
                f"Composite 21d change: **{_total169:+.1f} pts** · "
                f"Top driver: **{_top_driver169}** ({_vals169[0]:+.1f} pts). "
                f"Equal weight {_weight169:.1%} per sub-score."
            )
    except Exception as _e169:
        _err_track(_active_sub, _e169)
        st.caption(f"Factor decomp: {_e169}")


if _active_sub == 173:
    try:
        import plotly.graph_objects as _go173
        import numpy as _np173
        import pandas as _pd173
        _df173 = df.copy() if "df" in dir() else None
        _has173 = (_df173 is not None
                   and "composite_risk_score_smooth" in _df173.columns
                   and "hy_spread" in _df173.columns)
        if not _has173:
            st.info("composite_risk_score_smooth and hy_spread required.")
        else:
            st.subheader("Score Inflection Point Detector")
            st.caption("Second derivative of the composite score: when it crosses zero from negative to positive, the score's rate-of-change is accelerating upward (stress building). Crossing from positive to negative = deceleration (potential peak). These inflection points historically precede credit spread regime shifts by 2–4 weeks.")
            _comp173 = _df173["composite_risk_score_smooth"].dropna()
            _d1_173 = _comp173.diff(5).rolling(3).mean()   # smoothed 1st deriv
            _d2_173 = _d1_173.diff(5).rolling(3).mean()    # smoothed 2nd deriv
            # Detect zero crossings in d2
            _sign173 = _np173.sign(_d2_173.dropna().values)
            _crossings173 = []
            _d2_idx173 = _d2_173.dropna().index
            for _i173 in range(1, len(_sign173)):
                if _sign173[_i173 - 1] < 0 and _sign173[_i173] >= 0:
                    _crossings173.append((_d2_idx173[_i173], "Acceleration"))
                elif _sign173[_i173 - 1] > 0 and _sign173[_i173] <= 0:
                    _crossings173.append((_d2_idx173[_i173], "Deceleration"))
            _fig173 = _go173.Figure()
            _fig173.add_trace(_go173.Scatter(
                x=_comp173.index, y=_comp173.values,
                line=dict(color="#6366f1", width=2), name="Composite Score"
            ))
            for _dt173, _typ173 in _crossings173[-30:]:
                _fig173.add_vline(
                    x=_dt173, line_dash="dot",
                    line_color="#ef4444" if _typ173 == "Acceleration" else "#22c55e",
                    annotation_text=_typ173[0], annotation_position="top"
                )
            _fig173.add_hline(y=50, line_dash="dash", line_color="#9aa0aa")
            _fig173.update_layout(
                title="Composite Score with Inflection Points (last 30 crossings)",
                height=340, yaxis_title="Score (0–100)", yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig173, use_container_width=True)
            # 2nd derivative chart
            _fig173b = _go173.Figure()
            _fig173b.add_trace(_go173.Scatter(
                x=_d2_173.index, y=_d2_173.values,
                fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                line=dict(color="#6366f1", width=1), name="2nd Derivative"
            ))
            _fig173b.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
            _fig173b.update_layout(
                title="Score Second Derivative (positive = accelerating upward)",
                height=200, yaxis_title="d²Score/dt²",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig173b, use_container_width=True)
            # Forward HY outcomes at acceleration crossings
            _hy173 = _df173["hy_spread"]
            _accel_outcomes173 = []
            for _dt173, _typ173 in _crossings173:
                if _typ173 != "Acceleration":
                    continue
                try:
                    _loc173 = _hy173.index.get_loc(_dt173)
                    if _loc173 + 21 < len(_hy173):
                        _fwd173 = float(_hy173.iloc[_loc173 + 21]) - float(_hy173.iloc[_loc173])
                        _accel_outcomes173.append(_fwd173)
                except Exception:
                    pass
            _cur_d2173 = float(_d2_173.iloc[-1]) if _d2_173.notna().any() else float("nan")
            _last_type173 = _crossings173[-1][1] if _crossings173 else "Unknown"
            _last_dt173 = str(_crossings173[-1][0].date()) if _crossings173 else "—"
            if _accel_outcomes173:
                _med_fwd173 = float(_np173.median(_accel_outcomes173))
                st.caption(
                    f"Current 2nd derivative: **{_cur_d2173:.2f}** · Last inflection: **{_last_type173}** on {_last_dt173}. "
                    f"After acceleration crossings: median 21d HY Δ = **{_med_fwd173:+.0f} bps** "
                    f"(n={len(_accel_outcomes173)})."
                )
            else:
                st.caption(f"Current 2nd derivative: **{_cur_d2173:.2f}** · Last inflection: {_last_type173} ({_last_dt173}).")
    except Exception as _e173:
        _err_track(_active_sub, _e173)
        st.caption(f"Score inflection: {_e173}")


if _active_sub == "ov_sl":
    try:
        import plotly.graph_objects as _go_ov_sl
        import numpy as _np_ov_sl
        st.subheader("Signal Lab — Section Overview")
        st.caption("Composite score health, sub-score breakdown, and signal diagnostics at a glance. Select any sub-view from the sidebar.")
        _d = df
        _score_cols_sl = [c for c in [
            "macro_risk_score_smooth", "credit_market_risk_score_smooth",
            "complacency_score_smooth", "liquidity_regime_score_smooth",
            "treasury_stress_score_smooth", "fx_commodity_score_smooth",
            "enhanced_funding_stress_score_smooth", "cross_asset_divergence_score_smooth",
            "mean_reversion_score_smooth",
        ] if c in _d.columns]
        _comp_sl = float(_d["composite_risk_score_smooth"].dropna().iloc[-1]) if "composite_risk_score_smooth" in _d.columns else float("nan")
        _comp_s = _d["composite_risk_score_smooth"].dropna() if "composite_risk_score_smooth" in _d.columns else None
        _comp_chg = (float(_comp_s.iloc[-1]) - float(_comp_s.iloc[-22])) if _comp_s is not None and len(_comp_s) >= 22 else float("nan")
        _comp_pct = (float((_comp_s < _comp_sl).mean() * 100)) if _comp_s is not None else float("nan")
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("Composite Score", f"{_comp_sl:.1f}" if not _np_ov_sl.isnan(_comp_sl) else "—",
                   delta=f"{_comp_chg:+.1f} 21d" if not _np_ov_sl.isnan(_comp_chg) else None,
                   delta_color="inverse")
        _c2.metric("Score Percentile", f"{_comp_pct:.0f}th" if not _np_ov_sl.isnan(_comp_pct) else "—",
                   delta_color="off")
        _regime_sl = ("High Stress" if _comp_sl >= 70 else ("Elevated" if _comp_sl >= 50
                       else ("Moderate" if _comp_sl >= 30 else "Low Stress")))
        _c3.metric("Regime", _regime_sl)
        _n_above50 = sum(1 for c in _score_cols_sl
                         if len(_d[c].dropna()) and float(_d[c].dropna().iloc[-1]) >= 50)
        _c4.metric("Sub-scores ≥50", f"{_n_above50}/{len(_score_cols_sl)}")
        st.divider()
        # Sub-score bar chart (current values)
        if _score_cols_sl:
            _vals_sl = [float(_d[c].dropna().iloc[-1]) if len(_d[c].dropna()) else float("nan")
                        for c in _score_cols_sl]
            _labels_sl = [c.replace("_score_smooth", "").replace("_risk", "").replace("_", " ").title()
                          for c in _score_cols_sl]
            _fig_ov_sl = _go_ov_sl.Figure()
            _fig_ov_sl.add_trace(_go_ov_sl.Bar(
                x=_labels_sl, y=_vals_sl,
                marker_color=["#ef4444" if v >= 70 else "#f59e0b" if v >= 50 else "#22c55e"
                              for v in _vals_sl],
                text=[f"{v:.0f}" for v in _vals_sl], textposition="outside"
            ))
            _fig_ov_sl.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="Stress threshold")
            _fig_ov_sl.add_hline(y=70, line_dash="dash", line_color="#ef4444")
            _fig_ov_sl.update_layout(
                title="Current Sub-Score Readings",
                height=300, yaxis_title="Score", yaxis=dict(range=[0, 105]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False, xaxis=dict(tickangle=-25), margin=dict(t=40, b=20))
            st.plotly_chart(_fig_ov_sl, use_container_width=True)
        st.info(f"Composite at **{_comp_sl:.1f}** ({_regime_sl}). "
                f"35 sub-views: validation, PCA, bootstrap CI, lead-lag map, factor decomp, and more.")
    except Exception as _e_ov_sl:
        _err_track(_active_sub, _e_ov_sl)
        st.caption(f"Signal Lab overview: {_e_ov_sl}")


if _active_sub == 184:
    st.subheader("Forward Return Calibration")
    st.caption("Pre-computed HY spread forward changes binned by composite score — true signal validation using realized outcomes")
    try:
        import plotly.graph_objects as _go184
        import numpy as _np184
        import pandas as _pd184
        _score_col184 = next((c for c in ["composite_score_smooth","credit_risk_score_smooth","hy_spread_score_smooth"] if c in df.columns), None)
        if _score_col184 is None:
            st.warning("No composite score column found.")
        else:
            _df184 = df[[_score_col184, "hy_spread_forward_21d_change", "hy_spread_forward_63d_change", "hy_spread_forward_126d_change"]].dropna().copy()
            _df184["score_decile"] = _pd184.qcut(_df184[_score_col184], q=10, labels=[f"D{i}" for i in range(1, 11)])
            _last_score184 = float(df[_score_col184].dropna().iloc[-1])
            _score_pct184 = float((_df184[_score_col184] < _last_score184).mean() * 100)
            _c1_184, _c2_184, _c3_184 = st.columns(3)
            _c1_184.metric("Current Score", f"{_last_score184:.2f}")
            _c2_184.metric("Percentile", f"{_score_pct184:.0f}th")
            _c3_184.metric("Total Obs", f"{len(_df184):,}")
            st.divider()
            # Mean forward return by decile for each horizon
            _horizons184 = [("21d", "hy_spread_forward_21d_change"), ("63d", "hy_spread_forward_63d_change"), ("126d", "hy_spread_forward_126d_change")]
            _decile_stats184 = {}
            for _h184, _col184 in _horizons184:
                _stats184 = _df184.groupby("score_decile")[_col184].agg(["mean","std","count"]).reset_index()
                _stats184.columns = ["Decile", "Mean Fwd Chg", "Std", "N"]
                _decile_stats184[_h184] = _stats184
            # 3-panel bar charts
            _fig184a = _go184.Figure()
            for _h184, _col184 in _horizons184:
                _s184 = _decile_stats184[_h184]
                _fig184a.add_trace(_go184.Bar(
                    x=_s184["Decile"].astype(str),
                    y=_s184["Mean Fwd Chg"],
                    name=f"{_h184} mean",
                    marker_color=[("#ef4444" if v > 0 else "#3b82f6") for v in _s184["Mean Fwd Chg"]],
                    visible=(_h184 == "21d")
                ))
            # Buttons to switch horizon
            _fig184a.update_layout(
                title="Mean HY Spread Forward Change by Score Decile (D1=lowest score, D10=highest)",
                height=300,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                yaxis_title="Mean Forward Spread Change (bps)",
                xaxis_title="Score Decile",
                updatemenus=[dict(
                    type="buttons", direction="right", x=0.5, y=1.15, xanchor="center",
                    buttons=[dict(label=_h184, method="update",
                                  args=[{"visible": [_h184 == h2 for h2, _ in _horizons184]}])
                             for _h184, _ in _horizons184]
                )],
                margin=dict(t=60, b=40),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
            )
            st.plotly_chart(_fig184a, use_container_width=True)
            # Cumulative mean forward return: top vs bottom quartile
            _df184["score_quartile"] = _pd184.qcut(_df184[_score_col184], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
            _c1b_184, _c2b_184 = st.columns(2)
            with _c1b_184:
                st.markdown("**21d Forward Returns by Quartile**")
                _q21_184 = _df184.groupby("score_quartile")["hy_spread_forward_21d_change"].agg(["mean","std","count"])
                _q21_184.columns = ["Mean (bps)", "Std", "N"]
                st.dataframe(_q21_184.round(1), use_container_width=True)
            with _c2b_184:
                st.markdown("**63d Forward Returns by Quartile**")
                _q63_184 = _df184.groupby("score_quartile")["hy_spread_forward_63d_change"].agg(["mean","std","count"])
                _q63_184.columns = ["Mean (bps)", "Std", "N"]
                st.dataframe(_q63_184.round(1), use_container_width=True)
            # Score vs forward return scatter (63d)
            _fig184b = _go184.Figure()
            _fig184b.add_trace(_go184.Scatter(
                x=_df184[_score_col184], y=_df184["hy_spread_forward_63d_change"],
                mode="markers", marker=dict(size=2, color="#3b82f6", opacity=0.3),
                name="Obs"
            ))
            # Running mean
            _df184_sorted184 = _df184.sort_values(_score_col184)
            _running_mean184 = _df184_sorted184["hy_spread_forward_63d_change"].rolling(200, center=True).mean()
            _fig184b.add_trace(_go184.Scatter(
                x=_df184_sorted184[_score_col184], y=_running_mean184,
                mode="lines", line=dict(color="#f59e0b", width=2), name="200-obs rolling mean"
            ))
            _fig184b.add_vline(x=_last_score184, line_dash="dash", line_color="#10b981",
                               annotation_text=f"Current: {_last_score184:.2f}")
            _fig184b.update_layout(
                title=f"Score vs 63d Forward HY Change (current={_last_score184:.2f}, {_score_pct184:.0f}th pct)",
                height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                xaxis_title="Score", yaxis_title="63d HY Spread Change (bps)",
                margin=dict(t=40, b=30),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
            )
            st.plotly_chart(_fig184b, use_container_width=True)
            # Interpret current score
            _cur_decile184 = int(_score_pct184 // 10) + 1
            _cur_decile184 = min(_cur_decile184, 10)
            _dec_label184 = f"D{_cur_decile184}"
            _fwd21_184 = _decile_stats184["21d"][_decile_stats184["21d"]["Decile"].astype(str) == _dec_label184]["Mean Fwd Chg"]
            _fwd63_184 = _decile_stats184["63d"][_decile_stats184["63d"]["Decile"].astype(str) == _dec_label184]["Mean Fwd Chg"]
            _fwd21_val184 = float(_fwd21_184.iloc[0]) if len(_fwd21_184) > 0 else float("nan")
            _fwd63_val184 = float(_fwd63_184.iloc[0]) if len(_fwd63_184) > 0 else float("nan")
            st.caption(
                f"Score at {_score_pct184:.0f}th percentile ({_dec_label184}). "
                f"Historical mean 21d forward HY change at this decile: {_fwd21_val184:+.1f} bps; "
                f"63d: {_fwd63_val184:+.1f} bps. "
                f"Using {_score_col184}.")
    except Exception as _e184:
        _err_track(_active_sub, _e184)
        st.caption(f"Forward return calibration: {_e184}")

# --- sub185: Default Probability Monitor ---
