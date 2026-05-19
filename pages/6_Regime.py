"""
Regime — analytics section page.
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
    page_title='Regime — Credit Risk Dashboard',
    page_icon='🔄',
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
_SECTION_NAME = 'Regime'
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


if _active_sub == 8:
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


if _active_sub == 10:
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

if _active_sub == 11:
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

if _active_sub == 13:
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

if _active_sub == 14:
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

if _active_sub == 25:
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

if _active_sub == 32:
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

if _active_sub == 33:
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

if _active_sub == 34:
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

if _active_sub == 36:
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

if _active_sub == 43:
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

if _active_sub == 49:
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

if _active_sub == 131:
    try:
        import plotly.graph_objects as _go131
        import pandas as _pd131
        if "composite_risk_score_smooth" in df.columns and "hy_spread" in df.columns:
            _comp131 = df["composite_risk_score_smooth"].dropna()
            _hy131 = df["hy_spread"].dropna()
            # Define approximate recession-onset anchor dates
            _analogs131 = {
                "2001 (Tech bust)": "2001-03-01",
                "2008 (GFC)":       "2008-09-01",
                "2020 (COVID)":     "2020-02-15",
            }
            _window131 = 252  # ±1yr around onset
            _fig131a = _go131.Figure()
            _colors131 = {"2001 (Tech bust)": "#f59e0b", "2008 (GFC)": "#ef4444", "2020 (COVID)": "#10b981"}
            for name, dt_str in _analogs131.items():
                try:
                    _anchor = _pd131.Timestamp(dt_str)
                    _mask = (_comp131.index >= _anchor - _pd131.Timedelta(days=_window131)) & \
                            (_comp131.index <= _anchor + _pd131.Timedelta(days=_window131))
                    _slice = _comp131[_mask]
                    if len(_slice) > 10:
                        _x_rel = [(_d - _anchor).days for _d in _slice.index]
                        _fig131a.add_trace(_go131.Scatter(
                            x=_x_rel, y=_slice.values,
                            mode="lines", name=name,
                            line=dict(color=_colors131.get(name, "#9aa0aa"), width=1.5),
                            hovertemplate=f"{name}<br>Day %{{x}}: %{{y:.0f}}<extra></extra>",
                        ))
                except Exception:
                    pass
            # Current trajectory (last 252 days)
            _now131 = _comp131.index[-1]
            _curr_mask = _comp131.index >= (_now131 - _pd131.Timedelta(days=_window131))
            _curr_slice = _comp131[_curr_mask]
            if len(_curr_slice) > 10:
                _x_curr = [(_d - _now131).days for _d in _curr_slice.index]
                _fig131a.add_trace(_go131.Scatter(
                    x=_x_curr, y=_curr_slice.values,
                    mode="lines", name="Current",
                    line=dict(color="#ffffff", width=2.0, dash="dot"),
                    hovertemplate="Current<br>Day %{x}: %{y:.0f}<extra></extra>",
                ))
            _fig131a.add_vline(x=0, line_color="#6b7280", line_width=1, line_dash="dash",
                               annotation_text="Onset", annotation_position="top left",
                               annotation_font=dict(color="#6b7280", size=9))
            _fig131a.update_layout(
                height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                title=dict(text="Composite Score: Recession-Onset Analogs (±1yr window)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Days relative to recession onset"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Composite Score", range=[0, 100]),
                legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig131a, use_container_width=True)
            # Same for HY spread
            _fig131b = _go131.Figure()
            for name, dt_str in _analogs131.items():
                try:
                    _anchor = _pd131.Timestamp(dt_str)
                    _mask = (_hy131.index >= _anchor - _pd131.Timedelta(days=_window131)) & \
                            (_hy131.index <= _anchor + _pd131.Timedelta(days=_window131))
                    _slice = _hy131[_mask]
                    if len(_slice) > 10:
                        # Index to 100 at onset
                        _anchor_val = _slice[_slice.index >= _anchor].iloc[0] if len(_slice[_slice.index >= _anchor]) else _np130.nan
                        _x_rel = [(_d - _anchor).days for _d in _slice.index]
                        _fig131b.add_trace(_go131.Scatter(
                            x=_x_rel, y=_slice.values,
                            mode="lines", name=name,
                            line=dict(color=_colors131.get(name, "#9aa0aa"), width=1.5),
                            hovertemplate=f"{name}<br>Day %{{x}}: %{{y:.0f}}bps<extra></extra>",
                        ))
                except Exception:
                    pass
            # Current HY
            _curr_hy = _hy131[_hy131.index >= (_now131 - _pd131.Timedelta(days=_window131))]
            if len(_curr_hy) > 10:
                _x_hy_curr = [(_d - _now131).days for _d in _curr_hy.index]
                _fig131b.add_trace(_go131.Scatter(
                    x=_x_hy_curr, y=_curr_hy.values,
                    mode="lines", name="Current",
                    line=dict(color="#ffffff", width=2.0, dash="dot"),
                    hovertemplate="Current<br>Day %{x}: %{y:.0f}bps<extra></extra>",
                ))
            _fig131b.add_vline(x=0, line_color="#6b7280", line_width=1, line_dash="dash")
            _fig131b.update_layout(
                height=240, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=30, b=8),
                title=dict(text="HY Spread: Recession-Onset Analogs (±1yr window)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="Days relative to recession onset"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280",
                           title="HY Spread (bps)"),
                legend=dict(orientation="h", y=-0.35, x=0, font=dict(size=9)),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig131b, use_container_width=True)
            st.caption("Anchor = recession/crisis onset date. White dot = current trajectory. Compare shape and pace vs prior episodes.")
        else:
            st.info("Composite score or HY spread not found — run the full scoring pipeline.")
    except Exception as _e131:
        _err_track(_active_sub, _e131)
        st.caption(f"Recession analog: {_e131}")

# sub132 — Score Persistence (tab_siglab)

if _active_sub == 134:
    try:
        import plotly.graph_objects as _go134
        import numpy as _np134
        if "composite_risk_score_smooth" in df.columns:
            _comp134 = df["composite_risk_score_smooth"].dropna()
            # Define 5 composite risk regimes
            def _comp_regime_134(score):
                if score < 25:   return "Low"
                if score < 40:   return "Moderate"
                if score < 55:   return "Elevated"
                if score < 70:   return "High"
                return "Extreme"
            _REGIMES134 = ["Low", "Moderate", "Elevated", "High", "Extreme"]
            _regime_series = _comp134.apply(_comp_regime_134)
            # Build transition count matrix
            _trans134 = {r: {r2: 0 for r2 in _REGIMES134} for r in _REGIMES134}
            _prev = None
            for val in _regime_series:
                if _prev is not None:
                    _trans134[_prev][val] += 1
                _prev = val
            # Convert to probability matrix (row = from, col = to)
            _mat134 = []
            for _from in _REGIMES134:
                _row_total = sum(_trans134[_from].values())
                if _row_total > 0:
                    _mat134.append([_trans134[_from][_to] / _row_total * 100 for _to in _REGIMES134])
                else:
                    _mat134.append([0.0] * len(_REGIMES134))
            _fig134a = _go134.Figure(data=_go134.Heatmap(
                z=_mat134,
                x=_REGIMES134,
                y=_REGIMES134,
                colorscale=[[0.0, "#1a1f2e"], [0.5, "#1e40af"], [1.0, "#dc2626"]],
                zmin=0, zmax=100,
                text=[[f"{v:.0f}%" for v in row] for row in _mat134],
                texttemplate="%{text}",
                hovertemplate="From %{y} → %{x}: %{z:.1f}%<extra></extra>",
            ))
            _fig134a.update_layout(
                height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=50, b=8),
                title=dict(text="Composite Risk Regime Transition Probability Matrix (row=from, col=to)", font=dict(size=12, color="#9aa0aa")),
                xaxis=dict(color="#6b7280", title="Transition TO"),
                yaxis=dict(color="#6b7280", title="Transition FROM", autorange="reversed"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
            )
            st.plotly_chart(_fig134a, use_container_width=True)
            # Current regime and likely next
            _current_regime134 = _regime_series.iloc[-1]
            _next_probs = _trans134[_current_regime134]
            _total_next = sum(_next_probs.values())
            if _total_next > 0:
                _next_sorted = sorted(_next_probs.items(), key=lambda x: x[1], reverse=True)
                _fig134b = _go134.Figure()
                _fig134b.add_trace(_go134.Bar(
                    x=[r for r, _ in _next_sorted],
                    y=[c / _total_next * 100 for _, c in _next_sorted],
                    marker_color=["#ef4444" if r in ("High","Extreme") else ("#f59e0b" if r == "Elevated" else "#3b82f6") for r, _ in _next_sorted],
                    text=[f"{c / _total_next * 100:.0f}%" for _, c in _next_sorted],
                    textposition="auto",
                    hovertemplate="%{x}: %{y:.0f}%<extra></extra>",
                ))
                _fig134b.update_layout(
                    height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11), margin=dict(l=8, r=8, t=40, b=8),
                    title=dict(text=f"Next-Period Regime Probabilities (current: '{_current_regime134}')", font=dict(size=12, color="#9aa0aa")),
                    xaxis=dict(color="#6b7280"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#6b7280", title="Probability (%)"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(_fig134b, use_container_width=True)
            st.caption("Empirical transition probabilities from daily composite risk score history. Diagonal dominance = regime persistence (stickiness).")
        else:
            st.info("composite_risk_score_smooth not found — run the full scoring pipeline.")
    except Exception as _e134:
        _err_track(_active_sub, _e134)
        st.caption(f"Transition matrix: {_e134}")

# sub135 — Score Distributions (tab_siglab)

if _active_sub == 155:
    try:
        import plotly.graph_objects as _go155
        import numpy as _np155
        import pandas as _pd155
        _df155 = df.copy() if "df" in dir() else None
        _has155 = _df155 is not None and "composite_risk_score_smooth" in _df155.columns
        if not _has155:
            st.info("composite_risk_score_smooth required.")
        else:
            st.subheader("Regime Dwell Time Analysis")
            st.caption("How long does the composite score stay in each regime? Dwell-time distributions reveal regime stickiness. Knowing the typical episode length and historical exit rates by dwell-day sets probabilistic expectations for current conditions.")
            _comp155 = _df155["composite_risk_score_smooth"].dropna()
            def _reg155(s):
                if s < 30: return "Low Stress"
                elif s < 50: return "Moderate"
                elif s < 70: return "Elevated"
                else: return "High Stress"
            _regimes155 = _comp155.apply(_reg155)
            _runs155 = []
            _cur_r155 = None
            _run_l155 = 0
            for _r155 in _regimes155:
                if _r155 == _cur_r155:
                    _run_l155 += 1
                else:
                    if _cur_r155 is not None:
                        _runs155.append({"regime": _cur_r155, "days": _run_l155})
                    _cur_r155 = _r155
                    _run_l155 = 1
            if _cur_r155:
                _runs155.append({"regime": _cur_r155, "days": _run_l155})
            _runs_df155 = _pd155.DataFrame(_runs155)
            _fig155 = _go155.Figure()
            _rorder155 = ["Low Stress", "Moderate", "Elevated", "High Stress"]
            _rcols155 = {"Low Stress": "#22c55e", "Moderate": "#6366f1",
                         "Elevated": "#f59e0b", "High Stress": "#ef4444"}
            for _r155b in _rorder155:
                _rs155 = _runs_df155[_runs_df155["regime"] == _r155b]["days"]
                if len(_rs155) > 2:
                    _fig155.add_trace(_go155.Box(
                        y=_rs155.values, name=_r155b,
                        marker_color=_rcols155[_r155b], line_color=_rcols155[_r155b]
                    ))
            _fig155.update_layout(
                title="Regime Dwell Time Distribution (trading days)",
                height=350, yaxis_title="Days in Regime",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False
            )
            st.plotly_chart(_fig155, use_container_width=True)
            _sum155 = (_runs_df155.groupby("regime")["days"]
                       .agg(Count="count", Median="median", Mean="mean",
                            P90=lambda x: x.quantile(0.9))
                       .round(1).reset_index()
                       .rename(columns={"regime": "Regime"}))
            st.dataframe(_sum155, use_container_width=True, hide_index=True)
            _cur_dwell155 = int(_runs155[-1]["days"]) if _runs155 else 0
            _cur_rname155 = _runs155[-1]["regime"] if _runs155 else "Unknown"
            _rr155 = _runs_df155[_runs_df155["regime"] == _cur_rname155]["days"]
            _pct_surv155 = float((_rr155 > _cur_dwell155).mean() * 100) if len(_rr155) > 0 else float("nan")
            st.caption(
                f"Current regime: **{_cur_rname155}** · Dwell: **{_cur_dwell155} days** · "
                f"{_pct_surv155:.0f}% of historical {_cur_rname155} episodes lasted longer."
            )
    except Exception as _e155:
        _err_track(_active_sub, _e155)
        st.caption(f"Regime dwell time: {_e155}")


if _active_sub == 166:
    try:
        import pandas as _pd166
        _df166 = df.copy() if "df" in dir() else None
        if _df166 is None:
            st.info("Dataset required.")
        else:
            st.subheader("Macro Traffic Light Scorecard")
            st.caption("Multi-dimensional scorecard summarizing current conditions across six macro-financial channels. Each row evaluates one channel against its historical distribution and assigns a risk status.")

            def _safe166(col, default=float("nan")):
                if col in _df166.columns and _df166[col].notna().any():
                    return float(_df166[col].dropna().iloc[-1])
                return default

            def _pct166(col):
                if col not in _df166.columns:
                    return float("nan")
                s = _df166[col].dropna()
                if len(s) < 10:
                    return float("nan")
                v = float(s.iloc[-1])
                return float((s < v).mean() * 100)

            def _status166(pct, invert=False):
                if _pd166.isna(pct):
                    return "—"
                p = 100 - pct if invert else pct
                if p >= 75: return "🔴 High Risk"
                if p >= 50: return "🟡 Elevated"
                if p >= 25: return "🟢 Moderate"
                return "🟢 Low Risk"

            _rows166 = []

            # Growth
            _sahm166 = _safe166("sahm_like")
            _u_pct166 = _pct166("unemployment_change_90d")
            _grow_status166 = (_status166(_u_pct166) if not _pd166.isna(_u_pct166) else "—")
            _rows166.append({"Channel": "Growth / Labor",
                             "Key Metric": f"Sahm-like: {_sahm166:.2f}" if not _pd166.isna(_sahm166) else "—",
                             "Percentile": f"{_u_pct166:.0f}" if not _pd166.isna(_u_pct166) else "—",
                             "Status": _grow_status166,
                             "Note": "Sahm ≥0.5 = recession risk elevated"})

            # Inflation / Rates
            _be166 = _safe166("breakeven_10y")
            _ry166 = _safe166("real_yield_proxy")
            _ry_pct166 = _pct166("real_yield_z")
            _rows166.append({"Channel": "Inflation / Real Rates",
                             "Key Metric": f"Breakeven: {_be166:.1f}% | Real: {_ry166:.2f}" if not _pd166.isna(_be166) else "—",
                             "Percentile": f"{_ry_pct166:.0f}" if not _pd166.isna(_ry_pct166) else "—",
                             "Status": _status166(_ry_pct166),
                             "Note": "High real yields stress debt servicing"})

            # Financial Conditions
            _nfci166 = _safe166("nfci")
            _nfci_pct166 = _pct166("nfci")
            _rows166.append({"Channel": "Financial Conditions",
                             "Key Metric": f"NFCI: {_nfci166:.3f}" if not _pd166.isna(_nfci166) else "—",
                             "Percentile": f"{_nfci_pct166:.0f}" if not _pd166.isna(_nfci_pct166) else "—",
                             "Status": _status166(_nfci_pct166),
                             "Note": "NFCI > 0 = tighter than average"})

            # Credit
            _hy166 = _safe166("hy_spread")
            _hy_pct166 = _pct166("hy_spread")
            _rows166.append({"Channel": "Credit Spreads",
                             "Key Metric": f"HY OAS: {_hy166:.0f} bps" if not _pd166.isna(_hy166) else "—",
                             "Percentile": f"{_hy_pct166:.0f}" if not _pd166.isna(_hy_pct166) else "—",
                             "Status": _status166(_hy_pct166),
                             "Note": "Wide spreads = credit stress"})

            # Volatility
            _vix166 = _safe166("vix")
            _vix_pct166 = _pct166("vix")
            _rows166.append({"Channel": "Equity Volatility",
                             "Key Metric": f"VIX: {_vix166:.1f}" if not _pd166.isna(_vix166) else "—",
                             "Percentile": f"{_vix_pct166:.0f}" if not _pd166.isna(_vix_pct166) else "—",
                             "Status": _status166(_vix_pct166),
                             "Note": "VIX > 25 historically = stress regime"})

            # Yield Curve
            _curve166 = _safe166("spread")
            _curve_pct166 = _pct166("spread")
            _rows166.append({"Channel": "Yield Curve (2s10s)",
                             "Key Metric": f"Spread: {_curve166:.2f}%" if not _pd166.isna(_curve166) else "—",
                             "Percentile": f"{_curve_pct166:.0f}" if not _pd166.isna(_curve_pct166) else "—",
                             "Status": _status166(100 - _curve_pct166 if not _pd166.isna(_curve_pct166) else float("nan")),
                             "Note": "Inverted curve (low pct) = recession signal"})

            _sc166 = _pd166.DataFrame(_rows166)
            st.dataframe(_sc166, use_container_width=True, hide_index=True)

            _red_count166 = sum("🔴" in str(r) for r in _sc166["Status"])
            _yellow_count166 = sum("🟡" in str(r) for r in _sc166["Status"])
            st.caption(
                f"Scorecard: **{_red_count166}** high-risk · **{_yellow_count166}** elevated · "
                f"{6 - _red_count166 - _yellow_count166} moderate/low. "
                "Percentiles computed vs full available history."
            )
    except Exception as _e166:
        _err_track(_active_sub, _e166)
        st.caption(f"Macro scorecard: {_e166}")


if _active_sub == 168:
    try:
        import plotly.graph_objects as _go168
        import numpy as _np168
        import pandas as _pd168
        _df168 = df.copy() if "df" in dir() else None
        _feat_cols168 = [c for c in [
            "vix", "hy_spread", "spread", "unemployment", "nfci",
            "sp500_drawdown", "real_yield_proxy", "breakeven_10y",
            "hy_change_30d", "sahm_like",
        ] if _df168 is not None and c in _df168.columns]
        _has168 = _df168 is not None and len(_feat_cols168) >= 4 and "hy_spread" in _df168.columns
        if not _has168:
            st.info("At least 4 feature columns including hy_spread required.")
        else:
            st.subheader("Quantitative Regime Analogs")
            st.caption("Euclidean distance from today's standardized feature vector to every historical observation. Top similar dates = quantitative analogs. Forward outcomes (HY spread change at +21, +63d) reveal what historically followed the most similar macro-financial configurations.")
            _feat_df168 = _df168[_feat_cols168].dropna()
            # Standardize
            _mu168 = _feat_df168.mean()
            _sd168 = _feat_df168.std().replace(0, 1)
            _std_df168 = (_feat_df168 - _mu168) / _sd168
            _cur_vec168 = _std_df168.iloc[-1].values
            # Distances to all historical rows (exclude last 63 days to avoid self-match)
            _hist_std168 = _std_df168.iloc[:-63]
            _dists168 = _np168.sqrt((((_hist_std168.values - _cur_vec168) ** 2)).sum(axis=1))
            _dist_s168 = _pd168.Series(_dists168, index=_hist_std168.index)
            _top168 = _dist_s168.nsmallest(10)
            # Forward HY outcomes
            _hy168 = _df168["hy_spread"]
            _analog_rows168 = []
            for _dt168, _d168 in _top168.items():
                _fwd21168 = float("nan")
                _fwd63168 = float("nan")
                try:
                    _loc168 = _hy168.index.get_loc(_dt168)
                    if _loc168 + 21 < len(_hy168):
                        _fwd21168 = float(_hy168.iloc[_loc168 + 21]) - float(_hy168.iloc[_loc168])
                    if _loc168 + 63 < len(_hy168):
                        _fwd63168 = float(_hy168.iloc[_loc168 + 63]) - float(_hy168.iloc[_loc168])
                except Exception:
                    pass
                _analog_rows168.append({
                    "Analog Date": str(_dt168.date()),
                    "Distance": round(float(_d168), 2),
                    "HY at Analog": round(float(_hy168.get(_dt168, float("nan"))), 0),
                    "Fwd 21d HY Δ": round(_fwd21168, 0) if not _np168.isnan(_fwd21168) else None,
                    "Fwd 63d HY Δ": round(_fwd63168, 0) if not _np168.isnan(_fwd63168) else None,
                })
            _analog_df168 = _pd168.DataFrame(_analog_rows168)
            st.dataframe(_analog_df168, use_container_width=True, hide_index=True)
            # Summary of forward outcomes
            _fwd21_vals168 = [r["Fwd 21d HY Δ"] for r in _analog_rows168 if r["Fwd 21d HY Δ"] is not None]
            _fwd63_vals168 = [r["Fwd 63d HY Δ"] for r in _analog_rows168 if r["Fwd 63d HY Δ"] is not None]
            # Scatter of analog forward outcomes
            if _fwd21_vals168 and _fwd63_vals168:
                _fig168 = _go168.Figure()
                _fig168.add_trace(_go168.Scatter(
                    x=_fwd21_vals168, y=_fwd63_vals168[:len(_fwd21_vals168)],
                    mode="markers+text",
                    text=[r["Analog Date"][:7] for r in _analog_rows168[:len(_fwd21_vals168)]],
                    textposition="top center",
                    marker=dict(color="#6366f1", size=10), name="Analog"
                ))
                _fig168.add_vline(x=0, line_color="#9aa0aa", line_width=0.5)
                _fig168.add_hline(y=0, line_color="#9aa0aa", line_width=0.5)
                _fig168.update_layout(
                    title="Analog Forward Outcomes: 21d vs 63d HY Change (bps)",
                    height=340,
                    xaxis_title="Fwd 21d HY Δ (bps)",
                    yaxis_title="Fwd 63d HY Δ (bps)",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa"),
                    hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                    showlegend=False
                )
                st.plotly_chart(_fig168, use_container_width=True)
            _med21_168 = float(_np168.median(_fwd21_vals168)) if _fwd21_vals168 else float("nan")
            _med63_168 = float(_np168.median(_fwd63_vals168)) if _fwd63_vals168 else float("nan")
            _pct_wide21_168 = sum(v > 0 for v in _fwd21_vals168) / len(_fwd21_vals168) * 100 if _fwd21_vals168 else float("nan")
            if not _np168.isnan(_med21_168):
                st.caption(
                    f"Top {len(_analog_rows168)} analogs: median fwd 21d HY Δ = **{_med21_168:+.0f} bps**, "
                    f"63d = **{_med63_168:+.0f} bps**. "
                    f"{_pct_wide21_168:.0f}% of analogs showed HY widening over 21d. "
                    f"Features used: {', '.join(_feat_cols168)}."
                )
    except Exception as _e168:
        _err_track(_active_sub, _e168)
        st.caption(f"Quant analogs: {_e168}")


if _active_sub == 171:
    try:
        import plotly.graph_objects as _go171
        import numpy as _np171
        import pandas as _pd171
        _df171 = df.copy() if "df" in dir() else None
        _has171 = (_df171 is not None
                   and "composite_risk_score_smooth" in _df171.columns
                   and "hy_spread" in _df171.columns)
        if not _has171:
            st.info("composite_risk_score_smooth and hy_spread required.")
        else:
            st.subheader("Stress Probability Forecast")
            st.caption("Empirical probability of entering high-stress territory (composite score >70 or HY spread in top quartile) at forward horizons of 21, 63, and 126 days, conditioned on the current score bucket. Based on historical base rates — how often did similar starting conditions lead to stress?")
            _comp171 = _df171["composite_risk_score_smooth"].dropna()
            _hy171 = _df171["hy_spread"].dropna()
            _hy_q75171 = float(_hy171.quantile(0.75))
            # Define stress: score > 70 OR HY > 75th pct
            _stress171 = (_comp171 > 70) | (_hy171.reindex(_comp171.index).ffill() > _hy_q75171)
            # Current bucket
            _cur_score171 = float(_comp171.iloc[-1])
            def _bucket171(s):
                if s < 30: return "Low (<30)"
                elif s < 50: return "Moderate (30-50)"
                elif s < 70: return "Elevated (50-70)"
                else: return "High (>70)"
            _cur_bucket171 = _bucket171(_cur_score171)
            _horizons171 = [21, 63, 126]
            _rows171 = []
            for _bkt171 in ["Low (<30)", "Moderate (30-50)", "Elevated (50-70)", "High (>70)"]:
                _row171 = {"Score Bucket": _bkt171}
                _in_bucket171 = _comp171.apply(_bucket171) == _bkt171
                _in_idx171 = _comp171.index[_in_bucket171]
                for _h171 in _horizons171:
                    _probs171 = []
                    for _dt171 in _in_idx171:
                        try:
                            _loc171 = _stress171.index.get_loc(_dt171)
                            if _loc171 + _h171 < len(_stress171):
                                _future171 = _stress171.iloc[_loc171+1:_loc171+_h171+1]
                                _probs171.append(int(_future171.any()))
                        except Exception:
                            pass
                    _row171[f"P(stress) +{_h171}d"] = (
                        f"{_np171.mean(_probs171):.0%} (n={len(_probs171)})"
                        if _probs171 else "—"
                    )
                _rows171.append(_row171)
            _prob_df171 = _pd171.DataFrame(_rows171)
            # Highlight current bucket
            st.markdown(f"**Current bucket: {_cur_bucket171}** (composite score {_cur_score171:.1f})")
            st.dataframe(_prob_df171, use_container_width=True, hide_index=True)
            # Bar chart for current bucket only
            _cur_row171 = [r for r in _rows171 if r["Score Bucket"] == _cur_bucket171]
            if _cur_row171:
                _cr171 = _cur_row171[0]
                _bar_vals171 = []
                _bar_labs171 = []
                for _h171 in _horizons171:
                    _v171 = _cr171.get(f"P(stress) +{_h171}d", "—")
                    if _v171 != "—":
                        try:
                            _bar_vals171.append(float(_v171.split("%")[0]) / 100)
                            _bar_labs171.append(f"+{_h171}d")
                        except Exception:
                            pass
                if _bar_vals171:
                    _fig171 = _go171.Figure()
                    _fig171.add_trace(_go171.Bar(
                        x=_bar_labs171, y=_bar_vals171,
                        marker_color=["#ef4444" if v > 0.5 else "#f59e0b" if v > 0.3 else "#22c55e"
                                      for v in _bar_vals171],
                        text=[f"{v:.0%}" for v in _bar_vals171],
                        textposition="outside"
                    ))
                    _fig171.update_layout(
                        title=f"Stress Probability from '{_cur_bucket171}'",
                        height=260, yaxis_title="Probability", yaxis=dict(range=[0, 1]),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9aa0aa"),
                        hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                        showlegend=False
                    )
                    st.plotly_chart(_fig171, use_container_width=True)
            st.caption("Stress defined as composite score >70 OR HY spread above historical 75th percentile.")
    except Exception as _e171:
        _err_track(_active_sub, _e171)
        st.caption(f"Stress probability: {_e171}")


if _active_sub == "ov_reg":
    try:
        import plotly.graph_objects as _go_ov_reg
        import numpy as _np_ov_reg
        st.subheader("Regime — Section Overview")
        st.caption("Current regime classification, dwell time, and historical regime distribution. Select any sub-view from the sidebar.")
        _d = df
        def _last_reg(col): s = _d[col].dropna(); return float(s.iloc[-1]) if len(s) else float("nan")
        _comp_reg = _last_reg("composite_risk_score_smooth")
        _reg_label = ("High Stress" if _comp_reg >= 70 else ("Elevated" if _comp_reg >= 50
                       else ("Moderate" if _comp_reg >= 30 else "Low Stress")))
        _reg_col = {"High Stress": "#ef4444", "Elevated": "#f59e0b",
                    "Moderate": "#6366f1", "Low Stress": "#22c55e"}.get(_reg_label, "#9aa0aa")
        # Dwell time
        _comp_s_reg = _d["composite_risk_score_smooth"].dropna() if "composite_risk_score_smooth" in _d.columns else None
        _dwell_reg = 0
        if _comp_s_reg is not None:
            for i in range(len(_comp_s_reg) - 1, -1, -1):
                v = float(_comp_s_reg.iloc[i])
                lbl = ("High Stress" if v >= 70 else ("Elevated" if v >= 50
                        else ("Moderate" if v >= 30 else "Low Stress")))
                if lbl == _reg_label:
                    _dwell_reg += 1
                else:
                    break
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric("Current Regime", _reg_label)
        _c2.metric("Composite Score", f"{_comp_reg:.1f}" if not _np_ov_reg.isnan(_comp_reg) else "—")
        _c3.metric("Dwell Time", f"{_dwell_reg}d")
        # Regime distribution
        if _comp_s_reg is not None:
            _n_high = int((_comp_s_reg >= 70).sum())
            _n_total = len(_comp_s_reg)
            _c4.metric("% Time High Stress", f"{_n_high/_n_total:.0%}" if _n_total else "—")
        st.divider()
        # Score history colored by regime
        if _comp_s_reg is not None:
            _fig_ov_reg = _go_ov_reg.Figure()
            _cs_tail = _comp_s_reg.tail(504)
            _fig_ov_reg.add_trace(_go_ov_reg.Scatter(
                x=_cs_tail.index, y=_cs_tail.values,
                line=dict(color="#6366f1", width=2), name="Composite Score"))
            for _thresh, _col, _lbl in [(70, "rgba(239,68,68,0.15)", "High Stress"),
                                         (50, "rgba(245,158,11,0.10)", "Elevated")]:
                _fig_ov_reg.add_hrect(y0=_thresh, y1=100, fillcolor=_col, line_width=0,
                                      annotation_text=_lbl, annotation_position="right")
            _fig_ov_reg.add_hline(y=50, line_dash="dash", line_color="#9aa0aa")
            _fig_ov_reg.update_layout(
                title="Composite Score — Regime Bands (Last 2 Years)",
                height=300, yaxis_title="Score", yaxis=dict(range=[0, 100]),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9aa0aa"),
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                showlegend=False, margin=dict(t=40, b=20))
            st.plotly_chart(_fig_ov_reg, use_container_width=True)
        st.info(f"Currently in **{_reg_label}** for {_dwell_reg} days. "
                f"16 sub-views: performance, analogs, transition matrix, dwell time, quant analogs, and more.")
    except Exception as _e_ov_reg:
        _err_track(_active_sub, _e_ov_reg)
        st.caption(f"Regime overview: {_e_ov_reg}")

