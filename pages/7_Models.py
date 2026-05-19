"""
Models — analytics section page.
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
    page_title='Models — Credit Risk Dashboard',
    page_icon='🧮',
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
_SECTION_NAME = 'Models'
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


if _active_sub == "m1":
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



if _active_sub == "m2":
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


if _active_sub == "m3":
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


if _active_sub == "m4":
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


if _active_sub == "m5":
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


if _active_sub == "m6":
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


if _active_sub == "m7":
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

if _active_sub == "m8":
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

if _active_sub == 15:
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

if _active_sub == 16:
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
        _err_track(_active_sub, _ef_e)
        st.caption(f"Efficient frontier unavailable: {_ef_e}")


# =============================================================================
# ANALYTICS sub-tab 17: Kelly Criterion
# =============================================================================

if _active_sub == 17:
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

if _active_sub == 51:
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

if _active_sub == "m9":
    try:
        import plotly.graph_objects as _go_m9
        import numpy as _np_m9
        from src.threshold_robustness import run_threshold_robustness, DEFAULT_THRESHOLDS, SHIFT_GRID
        st.header("Threshold Robustness")
        st.caption(
            "Tests how sensitive the strategy's key metrics are to small shifts in "
            "composite score thresholds. A robust model should degrade gracefully — "
            "not collapse — when thresholds move ±5–10 pts."
        )
        with st.spinner("Running threshold grid..."):
            _th_result = run_threshold_robustness(df)
        if not _th_result or _th_result.get("error"):
            st.info(f"Threshold robustness unavailable: {_th_result.get('error', 'check pipeline')}")
        else:
            _th_df = _th_result.get("grid")
            if _th_df is not None and not _th_df.empty:
                st.subheader("Threshold Sensitivity Grid")
                _sharpe_col = [c for c in _th_df.columns if "sharpe" in c.lower()]
                _return_col = [c for c in _th_df.columns if "return" in c.lower() or "cagr" in c.lower()]
                _dd_col = [c for c in _th_df.columns if "drawdown" in c.lower() or "dd" in c.lower()]
                st.dataframe(
                    _th_df.style.background_gradient(
                        subset=_sharpe_col if _sharpe_col else _th_df.select_dtypes("number").columns[:1],
                        cmap="RdYlGn", axis=0
                    ).format(precision=2),
                    use_container_width=True,
                )
                # Heatmap of Sharpe if pivot-able
                if _sharpe_col and "caution_threshold" in _th_df.columns and "risk_off_threshold" in _th_df.columns:
                    try:
                        _pivot = _th_df.pivot(index="caution_threshold", columns="risk_off_threshold",
                                              values=_sharpe_col[0])
                        _fig_m9 = _go_m9.Figure(data=_go_m9.Heatmap(
                            z=_pivot.values.tolist(),
                            x=[str(c) for c in _pivot.columns],
                            y=[str(i) for i in _pivot.index],
                            colorscale="RdYlGn",
                            text=[[f"{v:.2f}" if not _np_m9.isnan(v) else "" for v in row]
                                  for row in _pivot.values.tolist()],
                            texttemplate="%{text}",
                            hovertemplate="Caution %{y} / RiskOff %{x}: Sharpe=%{z:.2f}<extra></extra>",
                        ))
                        _fig_m9.update_layout(
                            height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#9aa0aa", size=10), margin=dict(l=8, r=8, t=40, b=8),
                            title=dict(text="Sharpe Heatmap: Caution vs Risk-Off Threshold", font=dict(size=12, color="#9aa0aa")),
                            xaxis=dict(color="#6b7280", title="Risk-Off Threshold"),
                            yaxis=dict(color="#6b7280", title="Caution Threshold"),
                            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0")),
                        )
                        st.plotly_chart(_fig_m9, use_container_width=True)
                    except Exception:
                        pass
            _summary = _th_result.get("summary")
            if _summary:
                st.subheader("Robustness Summary")
                st.json(_summary)
            st.caption(f"Default thresholds: {DEFAULT_THRESHOLDS}. Grid shifts tested: {SHIFT_GRID}.")
    except Exception as _e_m9:
        _err_track(_active_sub, _e_m9)
        st.caption(f"Threshold robustness: {_e_m9}")

# ---------------------------------------------------------------------------
# Batch 20 — sub145–sub150
# ---------------------------------------------------------------------------

# sub145 — Real Yield Episodes (Rates & Macro)

if _active_sub == "ov_mod":
    try:
        import pandas as _pd_ov_mod
        st.subheader("Models — Section Overview")
        st.caption("Model validation, sensitivity analysis, and portfolio construction tools. Select any sub-view from the sidebar.")
        _d = df
        # Model health summary
        _comp_mod = float(_d["composite_risk_score_smooth"].dropna().iloc[-1]) if "composite_risk_score_smooth" in _d.columns else float("nan")
        _conf_col = "model_confidence" if "model_confidence" in _d.columns else None
        _conf_val = float(_d[_conf_col].dropna().iloc[-1]) if _conf_col else float("nan")
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("Composite Score", f"{_comp_mod:.1f}" if not _pd_ov_mod.isna(_comp_mod) else "—")
        _c2.metric("Model Confidence", f"{_conf_val:.1%}" if not _pd_ov_mod.isna(_conf_val) else "—")
        _n_scores = sum(1 for c in _d.columns if c.endswith("_score_smooth"))
        _c3.metric("Active Sub-scores", str(_n_scores))
        st.divider()
        _rows_mod = [
            {"Category": "Validation", "Views": "Sensitivity, OOS Splits, Walk-Forward, Transitions",
             "Purpose": "Test model robustness and avoid overfitting"},
            {"Category": "Risk", "Views": "Monte Carlo, Scenarios, Stress Test",
             "Purpose": "Forward simulation and tail risk quantification"},
            {"Category": "Portfolio", "Views": "Kelly Sizing, Risk Parity, Efficient Frontier",
             "Purpose": "Translate signals into position sizes"},
            {"Category": "Research", "Views": "Merton DD, Regimes, Sub-period",
             "Purpose": "Deep-dive model diagnostics and period analysis"},
        ]
        st.dataframe(_pd_ov_mod.DataFrame(_rows_mod), use_container_width=True, hide_index=True)
        st.info("14 sub-views covering model validation, scenario analysis, and portfolio construction.")
    except Exception as _e_ov_mod:
        _err_track(_active_sub, _e_ov_mod)
        st.caption(f"Models overview: {_e_ov_mod}")


if _active_sub == 192:
    st.subheader("Model Confidence Monitor")
    st.caption("Signal confidence and risk appetite alignment — when to trust the model and when regime uncertainty is elevated")
    try:
        import plotly.graph_objects as _go192
        import numpy as _np192
        import pandas as _pd192
        _score_col192 = next((c for c in ["composite_score_smooth","credit_risk_score_smooth"] if c in df.columns), None)
        _df192_cols = ["model_confidence","risk_appetite_score_smooth","hy_spread","vix"]
        if _score_col192:
            _df192_cols.append(_score_col192)
        _df192 = df[_df192_cols].dropna().copy()
        _last192 = _df192.iloc[-1]
        _mc_pct192 = float((_df192["model_confidence"] < _last192["model_confidence"]).mean() * 100)
        _ra_pct192 = float((_df192["risk_appetite_score_smooth"] < _last192["risk_appetite_score_smooth"]).mean() * 100)
        _c1_192, _c2_192, _c3_192, _c4_192 = st.columns(4)
        _c1_192.metric("Model Confidence", f"{_last192['model_confidence']:.2f}", f"{_mc_pct192:.0f}th pct")
        _c2_192.metric("Risk Appetite", f"{_last192['risk_appetite_score_smooth']:.2f}", f"{_ra_pct192:.0f}th pct")
        _c3_192.metric("VIX", f"{_last192['vix']:.1f}")
        _trust192 = _last192["model_confidence"] > _df192["model_confidence"].quantile(0.50)
        _c4_192.metric("Model Signal", "High confidence" if _trust192 else "Low confidence")
        st.divider()
        # Model confidence + risk appetite over time
        _fig192a = _go192.Figure()
        _fig192a.add_trace(_go192.Scatter(
            x=_df192.index, y=_df192["model_confidence"],
            name="Model Confidence", fill="tozeroy",
            fillcolor="rgba(16,185,129,0.15)", line=dict(color="#10b981", width=1.8), yaxis="y1"
        ))
        _fig192a.add_trace(_go192.Scatter(
            x=_df192.index, y=_df192["risk_appetite_score_smooth"],
            name="Risk Appetite Score", line=dict(color="#3b82f6", width=1.5), yaxis="y2"
        ))
        _fig192a.add_trace(_go192.Scatter(
            x=_df192.index, y=_df192["hy_spread"],
            name="HY Spread (bps)", line=dict(color="#ef4444", width=1, dash="dot"), yaxis="y3"
        ))
        _fig192a.update_layout(
            title="Model Confidence & Risk Appetite vs HY Spread",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            yaxis=dict(title="Confidence", side="left", color="#10b981"),
            yaxis2=dict(title="Risk Appetite", side="right", overlaying="y", color="#3b82f6", showgrid=False),
            yaxis3=dict(title="HY bps", overlaying="y", side="right", position=0.95, color="#ef4444", showgrid=False, visible=False),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=50, b=40),
            hoverlabel=dict(bgcolor="#1a1f2e", bordercolor="#2d3550", font=dict(color="#e2e8f0"))
        )
        st.plotly_chart(_fig192a, use_container_width=True)
        # Low confidence periods as context for score interpretation
        _low_conf_q25_192 = float(_df192["model_confidence"].quantile(0.25))
        _low_conf_mask192 = _df192["model_confidence"] < _low_conf_q25_192
        if _score_col192 and _score_col192 in _df192.columns:
            _c1b_192, _c2b_192 = st.columns(2)
            with _c1b_192:
                # Score in high vs low confidence regimes
                _df192["conf_regime"] = _low_conf_mask192.map({True: "Low Confidence", False: "High Confidence"})
                _conf_stats192 = _df192.groupby("conf_regime")[_score_col192].agg(["mean","std"]).round(3)
                _conf_stats192.columns = ["Mean Score", "Std"]
                _conf_hy192 = _df192.groupby("conf_regime")["hy_spread"].agg(["mean","std"]).round(1)
                _conf_hy192.columns = ["Mean HY", "HY Std"]
                _combined_conf192 = _conf_stats192.join(_conf_hy192)
                st.markdown("**Score & HY by Confidence Regime**")
                st.dataframe(_combined_conf192, use_container_width=True)
            with _c2b_192:
                # Model confidence vs VIX scatter
                _fig192b = _go192.Figure()
                _fig192b.add_trace(_go192.Scatter(
                    x=_df192["vix"], y=_df192["model_confidence"],
                    mode="markers", marker=dict(size=3, color="#10b981", opacity=0.3), name="History"
                ))
                _fig192b.add_trace(_go192.Scatter(
                    x=[_last192["vix"]], y=[_last192["model_confidence"]],
                    mode="markers", marker=dict(size=12, color="#f59e0b", symbol="star"), name="Now"
                ))
                _corr_mc_vix192 = round(float(_df192["model_confidence"].corr(_df192["vix"])), 2)
                _fig192b.update_layout(
                    title=f"Model Confidence vs VIX (r={_corr_mc_vix192:+.2f})",
                    height=230,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9aa0aa", size=11),
                    xaxis_title="VIX", yaxis_title="Model Confidence",
                    margin=dict(t=40, b=30))
                st.plotly_chart(_fig192b, use_container_width=True)
        # Risk appetite distribution
        _fig192c = _go192.Figure()
        _fig192c.add_trace(_go192.Histogram(
            x=_df192["risk_appetite_score_smooth"], nbinsx=50,
            marker_color="#3b82f6", opacity=0.7
        ))
        _fig192c.add_vline(x=float(_last192["risk_appetite_score_smooth"]), line_dash="dash",
                           line_color="#f59e0b",
                           annotation_text=f"Now: {_last192['risk_appetite_score_smooth']:.2f}")
        _fig192c.update_layout(
            title="Risk Appetite Score Distribution",
            height=180,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9aa0aa"),
            xaxis_title="Risk Appetite Score", yaxis_title="Count",
            margin=dict(t=40, b=25))
        st.plotly_chart(_fig192c, use_container_width=True)
        _div_signal192 = abs(_last192["model_confidence"] - _last192["risk_appetite_score_smooth"]) > 0.3
        st.caption(
            f"Model confidence {_last192['model_confidence']:.2f} ({_mc_pct192:.0f}th pct). "
            f"Risk appetite {_last192['risk_appetite_score_smooth']:.2f} ({_ra_pct192:.0f}th pct). "
            f"{'Confidence-risk appetite divergence — signal reliability may be reduced.' if _div_signal192 else 'Model confidence and risk appetite aligned.'} "
            f"Low-confidence periods (bottom quartile) have wider score dispersion."
        )
    except Exception as _e192:
        _err_track(_active_sub, _e192)
        st.caption(f"Model confidence: {_e192}")
