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
from src.regime_attribution import COMPOSITE_WEIGHTS, DISPLAY_NAMES, run_regime_attribution
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
def load_signal_validation(_df, _oos_cutoff="2022-01-01"):
    """Run IS/OOS signal validation against forward returns (cached)."""
    return validate_signals_vs_returns(_df, oos_cutoff=_oos_cutoff)


@st.cache_data
def load_signal_horizon_grid(_df, _oos_cutoff="2022-01-01"):
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Signal", "Charts", "Portfolio", "Backtest", "Analytics", "Models", "History", "Allocation"
])

with tab1:
    st.header("Current Signal Snapshot")

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

with tab2:
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _make_subplots

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

    # ── 12. Strategy vs SP500 Equity Curve ───────────────────────────────────
    if "strategy_equity_curve" in _df2.columns:
        st.subheader("Strategy vs SP500 — Cumulative Return")
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

with tab5:  # Analytics — 11 sub-tabs
    (_analytics_sub1, _analytics_sub2, _analytics_sub3, _analytics_sub4,
     _analytics_sub5, _analytics_sub6, _analytics_sub7, _analytics_sub8,
     _analytics_sub9, _analytics_sub10, _analytics_sub11) = st.tabs([
        "Validation", "Attribution", "Timeline", "Sig Decay",
        "Ortho", "Tail Risk", "Stress", "Performance", "Factors",
        "Regime Validity", "Failure Analysis",
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
    from src.backtester import OOS_CUTOFF, build_strategy_backtest, compute_oos_split
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

        _sq_c1, _sq_c2, _sq_c3 = st.columns(3)
        _sq_c1.metric(
            "Score↔SP500 30D Corr",
            f"{_corr_score:.2f}",
            help="Pearson correlation between composite_risk_score_smooth and SP500 30-day forward return. "
                 "Negative means higher risk score → lower future return (expected).",
        )
        _sq_c2.metric(
            "Unconditional Hit Rate",
            f"{_hit_all:.1%}",
            help="% of all days where SP500 30-day forward return was positive.",
        )
        _sq_c3.metric(
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
            "Strategy Sharpe": [_f2(_is.get("strategy_sharpe")),
                                _f2(_oos.get("strategy_sharpe")),
                                _f2(_fp.get("strategy_sharpe"))],
            "SP500 Sharpe":    [_f2(_is.get("sp500_sharpe")),
                                _f2(_oos.get("sp500_sharpe")),
                                _f2(_fp.get("sp500_sharpe"))],
            "Max Drawdown":    [_pct(_is.get("strategy_max_drawdown")),
                                _pct(_oos.get("strategy_max_drawdown")),
                                _pct(_fp.get("strategy_max_drawdown"))],
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

    # ── Credit Portfolio Backtest ─────────────────────────────────────────────
    st.subheader("3. Credit Portfolio Backtest")
    st.caption(
        "Performance of the regime-based HY/IG credit allocation. "
        "Total return = carry (all-in yield / 252) + duration-adjusted price return. "
        "HY duration ≈ 4y · IG duration ≈ 7y · 10bps one-way transaction cost."
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
