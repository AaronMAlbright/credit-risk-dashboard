import pandas as pd
import streamlit as st
from pathlib import Path

from src.bootstrap import run_bootstrap_analysis
from src.regime_attribution import COMPOSITE_WEIGHTS, DISPLAY_NAMES, run_regime_attribution
from src.regime_charts import (
    build_decision_timeline,
    build_score_history,
    build_sp500_with_regime_overlay,
)
from src.regime_transition import run_regime_analysis

st.set_page_config(
    page_title="Macro Credit Risk Dashboard",
    page_icon="📉",
    layout="wide",
)

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


df = load_data()
history = load_history()
latest = df.iloc[-1]

st.title("Macro Credit Risk Dashboard")
st.caption("Macro, credit, liquidity, complacency, and mean-reversion regime engine.")

decision = latest.get("final_decision", "N/A")
environment = latest.get("final_environment", "N/A")
action = latest.get("final_action", "N/A")

st.subheader(f"Current Decision: {decision}")
st.write(f"**Environment:** {environment}")
st.write(f"**Action:** {action}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Macro Risk", round(latest.get("macro_risk_score_smooth", 0), 1))
col2.metric("Credit Risk", round(latest.get("credit_market_risk_score_smooth", 0), 1))
col3.metric("Complacency", round(latest.get("complacency_score_smooth", 0), 1))
col4.metric("Mean Reversion", round(latest.get("mean_reversion_score_smooth", 0), 1))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Liquidity", round(latest.get("liquidity_regime_score_smooth", 0), 1))
col6.metric("Risk Appetite", round(latest.get("risk_appetite_score_smooth", 0), 1))
col7.metric("Treasury Stress", round(latest.get("treasury_stress_score_smooth", 0), 1))
col8.metric("SP500 Drawdown", f"{latest.get('sp500_drawdown', 0):.2%}")

col9, col10 = st.columns(2)
col9.metric("Composite Risk", round(latest.get("composite_risk_score_smooth", 0), 1))
col10.metric("Composite Regime", latest.get("composite_risk_label", "N/A"))

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Current Signal",
    "Charts",
    "Portfolio",
    "Validation",
    "Backtest",
    "History",
    "Sensitivity",
    "Regime Transitions",
    "Attribution",
    "Timeline",
])

with tab1:
    st.header("Current Signal Snapshot")

    left, right = st.columns(2)

    with left:
        st.subheader("Market Snapshot")
        st.write(f"**Spread:** {latest.get('spread', 0):.2f}")
        st.write(f"**Yield Curve Regime:** {latest.get('yield_curve_regime', 'N/A')}")
        st.write(f"**Unemployment:** {latest.get('unemployment', 0):.1f}")
        st.write(f"**HY Spread:** {latest.get('hy_spread', 0):.2f}")
        st.write(f"**VIX:** {latest.get('vix', 0):.2f}")
        st.write(f"**SP500:** {latest.get('sp500', 0):.2f}")

    with right:
        st.subheader("Signal Regime")
        st.write(f"**Macro Label:** {latest.get('macro_risk_label', 'N/A')}")
        st.write(f"**Credit Label:** {latest.get('credit_market_risk_label', 'N/A')}")
        st.write(f"**Liquidity Label:** {latest.get('liquidity_regime_label', 'N/A')}")
        st.write(f"**Complacency Label:** {latest.get('complacency_label', 'N/A')}")
        st.write(f"**Mean-Reversion Label:** {latest.get('mean_reversion_label', 'N/A')}")
        st.write(f"**Composite Regime:** {latest.get('composite_risk_label', 'N/A')}")
        st.write(f"**Transition Regime:** {latest.get('transition_regime', 'N/A')}")
        st.write(f"**Transition Signal:** {latest.get('transition_signal', 'N/A')}")

    st.subheader("Latest Text Report")
    if REPORT_PATH.exists():
        st.text(REPORT_PATH.read_text())
    else:
        st.warning("No signal report found. Run `python app.py` first.")

with tab2:
    st.header("Charts")

    chart_files = [
        "macro_credit_complacency.png",
        "composite_risk_score.png",
        "composite_regime_timeline.png",
        "backtest_equity_curve.png",
        "hy_credit_spread.png",
        "credit_impulse.png",
        "treasury_credit_macro.png",
        "cross_asset_liquidity.png",
        "sp500_drawdown.png",
        "risk_appetite_vs_complacency.png",
    ]

    for chart in chart_files:
        path = CHART_DIR / chart
        if path.exists():
            st.subheader(chart.replace("_", " ").replace(".png", "").title())
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"Missing chart: {chart}")

with tab3:
    st.header("Portfolio Stance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equity Weight", f"{latest.get('equity_weight', 0):.0%}")
    col2.metric("Credit Weight", f"{latest.get('credit_weight', 0):.0%}")
    col3.metric("Cash Weight", f"{latest.get('cash_weight', 0):.0%}")
    col4.metric("Duration Bias", latest.get("duration_bias", "N/A"))

    st.subheader("Decision Logic")
    st.write(f"**Decision:** {decision}")
    st.write(f"**Environment:** {environment}")
    st.write(f"**Action:** {action}")
    st.write(f"**Buy Trigger:** {latest.get('buy_trigger', 'N/A')}")
    st.write(f"**Risk-Off Trigger:** {latest.get('risk_off_trigger', 'N/A')}")

with tab4:
    st.header("Walk-Forward Validation")

    wf_windows, wf_regimes = load_walk_forward()

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

with tab5:
    st.header("Backtest")

    required_cols = [
        "strategy_equity_curve",
        "sp500_equity_curve",
        "strategy_daily_return",
        "sp500_daily_return",
        "strategy_drawdown",
        "sp500_backtest_drawdown",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        st.warning(f"Missing backtest columns: {missing_cols}. Run `python app.py` first.")
    else:
        strategy_total = df["strategy_equity_curve"].iloc[-1] - 1
        sp500_total = df["sp500_equity_curve"].iloc[-1] - 1
        strategy_total = df["strategy_equity_curve"].iloc[-1] - 1
        sp500_total = df["sp500_equity_curve"].iloc[-1] - 1

        strategy_vol = df["strategy_daily_return"].std() * (252 ** 0.5)
        sp500_vol = df["sp500_daily_return"].std() * (252 ** 0.5)

        strategy_sharpe = (
                                  df["strategy_daily_return"].mean()
                                  / df["strategy_daily_return"].std()
                          ) * (252 ** 0.5)

        sp500_sharpe = (
                               df["sp500_daily_return"].mean()
                               / df["sp500_daily_return"].std()
                       ) * (252 ** 0.5)

        capture_ratio = (
            strategy_total / sp500_total
            if sp500_total != 0
            else 0
        )

        drawdown_improvement = (
                abs(df["sp500_backtest_drawdown"].min())
                - abs(df["strategy_drawdown"].min())
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Total Return", f"{strategy_total:.2%}")
        col2.metric("SP500 Total Return", f"{sp500_total:.2%}")
        col3.metric("Strategy Max Drawdown", f"{df['strategy_drawdown'].min():.2%}")
        col4.metric("SP500 Max Drawdown", f"{df['sp500_backtest_drawdown'].min():.2%}")
        col5, col6, col7, col8 = st.columns(4)

        col5.metric(
            "Strategy Volatility",
            f"{strategy_vol:.2%}"
        )

        col6.metric(
            "SP500 Volatility",
            f"{sp500_vol:.2%}"
        )

        col7.metric(
            "Strategy Sharpe",
            f"{strategy_sharpe:.2f}"
        )

        col8.metric(
            "Capture Ratio",
            f"{capture_ratio:.2f}"
        )
        chart_path = CHART_DIR / "backtest_equity_curve.png"
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)

        st.subheader("Recent Strategy Weights")
        weight_cols = [
            "date",
            "final_decision",
            "composite_risk_label",
            "strategy_weight",
            "strategy_weight_lagged",
            "strategy_daily_return",
        ]
        existing_weight_cols = [c for c in weight_cols if c in df.columns]
        st.dataframe(df[existing_weight_cols].tail(50), use_container_width=True)

with tab6:
    st.header("Model Run History")

    if not history.empty:
        st.dataframe(history.tail(50), use_container_width=True)

        chart_cols = [
            "macro_risk",
            "credit_risk",
            "complacency",
            "mean_reversion",
        ]

        available_history_cols = [c for c in chart_cols if c in history.columns]

        if "timestamp" in history.columns and available_history_cols:
            history["timestamp"] = pd.to_datetime(history["timestamp"])
            st.line_chart(history.set_index("timestamp")[available_history_cols])
    else:
        st.warning("No run history found yet.")

with tab7:
    st.header("Parameter Sensitivity")

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

            # ── Heatmap images ───────────────────────────────────────────────
            heatmap_metrics = {
                "score_scale":        ["sharpe", "total_return"],
                "composite_weights":  ["fwd_mean_return", "fwd_hit_rate"],
                "backtester_weights": ["sharpe", "total_return"],
            }
            shown = False
            for metric in heatmap_metrics.get(group_key, []):
                img_path = SENS_HEATMAP_DIR / f"{group_key}_{metric}.png"
                if img_path.exists():
                    if not shown:
                        st.caption("Heatmaps (rows = perturbation value, columns = parameter)")
                        shown = True
                    st.image(str(img_path), use_container_width=True)

with tab8:
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

        # ── Transition probability heatmap ───────────────────────────────────
        st.subheader("Transition Probability Heatmap")
        heatmap_path = REGIME_TRANS_DIR / f"{view_col}_heatmap.png"
        if heatmap_path.exists():
            st.image(str(heatmap_path), use_container_width=True)
        else:
            st.caption("Heatmap not yet saved to disk — showing table instead.")
            probs = res["transition_probs"]
            st.dataframe(probs.style.background_gradient(cmap="YlOrRd", axis=1).format("{:.1%}"))

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
with tab9:
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
        delta_cols = [c for c in shifts.columns if c.endswith("_delta")]
        display_cols = ["date", "from_regime", "to_regime",
                        "primary_driver", "secondary_driver",
                        "composite_delta", "direction"] + delta_cols
        shift_disp = shifts[[c for c in display_cols if c in shifts.columns]].copy()
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

with tab10:
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
